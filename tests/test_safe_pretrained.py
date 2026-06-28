"""Tests for safe_load_pretrained() and its private helpers.

All tests are fully offline — no checkpoint is downloaded from the internet.
``get_pretrained_weights`` is patched with ``unittest.mock.patch`` so that
every test constructs its own synthetic checkpoint dict and feeds it directly
into the functions under test.

Test layout
-----------
TestExtractStateDict
    Tests for _extract_state_dict(): plain dict, Lightning format, model-key format.

TestStripPrefix
    Tests for _strip_prefix(): model. prefix, module. prefix, no prefix,
    mixed keys, custom prefixes.

TestSafeLoadPretrained
    Integration tests for safe_load_pretrained() via a small synthetic model
    (MinimalModel) that mirrors the encoder + head pattern of PARSeq:

    test_same_architecture_loads_all_layers
        Same shapes → every key is loaded, nothing is skipped.

    test_different_head_skips_classifier
        Head with different vocab size → encoder/decoder load, head is skipped.

    test_encoder_values_are_transferred
        Verifies that encoder weight *values* in the model actually match the
        checkpoint after loading (not just that no error was raised).

    test_head_values_unchanged_when_skipped
        Verifies that skipped head tensors retain their original init values.

    test_model_prefix_stripped
        Checkpoint keys with leading ``model.`` are matched correctly.

    test_module_prefix_stripped
        Checkpoint keys with leading ``module.`` are matched correctly.

    test_lightning_checkpoint_format
        Checkpoint wrapped as ``{"state_dict": {...}, "epoch": 5}`` is handled.

    test_plain_state_dict_format
        A plain state dict (no wrapper keys) is handled identically.

    test_missing_keys_reported
        Keys present in the model but absent from the checkpoint appear in
        the ``"missing"`` list and do not cause a RuntimeError.

    test_unexpected_keys_reported
        Keys in the checkpoint that do not exist in the model appear in
        the ``"unexpected"`` list.

    test_return_statistics_correct
        Asserts that all four returned lists contain exactly the right keys.

    test_no_runtime_error_on_shape_mismatch
        Confirms that a size mismatch does NOT raise RuntimeError.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# Functions under test
from strhub.models.utils import (
    _extract_state_dict,
    _strip_prefix,
    safe_load_pretrained,
)


# ---------------------------------------------------------------------------
# Synthetic models that mirror the PARSeq encoder → head pattern
# ---------------------------------------------------------------------------

EMBED_DIM = 8   # tiny embedding dimension — keeps tests fast


class MinimalModel(nn.Module):
    """Minimal model with an encoder block and a classifier head.

    Structure mirrors PARSeq:
      encoder.fc   – weight (EMBED_DIM, EMBED_DIM), bias (EMBED_DIM,)
      decoder.fc   – weight (EMBED_DIM, EMBED_DIM), bias (EMBED_DIM,)
      head         – weight (vocab, EMBED_DIM),      bias (vocab,)

    ``vocab`` is the only dimension that changes between charsets.
    """

    def __init__(self, vocab: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.decoder = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.head(self.decoder(self.encoder(x)))


def _state_dict_for(vocab: int) -> dict[str, torch.Tensor]:
    """Return a freshly initialised state dict for ``MinimalModel(vocab)``."""
    return MinimalModel(vocab).state_dict()


def _make_checkpoint(state_dict: dict, fmt: str = 'plain') -> dict:
    """Wrap *state_dict* in the requested checkpoint format.

    Parameters
    ----------
    fmt : ``'plain'`` | ``'lightning'`` | ``'model_key'``
    """
    if fmt == 'plain':
        return state_dict
    if fmt == 'lightning':
        return {'state_dict': state_dict, 'epoch': 5, 'global_step': 1000}
    if fmt == 'model_key':
        return {'model': state_dict, 'optimizer': {}}
    raise ValueError(f'Unknown fmt: {fmt!r}')


def _add_prefix(state_dict: dict, prefix: str) -> dict:
    """Return a copy of *state_dict* with every key prefixed by *prefix*."""
    return {prefix + k: v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Helpers: _extract_state_dict
# ---------------------------------------------------------------------------

class TestExtractStateDict:
    """Unit tests for _extract_state_dict()."""

    def test_plain_dict_returned_unchanged(self):
        sd = _state_dict_for(vocab=10)
        result = _extract_state_dict(sd)
        assert result is sd

    def test_lightning_format_extracts_state_dict(self):
        sd = _state_dict_for(vocab=10)
        ckpt = {'state_dict': sd, 'epoch': 3, 'global_step': 500}
        result = _extract_state_dict(ckpt)
        assert result is sd

    def test_model_key_format_extracts_model(self):
        sd = _state_dict_for(vocab=10)
        ckpt = {'model': sd, 'optimizer': {'param_groups': []}}
        result = _extract_state_dict(ckpt)
        assert result is sd

    def test_state_dict_takes_priority_over_model_key(self):
        """'state_dict' is checked before 'model'."""
        sd_real = _state_dict_for(vocab=10)
        sd_other = _state_dict_for(vocab=5)
        ckpt = {'state_dict': sd_real, 'model': sd_other}
        result = _extract_state_dict(ckpt)
        assert result is sd_real


# ---------------------------------------------------------------------------
# Helpers: _strip_prefix
# ---------------------------------------------------------------------------

class TestStripPrefix:
    """Unit tests for _strip_prefix()."""

    def test_model_prefix_stripped(self):
        sd = {'model.encoder.weight': torch.zeros(4, 4),
              'model.head.weight': torch.zeros(10, 4)}
        result = _strip_prefix(sd)
        assert set(result.keys()) == {'encoder.weight', 'head.weight'}

    def test_module_prefix_stripped(self):
        sd = {'module.encoder.weight': torch.zeros(4, 4),
              'module.head.bias': torch.zeros(10)}
        result = _strip_prefix(sd)
        assert set(result.keys()) == {'encoder.weight', 'head.bias'}

    def test_no_prefix_unchanged(self):
        sd = {'encoder.weight': torch.zeros(4, 4),
              'head.weight': torch.zeros(10, 4)}
        result = _strip_prefix(sd)
        assert set(result.keys()) == {'encoder.weight', 'head.weight'}

    def test_mixed_prefixes_handled(self):
        """Some keys have a prefix, others do not."""
        sd = {'model.encoder.weight': torch.zeros(4, 4),
              'head.weight': torch.zeros(10, 4)}
        result = _strip_prefix(sd)
        assert set(result.keys()) == {'encoder.weight', 'head.weight'}

    def test_only_first_prefix_stripped(self):
        """Double-prefixed key: only the outermost prefix is removed."""
        sd = {'model.model.encoder.weight': torch.zeros(4, 4)}
        result = _strip_prefix(sd)
        # 'model.' stripped once → 'model.encoder.weight'
        assert set(result.keys()) == {'model.encoder.weight'}

    def test_custom_prefixes(self):
        sd = {'backbone.layer.weight': torch.zeros(4, 4)}
        result = _strip_prefix(sd, prefixes=('backbone.',))
        assert set(result.keys()) == {'layer.weight'}

    def test_values_preserved(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        sd = {'model.vec': t}
        result = _strip_prefix(sd)
        assert torch.equal(result['vec'], t)


# ---------------------------------------------------------------------------
# Integration: safe_load_pretrained
# ---------------------------------------------------------------------------

MOCK_TARGET = 'strhub.models.utils.get_pretrained_weights'


class TestSafeLoadPretrained:
    """Integration tests for safe_load_pretrained().

    Every test patches ``get_pretrained_weights`` so no network call is made.
    The synthetic checkpoint is built from a MinimalModel state dict with
    controlled shapes so that each scenario is deterministic.
    """

    # ------------------------------------------------------------------ #
    # 1. Same architecture — all layers should load                        #
    # ------------------------------------------------------------------ #

    def test_same_architecture_loads_all_layers(self):
        """When checkpoint and model have identical architectures, every
        tensor is loaded and nothing is skipped."""
        vocab = 10
        ckpt_sd = _state_dict_for(vocab)
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert len(result['loaded']) == len(ckpt_sd)
        assert result['skipped'] == []
        assert result['missing'] == []
        assert result['unexpected'] == []

    def test_same_architecture_all_layers_count(self):
        """The total number of loaded keys equals the number of tensors
        in a MinimalModel state dict (6: enc.w, enc.b, dec.w, dec.b,
        head.w, head.b)."""
        vocab = 10
        model = MinimalModel(vocab)
        ckpt_sd = _state_dict_for(vocab)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert len(result['loaded']) == 6

    # ------------------------------------------------------------------ #
    # 2. Different classifier dimensions — encoder/decoder load, head skipped
    # ------------------------------------------------------------------ #

    def test_different_head_skips_classifier_weight(self):
        """Checkpoint has vocab=10; model has vocab=80 (Marathi).
        head.weight should appear in skipped."""
        ckpt_sd = _state_dict_for(vocab=10)
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert 'head.weight' in result['skipped']

    def test_different_head_skips_classifier_bias(self):
        ckpt_sd = _state_dict_for(vocab=10)
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert 'head.bias' in result['skipped']

    def test_different_head_loads_encoder(self):
        """encoder.weight and encoder.bias must be in 'loaded'."""
        ckpt_sd = _state_dict_for(vocab=10)
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert 'encoder.weight' in result['loaded']
        assert 'encoder.bias' in result['loaded']

    def test_different_head_loads_decoder(self):
        """decoder.weight and decoder.bias must be in 'loaded'."""
        ckpt_sd = _state_dict_for(vocab=10)
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert 'decoder.weight' in result['loaded']
        assert 'decoder.bias' in result['loaded']

    def test_skipped_count_is_two_for_head_mismatch(self):
        """Only head.weight and head.bias are skipped — exactly 2 tensors."""
        ckpt_sd = _state_dict_for(vocab=10)
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert len(result['skipped']) == 2
        assert len(result['loaded']) == 4   # enc.w, enc.b, dec.w, dec.b


    # ------------------------------------------------------------------ #
    # 3. Tensor values are actually transferred / preserved                #
    # ------------------------------------------------------------------ #

    def test_encoder_values_are_transferred(self):
        """After loading, the model's encoder.weight must equal the
        checkpoint's encoder.weight (not just 'no error')."""
        vocab = 10
        ckpt_sd = _state_dict_for(vocab)
        # Give the checkpoint encoder a distinct, recognisable value.
        sentinel = torch.full((EMBED_DIM, EMBED_DIM), fill_value=42.0)
        ckpt_sd['encoder.weight'] = sentinel

        model = MinimalModel(vocab)
        with patch(MOCK_TARGET, return_value=ckpt_sd):
            safe_load_pretrained(model, 'parseq')

        assert torch.equal(model.encoder.weight.data, sentinel)

    def test_head_values_unchanged_when_skipped(self):
        """When the head is skipped, the model retains its original
        (randomly-initialised) head weights, not the checkpoint values."""
        ckpt_sd = _state_dict_for(vocab=10)
        model = MinimalModel(vocab=80)

        # Record the model's initial head weight before loading.
        original_head_weight = model.head.weight.data.clone()

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            safe_load_pretrained(model, 'parseq')

        assert torch.equal(model.head.weight.data, original_head_weight)

    def test_decoder_values_are_transferred(self):
        """decoder.weight in the model matches the checkpoint value."""
        vocab = 10
        ckpt_sd = _state_dict_for(vocab)
        sentinel = torch.full((EMBED_DIM, EMBED_DIM), fill_value=7.0)
        ckpt_sd['decoder.weight'] = sentinel

        model = MinimalModel(vocab)
        with patch(MOCK_TARGET, return_value=ckpt_sd):
            safe_load_pretrained(model, 'parseq')

        assert torch.equal(model.decoder.weight.data, sentinel)

    # ------------------------------------------------------------------ #
    # 4. No RuntimeError on shape mismatch                                 #
    # ------------------------------------------------------------------ #

    def test_no_runtime_error_on_head_shape_mismatch(self):
        """A size-mismatched head must NOT raise RuntimeError."""
        ckpt_sd = _state_dict_for(vocab=10)
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            # Should complete without exception.
            safe_load_pretrained(model, 'parseq')

    # ------------------------------------------------------------------ #
    # 5. Prefix handling                                                   #
    # ------------------------------------------------------------------ #

    def test_model_prefix_stripped_and_loaded(self):
        """Checkpoint keys prefixed with 'model.' are normalised and loaded."""
        vocab = 10
        bare_sd = _state_dict_for(vocab)
        prefixed_sd = _add_prefix(bare_sd, 'model.')
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=prefixed_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert len(result['loaded']) == len(bare_sd)
        assert result['skipped'] == []
        assert result['unexpected'] == []

    def test_module_prefix_stripped_and_loaded(self):
        """Checkpoint keys prefixed with 'module.' are normalised and loaded."""
        vocab = 10
        bare_sd = _state_dict_for(vocab)
        prefixed_sd = _add_prefix(bare_sd, 'module.')
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=prefixed_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert len(result['loaded']) == len(bare_sd)
        assert result['skipped'] == []
        assert result['unexpected'] == []

    def test_model_prefix_with_head_mismatch(self):
        """model. prefix + different vocab: encoder loads, head skipped."""
        bare_sd = _state_dict_for(vocab=10)
        prefixed_sd = _add_prefix(bare_sd, 'model.')
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=prefixed_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert 'encoder.weight' in result['loaded']
        assert 'decoder.weight' in result['loaded']
        assert 'head.weight' in result['skipped']
        assert 'head.bias' in result['skipped']


    # ------------------------------------------------------------------ #
    # 6. Checkpoint format handling                                        #
    # ------------------------------------------------------------------ #

    def test_plain_state_dict_format(self):
        """A checkpoint that is already a plain state dict is handled."""
        vocab = 10
        ckpt = _make_checkpoint(_state_dict_for(vocab), fmt='plain')
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=ckpt):
            result = safe_load_pretrained(model, 'parseq')

        assert len(result['loaded']) == 6
        assert result['skipped'] == []

    def test_lightning_checkpoint_format(self):
        """A PyTorch Lightning checkpoint (state_dict key + metadata) is
        unwrapped automatically."""
        vocab = 10
        ckpt = _make_checkpoint(_state_dict_for(vocab), fmt='lightning')
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=ckpt):
            result = safe_load_pretrained(model, 'parseq')

        assert len(result['loaded']) == 6
        assert result['skipped'] == []

    def test_lightning_checkpoint_with_head_mismatch(self):
        """Lightning-wrapped checkpoint + different charset: head skipped."""
        ckpt = _make_checkpoint(_state_dict_for(vocab=10), fmt='lightning')
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=ckpt):
            result = safe_load_pretrained(model, 'parseq')

        assert 'head.weight' in result['skipped']
        assert 'encoder.weight' in result['loaded']

    def test_model_key_checkpoint_format(self):
        """A checkpoint dict with a 'model' key is unwrapped automatically."""
        vocab = 10
        ckpt = _make_checkpoint(_state_dict_for(vocab), fmt='model_key')
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=ckpt):
            result = safe_load_pretrained(model, 'parseq')

        assert len(result['loaded']) == 6
        assert result['skipped'] == []

    # ------------------------------------------------------------------ #
    # 7. Missing and unexpected key reporting                              #
    # ------------------------------------------------------------------ #

    def test_missing_keys_reported(self):
        """Keys in the model but absent from the checkpoint appear in
        'missing', and no RuntimeError is raised."""
        vocab = 10
        # Checkpoint is missing decoder keys entirely.
        bare_sd = _state_dict_for(vocab)
        reduced_sd = {k: v for k, v in bare_sd.items()
                      if not k.startswith('decoder')}
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=reduced_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert 'decoder.weight' in result['missing']
        assert 'decoder.bias' in result['missing']
        # The present layers still load.
        assert 'encoder.weight' in result['loaded']

    def test_unexpected_keys_reported(self):
        """Keys in the checkpoint that do not exist in the model appear in
        'unexpected', and no error is raised."""
        vocab = 10
        bare_sd = _state_dict_for(vocab)
        # Add a key that the model doesn't have.
        bare_sd['ghost.weight'] = torch.zeros(4, 4)
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=bare_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert 'ghost.weight' in result['unexpected']
        # All real layers still load.
        assert len(result['loaded']) == 6

    # ------------------------------------------------------------------ #
    # 8. Return statistics correctness                                     #
    # ------------------------------------------------------------------ #

    def test_return_statistics_correct_same_arch(self):
        """Full correctness check: same architecture, plain checkpoint."""
        vocab = 10
        ckpt_sd = _state_dict_for(vocab)
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        expected_keys = {'encoder.weight', 'encoder.bias',
                         'decoder.weight', 'decoder.bias',
                         'head.weight', 'head.bias'}
        assert set(result['loaded']) == expected_keys
        assert result['skipped'] == []
        assert result['missing'] == []
        assert result['unexpected'] == []

    def test_return_statistics_correct_head_mismatch(self):
        """Full correctness check: different classifier dimension."""
        ckpt_sd = _state_dict_for(vocab=10)
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert set(result['loaded']) == {
            'encoder.weight', 'encoder.bias',
            'decoder.weight', 'decoder.bias',
        }
        assert set(result['skipped']) == {'head.weight', 'head.bias'}
        assert result['missing'] == []
        assert result['unexpected'] == []

    def test_return_dict_has_all_four_keys(self):
        """The returned dict always contains exactly the four expected keys."""
        model = MinimalModel(vocab=10)
        ckpt_sd = _state_dict_for(vocab=10)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        assert set(result.keys()) == {'loaded', 'skipped', 'missing', 'unexpected'}

    def test_all_return_values_are_lists(self):
        """Every value in the returned dict is a list (not a set or tuple)."""
        model = MinimalModel(vocab=10)
        ckpt_sd = _state_dict_for(vocab=10)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result = safe_load_pretrained(model, 'parseq')

        for key, value in result.items():
            assert isinstance(value, list), f'result[{key!r}] should be a list'


    # ------------------------------------------------------------------ #
    # 9. Idempotency                                                       #
    # ------------------------------------------------------------------ #

    def test_calling_twice_is_idempotent(self):
        """Calling safe_load_pretrained twice on the same model with the same
        checkpoint should produce identical results both times."""
        vocab = 10
        ckpt_sd = _state_dict_for(vocab)
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result1 = safe_load_pretrained(model, 'parseq')
        with patch(MOCK_TARGET, return_value=ckpt_sd):
            result2 = safe_load_pretrained(model, 'parseq')

        assert result1 == result2

    # ------------------------------------------------------------------ #
    # 10. Model remains in evaluation-ready state after loading            #
    # ------------------------------------------------------------------ #

    def test_model_can_forward_after_loading(self):
        """After safe_load_pretrained the model should accept a forward pass
        without raising any exception."""
        vocab = 10
        ckpt_sd = _state_dict_for(vocab)
        model = MinimalModel(vocab)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            safe_load_pretrained(model, 'parseq')

        x = torch.zeros(2, EMBED_DIM)
        out = model(x)
        assert out.shape == (2, vocab)

    def test_model_can_forward_after_loading_with_head_mismatch(self):
        """After partial loading (head skipped), the model still runs inference
        with the new vocab dimension."""
        ckpt_sd = _state_dict_for(vocab=10)
        model = MinimalModel(vocab=80)

        with patch(MOCK_TARGET, return_value=ckpt_sd):
            safe_load_pretrained(model, 'parseq')

        x = torch.zeros(3, EMBED_DIM)
        out = model(x)
        assert out.shape == (3, 80)
