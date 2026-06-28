from pathlib import PurePath
from typing import Sequence

import yaml

import torch
from torch import nn


class InvalidModelError(RuntimeError):
    """Exception raised for any model-related error (creation, loading)"""


_WEIGHTS_URL = {
    'parseq-tiny': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt',
    'parseq-patch16-224': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_small_patch16_224-fcf06f5a.pt',
    'parseq': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
    'abinet': 'https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt',
    'trba': 'https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt',
    'vitstr': 'https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt',
    'crnn': 'https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt',
}


def _get_config(experiment: str, **kwargs):
    """Emulates hydra config resolution"""
    root = PurePath(__file__).parents[2]
    with open(root / 'configs/main.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / 'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    # Apply base model config
    model = exp['defaults'][0]['override /model']
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    # Apply experiment config
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    # Workaround for now: manually cast the lr to the correct type.
    config['lr'] = float(config['lr'])
    return config


def _get_model_class(key):
    if 'abinet' in key:
        from .abinet.system import ABINet as ModelClass
    elif 'crnn' in key:
        from .crnn.system import CRNN as ModelClass
    elif 'parseq' in key:
        from .parseq.system import PARSeq as ModelClass
    elif 'trba' in key:
        from .trba.system import TRBA as ModelClass
    elif 'trbc' in key:
        from .trba.system import TRBC as ModelClass
    elif 'vitstr' in key:
        from .vitstr.system import ViTSTR as ModelClass
    else:
        raise InvalidModelError(f"Unable to find model class for '{key}'")
    return ModelClass


def get_pretrained_weights(experiment):
    try:
        url = _WEIGHTS_URL[experiment]
    except KeyError:
        raise InvalidModelError(f"No pretrained weights found for '{experiment}'") from None
    return torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)


def _extract_state_dict(checkpoint: dict) -> dict:
    """Extract a plain ``state_dict`` from any common checkpoint format.

    Handles three layouts seen in the wild:

    * **Plain state dict** – the file *is* the state dict (keys are parameter
      names mapping directly to tensors).  Used by the official PARSeq weights
      downloaded via ``get_pretrained_weights()``.
    * **PyTorch Lightning checkpoint** – a dict that contains a ``"state_dict"``
      key alongside trainer metadata (``epoch``, ``global_step``, …).
    * **Generic model checkpoint** – a dict that contains a ``"model"`` key
      (common in many custom training scripts).

    Parameters
    ----------
    checkpoint:
        The raw object returned by ``torch.hub.load_state_dict_from_url`` or
        ``torch.load``.

    Returns
    -------
    A flat dict mapping parameter names → tensors, ready for key comparison.
    """
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    if 'model' in checkpoint:
        return checkpoint['model']
    # Assume it is already a plain state dict.
    return checkpoint


def _strip_prefix(state_dict: dict, prefixes: tuple[str, ...] = ('model.', 'module.')) -> dict:
    """Remove well-known wrapper prefixes from checkpoint keys.

    Checkpoint keys sometimes carry a leading ``model.`` (from a
    ``LightningModule`` that wraps an inner ``nn.Module``) or ``module.``
    (from ``DataParallel`` / ``DistributedDataParallel``).  Stripping these
    prefixes normalises the key space so that the checkpoint can be matched
    against a bare ``nn.Module`` state dict without any manual renaming.

    Only one prefix is stripped per key (the first matching one).  If a key
    does not start with any of the given prefixes it is returned unchanged.

    Parameters
    ----------
    state_dict:
        Raw checkpoint state dict, potentially containing prefixed keys.
    prefixes:
        Tuple of prefix strings to try, in order.  Defaults to the two most
        common ones: ``'model.'`` and ``'module.'``.

    Returns
    -------
    A new dict with prefixes removed where applicable.
    """
    stripped: dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                break
        stripped[new_key] = value
    return stripped


def safe_load_pretrained(model: nn.Module, experiment: str) -> dict[str, list[str]]:
    """Load pretrained weights into *model*, skipping any incompatible tensors.

    Motivation — multilingual OCR changes the classifier dimensions
    --------------------------------------------------------------
    PARSeq (and the other models in this hub) follow a consistent architecture:

    ``Image → Encoder (ViT backbone) → Decoder (attention) → Head (linear)``

    The **encoder** and **decoder** learn general visual and sequential
    features that are *script-agnostic* — they transfer well across languages
    because they capture stroke patterns, spatial relationships, and sequence
    context that are useful regardless of whether the target script is Latin,
    Devanagari, or any other writing system.

    The **classifier head** (``head.weight`` / ``head.bias`` in PARSeq; the
    final linear projection in ABINet, CRNN, ViTSTR, TRBA) is the only
    layer tied to the vocabulary size.  Its weight matrix has shape
    ``(len(charset_train) + num_special_tokens, embed_dim)``.  When the target
    charset differs from the pretrained charset — e.g. switching from 94
    English characters to ~80 Devanagari characters — this dimension changes
    and the tensor **cannot** be reused.

    Why incompatible classifier layers are intentionally skipped
    ------------------------------------------------------------
    Forcing a size-mismatched tensor into the model would either raise a
    ``RuntimeError`` (strict loading) or silently corrupt activations
    (truncation/padding).  The correct approach for transfer learning is to:

    1. Load the encoder/decoder weights from the source model (same shape).
    2. Re-initialise the classifier head randomly (new vocab size).
    3. Fine-tune end-to-end on the target language dataset.

    This is standard practice in cross-lingual OCR and NLP transfer learning.

    Why ``strict=False`` is not used
    ---------------------------------
    ``nn.Module.load_state_dict(strict=False)`` silently ignores *all* missing
    and unexpected keys.  A typo in a layer name, an accidental architecture
    divergence, or a subtle checkpoint format change would go undetected.

    Instead, this function performs **explicit key-by-key shape comparison**:

    1. Extract the state dict from the checkpoint (supports plain, Lightning,
       and ``{"model": ...}`` formats).
    2. Normalise common key prefixes (``model.``, ``module.``).
    3. Compare each checkpoint key against the current model by shape.
    4. Build a filtered copy of the current state dict, replacing only
       shape-compatible tensors.
    5. Call ``load_state_dict(strict=True)`` on the filtered dict — PyTorch
       still validates that every key in the model is accounted for.

    Parameters
    ----------
    model:
        The ``nn.Module`` to initialise.  For PARSeq this is ``system.model``
        (the inner ``nn.Module``); for other systems it is the
        ``LightningModule`` itself.  See ``train.py`` for the call-site
        convention.
    experiment:
        Pretrained model identifier (e.g. ``'parseq'``, ``'parseq-tiny'``).
        Passed directly to :func:`get_pretrained_weights`.

    Returns
    -------
    dict with four keys:

    * ``"loaded"``     – keys copied successfully from the checkpoint.
    * ``"skipped"``    – checkpoint keys skipped due to shape mismatch or
                          absence from the model (typically the classifier head
                          when fine-tuning on a different charset).
    * ``"missing"``    – model keys absent from the checkpoint (new layers not
                          present in the pretrained weights).
    * ``"unexpected"`` – checkpoint keys that had no corresponding model key
                          after prefix normalisation.
    """
    raw_checkpoint = get_pretrained_weights(experiment)
    # Step 1 — extract a plain state dict from whatever format was loaded.
    raw_state = _extract_state_dict(raw_checkpoint)
    # Step 2 — strip common wrapper prefixes so keys match bare nn.Module keys.
    ckpt_state = _strip_prefix(raw_state)

    current_state = model.state_dict()
    ckpt_keys = set(ckpt_state.keys())
    model_keys = set(current_state.keys())

    loaded: list[str] = []
    skipped: list[str] = []      # shape mismatch (present in both, but incompatible)
    missing: list[str] = []      # in model, absent from checkpoint
    unexpected: list[str] = []   # in checkpoint, absent from model

    # Step 3 — build a filtered state dict starting from current model weights.
    # Keys not overwritten will retain their randomly-initialised values.
    updated_state = {k: v.clone() for k, v in current_state.items()}

    for key, ckpt_tensor in ckpt_state.items():
        if key not in current_state:
            # Present in checkpoint but not in the current model.
            unexpected.append(key)
            continue
        if ckpt_tensor.shape != current_state[key].shape:
            # Shape mismatch — almost always the classifier head when the
            # target charset differs from the pretrained charset.
            skipped.append(key)
            continue
        updated_state[key] = ckpt_tensor
        loaded.append(key)

    # Collect model keys absent from the (prefix-normalised) checkpoint.
    for key in model_keys:
        if key not in ckpt_keys:
            missing.append(key)

    # Step 4 — strict=True load against the pre-filtered dict.
    # Every model key is present (unchanged or replaced), so PyTorch's
    # validation passes while we retain explicit control over what was loaded.
    model.load_state_dict(updated_state, strict=True)

    # ── Console summary ───────────────────────────────────────────────────────
    sep = '=' * 48
    print(sep)
    print('Safe Pretrained Loading'.center(48))
    print(sep)
    print(f'  Checkpoint tensors : {len(ckpt_state)}')
    print(f'  Model tensors      : {len(current_state)}')
    print(f'  Loaded             : {len(loaded)}')
    print(f'  Skipped            : {len(skipped)}')
    print(f'  Missing            : {len(missing)}')
    print(f'  Unexpected         : {len(unexpected)}')
    if skipped:
        print('  Skipped layers:')
        for k in skipped:
            print(f'    {k}')
    if missing:
        print('  Missing layers:')
        for k in missing:
            print(f'    {k}')
    if unexpected:
        print('  Unexpected layers:')
        for k in unexpected:
            print(f'    {k}')
    print(sep)

    return {
        'loaded': loaded,
        'skipped': skipped,
        'missing': missing,
        'unexpected': unexpected,
    }


def create_model(experiment: str, pretrained: bool = False, **kwargs):
    try:
        config = _get_config(experiment, **kwargs)
    except FileNotFoundError:
        raise InvalidModelError(f"No configuration found for '{experiment}'") from None
    ModelClass = _get_model_class(experiment)
    model = ModelClass(**config)
    if pretrained:
        m = model.model if 'parseq' in experiment else model
        # Use safe_load_pretrained so that models built with a different charset
        # (e.g. for multilingual fine-tuning) can still reuse the encoder and
        # decoder weights even when the classifier head size differs.
        safe_load_pretrained(m, experiment)
    return model


def load_from_checkpoint(checkpoint_path: str, **kwargs):
    if checkpoint_path.startswith('pretrained='):
        model_id = checkpoint_path.split('=', maxsplit=1)[1]
        model = create_model(model_id, True, **kwargs)
    else:
        ModelClass = _get_model_class(checkpoint_path)
        model = ModelClass.load_from_checkpoint(checkpoint_path, **kwargs)
    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true'  # special handling for bool
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)
    return kwargs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
