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


def safe_load_pretrained(model: nn.Module, experiment: str) -> dict[str, list[str]]:
    """Load pretrained weights into *model*, skipping any incompatible tensors.

    Standard ``load_state_dict()`` requires every tensor in the checkpoint to
    match the current model exactly (same key *and* same shape).  This is fine
    when fine-tuning on the same charset, but fails with a ``RuntimeError`` the
    moment the output head (classifier) has a different size — which is always
    the case when adapting a model trained on English (e.g. 94-character vocab)
    to a multilingual script such as Marathi (Devanagari, ~80+ characters).

    Why partial loading is safe for multilingual fine-tuning
    --------------------------------------------------------
    PARSeq (and the other models in this hub) share the same architecture
    pattern: a visual encoder + optional language decoder that produce
    *script-agnostic* feature representations, followed by a thin linear
    classifier head that maps those features to per-character logits.

    * **Encoder / decoder weights** – fully reusable.  The visual backbone and
      attention mechanism learn general image features that transfer across
      scripts with no modification.
    * **Classifier head** (``head.weight``, ``head.bias`` in PARSeq; the final
      linear layer in other models) – *not* reusable when the target charset
      differs.  Its first dimension equals ``len(charset) + num_special_tokens``
      and will be a different size for every distinct charset.  Loading these
      tensors would silently corrupt the model; skipping them lets the head
      be initialised randomly and learned from scratch during fine-tuning.

    Implementation
    --------------
    Rather than using ``strict=False`` (which silently ignores *all* missing /
    unexpected keys and can hide bugs), this function performs explicit,
    key-by-key shape comparison and only copies tensors that are safe to load.

    Parameters
    ----------
    model:
        The ``nn.Module`` whose weights should be initialised.  For PARSeq this
        is ``system.model`` (the inner ``nn.Module``); for other systems it is
        the ``LightningModule`` itself.  See ``train.py`` for the call-site
        convention.
    experiment:
        Pretrained model identifier, e.g. ``'parseq'`` or ``'parseq-tiny'``.
        Passed directly to :func:`get_pretrained_weights`.

    Returns
    -------
    dict with three keys:

    * ``"loaded"``  – keys successfully copied from the checkpoint.
    * ``"skipped"`` – keys present in the checkpoint but skipped because their
      tensor shape differs from the current model (typically the classifier head).
    * ``"missing"`` – keys present in the current model but absent from the
      checkpoint (e.g. new layers added for the target language).
    """
    checkpoint = get_pretrained_weights(experiment)
    current_state = model.state_dict()

    loaded: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []

    # Build an updated state dict: start from the current model weights so that
    # any key not present in the checkpoint retains its initialised value.
    updated_state = {k: v.clone() for k, v in current_state.items()}

    for key, ckpt_tensor in checkpoint.items():
        if key not in current_state:
            # Key exists in checkpoint but not in the current model.
            # This can happen when loading an older checkpoint after an
            # architecture refactor; safe to ignore.
            skipped.append(key)
            continue
        if ckpt_tensor.shape != current_state[key].shape:
            # Shape mismatch — almost always the classifier/output head when
            # the target charset differs from the pretrained charset.
            skipped.append(key)
            continue
        updated_state[key] = ckpt_tensor
        loaded.append(key)

    # Track keys that are in the current model but were absent from the checkpoint.
    ckpt_keys = set(checkpoint.keys())
    for key in current_state:
        if key not in ckpt_keys:
            missing.append(key)

    # Load with strict=True against the already-filtered state dict so that
    # PyTorch validates the final result (all remaining keys must match).
    model.load_state_dict(updated_state, strict=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    width = 40
    print('=' * width)
    print(' Pretrained Loading '.center(width, '='))
    print('=' * width)
    print(f'  Loaded layers : {len(loaded)}')
    print(f'  Skipped layers: {len(skipped)}')
    if skipped:
        print('  Skipped:')
        for k in skipped:
            print(f'    {k}')
    if missing:
        print(f'  Missing layers: {len(missing)}')
        for k in missing:
            print(f'    {k}')
    print('  Pretrained initialization completed.')
    print('=' * width)

    return {'loaded': loaded, 'skipped': skipped, 'missing': missing}


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
