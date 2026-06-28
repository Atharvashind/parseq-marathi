# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: imgaug has been replaced with NumPy / SciPy equivalents so that this
# module is compatible with NumPy 2.x.  The public API (rand_augment_transform)
# is unchanged.

from functools import partial

import numpy as np
from PIL import Image, ImageFilter

from timm.data import auto_augment

from strhub.data import aa_overrides

aa_overrides.apply()

_OP_CACHE = {}


def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def _get_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))


# ---------------------------------------------------------------------------
# Blur ops
# ---------------------------------------------------------------------------

def gaussian_blur(img, radius, **__):
    radius = _get_param(radius, img, 0.02)
    key = 'gaussian_blur_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)


def motion_blur(img, k, **__):
    """Horizontal motion blur implemented with scipy.ndimage (no imgaug)."""
    k = _get_param(k, img, 0.08, 3) | 1  # bin to odd values
    from scipy.ndimage import uniform_filter1d
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        result = uniform_filter1d(arr, size=k, axis=1, mode='reflect')
    else:
        result = np.stack(
            [uniform_filter1d(arr[..., c], size=k, axis=1, mode='reflect')
             for c in range(arr.shape[2])],
            axis=2,
        )
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Noise ops
# ---------------------------------------------------------------------------

def gaussian_noise(img, scale, **_):
    """Additive Gaussian noise implemented with NumPy (no imgaug)."""
    scale = _get_param(scale, img, 0.25) | 1  # bin to odd values
    arr = np.asarray(img, dtype=np.float32)
    rng = np.random.default_rng()
    noise = rng.normal(0.0, float(scale), arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def poisson_noise(img, lam, **_):
    """Additive Poisson noise implemented with NumPy (no imgaug)."""
    lam = _get_param(lam, img, 0.2) | 1  # bin to odd values
    arr = np.asarray(img, dtype=np.float32)
    rng = np.random.default_rng()
    noise = rng.poisson(float(lam), arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# RandAugment integration
# ---------------------------------------------------------------------------

def _level_to_arg(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return (level,)


_RAND_TRANSFORMS = auto_augment._RAND_INCREASING_TRANSFORMS.copy()
_RAND_TRANSFORMS.remove('SharpnessIncreasing')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.extend([
    'GaussianBlur',
    # 'MotionBlur',
    # 'GaussianNoise',
    'PoissonNoise',
])
auto_augment.LEVEL_TO_ARG.update({
    'GaussianBlur': partial(_level_to_arg, max=4),
    'MotionBlur': partial(_level_to_arg, max=20),
    'GaussianNoise': partial(_level_to_arg, max=0.1 * 255),
    'PoissonNoise': partial(_level_to_arg, max=40),
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': gaussian_blur,
    'MotionBlur': motion_blur,
    'GaussianNoise': gaussian_noise,
    'PoissonNoise': poisson_noise,
})


def rand_augment_transform(magnitude=5, num_layers=3):
    """Return a RandAugment transform compatible with torchvision's Compose.

    The signature is identical to the original so all call-sites are unchanged.
    """
    # These are tuned for magnitude=5, which means that effective magnitudes are half of these values.
    hparams = {
        'rotate_deg': 30,
        'shear_x_pct': 0.9,
        'shear_y_pct': 0.2,
        'translate_x_pct': 0.10,
        'translate_y_pct': 0.30,
    }
    ra_ops = auto_augment.rand_augment_ops(magnitude, hparams=hparams, transforms=_RAND_TRANSFORMS)
    # Supply weights to disable replacement in random selection (i.e. avoid applying the same op twice)
    choice_weights = [1.0 / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)
