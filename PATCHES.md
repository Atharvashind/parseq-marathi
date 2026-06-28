# Patches — parseq-marathi

This document records every infrastructure and compatibility change made to this
fork of the official [PARSeq](https://github.com/baudm/parseq) repository.
**No model architecture, encoder/decoder, tokenizer, or recognition-pipeline code
was modified.**

---

## 1. Unicode Compatibility Fix

**File:** `strhub/data/dataset.py`

### Problem
The original `_preprocess_labels` method unconditionally applied
`unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()` to
every label.  
For Devanagari (and any other non-ASCII script) this silently discards all
characters, producing empty labels and an empty dataset.

### Fix
Added a helper function `_is_ascii_charset(charset: str) -> bool` that checks
whether every character in the configured charset is a plain ASCII codepoint
(< 128).

`_preprocess_labels` now branches on this flag:

| Charset type | Normalisation applied |
|---|---|
| ASCII-only (English) | `NFKD` → strip non-ASCII bytes *(original behaviour, unchanged)* |
| Non-ASCII (e.g. Devanagari) | `NFC` only — composed characters such as Devanagari matras are preserved |

### Backward compatibility
English datasets continue to use exactly the original normalisation path.
The change is gated entirely on whether the charset itself contains non-ASCII
characters, so existing configs and checkpoints are unaffected.

---

## 2. NumPy 2.x / imgaug Removal

**Files:**  
- `strhub/data/augment.py`  
- `requirements/train.in`, `requirements/train.txt`  
- `requirements/tune.in`, `requirements/tune.txt`  
- `requirements/constraints.txt`

### Problem
`imgaug==0.4.0` uses internal NumPy APIs removed in NumPy 2.0
(`np.bool`, `np.int`, `np.float`, etc.), causing an `AttributeError` at
import time on any environment with NumPy ≥ 2.0.

### Fix
Replaced the three `imgaug` augmenters with equivalent implementations using
only **NumPy** and **SciPy** (both already required by the project):

| Original (imgaug) | Replacement |
|---|---|
| `iaa.MotionBlur(k)` | `scipy.ndimage.uniform_filter1d` along the horizontal axis with a box kernel of width `k` |
| `iaa.AdditiveGaussianNoise(scale=s)` | `np.random.default_rng().normal(0, s, shape)` added to the image array |
| `iaa.AdditivePoissonNoise(lam=l)` | `np.random.default_rng().poisson(l, shape)` added to the image array |

All three functions retain the same signature (`(img: PIL.Image, param, **kwargs) -> PIL.Image`)
so the `timm` RandAugment machinery that calls them is unaffected.

The public API `rand_augment_transform(magnitude, num_layers)` is **unchanged**.

`imgaug` and its imgaug-only transitive dependencies (`imageio` via imgaug,
`opencv-python` via imgaug, `scikit-image` via imgaug, `shapely`, `tifffile` as
a scikit-image dep, `lazy-loader`) have been removed from all requirement files.
`scipy` was already a transitive dependency and remains.

---

## 3. Marathi Dataset Support

**New files:**  
- `configs/charset/marathi.yaml`  
- `configs/dataset/marathi.yaml`

### charset — `configs/charset/marathi.yaml`
Defines `model.charset_train` with the full Devanagari character inventory used
in written Marathi:

- Independent vowels (स्वर): अ आ इ ई उ ऊ ऋ ए ऐ ओ औ
- Consonants (व्यंजन): क–ह including ळ क्ष ज्ञ
- Dependent vowel signs (मात्रा) and halant (्)
- Diacritics: anusvara (ं), visarga (ः), chandrabindu (ँ), nukta (़)
- Devanagari digits: ०–९
- Special symbols: avagraha (ऽ), Om (ॐ)

### dataset — `configs/dataset/marathi.yaml`
Sets `data.train_dir: marathi` and opts into Unicode-preserving normalisation
(`normalize_unicode: true`, which now applies NFC for this non-ASCII charset).

### Usage
```bash
python train.py charset=marathi dataset=marathi
```

Expected LMDB layout under `data.root_dir`:
```
<root_dir>/
  train/
    marathi/          ← data.train_dir
      data.mdb        ← single LMDB  OR  subdirectories, each with data.mdb
      lock.mdb
  val/
    data.mdb
    lock.mdb
  test/
    data.mdb
    lock.mdb
```

---

## 4. SceneTextDataModule — Flat LMDB Layout Support

**File:** `strhub/data/dataset.py`  (`build_tree_dataset`)

### Problem
The recursive `glob('**/data.mdb')` pattern only matches LMDB stores that are
**inside** a sub-directory of `root`.  
When the LMDB lives directly at `root/data.mdb` (the flat layout used by the
Marathi dataset and many single-split datasets), the glob returns nothing and
the returned `ConcatDataset` is empty.

### Fix
After the recursive glob, if no datasets were found and `root/data.mdb` exists,
the LMDB at `root` itself is opened directly.  
Both the tree layout and the flat layout are supported transparently without any
caller changes.

---

## 5. PyTorch Lightning 2.x / PyTorch 2.x Compatibility

**Files modified:** `train.py`, `tune.py`, `strhub/models/base.py`,
`strhub/models/parseq/system.py`, `strhub/models/abinet/system.py`,
`strhub/models/crnn/system.py`, `strhub/models/vitstr/system.py`,
`strhub/models/trba/system.py`

### 5a. `STEP_OUTPUT` removed from all model imports

**Problem:** Every model system file imported `STEP_OUTPUT` from
`pytorch_lightning.utilities.types`. This internal type alias is not guaranteed
stable across PL minor versions and triggers deprecation warnings in PL 2.x.

**Fix:** Removed the import entirely. `training_step` return types are now
annotated as `Tensor` (the real returned value). `validation_step`,
`test_step`, and `_eval_step` in `base.py` are annotated as
`Optional[dict[str, Any]]`. No runtime behavior changes.

### 5b. `summarize()` → `ModelSummary` in `train.py`

**Problem:** `pytorch_lightning.utilities.model_summary.summarize()` was
deprecated in PL 1.8 and removed in PL 2.x.

**Fix:** Replaced with `ModelSummary(model, max_depth=2)` — the canonical PL
2.x API that produces identical console output.

### 5c. `torch.get_autocast_gpu_dtype()` → `torch.get_autocast_dtype('cuda')`

**Problem:** `torch.get_autocast_gpu_dtype()` was deprecated in PyTorch 2.1.
The replacement is `torch.get_autocast_dtype('cuda')`.

**Fix:** Added a private helper `_get_autocast_dtype(device_type)` in
`train.py` that calls the new API first and falls back to the old one for
PyTorch < 2.1, avoiding warnings on modern installs.

### 5d. Integer precision `16` → string `'16-mixed'` in `tune.py`

**Problem:** `config.trainer.precision = 16` (integer) is deprecated in PL 2.x.
PL 2.x requires the explicit string `'16-mixed'` to distinguish mixed-precision
from full `float16`.

**Fix:** Set `config.trainer.precision = '16-mixed'`, consistent with `train.py`.

### 5e. `config.trainer.get('gpus', 0)` → `accelerator == 'gpu'` in `tune.py`

**Problem:** The `gpus` Trainer argument was removed in PL 2.0 in favour of
`accelerator='gpu'` + `devices=N`. The old key was never populated, so GPU
precision was silently never set.

**Fix:** Changed the guard to `config.trainer.get('accelerator') == 'gpu'`,
matching `train.py`.

### 5f. `air.RunConfig(local_dir=…)` → `storage_path=` in `tune.py`

**Problem:** Ray renamed `RunConfig.local_dir` to `storage_path` in Ray 2.7.
`requirements/tune.txt` pins Ray 2.9.2, so `local_dir` raises a deprecation
warning and may be removed in a later Ray release.

**Fix:** Replaced `local_dir=str(out_dir.parent.absolute())` with
`storage_path=str(out_dir.parent.absolute())`.

---

## 6. Safe Multilingual Pretrained Weight Loading

**Files modified:** `strhub/models/utils.py`, `train.py`

### Problem
`nn.Module.load_state_dict()` in strict mode raises a `RuntimeError` when
any tensor in the checkpoint has a different shape than the current model.
This is always the case when fine-tuning on a different charset (e.g. English
→ Marathi): the classifier/output head dimension equals
`len(charset) + num_special_tokens`, so it changes with every distinct
character set. The encoder, decoder, and all other layers are fully reusable.

### New function: `safe_load_pretrained(model, experiment)`

Added to `strhub/models/utils.py`.

**Algorithm (explicit filtering — no `strict=False`)**

1. Download the checkpoint via `get_pretrained_weights(experiment)`.
2. Snapshot the current model's `state_dict()`.
3. Walk every key in the checkpoint:
   - key absent from the model → **skipped**
   - tensor shapes differ → **skipped** (classifier head for multilingual case)
   - shapes match → **loaded**
4. Identify keys present in the model but absent from the checkpoint → **missing**.
5. Merge loaded tensors into a copy of the current state dict (unmatched keys
   keep their randomly-initialised values).
6. Call `load_state_dict(updated_state, strict=True)` — PyTorch still
   validates the final result; no silent failures.
7. Print a clean summary and return `{"loaded": [...], "skipped": [...], "missing": [...]}`.

**Why `strict=False` is not used**

`strict=False` silently ignores *all* missing and unexpected keys. Using an
explicit allow-list instead means any unexpected structural difference
(e.g. a refactored layer name) is visible in the `skipped`/`missing`
lists rather than hidden.

### Behaviour by scenario

| Scenario | Loaded | Skipped |
|---|---|---|
| English pretrained → English model (same charset) | All layers | 0 |
| English pretrained → Marathi model (different charset) | Encoder + decoder | `head.weight`, `head.bias` |

### Changes to `train.py`

- Replaced `from strhub.models.utils import get_pretrained_weights` with
  `from strhub.models.utils import safe_load_pretrained`.
- Replaced `m.load_state_dict(get_pretrained_weights(config.pretrained))`
  with `safe_load_pretrained(m, config.pretrained)`.

### Changes to `create_model()` in `strhub/models/utils.py`

- Replaced `m.load_state_dict(get_pretrained_weights(experiment))` with
  `safe_load_pretrained(m, experiment)` so the programmatic API
  (`hubconf.py`, `bench.py`) also benefits from safe loading.

### Console output example (multilingual fine-tuning)

```
========================================
============ Pretrained Loading ========
========================================
  Loaded layers : 287
  Skipped layers: 2
  Skipped:
    head.weight
    head.bias
  Pretrained initialization completed.
========================================
```

---

## Summary of Modified Files

| File | Change |
|---|---|
| `strhub/data/dataset.py` | Unicode auto-detection; flat LMDB layout support |
| `strhub/data/augment.py` | imgaug replaced with NumPy/SciPy; public API unchanged |
| `train.py` | `summarize` → `ModelSummary`; autocast compat helper; `safe_load_pretrained` |
| `tune.py` | `gpus` → `accelerator` check; integer precision → `'16-mixed'`; `local_dir` → `storage_path` |
| `strhub/models/utils.py` | Added `safe_load_pretrained()`; updated `create_model()` |
| `strhub/models/base.py` | Removed `STEP_OUTPUT` import; updated return type annotations |
| `strhub/models/parseq/system.py` | Removed `STEP_OUTPUT`; `training_step` → `Tensor` |
| `strhub/models/abinet/system.py` | Removed `STEP_OUTPUT`; `training_step` → `Tensor` |
| `strhub/models/crnn/system.py` | Removed `STEP_OUTPUT`; `training_step` → `Tensor` |
| `strhub/models/vitstr/system.py` | Removed `STEP_OUTPUT`; `training_step` → `Tensor` |
| `strhub/models/trba/system.py` | Removed `STEP_OUTPUT`; both `training_step`s → `Tensor` |
| `requirements/train.in` | Removed `imgaug` |
| `requirements/train.txt` | Removed `imgaug` and imgaug-only transitive deps |
| `requirements/tune.in` | Removed `imgaug` |
| `requirements/tune.txt` | Removed `imgaug` and imgaug-only transitive deps |
| `requirements/constraints.txt` | Removed `imgaug` entry; updated `# via` annotations |
| `configs/charset/marathi.yaml` | **New** — Devanagari charset for Marathi |
| `configs/dataset/marathi.yaml` | **New** — Marathi dataset config |
| `PATCHES.md` | **New** — this file |
