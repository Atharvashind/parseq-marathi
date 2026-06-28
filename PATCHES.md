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

### New helpers (private)

#### `_extract_state_dict(checkpoint)`
Detects and unwraps three common checkpoint formats:

| Format | Detection | Action |
|---|---|---|
| Plain state dict | No `state_dict` or `model` key | Use as-is |
| PyTorch Lightning | `"state_dict"` key present | Extract `checkpoint["state_dict"]` |
| Generic wrapper | `"model"` key present | Extract `checkpoint["model"]` |

#### `_strip_prefix(state_dict, prefixes)`
Strips leading `model.` or `module.` from checkpoint keys before matching,
so checkpoints saved from a `LightningModule` wrapper or `DataParallel` model
align correctly with a bare `nn.Module` state dict without manual key renaming.

### `safe_load_pretrained(model, experiment)` — updated algorithm

1. Download via `get_pretrained_weights(experiment)`.
2. **Extract** flat state dict via `_extract_state_dict()`.
3. **Normalise** key prefixes via `_strip_prefix()`.
4. Compare every checkpoint key against the current model:
   - shape matches → **loaded**
   - shape differs → **skipped** (classifier head for cross-charset case)
   - absent from model → **unexpected**
5. Collect model keys absent from checkpoint → **missing**.
6. Merge loaded tensors into a copy of the current state dict.
7. `load_state_dict(strict=True)` — PyTorch validates the final result.
8. Print detailed summary; return all four lists.

**Why `strict=False` is not used**

`strict=False` silently ignores *all* missing and unexpected keys, hiding
typos, refactoring errors, and format changes. Explicit filtering surfaces
every discrepancy in the printed summary while still producing a model that
PyTorch considers fully valid (`strict=True` on the filtered dict).

### Return value (updated)

```python
{
    "loaded":     [...],   # shape-compatible, copied from checkpoint
    "skipped":    [...],   # shape mismatch (classifier head)
    "missing":    [...],   # in model, absent from checkpoint
    "unexpected": [...],   # in checkpoint, absent from model
}
```

### Behaviour by scenario

| Scenario | Loaded | Skipped | Missing | Unexpected |
|---|---|---|---|---|
| English → English (same charset) | All | 0 | 0 | 0 |
| English → Marathi (different charset) | Encoder + decoder | `head.weight`, `head.bias` | 0 | 0 |
| Old checkpoint (missing new layers) | All compatible | 0 | New layer keys | 0 |
| Lightning ckpt with `model.` prefix | All (after strip) | shape mismatches only | 0 | 0 |

### Console output (multilingual fine-tuning)

```
================================================
           Safe Pretrained Loading
================================================
  Checkpoint tensors : 289
  Model tensors      : 289
  Loaded             : 287
  Skipped            : 2
  Missing            : 0
  Unexpected         : 0
  Skipped layers:
    head.weight
    head.bias
================================================
```

---

## 7. Transfer Learning Support

**Context:** `strhub/models/utils.py`, `train.py`, `configs/charset/marathi.yaml`,
`configs/dataset/marathi.yaml`

### Overview

The English pretrained PARSeq checkpoint can serve as a strong initialisation
point for Marathi scene-text recognition, following standard transfer learning
practice for OCR.

### What transfers

| Component | Transfers? | Reason |
|---|---|---|
| ViT encoder (patch embedding + transformer blocks) | ✅ Yes | Learns script-agnostic stroke and spatial features |
| Autoregressive decoder (attention layers) | ✅ Yes | Sequence modelling is character-system independent |
| Position embeddings | ✅ Yes | Same image resolution and patch layout |
| Layer norms, projection layers | ✅ Yes | Same shapes regardless of charset |
| Classifier head (`head.weight`, `head.bias`) | ❌ No | Output dimension = `len(charset) + specials`; differs per script |

### What is re-initialised

Only the classifier head layers are skipped and re-initialised from scratch
(random truncated normal, matching `init_weights()`). All other parameters
start from the English pretrained values and are fine-tuned jointly.

### Training command

```bash
python train.py \
  charset=marathi \
  dataset=marathi \
  pretrained=parseq
```

`safe_load_pretrained()` handles the shape mismatch automatically and prints
a loading summary confirming which layers were transferred and which were skipped.

### Why this works

Transfer learning from Latin-script OCR to Indic scripts is well-established.
The visual encoder trained on English text images already learns robust
low-level features (edges, curves, junctions) that are equally relevant for
Devanagari. Starting from these weights rather than random initialisation
typically accelerates convergence and improves final accuracy, especially when
the Marathi training set is smaller than the English one.

---

## 8. Evaluation and Inference Tools

**New files:** `tools/evaluate_marathi.py`, `tools/infer_marathi.py`

No existing files were modified.

### `tools/evaluate_marathi.py`

Evaluates a checkpoint against any LMDB validation split and writes a CSV.

```bash
python tools/evaluate_marathi.py \
    --checkpoint outputs/parseq/best.ckpt \
    --data_root  data \
    --lmdb_path  data/val \
    --batch_size 64 \
    --output     evaluation_predictions.csv
```

**What it reuses from the repository**
- `load_from_checkpoint()` — model loading, hyperparameter recovery
- `LmdbDataset` — LMDB reading, label filtering
- `SceneTextDataModule.get_transform()` — image preprocessing pipeline
- `model.tokenizer.decode()` — sequence decoding (no duplication)
- `model.charset_adapter` — post-decode character filtering

**Metrics** (all NFC-normalised, not NFKD)

| Metric | Definition |
|---|---|
| Exact Match Accuracy | `correct / total × 100` |
| Character Accuracy | `(1 - CER) × 100`, clamped to 0 |
| CER | `edit_distance(pred, gt) / len(gt)`, averaged |
| NED | `1 - mean(edit_distance / max(len(pred), len(gt)))` — ICDAR 2019 |

**Output CSV columns:** `index`, `ground_truth`, `prediction`, `confidence`

---

### `tools/infer_marathi.py`

Runs inference on a single image or every image in a folder.

```bash
# Single image
python tools/infer_marathi.py --checkpoint best.ckpt --image word.jpg

# Folder
python tools/infer_marathi.py --checkpoint best.ckpt --folder images/
```

**Single-image stdout output**

```
Prediction : मराठी
Confidence : 0.9231
```

**Output CSV columns:** `filename`, `prediction`, `confidence`

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
Images in a folder are processed in alphabetical order for reproducibility.
Predictions are NFC-normalised before printing and saving.

---

## Summary of Modified Files

| File | Change |
|---|---|
| `strhub/data/dataset.py` | Unicode auto-detection; flat LMDB layout support |
| `strhub/data/augment.py` | imgaug replaced with NumPy/SciPy; public API unchanged |
| `train.py` | `summarize` → `ModelSummary`; autocast compat helper; `safe_load_pretrained` |
| `tune.py` | `gpus` → `accelerator` check; integer precision → `'16-mixed'`; `local_dir` → `storage_path` |
| `strhub/models/utils.py` | Added `_extract_state_dict()`, `_strip_prefix()`, improved `safe_load_pretrained()`; updated `create_model()` |
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
| `tools/evaluate_marathi.py` | **New** — LMDB evaluation with metrics + CSV export |
| `tools/infer_marathi.py` | **New** — single-image and folder inference + CSV export |
| `PATCHES.md` | **New** — this file |
