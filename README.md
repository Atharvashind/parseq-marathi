## News
- **2024-02-22**: Updated for PyTorch 2.0 and Lightning 2.0
- **2024-01-16**: Featured in the [NVIDIA Developer Blog](https://developer.nvidia.com/blog/robust-scene-text-detection-and-recognition-introduction/)
- **2023-11-18**: [Interview with Deci AI at ECCV 2022](https://deeplearningdaily.substack.com/p/exclusive-interview-with-a-researcher) published
- **2023-09-07**: [Added](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/algorithm_rec_parseq_en.md) to [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), one of the most popular multilingual OCR toolkits
- **2023-06-15**: [Added](https://mindee.github.io/doctr/modules/models.html#doctr.models.recognition.parseq) to [docTR](https://github.com/mindee/doctr), a deep learning-based library for OCR
- **2022-07-14**: Initial public release (ranked #1 overall for STR on [Papers With Code](https://paperswithcode.com/paper/scene-text-recognition-with-permuted) at the time of release)
- **2022-07-04**: Accepted at ECCV 2022

<div align="center">

# Scene Text Recognition with<br/>Permuted Autoregressive Sequence Models
### Extended for Marathi (Devanagari) Scene Text Recognition

[![Apache License 2.0](https://img.shields.io/github/license/baudm/parseq)](https://github.com/baudm/parseq/blob/main/LICENSE)
[![arXiv preprint](http://img.shields.io/badge/arXiv-2207.06966-b31b1b)](https://arxiv.org/abs/2207.06966)
[![In Proc. ECCV 2022](http://img.shields.io/badge/ECCV-2022-6790ac)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/556_ECCV_2022_paper.php)
[![Gradio demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-Gradio-ff7c00)](https://huggingface.co/spaces/baudm/PARSeq-OCR)

[**Darwin Bautista**](https://github.com/baudm) and [**Rowel Atienza**](https://github.com/roatienza)

Electrical and Electronics Engineering Institute<br/>
University of the Philippines, Diliman

[Method](#method-tldr) | [Marathi Extension](#marathi-extension) | [Getting Started](#getting-started) | [Training](#training) | [Marathi Training](#marathi-training) | [Evaluation](#evaluation) | [Inference](#inference) | [FAQ](#frequently-asked-questions) | [Citation](#citation)

</div>

> **This fork** extends the official PARSeq repository with full Marathi (Devanagari) scene-text recognition support, NumPy 2.x / PyTorch Lightning 2.x compatibility fixes, safe multilingual weight loading, staged fine-tuning, and evaluation/inference tools. The model architecture, tokenizer, and training loop are **unchanged**. See [`PATCHES.md`](PATCHES.md) for a complete change log.

---

Scene Text Recognition (STR) models use language context to be more robust against noisy or corrupted images. Recent approaches like ABINet use a standalone or external Language Model (LM) for prediction refinement. In this work, we show that the external LM&mdash;which requires upfront allocation of dedicated compute capacity&mdash;is inefficient for STR due to its poor performance vs cost characteristics. We propose a more efficient approach using **p**ermuted **a**uto**r**egressive **seq**uence (PARSeq) models. View our ECCV [poster](https://drive.google.com/file/d/19luOT_RMqmafLMhKQQHBnHNXV7fOCRfw/view) and [presentation](https://drive.google.com/file/d/11VoZW4QC5tbMwVIjKB44447uTiuCJAAD/view) for a brief overview.

![PARSeq](.github/gh-teaser.png)

**NOTE:** _P-S and P-Ti are shorthands for PARSeq-S and PARSeq-Ti, respectively._

### Method tl;dr

Our main insight is that with an ensemble of autoregressive (AR) models, we could unify the current STR decoding methods (context-aware AR and context-free non-AR) and the bidirectional (cloze) refinement model:
<div align="center"><img src=".github/contexts-example.png" alt="Unified STR model" width="75%"/></div>

A single Transformer can realize different models by merely varying its attention mask. With the correct decoder parameterization, it can be trained with Permutation Language Modeling to enable inference for arbitrary output positions given arbitrary subsets of the input context. This *arbitrary decoding* characteristic results in a _unified_ STR model&mdash;PARSeq&mdash;capable of context-free and context-aware inference, as well as iterative prediction refinement using bidirectional context **without** requiring a standalone language model. PARSeq can be considered an ensemble of AR models with shared architecture and weights:

![System](.github/system.png)

### Sample Results
<div align="center">

| Input Image                                                                | PARSeq-S<sub>A</sub> | ABINet            | TRBA              | ViTSTR-S          | CRNN              |
|:--------------------------------------------------------------------------:|:--------------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| <img src="demo_images/art-01107.jpg" alt="CHEWBACCA" width="128"/>         | CHEWBACCA            | CHEWBA**GG**A     | CHEWBACCA         | CHEWBACCA         | CHEW**U**ACCA     |
| <img src="demo_images/coco-1166773.jpg" alt="Chevron" width="128"/>        | Chevro**l**          | Chevro\_          | Chevro\_          | Chevr\_\_         | Chevr\_\_         |
| <img src="demo_images/cute-184.jpg" alt="SALMON" height="128"/>            | SALMON               | SALMON            | SALMON            | SALMON            | SA\_MON           |
| <img src="demo_images/ic13_word_256.png" alt="Verbandstoffe" width="128"/> | Verbandst**e**ffe    | Verbandst**e**ffe | Verbandst**ell**e | Verbandst**e**ffe | Verbands**le**ffe |
| <img src="demo_images/ic15_word_26.png" alt="Kappa" width="128"/>          | Kappa                | Kappa             | Ka**s**pa         | Kappa             | Ka**ad**a         |
| <img src="demo_images/uber-27491.jpg" alt="3rdAve" height="128"/>          | 3rdAve               | 3=-Ave            | 3rdAve            | 3rdAve            | **Coke**          |

**NOTE:** _Bold letters and underscores indicate wrong and missing character predictions, respectively._
</div>

---

## Marathi Extension

This fork adds first-class Marathi (Devanagari) support to PARSeq. All changes
are infrastructure-only — the model architecture, tokenizer, and training loop
are identical to the upstream repository.

### What was added

| Area | Change |
|---|---|
| **Unicode** | NFC normalisation (not NFKD) for non-ASCII charsets; English datasets unchanged |
| **imgaug removal** | Replaced with NumPy/SciPy equivalents compatible with NumPy 2.x |
| **LMDB layout** | Flat `root/data.mdb` layout supported alongside the existing tree layout |
| **Charset** | `configs/charset/marathi.yaml` — full Devanagari inventory |
| **Dataset config** | `configs/dataset/marathi.yaml` |
| **Safe weight loading** | `safe_load_pretrained()` skips mismatched classifier layers for cross-charset transfer |
| **Staged fine-tuning** | Per-component freeze flags + differential learning rates via Hydra config |
| **Evaluation tool** | `tools/evaluate_marathi.py` — Exact Match, CER, NED, CSV export |
| **Inference tool** | `tools/infer_marathi.py` — single image or folder, CSV export |
| **PL 2.x compat** | Removed deprecated `STEP_OUTPUT`, `summarize()`, integer precision, `gpus` flag |

### Charset

The Marathi charset covers the full Devanagari Unicode block used in written Marathi:
vowels (अ–औ), consonants (क–ह, ळ, क्ष, ज्ञ), dependent vowel signs (मात्रा),
halant (्), diacritics (anusvara ं, visarga ः, chandrabindu ँ), Devanagari digits (०–९),
and common punctuation.

---

## Getting Started

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

```bash
# Use specific platform build. Other PyTorch 2.0 options: cu118, cu121, rocm5.7
platform=cpu
make torch-${platform}
pip install -r requirements/core.${platform}.txt -e .[train,test]
```

#### Updating dependency version pins
```bash
pip install pip-tools
make clean-reqs reqs
```

### Datasets

Download the [datasets](Datasets.md) from the following links:
1. [LMDB archives](https://drive.google.com/drive/folders/1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE) for MJSynth, SynthText, IIIT5k, SVT, SVTP, IC13, IC15, CUTE80, ArT, RCTW17, ReCTS, LSVT, MLT19, COCO-Text, and Uber-Text.
2. [LMDB archives](https://drive.google.com/drive/folders/1D9z_YJVa6f-O0juni-yG5jcwnhvYw-qC) for TextOCR and OpenVINO.

For Marathi datasets, the expected LMDB layout under `data.root_dir` is:

```
data/
  train/
    marathi/
      data.mdb        ← single LMDB or subdirectories each with data.mdb
      lock.mdb
  val/
    data.mdb
    lock.mdb
  test/
    data.mdb
    lock.mdb
```

### Pretrained Models via Torch Hub

```python
import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule

parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

img = Image.open('/path/to/image.png').convert('RGB')
img = img_transform(img).unsqueeze(0)

logits = parseq(img)
pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label = {}'.format(label[0]))
```

---

## Training

The training script can train any supported model. Use `./train.py --help` to see the default configuration.

<details><summary>Sample commands for standard training</summary><p>

### Finetune using pretrained weights
```bash
./train.py +experiment=parseq-tiny pretrained=parseq-tiny
```

### Train a model variant
```bash
./train.py +experiment=parseq-tiny
```

### Specify the character set
```bash
./train.py charset=94_full  # Other options: 36_lowercase, 62_mixed-case
```

### Specify the training dataset
```bash
./train.py dataset=real  # Other option: synth
```

### Change model parameters
```bash
./train.py model.img_size=[32, 128] model.max_label_length=25 model.batch_size=384
```

### Change data parameters
```bash
./train.py data.root_dir=data data.num_workers=2 data.augment=true
```

### Change Trainer parameters
```bash
./train.py trainer.max_epochs=20 trainer.accelerator=gpu trainer.devices=2
```

### Resume from checkpoint
```bash
./train.py +experiment=<model_exp> ckpt_path=outputs/<model>/<timestamp>/checkpoints/<checkpoint>.ckpt
```

</p></details>

---

## Marathi Training

### Quick start — fine-tune from English pretrained weights

```bash
python train.py \
  experiment=finetune_marathi \
  charset=marathi \
  dataset=marathi \
  pretrained=parseq
```

`safe_load_pretrained()` automatically loads all compatible layers (encoder,
decoder, position embeddings) and skips only the classifier head, which is
re-initialised for the Marathi vocab size. A loading summary is printed:

```
================================================
        Safe Pretrained Loading
================================================
  Checkpoint tensors : 289
  Model tensors      : 289
  Loaded             : 287
  Skipped            : 2
  Skipped layers:
    head.weight
    head.bias
================================================
```

### Staged fine-tuning experiments

The `finetune_marathi` experiment config supports three common transfer learning
strategies via Hydra CLI overrides — no code changes needed.

**Experiment 1 — freeze encoder + decoder, train head only with differential LRs**
```bash
python train.py experiment=finetune_marathi charset=marathi dataset=marathi \
  pretrained=parseq \
  model.freeze.encoder=true \
  model.freeze.decoder=true \
  model.backbone_lr=1e-5 \
  model.head_lr=5e-4
```

**Experiment 2 — freeze encoder only**
```bash
python train.py experiment=finetune_marathi charset=marathi dataset=marathi \
  pretrained=parseq \
  model.freeze.encoder=true \
  model.freeze.decoder=false
```

**Experiment 3 — train all layers (standard fine-tuning)**
```bash
python train.py charset=marathi dataset=marathi pretrained=parseq
```

A fine-tuning summary is printed at the start of training:

```
================================================
       Fine-tuning Configuration
================================================
  Encoder     : Frozen
  Decoder     : Trainable
  Head        : Trainable
  Text Embed  : Trainable
  Backbone LR : 1e-05
  Head LR     : 0.0005
================================================
```

### Freeze configuration reference

```yaml
# In your experiment config or as CLI overrides
model:
  freeze:
    encoder:    true   # freeze the ViT backbone
    decoder:    false  # keep decoder trainable
    head:       false  # always train (re-initialised for new charset)
    text_embed: false  # always train (new vocab embeddings)

  backbone_lr: 1.0e-5   # LR for encoder + decoder (when not null)
  head_lr:     5.0e-4   # LR for head + text_embed (when not null)
```

If `backbone_lr` and `head_lr` are both `null` (the default), the standard
single learning-rate optimizer is used unchanged.

---

## Evaluation

### Standard English benchmark evaluation

```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt
# or
./test.py pretrained=parseq
```

### Marathi validation set evaluation

```bash
python tools/evaluate_marathi.py \
  --checkpoint outputs/parseq/<timestamp>/checkpoints/best.ckpt \
  --lmdb_path  data/val \
  --batch_size 64 \
  --output     evaluation_predictions.csv
```

**Sample output:**
```
================================================
         Evaluation Results
================================================
  Samples          : 5000
  Exact Match Acc  : 82.34 %
  Character Acc    : 96.12 %
  CER              :  3.88 %
  NED              : 94.57 %
================================================
Predictions saved to: evaluation_predictions.csv
```

Output CSV columns: `index`, `ground_truth`, `prediction`, `confidence`.

All string comparisons use NFC normalisation (not NFKD) to correctly handle
Devanagari matras and composed characters.

<details><summary>Standard benchmark commands</summary><p>

### Lowercase alphanumeric comparison (Table 6)
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt
```

### Mixed-case and punctuation
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation
```

### New benchmark datasets (Table 5)
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --new
```

### Benchmark compute requirements
```bash
./bench.py model=parseq model.decode_ar=false model.refine_iters=3
```

### Orientation robustness
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --rotation 90
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --rotation 180
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --rotation 270
```

</p></details>

---

## Inference

### Original inference script (English)

```bash
./read.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --images demo_images/*
./read.py pretrained=parseq refine_iters:int=2 decode_ar:bool=false --images demo_images/*
```

### Marathi inference tool

**Single image**
```bash
python tools/infer_marathi.py \
  --checkpoint best.ckpt \
  --image      word.jpg
```

Output:
```
Prediction : मराठी
Confidence : 0.9231
```

**Folder of images**
```bash
python tools/infer_marathi.py \
  --checkpoint best.ckpt \
  --folder     images/ \
  --output     predictions.csv
```

Output CSV columns: `filename`, `prediction`, `confidence`.

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
Images in a folder are processed in alphabetical order.

---

## Tuning

```bash
./tune.py tune.num_samples=20
./tune.py +experiment=tune_abinet-lm
```

---

## Frequently Asked Questions

- How do I train on a new language? See Issues [#5](https://github.com/baudm/parseq/issues/5) and [#9](https://github.com/baudm/parseq/issues/9). For Marathi specifically, see the [Marathi Training](#marathi-training) section above.
- Can you export to TorchScript or ONNX? Yes, see Issue [#12](https://github.com/baudm/parseq/issues/12#issuecomment-1267842315).
- How do I test on my own dataset? See Issue [#27](https://github.com/baudm/parseq/issues/27).
- How do I finetune a custom dataset? See Issue [#7](https://github.com/baudm/parseq/issues/7) and the [Staged Fine-tuning](#staged-fine-tuning-experiments) section above.
- What is `val_NED`? See Issue [#10](https://github.com/baudm/parseq/issues/10).
- Why does loading a pretrained checkpoint fail with a shape mismatch? The classifier head size depends on the charset. Use `safe_load_pretrained()` (called automatically via the `pretrained=` flag) which skips incompatible layers instead of failing.

---

## Citation

```bibtex
@InProceedings{bautista2022parseq,
  title={Scene Text Recognition with Permuted Autoregressive Sequence Models},
  author={Bautista, Darwin and Atienza, Rowel},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  month={10},
  year={2022},
  publisher={Springer Nature Switzerland},
  address={Cham},
  doi={10.1007/978-3-031-19815-1_11},
  url={https://doi.org/10.1007/978-3-031-19815-1_11}
}
```
