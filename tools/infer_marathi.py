#!/usr/bin/env python3
"""Run inference with a PARSeq checkpoint on one image or a folder of images.

Single image
------------
python tools/infer_marathi.py \\
    --checkpoint best.ckpt \\
    --image      image.jpg

Folder of images
----------------
python tools/infer_marathi.py \\
    --checkpoint best.ckpt \\
    --folder     images/

Both modes print per-image results to stdout and write predictions.csv.

Output CSV columns
------------------
filename  |  prediction  |  confidence

Notes
-----
- Images are loaded as RGB and resized to the model's expected input size.
- Predictions are NFC-normalised before printing and saving.
- Supported image extensions: .jpg .jpeg .png .bmp .tiff .webp
- When --folder is given, images are sorted alphabetically for reproducibility.
- Confidence is the product of per-token probabilities (same as training).
"""
from __future__ import annotations

import argparse
import csv
import unicodedata
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import torch

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

# Image extensions accepted for folder inference.
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------

def _nfc(s: str) -> str:
    """NFC-normalise *s*.  Used for Marathi (Devanagari) strings so that
    composed characters such as matras and vowel signs are preserved.
    NFKD is explicitly NOT applied here."""
    return unicodedata.normalize('NFC', s)


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _collect_images(folder: Path) -> list[Path]:
    """Return all image files under *folder*, sorted alphabetically."""
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    return images


def _infer_batch(
    model: torch.nn.Module,
    image_paths: list[Path],
    transform,
    device: torch.device,
) -> list[tuple[str, str, float]]:
    """Run inference on a list of image paths.

    Returns a list of (filename, prediction, confidence) tuples.
    """
    results: list[tuple[str, str, float]] = []

    for path in tqdm(image_paths, desc='Inferring', unit='img'):
        try:
            img = Image.open(path).convert('RGB')
        except Exception as exc:  # noqa: BLE001
            print(f'  [WARN] Could not open {path}: {exc}')
            results.append((path.name, '', 0.0))
            continue

        tensor = transform(img).unsqueeze(0).to(device)
        logits = model(tensor)                     # (1, L, C)
        probs = logits.softmax(-1)
        preds, prob_seqs = model.tokenizer.decode(probs)

        pred = model.charset_adapter(preds[0])
        pred = _nfc(pred)
        confidence = prob_seqs[0].prod().item()

        results.append((path.name, pred, confidence))

    return results


def _print_and_save(
    results: list[tuple[str, str, float]],
    output_csv: Path,
) -> None:
    """Print a summary table and write the CSV file."""
    print('\n' + '─' * 60)
    print(f'  {"Filename":<30}  {"Prediction":<20}  Confidence')
    print('─' * 60)
    for filename, pred, conf in results:
        print(f'  {filename:<30}  {pred:<20}  {conf:.4f}')
    print('─' * 60)

    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'prediction', 'confidence'])
        writer.writeheader()
        for filename, pred, conf in results:
            writer.writerow({
                'filename':   filename,
                'prediction': pred,
                'confidence': f'{conf:.6f}',
            })
    print(f'\nPredictions saved to: {output_csv.resolve()}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run PARSeq inference on a single image or a folder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', required=True,
                        help="Path to .ckpt file, or 'pretrained=<id>'")
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--output',     default='predictions.csv',
                        help='Output CSV file path')

    # Input source — exactly one of --image or --folder must be provided.
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--image',  metavar='FILE',
                     help='Single image file to process')
    src.add_argument('--folder', metavar='DIR',
                     help='Directory of images to process')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')

    # ── Load model ───────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.checkpoint}')
    model = load_from_checkpoint(args.checkpoint).eval().to(device)
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    # ── Collect input images ─────────────────────────────────────────────────
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            print(f'Error: file not found: {image_path}', flush=True)
            raise SystemExit(1)
        image_paths = [image_path]
    else:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f'Error: directory not found: {folder}', flush=True)
            raise SystemExit(1)
        image_paths = _collect_images(folder)
        if not image_paths:
            print(f'No images found in {folder} '
                  f'(supported: {", ".join(sorted(_IMAGE_EXTS))})')
            raise SystemExit(0)
        print(f'Found {len(image_paths)} image(s) in {folder}')

    # ── Inference ─────────────────────────────────────────────────────────────
    results = _infer_batch(model, image_paths, transform, device)

    # ── Single-image convenience output ──────────────────────────────────────
    if args.image and results:
        _, pred, conf = results[0]
        print(f'\nPrediction : {pred}')
        print(f'Confidence : {conf:.4f}')

    # ── Print table + save CSV ────────────────────────────────────────────────
    _print_and_save(results, Path(args.output))


if __name__ == '__main__':
    main()
