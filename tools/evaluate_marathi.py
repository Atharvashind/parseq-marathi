#!/usr/bin/env python3
"""Evaluate a PARSeq checkpoint on a Marathi (or any) LMDB validation dataset.

Usage
-----
python tools/evaluate_marathi.py \\
    --checkpoint outputs/parseq/best.ckpt \\
    --data_root   data \\
    --lmdb_path   data/val \\
    --batch_size  64 \\
    --num_workers 4 \\
    --output      evaluation_predictions.csv

The script:
  1. Loads the model from *checkpoint* using the existing load_from_checkpoint().
  2. Opens the LMDB at *lmdb_path* directly (no intermediate DataModule needed).
  3. Runs inference on every sample with torch.inference_mode().
  4. Decodes predictions via the model's own tokenizer (no duplication).
  5. Compares predictions to ground-truth with NFC normalisation (not NFKD) so
     Devanagari matras and composed characters are handled correctly.
  6. Prints a metrics table: Exact Match Accuracy, Character Accuracy, CER, NED.
  7. Writes evaluation_predictions.csv with columns:
       index, ground_truth, prediction, confidence

Metrics definitions
-------------------
Exact Match Accuracy : fraction of samples where prediction == ground_truth
Character Accuracy   : fraction of individual characters that are correct
                       (1 - CER) × 100, capped at 0
CER                  : Character Error Rate = edit_distance / len(ground_truth)
                       averaged across all samples
NED                  : Normalised Edit Distance per ICDAR 2019 definition
                       = 1 - mean(edit_distance / max(len(pred), len(gt)))
"""
from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

from nltk import edit_distance
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from strhub.data.dataset import LmdbDataset
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------

def _nfc(s: str) -> str:
    """NFC-normalise a string.  Used for Marathi (Devanagari) comparisons so
    that composed characters such as matras are preserved.  NFKD is explicitly
    NOT used here because it strips Devanagari characters to ASCII."""
    return unicodedata.normalize('NFC', s)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    ground_truths: list[str],
    predictions: list[str],
) -> dict[str, float]:
    """Compute evaluation metrics over a list of (gt, pred) pairs.

    All strings are NFC-normalised before comparison.

    Returns
    -------
    dict with keys: exact_match, char_accuracy, cer, ned
    """
    assert len(ground_truths) == len(predictions)

    n = len(ground_truths)
    if n == 0:
        return {'exact_match': 0.0, 'char_accuracy': 0.0, 'cer': 0.0, 'ned': 0.0}

    exact_correct = 0
    total_char_errors = 0
    total_gt_chars = 0
    total_ned = 0.0

    for gt_raw, pred_raw in zip(ground_truths, predictions):
        gt = _nfc(gt_raw)
        pred = _nfc(pred_raw)

        # Exact match
        if pred == gt:
            exact_correct += 1

        # Edit distance for CER and NED
        dist = edit_distance(pred, gt)
        total_char_errors += dist
        total_gt_chars += max(len(gt), 1)  # avoid division by zero on empty gt

        # NED: ICDAR 2019 definition
        denom = max(len(pred), len(gt))
        total_ned += dist / denom if denom > 0 else 0.0

    exact_match = 100.0 * exact_correct / n
    cer = 100.0 * total_char_errors / total_gt_chars
    char_accuracy = max(0.0, 100.0 - cer)
    ned = 100.0 * (1.0 - total_ned / n)

    return {
        'exact_match': exact_match,
        'char_accuracy': char_accuracy,
        'cer': cer,
        'ned': ned,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate a PARSeq checkpoint on a Marathi LMDB dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint',   required=True,
                        help="Path to .ckpt file, or 'pretrained=<id>'")
    parser.add_argument('--data_root',    default='data',
                        help='Root data directory (used when lmdb_path is relative)')
    parser.add_argument('--lmdb_path',    default=None,
                        help='Path to the LMDB directory to evaluate. '
                             'Defaults to <data_root>/val')
    parser.add_argument('--batch_size',   type=int, default=64)
    parser.add_argument('--num_workers',  type=int, default=4)
    parser.add_argument('--device',       default='cuda')
    parser.add_argument('--output',       default='evaluation_predictions.csv',
                        help='Path for the CSV output file')
    args = parser.parse_args()

    # ── Resolve LMDB path ────────────────────────────────────────────────────
    if args.lmdb_path is None:
        lmdb_path = str(Path(args.data_root) / 'val')
    else:
        lmdb_path = args.lmdb_path

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')

    # ── Load model ───────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.checkpoint}')
    model = load_from_checkpoint(args.checkpoint).eval().to(device)
    hp = model.hparams

    # ── Build dataset / dataloader ────────────────────────────────────────────
    transform = SceneTextDataModule.get_transform(hp.img_size)
    dataset = LmdbDataset(
        root=lmdb_path,
        charset=hp.charset_test,
        max_label_len=hp.max_label_length,
        remove_whitespace=True,
        normalize_unicode=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        shuffle=False,
    )
    print(f'Dataset: {lmdb_path}  ({len(dataset)} samples)')

    # ── Inference ─────────────────────────────────────────────────────────────
    all_gts: list[str] = []
    all_preds: list[str] = []
    all_confs: list[float] = []

    for imgs, labels in tqdm(loader, desc='Evaluating', unit='batch'):
        imgs = imgs.to(device)
        logits = model(imgs)                     # (N, L, C)
        probs = logits.softmax(-1)
        preds, prob_seqs = model.tokenizer.decode(probs)

        for pred, prob_seq, gt in zip(preds, prob_seqs, labels):
            # Apply the model's charset adapter so evaluation mirrors training.
            pred = model.charset_adapter(pred)
            confidence = prob_seq.prod().item()

            all_gts.append(gt)
            all_preds.append(pred)
            all_confs.append(confidence)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(all_gts, all_preds)

    sep = '=' * 48
    print(f'\n{sep}')
    print('Evaluation Results'.center(48))
    print(sep)
    print(f'  Samples          : {len(all_gts)}')
    print(f'  Exact Match Acc  : {metrics["exact_match"]:.2f} %')
    print(f'  Character Acc    : {metrics["char_accuracy"]:.2f} %')
    print(f'  CER              : {metrics["cer"]:.2f} %')
    print(f'  NED              : {metrics["ned"]:.2f} %')
    print(sep)

    # ── CSV export ────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'ground_truth',
                                               'prediction', 'confidence'])
        writer.writeheader()
        for idx, (gt, pred, conf) in enumerate(zip(all_gts, all_preds, all_confs)):
            writer.writerow({
                'index':        idx,
                'ground_truth': gt,
                'prediction':   pred,
                'confidence':   f'{conf:.6f}',
            })

    print(f'\nPredictions saved to: {output_path.resolve()}')


if __name__ == '__main__':
    main()
