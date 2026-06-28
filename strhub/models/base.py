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

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from nltk import edit_distance

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

import pytorch_lightning as pl
from timm.optim import create_optimizer_v2

from strhub.data.utils import BaseTokenizer, CharsetAdapter, CTCTokenizer, Tokenizer


@dataclass
class BatchResult:
    num_samples: int
    correct: int
    ned: float
    confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int


# Type alias for a list of per-step output dicts collected across a validation epoch.
EPOCH_OUTPUT = list[dict[str, BatchResult]]


# Logical groupings of inner-model submodules for freeze / LR control.
# Each key is the user-facing config name; the value is the attribute name on
# the inner nn.Module (self.model for PARSeq-like systems).
_COMPONENT_MAP: dict[str, str] = {
    'encoder':    'encoder',
    'decoder':    'decoder',
    'head':       'head',
    'text_embed': 'text_embed',
}

# Components treated as the "backbone" (shared visual/sequential features).
_BACKBONE_COMPONENTS = ('encoder', 'decoder')
# Components treated as the "head" (task-specific classifier + embeddings).
_HEAD_COMPONENTS      = ('head', 'text_embed')


def _apply_freeze(inner_model: torch.nn.Module, freeze_cfg: dict[str, bool]) -> None:
    """Set ``requires_grad`` on submodules of *inner_model* according to *freeze_cfg*.

    Parameters
    ----------
    inner_model:
        The bare ``nn.Module`` (e.g. ``system.model`` for PARSeq).
    freeze_cfg:
        Dict mapping component name → True (freeze) / False (keep trainable).
        Missing keys default to False (trainable).
    """
    for cfg_key, module_attr in _COMPONENT_MAP.items():
        if not freeze_cfg.get(cfg_key, False):
            continue
        module = getattr(inner_model, module_attr, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(False)


def _print_finetune_summary(
    freeze_cfg: dict[str, bool],
    backbone_lr: float | None,
    head_lr: float | None,
    default_lr: float,
) -> None:
    """Print a one-time fine-tuning configuration table."""
    sep = '=' * 48
    print(sep)
    print('Fine-tuning Configuration'.center(48))
    print(sep)

    labels = {
        'encoder':    'Encoder    ',
        'decoder':    'Decoder    ',
        'head':       'Head       ',
        'text_embed': 'Text Embed ',
    }
    for cfg_key, label in labels.items():
        state = 'Frozen' if freeze_cfg.get(cfg_key, False) else 'Trainable'
        print(f'  {label} : {state}')

    if backbone_lr is not None and head_lr is not None:
        print(f'  Backbone LR : {backbone_lr}')
        print(f'  Head LR     : {head_lr}')
    else:
        print(f'  Learning Rate : {default_lr}')

    print(sep)


class BaseSystem(pl.LightningModule, ABC):

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        charset_test: str,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
        # Optional staged fine-tuning controls.  Both default to None so that
        # existing configs that do not specify them continue to work unchanged.
        backbone_lr: Optional[float] = None,
        head_lr: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.charset_adapter = CharsetAdapter(charset_test)
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.outputs: EPOCH_OUTPUT = []

    @abstractmethod
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        """Inference

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            max_length: Max sequence length of the output. If None, will use default.

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
        """
        raise NotImplementedError

    @abstractmethod
    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        """Like forward(), but also computes the loss (calls forward() internally).

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            labels: Text labels of the images

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
            loss: mean loss for the batch
            loss_numel: number of elements the loss was calculated from
        """
        raise NotImplementedError

    def _lr_scale(self) -> float:
        """Compute the linear LR scale factor (DDP + grad accumulation)."""
        agb = self.trainer.accumulate_grad_batches
        return agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.0

    def configure_optimizers(self):
        scale = self._lr_scale()
        lr = scale * self.lr

        use_diff_lr = (self.backbone_lr is not None) and (self.head_lr is not None)

        if use_diff_lr:
            # Build explicit parameter groups so each component gets its own LR.
            # Frozen parameters (requires_grad=False) are automatically excluded
            # by create_optimizer_v2 / AdamW since they have no gradient.
            inner = getattr(self, 'model', None)
            if inner is not None:
                backbone_params = [
                    p for comp in _BACKBONE_COMPONENTS
                    for p in getattr(inner, comp, torch.nn.Module()).parameters()
                    if p.requires_grad
                ]
                head_params = [
                    p for comp in _HEAD_COMPONENTS
                    for p in getattr(inner, comp, torch.nn.Module()).parameters()
                    if p.requires_grad
                ]
                # Any parameters not covered by the two groups (e.g. pos_queries,
                # dropout) fall back to the global lr.
                grouped_ids = {id(p) for p in backbone_params + head_params}
                other_params = [
                    p for p in self.parameters()
                    if p.requires_grad and id(p) not in grouped_ids
                ]
                param_groups = []
                if backbone_params:
                    param_groups.append({'params': backbone_params,
                                         'lr': scale * self.backbone_lr})
                if head_params:
                    param_groups.append({'params': head_params,
                                         'lr': scale * self.head_lr})
                if other_params:
                    param_groups.append({'params': other_params, 'lr': lr})
            else:
                # Fallback: no inner model — treat all params uniformly.
                param_groups = [{'params': list(self.parameters()), 'lr': lr}]

            # Use the maximum LR for the scheduler's max_lr so that no group
            # exceeds its configured ceiling during the warm-up phase.
            max_lr = [g['lr'] for g in param_groups]
            optim = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
            sched = OneCycleLR(
                optim, max_lr, self.trainer.estimated_stepping_batches,
                pct_start=self.warmup_pct, cycle_momentum=False,
            )
        else:
            # Original single-LR path — unchanged from the upstream repository.
            optim = create_optimizer_v2(self, 'adamw', lr, self.weight_decay)
            sched = OneCycleLR(
                optim, lr, self.trainer.estimated_stepping_batches,
                pct_start=self.warmup_pct, cycle_momentum=False,
            )

        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def _eval_step(self, batch, validation: bool) -> Optional[dict[str, Any]]:
        images, labels = batch

        correct = 0
        total = 0
        ned = 0
        confidence = 0
        label_length = 0
        if validation:
            logits, loss, loss_numel = self.forward_logits_loss(images, labels)
        else:
            # At test-time, we shouldn't specify a max_label_length because the test-time charset used
            # might be different from the train-time charset. max_label_length in eval_logits_loss() is computed
            # based on the transformed label, which could be wrong if the actual gt label contains characters existing
            # in the train-time charset but not in the test-time charset. For example, "aishahaleyes.blogspot.com"
            # is exactly 25 characters, but if processed by CharsetAdapter for the 36-char set, it becomes 23 characters
            # long only, which sets max_label_length = 23. This will cause the model prediction to be truncated.
            logits = self.forward(images)
            loss = loss_numel = None  # Only used for validation; not needed at test-time.

        probs = logits.softmax(-1)
        preds, probs = self.tokenizer.decode(probs)
        for pred, prob, gt in zip(preds, probs, labels):
            confidence += prob.prod().item()
            pred = self.charset_adapter(pred)
            # Follow ICDAR 2019 definition of N.E.D.
            ned += edit_distance(pred, gt) / max(len(pred), len(gt))
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
        return dict(output=BatchResult(total, correct, ned, confidence, label_length, loss, loss_numel))

    @staticmethod
    def _aggregate_results(outputs: EPOCH_OUTPUT) -> tuple[float, float, float]:
        if not outputs:
            return 0.0, 0.0, 0.0
        total_loss = 0
        total_loss_numel = 0
        total_n_correct = 0
        total_norm_ED = 0
        total_size = 0
        for result in outputs:
            result = result['output']
            total_loss += result.loss_numel * result.loss
            total_loss_numel += result.loss_numel
            total_n_correct += result.correct
            total_norm_ED += result.ned
            total_size += result.num_samples
        acc = total_n_correct / total_size
        ned = 1 - total_norm_ED / total_size
        loss = total_loss / total_loss_numel
        return acc, ned, loss

    def validation_step(self, batch, batch_idx) -> Optional[dict[str, Any]]:
        result = self._eval_step(batch, True)
        self.outputs.append(result)
        return result

    def on_validation_epoch_end(self) -> None:
        acc, ned, loss = self._aggregate_results(self.outputs)
        self.outputs.clear()
        self.log('val_accuracy', 100 * acc, sync_dist=True)
        self.log('val_NED', 100 * ned, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        self.log('hp_metric', acc, sync_dist=True)

    def test_step(self, batch, batch_idx) -> Optional[dict[str, Any]]:
        return self._eval_step(batch, False)


class CrossEntropySystem(BaseSystem):

    def __init__(
        self, charset_train: str, charset_test: str, batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
        backbone_lr: Optional[float] = None, head_lr: Optional[float] = None,
    ) -> None:
        tokenizer = Tokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay,
                         backbone_lr=backbone_lr, head_lr=head_lr)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        logits = self.forward(images, max_len)
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel


class CTCSystem(BaseSystem):

    def __init__(
        self, charset_train: str, charset_test: str, batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
        backbone_lr: Optional[float] = None, head_lr: Optional[float] = None,
    ) -> None:
        tokenizer = CTCTokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay,
                         backbone_lr=backbone_lr, head_lr=head_lr)
        self.blank_id = tokenizer.blank_id

    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        logits = self.forward(images)
        log_probs = logits.log_softmax(-1).transpose(0, 1)  # swap batch and seq. dims
        T, N, _ = log_probs.shape
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
        target_lengths = torch.as_tensor(list(map(len, labels)), dtype=torch.long, device=self.device)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        return logits, loss, N
