"""Reusable implementation of label-wise multilabel accuracy.

This module exposes :class:`LabelwiseAccuracy`, a small extension of Ignite's
``Accuracy`` metric for multilabel problems. When ``average="label-wise"`` is
selected, the metric returns one accuracy value per label as a tensor with shape
``(C,)``. When ``average="macro"`` is selected, it returns the mean of those
per-label accuracies as a scalar. In the default subset-accuracy mode, it
preserves the standard scalar behavior of Ignite's ``Accuracy``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Accuracy
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

__all__ = ["LabelwiseAccuracy"]


class LabelwiseAccuracy(Accuracy):
    """Accuracy metric with optional label-wise output for multilabel data.

    Args:
        output_transform: Callable used to transform engine output into ``(y_pred, y)``.
        is_multilabel: Must be ``True`` whenever ``average="label-wise"`` is requested.
        device: Target device for accumulation tensors.
        skip_unrolling: Whether to unroll multi-output predictions.
        average: ``None`` for standard subset accuracy, ``"label-wise"`` for
            per-label accuracy, or ``"macro"`` for the mean of the per-label
            accuracies.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
        average: str | None = None,
    ) -> None:
        if average not in {None, "label-wise", "macro"}:
            raise ValueError(f"average must be None, 'label-wise', or 'macro', got '{average}'.")
        if average in {"label-wise", "macro"} and not is_multilabel:
            raise ValueError("average='label-wise' and average='macro' are only supported when is_multilabel=True.")

        super().__init__(
            output_transform=output_transform,
            is_multilabel=is_multilabel,
            device=device,
            skip_unrolling=skip_unrolling,
        )
        self._average = average

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_correct = torch.tensor(0, device=self._device, dtype=torch.float64)
        self._num_examples = 0
        super().reset()

    def _reshape_multilabel_inputs(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_classes = y_pred.size(1)
        last_dim = y_pred.ndimension()

        if last_dim <= 1:
            return y_pred.reshape(-1, num_classes), y.reshape(-1, num_classes)

        y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
        y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
        return y_pred, y

    def _prepare_vector_accumulator(self, num_labels: int) -> None:
        self._num_correct = self._num_correct.to(self._device)
        if self._num_correct.numel() == 1:
            self._num_correct = torch.zeros(num_labels, device=self._device, dtype=torch.float64)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if self._average in {"label-wise", "macro"} and self._type == "multilabel":
            y_pred, y = self._reshape_multilabel_inputs(y_pred, y)
            correct = (y == y_pred.type_as(y)).to(dtype=torch.float64, device=self._device)
            self._prepare_vector_accumulator(correct.size(1))
            self._num_correct += correct.sum(dim=0).to(self._device)
            self._num_examples += correct.size(0)
            return

        if self._type == "binary":
            correct = torch.eq(y_pred.view(-1).to(y), y.view(-1))
        elif self._type == "multiclass":
            indices = torch.argmax(y_pred, dim=1)
            correct = torch.eq(indices, y).view(-1)
        elif self._type == "multilabel":
            y_pred, y = self._reshape_multilabel_inputs(y_pred, y)
            correct = torch.all(y == y_pred.type_as(y), dim=-1)
        else:
            raise ValueError(f"Unexpected type: {self._type}")

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float | torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Accuracy must have at least one example before it can be computed.")

        if self._average in {"label-wise", "macro"}:
            per_label_accuracy = self._num_correct / self._num_examples
            if self._average == "macro":
                return per_label_accuracy.mean().item()
            return per_label_accuracy

        return self._num_correct.item() / self._num_examples
