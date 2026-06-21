"""
Label-wise accuracy for multi-label classification.

Addresses pytorch/ignite issue #513:
  https://github.com/pytorch/ignite/issues/513

Ignite's Accuracy for multilabel inputs computes subset (exact-match) accuracy:
every label must match for a sample to be counted correct. This class adds
average='label-wise', which returns a per-label accuracy tensor of shape (C,),
consistent with how Precision(average=False) and Recall(average=False) behave.

Intended as a drop-in addition to ignite/metrics/accuracy.py.
"""

from collections.abc import Callable, Sequence

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Accuracy
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class LabelwiseAccuracy(Accuracy):
    """Accuracy metric extended with per-label support for multilabel problems.

    When ``average='label-wise'`` and ``is_multilabel=True``, ``compute()``
    returns a ``torch.Tensor`` of shape ``(C,)`` where entry ``i`` is the
    fraction of samples for which label ``i`` was predicted correctly.

    All other configurations fall back to the standard ``Accuracy`` behavior
    (subset accuracy, returning a float scalar).

    Args:
        output_transform: callable to transform engine output into ``(y_pred, y)``.
        is_multilabel: must be ``True`` when ``average='label-wise'``.
        device: device for accumulation tensors.
        skip_unrolling: whether to unroll multi-output predictions.
        average: ``None`` for standard subset accuracy, or ``'label-wise'``
            to return per-label accuracy. Only valid when ``is_multilabel=True``.

    Example::

        import torch
        from labelwise_accuracy import LabelwiseAccuracy

        y_pred = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])

        acc = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
        acc.reset()
        acc.update((y_pred, y_true))
        print(acc.compute())
        # tensor([1.0000, 0.6667, 0.3333], dtype=torch.float64)
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
        average: str | None = None,
    ):
        if average is not None and average != "label-wise":
            raise ValueError(f"average must be None or 'label-wise', got '{average}'.")
        if average == "label-wise" and not is_multilabel:
            raise ValueError("average='label-wise' is only supported when is_multilabel=True.")
        self._average = average
        super().__init__(
            output_transform=output_transform,
            is_multilabel=is_multilabel,
            device=device,
            skip_unrolling=skip_unrolling,
        )

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if self._type == "multilabel" and self._average == "label-wise":
            # Reshape (N, C, ...) -> (N * ..., C) to handle spatial inputs
            num_classes = y_pred.size(1)
            last_dim = y_pred.ndimension()
            y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
            y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)

            # Per-label correct counts: (N, C) summed over samples -> (C,)
            # Root cause fix for issue #513: remove torch.all(..., dim=-1) collapse
            correct_per_label = (y == y_pred.type_as(y)).to(dtype=torch.float64)
            # Non-in-place: on the first call _num_correct is scalar 0; broadcasting
            # scalar(0) + tensor(C,) works with = but fails with += (in-place reshape).
            self._num_correct = self._num_correct + correct_per_label.sum(dim=0).to(self._device)
            self._num_examples += y.shape[0]
            return

        # --- All other cases: standard Accuracy logic (subset / binary / multiclass) ---
        if self._type == "binary":
            correct = torch.eq(y_pred.view(-1).to(y), y.view(-1))
        elif self._type == "multiclass":
            indices = torch.argmax(y_pred, dim=1)
            correct = torch.eq(indices, y).view(-1)
        elif self._type == "multilabel":
            num_classes = y_pred.size(1)
            last_dim = y_pred.ndimension()
            y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
            y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
            correct = torch.all(y == y_pred.type_as(y), dim=-1)
        else:
            raise ValueError(f"Unexpected type: {self._type}")

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float | torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Accuracy must have at least one example before it can be computed.")
        if self._average == "label-wise":
            return self._num_correct / self._num_examples
        return self._num_correct.item() / self._num_examples
