"""Tests for the label-wise multilabel accuracy metric."""

import os
import sys

import pytest
import torch
from sklearn.metrics import accuracy_score

from ignite.exceptions import NotComputableError
from benchmark_labelwise_accuracy import _to_python_float
from labelwise_accuracy import LabelwiseAccuracy

sys.path.insert(0, os.path.dirname(__file__))

torch.manual_seed(42)


def _to_numpy_multilabel(y: torch.Tensor) -> torch.Tensor:
    y = y.transpose(1, 0).cpu().numpy()
    num_classes = y.shape[0]
    return y.reshape((num_classes, -1)).transpose(1, 0)


def _per_label_sklearn(y_pred: torch.Tensor, y_true: torch.Tensor) -> list[float]:
    np_pred = _to_numpy_multilabel(y_pred)
    np_true = _to_numpy_multilabel(y_true)
    return [accuracy_score(np_true[:, i], np_pred[:, i]) for i in range(np_true.shape[1])]


def test_to_python_float_handles_tensor_and_scalars() -> None:
    assert _to_python_float(torch.tensor(0.25)) == pytest.approx(0.25)
    assert _to_python_float(7.0) == pytest.approx(7.0)


def test_invalid_average_value() -> None:
    with pytest.raises(ValueError, match=r"average must be None or 'label-wise'"):
        LabelwiseAccuracy(average="macro")


def test_labelwise_requires_is_multilabel() -> None:
    with pytest.raises(ValueError, match=r"only supported when is_multilabel=True"):
        LabelwiseAccuracy(average="label-wise", is_multilabel=False)


def test_compute_raises_before_update() -> None:
    metric = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    with pytest.raises(NotComputableError):
        metric.compute()


def test_labelwise_matches_sklearn_per_label() -> None:
    y_pred = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.long)
    y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.long)

    metric = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    metric.update((y_pred, y_true))
    result = metric.compute()

    expected = _per_label_sklearn(y_pred, y_true)
    assert result.shape == (3,)
    for index, value in enumerate(expected):
        assert result[index].item() == pytest.approx(value)


def test_macro_average_matches_mean_of_per_label_accuracies() -> None:
    y_pred = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.long)
    y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.long)

    metric = LabelwiseAccuracy(is_multilabel=True, average="macro")
    metric.update((y_pred, y_true))
    result = metric.compute()

    expected = sum(_per_label_sklearn(y_pred, y_true)) / 3
    assert result == pytest.approx(expected)


def test_all_correct_returns_ones() -> None:
    y = torch.randint(0, 2, size=(10, 4), dtype=torch.long)
    metric = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    metric.update((y, y))
    result = metric.compute()

    assert result.shape == (4,)
    assert torch.all(result == 1.0)


def test_all_wrong_returns_zeros() -> None:
    y_pred = torch.zeros(10, 3, dtype=torch.long)
    y_true = torch.ones(10, 3, dtype=torch.long)

    metric = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    metric.update((y_pred, y_true))
    result = metric.compute()

    assert result.shape == (3,)
    assert torch.all(result == 0.0)


def test_multi_batch_matches_single_batch() -> None:
    torch.manual_seed(0)
    y_pred = torch.randint(0, 2, size=(40, 6), dtype=torch.long)
    y_true = torch.randint(0, 2, size=(40, 6), dtype=torch.long)

    single_metric = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    single_metric.update((y_pred, y_true))
    single_result = single_metric.compute()

    multi_metric = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    for start in range(0, 40, 8):
        multi_metric.update((y_pred[start : start + 8], y_true[start : start + 8]))
    multi_result = multi_metric.compute()

    assert single_result.shape == (6,)
    for index in range(6):
        assert single_result[index].item() == pytest.approx(multi_result[index].item())


def test_subset_accuracy_fallback() -> None:
    y_pred = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.long)
    y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.long)

    metric = LabelwiseAccuracy(is_multilabel=True, average=None)
    metric.update((y_pred, y_true))
    result = metric.compute()

    assert isinstance(result, float)
    assert result == pytest.approx(1 / 3)
