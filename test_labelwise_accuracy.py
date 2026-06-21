"""
Tests for LabelwiseAccuracy (issue #513 — label-wise multilabel accuracy).

Run from pytorch-ignite-contribution/:
    pip install -e "../ignite[dev]"
    pytest test_labelwise_accuracy.py -v

Follows the same patterns as ignite's own test_accuracy.py:
  - sklearn.metrics.accuracy_score as ground truth per label
  - to_numpy_multilabel helper to convert (N, C) tensors
  - Multi-batch updates verified to produce the same result as single-batch
  - pytest.raises for all invalid-input paths
"""

import sys
import os

# Allow running from pytorch-ignite-contribution/ directly
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import torch
from sklearn.metrics import accuracy_score

from ignite.exceptions import NotComputableError
from labelwise_accuracy import LabelwiseAccuracy

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers (mirrors test_accuracy.py::to_numpy_multilabel)
# ---------------------------------------------------------------------------

def to_numpy_multilabel(y: torch.Tensor):
    """Reshape (N, C) or (N, C, ...) tensor to (N*..., C) numpy array."""
    y = y.transpose(1, 0).cpu().numpy()
    num_classes = y.shape[0]
    return y.reshape((num_classes, -1)).transpose(1, 0)


def per_label_sklearn(y_pred: torch.Tensor, y: torch.Tensor):
    """Return list of per-label accuracy scores using sklearn as ground truth."""
    np_pred = to_numpy_multilabel(y_pred)
    np_y = to_numpy_multilabel(y)
    return [accuracy_score(np_y[:, i], np_pred[:, i]) for i in range(np_y.shape[1])]


# ---------------------------------------------------------------------------
# 1. Constructor validation
# ---------------------------------------------------------------------------

def test_invalid_average_value():
    """Unsupported average string must raise ValueError."""
    with pytest.raises(ValueError, match=r"average must be None or 'label-wise'"):
        LabelwiseAccuracy(average="macro")


def test_labelwise_requires_is_multilabel():
    """average='label-wise' is only valid when is_multilabel=True."""
    with pytest.raises(ValueError, match=r"only supported when is_multilabel=True"):
        LabelwiseAccuracy(average="label-wise", is_multilabel=False)


def test_no_update_raises():
    """compute() before any update must raise NotComputableError."""
    acc = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    with pytest.raises(NotComputableError):
        acc.compute()


# ---------------------------------------------------------------------------
# 2. Basic correctness — single batch
# ---------------------------------------------------------------------------

def test_labelwise_single_batch_shape_and_values():
    """Per-label result has shape (C,) and matches sklearn per-column."""
    y_pred = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])

    acc = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    acc.reset()
    acc.update((y_pred, y_true))
    result = acc.compute()

    assert isinstance(result, torch.Tensor), "label-wise result must be a Tensor"
    assert result.shape == (3,), f"expected shape (3,), got {result.shape}"

    expected = per_label_sklearn(y_pred, y_true)
    for i, exp in enumerate(expected):
        assert result[i].item() == pytest.approx(exp), f"label {i} mismatch"


def test_labelwise_all_correct():
    """All-correct predictions should give a tensor of 1.0s."""
    y = torch.randint(0, 2, size=(10, 4))
    acc = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    acc.reset()
    acc.update((y, y))
    result = acc.compute()
    assert result.shape == (4,)
    assert torch.all(result == 1.0), f"expected all 1.0, got {result}"


def test_labelwise_all_wrong():
    """Completely flipped predictions should give a tensor of 0.0s."""
    y = torch.ones(10, 3, dtype=torch.long)
    y_pred = torch.zeros(10, 3, dtype=torch.long)
    acc = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    acc.reset()
    acc.update((y_pred, y))
    result = acc.compute()
    assert result.shape == (3,)
    assert torch.all(result == 0.0), f"expected all 0.0, got {result}"


# ---------------------------------------------------------------------------
# 3. Multi-batch consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_times", range(3))
def test_labelwise_multi_batch_matches_single_batch(n_times):
    """Splitting data into batches must produce the same result as one update."""
    torch.manual_seed(n_times)
    N, C, batch_size = 40, 6, 8

    y_pred = torch.randint(0, 2, size=(N, C))
    y_true = torch.randint(0, 2, size=(N, C))

    # Single-batch reference
    acc_single = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    acc_single.reset()
    acc_single.update((y_pred, y_true))
    result_single = acc_single.compute()

    # Multi-batch (5 batches of 8)
    acc_multi = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    acc_multi.reset()
    n_iters = N // batch_size
    for i in range(n_iters):
        idx = i * batch_size
        acc_multi.update((y_pred[idx : idx + batch_size], y_true[idx : idx + batch_size]))
    result_multi = acc_multi.compute()

    assert result_single.shape == (C,)
    for i in range(C):
        assert result_single[i].item() == pytest.approx(result_multi[i].item()), f"label {i}"


# ---------------------------------------------------------------------------
# 4. Verify against sklearn per-label ground truth on random data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_times", range(3))
def test_labelwise_vs_sklearn(n_times):
    """Result must match sklearn's accuracy_score called per column."""
    torch.manual_seed(100 + n_times)
    N, C = 50, 7

    y_pred = torch.randint(0, 2, size=(N, C))
    y_true = torch.randint(0, 2, size=(N, C))

    acc = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    acc.reset()
    acc.update((y_pred, y_true))
    result = acc.compute()

    expected = per_label_sklearn(y_pred, y_true)
    for i, exp in enumerate(expected):
        assert result[i].item() == pytest.approx(exp), f"label {i}"


# ---------------------------------------------------------------------------
# 5. Fallback — subset accuracy still works when average=None
# ---------------------------------------------------------------------------

def test_subset_accuracy_fallback():
    """LabelwiseAccuracy without average should behave identically to Accuracy."""
    y_pred = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])

    acc = LabelwiseAccuracy(is_multilabel=True, average=None)
    acc.reset()
    acc.update((y_pred, y_true))
    result = acc.compute()

    # Only sample index 1 is fully correct → 1/3
    assert isinstance(result, float), "subset mode must return a float"
    assert result == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# 6. Edge case — reset() clears state for a fresh epoch
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    """After reset(), a second epoch's result must not include the first epoch's data."""
    y_pred = torch.ones(5, 3, dtype=torch.long)
    y_true = torch.zeros(5, 3, dtype=torch.long)  # all wrong

    acc = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    acc.reset()
    acc.update((y_pred, y_true))
    first_result = acc.compute()
    assert torch.all(first_result == 0.0)

    # After reset, feed all-correct data
    acc.reset()
    y_correct = torch.ones(5, 3, dtype=torch.long)
    acc.update((y_correct, y_correct))
    second_result = acc.compute()
    assert torch.all(second_result == 1.0), f"reset did not clear state: {second_result}"


# ---------------------------------------------------------------------------
# 7. Edge case — spatial (NHW) inputs
# ---------------------------------------------------------------------------

def test_labelwise_spatial_input():
    """Spatial inputs (N, C, H, W) should be flattened correctly before per-label sum."""
    N, C, H, W = 4, 3, 5, 5
    y_pred = torch.randint(0, 2, size=(N, C, H, W))
    y_true = torch.randint(0, 2, size=(N, C, H, W))

    acc = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    acc.reset()
    acc.update((y_pred, y_true))
    result = acc.compute()

    assert result.shape == (C,), f"expected shape ({C},), got {result.shape}"

    # Ground truth: flatten spatial dims, then compute per-label with sklearn
    expected = per_label_sklearn(
        y_pred.reshape(N * H * W, C),
        y_true.reshape(N * H * W, C),
    )
    # Note: to_numpy_multilabel handles the transposition, so pass raw (N,C,H,W)
    np_pred = to_numpy_multilabel(y_pred)
    np_y = to_numpy_multilabel(y_true)
    expected = [accuracy_score(np_y[:, i], np_pred[:, i]) for i in range(C)]
    for i, exp in enumerate(expected):
        assert result[i].item() == pytest.approx(exp), f"label {i} spatial mismatch"
