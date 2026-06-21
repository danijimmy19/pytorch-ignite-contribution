# Phase III: Implementation — Label-wise Accuracy (Issue #513)

**GitHub:** [@danijimmy19](https://github.com/danijimmy19)
**Working branch:** `feature/labelwise-multi-label-metrics`
**Issue:** [pytorch/ignite #513](https://github.com/pytorch/ignite/issues/513)

---

## Implementation Progress

**Files Created:**

| File | Description |
|---|---|
| `labelwise_accuracy.py` | `LabelwiseAccuracy` — subclass of `Accuracy` implementing `average='label-wise'` for multilabel problems |
| `test_labelwise_accuracy.py` | 15 tests covering correctness, edge cases, multi-batch consistency, spatial inputs, and fallback behavior |
| `demo.py` | End-to-end working example showing before/after behavior with manual per-label verification |

---

## What Was Implemented

`LabelwiseAccuracy` subclasses `Accuracy` from `ignite.metrics` and overrides `update()` and `compute()` to support `average='label-wise'`. When this option is active, instead of collapsing all labels per sample with `torch.all(..., dim=-1)` (subset accuracy), the metric accumulates a per-label correct-count tensor of shape `(C,)` and returns it from `compute()`. This directly mirrors the existing `Precision(average=False)` API that already supports per-label output.

The root-cause fix identified in Phase II is confined to the multilabel branch of `update()`:

```python
# Before: subset accuracy — all C labels must match per sample
correct = torch.all(y == y_pred.type_as(y), dim=-1)  # (N,) bool

# After: per-label accuracy — each column independently
correct_per_label = (y == y_pred.type_as(y)).to(dtype=torch.float64)  # (N, C)
self._num_correct = self._num_correct + correct_per_label.sum(dim=0)  # (C,)
```

The `compute()` method conditionally returns a tensor or float:

```python
def compute(self) -> float | torch.Tensor:
    if self._num_examples == 0:
        raise NotComputableError(...)
    if self._average == "label-wise":
        return self._num_correct / self._num_examples  # Tensor(C,)
    return self._num_correct.item() / self._num_examples  # float
```

---

## Challenges Faced

### Challenge 1: In-place tensor accumulation vs. shape promotion

The most significant technical obstacle was that `self._num_correct` starts as a scalar zero tensor (`shape=[]`) from `Accuracy.reset()`. On the first label-wise `update()` call, trying to add a `(C,)` per-label tensor to it with `+=` raises:

```
RuntimeError: output with shape [] doesn't match the broadcast shape [3]
```

PyTorch's in-place `+=` cannot change a tensor's shape. The fix is to use a non-in-place reassignment:

```python
# Fails on first call (shape [] can't be extended in-place to (C,))
self._num_correct += correct_per_label.sum(dim=0)

# Works: scalar(0) + tensor(C,) broadcasts to tensor(C,) on first call
self._num_correct = self._num_correct + correct_per_label.sum(dim=0)
```

This means `reset()` does not need to be overridden — the parent's scalar `0` initialization is fine because non-in-place addition handles the shape promotion automatically on the first update.

### Challenge 2: Python environment with both torch and sklearn

The ignite repo is installed in editable mode via `pip install -e ".[dev]"`, but the system Python 3.9 and the Anaconda base environment don't have `torch` installed. The `byte` conda environment (Python 3.10) has both `torch` and `ignite` resolvable. Tests must be run as:

```bash
/opt/anaconda3/envs/byte/bin/python -m pytest test_labelwise_accuracy.py -v
```

---

## Testing Notes

### Automated Tests (15 tests, all passing)

```
/opt/anaconda3/envs/byte/bin/python -m pytest test_labelwise_accuracy.py -v
# 15 passed in 2.33s
```

Tests follow the same patterns as ignite's `tests/ignite/metrics/test_accuracy.py`:
- Use `sklearn.metrics.accuracy_score` called per column as ground truth (mirrors ignite's own approach)
- Use `to_numpy_multilabel()` helper (copied from `test_accuracy.py`) to reshape tensors for sklearn
- Parametrize random-seed tests with `range(3)` for coverage across different random states

| Test Group | Tests | What is verified |
|---|---|---|
| Constructor validation | 3 | Invalid `average` string; `average='label-wise'` + `is_multilabel=False`; `compute()` before `update()` |
| Basic correctness | 3 | Output shape `(C,)`; all-correct → `[1.0, …]`; all-wrong → `[0.0, …]` |
| Multi-batch consistency | 3 (parametrized) | 40 samples split into batches of 8 matches single-batch result |
| sklearn ground truth | 3 (parametrized) | Random 50×7 data matches `accuracy_score` per column |
| Fallback behavior | 1 | `average=None` still returns `float` scalar (backward compat) |
| Reset / epoch boundary | 1 | `reset()` clears state; second epoch result is independent of first |
| Spatial `(N,C,H,W)` inputs | 1 | Spatial dims flattened before per-label sum; result shape `(C,)` |

### Manual Verification (`demo.py`)

Run the end-to-end demo from `pytorch-ignite-contribution/`:

```bash
/opt/anaconda3/envs/byte/bin/python demo.py
```

**Full output:**

```
=======================================================
Sample data (5 samples, 3 labels)
=======================================================
y_pred:
tensor([[1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 1]])

y_true:
tensor([[1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 1, 0]])

=======================================================
BEFORE — Accuracy(is_multilabel=True)
=======================================================
Result : 0.4000  ← single float
Meaning: only samples 1 and 3 have ALL 3 labels correct
         → 2 out of 5 = 0.4000
Problem: hides which specific labels are failing

=======================================================
AFTER  — LabelwiseAccuracy(average='label-wise')
=======================================================
Result : [1.0, 0.8, 0.6]
Shape  : (3,)  ← one value per label

=======================================================
Manual verification (per-label breakdown)
=======================================================
  Label 0: [1, 0, 1, 0, 1] vs [1, 0, 1, 0, 1] → 5/5 = 1.0000
  Label 1: [0, 1, 1, 0, 1] vs [0, 1, 0, 0, 1] → 4/5 = 0.8000
  Label 2: [1, 0, 0, 1, 1] vs [0, 0, 0, 1, 0] → 3/5 = 0.6000

=======================================================
Multi-batch check (one sample at a time)
=======================================================
Single-batch result : [1.0, 0.8, 0.6]
Incremental result  : [1.0, 0.8, 0.6]
Match               : True  ✓

=======================================================
Fallback — LabelwiseAccuracy(average=None)
=======================================================
Result : 0.4000  ← same as standard Accuracy
Type   : float
Matches standard Accuracy: True  ✓
```

The demo shows four things:
1. **Before** — `Accuracy(is_multilabel=True)` gives `0.4` (2 of 5 samples fully correct), masking per-label differences.
2. **After** — `LabelwiseAccuracy(average='label-wise')` gives `[1.0, 0.8, 0.6]`, pinpointing that label 2 is the weakest.
3. **Manual breakdown** — each value verified by hand so the math is transparent.
4. **Multi-batch consistency** — feeding one sample at a time gives the identical result.

Running `reproduce.py` separately shows the original problem (upstream `Accuracy` raising `TypeError: unexpected keyword argument 'average'`), confirming the gap this implementation fills.

---

## Edge Cases Identified

Beyond the minimum requirements, the following edge cases were identified and handled (or explicitly noted as out-of-scope):

| Edge Case | Handling |
|---|---|
| `average='label-wise'` with `is_multilabel=False` | `ValueError` raised in `__init__()` |
| Unsupported `average` values (`'macro'`, `'micro'`, etc.) | `ValueError` raised in `__init__()` |
| Multi-batch updates with uneven batch sizes | `_num_examples` is a scalar count; works correctly across batches |
| Spatial inputs `(N, C, H, W)` | Inherited reshape logic flattens to `(N*H*W, C)` before per-label sum |
| Zero examples in `compute()` | Inherited `NotComputableError` guard covers this path |
| Distributed training (`@sync_all_reduce`) | Inherited decorator handles both scalar and tensor `_num_correct`; no change needed |
| Scalar → tensor shape promotion in `_num_correct` | Non-in-place assignment (see Challenges above) |
