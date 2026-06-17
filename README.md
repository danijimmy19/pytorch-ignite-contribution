# Open Source Contribution — CodePath AI301
**GitHub:** [@danijimmy19](https://github.com/danijimmy19)

---

## About Me
I'm **Jimmy Dani**, a Computer Science PhD Candidate at Texas A&M University specializing in machine learning for cybersecurity, privacy, and cryptographic evaluation. My research spans deep learning, LLMs, adversarial ML, and secure multi-party computation. I have hands-on experience with PyTorch, scikit-learn, TensorFlow, and the broader ML/AI ecosystem, and I'm participating in CodePath's AI301 course (Summer 2026) to contribute to impactful open source ML projects.

---

## Selected Issue

### Label-wise Metrics (Accuracy, Precision, Recall) for Multi-label Problems
**Issue:** [#513 — pytorch/ignite](https://github.com/pytorch/ignite/issues/513)
**Repository:** [pytorch/ignite](https://github.com/pytorch/ignite)
**Organization:** PyTorch
**Languages:** Python, Jupyter Notebook, Shell
**Tags:** `enhancement` `help wanted` `module: metrics`

---

## Why I Chose This Issue

As a PhD researcher working extensively with PyTorch and multi-label classification tasks in my ML and security research, I immediately recognized this gap as a practical and meaningful problem. My day-to-day work requires evaluating models at the per-label level — a single averaged accuracy score often hides which specific classes a model is struggling with, which is critical for improving model performance.

This issue is a strong skill match: I have deep familiarity with PyTorch's tensor operations, scikit-learn's metrics API, and the kind of metric computation logic involved here. My learning goal is to go beyond using open source libraries and understand how a production-grade ML library like Ignite structures its metrics module, handles edge cases, and maintains test coverage — skills directly transferable to my research engineering work.

---

## Problem Summary

PyTorch Ignite's `Accuracy` metric for multi-label classification currently returns only a single averaged scalar (subset accuracy), making it impossible to diagnose per-label model performance. `Precision` and `Recall` already support per-label output via `average=False`, but `Accuracy` has no equivalent. This matters because an overall accuracy of 70% could hide a single failing label or reflect uniform errors across all labels — two very different situations requiring different fixes. The feature was requested in 2019, a community PR (#516) was submitted but never merged due to unresolved API design questions, and as of v0.5.4 the gap still exists.

---

## Forked Repository

[github.com/danijimmy19/ignite](https://github.com/danijimmy19/ignite)

**Working branch:** `feature/labelwise-multi-label-metrics`

---

## Phase II: Reproduce & Plan

---

### 1. Environment Setup

**Setup approach:** I followed the repository's `README.md` and `CONTRIBUTING.md` instructions, using the local Python environment (Python 3.11) and installing the dev extras via pip.

```bash
git clone https://github.com/danijimmy19/ignite.git
cd ignite
git checkout feature/labelwise-multi-label-metrics
pip install -e ".[dev]"
pip install pytest scikit-learn torch
```

**Challenges encountered and how I resolved them:**

- **Pre-commit hooks failing on first commit.** The repo uses `ufmt` (black + usort) for formatting. My initial edits had minor formatting issues that caused the pre-commit hook to fail. I ran `ufmt format ignite/metrics/accuracy.py` and re-committed. Resolution: always run `ufmt format` before committing.

- **`pytest` couldn't find `ignite` module.** Running `pytest tests/` from the repo root failed with `ModuleNotFoundError: No module named 'ignite'` until I installed the package in editable mode with `pip install -e ".[dev]"`. The `[dev]` extra is required; a bare install does not pull test dependencies.

- **Identifying the right test runner flags.** Running the full test suite triggers distributed training tests (which require multiple GPUs or mock setup). I isolated just the metrics tests with `pytest tests/ignite/metrics/test_accuracy.py -v` to get fast, clean feedback.

---

### 2. Reproduction

This is an enhancement issue (missing feature), so "reproduction" means demonstrating the current behavior and showing exactly where it falls short of what is needed.

#### Reproduction Steps

1. Clone the fork and checkout the working branch:
   ```bash
   git clone https://github.com/danijimmy19/ignite.git && cd ignite
   git checkout feature/labelwise-multi-label-metrics
   pip install -e ".[dev]"
   ```

2. Open a Python interpreter (or run the script below):
   ```python
   import torch
   from ignite.metrics import Accuracy, Precision

   # --- Accuracy: current multilabel behavior ---
   acc = Accuracy(is_multilabel=True)
   acc.reset()

   y_true = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
   y_pred = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])  # label 2 wrong for samples 0 and 2

   acc.update((y_pred, y_true))
   print("Accuracy result:", acc.compute())

   # --- Precision: already supports per-label via average=False ---
   prec = Precision(is_multilabel=True, average=False)
   prec.reset()
   prec.update((y_pred.float(), y_true.float()))
   print("Per-label Precision:", prec.compute())
   ```

3. Observe the output:
   ```
   Accuracy result: 0.3333333333333333
   Per-label Precision: tensor([1.0000, 1.0000, 0.0000], dtype=torch.float64)
   ```

#### Expected vs. Actual Behavior

| | Accuracy (current) | Precision (current) | Desired Accuracy |
|---|---|---|---|
| **Output type** | `float` scalar | `torch.Tensor` of shape `(C,)` | `torch.Tensor` of shape `(C,)` |
| **What it computes** | Subset accuracy (all C labels must match per sample) | True positives / predicted positives per label | Correct predictions per label independently |
| **Example output** | `0.333` | `tensor([1.0, 1.0, 0.0])` | `tensor([1.0, 0.667, 0.333])` |

**Specific detail:** With the 3-sample batch above, only 1 of 3 samples has ALL labels correct (sample 1), so the current subset accuracy is `1/3 = 0.333`. But per-label, label 0 is correct in all 3 samples, label 1 correct in 2 of 3, and label 2 correct in 1 of 3 — giving `[1.0, 0.667, 0.333]`.

#### Files and Functions Involved

- **`ignite/metrics/accuracy.py`** — `Accuracy` class, specifically:
  - `Accuracy.__init__()` (line 276): no `average` parameter exists; needs one
  - `Accuracy.reset()` (line 288): initializes `self._num_correct` as a scalar `torch.tensor(0)` — needs to become a per-label tensor when label-wise mode is active
  - `Accuracy.update()` (line 294): the multilabel branch (lines 304–310) computes `correct = torch.all(y == y_pred.type_as(y), dim=-1)` — the `torch.all(..., dim=-1)` collapses labels into a per-sample bool before summing; per-label accuracy requires removing `torch.all` and summing per-column instead
  - `Accuracy.compute()` (line 317): divides scalar `_num_correct` by `_num_examples`; needs a conditional path for tensor output

- **`ignite/metrics/precision.py`** — `_BasePrecisionRecall._prepare_output()` (line 61) and `compute()` (line 128): the reference implementation for per-label metrics; the pattern `self._numerator / self._denominator` returning a tensor when `average=False` is the model to follow

- **`tests/ignite/metrics/test_accuracy.py`** — `test_multilabel_input()` (line 158) and `to_numpy_multilabel()` (line 129): the existing test helper and multilabel test function show the fixture pattern and how sklearn is used for ground-truth comparison

---

### 3. Solution Plan (UMPIRE)

#### U — Understand the Problem

The issue requests per-label accuracy for multi-label classification, analogous to how `Precision(average=False)` already returns a per-label tensor. The root cause is in `Accuracy.update()` at **`accuracy.py:310`**:

```python
correct = torch.all(y == y_pred.type_as(y), dim=-1)   # per-sample: ALL labels must match
```

`torch.all(..., dim=-1)` reduces the label dimension to a boolean — 1 if every label matches, 0 otherwise. This is "exact match ratio" (subset accuracy). Label-wise accuracy instead asks: for each label independently, what fraction of samples got it right? That means removing the `torch.all` and keeping per-column correct counts.

Additionally, `_num_correct` is initialized as a scalar zero tensor (`torch.tensor(0)`) at line 289. There is no `average` parameter on `Accuracy` at all — unlike `Precision` and `Recall` which both accept one.

#### M — Match (Analogous Patterns in the Codebase)

**`_BasePrecisionRecall`** (`precision.py:16`) is the direct model to follow:

- It has an `average` parameter validated in `__init__()` (line 27–36).
- `_numerator` and `_denominator` accumulate as tensors of shape `(C,)` when `average` is `False`/`None`/`'macro'`/`'weighted'` (lines 120–122).
- `compute()` returns `fraction = self._numerator / self._denominator` as a tensor when `average` is `False` or `None` (line 160).
- The test suite in `test_precision.py` validates each `average` variant and cross-checks against sklearn.

I confirmed via `git log --oneline ignite/metrics/accuracy.py` that `_num_correct` as a scalar was introduced in commit `ce00b72a` ("Accuracy MultiLabel Handling and Error Message") and has never been made per-label. The `_BasePrecisionRecall` path was added in `1a8ead8b` ("Multilabel Metrics") and was designed with the tensor-accumulator pattern from the start.

#### P — Plan

The change is confined to `accuracy.py` and its test file. No changes needed to `precision.py`, `recall.py`, or the base `Metric` class.

**Step 1 — Add `average` parameter to `Accuracy.__init__()`:**

Accept `average: bool | str | None = False` (defaulting to `False` for backward-compat with the existing scalar behavior, mirroring Precision/Recall). Validate that `average="label-wise"` is only valid when `is_multilabel=True`, and raise a `ValueError` otherwise.

**Step 2 — Update `Accuracy.reset()`:**

When `average="label-wise"` and `is_multilabel=True`, initialize `_num_correct` as a tensor of zeros of shape `(num_classes,)`. Because `num_classes` is not known until the first `update()` call (it's inferred from `y_pred.shape[1]`), initialize to a scalar `0` and replace with the appropriately shaped tensor on the first update call, matching how `_BasePrecisionRecall` handles deferred shape initialization.

**Step 3 — Update `Accuracy.update()` multilabel branch:**

Change:
```python
correct = torch.all(y == y_pred.type_as(y), dim=-1)   # (N,)  — wrong for label-wise
self._num_correct += torch.sum(correct).to(self._device)
self._num_examples += correct.shape[0]
```

To (when `average="label-wise"`):
```python
correct = (y == y_pred.type_as(y)).float()   # (N, C) — per-label per-sample
self._num_correct += correct.sum(dim=0).to(self._device)  # (C,)
self._num_examples += y.shape[0]
```

Keep the existing path unchanged for all other `average` values (preserves backward compat).

**Step 4 — Update `Accuracy.compute()`:**

Add a conditional:
```python
if self._average == "label-wise":
    return self._num_correct / self._num_examples   # returns Tensor of shape (C,)
return self._num_correct.item() / self._num_examples  # existing scalar path
```

**Step 5 — Update `@sync_all_reduce` decorator:**

The existing `@sync_all_reduce("_num_examples", "_num_correct")` handles distributed reduction. For the label-wise case, `_num_correct` is a tensor instead of a scalar — `sync_all_reduce` already handles both scalars and tensors (confirmed by inspecting `metric.py`), so no decorator change is needed.

**Step 6 — Update docstring and `_state_dict_all_req_keys`:**

Add an RST doctest example for `average="label-wise"` consistent with the existing multilabel testcode block. No other documentation file needs to change.

**Step 7 — Add tests in `test_accuracy.py`:**

Following the existing `test_multilabel_input` pattern:
- Use sklearn's `accuracy_score` called per-column as ground truth: `[accuracy_score(y[:, i], y_pred[:, i]) for i in range(C)]`
- Test single-batch and multi-batch update
- Test `ValueError` raised when `average="label-wise"` and `is_multilabel=False`
- Test shape of returned tensor is `(C,)` for a C-label problem

#### I — Identify Files to Modify

| File | Change |
|---|---|
| `ignite/metrics/accuracy.py` | Add `average` param; branch update/compute on `"label-wise"` |
| `tests/ignite/metrics/test_accuracy.py` | New test functions for `average="label-wise"` |

No other files require modification. `Precision` and `Recall` already support per-label output for multilabel via `average=False`.

#### R — Root Cause (Not Symptom)

The root cause is a design decision made in 2019 when multilabel support was added (`commit 1a8ead8b`): `Accuracy` was implemented to compute subset accuracy (exact match ratio) rather than per-label accuracy, and `_num_correct` was made a scalar with no `average` dispatch mechanism. `Precision` and `Recall`, added in the same commit, were given the `_BasePrecisionRecall` architecture that includes `average` dispatch and tensor accumulators. `Accuracy` was never brought to parity.

The specific fix is at `accuracy.py:304–315`: replace `torch.all(..., dim=-1)` with element-wise comparison and column-wise summing when `average="label-wise"` is requested.

#### E — Edge Cases to Handle

| Edge Case | Handling |
|---|---|
| `average="label-wise"` with `is_multilabel=False` | Raise `ValueError` in `__init__()` |
| Multi-batch update with different batch sizes | `_num_examples` is a count (not a tensor), so cross-batch averaging is correct |
| NHW (spatial) multilabel inputs `(N, C, H, W)` | The reshape at line 308–309 flattens to `(N*H*W, C)` before the `correct` computation; per-label correct sum over `dim=0` still gives `(C,)` correctly |
| Distributed training (`@sync_all_reduce`) | Tensor `_num_correct` of shape `(C,)` is all-reduced correctly; scalar `_num_examples` is unchanged |
| Zero examples in `compute()` | The existing `NotComputableError` guard at line 319 covers this path |

---

## Contribution Plan

| Phase | Task |
|-------|------|
| **Exploration** | Fork the repo, study `ignite/metrics/accuracy.py`, `precision.py`, `recall.py` and existing tests ✓ |
| **Implementation** | Add `average="label-wise"` to `Accuracy` for `is_multilabel=True` |
| **Testing** | Write unit tests consistent with existing test patterns (compared against scikit-learn per-column) |
| **Documentation** | Update docstrings with RST-formatted examples |
| **PR** | Submit a pull request referencing issue #513 |

---

## Resources

- [pytorch/ignite source — metrics module](https://github.com/pytorch/ignite/tree/master/ignite/metrics)
- [Original issue #513](https://github.com/pytorch/ignite/issues/513)
- [Stalled PR #516 (reference implementation)](https://github.com/pytorch/ignite/pull/516)
- [scikit-learn multilabel metrics (for test reference)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [Ignite metrics test suite](https://github.com/pytorch/ignite/blob/master/tests/ignite/metrics/test_accuracy.py)
