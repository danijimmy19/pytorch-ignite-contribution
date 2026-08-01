# Label-wise accuracy contribution prototype

This folder contains a small, self-contained prototype for adding label-wise
multilabel accuracy support to Ignite.

## Files

- [labelwise_accuracy.py](labelwise_accuracy.py): implementation-focused module that defines the reusable `LabelwiseAccuracy` class.
- [labelwise_accuracy_example.py](labelwise_accuracy_example.py): minimal runnable example for the metric.
- [benchmark_labelwise_accuracy.py](benchmark_labelwise_accuracy.py): real-data benchmark script using CIFAR-10 samples and a simple ResNet-18 model, comparing Ignite's subset accuracy and label-wise accuracy against scikit-learn references and a classification report.
- [test_labelwise_accuracy.py](test_labelwise_accuracy.py): compact regression tests for the implementation.

## Run the example

```bash
cd /Users/jimmydani/Documents/Pixel Playground/CodePath-Summer-2026/pytorch-ignite-contribution
python3 labelwise_accuracy_example.py
```

## Run the tests

```bash
cd /Users/jimmydani/Documents/Pixel Playground/CodePath-Summer-2026/pytorch-ignite-contribution
python3 -m pytest -q test_labelwise_accuracy.py
```

## Run the benchmark script

### Lightweight benchmark

```bash
cd /Users/jimmydani/Documents/Pixel Playground/CodePath-Summer-2026/pytorch-ignite-contribution
python3 benchmark_labelwise_accuracy.py --mode small
```

### Full benchmark

```bash
cd /Users/jimmydani/Documents/Pixel Playground/CodePath-Summer-2026/pytorch-ignite-contribution
python3 benchmark_labelwise_accuracy.py --mode large
```

You can also control how many samples are used per class:

```bash
python3 benchmark_labelwise_accuracy.py --mode small --samples-per-class 2
```

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
