# Open Source Contribution — CodePath AI301
**GitHub:** [@danijimmy19](https://github.com/danijimmy19)

---

## About Me
I'm **Jimmy Dani**, a Computer Science PhD Candidate at Texas A&M University specializing in machine learning for cybersecurity, privacy, and cryptographic evaluation. My research spans deep learning, LLMs, adversarial ML, and secure multi-party computation. I have hands-on experience with PyTorch, scikit-learn, TensorFlow, and the broader ML/AI ecosystem, and I'm participating in CodePath's AI301 course (Summer 2026) to contribute to impactful open source ML projects.

---

## Selected Issue

### 🏷️ Label-wise Metrics (Accuracy, Precision, Recall) for Multi-label Problems
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

PyTorch Ignite's `Accuracy`, `Precision`, and `Recall` metrics for multi-label classification currently return only a single averaged scalar, making it impossible to diagnose per-label model performance. This matters because an overall accuracy of 70% could hide a single failing label or reflect uniform errors across all labels — two very different situations requiring different fixes. The feature was requested in 2019, a community PR (#516) was submitted but never merged due to unresolved API design questions, and as of v0.5.4 the gap still exists. I chose this issue because it is well-scoped, directly relevant to real ML workflows, and my background in PyTorch and metric evaluation makes me well-positioned to complete it properly.

---

## What Needs to Be Built

Extend Ignite's `Accuracy`, `Precision`, and `Recall` metric classes to support a label-wise mode, so that for a multi-label problem with `C` classes, the metric returns a tensor of `C` values (one per label) instead of a single scalar.

**Example of desired behavior:**
```python
# Currently: returns a single float (averaged across all labels)
metric = Accuracy(is_multilabel=True)

# Desired: returns a tensor of per-label accuracies
metric = Accuracy(is_multilabel=True, average="label-wise")
# → tensor([0.75, 0.75, 0.50])  # accuracy for each of the 3 labels
```

### Current State

- The issue has been open since **May 2019** — the feature was requested but never officially merged
- A community PR (#516) was submitted with a partial implementation for `Accuracy` only, but was **closed without merging** due to unresolved API design questions and missing tests for `Precision` and `Recall`
- As of the latest release (v0.5.4), the `Accuracy` class **still does not support label-wise output** for multi-label cases
- This leaves a real gap for practitioners working on multi-label ML tasks

---

## Contribution Plan

| Phase | Task |
|-------|------|
| **Exploration** | Fork the repo, study `ignite/metrics/accuracy.py`, `precision.py`, `recall.py` and existing tests |
| **Implementation** | Add label-wise support to `Accuracy`, `Precision`, and `Recall` for `is_multilabel=True` |
| **Testing** | Write unit tests consistent with existing test patterns (compared against scikit-learn) |
| **Documentation** | Update docstrings with RST-formatted examples |
| **PR** | Submit a pull request referencing issue #513 |

---

## Forked Repository

[github.com/danijimmy19/ignite](https://github.com/danijimmy19/ignite)

---

## Resources

- [pytorch/ignite source — metrics module](https://github.com/pytorch/ignite/tree/master/ignite/metrics)
- [Original issue #513](https://github.com/pytorch/ignite/issues/513)
- [Stalled PR #516 (reference implementation)](https://github.com/pytorch/ignite/pull/516)
- [scikit-learn multilabel metrics (for test reference)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [Ignite metrics test suite](https://github.com/pytorch/ignite/blob/master/tests/ignite/metrics/test_accuracy.py)
