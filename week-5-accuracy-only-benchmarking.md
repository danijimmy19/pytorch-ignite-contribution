# Week 5: Accuracy-Only Benchmarking — Issue #513

**GitHub:** [@danijimmy19](https://github.com/danijimmy19)  
**Issue:** [pytorch/ignite #513](https://github.com/pytorch/ignite/issues/513)  
**Focus:** accuracy-only benchmarking for multilabel classification and label-wise evaluation

---

## Summary

This week focuses on the accuracy-only interpretation of the multilabel metric problem raised in issue #513. The central question is simple but important:

- Standard `Accuracy` for multilabel tasks reports subset accuracy, which is a single sample-level score.
- That makes it hard to see which labels are contributing most to errors.
- A label-wise accuracy view makes the failure mode much more interpretable, especially for imbalanced multilabel datasets.

This is especially relevant for the original use case described in the issue: a multilabel classifier trained with `BCEWithLogitsLoss`, where each sample can have multiple positive labels and the model should be evaluated per label rather than only by exact-match correctness.

---

## Background from the Original Issue

The original issue was raised by [@jphdotam](https://github.com/jphdotam) and described a multilabel setting where a sample could have multiple binary labels, such as:

```python
# Example shape
# y_pred and y might look like [0, 1, 1]
```

The user noted that they were using metrics such as:

```python
Accuracy(output_transform=thresholded_output_transform, is_multilabel=True)
Precision(output_transform=thresholded_output_transform, is_multilabel=True, average=True)
```

and wanted label-specific metrics like three separate accuracies instead of one aggregate score.

### Original Comment (excerpt)

> Hi,
>
> I've made a multi-label classifier using `BCEWithLogitsLoss`. In summary a data sample can be one of 3 binary classes, which aren't mutually exclusive, so `y_pred` and `y` can look something like `[0, 1, 1]`.
>
> My metrics include `Accuracy(output_transform=thresholded_output_transform, is_multilabel=True)` and `Precision(output_transform=thresholded_output_transform, is_multilabel=True, average=True)`.
>
> However, I'm interested in having label-specific metrics (i.e. having 3 accuracies etc.). This is important because it allows me to see what labels are compromising my overall accuracy the most.
>
> There is no option to disable averaging for `Accuracy()` as with the others, and setting `average=False` for `Precision()` does not do what I expected (it yields a binary result per datum, not per label).
>
> Is there a way to get label-wise metrics in multilabel problems? Or a plan to introduce it?

---

## What This Week Covers

### Main Goal

Benchmark the difference between:

1. Standard multilabel subset accuracy
2. Per-label accuracy for each class/label

### Why This Matters

Subset accuracy can hide important weaknesses. For example, a model might appear to have moderate overall performance while one specific label consistently fails. A per-label accuracy breakdown makes that visible immediately.

### Benchmarking Angle

This week’s work focuses on evaluating:

- how accuracy-only metrics behave in multilabel settings,
- how per-label accuracy differs from sample-wise exact-match accuracy,
- and how this can be documented clearly for contributors and users.

---

## Suggested GitHub Comment Draft

Here is a polished comment you can post on the issue thread:

> Hi everyone, I’m Jimmy Dani (@danijimmy19), and I’ve been exploring this issue in the context of multilabel classification and metric design.
>
> The core problem is that standard `Accuracy` for multilabel tasks currently behaves like subset accuracy: it produces one score per sample, which can obscure which labels are actually driving the error. For this use case, a per-label accuracy view is much more informative.
>
> I’m focusing on the accuracy-only part of this problem and documenting how a label-wise accuracy interpretation can complement the existing multilabel metric behavior. I think this direction is especially useful for imbalanced multilabel datasets and for debugging model performance label by label.
>
> I’d be glad to help contribute to this area further and would appreciate any guidance from maintainers on the intended scope and API direction.

---

## Notes for the Contribution Repo

The work in the contribution repository is centered around a lightweight prototype that demonstrates this behavior clearly and keeps the implementation isolated from the core Ignite package.

### Current contribution focus

- A reusable implementation for label-wise accuracy
- A minimal runnable example
- A compact test suite
- A documentation note that explains the motivation and benchmarking angle

### How to test the codebase

From the contribution directory, run:

```bash
cd pytorch-ignite-contribution
python -m pytest test_labelwise_accuracy.py -v
```

To run the example script:

```bash
python labelwise_accuracy_example.py
```

If you want to verify the behavior manually from the repo root:

```bash
cd pytorch-ignite-contribution
python -m pytest -q
```

These commands help validate the implementation, confirm that the metric returns the expected per-label values, and ensure the example remains runnable.

---

## Closing Note

This issue remains an important one because it connects directly to the practical evaluation needs of multilabel models. The distinction between sample-wise accuracy and label-wise accuracy is not just academic; it can materially change how a model is interpreted and improved.
