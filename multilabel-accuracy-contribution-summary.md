# AI301 Contribution Submission

## Week/Phase Summary

This submission documents the work completed for the contribution issue focused on improving multilabel accuracy support in the Ignite contribution prototype.

## Issue Context

The goal of this work was to extend the contribution-side implementation of label-wise multilabel accuracy so that it can be compared more directly with classification-report-style metrics and used in a lightweight benchmark.

## What Was Implemented

### 1. Label-wise multilabel accuracy metric
- Added a reusable implementation of per-label accuracy for multilabel classification in the contribution repository.
- The metric supports:
  - `average="label-wise"` for returning one accuracy value per label
  - `average="macro"` for returning the mean of those per-label accuracies as a scalar
  - `average=None` for preserving standard subset-accuracy behavior

### 2. Benchmark script
- Added a runnable benchmark script that evaluates the metric on real torchvision datasets.
- The script supports:
  - `--mode small` and `--mode large`
  - `--samples-per-class` for controlling dataset size
  - automatic dataset download when data is missing locally

### 3. Example and tests
- Added a minimal runnable example demonstrating the difference between subset accuracy and label-wise accuracy.
- Added regression tests covering:
  - per-label correctness
  - macro-average behavior
  - multi-batch consistency
  - invalid configuration handling

## Files Added or Updated

- `labelwise_accuracy.py`
- `benchmark_labelwise_accuracy.py`
- `labelwise_accuracy_example.py`
- `test_labelwise_accuracy.py`
- `README.md`

## Verification

The implementation was exercised through the benchmark script and validated through the local test suite for the metric behavior.

## Notes for Reviewers

This work is intentionally scoped as a contribution-side prototype rather than a change to the upstream Ignite repository. The goal is to provide a reviewable implementation and supporting artifacts that demonstrate the feature clearly.

## Submission Status

- Phase completed: implementation + validation for the contribution issue
- Current artifact for check-in: this contribution README / submission document
