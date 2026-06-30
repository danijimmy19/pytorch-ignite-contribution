# Week 4: Benchmarking Research — Dataset Curation

**GitHub:** [@danijimmy19](https://github.com/danijimmy19)
**Working branch:** `feature/labelwise-multi-label-metrics`
**Issue:** [pytorch/ignite #513](https://github.com/pytorch/ignite/issues/513)

---

## Focus This Week

Following the Phase III implementation of `LabelwiseAccuracy`, this week shifts to validating the metric against real-world multilabel datasets. The goal is to identify a curated set of benchmarks where per-label accuracy is a meaningful diagnostic — i.e., datasets with class imbalance, label correlation, or varying per-label difficulty where subset accuracy would mask important signal.

---

## Activities

- **Studying the ignite repo** — reviewing how existing metrics (e.g., `Precision`, `Recall`, `F1`) are benchmarked and tested against real datasets in the `examples/` and `tests/` directories, to understand the expected integration pattern.
- **Curating benchmark datasets** — identifying multilabel classification datasets suitable for demonstrating `LabelwiseAccuracy` in a realistic setting. Criteria:
  - Multilabel (each sample can have multiple simultaneous labels)
  - Publicly available and loadable via `torchvision` or `torchtext` (or minimal extra deps)
  - Known class imbalance or label correlation, so per-label breakdown is informative

---

## Dataset Candidates Under Consideration

| Dataset | Labels | Domain | Notes |
|---|---|---|---|
| **MS-COCO** | 80 | Image (object detection) | Highly imbalanced; per-label accuracy exposes rare-class failures |
| **Pascal VOC 2007/2012** | 20 | Image (object classification) | Standard multilabel benchmark; available via `torchvision.datasets.VOCDetection` |
| **NUS-WIDE** | 81 | Image + tags | Real-world label co-occurrence; highlights correlated label groups |
| **Reuters-21578** | ~90 | Text (news) | Classic multilabel text benchmark; imbalanced across topic labels |
| **Delicious** | 983 | Text (bookmarks) | High-cardinality labels; stress-tests per-label output shape |

---

## Next Steps

- Run `LabelwiseAccuracy` against baselines on selected datasets and compare output to `sklearn.metrics.accuracy_score` per column (same validation approach used in the unit tests).
- Document results as a benchmarking demo, suitable for inclusion in the PR or issue thread.
