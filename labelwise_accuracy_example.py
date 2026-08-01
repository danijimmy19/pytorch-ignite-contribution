"""Minimal runnable example for the label-wise accuracy metric."""

import torch
from ignite.metrics import Accuracy

from labelwise_accuracy import LabelwiseAccuracy


def main() -> None:
    y_pred = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=torch.long,
    )
    y_true = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=torch.long,
    )

    subset_metric = Accuracy(is_multilabel=True)
    subset_metric.update((y_pred, y_true))
    subset_result = subset_metric.compute()

    labelwise_metric = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    labelwise_metric.update((y_pred, y_true))
    labelwise_result = labelwise_metric.compute()

    print("Subset accuracy:", subset_result)
    print("Per-label accuracy:", labelwise_result.tolist())


if __name__ == "__main__":
    main()
