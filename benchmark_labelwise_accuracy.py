"""Benchmark label-wise multilabel accuracy on real torchvision datasets.

This script can run in two modes:

- ``small``: a lightweight benchmark using a few smaller vision datasets and a
  modest number of examples per class.
- ``large``: a fuller benchmark that includes more datasets and more samples,
  which is better suited to a machine with more memory and compute.

The predictions and labels are converted to a multilabel one-vs-rest form so
that:

- Ignite's standard subset accuracy can be compared with scikit-learn's subset
  accuracy,
- ``LabelwiseAccuracy`` can report one accuracy per class,
- and scikit-learn's classification report can provide a familiar summary.

The purpose is to make the behavior easier to inspect and discuss in issue
threads, PR reviews, or educational demos.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from ignite.metrics import Accuracy
from sklearn.metrics import accuracy_score, classification_report

from labelwise_accuracy import LabelwiseAccuracy


@dataclass
class BenchmarkResult:
    dataset_name: str
    subset_accuracy: float
    sklearn_subset_accuracy: float
    labelwise_accuracy: list[float]
    sklearn_labelwise_accuracy: list[float]
    classification_report_text: str


@dataclass
class SampleBatch:
    images: torch.Tensor
    labels: torch.Tensor


@dataclass
class DatasetSpec:
    name: str
    loader: Callable[..., Any]
    num_classes: int
    kwargs: dict[str, Any]
    transform: transforms.Compose
    class_names: list[str]


def _build_transform(grayscale: bool = False) -> transforms.Compose:
    pipeline: list[Any] = [transforms.Resize((224, 224))]
    if grayscale:
        pipeline.append(transforms.Grayscale(num_output_channels=3))
    pipeline.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(pipeline)


def _to_python_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _sample_per_class(dataset: Any, samples_per_class: int, num_classes: int) -> SampleBatch:
    class_to_samples: dict[int, list[tuple[torch.Tensor, int]]] = {i: [] for i in range(num_classes)}

    for sample in dataset:
        image, label = sample
        class_to_samples[int(label)].append((image, int(label)))

    selected: list[tuple[torch.Tensor, int]] = []
    for class_idx in range(num_classes):
        selected.extend(class_to_samples[class_idx][:samples_per_class])

    images = torch.stack([image for image, _ in selected])
    labels = torch.tensor([label for _, label in selected], dtype=torch.long)
    return SampleBatch(images=images, labels=labels)


def load_dataset(spec: DatasetSpec, root: str | Path, samples_per_class: int) -> SampleBatch | None:
    root_path = Path(root)
    try:
        dataset = spec.loader(root=root_path, **spec.kwargs, transform=spec.transform)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Skipping {spec.name}: {exc}")
        return None
    return _sample_per_class(dataset, samples_per_class=samples_per_class, num_classes=spec.num_classes)


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    torch.manual_seed(7)
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.to(device)
    model.eval()
    return model


def get_dataset_specs(mode: str) -> list[DatasetSpec]:
    if mode == "small":
        return [
            DatasetSpec(
                name="MNIST",
                loader=datasets.MNIST,
                num_classes=10,
                kwargs={"train": False, "download": True},
                transform=_build_transform(grayscale=True),
                class_names=[str(idx) for idx in range(10)],
            ),
            DatasetSpec(
                name="FashionMNIST",
                loader=datasets.FashionMNIST,
                num_classes=10,
                kwargs={"train": False, "download": True},
                transform=_build_transform(grayscale=True),
                class_names=[
                    "T-shirt/top",
                    "Trouser",
                    "Pullover",
                    "Dress",
                    "Coat",
                    "Sandal",
                    "Shirt",
                    "Sneaker",
                    "Bag",
                    "Ankle boot",
                ],
            ),
        ]

    return [
        DatasetSpec(
            name="CIFAR-10",
            loader=datasets.CIFAR10,
            num_classes=10,
            kwargs={"train": False, "download": True},
            transform=_build_transform(grayscale=False),
            class_names=[
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
        ),
        DatasetSpec(
            name="CIFAR-100",
            loader=datasets.CIFAR100,
            num_classes=100,
            kwargs={"train": False, "download": True},
            transform=_build_transform(grayscale=False),
            class_names=[f"class_{idx}" for idx in range(100)],
        ),
        DatasetSpec(
            name="SVHN",
            loader=datasets.SVHN,
            num_classes=10,
            kwargs={"split": "test", "download": True},
            transform=_build_transform(grayscale=False),
            class_names=[str(idx) for idx in range(10)],
        ),
    ]


def evaluate_dataset(spec: DatasetSpec, samples_per_class: int, device: str) -> BenchmarkResult:
    device_obj = torch.device(device)
    batch = load_dataset(spec, root=".", samples_per_class=samples_per_class)
    if batch is None:
        return BenchmarkResult(
            dataset_name=spec.name,
            subset_accuracy=float("nan"),
            sklearn_subset_accuracy=float("nan"),
            labelwise_accuracy=[],
            sklearn_labelwise_accuracy=[],
            classification_report_text="Skipped: dataset not available locally.",
        )

    model = build_model(spec.num_classes, device_obj)

    with torch.no_grad():
        logits = model(batch.images.to(device_obj))
    predictions = logits.argmax(dim=1).cpu()

    true_labels = batch.labels
    true_multilabel = torch.nn.functional.one_hot(true_labels, num_classes=spec.num_classes).float()
    pred_multilabel = torch.nn.functional.one_hot(predictions, num_classes=spec.num_classes).float()

    subset_metric = Accuracy(is_multilabel=True)
    subset_metric.update((pred_multilabel, true_multilabel))
    ignite_subset = _to_python_float(subset_metric.compute())

    labelwise_metric = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
    labelwise_metric.update((pred_multilabel, true_multilabel))
    labelwise_result = labelwise_metric.compute()
    if isinstance(labelwise_result, torch.Tensor):
        ignite_labelwise = [_to_python_float(value) for value in labelwise_result.detach().cpu().tolist()]
    else:
        ignite_labelwise = [_to_python_float(labelwise_result)]

    sklearn_subset = float(accuracy_score(true_multilabel.numpy(), pred_multilabel.numpy()))
    sklearn_labelwise = [
        float(accuracy_score(true_multilabel[:, idx].numpy(), pred_multilabel[:, idx].numpy()))
        for idx in range(true_multilabel.shape[1])
    ]

    report_text = classification_report(
        true_labels.numpy(),
        predictions.numpy(),
        target_names=spec.class_names,
        zero_division=0,
    )

    return BenchmarkResult(
        dataset_name=spec.name,
        subset_accuracy=ignite_subset,
        sklearn_subset_accuracy=sklearn_subset,
        labelwise_accuracy=ignite_labelwise,
        sklearn_labelwise_accuracy=sklearn_labelwise,
        classification_report_text=report_text,
    )


def print_report(results: Sequence[BenchmarkResult]) -> None:
    print("Real-data multilabel benchmark")
    print("=" * 48)
    for result in results:
        print(f"\nDataset: {result.dataset_name}")
        if not result.labelwise_accuracy:
            print(f"  {result.classification_report_text}")
            continue
        print(f"  Subset accuracy | Ignite: {result.subset_accuracy:.4f} | sklearn: {result.sklearn_subset_accuracy:.4f}")
        print("  Per-label accuracy:")
        for idx, (ignite_value, sklearn_value) in enumerate(zip(result.labelwise_accuracy, result.sklearn_labelwise_accuracy)):
            print(f"    Label {idx:02d}: Ignite {ignite_value:.4f} | sklearn {sklearn_value:.4f}")
        print("\n  Classification report (multiclass view):")
        print(result.classification_report_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark label-wise accuracy on multiple torchvision datasets")
    parser.add_argument(
        "--mode",
        choices=["small", "large"],
        default="small",
        help="small keeps the run lightweight and avoids downloading large datasets; large opts into additional datasets",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="How many samples to draw from each class (defaults to 4 for small mode and 20 for large mode)",
    )
    parser.add_argument("--device", default="cpu", help="Computation device to use")
    args = parser.parse_args()

    if args.samples_per_class is None:
        samples_per_class = 4 if args.mode == "small" else 20
    else:
        if args.samples_per_class <= 0:
            raise ValueError("--samples-per-class must be a positive integer")
        samples_per_class = args.samples_per_class

    specs = get_dataset_specs(mode=args.mode)
    results = [evaluate_dataset(spec, samples_per_class=samples_per_class, device=args.device) for spec in specs]
    print_report(results)


if __name__ == "__main__":
    main()
