"""
Reproduction script for pytorch/ignite issue #513:
"Label-wise Metrics (Accuracy, Precision, Recall) for Multi-label Problems"

Run from the root of the ignite fork:
    pip install -e ".[dev]"
    python reproduce.py
"""

import torch
from ignite.engine import Engine
from ignite.metrics import Accuracy, Precision, Recall


def eval_step(engine, batch):
    return batch


evaluator = Engine(eval_step)

y_pred = torch.tensor([
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 1],
], dtype=torch.float)

y_true = torch.tensor([
    [0, 0, 1, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1],
], dtype=torch.float)

print("=" * 60)
print("Issue #513 — Label-wise Metrics for Multi-label Problems")
print("=" * 60)

# --- 1. Current Accuracy behavior (subset/exact-match accuracy) ---
print("\n[1] Accuracy(is_multilabel=True) — current behavior")
acc = Accuracy(is_multilabel=True)
acc.attach(evaluator, "accuracy")
state = evaluator.run([[y_pred, y_true]])
print(f"    Result : {state.metrics['accuracy']:.4f}  (single scalar)")
print(f"    Meaning: only sample 1 has ALL 5 labels correct → 1/5 = 0.2")
print(f"    Problem: hides per-label performance entirely")

# --- 2. Attempting the desired API (does not exist yet) ---
print("\n[2] Accuracy(is_multilabel=True, average='label-wise') — desired")
try:
    acc_lw = Accuracy(is_multilabel=True, average="label-wise")
    print("    Result: (unexpected — parameter should not exist yet)")
except TypeError as e:
    print(f"    TypeError: {e}")
    print("    Confirmed: 'average' parameter does not exist on Accuracy")

# --- 3. What per-label accuracy should look like ---
print("\n[3] Expected per-label accuracy (computed manually)")
correct_per_label = (y_pred == y_true).float().mean(dim=0)
rounded = [round(v, 4) for v in correct_per_label.tolist()]
print(f"    Expected: {rounded}")
print(f"    Label 0: {correct_per_label[0]:.2f}  Label 1: {correct_per_label[1]:.2f}  "
      f"Label 2: {correct_per_label[2]:.2f}  Label 3: {correct_per_label[3]:.2f}  "
      f"Label 4: {correct_per_label[4]:.2f}")

# --- 4. Precision already supports per-label via average=False ---
print("\n[4] Precision(is_multilabel=True, average=False) — already works")
prec = Precision(is_multilabel=True, average=False)
prec.attach(evaluator, "precision")
state = evaluator.run([[y_pred, y_true]])
print(f"    Result: {state.metrics['precision'].tolist()}")
print("    Precision returns a per-label tensor — Accuracy needs the same")

# --- 5. Recall already supports per-label via average=False ---
print("\n[5] Recall(is_multilabel=True, average=False) — already works")
rec = Recall(is_multilabel=True, average=False)
rec.attach(evaluator, "recall")
state = evaluator.run([[y_pred, y_true]])
print(f"    Result: {state.metrics['recall'].tolist()}")
print("    Recall also returns a per-label tensor — Accuracy is the gap")

print("\n" + "=" * 60)
print("Root cause: accuracy.py update() line uses torch.all(..., dim=-1)")
print("which collapses all label dimensions to a per-sample bool before")
print("summing. Fix: accumulate per-label correct counts when average='label-wise'.")
print("=" * 60)
