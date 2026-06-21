"""
Demo: LabelwiseAccuracy in action (issue #513 fix)

Run from pytorch-ignite-contribution/:
    /opt/anaconda3/envs/byte/bin/python demo.py

Shows the before/after behavior and verifies the result manually.
"""

import torch
from ignite.metrics import Accuracy
from labelwise_accuracy import LabelwiseAccuracy

# ---------------------------------------------------------------------------
# Sample data: 5 samples, 3 labels
# ---------------------------------------------------------------------------
y_pred = torch.tensor([
    [1, 0, 1],   # sample 0
    [0, 1, 0],   # sample 1
    [1, 1, 0],   # sample 2
    [0, 0, 1],   # sample 3
    [1, 1, 1],   # sample 4
])

y_true = torch.tensor([
    [1, 0, 0],   # sample 0 — label 2 wrong
    [0, 1, 0],   # sample 1 — all correct
    [1, 0, 0],   # sample 2 — label 1 wrong
    [0, 0, 1],   # sample 3 — all correct
    [1, 1, 0],   # sample 4 — label 2 wrong
])

print("=" * 55)
print("Sample data (5 samples, 3 labels)")
print("=" * 55)
print(f"y_pred:\n{y_pred}\n")
print(f"y_true:\n{y_true}\n")

# ---------------------------------------------------------------------------
# BEFORE: standard Accuracy returns a single scalar (subset accuracy)
# ---------------------------------------------------------------------------
acc = Accuracy(is_multilabel=True)
acc.reset()
acc.update((y_pred, y_true))
subset_result = acc.compute()

print("=" * 55)
print("BEFORE — Accuracy(is_multilabel=True)")
print("=" * 55)
print(f"Result : {subset_result:.4f}  ← single float")
print("Meaning: only samples 1 and 3 have ALL 3 labels correct")
print("         → 2 out of 5 = 0.4000")
print("Problem: hides which specific labels are failing\n")

# ---------------------------------------------------------------------------
# AFTER: LabelwiseAccuracy returns a tensor of shape (C,)
# ---------------------------------------------------------------------------
lw_acc = LabelwiseAccuracy(is_multilabel=True, average="label-wise")
lw_acc.reset()
lw_acc.update((y_pred, y_true))
labelwise_result = lw_acc.compute()

print("=" * 55)
print("AFTER  — LabelwiseAccuracy(average='label-wise')")
print("=" * 55)
print(f"Result : {labelwise_result.tolist()}")
print(f"Shape  : {tuple(labelwise_result.shape)}  ← one value per label\n")

# ---------------------------------------------------------------------------
# Manual verification (so you can see exactly why each number is what it is)
# ---------------------------------------------------------------------------
print("=" * 55)
print("Manual verification (per-label breakdown)")
print("=" * 55)
for label_idx in range(y_pred.shape[1]):
    col_pred = y_pred[:, label_idx]
    col_true = y_true[:, label_idx]
    correct = (col_pred == col_true)
    n_correct = correct.sum().item()
    n_total = len(correct)
    acc_val = n_correct / n_total
    print(f"  Label {label_idx}: {col_pred.tolist()} vs {col_true.tolist()} "
          f"→ {n_correct}/{n_total} = {acc_val:.4f}")
print()

# ---------------------------------------------------------------------------
# Multi-batch: verify that splitting into batches gives the same result
# ---------------------------------------------------------------------------
lw_acc.reset()
for i in range(5):                          # feed one sample at a time
    lw_acc.update((y_pred[i:i+1], y_true[i:i+1]))
incremental_result = lw_acc.compute()

print("=" * 55)
print("Multi-batch check (one sample at a time)")
print("=" * 55)
print(f"Single-batch result : {labelwise_result.tolist()}")
print(f"Incremental result  : {incremental_result.tolist()}")
match = torch.allclose(labelwise_result, incremental_result)
print(f"Match               : {match}  ✓\n")

# ---------------------------------------------------------------------------
# Fallback: average=None still returns a float (backward compatible)
# ---------------------------------------------------------------------------
fallback = LabelwiseAccuracy(is_multilabel=True, average=None)
fallback.reset()
fallback.update((y_pred, y_true))
fallback_result = fallback.compute()

print("=" * 55)
print("Fallback — LabelwiseAccuracy(average=None)")
print("=" * 55)
print(f"Result : {fallback_result:.4f}  ← same as standard Accuracy")
print(f"Type   : {type(fallback_result).__name__}")
print(f"Matches standard Accuracy: {abs(fallback_result - subset_result) < 1e-9}  ✓")
