#!/usr/bin/env python3
"""Validation 기반 Threshold 최적화.

训练된 모델의 validation set에서 최적 임계값을 찾아서
제출 시 성능을 극대화.

사용법:
  python scripts/optimize_thresholds.py \
      --model runs/raptor_v2/best.pth \
      --val train_data/manifest_val.csv \
      --output runs/raptor_v2/thresholds.json
"""

import argparse
import json
import csv
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_curve


def compute_eer_threshold(y_true, y_score):
    """Find optimal threshold for EER."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return 0.5, 1.0

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    threshold = float(thresholds[idx])
    return threshold, eer


def find_optimal_thresholds(model, val_loader, device):
    """Find optimal thresholds for each output."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for wav, y in val_loader:
            logits = model(wav.to(device))
            scores = torch.sigmoid(logits).cpu().tolist()
            all_scores.extend(scores)
            all_labels.extend(y.squeeze(-1).tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Find threshold for different operating points
    thresholds = {}

    # 1. EER threshold (equal error rate)
    eer_thresh, eer = compute_eer_threshold(all_labels, all_scores)
    thresholds["eer_threshold"] = eer_thresh
    thresholds["eer"] = eer

    # 2. FPR=1% threshold (low false positive rate)
    fpr, tpr, threshs = roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    idx_1pct = int(np.argmin(np.abs(fpr - 0.01)))
    thresholds["fpr1_threshold"] = float(threshs[idx_1pct])
    thresholds["fpr1_tpr"] = float(tpr[idx_1pct])

    # 3. FNR=1% threshold (low false negative rate)
    idx_fnr1 = int(np.argmin(np.abs(fnr - 0.01)))
    thresholds["fnr1_threshold"] = float(threshs[idx_fnr1])
    thresholds["fnr1_fpr"] = float(fpr[idx_fnr1])

    # 4. Youden's J statistic (maximizes TPR - FPR)
    j_scores = tpr - fpr
    idx_j = int(np.argmax(j_scores))
    thresholds["youden_threshold"] = float(threshs[idx_j])
    thresholds["youden_j"] = float(j_scores[idx_j])

    # 5. Class-weighted optimal (since competition weights fake detection)
    # Minimize weighted error: 0.5 * FPR + 0.5 * FNR
    weighted_error = 0.5 * fpr + 0.5 * fnr
    idx_weighted = int(np.argmin(weighted_error))
    thresholds["weighted_threshold"] = float(threshs[idx_weighted])
    thresholds["weighted_error"] = float(weighted_error[idx_weighted])

    # Score distribution statistics
    thresholds["score_mean_real"] = float(np.mean(all_scores[all_labels == 0]))
    thresholds["score_mean_fake"] = float(np.mean(all_scores[all_labels == 1]))
    thresholds["score_std_real"] = float(np.std(all_scores[all_labels == 0]))
    thresholds["score_std_fake"] = float(np.std(all_scores[all_labels == 1]))

    return thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--output", type=Path, default=Path("runs/raptor_v2/thresholds.json"))
    ap.add_argument("--bs", type=int, default=24)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    from train_raptor_v2 import UltimateDetector, FakeDetectDataset, SEG
    from torch.utils.data import DataLoader

    ck = torch.load(args.model, map_location=device, weights_only=False)
    model = UltimateDetector(dropout=0.1)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    # Load validation data
    val_ds = FakeDetectDataset(args.val, train=False)
    val_loader = DataLoader(val_ds, batch_size=args.bs, num_workers=2, pin_memory=True)

    print(f"Optimizing thresholds on {len(val_ds)} validation samples...")
    thresholds = find_optimal_thresholds(model, val_loader, device)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nOptimal thresholds:")
    print(f"  EER threshold: {thresholds['eer_threshold']:.4f} (EER={thresholds['eer']:.4f})")
    print(f"  FPR=1% threshold: {thresholds['fpr1_threshold']:.4f} (TPR={thresholds['fpr1_tpr']:.4f})")
    print(f"  Youden threshold: {thresholds['youden_threshold']:.4f} (J={thresholds['youden_j']:.4f})")
    print(f"  Score distribution:")
    print(f"    Real: mean={thresholds['score_mean_real']:.4f}, std={thresholds['score_std_real']:.4f}")
    print(f"    Fake: mean={thresholds['score_mean_fake']:.4f}, std={thresholds['score_std_fake']:.4f}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
