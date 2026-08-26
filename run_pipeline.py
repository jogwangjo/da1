#!/usr/bin/env python3
"""0.8+ 파이프라인 — 하나로 끝.

실행:
  python run_pipeline.py              # 전체 파이프라인
  python run_pipeline.py --step data  # 데이터만
  python run_pipeline.py --step train # 학습만
  python run_pipeline.py --step infer # 추론만
  python run_pipeline.py --step zip   # 제출 zip만

GPU 없으면: data + zip만 가능 (학습/추론은 GPU 필요)
GPU 있으면: 전체 자동 실행
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
TRAIN_DATA = PROJECT / "train_data"
RUNS_DIR = PROJECT / "runs"
SUBMIT_DIR = PROJECT / "submit"
MODEL_DIR = SUBMIT_DIR / "model"
PSEUDO_DIR = PROJECT / "pseudo_eval"


def has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def gpu_name():
    if has_gpu():
        import torch
        return torch.cuda.get_device_name(0)
    return "CPU only"


def sh(cmd, desc=""):
    print(f"  > {desc or cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[-500:]}")
        return False
    return True


# ============================================================
# Step 1: 학습 데이터 구축
# ============================================================
def step_data():
    print("\n" + "=" * 60)
    print("STEP 1: 학습 데이터 구축")
    print("=" * 60)

    if (TRAIN_DATA / "manifest_train.csv").exists():
        print("  Already exists. Skipping. (use --force-data to rebuild)")
        return True

    cmd = (
        f'"{sys.executable}" scripts/build_train_data.py '
        f'--out {TRAIN_DATA} '
        f'--auto-download '
        f'--max-voice-real 2000 '
        f'--max-voice-fake 3000 '
        f'--max-music-real 500 '
        f'--max-music-fake 2000'
    )
    return sh(cmd, "Building training data")


# ============================================================
# Step 2: RAPTOR 파인튜닝
# ============================================================
def step_train():
    print("\n" + "=" * 60)
    print("STEP 2: RAPTOR 파인튜닝")
    print("=" * 60)

    if not has_gpu():
        print("  [SKIP] GPU not available. Training requires GPU.")
        print("  Use Colab with T4 GPU for training.")
        return False

    best_pth = RUNS_DIR / "raptor_v1" / "best.pth"
    if best_pth.exists():
        print(f"  Already exists: {best_pth}")
        print("  Skipping. (use --force-train to retrain)")
        return True

    manifest_train = TRAIN_DATA / "manifest_train.csv"
    manifest_val = TRAIN_DATA / "manifest_val.csv"
    if not manifest_train.exists():
        print("  [ERROR] Training data not found. Run --step data first.")
        return False

    cmd = (
        f'"{sys.executable}" scripts/train_raptor.py '
        f'--train {manifest_train} '
        f'--val {manifest_val} '
        f'--backbone utter-project/mHuBERT-147 '
        f'--out {RUNS_DIR}/raptor_v1 '
        f'--epochs 20 --bs 24 --lr 1e-6 --lr-head 3e-4 '
        f'--consistency-w 0.25 --p-aug 0.5'
    )
    return sh(cmd, "Training RAPTOR")


# ============================================================
# Step 3: 추론
# ============================================================
def step_infer():
    print("\n" + "=" * 60)
    print("STEP 3: 추론 (제출 파일 생성)")
    print("=" * 60)

    if not has_gpu():
        print("  [SKIP] GPU not available. Inference requires GPU.")
        print("  Use Colab with T4 GPU for inference.")
        return False

    # 파인튜닝 모델 복사
    best_pth = RUNS_DIR / "raptor_v1" / "best.pth"
    raptor_dst = MODEL_DIR / "raptor_best.pth"
    if best_pth.exists():
        shutil.copy(best_pth, raptor_dst)
        print(f"  Copied: {best_pth} -> {raptor_dst}")

    test_dir = DATA_DIR / "test"
    sample_sub = DATA_DIR / "sample_submission.csv"
    output = PROJECT / "output" / "submission.csv"

    if not test_dir.exists():
        print("  [ERROR] data/test/ not found.")
        print("  Download test data from DACON competition page.")
        return False
    if not sample_sub.exists():
        print("  [ERROR] data/sample_submission.csv not found.")
        return False

    # script_v2.py 사용 (RAPTOR 있으면 포함, 없으면 DF-Arena only)
    cmd = (
        f'"{sys.executable}" submit/script_v2.py '
        f'--test-dir {test_dir} '
        f'--sample-submission {sample_sub} '
        f'--output {output} '
        f'--device cuda '
        f'--tta 2'
    )
    return sh(cmd, "Running inference")


# ============================================================
# Step 4: 제출 zip 생성
# ============================================================
def step_zip():
    print("\n" + "=" * 60)
    print("STEP 4: 제출 zip 생성")
    print("=" * 60)

    # 파인튜닝 모델이 있으면 script_v2, 없으면 script_baseline
    raptor_dst = MODEL_DIR / "raptor_best.pth"
    if raptor_dst.exists():
        script = SUBMIT_DIR / "script_v2.py"
        print("  Using script_v2.py (with RAPTOR)")
    else:
        script = SUBMIT_DIR / "script_baseline.py"
        print("  Using script_baseline.py (DF-Arena only)")

    output_zip = PROJECT / "submit.zip"
    if output_zip.exists():
        output_zip.unlink()

    with __import__("zipfile").ZipFile(output_zip, "w", __import__("zipfile").ZIP_STORED) as zf:
        # script.py
        zf.write(script, "script.py")
        print(f"  Added: script.py")

        # model/
        for p in sorted(MODEL_DIR.rglob("*")):
            if p.is_file():
                zf.write(str(p), f"model/{p.relative_to(MODEL_DIR)}")
        n_model = sum(1 for _ in MODEL_DIR.rglob("*") if _.is_file())
        print(f"  Added: model/ ({n_model} files)")

        # vendor_sonics/
        vendor = SUBMIT_DIR / "vendor_sonics"
        if vendor.exists():
            for p in sorted(vendor.rglob("*")):
                if p.is_file():
                    zf.write(str(p), f"vendor_sonics/{p.relative_to(vendor)}")
            print(f"  Added: vendor_sonics/")

    size_gb = output_zip.stat().st_size / (1024 ** 3)
    print(f"  Created: {output_zip} ({size_gb:.1f} GB)")
    return True


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="0.8+ pipeline")
    ap.add_argument("--step", choices=["data", "train", "infer", "zip", "all"],
                    default="all")
    ap.add_argument("--force-data", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("0.8+ Audio Deepfake Detection Pipeline")
    print("=" * 60)
    print(f"GPU: {gpu_name()}")
    print(f"Step: {args.step}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.force_data and (TRAIN_DATA / "manifest_train.csv").exists():
        shutil.rmtree(TRAIN_DATA)
    if args.force_train:
        best_pth = RUNS_DIR / "raptor_v1" / "best.pth"
        if best_pth.exists():
            best_pth.unlink()

    steps = {
        "data": step_data,
        "train": step_train,
        "infer": step_infer,
        "zip": step_zip,
    }

    if args.step == "all":
        order = ["data", "train", "infer", "zip"]
    else:
        order = [args.step]

    for s in order:
        ok = steps[s]()
        if not ok and s in ("train", "infer"):
            print(f"\n  Step '{s}' failed. Continuing...")
        elif not ok:
            print(f"\n  Step '{s}' failed. Stopping.")
            break

    print("\n" + "=" * 60)
    print("Done!")
    if has_gpu():
        print(f"Submit: output/submission.csv")
    print(f"Zip: submit.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
