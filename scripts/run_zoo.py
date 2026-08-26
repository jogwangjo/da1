#!/usr/bin/env python3
"""zoo 모델들을 pseudo-eval에 일괄 추론.

출력
  preds/<model>__voice.csv : ID, PROB (refs/voice 트랙)
  preds/<model>__music.csv : ID, PROB (refs/music 트랙)
  preds/<model>__file.csv  : ID, PROB (test/ 파일 트랙)
  preds/panns__presence.csv: ID, VOICE_PRESENT_PROB, MUSIC_PRESENT_PROB

극성 자동판정: refs 라벨과 비교해 fake 쪽 점수가 낮으면 자동 반전 후 저장.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def read_labels(path):
    return pd.read_csv(path)


def polarity_fix(probs, fakes):
    """fake 평균이 real보다 낮으면 반전."""
    probs = np.asarray(probs)
    fakes = np.asarray(fakes)
    if len(np.unique(fakes)) < 2:
        return probs, False
    if probs[fakes == 1].mean() < probs[fakes == 0].mean():
        return -probs + min(probs) * 2, True  # 순위 보존 선형반전(음수 방지)
    return probs, False


def run_fake_model(model, wav_dir, label_csv, out_csv, device):
    labels = read_labels(label_csv)
    rows = []
    for _, r in labels.iterrows():
        p = model.predict_file(str(wav_dir / f"{r['ID']}.wav"))
        rows.append((r["ID"], p))
        if len(rows) % 50 == 0:
            print(f"  {len(rows)}/{len(labels)}")
    probs, inverted = polarity_fix([p for _, p in rows], labels["FAKE"].values)
    if inverted:
        print(f"  [!] 극성 반전 적용")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "PROB"])
        for (i, _), pr in zip(rows, probs):
            w.writerow([i, round(float(pr), 10)])


def run_presence(model, test_dir, out_csv):
    ids = sorted(p.stem for p in test_dir.glob("TEST_*.wav"))
    rows = []
    for i in ids:
        vp, mp = model.predict_file(str(test_dir / f"{i}.wav"))
        rows.append([i, round(vp, 10), round(mp, 10)])
        if len(rows) % 50 == 0:
            print(f"  {len(rows)}/{len(ids)}")
    pd.DataFrame(rows, columns=["ID", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"]).to_csv(
        out_csv, index=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-dir", type=Path, default=Path("pseudo_eval"))
    ap.add_argument("--out-dir", type=Path, default=Path("preds"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tracks", nargs="+", default=["voice", "music", "file"])
    args = ap.parse_args()

    from zoo import REGISTRY

    args.out_dir.mkdir(exist_ok=True)
    for name in args.models:
        print(f"=== {name} ===")
        model = REGISTRY[name]()
        model.load(args.device)

        if model.capability == "presence":
            run_presence(model, args.pseudo_dir / "test",
                         args.out_dir / f"{name}__presence.csv")
        else:
            if "voice" in args.tracks:
                run_fake_model(model, args.pseudo_dir / "refs" / "voice",
                               args.pseudo_dir / "refs_voice.csv",
                               args.out_dir / f"{name}__voice.csv", args.device)
            if "music" in args.tracks:
                run_fake_model(model, args.pseudo_dir / "refs" / "music",
                               args.pseudo_dir / "refs_music.csv",
                               args.out_dir / f"{name}__music.csv", args.device)
            if "file" in args.tracks:
                # file 트랙: test 전체. 극성은 voice 트랙 결과 재사용.
                lab = args.pseudo_dir / "labels.csv"
                df = read_labels(lab)
                rows = []
                for _, r in df.iterrows():
                    p = model.predict_file(str(args.pseudo_dir / "test" / f"{r['ID']}.wav"))
                    rows.append((r["ID"], p))
                vcsv = args.out_dir / f"{name}__voice.csv"
                inv = False
                if vcsv.exists():
                    probe = read_labels(args.pseudo_dir / "refs_voice.csv")
                    pv = pd.read_csv(vcsv)
                    merged = probe.merge(pv, on="ID")
                    _, inv = polarity_fix(merged["PROB"], merged["FAKE"])
                if inv:
                    rows = [(i, 1.0 - p) for i, p in rows]
                    print("  [!] file 트랙 극성 반전 적용")
                with open(args.out_dir / f"{name}__file.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["ID", "PROB"])
                    for i, p in rows:
                        w.writerow([i, round(float(min(max(p, 1e-9), 1 - 1e-9)), 10)])
        del model
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("done.")


if __name__ == "__main__":
    main()
