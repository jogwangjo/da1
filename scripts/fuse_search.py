#!/usr/bin/env python3
"""파일 단위 fusion 그리드서치 + 공식 산식 채점.

역할 배정 예:
  --voice-role ssl_aasist_la:ssl_aasist_df  (쉼표=앙상블 평균)
  --music-role sonics_g120s
  --file-role df_arena
presence:
  --presence oracle | preds/panns__presence.csv 경로

FILE_FAKE 후보식:
  F1 max(VP*VF, MP*MF)          (baseline)
  F2 noisy-or(1-(1-VPr)(1-MPr))
  F3 max(VF, MF)
  F4 mean(VPr, MPr) 가중 alpha 혼합 + presence 게임
각 식 x alpha 그리드를 돌려 TOTAL 랭킹 출력.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dacon_metric import evaluate  # noqa


def load_scores(preds_dir, spec, track):
    """spec: model1:model2 (콜론=평균 앙상블). 없는 모델은 skip."""
    parts = []
    for name in spec.split(":"):
        p = Path(preds_dir) / f"{name}__{track}.csv"
        if p.exists():
            parts.append(pd.read_csv(p).set_index("ID")["PROB"])
    if not parts:
        return None
    return pd.concat(parts, axis=1).mean(axis=1).rename("PROB")


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-dir", type=Path, default=Path("pseudo_eval"))
    ap.add_argument("--preds-dir", type=Path, default=Path("preds"))
    ap.add_argument("--voice-role", default="df_arena")
    ap.add_argument("--music-role", default="sonics_g120s")
    ap.add_argument("--file-role", default="df_arena")
    ap.add_argument("--presence", default="oracle",
                    help="'oracle' 또는 presence csv 경로")
    args = ap.parse_args()

    labels = pd.read_csv(args.pseudo_dir / "labels.csv").set_index("ID")

    if args.presence == "oracle":
        vp = labels["VOICE_PRESENT"].astype(float)
        mp = labels["MUSIC_PRESENT"].astype(float)
    else:
        pr = pd.read_csv(args.presence).set_index("ID")
        vp = pr["VOICE_PRESENT_PROB"]
        mp = pr["MUSIC_PRESENT_PROB"]

    vf = load_scores(args.preds_dir, args.voice_role, "voice").reindex(labels.index)
    mf = load_scores(args.preds_dir, args.music_role, "music").reindex(labels.index)

    # 컴포넌트 트랙 점수 -> 파일 트랙에 복사(해당 성분 존재 시 사용)
    VF = np.where(labels["VOICE_PRESENT"] == 1, vf.values, 0.0)
    MF = np.where(labels["MUSIC_PRESENT"] == 1, mf.values, 0.0)

    results = []

    def score_file(file_fake, tag):
        pred = pd.DataFrame({
            "ID": labels.index,
            "FILE_FAKE_PROB": file_fake,
            "VOICE_FAKE_PROB": VF,
            "MUSIC_FAKE_PROB": MF,
            "VOICE_PRESENT_PROB": vp.values,
            "MUSIC_PRESENT_PROB": mp.values,
        })
        r = evaluate(pred.reset_index(drop=True), labels.reset_index())
        results.append((tag, r["TOTAL"], r["ADS"], r["CPS"],
                        r["File_EER"], r["Voice_EER"], r["Music_EER"]))

    vprisk = vp.values * VF
    mprisk = mp.values * MF
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    # F1 baseline max-product
    score_file(np.maximum(vprisk, mprisk), "F1_max_product(baseline)")

    # F2 noisy-or
    score_file(1 - (1 - vprisk) * (1 - mprisk), "F2_noisy_or")

    # F3 raw max
    score_file(np.maximum(VF, MF), "F3_max_raw")

    # F4 alpha 혼합(logit 도메인) + presence 곱
    for a in alphas:
        mix = sigmoid(a * logit(VF) + (1 - a) * logit(MF))
        for g in [0.0, 0.5, 1.0]:
            gate = ((vp.values * mp.values) ** g)
            s = np.maximum(mix * gate * VF, mix * gate * MF) if g > 0 else mix
            score_file(s, f"F4_alpha{a}_gate{g}")

    res = pd.DataFrame(results, columns=[
        "formula", "TOTAL", "ADS", "CPS", "FileEER", "VoiceEER", "MusicEER"
    ]).sort_values("TOTAL", ascending=False)
    res.to_csv(args.preds_dir / "fusion_results.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    main()
