#!/usr/bin/env python3
"""DACON 236749 공식 평가 산식 오프라인 재현.

Score = 0.9 * ADS + 0.1 * CPS
ADS   = 0.5*(1-File EER) + 0.2*(1-Voice EER) + 0.3*(1-Music EER)
CPS   = 0.5*VoicePresenceAUC + 0.5*MusicPresenceAUC

EER: roc_curve(pos_label=1, drop_intermediate=False) 후 |fpr-fnr| 최소점에서 (fpr+fnr)/2
Voice EER은 음성 존재 샘플만, Music EER은 음악 존재 샘플만 계산.

사용법:
    python dacon_metric.py --pred submission.csv --labels labels.csv
labels.csv 컬럼: ID, FILE_FAKE, VOICE_FAKE, MUSIC_FAKE, VOICE_PRESENT, MUSIC_PRESENT
pred csv 컬럼:   ID, FILE_FAKE_PROB, VOICE_FAKE_PROB, MUSIC_FAKE_PROB, VOICE_PRESENT_PROB, MUSIC_PRESENT_PROB
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return (fpr[idx] + fnr[idx]) / 2


def evaluate(pred_df, label_df):
    df = label_df.merge(pred_df, on="ID", how="inner", validate="one_to_one")
    if len(df) != len(label_df):
        raise ValueError(f"ID mismatch: {len(df)} vs {len(label_df)}")

    file_eer = compute_eer(df["FILE_FAKE"], df["FILE_FAKE_PROB"])

    v_mask = df["VOICE_PRESENT"] == 1
    m_mask = df["MUSIC_PRESENT"] == 1
    voice_eer = compute_eer(df.loc[v_mask, "VOICE_FAKE"], df.loc[v_mask, "VOICE_FAKE_PROB"])
    music_eer = compute_eer(df.loc[m_mask, "MUSIC_FAKE"], df.loc[m_mask, "MUSIC_FAKE_PROB"])

    ads = 0.5 * (1 - file_eer) + 0.2 * (1 - voice_eer) + 0.3 * (1 - music_eer)
    cps = 0.5 * roc_auc_score(df["VOICE_PRESENT"], df["VOICE_PRESENT_PROB"]) \
        + 0.5 * roc_auc_score(df["MUSIC_PRESENT"], df["MUSIC_PRESENT_PROB"])
    total = 0.9 * ads + 0.1 * cps

    return {
        "n": len(df),
        "File_EER": file_eer,
        "Voice_EER": voice_eer,
        "Music_EER": music_eer,
        "VoicePresence_AUC": roc_auc_score(df["VOICE_PRESENT"], df["VOICE_PRESENT_PROB"]),
        "MusicPresence_AUC": roc_auc_score(df["MUSIC_PRESENT"], df["MUSIC_PRESENT_PROB"]),
        "ADS": ads,
        "CPS": cps,
        "TOTAL": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    pred = pd.read_csv(args.pred)
    labels = pd.read_csv(args.labels)
    r = evaluate(pred, labels)
    print(f"n={r['n']}")
    print(f"File EER          : {r['File_EER']:.6f}")
    print(f"Voice EER         : {r['Voice_EER']:.6f}  (n_voice={int((labels['VOICE_PRESENT']==1).sum())})")
    print(f"Music EER         : {r['Music_EER']:.6f}  (n_music={int((labels['MUSIC_PRESENT']==1).sum())})")
    print(f"Voice Presence AUC: {r['VoicePresence_AUC']:.6f}")
    print(f"Music Presence AUC: {r['MusicPresence_AUC']:.6f}")
    print("-" * 40)
    print(f"ADS   : {r['ADS']:.6f}")
    print(f"CPS   : {r['CPS']:.6f}")
    print(f"TOTAL : {r['TOTAL']:.6f}")


if __name__ == "__main__":
    main()
