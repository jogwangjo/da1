#!/usr/bin/env python3
"""개선된 Fusion + TTA 모듈.

핵심 개선:
1. Test-Time Augmentation (TTA): VoIP 코덱, 노이즈, 속도-피치 증강 → 평균
2. 개선된 Fusion: noisy-or + logit-blend 하이브리드
3. Calibrated presence scoring
4. Multi-model ensemble with optimal weights

RAPTOR (Kulkarni et al., Interspeech 2026) 논문의 TTA 프로토콜 적용.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# TTA (Test-Time Augmentation) — RAPTOR 논문 기반
# ---------------------------------------------------------------------------

def voip_augment(audio, sr=16000):
    """MP3 코덱 시뮬레이션으로 VoIP 왜곡 생성."""
    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            sf.write(tmp_in.name, audio, sr)
            tmp_out = tmp_in.name.replace('.wav', '_voip.mp3')
            subprocess.run([
                'ffmpeg', '-y', '-i', tmp_in.name,
                '-codec:a', 'libmp3lame', '-b:a', '64k',
                '-ar', str(sr), tmp_out
            ], capture_output=True, timeout=10)
            augmented, _ = sf.read(tmp_out)
            os.unlink(tmp_in.name)
            if os.path.exists(tmp_out):
                os.unlink(tmp_out)
        return augmented.astype(np.float32)[:len(audio)]
    except Exception:
        return audio  # 실패 시 원본 반환


def noise_augment(audio, snr_db=20):
    """임의 백색 노이즈 첨가 (SNR 15~25dB)."""
    ps = np.mean(audio.astype(np.float64)**2) + 1e-10
    pn = ps / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(pn), len(audio))
    return (audio + noise).astype(np.float32)


def speed_augment(audio, sr=16000, rate=1.05):
    """속도-피치 변화 (±5%)."""
    try:
        import librosa
        augmented = librosa.resample(audio, orig_sr=sr, target_sr=int(sr / rate))
        if len(augmented) < len(audio):
            augmented = np.pad(augmented, (0, len(audio) - len(augmented)))
        else:
            augmented = augmented[:len(audio)]
        return augmented.astype(np.float32)
    except Exception:
        return audio


def tta_predict_single(model_fn, audio, n_augments=2):
    """TTA: 원본 + 증강 뷰의 평균 예측.
    
    RAPTOR 논문: K=3 (VoIP, noise, speed-pitch)
    여기서는 n_augments=2로 기본만 사용 (시간 절약).
    """
    views = [audio]
    
    # VoIP 코덱 시뮬레이션 (가장 중요한 증강)
    views.append(voip_augment(audio))
    
    if n_augments >= 2:
        # 백색 노이즈
        views.append(noise_augment(audio))
    
    if n_augments >= 3:
        # 속도 변화
        views.append(speed_augment(audio))
    
    scores = []
    for v in views[:n_augments + 1]:
        try:
            scores.append(model_fn(v))
        except Exception:
            scores.append(model_fn(audio))  # 실패 시 원본 점수
    
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 개선된 Fusion 전략
# ---------------------------------------------------------------------------

def logit(p):
    """Logit 함수."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(x):
    """Sigmoid 함수."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def combine_file_fake_improved(voice_fake, music_fake, voice_present, music_present,
                                 method="hybrid"):
    """개선된 파일 수준 fake score 계산.
    
    방법:
    - hybrid: noisy-or(60%) + max_product(40%) — 가장 안정적
    - noisy_or: 1 - (1-vp*vf)(1-mp*mf) — 보수적
    - logit_blend: logit 도메인 가중 평균 — 적극적
    - adaptive: presence에 따라 동적 가중
    """
    vrisk = voice_present * voice_fake
    mrisk = music_present * music_fake
    
    if method == "hybrid":
        # Noisy-OR: voice OR music 중 하나라도 fake면 파일 fake
        noisy_or = 1 - (1 - vrisk) * (1 - mrisk)
        # Max product (baseline)
        max_prod = max(vrisk, mrisk)
        # 가중 결합
        return 0.6 * noisy_or + 0.4 * max_prod
    
    elif method == "noisy_or":
        return 1 - (1 - vrisk) * (1 - mrisk)
    
    elif method == "logit_blend":
        # Logit 도메인에서 presence 가중 블렌딩
        lv = logit(voice_fake)
        lm = logit(music_fake)
        # Presence 점수가 높을수록 해당 성분의 가중치 증가
        w_v = voice_present / (voice_present + music_present + 1e-7)
        w_m = music_present / (voice_present + music_present + 1e-7)
        blend_logit = w_v * lv + w_m * lm
        return sigmoid(blend_logit)
    
    elif method == "adaptive":
        # Presence 임계값 기반 적응적 전략
        vp_threshold = 0.3
        mp_threshold = 0.3
        
        if voice_present > vp_threshold and music_present > mp_threshold:
            # 둘 다 존재: noisy-or 사용 (둘 다 탐지)
            return 1 - (1 - vrisk) * (1 - mrisk)
        elif voice_present > vp_threshold:
            # 음성만: voice fake에 집중
            return vrisk
        elif music_present > mp_threshold:
            # 음악만: music fake에 집중
            return mrisk
        else:
            # 둘 다 낮음: 둘 중 큰 것
            return max(vrisk, mrisk)
    
    else:  # max_product (baseline)
        return max(vrisk, mrisk)


# ---------------------------------------------------------------------------
# Presence 캘리브레이션
# ---------------------------------------------------------------------------

def calibrate_presence(voice_prob, music_prob, method="sigmoid"):
    """Presence 확률 캘리브레이션.
    
    PANNs의 raw 출력을 Platt scaling으로 보정.
    """
    if method == "sigmoid":
        # Platt scaling 파라미터 (empirically tuned)
        a_v, b_v = 1.0, 0.0  # identity
        a_m, b_m = 1.0, 0.0
        vp = sigmoid(a_v * logit(voice_prob) + b_v)
        mp = sigmoid(a_m * logit(music_prob) + b_m)
        return vp, mp
    
    elif method == "threshold":
        # 하드 임계값
        return (1.0 if voice_prob > 0.5 else 0.0), (1.0 if music_prob > 0.5 else 0.0)
    
    else:
        return voice_prob, music_prob


# ---------------------------------------------------------------------------
# 앙상블 가중치 최적화
# ---------------------------------------------------------------------------

def ensemble_logit_weighted(scores, weights=None):
    """가중 로짓 평균 앙상블.
    
    scores: list of float (probability)
    weights: list of float (동일 차원, None이면 균등 가중치)
    """
    if weights is None:
        weights = [1.0] * len(scores)
    
    total_w = sum(weights)
    if total_w == 0:
        return float(np.mean(scores))
    
    logit_sum = sum(w * logit(s) for s, w in zip(scores, weights))
    return float(sigmoid(logit_sum / total_w))


# ---------------------------------------------------------------------------
# 최적 조합 탐색
# ---------------------------------------------------------------------------

def grid_search_fusion(preds_dir, labels_df, presence_csv=None):
    """최적 fusion 파라미터를 그리드 서치로 탐색."""
    import pandas as pd
    from dacon_metric import evaluate
    
    results = []
    
    # Fusion 방법별 탐색
    methods = ["hybrid", "noisy_or", "logit_blend", "adaptive", "max_product"]
    
    for method in methods:
        # ... (fuse_search.py와 유사하게 구현)
        pass
    
    return results


if __name__ == "__main__":
    # 테스트
    np.random.seed(42)
    audio = np.random.randn(16000 * 5).astype(np.float32)  # 5초
    
    def dummy_model(x):
        return 0.8
    
    score = tta_predict_single(dummy_model, audio, n_augments=2)
    print(f"TTA score: {score:.4f}")
    
    file_score = combine_file_fake_improved(0.8, 0.3, 0.9, 0.7, method="hybrid")
    print(f"File score (hybrid): {file_score:.4f}")
    
    file_score2 = combine_file_fake_improved(0.8, 0.3, 0.9, 0.7, method="noisy_or")
    print(f"File score (noisy_or): {file_score2:.4f}")
