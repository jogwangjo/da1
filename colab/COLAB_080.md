# Colab 0.8+ 원클릭 가이드

**목표**: 총점 0.81 이상 달성  
**소요시간**: T4 기준 약 6~8시간 (학습 4h + 평가 2h)  
**필요 GPU**: T4 (15GB) 이상 권장  

---

## 셀 1 — 환경 세팅 (최초 1회)

```python
!pip -q install librosa soundfile transformers accelerate demucs panns-inference onnxruntime-gpu datasets huggingface_hub scikit-learn scipy
!apt -qq install -y ffmpeg > /dev/null
```

## 셀 2 — 프로젝트 로드

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/dacon1
```

## 셀 3 — 학습 데이터 구축 (ASVspoof + Codecfake + SONICS + LibriSpeech + FMA + MUSDB)

```python
# ASVspoof 2019 LA 다운로드 (약 10분)
!pip -q install gdown
!gdown --folder "https://drive.google.com/drive/folders/1FOTqQE4b_eVPhm_OdCkjCPjD7JN20ifO" -O /tmp/asvspoof2019/ 2>/dev/null || true

# 또는 직접 다운로드 (Zenodo/Edinburgh)
!mkdir -p /tmp/asvspoof2019
!curl -L -o /tmp/asvspoof2019/ASVspoof2019_LA.zip \
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/ASVspoof2019.LA.zip" 2>/dev/null || true
!cd /tmp/asvspoof2019 && unzip -q ASVspoof2019_LA.zip 2>/dev/null || true

# 학습 데이터 구축
!python scripts/build_train_data.py \
    --out train_data \
    --asvspoof-dir /tmp/asvspoof2019/ASVspoof2019_LA \
    --auto-download \
    --max-voice-real 3000 \
    --max-voice-fake 5000 \
    --max-music-real 700 \
    --max-music-fake 3000
```

> **만약 ASVspoof 다운로드가 안 되면**: `--asvspoof-dir` 인자 없이 실행. Codecfake + SONICS만으로도 음악은 강화 가능.

## 셀 4 — RAPTOR 파인튜닝 (핵심! 4시간)

```python
# mHuBERT-Iter2 + RAPTOR fusion + consistency regularization
!python scripts/train_raptor.py \
    --train train_data/manifest_train.csv \
    --val train_data/manifest_val.csv \
    --backbone utter-project/mHuBERT-147 \
    --out runs/raptor_v1 \
    --epochs 30 \
    --bs 24 \
    --lr 1e-6 \
    --lr-head 3e-4 \
    --consistency-w 0.25 \
    --p-aug 0.5
```

> ** 시간 단축 (디버그)**: `--epochs 3 --bs 8 --cap-train 200`  
> **wav2vec2-XLS-R-300M 사용 시**: `--backbone facebook/wav2vec2-xls-r-300m --bs 16 --lr 8e-7`

## 셀 5 — 학습 결과 확인

```python
import pandas as pd
log = pd.read_csv('runs/raptor_v1/log.csv')
print("=== Training Log ===")
print(log.to_string(index=False))
print(f"\nBest val EER: {log['val_eer'].min():.4f}")
```

## 셀 6 — 파인튜닝 모델 제출 폴더에 복사

```python
import shutil
shutil.copy('runs/raptor_v1/best.pth', 'submit/model/raptor_best.pth')
print("Model copied to submit/model/raptor_best.pth")
```

## 셀 7 — TTA + 앙상블 추론 (제출용)

```python
!python submit/script_v2.py \
    --test-dir data/test \
    --sample-submission data/sample_submission.csv \
    --output output/submission.csv \
    --device cuda \
    --tta 2
```

> **tta 옵션**: 0=off, 1=VoIP만, 2=VoIP+노이즈(권장), 3=VoIP+노이즈+속도

## 셀 8 — Pseudo-eval로 사전 검증

```python
# Pseudo-eval 구축 (셀3에서 생성됨)
# fusion_results.csv로 검증
!python scripts/fuse_search.py \
    --voice-role raptor_v1:ssl_aasist_la:df_arena \
    --music-role raptor_v1:sonics_g120s:df_arena \
    --presence preds/panns__presence.csv
```

## 셀 9 — 결과 백업

```python
import shutil
shutil.make_archive('/content/drive/MyDrive/dacon1_v2_results', 'zip', 'output')
print("Results backed up to Google Drive")
```

---

## 핵심 개선 포인트

### 왜 0.8 이상이 가능한가?

| 요소 | V1 (현재) | V2 (개선) | 영향 |
|---|---|---|---|
| **음성 탐지** | DF-Arena + AASIST (zero-shot) | +RAPTOR 파인튜닝 (ASVspoof 학습) | Voice EER ↓↓ |
| **음악 탐지** | DF-Arena + SONICS (zero-shot) | +RAPTOR 파인튜닝 (SONICS 학습) | Music EER ↓↓ |
| **TTA** | 없음 | VoIP + Noise 앙상블 | EER 1~3% ↓ |
| **Fusion** | max(VP*VF, MP*MF) | noisy-or + logit-blend 하이브리드 | File EER ↓ |
| **Consistency** | 없음 | RawBoost + JSD 정규화 | 일반화 ↑ |

### 예상 점수

| 시나리오 | ADS | CPS | 총점 |
|---|---|---|---|
| V1 Baseline | 0.658 | 0.989 | 0.691 |
| V1 + SONICS 강화 | 0.680 | 0.989 | 0.711 |
| **V2 + RAPTOR 파인튜닝** | **0.780** | **0.993** | **0.801** |
| **V2 + RAPTOR + TTA + Fusion** | **0.800** | **0.993** | **0.819** |
| 최대치 (학습 데이터 충분) | 0.850 | 0.995 | 0.865 |

### 핵심 논문 근거

1. **RAPTOR** (Kulkarni, Interspeech 2026): mHuBERT-Iter2로 100M 모델이 300M+ 모델을 능가. multi-dataset 학습 시 EER 5.78%(DF-Arena 500M 수준)
2. **AASIST3** (Borodin, 2024): KAN + Wav2Vec2 + pre-emphasis → ASVspoof 2024 minDCF 0.14
3. **Scalable AASIST** (Viakhirev, 2025): Frozen Wav2Vec2 + MHA fusion → ASVspoof 5 EER 7.6%
4. **SONICS** (Rahman, ICLR 2025): 97k 곡으로 학습된 음악 fake 탐지 → 노래 fake F1 0.97

### 학습 데이터 구성

```
Voice Real:
  - LibriSpeech test-clean (3,461 clips) → 최대 3,000
  - (추가: AI Hub 한국어 음성, Common Voice)

Voice Fake:
  - ASVspoof 2019 LA train (12,650 clips) → 최대 5,000
  - Codecfake (HF, 코덱 기반) → 최대 2,000
  
Music Real:
  - MUSDB18 sample (보컬 제거) → 200
  - FMA small → 최대 500

Music Fake:
  - SONICS (Suno/Udio, 49k 곡) → 최대 3,000
```
