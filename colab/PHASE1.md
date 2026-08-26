# Colab Phase 1 실행 가이드 (zero-shot 모델 평가 + fusion 탐색)

준비물: 이 프로젝트 폴더(`dacon1`)을 Google Drive 루트에 업로드.
(모델 4.6GB 포함이라 시간 걸림. 대안: 아래 셀0에서 DF-Arena를 HF에서 직접 받도록 함 — Drive엔 `scripts/`, `zoo/`, `dacon_metric.py`만 있으면 됨)

아래 셀을 순서대로 Colab(GPU T4)에 붙여넣어 실행.

---

## 셀 1 — 환경 세팅 (런타임 연결 후 최초 1회)

```python
!pip -q install librosa soundfile edge-tts transformers accelerate demucs panns-inference gdown huggingface_hub
!apt -qq install -y ffmpeg > /dev/null
```

## 셀 2 — 프로젝트 코드 확보 (Drive 마운트 or zip 업로드)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content
import shutil
# Drive 루트의 dacon1 폴더를 로컬로 복사 (모델 제외, 코드만)
shutil.copytree('/content/drive/MyDrive/dacon1', '/content/dacon1',
                ignore=shutil.ignore_patterns('open1', '_cache', 'pseudo_eval', 'preds'))
%cd /content/dacon1
```

> Drive에 올리기 부담이면: 로컬에서 dacon1 폴더 중 scripts/, zoo/, 전략 문서만 zip으로 묶어
> Colab 파일 업로드 후 `unzip` 해도 됨.

## 셀 3 — pseudo-eval 구축 (~30~50분, FMA 다운로드가 병목)

```python
!python scripts/build_pseudo_eval.py --out pseudo_eval --seed 42 \
    --n-voice-real 180 --n-voice-fake-edge 50 --n-voice-fake-mms 30 \
    --n-music-real-fma 120 --n-song-sonics 80 --n-mix 140 --fx-prob 0.45
```

MusicGen 가짜 음악도 넣고 싶으면(선택, +20분):
```python
!python scripts/build_pseudo_eval.py --out pseudo_eval --seed 42 \
    --n-music-fake-musicgen 40   # 기존 결과 재사용하고 musicgen만 추가
```
(주의: 위 명령은 labels를 재생성하지 않음 — musicgen은 refs_music에만 추가됨.
파일 트랙에 반영하려면 빌더 재실행 필요. 우선 refs 트랙만으로 충분.)

## 셀 4 — baseline(PANNs presence + DF-Arena) 추론

```python
# DF-Arena는 HF에서 직접 다운로드 (Drive에 없어도 OK)
!python scripts/run_zoo.py --models panns df_arena --device cuda
```
소요: panns ~5분, df_arena(1B) ~30-60분 (pseudo-eval 크기에 비례).
시간 부족 시 먼저 `--tracks voice music`으로 컴포넌트 트랙만.

## 셀 5 — 경쟁 모델 추가

```python
# SSL AASIST (ONNX 1.26GB 자동 다운로드, onnxruntime-gpu 사용)
!pip -q install onnxruntime-gpu
!python scripts/run_zoo.py --models ssl_aasist_la

# XLSR-Mamba (mamba-ssm 필요 — 실패하면 skip 가능)
!pip -q install mamba-ssm causal-conv1d || echo "mamba 설치 실패 - skip"
!git clone https://github.com/swagshaw/XLSR-Mamba xlsr_mamba
!python scripts/run_zoo.py --models xlsr_mamba_la xlsr_mamba_df

# SONICS (음악/노래 fake 특화) — music 트랙이 핵심
!pip -q install git+https://github.com/awsaf49/sonics.git
!python scripts/run_zoo.py --tracks voice music --models sonics_a5s sonics_b5s sonics_g120s
```

각 모델별 컴포넌트 성능 즉시 확인:
```python
import sys; sys.path.insert(0,'scripts')
from dacon_metric import compute_eer
import pandas as pd
for m in ['ssl_aasist_la','xlsr_mamba_la','xlsr_mamba_df','df_arena']:
    try:
        v = pd.read_csv(f'preds/{m}__voice.csv'); lv = pd.read_csv('pseudo_eval/refs_voice.csv')
        mu = pd.read_csv(f'preds/{m}__music.csv'); lm = pd.read_csv('pseudo_eval/refs_music.csv')
        j = lv.merge(v,on='ID'); jm = lm.merge(mu,on='ID')
        print(f"{m:18s} VoiceEER={compute_eer(j.FAKE,j.PROB):.3f} MusicEER={compute_eer(jm.FAKE,jm.PROB):.3f}")
    except FileNotFoundError:
        pass
```

## 셀 6 — fusion 탐색 + 리더보드 추정

```python
!python scripts/fuse_search.py \
    --voice-role ssl_aasist_la:df_arena \
    --music-role sonics_g120s:sonics_b5s:df_arena \
    --presence preds/panns__presence.csv
```
→ `preds/fusion_results.csv` 상위 조합이 곧 우리 예상 리더보드 점수.
(presence oracle로 돌리면 이론 상한 확인 가능)

## 셀 7 — 결과 백업 (Drive)

```python
shutil.make_archive('/content/drive/MyDrive/dacon1_results', 'zip', '/content/dacon1/preds')
shutil.copy('/content/dacon1/pseudo_eval/labels.csv','/content/drive/MyDrive/pseudo_labels.csv')
```

---

### 체크포인트 의미
- 셀4 결과가 Baseline(ADS≈0.658)과 비슷하게 나오면 pseudo-eval이 실제 분포를 잘 모사하는 것 → 신뢰 확보
- 셀5에서 MusicEER을 크게 낮추는 모델이 보이면 그것이 1월 위 Devfiance(+0.011)를 넘을 카드
- fusion 상위식이 F1(baseline 식)보다 좋으면 그대로 제출 스크립트에 반영
