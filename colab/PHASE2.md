# Colab Phase 2 — 파인튜닝 (PHASE1 결과 확인 후 실행)

전제: PHASE1 완료. fusion_results.csv에서 어떤 역할이 약한지 확인했음.
목표: pseudo-eval + 외부 코퍼스로 우리 자체 탐지기를 만들어 voice/music role에 투입.

## 셀 1 — 준비

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/dacon1   # PHASE1에서 쓰던 작업 폴더
!pip -q install librosa soundfile transformers accelerate onnxruntime-gpu
```

## 셀 2 — manifest 생성 (pseudo-eval refs 기반)

```python
import pandas as pd, csv
def make_manifest(ref_csv, wav_dir, out_csv, fake_extra=None):
    df = pd.read_csv(ref_csv)
    rows = [[str(wav_dir / f"{r.ID}.wav"), int(r.FAKE)] for r in df.itertuples()]
    if fake_extra:
        rows += fake_extra
    with open(out_csv,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['filepath','label']); w.writerows(rows)

make_manifest('pseudo_eval/refs_voice.csv','pseudo_eval/refs/voice','manifest_voice.csv')
make_manifest('pseudo_eval/refs_music.csv','pseudo_eval/refs/music','manifest_music.csv')

# 통합(음성+음악 한 모델)이면:
with open('manifest_all.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['filepath','label'])
    for p in ['manifest_voice.csv','manifest_music.csv']:
        for i,r in enumerate(open(p).read().splitlines()[1:]):
            f.write(r+'\n')

# val/test 분리 (source별 stratified 권장 — 일단 랜덤 15%)
from sklearn.model_selection import train_test_split
rows = list(csv.DictReader(open('manifest_all.csv')))
tr, va = train_test_split(rows, test_size=0.15, random_state=42, stratify=[r['label'] for r in rows])
for name, data in [('manifest_train.csv',tr),('manifest_val.csv',va)]:
    with open(name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['filepath','label']); w.writeheader(); w.writerows(data)
```

> 주의: SONICS 곡은 refs/music에 없고 test 트랙(song)에만 있음.
> 음악 fake 학습량을 늘리려면 PHASE1의 `_src_sonics` 폴더를 manifest에 추가:
```python
import csv, glob, random
extra = [(p, 1) for p in glob.glob('pseudo_eval/_src_sonics/*')]
random.seed(0); random.shuffle(extra)
extra = extra[:200]  # 학습 분량 조절
rows = list(csv.DictReader(open('manifest_train.csv')))
rows = [{'filepath':p,'label':l} for p,l in rows]
va_rows = list(csv.DictReader(open('manifest_val.csv')))
n_val = max(10,len(extra)//8)
rows += [{'filepath':p,'label':l} for p,l in extra[n_val:]]
va_rows += [{'filepath':p,'label':l} for p,l in extra[:n_val]]
with open('manifest_train.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['filepath','label']); w.writeheader(); w.writerows(rows)
with open('manifest_val.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['filepath','label']); w.writeheader(); w.writerows(va_rows)
```

## 셀 3 — 학습 (T4 기준 3~5시간 예상, 세션 끊김 대비 epoch 저장)

```python
!python scripts/train_custom.py --train manifest_train.csv --val manifest_val.csv \
    --out runs/exp1 --epochs 4 --bs 12 --lr 6e-6 --lr-head 3e-4 --cap-train 0
```

디버그용 소규모 선행 실행(30분): `--epochs 1 --bs 8 --cap-train 150`

## 셀 4 — 학습 모델 ONNX 변환 (제출 패키징용)

```python
import torch, sys
sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
from scripts.train_custom import Net
m = Net('facebook/wav2vec2-xls-r-300m').cuda()
ck = torch.load('runs/exp1/best.pth'); m.load_state_dict(ck['model']); m.eval()
dummy = torch.randn(1, 64600).cuda()
torch.onnx.export(m, dummy, 'ours.onnx',
                  input_names=['wav'], output_names=['spoof_logit'],
                  dynamic_axes={'wav':{0:'batch'}}, opset_version=17)
print('saved ours.onnx')
```

## 셀 5 — zoo에 편입 & 재채점

`zoo/ssl_aasist.py`와 동일 구조의 `zoo/ours.py`를 만들거나, 간단히
onnx 경로만 바꿔 SSLAasist(ckpt='ours.onnx') 인스턴스를 REGISTRY에 추가:

```python
# zoo/__init__.py 에 임시 추가 후:
!python scripts/run_zoo.py --models ours --device cuda
!python scripts/fuse_search.py --voice-role ours:ssl_aasist_la:df_arena \
    --music-role ours:sonics_g120s:df_arena --presence preds/panns__presence.csv
```

## 확장 데이터 소스 (학습 성능 부족 시 추가 다운로드)
- ASVspoof 2019 LA (Edinburgh datashare DS_10283_3336) — 음성 fake 대량
- ASVspoof 5 (HF jungjee/asvspoof5) — 최신 공격 포함
- MLAAD v6 — 다국어 TTS 공격 (Korean 포함)
- CtrSVDD (Zenodo 10467648/10742049) — 노래 보컬 fake
- In-the-Wild (intwild) — 실환경 fake
- AI Hub 한국어 음성 — 한국어 real 다양화
추가 규칙: real:fake 비율 1:1~1:2 유지, source 단위로 train/val 분리(누수 방지).
