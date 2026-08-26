#!/usr/bin/env python3
"""통합 코퍼스 파인튜닝: wav2vec2-XLS-R-300M + 통계풀링 헤드 (fake 확률 회귀).

설계
  - front-end: facebook/wav2vec2-xls-r-300m (feature_encoder conv 동결, 나머지 학습)
  - head: GAP + attention-stats-pooling(2파라미터) → MLP → spoof logit
  - 입력: 16kHz, 무작위 64600 크롭
  - 증강: noise / tel bandpass / µlaw / gain / lowpass / speed (numpy·scipy만)
  - loss: BCEWithLogits(label smoothing)
  - 매 에폭 val EER 채점(dacon_metric.compute_eer) 후 best 저장

manifest 형식(csv): filepath,label   (label: 0=real, 1=fake)

사용법 (Colab T4):
  python scripts/train_ssl_aasist.py --train manifest_train.csv --val manifest_val.csv \
      --out runs/exp1 --epochs 6 --bs 12 --lr 8e-6
"""
import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEG = 64600


# ------------------------------------------------------------------ augment --
def rand_gain(x, rng):
    return x * float(rng.uniform(0.3, 1.0))


def add_noise(x, rng):
    snr = float(rng.uniform(8, 25))
    ps = np.mean(x**2) + 1e-10
    pn = ps / (10 ** (snr / 10))
    return x + rng.normal(0, np.sqrt(pn), len(x))


def tel_bandpass(x):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [300, 3400], btype="bandpass", fs=16000, output="sos")
    return sosfiltfilt(sos, x)


def lowpass_hp(x):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, 4000, btype="lowpass", fs=16000, output="sos")
    return sosfiltfilt(sos, x)


def mulaw_q(x):
    mu = 255.0
    enc = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round(enc * 127) / 127
    return (np.sign(q) * (np.power(1 + mu, np.abs(q)) - 1) / mu).astype(np.float32)


def speed_perturb(x, rng):
    import librosa

    rate = float(rng.uniform(0.9, 1.1))
    y = librosa.resample(x, orig_sr=16000, target_sr=int(16000 / rate))
    if len(y) < SEG:
        y = np.tile(y, int(math.ceil(SEG / len(y))))
    s = int(rng.integers(0, len(y) - SEG + 1))
    return y[s : s + SEG]


AUGS = [add_noise, tel_bandpass, mulaw_q, lowpass_hp]


# ------------------------------------------------------------------- data ---
class Manifest(Dataset):
    def __init__(self, rows, train=True, p_aug=0.5):
        self.rows = rows
        self.train = train
        self.p_aug = p_aug
        self.rng = random.Random(0)

    def __len__(self):
        return len(self.rows)

    def _load(self, path):
        import librosa

        x, _ = librosa.load(path, sr=16000, mono=True)
        return x.astype(np.float32)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        x = self._load(path)
        rng = np.random.default_rng(self.rng.randrange(1 << 31))
        if self.train and rng.random() < 0.18:
            x = speed_perturb(x, rng)
        n = len(x)
        if n >= SEG:
            s = int(rng.integers(0, n - SEG + 1)) if self.train else (n - SEG) // 2
            x = x[s : s + SEG]
        else:
            reps = int(np.ceil(SEG / max(1, n)))
            x = np.tile(x, reps)[:SEG]
        if self.train and rng.random() < self.p_aug:
            f = AUGS[int(rng.integers(0, len(AUGS)))]
            try:
                x = f(x, rng) if f in (add_noise, rand_gain) else f(x)
            except Exception:
                pass
            x = x.astype(np.float32)
        peak = float(np.max(np.abs(x))) + 1e-9
        if peak > 1.0:
            x = x * (1.0 / peak)
        return torch.from_numpy(x), torch.tensor([float(label)])


def read_manifest(path, limit_per_class=0, seed=13):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["filepath"], int(r["label"])))
    if limit_per_class:
        rng = random.Random(seed)
        by_cls = {0: [], 1: []}
        for r in rows:
            by_cls[r[1]].append(r)
        rows = []
        for c in (0, 1):
            rng.shuffle(by_cls[c])
            rows += by_cls[c][:limit_per_class]
    return rows


# ------------------------------------------------------------------ model ---
class AttnStatsPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x):  # x: [B, T, C]
        w = torch.softmax(self.att(x), dim=1)
        mean = (w * x).sum(dim=1)
        var = (w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        return torch.cat([mean, var.sqrt()], dim=-1)


class Net(nn.Module):
    def __init__(self, ssl_name="facebook/wav2vec2-xls-r-300m", dropout=0.1):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.ssl = Wav2Vec2Model.from_pretrained(ssl_name)
        self.ssl.feature_extractor._freeze_parameters()
        dim = self.ssl.config.hidden_size  # 1024
        self.pool = AttnStatsPool(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, wav):  # [B, T]
        h = self.ssl(wav).last_hidden_state  # [B, T', C]
        return self.head(self.pool(h)).squeeze(-1)


# ------------------------------------------------------------------- train --
def evaluate_eer(model, loader, device):
    from sklearn.metrics import roc_curve

    ys, ss = [], []
    model.eval()
    with torch.inference_mode():
        for wav, y in loader:
            logits = model(wav.to(device))
            ss.extend(torch.sigmoid(logits).cpu().tolist())
            ys.extend(y.squeeze(-1).tolist())
    ys = np.asarray(ys, dtype=np.int8)
    ss = np.asarray(ss)
    if len(np.unique(ys)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(ys, ss, pos_label=1)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return (fpr[i] + fnr[i]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/exp1"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--p-aug", type=float, default=0.5)
    ap.add_argument("--cap-train", type=int, default=0, help="클래스당 최대 샘플(디버그)")
    ap.add_argument("--ssl", default="facebook/wav2vec2-xls-r-300m")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)

    tr_rows = read_manifest(args.train, args.cap_train)
    va_rows = read_manifest(args.val)
    print(f"train={len(tr_rows)} val={len(va_rows)}")

    dl_tr = DataLoader(
        Manifest(tr_rows, True, args.p_aug), batch_size=args.bs, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    dl_va = DataLoader(
        Manifest(va_rows, False), batch_size=args.bs, num_workers=2, pin_memory=True,
    )

    model = Net(args.ssl).to(device)
    backbone_params = list(model.ssl.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("ssl.")]
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    bce = nn.BCEWithLogitsLoss()

    best = 1e9
    log = open(args.out / "log.csv", "a", newline="")
    lg = csv.writer(log)
    lg.writerow(["epoch", "loss", "val_eer"])
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for wav, y in dl_tr:
            wav, y = wav.to(device), y.to(device).squeeze(-1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device == "cuda"):
                logits = model(wav)
                loss = bce(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
        sched.step()
        eer = evaluate_eer(model, dl_va, device)
        lg.writerow([ep, round(tot / max(nb, 1), 4), round(eer, 6)])
        log.flush()
        print(f"ep{ep} loss={tot/max(nb,1):.4f} valEER={eer:.4f}")
        ck = {"model": model.state_dict(), "ssl_name": args.ssl}
        torch.save(ck, args.out / ("best.pth" if eer < best else "last.pth"))
        if eer < best:
            best = eer
    log.close()
    print(f"best valEER={best:.4f}")


if __name__ == "__main__":
    main()
