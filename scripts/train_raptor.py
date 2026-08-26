#!/usr/bin/env python3
"""RAPTOR 스타일 파인튜닝 — 0.8+ 달성을 위한 핵심 학습 코드.

설계 (Kulkarni et al., Interspeech 2026 기반):
  - Backbone: mHuBERT-Iter2 (95M, utter-project/mHuBERT-147)
           또는 wav2vec2-XLS-R-300M (facebook/wav2vec2-xls-r-300m)
  - RAPTOR Fusion: 인접 SSL 레이어 pairwise softmax gated fusion
  - Attention pooling → binary classifier
  - Consistency regularization: RawBoost aug + JSD (λ=0.25)
  - 입력: 16kHz, 4초 (64000 샘플)
  - 증강: RawBoost (noise, reverb, codec, speed)
  - 손실: BCE with label smoothing
  - Validation: EER 기반 best 모델 저장

사용법:
  python scripts/train_raptor.py \\
      --train train_data/manifest_train.csv \\
      --val train_data/manifest_val.csv \\
      --backbone utter-project/mHuBERT-147 \\
      --out runs/raptor_v1 \\
      --epochs 30 --bs 24 --lr 1e-6

  # wav2vec2-XLS-R-300M 사용 시:
  python scripts/train_raptor.py \\
      --backbone facebook/wav2vec2-xls-r-300m \\
      --epochs 30 --bs 16 --lr 8e-7
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEG = 64000  # 4초 @16kHz


# ============================================================
# Data Augmentation (RawBoost 스타일)
# ============================================================

def additive_noise(x, snr_range=(5, 25), rng=None):
    """Additive colored noise."""
    snr = float(rng.uniform(*snr_range))
    ps = np.mean(x**2) + 1e-10
    pn = ps / (10 ** (snr / 10))
    return x + rng.normal(0, np.sqrt(pn), len(x)).astype(np.float32)


def impulse_response(x, rng=None):
    """Simulated room impulse response (간단한 지연+감쇠)."""
    delay = int(rng.integers(1, 50))
    decay = float(rng.uniform(0.1, 0.5))
    ir = np.zeros(len(x) + delay, dtype=np.float32)
    ir[delay:delay+len(x)] = x
    ir[delay:] += decay * x[:len(ir)-delay]
    return ir[:len(x)]


def codec_simulation(x, rng=None):
    """MP3/Opus 코덱 시뮬레이션 (lossy compression artifact)."""
    try:
        import tempfile, subprocess, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, x, 16000)
            codec = str(rng.choice(["mp3", "opus"]))
            if codec == "mp3":
                kbps = int(rng.choice([32, 48, 64, 96]))
                out = tmp.name.replace('.wav', f'_{codec}.mp3')
                subprocess.run([
                    'ffmpeg', '-y', '-i', tmp.name,
                    '-codec:a', 'libmp3lame', '-b:a', f'{kbps}k',
                    '-ar', '16000', out
                ], capture_output=True, timeout=5)
            else:
                out = tmp.name.replace('.wav', f'_{codec}.ogg')
                subprocess.run([
                    'ffmpeg', '-y', '-i', tmp.name,
                    '-codec:a', 'libopus', '-b:a', '32k',
                    '-ar', '16000', out
                ], capture_output=True, timeout=5)
            augmented, _ = sf.read(out)
            Path(tmp.name).unlink(missing_ok=True)
            Path(out).unlink(missing_ok=True)
        return augmented.astype(np.float32)[:len(x)]
    except Exception:
        return x


def speed_perturb(x, rng=None):
    """Speed perturbation ±10%."""
    import librosa
    rate = float(rng.uniform(0.9, 1.1))
    augmented = librosa.resample(x, orig_sr=16000, target_sr=int(16000 / rate))
    if len(augmented) < len(x):
        augmented = np.pad(augmented, (0, len(x) - len(augmented)))
    return augmented[:len(x)].astype(np.float32)


def pre_emphasis(x, coeff=0.97):
    """Pre-emphasis filter (AASIST3 style) — 고주파 강조."""
    return np.append(x[0], x[1:] - coeff * x[:-1]).astype(np.float32)


def rand_gain(x, rng=None):
    """Random gain ±6dB."""
    gain = float(rng.uniform(0.25, 2.0))
    return (x * gain).astype(np.float32)


RAWBOOST_AUGS = [
    additive_noise,
    impulse_response,
    codec_simulation,
    speed_perturb,
    rand_gain,
]


def apply_rawboost(x, rng, p=0.5):
    """RawBoost 스타일 온라인 증강 (확률 p로 1~2개 적용)."""
    if rng.random() > p:
        return x
    n_augs = rng.integers(1, 3)
    for _ in range(n_augs):
        aug_fn = rng.choice(RAWBOOST_AUGS)
        try:
            x = aug_fn(x, rng=rng)
        except Exception:
            pass
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class FakeDetectDataset(Dataset):
    def __init__(self, manifest_path, train=True, p_aug=0.5, seg=SEG):
        self.rows = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append((r["filepath"], int(r["label"])))
        self.train = train
        self.p_aug = p_aug
        self.seg = seg
        self.rng = random.Random(42 if not train else 0)

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

        # Random crop or pad
        n = len(x)
        if n >= self.seg:
            if self.train:
                s = int(rng.integers(0, n - self.seg + 1))
            else:
                s = (n - self.seg) // 2
            x = x[s:s + self.seg]
        else:
            reps = int(np.ceil(self.seg / max(1, n)))
            x = np.tile(x, reps)[:self.seg]

        # Pre-emphasis (고주파 강조, AASIST3)
        x = pre_emphasis(x)

        # RawBoost augmentation
        if self.train and rng.random() < self.p_aug:
            x = apply_rawboost(x, rng, p=1.0)

        # Peak normalize
        peak = float(np.max(np.abs(x))) + 1e-9
        if peak > 1.0:
            x = x * (1.0 / peak)

        return torch.from_numpy(x), torch.tensor([float(label)])


# ============================================================
# RAPTOR Model
# ============================================================

class PairwiseGate(nn.Module):
    """인접 SSL 레이어의 pairwise softmax gated fusion."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, h1, h2):
        # h1, h2: [B, T, D]
        g = self.gate(torch.cat([h1, h2], dim=-1))  # [B, T, D]
        return g * h1 + (1 - g) * h2


class HierarchicalGate(nn.Module):
    """계층적 pairwise fusion: 레이어 쌍 → 쌍의 쌍 → ... → 단일 벡터."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, pairs):
        # pairs: list of [B, T, D] — 인접 쌍들
        while len(pairs) > 1:
            new_pairs = []
            for i in range(0, len(pairs) - 1, 2):
                h1, h2 = pairs[i], pairs[i + 1]
                g = self.gate(torch.cat([h1, h2], dim=-1))
                new_pairs.append(g * h1 + (1 - g) * h2)
            if len(pairs) % 2 == 1:
                new_pairs.append(pairs[-1])
            pairs = new_pairs
        return pairs[0]


class AttentionPooling(nn.Module):
    """Attention pooling: [B, T, D] → [B, D]."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]
        return (w * x).sum(dim=1)  # [B, D]


class RAPTOR(nn.Module):
    """RAPTOR: Representation Aware Pairwise-gated Transformer.
    
    SSL backbone → pairwise gated layer fusion → attention pooling → classifier.
    """
    def __init__(self, backbone_name="utter-project/mHuBERT-147", dropout=0.1):
        super().__init__()
        
        # Load SSL backbone
        self.backbone_name = backbone_name
        self._load_backbone(backbone_name)
        
        dim = self.backbone_dim  # 768 for mHuBERT, 1024 for XLS-R
        
        # RAPTOR: pairwise gated fusion
        n_layers = self.n_layers
        n_pairs = n_layers // 2
        
        self.pair_gates = nn.ModuleList([
            PairwiseGate(dim) for _ in range(n_pairs)
        ])
        self.hier_gate = HierarchicalGate(dim)
        
        # Attention pooling
        self.pool = AttentionPooling(dim)
        
        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _load_backbone(self, name):
        if "mHuBERT" in name or "hubert" in name.lower():
            from transformers import HubertModel, AutoConfig
            config = AutoConfig.from_pretrained(name)
            self.backbone = HubertModel.from_pretrained(name)
            self.backbone_dim = config.hidden_size
            self.n_layers = config.num_hidden_layers
        elif "xls-r" in name.lower() or "xlsr" in name.lower():
            from transformers import Wav2Vec2Model
            self.backbone = Wav2Vec2Model.from_pretrained(name)
            self.backbone_dim = self.backbone.config.hidden_size
            self.n_layers = self.backbone.config.num_hidden_layers
        else:
            raise ValueError(f"Unknown backbone: {name}")

    def forward(self, wav):
        """wav: [B, T] raw waveform → logit: [B]."""
        # SSL forward — 모든 레이어 출력 반환
        outputs = self.backbone(wav, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [B, T', D]
        
        # Pairwise gated fusion
        pairs = []
        for i in range(0, len(hidden_states) - 1, 2):
            h1 = hidden_states[i]
            h2 = hidden_states[i + 1]
            fused = self.pair_gates[i // 2](h1, h2)
            pairs.append(fused)
        
        # Handle odd number of layers
        if len(hidden_states) % 2 == 1:
            pairs.append(hidden_states[-1])
        
        # Hierarchical fusion
        fused = self.hier_gate(pairs)  # [B, T', D]
        
        # Attention pooling
        pooled = self.pool(fused)  # [B, D]
        
        # Classification
        return self.head(pooled).squeeze(-1)  # [B]


# ============================================================
# Consistency Regularization (RAPTOR)
# ============================================================

def consistency_loss(model, wav_clean, wav_aug, lambda_cons=0.25):
    """RawBoost 증강에 대한 게이팅 안정화.
    
    JSD between gate distributions of clean vs augmented views.
    """
    with torch.no_grad():
        # Clean forward (게이팅 분포만 필요)
        out_clean = model.backbone(wav_clean, output_hidden_states=True)
        hs_clean = out_clean.hidden_states
    
    out_aug = model.backbone(wav_aug, output_hidden_states=True)
    hs_aug = out_aug.hidden_states
    
    # Pairwise gate activations 비교
    jsd_total = 0.0
    n_pairs = min(len(hs_clean), len(hs_aug)) // 2
    
    for i in range(n_pairs):
        h1_c, h2_c = hs_clean[2*i], hs_clean[2*i+1]
        h1_a, h2_a = hs_aug[2*i], hs_aug[2*i+1]
        
        # Gate output: [B, T, D] → mean over T, D for simplicity
        g_c = model.pair_gates[i].gate(torch.cat([h1_c, h2_c], dim=-1)).mean(dim=(1, 2))
        g_a = model.pair_gates[i].gate(torch.cat([h1_a, h2_a], dim=-1)).mean(dim=(1, 2))
        
        # Binary JSD: gate outputs are [0,1] after sigmoid
        p = torch.stack([g_c, 1 - g_c], dim=-1)  # [B, 2]
        q = torch.stack([g_a, 1 - g_a], dim=-1)
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(m.log(), p, reduction="batchmean")
        kl_qm = F.kl_div(m.log(), q, reduction="batchmean")
        jsd_total += 0.5 * (kl_pm + kl_qm)
    
    return lambda_cons * jsd_total / max(n_pairs, 1)


# ============================================================
# Evaluation
# ============================================================

def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 1.0
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


@torch.no_grad()
def evaluate_eer(model, loader, device):
    model.eval()
    ys, ss = [], []
    for wav, y in loader:
        logits = model(wav.to(device))
        ss.extend(torch.sigmoid(logits).cpu().tolist())
        ys.extend(y.squeeze(-1).tolist())
    return compute_eer(ys, ss)


# ============================================================
# Training
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="RAPTOR fine-tuning for 0.8+")
    ap.add_argument("--train", required=True, help="manifest_train.csv")
    ap.add_argument("--val", required=True, help="manifest_val.csv")
    ap.add_argument("--backbone", default="utter-project/mHuBERT-147",
                    help="SSL backbone name")
    ap.add_argument("--out", type=Path, default=Path("runs/raptor_v1"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-6, help="backbone LR")
    ap.add_argument("--lr-head", type=float, default=3e-4, help="head LR")
    ap.add_argument("--consistency-w", type=float, default=0.25,
                    help="consistency regularization weight")
    ap.add_argument("--p-aug", type=float, default=0.5, help="augmentation probability")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--cap-train", type=int, default=0,
                    help="max samples per class (0=unlimited)")
    ap.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)

    # Data
    tr_ds = FakeDetectDataset(args.train, train=True, p_aug=args.p_aug)
    va_ds = FakeDetectDataset(args.val, train=False)
    
    if args.cap_train > 0:
        # Cap per class
        from collections import defaultdict
        by_cls = defaultdict(list)
        for i, (_, l) in enumerate(tr_ds.rows):
            by_cls[l].append(i)
        capped = []
        for c, idxs in by_cls.items():
            capped.extend(idxs[:args.cap_train])
        tr_ds.rows = [tr_ds.rows[i] for i in capped]

    print(f"Train: {len(tr_ds)} samples, Val: {len(va_ds)} samples")
    
    dl_tr = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)
    dl_va = DataLoader(va_ds, batch_size=args.bs, num_workers=2, pin_memory=True)

    # Model
    print(f"Loading backbone: {args.backbone}")
    model = RAPTOR(args.backbone, dropout=0.1).to(device)
    
    # Optimizer: different LR for backbone vs head
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=1e-4)
    
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    
    # Loss with label smoothing
    bce = nn.BCEWithLogitsLoss()
    
    best_eer = 1e9
    
    # Resume
    start_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best_eer = ck.get("best_eer", 1e9)
        print(f"Resumed from epoch {start_epoch}, best EER={best_eer:.4f}")

    log_path = args.out / "log.csv"
    with open(log_path, "a", newline="") as log_file:
        lg = csv.writer(log_file)
        if start_epoch == 1:
            lg.writerow(["epoch", "loss", "val_eer", "time"])
        
        for ep in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            model.train()
            tot, nb = 0.0, 0
            
            for wav, y in dl_tr:
                wav, y = wav.to(device), y.to(device).squeeze(-1)
                
                # RawBoost augmentation view for consistency
                rng_cons = np.random.default_rng(int(time.time() * 1000) % (2**31))
                wav_aug = wav.clone()
                for i in range(len(wav_aug)):
                    wav_aug[i] = torch.from_numpy(
                        apply_rawboost(wav_aug[i].cpu().numpy(), rng_cons, p=1.0)
                    ).to(device)
                
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=(device == "cuda")):
                    logits = model(wav)
                    
                    # Classification loss with label smoothing
                    targets = y * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
                    loss_cls = bce(logits.float(), targets)
                    
                    # Consistency loss
                    loss_cons = consistency_loss(model, wav, wav_aug, args.consistency_w)
                    
                    loss = loss_cls + loss_cons
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                
                tot += loss.item()
                nb += 1
            
            sched.step()
            elapsed = time.time() - t0
            
            # Validation
            val_eer = evaluate_eer(model, dl_va, device)
            
            lg.writerow([ep, round(tot / max(nb, 1), 4), round(val_eer, 6), round(elapsed, 1)])
            log_file.flush()
            
            print(f"Epoch {ep}/{args.epochs}  loss={tot/max(nb,1):.4f}  "
                  f"valEER={val_eer:.4f}  time={elapsed:.0f}s")
            
            # Save best
            ck = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": ep,
                "backbone": args.backbone,
                "best_eer": best_eer,
            }
            if val_eer < best_eer:
                best_eer = val_eer
                ck["best_eer"] = best_eer
                torch.save(ck, args.out / "best.pth")
                print(f"  → New best EER: {best_eer:.4f}")
            
            # Always save last
            torch.save(ck, args.out / "last.pth")
    
    print(f"\nDone. Best val EER = {best_eer:.4f}")
    print(f"Model saved to: {args.out / 'best.pth'}")


if __name__ == "__main__":
    main()
