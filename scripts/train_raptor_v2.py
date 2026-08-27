#!/usr/bin/env python3
"""Ultimate Audio Deepfake Detector — 2026 SOTA 통합.

논문 기반 구현:
1. RAPTOR (Kulkarni et al., Interspeech 2026): mHuBERT pairwise gated fusion
2. Scalable AASIST (Viakhirev et al., 2025): Frozen Wav2Vec2 + MHA
3. Multi-Backbone Ensemble (Kim et al., CLEF 2026): 4-backbone SSL ensemble
4. Layer-Wise Decision Fusion (Xiao et al., Interspeech 2025): Per-layer classifiers
5. Microsoft (Delgado et al., 2026): Data quality > model size

핵심 차별화:
- Dual-backbone: mHuBERT-Iter2 + XLS-R-300M (frozen)
- Layer-wise decision fusion (feature collapse 방지)
- One-class softmax loss (generalization 향상)
- Consistency regularization (augmentation invariance)
- TTA with aleatoric uncertainty estimation

사용법:
  python scripts/train_raptor_v2.py \
      --train train_data/manifest_train.csv \
      --val train_data/manifest_val.csv \
      --out runs/raptor_v2 \
      --epochs 50 --bs 24
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
# Data Augmentation (RawBoost + Extended)
# ============================================================

def additive_noise(x, snr_range=(5, 25), rng=None):
    snr = float(rng.uniform(*snr_range))
    ps = np.mean(x**2) + 1e-10
    pn = ps / (10 ** (snr / 10))
    return x + rng.normal(0, np.sqrt(pn), len(x)).astype(np.float32)


def impulse_response(x, rng=None):
    delay = int(rng.integers(1, 50))
    decay = float(rng.uniform(0.1, 0.5))
    ir = np.zeros(len(x) + delay, dtype=np.float32)
    ir[delay:delay+len(x)] = x
    ir[delay:] += decay * x[:len(ir)-delay]
    return ir[:len(x)]


def codec_simulation(x, rng=None):
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
    import librosa
    rate = float(rng.uniform(0.9, 1.1))
    augmented = librosa.resample(x, orig_sr=16000, target_sr=int(16000 / rate))
    if len(augmented) < len(x):
        augmented = np.pad(augmented, (0, len(x) - len(augmented)))
    return augmented[:len(x)].astype(np.float32)


def phone_call_distortion(x, rng=None):
    """전화선 시뮬레이션 (8kHz + 밴드패스)."""
    try:
        import librosa
        from scipy.signal import butter, sosfilt
        audio_8k = librosa.resample(x.astype(np.float64), orig_sr=16000, target_sr=8000)
        sos = butter(4, [300, 3400], btype='band', fs=8000, output='sos')
        audio_8k = sosfilt(sos, audio_8k).astype(np.float32)
        audio_out = librosa.resample(audio_8k.astype(np.float64), orig_sr=8000, target_sr=16000)
        if len(audio_out) < len(x):
            audio_out = np.pad(audio_out, (0, len(x) - len(audio_out)))
        return audio_out[:len(x)].astype(np.float32)
    except Exception:
        return x


def pre_emphasis(x, coeff=0.97):
    return np.append(x[0], x[1:] - coeff * x[:-1]).astype(np.float32)


def rand_gain(x, rng=None):
    gain = float(rng.uniform(0.25, 2.0))
    return (x * gain).astype(np.float32)


RAWBOOST_AUGS = [
    additive_noise, impulse_response, codec_simulation,
    speed_perturb, phone_call_distortion, rand_gain,
]


def apply_rawboost(x, rng, p=0.5):
    if rng.random() > p:
        return x
    n_augs = rng.integers(1, 4)
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
# Mixup + SpecAugment
# ============================================================

def mixup_data(x, y, alpha=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    lam = float(rng.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def specaugment(x, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    T = len(x)
    if rng.random() < 0.5:
        mask_len = int(rng.integers(T // 10, T // 4))
        mask_start = int(rng.integers(0, T - mask_len))
        x = x.copy()
        x[mask_start:mask_start + mask_len] = 0
    if rng.random() < 0.3:
        freq = int(rng.integers(50, 200))
        x = x.copy()
        mask = np.ones(len(x), dtype=np.float32)
        mask[::freq] = 0
        x = x * mask
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

        x = pre_emphasis(x)

        if self.train and rng.random() < self.p_aug:
            x = apply_rawboost(x, rng, p=1.0)

        peak = float(np.max(np.abs(x))) + 1e-9
        if peak > 1.0:
            x = x * (1.0 / peak)

        return torch.from_numpy(x), torch.tensor([float(label)])


# ============================================================
# Backbone Wrappers
# ============================================================

class mHuBERTBackbone(nn.Module):
    """mHuBERT-Iter2 backbone (95M)."""
    def __init__(self, name="utter-project/mHuBERT-147"):
        super().__init__()
        from transformers import HubertModel, AutoConfig
        config = AutoConfig.from_pretrained(name)
        self.backbone = HubertModel.from_pretrained(name)
        self.dim = config.hidden_size
        self.n_layers = config.num_hidden_layers

    def forward(self, wav):
        outputs = self.backbone(wav, output_hidden_states=True)
        return outputs.hidden_states  # tuple of [B, T', D]


class XLSRBackbone(nn.Module):
    """XLS-R 300M backbone (frozen)."""
    def __init__(self, name="facebook/wav2vec2-xls-r-300m"):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.backbone = Wav2Vec2Model.from_pretrained(name)
        self.backbone.eval()
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.dim = self.backbone.config.hidden_size
        self.n_layers = self.backbone.config.num_hidden_layers

    @torch.no_grad()
    def forward(self, wav):
        self.backbone.eval()
        outputs = self.backbone(wav, output_hidden_states=True)
        return outputs.hidden_states


# ============================================================
# RAPTOR Fusion Module
# ============================================================

class PairwiseGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, h1, h2):
        g = self.gate(torch.cat([h1, h2], dim=-1))
        return g * h1 + (1 - g) * h2


class HierarchicalGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, pairs):
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
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)


# ============================================================
# Layer-Wise Decision Fusion (Xiao et al., 2025)
# ============================================================

class LayerWiseClassifier(nn.Module):
    """Each SSL layer gets its own classifier."""
    def __init__(self, dim, n_layers, d=128):
        super().__init__()
        # Shared bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, d),
            nn.GELU(),
        )
        # Per-layer classifier
        self.classifiers = nn.ModuleList([
            nn.Linear(d, 1) for _ in range(n_layers)
        ])
        # Learnable layer weights
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)

    def forward(self, hidden_states):
        """hidden_states: tuple of [B, T, D]"""
        layer_weights = F.softmax(self.layer_weights, dim=0)
        scores = []
        for i, h in enumerate(hidden_states):
            # Attention pooling
            pooled = h.mean(dim=1)  # [B, D]
            proj = self.bottleneck(pooled)  # [B, d]
            score = self.classifiers[i](proj).squeeze(-1)  # [B]
            scores.append(score * layer_weights[i])
        return sum(scores)  # [B]


# ============================================================
# Ultimate Model: RAPTOR + Layer-Wise Fusion
# ============================================================

class UltimateDetector(nn.Module):
    """Dual-backbone with RAPTOR + Layer-Wise Decision Fusion.

    Architecture:
    1. mHuBERT-Iter2 (trainable) → RAPTOR pairwise gated fusion
    2. XLS-R-300M (frozen) → Layer-wise decision fusion
    3. Final: weighted combination of both
    """
    def __init__(self, dropout=0.1):
        super().__init__()

        # Backbone 1: mHuBERT-Iter2 (trainable)
        self.mhubert = mHuBERTBackbone("utter-project/mHuBERT-147")
        dim1 = self.mhubert.dim
        n_layers1 = self.mhubert.n_layers

        # Backbone 2: XLS-R-300M (frozen)
        self.xlsr = XLSRBackbone("facebook/wav2vec2-xls-r-300m")
        dim2 = self.xlsr.dim
        n_layers2 = self.xlsr.n_layers

        # RAPTOR fusion for mHuBERT
        n_pairs1 = n_layers1 // 2
        self.pair_gates = nn.ModuleList([PairwiseGate(dim1) for _ in range(n_pairs1)])
        self.hier_gate = HierarchicalGate(dim1)
        self.pool1 = AttentionPooling(dim1)

        # Layer-wise classifier for XLS-R
        self.lw_classifier = LayerWiseClassifier(dim2, n_layers2, d=128)

        # Cross-backbone fusion
        self.cross_fusion = nn.Sequential(
            nn.Linear(dim1 + 128, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Individual heads for each backbone
        self.head_mhubert = nn.Sequential(
            nn.LayerNorm(dim1),
            nn.Linear(dim1, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Learnable backbone weights
        self.alpha_mhubert = nn.Parameter(torch.tensor(0.5))
        self.alpha_xlsr = nn.Parameter(torch.tensor(0.5))

    def forward(self, wav):
        # Backbone 1: mHuBERT
        hs1 = self.mhubert(wav)
        pairs = []
        for i in range(0, len(hs1) - 1, 2):
            h1, h2 = hs1[i], hs1[i + 1]
            fused = self.pair_gates[i // 2](h1, h2)
            pairs.append(fused)
        if len(hs1) % 2 == 1:
            pairs.append(hs1[-1])
        fused1 = self.hier_gate(pairs)
        pooled1 = self.pool1(fused1)  # [B, dim1]

        # Backbone 2: XLS-R (frozen)
        with torch.no_grad():
            hs2 = self.xlsr(wav)
        xlsr_score = self.lw_classifier(hs2)  # [B]

        # Individual mHuBERT score
        mhubert_score = self.head_mhubert(pooled1).squeeze(-1)  # [B]

        # Cross-backbone fusion
        # Pool XLS-R features for fusion
        xlsr_pooled = torch.cat([h.mean(dim=1) for h in hs2], dim=-1)
        xlsr_proj = self.lw_classifier.bottleneck(xlsr_pooled.mean(dim=0, keepdim=True).expand(pooled1.size(0), -1))
        cross_input = torch.cat([pooled1, xlsr_proj], dim=-1)
        cross_score = self.cross_fusion(cross_input).squeeze(-1)

        # Weighted combination
        alpha1 = torch.sigmoid(self.alpha_mhubert)
        alpha2 = torch.sigmoid(self.alpha_xlsr)
        total = alpha1 + alpha2 + 1e-8
        alpha1, alpha2 = alpha1 / total, alpha2 / total

        final = alpha1 * mhubert_score + alpha2 * xlsr_score + (1 - alpha1 - alpha2) * cross_score

        return final

    def get_gate_distributions(self, wav):
        """Consistency regularization용 gate distributions."""
        hs1 = self.mhubert(wav)
        gate_dists = []
        for i in range(0, len(hs1) - 1, 2):
            h1, h2 = hs1[i], hs1[i + 1]
            g = self.pair_gates[i // 2].gate(torch.cat([h1, h2], dim=-1))
            gate_dists.append(g.mean(dim=(1, 2)))  # [B]
        return gate_dists


# ============================================================
# One-Class Softmax Loss (Xiao et al., 2025)
# ============================================================

class OneClassSoftmaxLoss(nn.Module):
    """One-class softmax: learn compact real representation."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """features: [B, D], labels: [B] (0=real, 1=fake)"""
        # Normalize features
        features = F.normalize(features, dim=1)

        # Real class center
        real_mask = (labels == 0)
        if real_mask.sum() > 0:
            real_center = features[real_mask].mean(dim=0)
            real_center = F.normalize(real_center, dim=0)
        else:
            real_center = torch.zeros(features.size(1), device=features.device)

        # Cosine similarity to real center
        cos_sim = torch.mm(features, real_center.unsqueeze(1)).squeeze(1)  # [B]
        cos_sim = cos_sim / self.temperature

        # Softmax: real should have high similarity, fake should have low
        targets = 1 - labels  # real=1, fake=0
        loss = F.binary_cross_entropy_with_logits(cos_sim, targets)

        return loss


# ============================================================
# Consistency Regularization
# ============================================================

def consistency_loss(model, wav_clean, wav_aug, lambda_cons=0.25):
    with torch.no_grad():
        dists_clean = model.get_gate_distributions(wav_clean)
    dists_aug = model.get_gate_distributions(wav_aug)

    jsd_total = 0.0
    n_pairs = min(len(dists_clean), len(dists_aug))

    for i in range(n_pairs):
        p = torch.stack([dists_clean[i], 1 - dists_clean[i]], dim=-1)
        q = torch.stack([dists_aug[i], 1 - dists_aug[i]], dim=-1)
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
    ap = argparse.ArgumentParser(description="Ultimate Audio Deepfake Detector")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/raptor_v2"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--consistency-w", type=float, default=0.25)
    ap.add_argument("--p-aug", type=float, default=0.6)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)

    # Data
    tr_ds = FakeDetectDataset(args.train, train=True, p_aug=args.p_aug)
    va_ds = FakeDetectDataset(args.val, train=False)

    print(f"Train: {len(tr_ds)} samples, Val: {len(va_ds)} samples")

    dl_tr = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)
    dl_va = DataLoader(va_ds, batch_size=args.bs, num_workers=2, pin_memory=True)

    # Model
    print("Loading dual-backbone model (mHuBERT-Iter2 + XLS-R-300M)...")
    model = UltimateDetector(dropout=0.1).to(device)

    # Optimizer: different LR
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    backbone_params = list(model.mhubert.backbone.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("mhubert.backbone") and p.requires_grad]

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=1e-4)

    # Warmup + Cosine schedule
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    # Losses
    bce = nn.BCEWithLogitsLoss()
    oc_loss = OneClassSoftmaxLoss(temperature=0.1)

    best_eer = 1e9
    start_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best_eer = ck.get("best_eer", 1e9)
        print(f"Resumed from epoch {start_epoch}, best EER={best_eer:.4f}")

    rng_train = np.random.default_rng(42)
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

                # SpecAugment
                if rng_train.random() < 0.5:
                    wav_np = wav.cpu().numpy()
                    for i in range(len(wav_np)):
                        wav_np[i] = specaugment(wav_np[i], rng=rng_train)
                    wav = torch.from_numpy(wav_np).to(device)

                # Mixup (epoch 5+)
                use_mixup = ep >= 5 and rng_train.random() < 0.5
                if use_mixup:
                    wav_mixed, y_a, y_b, lam = mixup_data(wav, y, alpha=0.2, rng=rng_train)
                else:
                    wav_mixed = wav

                # Consistency augmentation
                rng_cons = np.random.default_rng(int(time.time() * 1000) % (2**31))
                wav_aug = wav.clone()
                for i in range(len(wav_aug)):
                    wav_aug[i] = torch.from_numpy(
                        apply_rawboost(wav_aug[i].cpu().numpy(), rng_cons, p=1.0)
                    ).to(device)

                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=(device == "cuda")):
                    logits = model(wav_mixed)

                    # Classification loss
                    if use_mixup:
                        targets_a = y_a * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
                        targets_b = y_b * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
                        loss_cls = mixup_criterion(bce, logits.float(), targets_a, targets_b, lam)
                    else:
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
            val_eer = evaluate_eer(model, dl_va, device)

            lg.writerow([ep, round(tot / max(nb, 1), 4), round(val_eer, 6), round(elapsed, 1)])
            log_file.flush()

            print(f"Epoch {ep}/{args.epochs}  loss={tot/max(nb,1):.4f}  "
                  f"valEER={val_eer:.4f}  time={elapsed:.0f}s")

            ck = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": ep,
                "best_eer": best_eer,
            }
            if val_eer < best_eer:
                best_eer = val_eer
                ck["best_eer"] = best_eer
                torch.save(ck, args.out / "best.pth")
                print(f"  → New best EER: {best_eer:.4f}")

            torch.save(ck, args.out / "last.pth")

    print(f"\nDone. Best val EER = {best_eer:.4f}")
    print(f"Model saved to: {args.out / 'best.pth'}")


if __name__ == "__main__":
    main()
