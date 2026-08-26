"""Vendored SONICS SpecTTTra loader (awsaf49/sonics, Apache-2.0).

패키지 설치 없이 로컬 스냅샷(config.json + pytorch_model.bin)을 로드한다.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _NS:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = _NS(v)
            setattr(self, k, v)


class SonicsNet(nn.Module):
    """sonics.models.model.AudioClassifier의 추론 전용 복제(속성명 동일)."""

    def __init__(self, cfg):
        super().__init__()
        from sonics.layers.feature import FeatureExtractor
        from sonics.models.spectttra import SpecTTTra

        self.cfg = cfg
        self.model_name = cfg.model.name
        self.input_shape = tuple(cfg.model.input_shape)
        self.num_classes = cfg.num_classes
        self.use_global_pool = True
        self.ft_extractor = FeatureExtractor(cfg)
        self.encoder = SpecTTTra(
            input_spec_dim=cfg.model.input_shape[0],
            input_temp_dim=cfg.model.input_shape[1],
            embed_dim=cfg.model.embed_dim,
            t_clip=cfg.model.t_clip,
            f_clip=cfg.model.f_clip,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            pre_norm=cfg.model.pre_norm,
            pe_learnable=cfg.model.pe_learnable,
            pos_drop_rate=getattr(cfg.model, "pos_drop_rate", 0.0),
            attn_drop_rate=getattr(cfg.model, "attn_drop_rate", 0.0),
            proj_drop_rate=getattr(cfg.model, "proj_drop_rate", 0.0),
            mlp_ratio=getattr(cfg.model, "mlp_ratio", 4.0),
        )
        self.embed_dim = cfg.model.embed_dim
        self.classifier = nn.Linear(cfg.model.embed_dim, cfg.num_classes)

    def forward(self, audio):  # [B, T]
        spec = self.ft_extractor(audio)
        spec = spec.unsqueeze(1)
        spec = F.interpolate(spec, size=tuple(self.input_shape), mode="bilinear")
        features = self.encoder(spec)
        embeds = features.mean(dim=1) if self.use_global_pool else features
        return self.classifier(embeds)


def load_sonics(model_dir, device="cpu"):
    pkg = Path(__file__).resolve().parent
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    model_dir = Path(model_dir)
    cfg = _NS(json.loads((model_dir / "config.json").read_text(encoding="utf-8")))
    net = SonicsNet(cfg)
    sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("augment")]
    assert not real_missing, f"missing keys: {real_missing[:8]}"
    net = net.to(device).eval()
    max_len = int(cfg.audio.max_len)          # 5s 모델: 80000
    sr = int(cfg.audio.sample_rate)           # 16000
    normalize = bool(getattr(cfg.audio, "normalize", True))
    return net, dict(win=max_len, sr=sr, normalize=normalize)


@torch.inference_mode()
def sonics_predict(net, meta, wav, device):
    """wav: np.float32 mono 16k -> fake 확률 (윈도우 분할 후 평균)."""
    import numpy as np

    win, sr = meta["win"], meta["sr"]
    x = np.asarray(wav, dtype=np.float32)
    n = len(x)
    if n < win:
        reps = int(np.ceil(win / max(1, n)))
        x = np.tile(x, reps)[:win]
    else:
        step = win // 2
        starts = list(range(0, n - win + 1, step))
        x = np.concatenate([x[s : s + win][None] for s in starts], axis=0)
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    logits = []
    for i in range(0, len(t), 8):
        logits.extend(net(t[i : i + 8]).flatten().float().cpu().tolist())
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    return float(probs.mean())
