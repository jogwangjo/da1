#!/usr/bin/env python3
"""DACON 236749 제출 스크립트 V2 — 0.8+ 타겟.

핵심 개선 (V1 대비):
  1. TTA (Test-Time Augmentation): VoIP 코덱, 노이즈, 속도 → 평균
  2. Multi-model ensemble: DF-Arena + AASIST + SONICS + (파인튜닝 모델)
  3. 개선된 Fusion: noisy-or + logit-blend 하이브리드
  4. Presence 캘리브레이션
  5. RAPTOR 파인튜닝 모델 통합 (학습 시)

사용: python script_v2.py [--test-dir data/test] [--device cuda]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import librosa
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from tqdm import tqdm

try:
    from demucs.separate import load_track
except ImportError:
    from demucs.audio import convert_audio
    def load_track(path, audio_channels, samplerate):
        try:
            wav, sr = torchaudio.load(str(path))
        except Exception:
            y, sr = librosa.load(str(path), sr=None, mono=False)
            if y.ndim == 1:
                y = y[None]
            wav = torch.from_numpy(y.astype(np.float32))
        return convert_audio(wav, sr, samplerate, audio_channels)

# Paths
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"
AASIST_ONNX = MODEL_DIR / "w2v2-aasist.onnx"
RAPTOR_CKPT = MODEL_DIR / "raptor_best.pth"  # 파인튜닝 모델 (있으면 사용)
VENDOR_SONICS = BASE_DIR / "vendor_sonics" / "sonics_pkg"

DEFAULT_TEST_DIR = Path("data") / "test"
DEFAULT_SAMPLE_SUBMISSION = Path("data") / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = Path("output") / "submission.csv"

AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600
SILENCE_RMS = 1e-5

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]
SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}

# Ensemble weights (tuned on pseudo-eval)
W_DF_VOICE = 0.4       # DF-Arena in voice ensemble
W_DF_MUSIC = 0.3       # DF-Arena in music ensemble
W_AASIST = 0.3          # AASIST in voice ensemble
W_SONICS = 0.5          # SONICS in music ensemble
W_RAPTOR = 0.3          # RAPTOR in ensemble (if available)

# TTA settings
TTA_N_AUGMENTS = 2  # 0=off, 1=+voip, 2=+voip+noise, 3=+voip+noise+speed


# ============================================================
# Utility
# ============================================================

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tta", type=int, default=TTA_N_AUGMENTS,
                        help="TTA augmentations (0=off)")
    parser.add_argument("--no-aasist", action="store_true")
    parser.add_argument("--no-sonics", action="store_true")
    parser.add_argument("--no-df", action="store_true")
    parser.add_argument("--no-raptor", action="store_true")
    return parser.parse_args(argv)


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> CPU")
        return torch.device("cpu")
    return torch.device(name)


def find_audio_files(test_dir):
    audio_files = []
    for p in test_dir.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.append(p)
    audio_files.sort(key=lambda p: p.stem)
    if not audio_files:
        raise FileNotFoundError("No audio files found")
    return audio_files


def read_sample_submission(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    return cols, rows


def order_audio_files(audio_files, submission_rows):
    audio_by_id = {p.stem: p for p in audio_files}
    return [audio_by_id[r["ID"]] for r in submission_rows]


def load_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
    return audio


def get_segment_starts(audio_length):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]
    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio, start):
    if audio.size < SEGMENT_SAMPLES:
        reps = SEGMENT_SAMPLES // audio.size + 1
        return np.tile(audio, reps)[:SEGMENT_SAMPLES].astype(np.float32)
    return audio[start:start + SEGMENT_SAMPLES].astype(np.float32, copy=False)


def softmax2(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# ============================================================
# TTA (Test-Time Augmentation) — RAPTOR 논문 기반
# ============================================================

def voip_augment(audio):
    """MP3 코덱 시뮬레이션."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            import soundfile as sf
            sf.write(tmp_in.name, audio, AUDIO_SAMPLE_RATE)
            tmp_out = tmp_in.name.replace('.wav', '_voip.mp3')
            subprocess.run([
                'ffmpeg', '-y', '-i', tmp_in.name,
                '-codec:a', 'libmp3lame', '-b:a', '64k',
                '-ar', str(AUDIO_SAMPLE_RATE), tmp_out
            ], capture_output=True, timeout=10)
            augmented, _ = sf.read(tmp_out)
            os.unlink(tmp_in.name)
            if os.path.exists(tmp_out):
                os.unlink(tmp_out)
        return augmented.astype(np.float32)[:len(audio)]
    except Exception:
        return audio


def noise_augment(audio, snr_db=20):
    """백색 노이즈 첨가."""
    ps = np.mean(audio.astype(np.float64)**2) + 1e-10
    pn = ps / (10 ** (snr_db / 10))
    return (audio + np.random.normal(0, np.sqrt(pn), len(audio))).astype(np.float32)


def tta_views(audio, n_augments=2):
    """TTA 뷰 생성: 원본 + n_augments 증강."""
    views = [audio]
    if n_augments >= 1:
        views.append(voip_augment(audio))
    if n_augments >= 2:
        views.append(noise_augment(audio))
    return views[:n_augments + 1]


# ============================================================
# Fusion — 개선된 합성
# ============================================================

def combine_file_fake(voice_fake, music_fake, voice_present, music_present,
                       method="hybrid"):
    """개선된 파일 수준 fake score."""
    vrisk = voice_present * voice_fake
    mrisk = music_present * music_fake
    
    if method == "hybrid":
        noisy_or = 1 - (1 - vrisk) * (1 - mrisk)
        max_prod = max(vrisk, mrisk)
        return 0.6 * noisy_or + 0.4 * max_prod
    elif method == "noisy_or":
        return 1 - (1 - vrisk) * (1 - mrisk)
    else:
        return max(vrisk, mrisk)


def ensemble_logit_mean(scores, weights=None):
    """가중 로짓 평균 앙상블."""
    if weights is None:
        weights = [1.0] * len(scores)
    total_w = sum(weights)
    if total_w == 0:
        return float(np.mean(scores))
    logit_sum = sum(w * logit(s) for s, w in zip(scores, weights))
    return float(sigmoid(logit_sum / total_w))


# ============================================================
# PANNs Presence
# ============================================================

def prepare_panns_labels():
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(source, target)


def load_panns_model(device):
    prepare_panns_labels()
    from panns_inference import AudioTagging, labels
    model = AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device.type,
    )
    config_path = PANNS_DIR / "component_labels.json"
    label_groups = json.loads(config_path.read_text(encoding="utf-8"))
    label_to_index = {label: index for index, label in enumerate(labels)}
    voice_indices = [label_to_index[l] for l in label_groups["voice"]]
    music_indices = [label_to_index[l] for l in label_groups["music"]]
    return model, voice_indices, music_indices


def predict_presence(model, voice_indices, music_indices, audio):
    segments = []
    for start in get_segment_starts(audio.size):
        seg = extract_segment(audio, start)
        seg = librosa.resample(seg, orig_sr=AUDIO_SAMPLE_RATE,
                               target_sr=PANNS_SAMPLE_RATE, res_type="soxr_hq")
        segments.append(seg.astype(np.float32))
    segments = np.stack(segments)
    predictions, _ = model.inference(segments)
    vp = float(predictions[:, voice_indices].max())
    mp = float(predictions[:, music_indices].max())
    return vp, mp


def predict_presence_all(audio_files, device):
    model, vi, mi = load_panns_model(device)
    scores = {}
    for p in tqdm(audio_files, desc="Presence"):
        audio = load_audio(p)
        scores[p.stem] = predict_presence(model, vi, mi, audio)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


# ============================================================
# HTDemucs
# ============================================================

def load_htdemucs():
    orig = torch.load
    def trusted(*a, **kw):
        kw.setdefault("weights_only", False)
        return orig(*a, **kw)
    torch.load = trusted
    try:
        m = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = orig
    return m.cpu().eval()


def separate(audio_path, model, device):
    waveform = load_track(audio_path, model.audio_channels, model.samplerate).float()
    mono = waveform.mean(0)
    mean, std = mono.mean(), mono.std()
    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * AUDIO_SAMPLE_RATE / model.samplerate)
        s = np.zeros(max(1, length), dtype=np.float32)
        return s, s.copy()
    normed = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(model, normed[None], device=device, shifts=0,
                              split=True, overlap=0.25, progress=False)[0]
    sources = sources * std + mean
    vi = model.sources.index("vocals")
    voice = sources[vi].mean(0, keepdim=True)
    music = torch.stack([sources[i] for i, n in enumerate(model.sources) if n != "vocals"]).sum(0).mean(0, keepdim=True)
    voice = torchaudio.functional.resample(voice, model.samplerate, AUDIO_SAMPLE_RATE)[0]
    music = torchaudio.functional.resample(music, model.samplerate, AUDIO_SAMPLE_RATE)[0]
    return voice.cpu().numpy().astype(np.float32), music.cpu().numpy().astype(np.float32)


# ============================================================
# DF-Arena 1B
# ============================================================

def load_df_arena(device):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing
    cwd = Path.cwd()
    os.chdir(DF_ARENA_DIR)
    try:
        m = DF_Arena_1B_Antispoofing.from_pretrained(
            str(DF_ARENA_DIR), local_files_only=True, low_cpu_mem_usage=True)
    finally:
        os.chdir(cwd)
    m = m.to(device).eval()
    idx = int(m.config.label2id["spoof"])
    return m, idx


class DFArenaScorer:
    def __init__(self, model, idx, device):
        self.model, self.idx, self.device = model, idx, device

    def _predict(self, audio):
        if float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < SILENCE_RMS:
            return 0.0
        probs = []
        for start in get_segment_starts(audio.size):
            seg = extract_segment(audio, start)
            t = torch.from_numpy(seg).to(self.device)
            with torch.inference_mode():
                logits = self.model(input_values=t)["logits"]
                probs.append(float(torch.softmax(logits.float(), dim=-1)[0, self.idx]))
        return float(np.max(probs))

    def predict_tta(self, audio, n_augments=0):
        """TTA support: original + augmented views."""
        if n_augments == 0:
            return self._predict(audio)
        views = tta_views(audio, n_augments)
        scores = [self._predict(v) for v in views]
        return float(np.mean(scores))


# ============================================================
# AASIST ONNX
# ============================================================

class AASISTScorer:
    def __init__(self, onnx_path, device):
        import onnxruntime as ort
        providers = []
        if device.type == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        avail = ort.get_available_providers()
        providers = [p for p in providers if p in avail]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.batch = 8 if "CUDA" in str(self.sess.get_providers()) else 1

    def _predict(self, audio):
        segs = np.stack([extract_segment(audio, s) for s in get_segment_starts(audio.size)])
        scores = []
        for i in range(0, len(segs), self.batch):
            chunk = segs[i:i+self.batch].astype(np.float32)
            logits = self.sess.run([self.out_name], {self.in_name: chunk})[0]
            scores.extend(softmax2(logits)[:, 0].tolist())
        return float(np.max(scores))

    def predict_tta(self, audio, n_augments=0):
        if n_augments == 0:
            return self._predict(audio)
        views = tta_views(audio, n_augments)
        scores = [self._predict(v) for v in views]
        return float(np.mean(scores))


# ============================================================
# SONICS
# ============================================================

class SONICSScorer:
    def __init__(self, dirs, device):
        sys.path.insert(0, str(VENDOR_SONICS))
        from load_sonics import load_sonics
        self.models = []
        for d in dirs:
            net, meta = load_sonics(d, device.type)
            self.models.append((net, meta))

    def _predict(self, audio):
        per_model = []
        for net, meta in self.models:
            win = meta["win"]
            n = audio.size
            if n < win:
                reps = int(np.ceil(win / max(1, n)))
                windows = np.tile(audio, reps)[:win][None]
            else:
                step = win // 2
                starts = list(range(0, n - win + 1, step))
                windows = np.stack([audio[s:s+win] for s in starts])
            t = torch.from_numpy(np.ascontiguousarray(windows)).to(
                next(net.parameters()).device)
            logits = []
            for i in range(0, len(t), 8):
                logits.extend(net(t[i:i+8]).flatten().float().cpu().tolist())
            per_model.append(float(sigmoid(np.asarray(logits)).max()))
        return float(np.mean(per_model)) if per_model else 0.0

    def predict_tta(self, audio, n_augments=0):
        if n_augments == 0:
            return self._predict(audio)
        views = tta_views(audio, n_augments)
        scores = [self._predict(v) for v in views]
        return float(np.mean(scores))


# ============================================================
# RAPTOR (Fine-tuned model)
# ============================================================

class RAPTORScorer:
    def __init__(self, ckpt_path, device):
        self.ckpt = Path(ckpt_path)
        self.device = device

    def _load_model(self):
        if not self.ckpt.exists():
            return None
        # Import RAPTOR from training code
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from train_raptor import RAPTOR
        ck = torch.load(self.ckpt, map_location=self.device)
        model = RAPTOR(ck.get("backbone", "utter-project/mHuBERT-147"))
        model.load_state_dict(ck["model"])
        model = model.to(self.device).eval()
        return model

    def _predict(self, audio, model):
        probs = []
        for start in get_segment_starts(audio.size):
            seg = extract_segment(audio, start)
            t = torch.from_numpy(seg).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logit_val = model(t)
                probs.append(float(torch.sigmoid(logit_val)))
        return float(np.max(probs))

    def predict_tta(self, audio, n_augments=0):
        model = self._load_model()
        if model is None:
            return None
        if n_augments == 0:
            return self._predict(audio, model)
        views = tta_views(audio, n_augments)
        scores = [self._predict(v, model) for v in views]
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return float(np.mean(scores))


# ============================================================
# Main
# ============================================================

def main():
    args = parse_arguments(sys.argv[1:])
    device = select_device(args.device)
    use_aasist = not args.no_aasist and AASIST_ONNX.exists()
    use_sonics = not args.no_sonics
    use_df = not args.no_df and DF_ARENA_DIR.exists()
    use_raptor = not args.no_raptor and RAPTOR_CKPT.exists()

    # Files
    audio_files_all = find_audio_files(args.test_dir)
    cols, rows = read_sample_submission(args.sample_submission)
    audio_files = order_audio_files(audio_files_all, rows)
    if args.limit:
        audio_files = audio_files[:args.limit]
        rows = rows[:args.limit]
    print(f"files: {len(audio_files)}, TTA: {args.tta}")

    # 1) Presence
    print("Loading PANNs for presence ...")
    presence_scores = predict_presence_all(audio_files, device)

    # 2) Load scorers
    df_scorer = aasist_scorer = sonics_scorer = raptor_scorer = None
    if use_df:
        print("Loading DF-Arena 1B ...")
        m, idx = load_df_arena(device)
        df_scorer = DFArenaScorer(m, idx, device)
    if use_aasist:
        print("Loading AASIST ONNX ...")
        aasist_scorer = AASISTScorer(AASIST_ONNX, device)
    if use_sonics:
        print("Loading SONICS ...")
        sonics_scorer = SONICSScorer(
            [MODEL_DIR / "sonics-alpha-5s", MODEL_DIR / "sonics-beta-5s"], device)
    if use_raptor:
        print("Loading RAPTOR ...")
        raptor_scorer = RAPTORScorer(RAPTOR_CKPT, device)

    # 3) HTDemucs
    print("Loading HTDemucs ...")
    htdemucs = load_htdemucs()

    # 4) Inference loop
    t0 = time.time()
    for idx_file, audio_path in enumerate(tqdm(audio_files, desc="Inference")):
        voice_audio, music_audio = separate(audio_path, htdemucs, device)

        # Voice ensemble
        vf_parts, vf_weights = [], []
        if df_scorer is not None:
            vf_parts.append(df_scorer.predict_tta(voice_audio, args.tta))
            vf_weights.append(W_DF_VOICE)
        if aasist_scorer is not None:
            vf_parts.append(aasist_scorer.predict_tta(voice_audio, args.tta))
            vf_weights.append(W_AASIST)
        if raptor_scorer is not None:
            rp = raptor_scorer.predict_tta(voice_audio, args.tta)
            if rp is not None:
                vf_parts.append(rp)
                vf_weights.append(W_RAPTOR)

        # Music ensemble
        mf_parts, mf_weights = [], []
        if df_scorer is not None:
            mf_parts.append(df_scorer.predict_tta(music_audio, args.tta))
            mf_weights.append(W_DF_MUSIC)
        if sonics_scorer is not None:
            mf_parts.append(sonics_scorer.predict_tta(music_audio, args.tta))
            mf_weights.append(W_SONICS)
        if raptor_scorer is not None:
            rp = raptor_scorer.predict_tta(music_audio, args.tta)
            if rp is not None:
                mf_parts.append(rp)
                mf_weights.append(W_RAPTOR)

        voice_fake = ensemble_logit_mean(vf_parts, vf_weights) if vf_parts else 0.0
        music_fake = ensemble_logit_mean(mf_parts, mf_weights) if mf_parts else 0.0

        voice_present, music_present = presence_scores[audio_path.stem]
        file_fake = combine_file_fake(voice_fake, music_fake,
                                       voice_present, music_present, method="hybrid")

        row = rows[idx_file]
        row["FILE_FAKE_PROB"] = round(file_fake, 10)
        row["VOICE_FAKE_PROB"] = round(voice_fake, 10)
        row["MUSIC_FAKE_PROB"] = round(music_fake, 10)
        row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present, 10)

    elapsed = time.time() - t0
    print(f"[time] {elapsed:.0f}s ({elapsed/len(audio_files):.2f}s/file)")

    # 5) Save
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
