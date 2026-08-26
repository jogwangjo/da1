#!/usr/bin/env python3
"""_DACON 236749 제출 스크립트 — 안전한 baseline回归 + TTA.

문제: AASIST/SONICS 앙상블이 점수를 오히려 떨어뜨림 (0.641 vs baseline 0.691)
해결: DF-Arena만 사용 + TTA(VoIP+노이즈)로 안전하게 개선

현재 리더보드: baseline 0.69091
목표: TTA로 0.70+ (AASIST/SONICS는 검증 후 추가)
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

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"

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

# TTA settings
TTA_ENABLED = True


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
    parser.add_argument("--no-tta", action="store_true")
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


def calculate_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


# ============================================================
# TTA (Test-Time Augmentation) — RAPTOR 논문 기반
# ============================================================

def voip_augment(audio):
    """MP3 코덱 시뮬레이션 (VoIP 왜곡)."""
    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
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


# ============================================================
# PANNs Presence (baseline과 동일)
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
# HTDemucs (baseline과 동일)
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
# DF-Arena 1B (baseline과 동일 + TTA)
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


def predict_fake_single(model, fake_label_index, audio, device):
    """Baseline과 동일: single prediction (no TTA)."""
    if calculate_rms(audio) < SILENCE_RMS:
        return 0.0
    probs = []
    for start in get_segment_starts(audio.size):
        seg = extract_segment(audio, start)
        t = torch.from_numpy(seg).to(device)
        with torch.inference_mode():
            logits = model(input_values=t)["logits"]
            probs.append(float(torch.softmax(logits.float(), dim=-1)[0, fake_label_index]))
    return max(probs)


def predict_fake_tta(model, fake_label_index, audio, device):
    """TTA: original + VoIP + noise → 평균."""
    views = [audio, voip_augment(audio), noise_augment(audio)]
    scores = []
    for v in views:
        scores.append(predict_fake_single(model, fake_label_index, v, device))
    return float(np.mean(scores))


def combine_file_fake_score(voice_fake, music_fake, voice_present, music_present):
    """Baseline과 동일: max(VP*VF, MP*MF)."""
    voice_score = voice_present * voice_fake
    music_score = music_present * music_fake
    return max(voice_score, music_score)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_arguments(sys.argv[1:])
    device = select_device(args.device)
    use_tta = not args.no_tta

    # Files
    audio_files_all = find_audio_files(args.test_dir)
    cols, rows = read_sample_submission(args.sample_submission)
    audio_files = order_audio_files(audio_files_all, rows)
    if args.limit:
        audio_files = audio_files[:args.limit]
        rows = rows[:args.limit]
    print(f"files: {len(audio_files)}, TTA: {use_tta}")

    # 1) Presence
    print("Loading PANNs for presence ...")
    presence_scores = predict_presence_all(audio_files, device)

    # 2) DF-Arena only (baseline回归)
    print("Loading DF-Arena 1B ...")
    df_model, fake_idx = load_df_arena(device)

    # 3) HTDemucs
    print("Loading HTDemucs ...")
    htdemucs = load_htdemucs()

    # 4) Inference loop
    t0 = time.time()
    for idx_file, audio_path in enumerate(tqdm(audio_files, desc="Inference")):
        voice_audio, music_audio = separate(audio_path, htdemucs, device)

        # DF-Arena only (safe baseline)
        if use_tta:
            voice_fake = predict_fake_tta(df_model, fake_idx, voice_audio, device)
            music_fake = predict_fake_tta(df_model, fake_idx, music_audio, device)
        else:
            voice_fake = predict_fake_single(df_model, fake_idx, voice_audio, device)
            music_fake = predict_fake_single(df_model, fake_idx, music_audio, device)

        voice_present, music_present = presence_scores[audio_path.stem]
        file_fake = combine_file_fake_score(
            voice_fake, music_fake, voice_present, music_present)

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
