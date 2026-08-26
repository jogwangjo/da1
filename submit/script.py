#!/usr/bin/env python3
"""DACON 236749 코드 제출용 추론 스크립트 (앙상블 버전).

구성
  presence : PANNs Cnn14            (baseline 동일)
  분리     : HTDemucs               (baseline 동일)
  voice    : DF-Arena 1B + w2v2-AASIST(ONNX)  logit-mean 앙상블
  music    : DF-Arena 1B + SONICS SpecTTTra(a+b) logit-mean 앙상블
  file     : max(VP*VF, MP*MF)      (fusion 모드 설정 가능)

실행: python script.py [--limit N] [--no-aasist] [--no-sonics] [--no-df]
"""
import argparse
import csv
import json
import os
import shutil
import sys
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

try:  # demucs <=4.0
    from demucs.separate import load_track
except ImportError:  # demucs >=4.1 폴백
    from demucs.audio import convert_audio

    def load_track(path, audio_channels, samplerate):
        try:
            wav, sr = torchaudio.load(str(path))
        except Exception:
            import librosa

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
AASIST_ONNX = MODEL_DIR / "w2v2-aasist.onnx"
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

FUSION_MODE = "max_product"   # or "noisy_or" / "max_raw"
W_DF_VOICE = 0.5              # voice 앙상블 내 DF-Arena 비중 (logit 도메인)
W_DF_MUSIC = 0.5


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--sample-submission", type=Path,
                        default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-aasist", action="store_true")
    parser.add_argument("--no-sonics", action="store_true")
    parser.add_argument("--no-df", action="store_true")
    return parser.parse_args(argv)


def select_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> CPU")
        device_name = "cpu"
    return torch.device(device_name)


def find_audio_files(test_dir):
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    audio_files = []
    for path in test_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.append(path)
    audio_files.sort(key=lambda path: path.stem)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")
    audio_ids = [path.stem for path in audio_files]
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("Audio IDs must be unique")
    return audio_files


def read_sample_submission(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sample submission not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        column_names = reader.fieldnames
        rows = list(reader)
    if column_names is None or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")
    required_columns = ["ID"] + PREDICTION_COLUMNS
    missing = [n for n in required_columns if n not in column_names]
    if missing:
        raise ValueError(f"Sample submission is missing columns: {missing}")
    seen = set()
    for row in rows:
        audio_id = str(row["ID"]).strip()
        if not audio_id:
            raise ValueError("Empty ID in sample submission")
        if audio_id in seen:
            raise ValueError(f"Duplicate ID: {audio_id}")
        seen.add(audio_id)
        row["ID"] = audio_id
    return column_names, rows


def order_audio_files(audio_files, submission_rows):
    audio_by_id = {path.stem: path for path in audio_files}
    submission_ids = [row["ID"] for row in submission_rows]
    missing = [i for i in submission_ids if i not in audio_by_id]
    extra = [i for i in audio_by_id if i not in submission_ids]
    if missing or extra:
        raise ValueError(
            f"IDs mismatch. Missing: {missing[:5]}, Extra: {extra[:5]}"
        )
    return [audio_by_id[i] for i in submission_ids]


def load_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True,
                            dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {audio_path}")
    return audio


# ---------------------------------------------------------------- segments --
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
        repeat_count = SEGMENT_SAMPLES // audio.size + 1
        padded = np.tile(audio, repeat_count)
        return padded[:SEGMENT_SAMPLES].astype(np.float32)
    end = start + SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


def softmax2(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# ------------------------------------------------------------- PANNs ------
def prepare_panns_labels():
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
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
    voice_indices = [label_to_index[label] for label in label_groups["voice"]]
    music_indices = [label_to_index[label] for label in label_groups["music"]]
    return model, voice_indices, music_indices


def make_panns_segments(audio):
    segments = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment = librosa.resample(
            segment, orig_sr=AUDIO_SAMPLE_RATE, target_sr=PANNS_SAMPLE_RATE,
            res_type="soxr_hq",
        )
        segments.append(segment.astype(np.float32))
    return np.stack(segments)


def predict_presence(model, voice_indices, music_indices, audio):
    segments = make_panns_segments(audio)
    predictions, _ = model.inference(segments)
    voice_probability = float(predictions[:, voice_indices].max())
    music_probability = float(predictions[:, music_indices].max())
    return voice_probability, music_probability


def predict_presence_for_all_files(audio_files, device):
    model, voice_indices, music_indices = load_panns_model(device)
    presence_scores = {}
    t0 = time.time()
    for audio_path in tqdm(audio_files, desc="Presence"):
        audio = load_audio(audio_path)
        presence_scores[audio_path.stem] = predict_presence(
            model, voice_indices, music_indices, audio
        )
    print(f"[time] presence: {time.time()-t0:.0f}s")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return presence_scores


# ------------------------------------------------------------- HTDemucs ---
def load_htdemucs_model():
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = original_torch_load
    return model.cpu().eval()


def separate_voice_and_music(audio_path, model, device):
    waveform = load_track(audio_path, model.audio_channels, model.samplerate).float()
    mono_waveform = waveform.mean(0)
    mean = mono_waveform.mean()
    std = mono_waveform.std()

    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * AUDIO_SAMPLE_RATE / model.samplerate)
        silence = np.zeros(max(1, length), dtype=np.float32)
        return silence, silence.copy()

    normalized_waveform = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model, normalized_waveform[None], device=device, shifts=0,
            split=True, overlap=0.25, progress=False,
        )[0]
    sources = sources * std + mean

    vocal_index = model.sources.index("vocals")
    voice_audio = sources[vocal_index].mean(0, keepdim=True)

    music_sources = []
    for index, source_name in enumerate(model.sources):
        if source_name != "vocals":
            music_sources.append(sources[index])
    music_audio = torch.stack(music_sources).sum(0).mean(0, keepdim=True)

    voice_audio = torchaudio.functional.resample(
        voice_audio, model.samplerate, AUDIO_SAMPLE_RATE)[0]
    music_audio = torchaudio.functional.resample(
        music_audio, model.samplerate, AUDIO_SAMPLE_RATE)[0]
    return (
        voice_audio.cpu().numpy().astype(np.float32),
        music_audio.cpu().numpy().astype(np.float32),
    )


# ------------------------------------------------------------ DF-Arena ----
def load_df_arena_model(device):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    previous_directory = Path.cwd()
    os.chdir(DF_ARENA_DIR)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(DF_ARENA_DIR), local_files_only=True, low_cpu_mem_usage=True,
        )
    finally:
        os.chdir(previous_directory)

    model = model.to(device).eval()
    fake_label_index = int(model.config.label2id["spoof"])
    return model, fake_label_index


def calculate_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


class DFArenaScorer:
    def __init__(self, model, fake_label_index, device, batch=1):
        self.model, self.idx, self.device = model, fake_label_index, device

    def predict(self, audio):
        if calculate_rms(audio) < SILENCE_RMS:
            return 0.0
        probs = []
        for start in get_segment_starts(audio.size):
            seg = extract_segment(audio, start)
            t = torch.from_numpy(seg).to(self.device)  # 1D 필수 (backbone이 unsqueeze)
            with torch.inference_mode():
                logits = self.model(input_values=t)["logits"]
                probs.append(float(torch.softmax(logits.float(), dim=-1)[0, self.idx]))
        return float(np.max(probs))


# ------------------------------------------------------------ AASIST ------
class AasistOnnxScorer:
    """w2v2-AASIST ONNX. logits[:,0]=spoof."""

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
        self.batch = 8 if "CUDA" in self.sess.get_providers()[0] else 1

    def predict(self, audio):
        segs = np.stack([extract_segment(audio, s)
                         for s in get_segment_starts(audio.size)])
        scores = []
        for i in range(0, len(segs), self.batch):
            chunk = segs[i:i + self.batch]
            logits = self.sess.run([self.out_name], {self.in_name: chunk})[0]
            scores.extend(softmax2(logits)[:, 0].tolist())
        return float(np.max(scores))


# ------------------------------------------------------------ SONICS ------
class SonicsEnsembleScorer:
    def __init__(self, dirs, device):
        sys.path.insert(0, str(VENDOR_SONICS))
        from load_sonics import load_sonics

        self.models = []
        for d in dirs:
            net, meta = load_sonics(d, device.type)
            self.models.append((net, meta))

    @staticmethod
    def _windows(audio, win):
        n = audio.size
        if n < win:
            reps = int(np.ceil(win / max(1, n)))
            return np.tile(audio, reps)[:win][None]
        step = win // 2
        starts = list(range(0, n - win + 1, step))
        return np.stack([audio[s:s + win] for s in starts])

    def predict(self, audio):
        per_model = []
        for net, meta in self.models:
            wins = self._windows(audio, meta["win"])
            t = torch.from_numpy(np.ascontiguousarray(wins)).to(next(net.parameters()).device)
            logits = []
            for i in range(0, len(t), 8):
                logits.extend(net(t[i:i + 8]).flatten().float().cpu().tolist())
            per_model.append(float(sigmoid(np.asarray(logits)).max()))
        return float(np.mean(per_model)) if per_model else 0.0


# ---------------------------------------------------------------- fusion --
def combine_file_fake_score(vf, mf, vp, mp):
    vrisk, mrisk = vp * vf, mp * mf
    if FUSION_MODE == "noisy_or":
        return 1 - (1 - vrisk) * (1 - mrisk)
    if FUSION_MODE == "max_raw":
        return max(vf, mf)
    return max(vrisk, mrisk)


def ensemble_logit_mean(scores, weights):
    vals = [(logit(s), w) for s, w in zip(scores, weights)]
    tot_w = sum(w for _, w in vals) or 1.0
    return float(sigmoid(sum(l * w for l, w in vals) / tot_w))


# ------------------------------------------------------------------ main --
def main():
    global FUSION_MODE
    args = parse_arguments(sys.argv[1:])
    device = select_device(args.device)
    use_aasist = not args.no_aasist
    use_sonics = not args.no_sonics
    use_df = not args.no_df

    audio_files_all = find_audio_files(args.test_dir)
    column_names, submission_rows = read_sample_submission(args.sample_submission)
    audio_files = order_audio_files(audio_files_all, submission_rows)
    if args.limit:
        audio_files = audio_files[: args.limit]
        submission_rows = submission_rows[: args.limit]
    print(f"files: {len(audio_files)}")

    # 1) presence
    presence_scores = predict_presence_for_all_files(audio_files, device)

    # 2) scorers
    df_scorer = aasist_scorer = sonics_scorer = None
    if use_df:
        print("loading DF-Arena ...")
        m, idx = load_df_arena_model(device)
        df_scorer = DFArenaScorer(m, idx, device)
    if use_aasist:
        print("loading AASIST(onnx) ...")
        aasist_scorer = AasistOnnxScorer(AASIST_ONNX, device)
    if use_sonics:
        print("loading SONICS ...")
        sonics_scorer = SonicsEnsembleScorer(
            [MODEL_DIR / "sonics-alpha-5s", MODEL_DIR / "sonics-beta-5s"], device)

    htdemucs_model = load_htdemucs_model()

    # 3) loop
    t0 = time.time()
    for index, audio_path in enumerate(tqdm(audio_files, desc="Components")):
        voice_audio, music_audio = separate_voice_and_music(
            audio_path, htdemucs_model, device)

        vf_parts, vf_weights, mf_parts, mf_weights = [], [], [], []
        if df_scorer is not None:
            vf_parts.append(df_scorer.predict(voice_audio)); vf_weights.append(W_DF_VOICE)
            mf_parts.append(df_scorer.predict(music_audio)); mf_weights.append(W_DF_MUSIC)
        if aasist_scorer is not None:
            vf_parts.append(aasist_scorer.predict(voice_audio)); vf_weights.append(1 - W_DF_VOICE)
        if sonics_scorer is not None:
            mf_parts.append(sonics_scorer.predict(music_audio)); mf_weights.append(1 - W_DF_MUSIC)

        voice_fake = ensemble_logit_mean(vf_parts, vf_weights) if vf_parts else 0.0
        music_fake = ensemble_logit_mean(mf_parts, mf_weights) if mf_parts else 0.0

        voice_present, music_present = presence_scores[audio_path.stem]
        file_fake = combine_file_fake_score(
            voice_fake, music_fake, voice_present, music_present)

        row = submission_rows[index]
        row["FILE_FAKE_PROB"] = round(file_fake, 10)
        row["VOICE_FAKE_PROB"] = round(voice_fake, 10)
        row["MUSIC_FAKE_PROB"] = round(music_fake, 10)
        row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present, 10)

    print(f"[time] component loop: {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/len(audio_files):.2f}s/file)")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(submission_rows)
    print(f"Saved {len(submission_rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
