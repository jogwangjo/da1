#!/usr/bin/env python3
"""학습 데이터 자동 구축 — 0.8+ 파인튜닝용.

다운로드:
  1. ASVspoof 2019 LA (음성 fake, 12.6GB) — Edinburgh datashare
  2. Codecfake (코덱 기반 음성 fake) — HuggingFace
  3. SONICS (노래/음악 fake, 49k곡) — HuggingFace awsaf49/sonics
  4. LibriTTS-R (음성 real) — _cache/librispeech 재사용
  5. MUSDB18 (음악 real) — _cache/musdb 재사용
  6. FMA small (음악 real) — _cache/fma 재사용

출력:
  manifest_voice_train.csv : voice 모델 학습 (real/fake)
  manifest_music_train.csv : music 모델 학습 (real/fake)
  manifest_all_train.csv   : 통합 모델 학습

사용법:
  python scripts/build_train_data.py --out train_data --asvspoof-dir /path/to/asvspoof2019la
  또는 HuggingFace에서 자동 다운로드:
  python scripts/build_train_data.py --out train_data --auto-download
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np


SR = 16000


def save_wav(x, path):
    import soundfile as sf
    sf.write(str(path), np.clip(x, -1, 1), SR, subtype="PCM_16")


def load_wav(path):
    import librosa
    x, _ = librosa.load(str(path), sr=SR, mono=True)
    return x.astype(np.float32)


def trim_or_pad(x, dur_sec, rng=None):
    n = int(dur_sec * SR)
    if len(x) >= n:
        if rng is not None:
            s = int(rng.integers(0, len(x) - n + 1))
            return x[s:s+n]
        s = (len(x) - n) // 2
        return x[s:s+n]
    reps = int(np.ceil(n / max(1, len(x))))
    return np.tile(x, reps)[:n]


# ============================================================
# 1. ASVspoof 2019 LA — 음성 fake 학습의 핵심 (자동 다운로드)
# ============================================================

def download_asvspoof_2019(out_dir):
    """ASVspoof 2019 LA 자동 다운로드 (Edinburgh DataShare)."""
    out_dir = Path(out_dir)
    flac_dir = out_dir / "flac"
    if flac_dir.exists() and len(list(flac_dir.glob("*.flac"))) > 1000:
        print(f"  ASVspoof 2019 already exists at {out_dir}")
        return True

    print("  Downloading ASVspoof 2019 LA...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 공개 다운로드 URL (Edinburgh DataShare)
    urls = {
        "train_flac": "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/LA_trn.flac.zip",
        "train_dev": "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/LA_dev_metadata.txt",
        "train_label": "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/LA_cm/train/trial_metadata.txt",
    }

    try:
        # Try downloading train flac (main data)
        zip_path = out_dir / "LA_trn.flac.zip"
        if not zip_path.exists() and not flac_dir.exists():
            print("  Downloading ASVspoof 2019 LA train flac (~2.5GB)...")
            subprocess.run([
                "wget", "-q", "--show-progress", "-O", str(zip_path),
                urls["train_flac"]
            ], timeout=600)

        # Extract
        if zip_path.exists() and not flac_dir.exists():
            print("  Extracting...")
            subprocess.run([
                "unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)
            ], timeout=300)
            # Clean up zip to save space
            zip_path.unlink(missing_ok=True)

        # Download label file
        label_path = out_dir / "cm" / "train" / "trial_metadata.txt"
        if not label_path.exists():
            label_dir = out_dir / "cm" / "train"
            label_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "wget", "-q", "-O", str(label_path),
                urls["train_label"]
            ], timeout=60)

        return flac_dir.exists() and len(list(flac_dir.glob("*.flac"))) > 0

    except Exception as e:
        print(f"  ASVspoof download failed: {e}")
        return False


def prepare_asvspoof_2019_la(asvspoof_dir, out_dir, max_per_class=5000):
    """ASVspoof 2019 LA train → manifest."""
    asvspoof_dir = Path(asvspoof_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Label file path
    label_path = None
    for candidate in [
        asvspoof_dir / "cm" / "train" / "trial_metadata.txt",
        asvspoof_dir / "LA_cm" / "train" / "trial_metadata.txt",
    ]:
        if candidate.exists():
            label_path = candidate
            break

    # Fallback: search for any txt with bonafide/spoof
    if label_path is None:
        for lf in asvspoof_dir.rglob("*.txt"):
            try:
                content = lf.read_text()[:2000]
                if "bonafide" in content or "spoof" in content:
                    label_path = lf
                    break
            except Exception:
                continue

    flac_dir = asvspoof_dir / "flac"
    if not flac_dir.exists():
        # Try parent dirs
        for p in asvspoof_dir.rglob("flac"):
            if p.is_dir() and len(list(p.glob("*.flac"))) > 100:
                flac_dir = p
                break

    if not flac_dir.exists() or not label_path:
        print(f"  WARNING: ASVspoof structure not found in {asvspoof_dir}")
        # Fallback: use whatever flac files exist
        if flac_dir.exists():
            flacs = sorted(flac_dir.glob("*.flac"))
            for i, f in enumerate(flacs[:max_per_class * 2]):
                label = 1 if i >= max_per_class else 0
                rows.append((str(f), label, "asvspoof19"))
        return rows

    # Parse label file
    real_count, fake_count = 0, 0
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id, label_str = parts[0], parts[1]
            is_fake = 1 if label_str == "spoof" else 0

            # Find flac file
            flac_path = flac_dir / f"{utt_id}.flac"
            if not flac_path.exists():
                # Try subdirectory structure
                for p in flac_dir.rglob(f"{utt_id}.flac"):
                    flac_path = p
                    break
                else:
                    continue

            if is_fake == 0 and real_count < max_per_class:
                rows.append((str(flac_path), 0, "asvspoof19"))
                real_count += 1
            elif is_fake == 1 and fake_count < max_per_class * 2:
                rows.append((str(flac_path), 1, "asvspoof19"))
                fake_count += 1

    return rows


# ============================================================
# 2. Codecfake — 코덱 기반 음성 fake
# ============================================================
def prepare_codecfake(out_dir, max_samples=3000):
    """Codecfake HuggingFace에서 다운로드."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        from datasets import load_dataset
        ds = load_dataset("rogertseng/CodecFake", split="train", streaming=True, revision="refs/convert/parquet")

        rows = []
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            audio = sample["audio"]
            wav_path = out_dir / f"codecfake_{i:06d}.wav"
            if not wav_path.exists():
                import soundfile as sf
                audio_array = np.array(audio["array"], dtype=np.float32)
                if audio["sampling_rate"] != SR:
                    import librosa
                    audio_array = librosa.resample(
                        audio_array, orig_sr=audio["sampling_rate"], target_sr=SR
                    )
                sf.write(str(wav_path), np.clip(audio_array, -1, 1), SR)
            rows.append((str(wav_path), 1, "codecfake"))

        return rows
    except Exception as e:
        print(f"Codecfake download failed: {e}")
        return []


# ============================================================
# 2b. Diverse TTS Fakes — 여러 TTS 엔진으로 생성
# ============================================================
def prepare_diverse_tts_fakes(out_dir, max_samples=3000):
    """generate_diverse_fakes.py 호출해서 다양한 TTS fake 생성."""
    try:
        script = Path(__file__).parent / "generate_diverse_fakes.py"
        if not script.exists():
            print(f"  generate_diverse_fakes.py not found")
            return []

        tts_dir = out_dir / "tts_fakes"
        tts_dir.mkdir(parents=True, exist_ok=True)

        # Check existing
        existing = len(list(tts_dir.glob("tts_fake_*.wav")))
        if existing >= max_samples:
            print(f"  Diverse TTS: {existing} files already exist")
        else:
            print(f"  Generating {max_samples} diverse TTS fakes...")
            subprocess.run([
                sys.executable, str(script),
                "--out", str(tts_dir),
                "--max", str(max_samples),
                "--augment",
            ], timeout=3600)

        rows = []
        for wav in sorted(tts_dir.glob("tts_fake_*.wav")):
            rows.append((str(wav), 1, "diverse_tts"))

        return rows[:max_samples]

    except Exception as e:
        print(f"  Diverse TTS generation failed: {e}")
        return []


# ============================================================
# 3. SONICS — 노래/음악 fake (Suno/Udio)
# ============================================================
def prepare_sonics(out_dir, max_samples=5000):
    """SONICS HF에서 오디오 다운로드.
    
    Note: awsaf49/sonics는 메타데이터만 있는 경우가 많음.
    audio 컬럼이 없으면 youtube_id로 다운로드 시도.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("awsaf49/sonics", split="train", streaming=True)
        
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        count = 0
        
        sample = next(iter(ds))
        cols = list(sample.keys())
        print(f"  SONICS columns: {cols}")
        
        has_audio = "audio" in cols
        
        if not has_audio:
            # Try to download audio using yt-dlp from youtube_id
            print("  SONICS: No audio column, trying yt-dlp download...")
            ds = load_dataset("awsaf49/sonics", split="train", streaming=True)
            
            try:
                import subprocess, tempfile
                for i, s in enumerate(ds):
                    if count >= max_samples:
                        break
                    ytid = s.get("youtube_id") or s.get("id")
                    if not ytid:
                        continue
                    wav_path = out_dir / f"sonics_fake_{count:06d}.wav"
                    if wav_path.exists():
                        rows.append((str(wav_path), 1, "sonics"))
                        count += 1
                        continue
                    try:
                        url = f"https://www.youtube.com/watch?v={ytid}"
                        with tempfile.TemporaryDirectory() as td:
                            out_file = os.path.join(td, "audio")
                            r = subprocess.run(
                                ["yt-dlp", "-x", "--audio-format", "wav",
                                 "--max-filesize", "10M",
                                 "-o", out_file + ".%(ext)s", url],
                                capture_output=True, timeout=30
                            )
                            # Find the output wav
                            for f in os.listdir(td):
                                if f.endswith(".wav"):
                                    shutil.copy2(os.path.join(td, f), str(wav_path))
                                    rows.append((str(wav_path), 1, "sonics"))
                                    count += 1
                                    break
                    except Exception:
                        continue
                    if count % 100 == 0 and count > 0:
                        print(f"  SONICS yt-dlp: {count} files...")
            except Exception as e2:
                print(f"  SONICS yt-dlp failed: {e2}")
            
            if count == 0:
                print("  SONICS: skipping (no audio available)")
            return rows
        
        # Has audio column - direct download
        ds = load_dataset("awsaf49/sonics", split="train", streaming=True)
        for i, sample in enumerate(ds):
            if count >= max_samples:
                break
            try:
                audio = sample["audio"]
                wav_path = out_dir / f"sonics_fake_{count:06d}.wav"
                if not wav_path.exists():
                    import soundfile as sf
                    audio_array = np.array(audio["array"], dtype=np.float32)
                    if audio["sampling_rate"] != SR:
                        import librosa
                        audio_array = librosa.resample(
                            audio_array, orig_sr=audio["sampling_rate"], target_sr=SR
                        )
                    sf.write(str(wav_path), np.clip(audio_array, -1, 1), SR)
                rows.append((str(wav_path), 1, "sonics"))
                count += 1
                if count % 500 == 0:
                    print(f"  SONICS: {count} files...")
            except Exception:
                continue
        
        return rows
    except Exception as e:
        print(f"SONICS download failed: {e}")
        return []


# ============================================================
# 4. LibriSpeech / LibriTTS-R — 음성 real
# ============================================================
def prepare_librispeech_real(out_dir, cache_dir=Path("_cache/librispeech"), max_samples=3000):
    """LibriSpeech test-clean에서 real 음성."""
    root = cache_dir / "LibriSpeech" / "test-clean"
    if not root.exists():
        print(f"LibriSpeech not found at {root}")
        return []
    
    flacs = sorted(root.rglob("*.flac"))
    if len(flacs) == 0:
        print(f"LibriSpeech: no .flac files found at {root}")
        return []
    
    rng = np.random.default_rng(42)
    sel = rng.choice(len(flacs), size=min(max_samples, len(flacs)), replace=False)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, idx in enumerate(sorted(sel)):
        src = flacs[int(idx)]
        wav_path = out_dir / f"librispeech_real_{i:06d}.wav"
        if not wav_path.exists():
            x = load_wav(src)
            save_wav(x[:int(10 * SR)], wav_path)  # 최대 10초
        rows.append((str(wav_path), 0, "librispeech"))
    
    return rows


def prepare_edge_tts_real(out_dir, max_samples=2000):
    """edge-tts로 실제 음성 데이터 생성 (fallback)."""
    try:
        import edge_tts
        import asyncio
    except ImportError:
        print("  edge-tts not installed")
        return []
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Various English voices for diversity
    voices = [
        "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
        "en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural",
        "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural",
        "en-IN-PrabhatNeural",
    ]
    
    # Short sentences for voice samples
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A stitch in time saves nine.",
        "Practice makes perfect.",
        "The early bird catches the worm.",
        "Actions speak louder than words.",
        "Knowledge is power.",
        "Time flies when you are having fun.",
        "All that glitters is not gold.",
        "Where there is a will, there is a way.",
        "The pen is mightier than the sword.",
        "Rome was not built in a day.",
        "Better late than never.",
        "The grass is always greener on the other side.",
        "Do not put all your eggs in one basket.",
        "Every cloud has a silver lining.",
    ]
    
    rows = []
    count = 0
    
    async def generate_one(text, voice, path):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(path))
    
    for vi, voice in enumerate(voices):
        if count >= max_samples:
            break
        for si, sent in enumerate(sentences):
            if count >= max_samples:
                break
            wav_path = out_dir / f"tts_real_{count:06d}.wav"
            mp3_path = out_dir / f"tts_real_{count:06d}.mp3"
            if wav_path.exists():
                rows.append((str(wav_path), 0, "edge_tts"))
                count += 1
                continue
            try:
                asyncio.run(generate_one(sent, voice, mp3_path))
                # Convert mp3 to wav
                import subprocess
                subprocess.run(["ffmpeg", "-y", "-i", str(mp3_path),
                              "-ar", str(SR), "-ac", "1", str(wav_path)],
                             capture_output=True, timeout=10)
                mp3_path.unlink(missing_ok=True)
                if wav_path.exists():
                    rows.append((str(wav_path), 0, "edge_tts"))
                    count += 1
            except Exception:
                mp3_path.unlink(missing_ok=True)
                continue
            if count % 200 == 0 and count > 0:
                print(f"  edge-tts: {count} files...")
    
    return rows


# ============================================================
# 5. MUSDB18 — 음악 real (보컬 없는 instrumental)
# ============================================================
def prepare_musdb_real(out_dir, cache_dir=Path("_cache/musdb"), max_samples=200):
    """MUSDB18 sample에서 drums+bass+other 합성 → 보컬 없는 음악 real."""
    ext_dir = cache_dir / "extracted"
    if not ext_dir.exists():
        zpath = cache_dir / "sample.zip"
        if zpath.exists():
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(ext_dir)
        else:
            return []
    
    stems_dirs = [d for d in sorted(ext_dir.rglob("*")) 
                  if d.is_dir() and (d / "other.wav").exists()]
    
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, d in enumerate(stems_dirs[:max_samples]):
        wav_path = out_dir / f"musdb_real_{i:06d}.wav"
        if not wav_path.exists():
            parts = []
            for stem in ["drums", "bass", "other"]:
                x, _ = __import__("soundfile").read(str(d / f"{stem}.wav"), dtype="float32")
                parts.append(x.mean(axis=1) if x.ndim > 1 else x)
            L = min(len(p) for p in parts)
            mix = np.sum(parts, axis=0)[:L] / len(parts)
            save_wav(mix.astype(np.float32), wav_path)
        rows.append((str(wav_path), 0, "musdb"))
    
    return rows


# ============================================================
# 6. FMA small — 음악 real
# ============================================================
def prepare_fma_real(out_dir, cache_dir=Path("_cache/fma"), max_samples=500):
    """FMA small zip에서 음악 real."""
    zpath = cache_dir / "fma_small.zip"
    extract_root = cache_dir / "fma_small_extracted"
    
    if not zpath.exists():
        print("FMA small zip not found")
        return []
    
    extract_root.mkdir(parents=True, exist_ok=True)
    
    # 일부 디렉터리만 추출
    want_dirs = {"000", "001", "002", "003", "004", "005"}
    if not any(extract_root.iterdir()):
        with zipfile.ZipFile(zpath) as zf:
            members = [
                m for m in zf.namelist()
                if m.endswith(".mp3") and m.split("/")[1][:3] in want_dirs
            ]
            for m in members[:max_samples * 2]:
                tgt = extract_root / m
                if not tgt.exists():
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as s, open(tgt, "wb") as d:
                        d.write(s.read())
    
    files = sorted(extract_root.rglob("*.mp3"))
    rng = np.random.default_rng(1)
    sel = rng.choice(len(files), size=min(max_samples, len(files)), replace=False)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, idx in enumerate(sorted(sel)):
        src = files[int(idx)]
        wav_path = out_dir / f"fma_real_{i:06d}.wav"
        if not wav_path.exists():
            x = load_wav(src)
            save_wav(x[:int(30 * SR)], wav_path)  # 최대 30초
        rows.append((str(wav_path), 0, "fma"))
    
    return rows


# ============================================================
# manifest 생성
# ============================================================
def write_manifest(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label", "source"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="0.8+ 학습 데이터 구축")
    ap.add_argument("--out", type=Path, default=Path("train_data"))
    ap.add_argument("--asvspoof-dir", type=Path, default=None,
                    help="ASVspoof 2019 LA 디렉토리 (없으면 skip)")
    ap.add_argument("--max-voice-real", type=int, default=3000)
    ap.add_argument("--max-voice-fake", type=int, default=5000)
    ap.add_argument("--max-music-real", type=int, default=700)
    ap.add_argument("--max-music-fake", type=int, default=3000)
    ap.add_argument("--auto-download", action="store_true",
                    help="HuggingFace에서 자동 다운로드")
    args = ap.parse_args()
    
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    voice_real_dir = out / "voice_real"
    voice_fake_dir = out / "voice_fake"
    music_real_dir = out / "music_real"
    music_fake_dir = out / "music_fake"
    
    print("=" * 60)
    print("Building training data for 0.8+ target")
    print("=" * 60)
    
    # ---- Voice real ----
    print("\n[1/4] Voice real data ...")
    vr = []
    vr += prepare_librispeech_real(voice_real_dir, max_samples=args.max_voice_real)
    print(f"  LibriSpeech real: {len(vr)} files")
    
    # Fallback: if no LibriSpeech, use edge-tts to generate real voice
    if len(vr) == 0 and args.auto_download:
        print("  LibriSpeech not available, using edge-tts fallback...")
        vr += prepare_edge_tts_real(voice_real_dir, max_samples=args.max_voice_real)
        print(f"  edge-tts real: {len(vr)} files")
    
    write_manifest(vr, out / "manifest_voice_real.csv")
    
    # ---- Voice fake ----
    print("\n[2/4] Voice fake data ...")
    vf = []

    # ASVspoof 2019 LA (핵심!) — 자동 다운로드
    if args.auto_download:
        asvspoof_cache = Path("_cache/asvspoof2019")
        if download_asvspoof_2019(asvspoof_cache):
            asv_rows = prepare_asvspoof_2019_la(asvspoof_cache, voice_fake_dir, args.max_voice_fake)
            vf += asv_rows
            print(f"  ASVspoof 2019 LA: {len(asv_rows)} files")
        else:
            print("  ASVspoof 2019 LA: download failed, skipping")
    elif args.asvspoof_dir:
        asv_rows = prepare_asvspoof_2019_la(args.asvspoof_dir, voice_fake_dir, args.max_voice_fake)
        vf += asv_rows
        print(f"  ASVspoof 2019 LA: {len(asv_rows)} files")

    # Codecfake
    if args.auto_download:
        remaining = args.max_voice_fake - len(vf)
        if remaining > 0:
            cf = prepare_codecfake(voice_fake_dir, max_samples=min(2000, remaining))
            vf += cf
            print(f"  Codecfake: {len(cf)} files")

    # Diverse TTS fakes (edge-tts로 다양한 화자/텍스트)
    if args.auto_download:
        remaining = args.max_voice_fake - len(vf)
        if remaining > 0:
            tts = prepare_diverse_tts_fakes(voice_fake_dir, max_samples=min(3000, remaining))
            vf += tts
            print(f"  Diverse TTS: {len(tts)} files")

    write_manifest(vf, out / "manifest_voice_fake.csv")
    
    # ---- Music real ----
    print("\n[3/4] Music real data ...")
    mr = []
    mr += prepare_musdb_real(music_real_dir, max_samples=min(200, args.max_music_real))
    print(f"  MUSDB18: {len([r for r in mr if 'musdb' in r[2]])} files")
    mr += prepare_fma_real(music_real_dir, max_samples=args.max_music_real - len(mr))
    print(f"  FMA: {len([r for r in mr if 'fma' in r[2]])} files")
    write_manifest(mr, out / "manifest_music_real.csv")
    
    # ---- Music fake ----
    print("\n[4/4] Music fake data ...")
    mf = []
    if args.auto_download:
        mf += prepare_sonics(music_fake_dir, max_samples=args.max_music_fake)
        print(f"  SONICS: {len(mf)} files")
    write_manifest(mf, out / "manifest_music_fake.csv")

    # ---- Data Augmentation (증강으로 데이터 다양화) ----
    print("\n[5/5] Data augmentation...")
    aug_dir = out / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)

    # Voice augmented copies
    for src_dir, label, name in [
        (voice_real_dir, 0, "voice_real"),
        (voice_fake_dir, 1, "voice_fake"),
    ]:
        src_wavs = sorted(src_dir.glob("*.wav"))
        if len(src_wavs) == 0:
            continue
        aug_out = aug_dir / name
        aug_out.mkdir(parents=True, exist_ok=True)
        existing = len(list(aug_out.glob("*.wav")))
        if existing < len(src_wavs) * 2:
            print(f"  Augmenting {name} ({len(src_wavs)} files)...")
            try:
                subprocess.run([
                    sys.executable, str(Path(__file__).parent / "augment_data.py"),
                    "--input", str(src_dir),
                    "--output", str(aug_out),
                    "--n-copies", "2",
                ], timeout=3600)
            except Exception as e:
                print(f"  Augmentation failed: {e}")

        # Add augmented files to rows
        for wav in sorted(aug_out.glob("*.wav")):
            if label == 0:
                vr.append((str(wav), 0, f"aug_{name}"))
            else:
                vf.append((str(wav), 1, f"aug_{name}"))

    print(f"  After augmentation: Voice real={len(vr)}, Voice fake={len(vf)}")

    # ---- 통합 manifest ----
    all_rows = []
    for rows_list in [vr, vf, mr, mf]:
        for filepath, label, source in rows_list:
            all_rows.append([filepath, label, source])
    
    write_manifest(all_rows, out / "manifest_all.csv")
    
    # 통합 manifest에서 train/val 분리
    if len(all_rows) == 0:
        print("\nERROR: No training data found! Check your data sources.")
        print("Required: LibriSpeech, ASVspoof, Codecfake, edge-tts, etc.")
        return
    
    from sklearn.model_selection import train_test_split
    all_for_split = [[r[0], r[1]] for r in all_rows]
    
    # Check if we have both classes for stratified split
    labels = [r[1] for r in all_for_split]
    has_both = 0 in labels and 1 in labels
    
    if has_both and len(all_for_split) >= 4:
        tr, va = train_test_split(all_for_split, test_size=0.15, random_state=42,
                                   stratify=labels)
    else:
        # Can't stratify with single class or too few samples
        tr, va = train_test_split(all_for_split, test_size=0.15, random_state=42)
    
    write_manifest([[r[0], r[1], ""] for r in tr], out / "manifest_train.csv")
    write_manifest([[r[0], r[1], ""] for r in va], out / "manifest_val.csv")
    
    print(f"\n{'=' * 60}")
    print(f"Total: {len(all_rows)} files")
    print(f"  Voice real: {len(vr)}, Voice fake: {len(vf)}")
    print(f"  Music real: {len(mr)}, Music fake: {len(mf)}")
    print(f"  Train: {len(tr)}, Val: {len(va)}")
    print(f"Output: {out}")
    print(f"  manifest_train.csv / manifest_val.csv")
    print(f"  manifest_voice_real.csv / manifest_voice_fake.csv")
    print(f"  manifest_music_real.csv / manifest_music_fake.csv")


if __name__ == "__main__":
    main()
