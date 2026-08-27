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
# 1. ASVspoof 2019 LA — 음성 fake 학습의 핵심
# ============================================================
def prepare_asvspoof_2019_la(asvspoof_dir, out_dir, max_per_class=5000):
    """ASVspoof 2019 LA train/dev → manifest."""
    asvspoof_dir = Path(asvspoof_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    # flac 디렉토리
    flac_dir = asvspoof_dir / "flac"
    if not flac_dir.exists():
        # zip 파일이 있으면 풀기
        for ext in ["zip", "tar.gz", "tgz"]:
            cand = asvspoof_dir / f"ASVspoof2019_LA_{ext}"
            if cand.exists():
                print(f"extracting {cand} ...")
                if ext == "zip":
                    with zipfile.ZipFile(cand) as zf:
                        zf.extractall(asvspoof_dir)
                else:
                    subprocess.run(["tar", "xzf", str(cand)], cwd=asvspoof_dir, check=True)
                break
    
    # label 파일 파싱
    label_files = list(asvspoof_dir.rglob("*.txt"))
    label_file = None
    for lf in label_files:
        content = lf.read_text()
        if "bonafide" in content or "spoof" in content:
            label_file = lf
            break
    
    if label_file is None:
        print(f"WARNING: label file not found in {asvspoof_dir}, using flac listing")
        flacs = sorted(flac_dir.glob("*.flac")) if flac_dir.exists() else []
        real_count, fake_count = 0, 0
        for f in flacs:
            # 파일명 규칙: LA_D_XXX_YYYY.flac → D=DEV, T=TRAIN
            parts = f.stem.split("_")
            if len(parts) >= 3 and parts[1] == "T":  # train partition
                if real_count < max_per_class:
                    rows.append((str(f), 0, "asvspoof19_train_real"))
                    real_count += 1
                elif fake_count < max_per_class * 2:
                    rows.append((str(f), 1, "asvspoof19_train_fake"))
                    fake_count += 1
        return rows
    
    # label 파싱: utterance_label bonafide/spoof
    real_count, fake_count = 0, 0
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id, label = parts[0], parts[1]
            # flac 파일 경로
            flac_path = flac_dir / f"{utt_id}.flac"
            if not flac_path.exists():
                continue
            
            is_fake = 1 if label == "spoof" else 0
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
    if args.asvspoof_dir:
        vf += prepare_asvspoof_2019_la(args.asvspoof_dir, voice_fake_dir, args.max_voice_fake)
        print(f"  ASVspoof 2019 LA: {len([r for r in vf if 'asvspoof' in r[2]])} files")
    
    if args.auto_download:
        cf = prepare_codecfake(voice_fake_dir, max_samples=min(2000, args.max_voice_fake - len(vf)))
        vf += cf
        print(f"  Codecfake: {len(cf)} files")
    
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
