#!/usr/bin/env python3
"""데이터 증강 모듈 — Neural Codec, Phone Call Distortion, SpecAugment.

경쟁자들이 안 쓰는 고급 증강 기법:
1. Neural Codec Augmentation: 오디오를 다양한 코덱으로 재인코딩
2. Phone Call Distortion: 8kHz 전화선 시뮬레이션 (GSM, μ-law)
3. Room Impulse Response: 실제 방 잔향 시뮬레이션
4. Speed/Pitch Perturbation: 다양한 속도/피치 변환
5. Music Background Mixing: 음악 배경음 혼합

사용법:
  python scripts/augment_data.py --input train_data --output train_data_augmented
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


# ============================================================
# 1. Neural Codec Augmentation
# ============================================================

def codec_augment(audio, sr=SR, rng=None):
    """다양한 코덱으로 재인코딩 → codec artifact 생성."""
    if rng is None:
        rng = np.random.default_rng()

    codecs = ["mp3", "opus", "aac", "wma"]
    codec = rng.choice(codecs)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
        sf.write(tmp_in.name, audio, sr)

        try:
            if codec == "mp3":
                kbps = int(rng.choice([32, 48, 64, 96, 128]))
                out = tmp_in.name.replace('.wav', f'_codec.mp3')
                subprocess.run([
                    'ffmpeg', '-y', '-i', tmp_in.name,
                    '-codec:a', 'libmp3lame', '-b:a', f'{kbps}k',
                    '-ar', str(sr), out
                ], capture_output=True, timeout=10)
            elif codec == "opus":
                out = tmp_in.name.replace('.wav', f'_codec.ogg')
                subprocess.run([
                    'ffmpeg', '-y', '-i', tmp_in.name,
                    '-codec:a', 'libopus', '-b:a', '32k',
                    '-ar', str(sr), out
                ], capture_output=True, timeout=10)
            elif codec == "aac":
                out = tmp_in.name.replace('.wav', f'_codec.aac')
                subprocess.run([
                    'ffmpeg', '-y', '-i', tmp_in.name,
                    '-codec:a', 'aac', '-b:a', '64k',
                    '-ar', str(sr), out
                ], capture_output=True, timeout=10)
            elif codec == "wma":
                out = tmp_in.name.replace('.wav', f'_codec.wma')
                subprocess.run([
                    'ffmpeg', '-y', '-i', tmp_in.name,
                    '-codec:a', 'wmav2', '-b:a', '64k',
                    '-ar', str(sr), out
                ], capture_output=True, timeout=10)
            else:
                return audio

            if os.path.exists(out):
                augmented, _ = sf.read(out)
                os.unlink(out)
                os.unlink(tmp_in.name)
                return augmented.astype(np.float32)[:len(audio)]
            else:
                os.unlink(tmp_in.name)
                return audio

        except Exception:
            os.unlink(tmp_in.name)
            return audio


# ============================================================
# 2. Phone Call Distortion (8kHz, GSM, μ-law)
# ============================================================

def phone_call_distortion(audio, sr=SR, rng=None):
    """전화선 시뮬레이션: 8kHz 다운샘플 + 코덱 + 밴드패스."""
    if rng is None:
        rng = np.random.default_rng()

    try:
        import librosa

        # 1. 8kHz 다운샘플
        audio_8k = librosa.resample(
            audio.astype(np.float64), orig_sr=sr, target_sr=8000
        )

        # 2. 밴드패스 필터 (300Hz-3400Hz, 전화 주파수 대역)
        from scipy.signal import butter, sosfilt
        sos = butter(4, [300, 3400], btype='band', fs=8000, output='sos')
        audio_8k = sosfilt(sos, audio_8k).astype(np.float32)

        # 3. μ-law 압축 시뮬레이션
        if rng.random() < 0.5:
            mu = 255
            audio_8k = np.sign(audio_8k) * np.log(1 + mu * np.abs(audio_8k)) / np.log(1 + mu)

        # 4. 16kHz로 업샘플
        audio_out = librosa.resample(
            audio_8k.astype(np.float64), orig_sr=8000, target_sr=sr
        )

        # 5. 길이 맞추기
        if len(audio_out) < len(audio):
            audio_out = np.pad(audio_out, (0, len(audio) - len(audio_out)))
        audio_out = audio_out[:len(audio)]

        # 노이즈 추가 (전화선 노이즈)
        noise_level = rng.uniform(0.001, 0.01)
        audio_out += rng.normal(0, noise_level, len(audio_out)).astype(np.float32)

        return np.clip(audio_out, -1, 1).astype(np.float32)

    except Exception:
        return audio


# ============================================================
# 3. Room Impulse Response (실제 방 잔향)
# ============================================================

def room_impulse_response(audio, sr=SR, rng=None):
    """방 잔향 시뮬레이션 (여러 지연 + 감쇠)."""
    if rng is None:
        rng = np.random.default_rng()

    n_reflections = rng.integers(3, 8)
    output = np.zeros(len(audio) + sr, dtype=np.float64)  # 1초 여유

    for _ in range(n_reflections):
        delay_ms = rng.uniform(5, 80)
        delay_samples = int(delay_ms * sr / 1000)
        decay = rng.uniform(0.05, 0.4)
        # 고주파 감쇠 (실제 방)
        if rng.random() < 0.5:
            from scipy.signal import butter, sosfilt
            sos = butter(2, 2000, btype='low', fs=sr, output='sos')
            reflection = sosfilt(sos, audio) * decay
        else:
            reflection = audio * decay
        output[delay_samples:delay_samples + len(audio)] += reflection

    output[:len(audio)] += audio
    return output[:len(audio)].astype(np.float32)


# ============================================================
# 4. Speed/Pitch Perturbation
# ============================================================

def speed_perturbation(audio, sr=SR, rng=None):
    """속도 변환 ±15%."""
    if rng is None:
        rng = np.random.default_rng()
    try:
        import librosa
        rate = rng.uniform(0.85, 1.15)
        augmented = librosa.resample(
            audio.astype(np.float64), orig_sr=sr, target_sr=int(sr / rate)
        )
        if len(augmented) < len(audio):
            augmented = np.pad(augmented, (0, len(audio) - len(augmented)))
        return augmented[:len(audio)].astype(np.float32)
    except Exception:
        return audio


def pitch_shift(audio, sr=SR, rng=None):
    """피치 변환 ±3 스텝."""
    if rng is None:
        rng = np.random.default_rng()
    try:
        import librosa
        n_steps = rng.uniform(-3, 3)
        return librosa.effects.pitch_shift(
            audio.astype(np.float64), sr=sr, n_steps=n_steps
        ).astype(np.float32)
    except Exception:
        return audio


# ============================================================
# 5. Background Noise Mix
# ============================================================

def add_colored_noise(audio, snr_range=(5, 30), rng=None):
    """색 노이즈 첨가 (백색/분홍/ 갈색)."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(audio)
    snr_db = rng.uniform(*snr_range)

    # 색 노이즈 생성
    color = rng.choice(["white", "pink", "brown"])
    white = rng.normal(0, 1, n)

    if color == "pink":
        # Voss-McCartney 알고리즘 (간소화)
        from scipy.signal import lfilter
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004709510])
        a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
        noise = lfilter(b, a, white)
    elif color == "brown":
        from scipy.signal import lfilter
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004709510])
        a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
        brown = lfilter(b, a, white)
        noise = np.cumsum(brown)
        noise = noise / (np.max(np.abs(noise)) + 1e-10)
    else:
        noise = white

    ps = np.mean(audio.astype(np.float64) ** 2) + 1e-10
    pn = ps / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(pn / (np.mean(noise ** 2) + 1e-10))

    return np.clip(audio + noise.astype(np.float32), -1, 1).astype(np.float32)


# ============================================================
# 6. Random Crop + Pad
# ============================================================

def random_crop_pad(audio, target_len, rng=None):
    """랜덤 크롭 또는 패딩."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(audio)
    if n >= target_len:
        start = rng.integers(0, n - target_len + 1)
        return audio[start:start + target_len]
    else:
        reps = int(np.ceil(target_len / max(1, n)))
        return np.tile(audio, reps)[:target_len]


# ============================================================
# Batch Augmentation Pipeline
# ============================================================

AUGMENTATIONS = [
    ("codec", codec_augment),
    ("phone", phone_call_distortion),
    ("rir", room_impulse_response),
    ("speed", speed_perturbation),
    ("pitch", pitch_shift),
    ("noise", add_colored_noise),
]


def apply_augmentation_pipeline(audio, rng, n_augs=2):
    """파이프라인: n_augs개 랜덤 증강 순차 적용."""
    selected = rng.choice(len(AUGMENTATIONS), size=min(n_augs, len(AUGMENTATIONS)), replace=False)
    for idx in selected:
        name, aug_fn = AUGMENTATIONS[idx]
        try:
            audio = aug_fn(audio, SR, rng=rng)
        except Exception:
            pass
    return audio


def augment_directory(input_dir, output_dir, n_copies=2, target_len=64000):
    """디렉토리 내 모든 WAV 파일을 증강해서 복사."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob("*.wav"))
    print(f"Augmenting {len(wav_files)} files × {n_copies} copies...")

    rng = np.random.default_rng(42)
    count = 0

    for wav_path in wav_files:
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        # 원본 복사
        out_path = output_dir / wav_path.name
        if not out_path.exists():
            sf.write(str(out_path), random_crop_pad(audio, target_len, rng), SR)
        count += 1

        # 증강 복사
        for c in range(n_copies):
            aug_name = f"{wav_path.stem}_aug{c:02d}.wav"
            aug_path = output_dir / aug_name
            if aug_path.exists():
                continue
            aug_audio = apply_augmentation_pipeline(audio.copy(), rng, n_augs=rng.integers(1, 4))
            aug_audio = random_crop_pad(aug_audio, target_len, rng)
            peak = np.max(np.abs(aug_audio)) + 1e-9
            if peak > 0.99:
                aug_audio = aug_audio * (0.99 / peak)
            sf.write(str(aug_path), aug_audio, SR)
            count += 1

        if count % 1000 == 0:
            print(f"  {count} files processed...")

    print(f"Done! {count} files in {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="Data augmentation pipeline")
    ap.add_argument("--input", required=True, help="Input directory with WAV files")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--n-copies", type=int, default=2,
                    help="Number of augmented copies per file")
    ap.add_argument("--target-len", type=int, default=64000,
                    help="Target length in samples (4s @ 16kHz)")
    args = ap.parse_args()

    augment_directory(args.input, args.output, args.n_copies, args.target_len)


if __name__ == "__main__":
    main()
