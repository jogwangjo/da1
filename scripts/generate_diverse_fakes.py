#!/usr/bin/env python3
"""다양한 TTS 엔진으로 fake 음성 데이터 생성.

핵심: 코드크파이크와 다른 공격 기법(TTS)으로 diverse fake 데이터 확보.
- edge-tts: Microsoft Azure 기반, 다양한 화자
- pyttsx3: 로컬 TTS (오프라인)
- gTTS: Google TTS

대회에서_codecfake만으로는 부족 → 다양한 TTS 공격으로 robustness 확보.

사용법:
  python scripts/generate_diverse_fakes.py --out train_data/voice_fake --max 3000
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000

# edge-tts 화자 목록 (다양한 성별/나이/억양)
EDGE_TTS_VOICES = [
    # American English
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural",
    "en-US-DavisNeural", "en-US-JaneNeural", "en-US-JasonNeural",
    "en-US-SaraNeural", "en-US-TonyNeural", "en-US-NancyNeural",
    "en-US-AmberNeural", "en-US-AvaNeural", "en-US-BrandonNeural",
    "en-US-ChristopherNeural", "en-US-CoraNeural", "en-US-ElizabethNeural",
    # British English
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural",
    "en-GB-ThomasNeural",
    # Australian English
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    # Indian English
    "en-IN-PrabhatNeural", "en-IN-NeerjaNeural",
    # Other
    "en-IE-ConnorNeural", "en-ZA-LeahNeural", "en-NG-AbeoNeural",
]

# 다양한 텍스트 주제 (자연스러운 문장)
TOPICS = [
    # 일상 대화
    "Good morning, how are you doing today?",
    "The weather is really nice outside, don't you think?",
    "I went to the grocery store yesterday and bought some fresh vegetables.",
    "Have you seen the new movie that came out last week?",
    "My cat loves to sleep on the warm couch in the afternoon.",
    "Could you pass me the salt, please?",
    "I think we should take a different route to avoid traffic.",
    "The concert was absolutely amazing last night.",
    "She told me she would be here by three o'clock.",
    "We need to finish this project before the deadline on Friday.",
    # 비즈니스
    "The quarterly earnings report shows a fifteen percent increase.",
    "Please submit your expense reports by the end of the month.",
    "Our team is working on a new product launch for next quarter.",
    "The client meeting has been rescheduled to next Tuesday.",
    "We should analyze the market trends before making a decision.",
    "The presentation went very well and the board approved the budget.",
    "I would like to schedule a call with the development team.",
    "Our company is expanding into new international markets.",
    "The new hiring process will begin next month.",
    "Customer satisfaction scores have improved significantly this year.",
    # 뉴스/정보
    "Scientists have discovered a new species of deep sea fish.",
    "The government announced new policies for renewable energy.",
    "SpaceX successfully launched another batch of satellites today.",
    "The stock market reached an all time high yesterday.",
    "Researchers are developing a new treatment for rare diseases.",
    "The city council approved the construction of a new park.",
    "A major earthquake was recorded off the coast of Japan.",
    "The education department released new curriculum guidelines.",
    "Electric vehicle sales have doubled compared to last year.",
    "The international summit focused on climate change solutions.",
    # 스토리텔링
    "Once upon a time there was a wise old man living in the mountains.",
    "The brave knight rode through the enchanted forest at dawn.",
    "She opened the mysterious box and found a golden key inside.",
    "The detective carefully examined the evidence at the crime scene.",
    "Years later he returned to the village where he grew up.",
    "The children laughed as they played in the autumn leaves.",
    "He gazed at the stars and wondered about distant galaxies.",
    "The old library was filled with thousands of ancient books.",
    "They traveled across the desert following the guidance of maps.",
    "The garden bloomed with colorful flowers every spring.",
    # 기술/과학
    "Machine learning models require large datasets for training.",
    "The new processor delivers significantly better performance.",
    "Quantum computing could revolutionize cryptography in the future.",
    "The satellite orbits the earth at an altitude of four hundred kilometers.",
    "Artificial intelligence is transforming the healthcare industry.",
    "The research team published their findings in a leading journal.",
    "Deep learning algorithms can recognize patterns in complex data.",
    "The experiments confirmed the hypothesis about protein folding.",
    "Blockchain technology provides secure and transparent transactions.",
    "The telescope captured images of galaxies billions of light years away.",
    # 감정/설명
    "I really enjoy spending time with my family during holidays.",
    "Learning a new language takes patience and consistent practice.",
    "The sunset painted the sky in beautiful shades of orange and pink.",
    "Music has the power to change your mood and lift your spirits.",
    "Traveling to new places broadens your perspective on life.",
    "A good book can transport you to another world entirely.",
    "Cooking is both an art and a science that requires precision.",
    "The sound of rain on the roof helps me fall asleep at night.",
    "Exercise is essential for maintaining both physical and mental health.",
    "Friendship is one of the most valuable things in life.",
    # 긴 문장 (발화 길이 다양화)
    "In the early morning hours, before the sun had fully risen, "
    "she made her way through the quiet streets toward the train station, "
    "carrying only a small bag and a heart full of memories.",
    "The laboratory was filled with advanced equipment and the scientists "
    "worked diligently to complete their research before the grant deadline.",
    "Despite the challenges they faced during the project, the team "
    "remained optimistic and continued to push forward with determination.",
    "The ancient ruins were discovered by a team of archaeologists "
    "who had been exploring the remote region for several months.",
    "As the train pulled into the station, he could see her standing "
    "on the platform, waving enthusiastically with a bright smile.",
]


def generate_edge_tts(text, voice, output_path):
    """edge-tts로 음성 생성."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)

        async def _gen():
            await communicate.save(str(output_path))

        asyncio.run(_gen())
        return True
    except Exception as e:
        print(f"  edge-tts error: {e}")
        return False


def mp3_to_wav(mp3_path, wav_path, sr=SR):
    """MP3 → WAV 변환."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(mp3_path),
            "-ar", str(sr), "-ac", "1", "-sample_fmt", "s16",
            str(wav_path)
        ], capture_output=True, timeout=10)
        return wav_path.exists()
    except Exception:
        return False


def apply_random_augmentation(audio, rng):
    """TTS 생성 음성에 랜덤 증강 적용 (자연스러움 추가)."""
    import librosa

    # 1. 랜덤 피치 시프트
    if rng.random() < 0.3:
        n_steps = rng.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(
            audio.astype(np.float64), sr=SR, n_steps=n_steps
        ).astype(np.float32)

    # 2. 랜덤 속도 변환
    if rng.random() < 0.3:
        rate = rng.uniform(0.9, 1.1)
        augmented = librosa.resample(
            audio.astype(np.float64), orig_sr=SR, target_sr=int(SR / rate)
        )
        if len(augmented) < len(audio):
            augmented = np.pad(augmented, (0, len(audio) - len(augmented)))
        audio = augmented[:len(audio)].astype(np.float32)

    # 3. 배경 노이즈 (약간)
    if rng.random() < 0.4:
        snr_db = rng.uniform(20, 35)
        ps = np.mean(audio.astype(np.float64) ** 2) + 1e-10
        pn = ps / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(pn), len(audio))
        audio = (audio + noise.astype(np.float32))

    # 4. 랜덤 gain
    if rng.random() < 0.3:
        gain = rng.uniform(0.7, 1.3)
        audio = audio * gain

    # 정규화
    peak = np.max(np.abs(audio)) + 1e-9
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    return np.clip(audio, -1, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Multi-TTS diverse fake data generator")
    ap.add_argument("--out", type=Path, default=Path("train_data/voice_fake"))
    ap.add_argument("--max", type=int, default=3000, help="Max files to generate")
    ap.add_argument("--offset", type=int, default=0,
                    help="Start index (for resuming)")
    ap.add_argument("--augment", action="store_true", default=True,
                    help="Apply random augmentation to TTS output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Already generated files
    existing = len(list(args.out.glob("tts_fake_*.wav")))
    if existing >= args.max:
        print(f"Already have {existing} files, skipping.")
        return

    start_idx = max(args.offset, existing)
    remaining = args.max - start_idx
    print(f"Generating {remaining} diverse TTS fake files (starting from {start_idx})...")

    count = 0
    total_attempts = 0
    max_attempts = remaining * 3  # Allow retries

    while count < remaining and total_attempts < max_attempts:
        total_attempts += 1

        # Random topic and voice
        text = rng.choice(TOPICS)
        voice = rng.choice(EDGE_TTS_VOICES)

        idx = start_idx + count
        wav_path = args.out / f"tts_fake_{idx:06d}.wav"

        if wav_path.exists():
            count += 1
            continue

        with tempfile.TemporaryDirectory() as td:
            mp3_path = Path(td) / "temp.mp3"

            if generate_edge_tts(text, voice, mp3_path):
                if mp3_to_wav(mp3_path, Path(td) / "temp.wav"):
                    audio, _ = sf.read(Path(td) / "temp.wav", dtype="float32")

                    # Resample if needed
                    if len(audio) > 0:
                        if args.augment:
                            audio = apply_random_augmentation(audio, rng)

                        # Trim or pad to reasonable length (2-10 sec)
                        min_len = int(2 * SR)
                        max_len = int(10 * SR)
                        if len(audio) < min_len:
                            reps = int(np.ceil(min_len / max(1, len(audio))))
                            audio = np.tile(audio, reps)[:min_len]
                        elif len(audio) > max_len:
                            # Random crop
                            start = rng.integers(0, len(audio) - max_len)
                            audio = audio[start:start + max_len]

                        sf.write(str(wav_path), audio, SR)
                        count += 1

                        if count % 200 == 0:
                            print(f"  Generated {count}/{remaining} files...")

    print(f"Done! Generated {count} diverse TTS fake files in {args.out}")
    print(f"  Total attempts: {total_attempts}, Success rate: {count/max(1,total_attempts):.1%}")


if __name__ == "__main__":
    main()
