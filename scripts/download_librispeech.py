#!/usr/bin/env python3
"""LibriSpeech test-clean 다운로드 → _cache/librispeech/LibriSpeech/test-clean"""
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip -q install datasets")
    from datasets import load_dataset

import numpy as np

out_dir = "_cache/librispeech/LibriSpeech/test-clean"

if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 10:
    print(f"LibriSpeech already exists at {out_dir}")
    sys.exit(0)

print("Downloading LibriSpeech test-clean from HuggingFace...")
os.makedirs(out_dir, exist_ok=True)

try:
    ds = load_dataset("librispeech_asr", "clean", split="test")
    print(f"Loaded {len(ds)} samples")
except Exception as e:
    print(f"librispeech_asr failed: {e}")
    print("Trying alternative...")
    ds = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test[:2000]")
    print(f"Loaded {len(ds)} samples from common_voice")

import soundfile as sf

count = 0
for i, s in enumerate(ds):
    spk = str(s.get("speaker_id", i // 100))
    ch = str(s.get("chapter_id", 0))
    sub = os.path.join(out_dir, spk, ch)
    os.makedirs(sub, exist_ok=True)
    aud = np.array(s["audio"]["array"], dtype=np.float32)
    fname = f"{spk}-{ch}-{i:04d}.flac"
    sf.write(os.path.join(sub, fname), aud, s["audio"]["sampling_rate"])
    count += 1
    if count >= 2000:
        break
    if count % 500 == 0:
        print(f"  {count} files downloaded...")

print(f"Done! Downloaded {count} files to {out_dir}")
