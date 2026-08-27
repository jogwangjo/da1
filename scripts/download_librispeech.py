#!/usr/bin/env python3
"""LibriSpeech test-clean download — streaming mode (no disk cache)."""
import os, sys

out_dir = "_cache/librispeech/LibriSpeech/test-clean"

if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 100:
    print(f"LibriSpeech already exists at {out_dir} ({len(os.listdir(out_dir))} files)")
    sys.exit(0)

print("Downloading LibriSpeech test-clean (streaming)...")
os.makedirs(out_dir, exist_ok=True)

from datasets import load_dataset
import soundfile as sf
import numpy as np

# Streaming mode: doesn't cache the entire dataset to disk
ds = load_dataset("openslr/librispeech_asr", "default", split="test",
                  revision="refs/convert/parquet", streaming=True)

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
    if count >= 2500:
        break
    if count % 500 == 0:
        print(f"  {count} files downloaded...")

print(f"Done! Downloaded {count} files to {out_dir}")
