#!/usr/bin/env python3
"""pseudo-eval 구축: 대회 테스트 분포를 모사하는 라벨 있는 검증셋 생성.

트랙 구성
  refs/voice  : 음성 단독 클립 (real: LibriSpeech / fake: edge-tts, MMS-TTS-ko)
  refs/music  : 음악 단독 클립 (real: FMA, MUSDB18-sample stems / fake: MusicGen)
  test/       : 평가용 파일 (voice-only, music-only, song=SONICS, mixed) + 채널 FX
  labels.csv  : ID, FILE_FAKE, VOICE_FAKE, MUSIC_FAKE, VOICE_PRESENT, MUSIC_PRESENT + 메타
  refs_voice.csv / refs_music.csv : 컴포넌트 트랙 라벨

사용법 (Colab 권장):
  python scripts/build_pseudo_eval.py --out pseudo_eval --seed 42 [--skip-musicgen]
"""
import argparse
import asyncio
import csv
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np

SR = 16000
SENTENCES_KO = [
    "안녕하세요, 저는 한국에서 태어났습니다.",
    "오늘 날씨가 정말 좋네요. 산책 가실래요?",
    "계좌 번호를 알려주시면 송금해 드리겠습니다.",
    "긴급합니다. 지금 바로 전화 좀 받아 주세요.",
    "저는 은행 직원입니다. 본인 확인이 필요합니다.",
    "이번 주말에 부모님 댁에 다녀올 예정입니다.",
    "회의가 30분 정도 늦어질 것 같습니다.",
    "수고하셨습니다. 내일 다시 연락드리겠습니다.",
    "그 가게 음식이 정말 맛있었어요. 추천합니다.",
    "지하철 2호선을 타고 시청역에서 내리세요.",
    "여권을 분실해서 대사관에 방문해야 합니다.",
    "아이가 학교에서 돌아올 시간이 됐네요.",
    "건강을 위해 매일 아침 조깅을 시작했습니다.",
    "이 책은 인공지능의 역사를 다루고 있습니다.",
    "택배 기사님께서 문 앞에 두고 가셨어요.",
    "다음 주에 이사를 하게 되어 정신이 없습니다.",
]
SENTENCES_EN = [
    "Hello, I was born in Seoul and moved here last year.",
    "The weather is beautiful today, would you like to go for a walk?",
    "Please confirm your identity before we proceed with the transfer.",
    "This is an urgent matter, please pick up the phone right now.",
    "I work at the bank and I need to verify your account details.",
    "The meeting will probably be delayed by about thirty minutes.",
    "Thank you for your help, I will contact you again tomorrow.",
    "I lost my passport and I have to visit the embassy tomorrow.",
    "My child comes home from school around four in the afternoon.",
    "I started jogging every morning to improve my health.",
    "This book covers the modern history of artificial intelligence.",
    "We are moving to a new apartment next week, it is very busy.",
    "Take the subway line two and get off at city hall station.",
    "The delivery driver left the package in front of the door.",
    "She said the restaurant near the harbor was excellent.",
    "He is preparing for the exam that starts early next month.",
]
MUSICGEN_PROMPTS = [
    "upbeat pop song with guitar and drums",
    "smooth jazz piano trio",
    "electronic dance music with heavy bass",
    "classical string quartet",
    "acoustic folk ballad",
    "hip hop beat with vinyl texture",
    "cinematic orchestral theme",
    "lo-fi chillhop groove",
]


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{r.stderr[-2000:]}")


def have_ffmpeg():
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------- audio io --
def load_wav(path, sr=SR):
    import librosa

    x, _ = librosa.load(str(path), sr=sr, mono=True)
    return x.astype(np.float32)


def save_wav(x, path):
    import soundfile as sf

    sf.write(str(path), np.clip(x, -1, 1), SR, subtype="PCM_16")


def trim_or_pad(x, dur, rng=None, mode="mid"):
    n = int(dur * SR)
    if len(x) >= n:
        if mode == "rand" and rng is not None:
            s = int(rng.integers(0, len(x) - n + 1))
            return x[s : s + n]
        s = (len(x) - n) // 2
        return x[s : s + n]
    reps = int(np.ceil(n / max(1, len(x))))
    return np.tile(x, reps)[:n]


# ---------------------------------------------------------------- audio FX --
def fx_mp3(x, kbps, tmpdir):
    src = Path(tmpdir) / "_s.wav"
    dst = src.with_suffix(".mp3")
    save_wav(x, src)
    sh(f'ffmpeg -y -loglevel error -i "{src}" -b:a {kbps}k "{dst}"')
    out = load_wav(dst)
    src.unlink(missing_ok=True)
    dst.unlink(missing_ok=True)
    return out


def fx_telephone(x):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
    y = sosfiltfilt(sos, x)
    return np.tanh(1.6 * y).astype(np.float32)


def fx_mulaw(x):
    mu = 255.0
    enc = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round(enc * 127) / 127
    return (np.sign(q) * (np.power(1 + mu, np.abs(q)) - 1) / mu).astype(np.float32)


def fx_noise(x, snr_db, rng):
    p_sig = np.mean(x**2) + 1e-10
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (x + rng.normal(0, np.sqrt(p_noise), len(x))).astype(np.float32)


def fx_tilt(x, rng):
    from scipy.signal import butter, sosfiltfilt

    if rng.random() < 0.5:
        sos = butter(2, 4000, btype="lowpass", fs=SR, output="sos")
    else:
        sos = butter(2, 150, btype="highpass", fs=SR, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def apply_fx(x, rng, tmpdir, p=0.5):
    applied = []
    if have_ffmpeg() and rng.random() < p:
        kind = str(rng.choice(["mp3", "tel", "mulaw", "noise", "tilt", "mp3+noise", "tel+mp3"]))
        if "mp3" in kind:
            kbps = int(rng.choice([32, 48, 64, 96, 128]))
            x = fx_mp3(x, kbps, tmpdir)
            applied.append(f"mp3-{kbps}k")
        if "tel" in kind:
            x = fx_telephone(x)
            applied.append("tel")
        if "mulaw" in kind:
            x = fx_mulaw(x)
            applied.append("mulaw")
        if "noise" in kind:
            x = fx_noise(x, float(rng.uniform(8, 25)), rng)
            applied.append("noise")
        if "tilt" in kind:
            x = fx_tilt(x, rng)
            applied.append("tilt")
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32), "+".join(applied) if applied else "none"


# ---------------------------------------------------------------- sources ---
def fetch_librispeech(out_dir, n, cache=Path("_cache/librispeech")):
    """real voice. openslr test-clean (346MB)."""
    idx_file = out_dir / "sources_librispeech.txt"
    if idx_file.exists():
        paths = [Path(p) for p in idx_file.read_text().splitlines()]
    else:
        tgz = cache / "test-clean.tar.gz"
        tgz.parent.mkdir(parents=True, exist_ok=True)
        if not tgz.exists():
            print("downloading LibriSpeech test-clean ...")
            subprocess.run(
                ["curl", "-L", "-o", str(tgz),
                 "https://www.openslr.org/resources/12/test-clean.tar.gz"],
                check=True,
            )
        root = cache / "LibriSpeech"
        if not root.exists():
            with tarfile.open(tgz) as tf:
                tf.extractall(cache)
        flacs = sorted(root.rglob("*.flac"))
        rng = np.random.default_rng(0)
        sel = rng.choice(len(flacs), size=min(n, len(flacs)), replace=False)
        paths = [flacs[int(i)] for i in sorted(sel)]
        idx_file.write_text("\n".join(map(str, paths)))
    return paths


async def _edge_one(text, voice, out_path, rate="+0%"):
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate).save(str(out_path))


def gen_edge_tts(out_dir, n, seed=42):
    """fake voice: MS neural TTS (ko/en)."""
    rng = np.random.default_rng(seed)
    ko_voices = ["ko-KR-InJoonNeural", "ko-KR-SunHiNeural"]
    en_voices = ["en-US-GuyNeural", "en-US-AriaNeural", "en-GB-RyanNeural"]
    voices = ko_voices * 2 + en_voices
    out_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(out_dir.glob("edge_*.wav"))
    if len(done) >= n:
        return done[:n]

    async def run_all():
        jobs = []
        for i in range(n):
            f = out_dir / f"edge_{i:04d}.wav"
            if f.exists():
                continue
            voice = voices[i % len(voices)]
            sents = SENTENCES_KO if "ko" in voice else SENTENCES_EN
            text = " ".join(rng.choice(sents, size=rng.integers(2, 4)))
            rate = rng.choice(["-10%", "+0%", "+15%"])
            jobs.append(_edge_one(text, voice, f, rate))
        await asyncio.gather(*jobs)

    asyncio.run(run_all())
    return sorted(out_dir.glob("edge_*.wav"))[:n]


def gen_mms_kor(out_dir, n):
    """fake voice: facebook/mms-tts-kor (VITS)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(out_dir.glob("mms_*.wav"))
    if len(done) >= n:
        return done[:n]
    import torch
    from transformers import VitsModel, AutoTokenizer

    model = VitsModel.from_pretrained("facebook/mms-tts-kor").eval()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
    rng = np.random.default_rng(43)
    for i in range(n):
        f = out_dir / f"mms_{i:04d}.wav"
        if f.exists():
            continue
        text = " ".join(rng.choice(SENTENCES_KO, size=int(rng.integers(2, 4))))
        inputs = tok(text, return_tensors="pt")
        with torch.inference_mode():
            wav = model(inputs["input_values"], attention_mask=None).waveform
        save_wav(wav.squeeze().float().numpy(), f)
    return sorted(out_dir.glob("mms_*.wav"))[:n]


def fetch_fma_small(out_dir, n, cache=Path("_cache/fma")):
    """real music: FMA small zip에서 일부 디렉터리만 추출 후 mid-segment."""
    idx_file = out_dir / "sources_fma.txt"
    if idx_file.exists():
        return [Path(p) for p in idx_file.read_text().splitlines()]
    zpath = cache / "fma_small.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    extract_root = cache / "fma_small_extracted"
    if not zpath.exists():
        print("downloading FMA small (7.2GB) ...")
        subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(zpath),
             "https://os.unil.cloud.switch.ch/fma/fma_small.zip"],
            check=True,
        )
    extract_root.mkdir(parents=True, exist_ok=True)
    want_dirs = {"000", "001", "002", "003"}
    with zipfile.ZipFile(zpath) as zf:
        members = [
            m for m in zf.namelist()
            if m.endswith(".mp3") and m.split("/")[1][:3] in want_dirs
        ]
        for m in members:
            tgt = extract_root / m
            if not tgt.exists():
                tgt.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as s, open(tgt, "wb") as d:
                    d.write(s.read())
    files = sorted(extract_root.rglob("*.mp3"))
    rng = np.random.default_rng(1)
    sel = rng.choice(len(files), size=min(n, len(files)), replace=False)
    paths = [files[int(i)] for i in sorted(sel)]
    idx_file.write_text("\n".join(map(str, paths)))
    return paths


def fetch_musdb_sample_stems(out_dir, cache=Path("_cache/musdb")):
    """real music(보컬 없음 보장): MUSDB18 sample stems 중 drums+bass+other 합성."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(out_dir.glob("musdb_*.wav"))
    if done:
        return done
    zpath = cache / "sample.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    if not zpath.exists():
        print("downloading MUSDB18 sample ...")
        subprocess.run(
            ["curl", "-L", "-o", str(zpath),
             "https://zenodo.org/records/3338373/files/sample.zip?download=1"],
            check=True,
        )
    ext = cache / "extracted"
    if not ext.exists():
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(ext)
    stems_dirs = [d for d in sorted(ext.rglob("*")) if d.is_dir() and (d / "other.wav").exists()]
    i = 0
    for d in stems_dirs:
        parts = []
        for stem in ["drums", "bass", "other"]:
            x, _ = __import__("soundfile").read(str(d / f"{stem}.wav"), dtype="float32")
            parts.append(x.mean(axis=1) if x.ndim > 1 else x)
        L = min(len(p) for p in parts)
        mix = np.sum(parts, axis=0)[:L] / len(parts)
        save_wav(mix.astype(np.float32), out_dir / f"musdb_{i:03d}.wav")
        i += 1
    return sorted(out_dir.glob("musdb_*.wav"))


def fetch_sonics_fakes(out_dir, n, repo="awsaf49/sonics"):
    """fake song(Suno/Udio end-to-end): SONICS HF dataset에서 오디오 파일만 n개."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(p for p in out_dir.iterdir() if p.suffix.lower() in {".wav", ".mp3", ".flac"})
    if len(done) >= n:
        return done[:n]
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    audio = [f for f in files if Path(f).suffix.lower() in {".wav", ".mp3", ".flac"}]
    audio.sort()
    step = max(1, len(audio) // n)
    picked = audio[::step][:n]
    got = []
    for f in picked:
        local = hf_hub_download(repo, f, repo_type="dataset")
        tgt = out_dir / Path(f).name
        if not tgt.exists():
            shutil.copy(local, tgt)
        got.append(tgt)
    return got


def gen_musicgen(out_dir, n, device="cuda"):
    """fake music: transformers musicgen-small."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(out_dir.glob("musicgen_*.wav"))
    if len(done) >= n:
        return done[:n]
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small"
    ).to(device)
    sr_model = model.config.audio_encoder.sampling_rate
    for i in range(n):
        f = out_dir / f"musicgen_{i:03d}.wav"
        if f.exists():
            continue
        prompt = MUSICGEN_PROMPTS[i % len(MUSICGEN_PROMPTS)]
        inputs = proc(text=[prompt], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            wav = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                                 max_new_tokens=512)
        y = wav[0, 0].float().cpu().numpy()
        if sr_model != SR:
            import librosa

            y = librosa.resample(y, orig_sr=sr_model, target_sr=SR)
        save_wav(y.astype(np.float32), f)
    return sorted(out_dir.glob("musicgen_*.wav"))[:n]


# ------------------------------------------------------------- composition --
def overlay(voice, music, rng, mode="overlap"):
    n = max(len(voice), len(music))
    v = np.zeros(n, dtype=np.float32)
    m = np.zeros(n, dtype=np.float32)
    v[: len(voice)] += voice
    if mode == "seq" and len(voice) < n:
        # 음성 -> 음악 순차 배치
        off = len(voice)
        m[off : off + len(music)] += music[: n - off]
    else:
        g = float(rng.uniform(0.35, 0.7))
        m[: len(music)] += g * music
    y = 0.95 * v + m
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("pseudo_eval"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-voice-real", type=int, default=180)
    ap.add_argument("--n-voice-fake-edge", type=int, default=50)
    ap.add_argument("--n-voice-fake-mms", type=int, default=30)
    ap.add_argument("--n-music-real-fma", type=int, default=120)
    ap.add_argument("--n-music-real-musdb", type=int, default=14)
    ap.add_argument("--n-song-sonics", type=int, default=80)
    ap.add_argument("--n-music-fake-musicgen", type=int, default=0)
    ap.add_argument("--n-mix", type=int, default=140)
    ap.add_argument("--fx-prob", type=float, default=0.45)
    ap.add_argument("--skip-musicgen", action="store_true")
    ap.add_argument("--force", action="store_true", help="labels.csv가 있어도 재생성")
    args = ap.parse_args()

    if (args.out / "labels.csv").exists() and not args.force:
        print("labels.csv already exists. use --force to rebuild.")
        return

    rng = np.random.default_rng(args.seed)
    out = args.out
    (out / "refs" / "voice").mkdir(parents=True, exist_ok=True)
    (out / "refs" / "music").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    tmpdir = out / "_tmp"
    tmpdir.mkdir(exist_ok=True)

    # ---- refs/voice
    vr_src = fetch_librispeech(out, args.n_voice_real)
    vf_edge = gen_edge_tts(out / "_src_edge", args.n_voice_fake_edge, args.seed)
    vf_mms = gen_mms_kor(out / "_src_mms", args.n_voice_fake_mms)

    rows_voice = []
    i = 0
    for src in vr_src:
        x = trim_or_pad(load_wav(src), float(rng.uniform(4, 10)), rng, "rand")
        p = out / "refs" / "voice" / f"V{i:05d}.wav"
        save_wav(x, p)
        rows_voice.append(("V%05d" % i, 0, "librispeech"))
        i += 1
    j = 0
    for src in list(vf_edge) + list(vf_mms):
        name = "E%05d" % j if "edge" in src.stem else "M%05d" % j
        x = trim_or_pad(load_wav(src), float(rng.uniform(4, 10)), rng, "rand")
        save_wav(x, out / "refs" / "voice" / f"{name}.wav")
        rows_voice.append((name, 1, "edge" if name.startswith("E") else "mms"))
        j += 1
    with open(out / "refs_voice.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "FAKE", "SOURCE"])
        w.writerows(rows_voice)

    # ---- refs/music
    mr_fma = fetch_fma_small(out, args.n_music_real_fma)
    mr_musdb = fetch_musdb_sample_stems(out / "_src_musdb") if args.n_music_real_musdb > 0 else []
    mf_musicgen = []
    if args.n_music_fake_musicgen > 0 and not args.skip_musicgen:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mf_musicgen = gen_musicgen(out / "_src_musicgen", args.n_music_fake_musicgen, dev)

    rows_music = []
    i = 0
    for src in mr_fma:
        x = trim_or_pad(load_wav(src), 10.0, rng, "rand")
        save_wav(x, out / "refs" / "music" / f"R{i:05d}.wav")
        rows_music.append(("R%05d" % i, 0, "fma"))
        i += 1
    for src in mr_musdb:
        if i >= args.n_music_real_fma + args.n_music_real_musdb:
            break
        x = trim_or_pad(load_wav(src), 10.0, rng, "rand")
        save_wav(x, out / "refs" / "music" / f"R{i:05d}.wav")
        rows_music.append(("R%05d" % i, 0, "musdb"))
        i += 1
    j = 0
    for src in mf_musicgen:
        x = trim_or_pad(load_wav(src), 10.0, rng, "rand")
        save_wav(x, out / "refs" / "music" / f"G{j:05d}.wav")
        rows_music.append(("G%05d" % j, 1, "musicgen"))
        j += 1
    with open(out / "refs_music.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "FAKE", "SOURCE"])
        w.writerows(rows_music)

    # ---- song (SONICS): 혼합으로 취급 (vocals=음성 fake, 반주=음악 fake)
    sonics = fetch_sonics_fakes(out / "_src_sonics", args.n_song_sonics)

    # ---- test 조립 목록 생성
    voice_pool = [(f, fk, s) for f, fk, s in rows_voice]
    music_pool = [(f, fk, s) for f, fk, s in rows_music]
    tasks = []  # dict: v,m,vf,mf,kind
    for f, fk, _ in voice_pool:
        tasks.append(dict(v=f, vf=fk, m=None, mf=0, kind="voice"))
    for f, fk, _ in music_pool:
        tasks.append(dict(v=None, vf=0, m=f, mf=fk, kind="music"))
    for p in sonics:
        tasks.append(dict(v=None, vf=0, m=p.name, mf=1, kind="song"))
    ids_v = [t for t in tasks if t["kind"] == "voice"]
    ids_m = [t for t in tasks if t["kind"] in ("music", "song")]
    per_mix = max(1, args.n_mix // 4)
    
    # Avoid empty pool errors
    v_real = [t for t in ids_v if t["vf"] == 0]
    v_fake = [t for t in ids_v if t["vf"] == 1]
    m_real = [t for t in ids_m if t["mf"] == 0]
    m_fake = [t for t in ids_m if t["mf"] == 1]
    
    picks = []
    if v_real and m_real:
        picks += [(rng.choice(v_real), rng.choice(m_real)) for _ in range(per_mix)]
    if v_fake and m_real:
        picks += [(rng.choice(v_fake), rng.choice(m_real)) for _ in range(per_mix)]
    if v_real and m_fake:
        picks += [(rng.choice(v_real), rng.choice(m_fake)) for _ in range(per_mix)]
    if v_fake and m_fake:
        picks += [(rng.choice(v_fake), rng.choice(m_fake)) for _ in range(per_mix)]
    for tv, tm in picks:
        tasks.append(dict(v=tv["v"], vf=tv["vf"], m=tm["m"], mf=tm["mf"], kind="mix"))

    # ---- render
    labels = []
    ref_voice_audio = {r[0]: None for r in rows_voice}
    for r_id, _, _ in rows_voice:
        ref_voice_audio[r_id] = load_wav(out / "refs" / "voice" / f"{r_id}.wav")
    ref_music_audio = {}
    for r_id, _, _ in rows_music:
        ref_music_audio[r_id] = load_wav(out / "refs" / "music" / f"{r_id}.wav")

    for k, t in enumerate(tasks):
        fid = "TEST_%04d" % k
        if t["kind"] == "song":
            raw = load_wav(out / "_src_sonics" / t["m"])
            dur = float(rng.uniform(20, 40))
            x = trim_or_pad(raw, dur, rng, "rand")
            vp, mp, vf_, mf_ = 1, 1, 1, 1
            sv, sm = "sonics", "sonics"
        elif t["kind"] == "mix":
            va = ref_voice_audio[t["v"]]
            ma = ref_music_audio[t["m"]]
            mode = "seq" if rng.random() < 0.3 else "ov"
            dur = float(rng.uniform(6, 16))
            x = overlay(trim_or_pad(va, min(dur, len(va) / SR)), trim_or_pad(ma, dur), rng, mode)
            x = trim_or_pad(x, dur, None)
            vp, mp, vf_, mf_ = 1, 1, t["vf"], t["mf"]
            sv, sm = t["v"], t["m"]
        elif t["kind"] == "voice":
            x = ref_voice_audio[t["v"]]
            vp, mp, vf_, mf_ = 1, 0, t["vf"], 0
            sv, sm = t["v"], "-"
        else:
            x = ref_music_audio[t["m"]]
            vp, mp, vf_, mf_ = 0, 1, 0, t["mf"]
            sv, sm = "-", t["m"]
        x, fx_name = apply_fx(x, rng, tmpdir, args.fx_prob)
        save_wav(x, out / "test" / f"{fid}.wav")
        file_fake = int(max(vf_, mf_) > 0)
        labels.append([fid, file_fake, vf_, mf_, vp, mp, sv, sm, fx_name])
        if (k + 1) % 100 == 0:
            print(f"rendered {k + 1}/{len(tasks)}")

    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE",
                    "VOICE_PRESENT", "MUSIC_PRESENT", "SRC_V", "SRC_M", "FX"])
        w.writerows(labels)
    print(f"done: {len(labels)} files -> {out/'labels.csv'}")


if __name__ == "__main__":
    main()
