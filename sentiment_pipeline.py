"""

Pipeline:
  1. Load audio from HuggingFace dataset
  2. Transcribe with Whisper (or use existing transcript if available)
  3. Score sentiment with DistilBERT SST-2
  4. Print per-call results + aggregate report

Install:
    pip install datasets transformers torch soundfile librosa audioread
"""

import warnings, io, os
warnings.filterwarnings("ignore")

import numpy as np
import torch
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from transformers import pipeline

# ══════════════════════════════════════════════════════════════════════
#  DATASET OPTIONS
# ══════════════════════════════════════════════════════════════════════
DATASET_OPTIONS = {
    # 1. PolyAI/minds14  ← DEFAULT (best for contact-center POC)
    #    Real customer service audio: banking, telecom, insurance intents
    #    563 en-US samples | free | no login needed
    #    https://huggingface.co/datasets/PolyAI/minds14
    "minds14": {
        "name"      : "PolyAI/minds14",
        "config"    : "en-US",
        "split"     : "train",
        "audio_col" : "audio",
        "text_col"  : "transcription",
        "label_col" : "intent_class",
    },

    # 2. LibriSpeech dummy  (clean English speech, ~73 samples, no login)
    #    https://huggingface.co/datasets/patrickvonplaten/librispeech_asr_dummy
    "librispeech_dummy": {
        "name"      : "patrickvonplaten/librispeech_asr_dummy",
        "config"    : "clean",
        "split"     : "validation",
        "audio_col" : "audio",
        "text_col"  : "text",
        "label_col" : None,
    },

    # 3. Robocall audio dataset  (1,432 real phone calls, open access)
    #    https://huggingface.co/datasets/wspr-ncsu/robocall-audio-dataset
    "robocall": {
        "name"      : "wspr-ncsu/robocall-audio-dataset",
        "config"    : None,
        "split"     : "train",
        "audio_col" : "audio",
        "text_col"  : "transcript",
        "label_col" : None,
    },
}

# ── Config ─────────────────────────────────────────────────────────────────────
ACTIVE_DATASET  = "minds14"          # change to "librispeech_dummy" or "robocall"
NUM_SAMPLES     = 20                 # How many calls to process (None = all)
WHISPER_MODEL   = "openai/whisper-base"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SAMPLE_RATE     = 16_000
# ══════════════════════════════════════════════════════════════════════

cfg = DATASET_OPTIONS[ACTIVE_DATASET]

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("=" * 65)
print("  Contact Center Audio → Sentiment Analyzer")
print("=" * 65)
print(f"\n[1/4] Loading dataset: {cfg['name']} …")

load_kwargs = dict(trust_remote_code=True)
if cfg["config"]:
    ds = load_dataset(cfg["name"], cfg["config"], **load_kwargs)
else:
    ds = load_dataset(cfg["name"], **load_kwargs)

split_name = cfg["split"] if cfg["split"] in ds else list(ds.keys())[0]
dataset    = ds[split_name]

# Disable auto-decode to avoid torchcodec dependency
dataset = dataset.cast_column(cfg["audio_col"], Audio(decode=False))

print(f"      Split '{split_name}' → {len(dataset)} samples total.")

if NUM_SAMPLES:
    dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
    print(f"      Processing {len(dataset)} sample(s).")

# ── 2. Load Models ─────────────────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1
hw     = "GPU" if device == 0 else "CPU"

print(f"\n[2/4] Loading Whisper ASR  ({WHISPER_MODEL}) on {hw} …")
asr = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_MODEL,
    device=device,
    chunk_length_s=30,
    stride_length_s=5,
)

print(f"\n[3/4] Loading Sentiment model  ({SENTIMENT_MODEL}) …")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    device=device,
    truncation=True,
    max_length=512,
)
print("      Models ready.\n")

# ── Helpers ────────────────────────────────────────────────────────────────────
def decode_audio(sample: dict, audio_col: str) -> np.ndarray:
    audio = sample[audio_col]

    # Already decoded
    if isinstance(audio, dict) and audio.get("array") is not None:
        arr = np.array(audio["array"], dtype=np.float32)
        sr  = audio.get("sampling_rate", SAMPLE_RATE)
    else:
        raw   = audio.get("bytes") if isinstance(audio, dict) else (audio if isinstance(audio, bytes) else None)
        fpath = audio.get("path")  if isinstance(audio, dict) else None

        if raw:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        elif fpath and os.path.exists(fpath):
            arr, sr = sf.read(fpath, dtype="float32", always_2d=False)
        else:
            raise ValueError("Cannot read audio: no bytes or valid path found.")

    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != SAMPLE_RATE:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
    peak = np.abs(arr).max()
    if peak > 0:
        arr /= peak
    return arr.astype(np.float32)


def to_score(label: str, conf: float) -> float:
    return conf if label == "POSITIVE" else -conf


# ── 4. Process Calls ───────────────────────────────────────────────────────────
print(f"[4/4] Transcribing & scoring {len(dataset)} call(s) …")
print("-" * 65)

results = []

for idx, sample in enumerate(dataset):
    print(f"\n📞  Call {idx + 1}/{len(dataset)}")

    # Decode audio
    try:
        audio_arr = decode_audio(sample, cfg["audio_col"])
    except Exception as e:
        print(f"   ⚠  Audio decode failed: {e}")
        continue

    duration = len(audio_arr) / SAMPLE_RATE
    print(f"   🔊 Duration : {duration:.1f}s")

    # Use existing transcript if available, else run Whisper
    text_col = cfg.get("text_col")
    existing = sample.get(text_col, "").strip() if text_col else ""

    if existing:
        text = existing
        src  = "transcript"
    else:
        try:
            text = asr({"raw": audio_arr, "sampling_rate": SAMPLE_RATE})["text"].strip()
            src  = "Whisper"
        except Exception as e:
            print(f"   ⚠  ASR failed: {e}")
            continue

    if not text:
        print("   ⚠  Empty transcript — skipping.")
        continue

    print(f"   📝 [{src}] {text[:110]}{'…' if len(text) > 110 else ''}")

    # Sentiment
    try:
        res   = sentiment_pipe(text)[0]
        label = res["label"]
        conf  = round(res["score"], 4)
        score = round(to_score(label, conf), 4)
    except Exception as e:
        print(f"   ⚠  Sentiment failed: {e}")
        label, conf, score = "UNKNOWN", 0.0, 0.0

    emoji = "😊" if label == "POSITIVE" else "😟"
    print(f"   {emoji} Sentiment : {label}  (conf={conf:.2%}, score={score:+.4f})")

    intent = ""
    if cfg["label_col"] and cfg["label_col"] in sample:
        intent = str(sample[cfg["label_col"]])
        print(f"   🏷️  Intent   : {intent}")

    results.append({
        "call"    : idx + 1,
        "text"    : text,
        "label"   : label,
        "conf"    : conf,
        "score"   : score,
        "intent"  : intent,
        "duration": round(duration, 1),
    })

# ── 5. Aggregate Report ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  📊  AGGREGATE SENTIMENT REPORT")
print("=" * 65)

if not results:
    print("  No results to display.")
else:
    scores = [r["score"] for r in results]
    pos    = sum(1 for r in results if r["label"] == "POSITIVE")
    neg    = len(results) - pos
    avg    = float(np.mean(scores))
    std    = float(np.std(scores))

    print(f"  Calls processed  : {len(results)}")
    print(f"  Positive         : {pos}  ({pos/len(results):.0%})")
    print(f"  Negative         : {neg}  ({neg/len(results):.0%})")
    print(f"  Avg score        : {avg:+.4f}  (–1 very negative → +1 very positive)")
    print(f"  Std deviation    : {std:.4f}")

    if   avg >=  0.5: verdict = "🟢 HIGHLY POSITIVE"
    elif avg >=  0.1: verdict = "🟡 MILDLY POSITIVE"
    elif avg >= -0.1: verdict = "⚪ NEUTRAL"
    elif avg >= -0.5: verdict = "🟠 MILDLY NEGATIVE"
    else:             verdict = "🔴 HIGHLY NEGATIVE"

    print(f"\n  Overall verdict  : {verdict}")
    print("=" * 65)

    print(f"\n  {'#':>3}  {'Score':>7}  {'Label':<10}  {'Conf':>7}  {'Dur':>6}  Intent")
    print(f"  {'-'*3}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*6}  {'-'*20}")
    for r in results:
        intent_s = (r["intent"][:18] + "…") if len(r["intent"]) > 20 else r["intent"]
        print(f"  {r['call']:>3}  {r['score']:>+7.4f}  {r['label']:<10}  "
              f"{r['conf']:>7.2%}  {r['duration']:>4.1f}s  {intent_s}")

print("\n✅  Done.\n")