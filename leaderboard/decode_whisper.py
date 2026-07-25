#!/usr/bin/env python3
"""Decode Ben-10 test with a Whisper ASR pipeline. Hyps stay private."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm
from transformers import pipeline


def load_wav(path: str, target_sr: int = 16000) -> dict:
    """Load wav as pipeline audio dict (avoids ffmpeg PATH dependency)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # linear resample without librosa dependency if rates rare
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": sr}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--model", default="bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--chunk-length-s", type=float, default=30.0)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    gold = pd.read_csv(args.gold)
    files = []
    for fn in gold["file_name"].tolist():
        path = args.audio_dir / fn
        if not path.exists():
            raise SystemExit(f"missing audio: {path}")
        files.append((fn, str(path)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, str] = {}
    if args.out.exists():
        prev = pd.read_csv(args.out)
        if {"file_name", "sentence"} <= set(prev.columns):
            done = dict(zip(prev["file_name"].astype(str), prev["sentence"].astype(str)))
            print(f"resume: {len(done)} existing hyps in {args.out}")

    todo = [(fn, path) for fn, path in files if fn not in done]
    print(f"total={len(files)} todo={len(todo)} model={args.model}")

    device = 0 if args.device == "0" and torch.cuda.is_available() else -1
    dtype = torch.float16 if device == 0 else torch.float32
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=args.model,
        chunk_length_s=args.chunk_length_s,
        device=device,
        dtype=dtype,
        ignore_warning=True,
    )
    target_sr = int(getattr(pipe.feature_extractor, "sampling_rate", 16000))

    # rewrite full file each checkpoint for crash-safety
    def flush() -> None:
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file_name", "sentence"])
            w.writeheader()
            for fn, _ in files:
                if fn in done:
                    w.writerow({"file_name": fn, "sentence": done[fn]})

    batch_audio: list[dict] = []
    batch_names: list[str] = []

    def run_batch() -> None:
        nonlocal batch_audio, batch_names
        if not batch_audio:
            return
        outs = pipe(batch_audio, batch_size=len(batch_audio))
        if isinstance(outs, dict):
            outs = [outs]
        for fn, out in zip(batch_names, outs):
            done[fn] = (out.get("text") or "").strip()
        batch_audio, batch_names = [], []
        flush()

    for fn, path in tqdm(todo, desc="decode"):
        batch_names.append(fn)
        batch_audio.append(load_wav(path, target_sr=target_sr))
        if len(batch_audio) >= args.batch_size:
            run_batch()
    run_batch()

    meta = {
        "model": args.model,
        "n": len(done),
        "device": device,
        "batch_size": args.batch_size,
        "chunk_length_s": args.chunk_length_s,
        "dtype": str(dtype),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out} ({len(done)} rows)")


if __name__ == "__main__":
    main()
