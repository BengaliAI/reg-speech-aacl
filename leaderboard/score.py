#!/usr/bin/env python3
"""Score Ben-10 test hypotheses. Keep hyps private; publish metrics only."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
from jiwer import wer


def norm_id(x: str) -> str:
    x = str(x).strip()
    if x.endswith(".wav"):
        x = x[:-4]
    return x.replace("16kHz_test_audio/", "")


def region_of(utt_id: str) -> str:
    m = re.match(r"test_([a-zA-Z]+)", utt_id)
    return m.group(1).lower() if m else "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--hyps", required=True, type=Path)
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-url", default="")
    p.add_argument("--backend", default="unknown")
    p.add_argument("--scorer-commit", default="")
    p.add_argument("--decode-commit", default="")
    p.add_argument("--requested-by", default="maintainer")
    p.add_argument("--notes", default="")
    p.add_argument("--out", type=Path, default=Path("metrics.json"))
    p.add_argument(
        "--record-env",
        action="store_true",
        help="Attach env_versions() package pins to metrics.json",
    )
    args = p.parse_args()

    gold = pd.read_csv(args.gold)
    hyps = pd.read_csv(args.hyps)
    if "sentence" not in hyps.columns:
        raise SystemExit("hyps CSV needs columns: file_name,sentence")
    id_col = "file_name" if "file_name" in hyps.columns else "id"
    gold = gold.copy()
    hyps = hyps.copy()
    gold["id"] = gold["file_name"].map(norm_id)
    hyps["id"] = hyps[id_col].map(norm_id)
    gold["region"] = gold["id"].map(region_of)

    merged = gold.merge(hyps[["id", "sentence"]], on="id", how="left", indicator=True)
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise SystemExit(f"missing hyps for {missing} / {len(gold)} gold rows")

    merged["ref"] = merged["transcripts"].fillna("").astype(str).str.replace("\n", " ").str.strip()
    merged["hyp"] = merged["sentence"].fillna("").astype(str).str.replace("\n", " ").str.strip()

    overall = float(wer(merged["ref"].tolist(), merged["hyp"].tolist()))
    by_region = {
        r: float(wer(g["ref"].tolist(), g["hyp"].tolist()))
        for r, g in merged.groupby("region")
    }

    row = {
        "model_id": args.model_id,
        "model_url": args.model_url,
        "wer": overall,
        "wer_by_region": by_region,
        "backend": args.backend,
        "scorer_commit": args.scorer_commit,
        "decode_commit": args.decode_commit,
        "evaluated_at": date.today().isoformat(),
        "requested_by": args.requested_by,
        "notes": args.notes,
        "n_utt": int(len(merged)),
    }
    if args.record_env:
        from env_versions import versions

        row["env"] = versions()
    args.out.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"\nWrote {args.out} — append scores (not hyps) to bengaliAI/ben10-asr-results")


if __name__ == "__main__":
    main()
