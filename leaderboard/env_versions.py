#!/usr/bin/env python3
"""Print pinned runtime versions for leaderboard provenance."""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version


PKGS = [
    "torch",
    "transformers",
    "accelerate",
    "jiwer",
    "soundfile",
    "librosa",
    "pandas",
    "tqdm",
    "numpy",
]


def versions() -> dict:
    out: dict[str, str] = {}
    for name in PKGS:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "missing"
    try:
        import torch

        out["torch_cuda"] = str(torch.version.cuda)
        out["cuda_available"] = str(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": out,
    }


def main() -> None:
    print(json.dumps(versions(), indent=2))


if __name__ == "__main__":
    main()
