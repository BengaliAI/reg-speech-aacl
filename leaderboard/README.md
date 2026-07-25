# Ben-10 maintainer-run leaderboard tooling

Private eval against `ben-10-test` (local/Drive only). Public surfaces:

- Results: https://huggingface.co/datasets/bengaliAI/ben10-asr-results
- Board: https://huggingface.co/spaces/bengaliAI/ben10-asr-leaderboard

## Environment (uv)

Pinned for recreate of the 2026-07-19 regional Whisper baseline:

```bash
cd leaderboard
uv sync          # creates .venv from uv.lock (Python 3.12, torch cu124)
```

Runtime package receipt lives inside each run's `metrics.json` (`score.py --record-env`).

| Pin | Value |
| --- | --- |
| Python | 3.12 (see `.python-version`) |
| Lock | `uv.lock` |
| Torch | 2.5.1+cu124 |
| Transformers | 5.14.1 |
| jiwer | 4.0.0 |

Baseline used conda `pytorch_env` before this lock existed; versions above match that run. New evals should use `uv sync` + `uv run`.

## Decode (private hyps)

```bash
uv run python decode_whisper.py \
  --audio-dir /path/to/ben10-hidden-test/16kHz_test_audio \
  --gold /path/to/ben10-hidden-test/16kHz_test_audio/test.csv \
  --model bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium \
  --out /path/to/private_run/hyps.csv \
  --batch-size 4
```

## Score (never publish hyps)

```bash
uv run python score.py \
  --gold /path/to/ben10-hidden-test/16kHz_test_audio/test.csv \
  --hyps /path/to/private_run/hyps.csv \
  --model-id bengaliAI/some-model \
  --model-url https://huggingface.co/bengaliAI/some-model \
  --backend whisper-transformers-pipeline-fp16 \
  --record-env \
  --out /path/to/private_run/metrics.json
```

Hyps CSV columns: `file_name,sentence` (ids may omit `.wav`).

Then append a row to `ben10-asr-results` `results.csv` (scores only). Record `scorer_commit` / lock SHA in the row notes. Baseline snapshot (WER + env): `baselines/2026-07-19-regional-whisper.metrics.json`.
