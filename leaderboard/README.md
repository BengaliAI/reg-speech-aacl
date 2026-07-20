# Ben-10 maintainer-run leaderboard tooling

Private eval against `ben-10-test` (local/Drive only). Public surfaces:

- Results: https://huggingface.co/datasets/bengaliAI/ben10-asr-results
- Board: https://huggingface.co/spaces/bengaliAI/ben10-asr-leaderboard

## Score hyps (never publish hyps)

```bash
python leaderboard/score.py \
  --gold /path/to/ben10-hidden-test/16kHz_test_audio/test.csv \
  --hyps /path/to/private_hyps.csv \
  --model-id bengaliAI/some-model \
  --model-url https://huggingface.co/bengaliAI/some-model \
  --backend whisper \
  --out /tmp/metrics.json
```

Hyps CSV columns: `file_name,sentence` (ids may omit `.wav`).

Then append a row to `ben10-asr-results` `results.csv` (scores only).
