# Release gate

Compares candidate wheels against the latest PyPI release; publication requires
passing the gate. Thresholds are in `run_gate.py` (`RULES`).

| Task | CI workload | Failure threshold |
| --- | --- | --- |
| Whisper tiny | 20 LibriSpeech dummy clean samples | WER increase >0.01 (1 percentage point) |
| opus-mt-en-de | 500 WMT14 sentences | BLEU decrease >1 point |
| T5Gemma 2 270M | 50 SQuAD validation prompts, FP32 greedy, up to 64 output tokens | Missing or empty completions |
| All tasks | Median throughput | Confirmed drop >10% |

T5Gemma uses `jordimas/t5gemma-2-270m-270m` at a pinned revision through
`Translator`. Changed text is informational, not an answer-quality check.
Prompts use `rajpurkar/squad` at revision `7b6d24c440a36b6815f21b70d25016731768db1f`,
shuffled with seed 42 and capped at 512 input tokens.
Invalid metrics and runtime errors fail immediately.

## Performance

Each task runs three sequential baseline/candidate pairs in alternating order.
Each fresh process warms up on one batch (one sample for Whisper), then times
inference only. CPU thread counts are fixed at two, with one CTranslate2 worker.

A median throughput drop >10% triggers three more pairs, starting with the
candidate. The same metric must regress in both rounds to fail; quality failures
from any measurement remain blocking. Individual measurements and medians are
logged. This reduces runner noise at the cost of 3–6× benchmark work plus warm-up.

## Running

CI tests one CPython 3.12 wheel per platform: Linux x86-64/ARM64, macOS ARM64,
and Windows x86-64. The gate runs automatically on pull requests and on tag
pushes, reusing the wheels built earlier in the CI pipeline.

```sh
python release-gates/run_gate.py --candidate-wheel dist/ctranslate2-*.whl
```

Use `--candidate-wheel-pattern "dist/*.whl"` to require exactly one matching wheel.
Omit `--candidate-wheel` to compare PyPI against itself. `--bleu-limit` sets
translation coverage (default 500), independently of `--limit`.
Task implementations are in `tasks.py`.
