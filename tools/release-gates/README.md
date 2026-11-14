# Release gate

Compares candidate wheels with the latest PyPI release and blocks publication on
regressions. Tasks and thresholds are defined in `tasks.py` and `run_gate.py`.

| Task | CI workload | Failure threshold |
| --- | --- | --- |
| Whisper tiny | 20 LibriSpeech dummy clean samples | WER increase >0.01 (1 percentage point) |
| opus-mt-en-de | 500 WMT14 sentences | BLEU decrease >1 point |
| T5Gemma 2 270M | 20 SQuAD prompts, up to 64 output tokens | Missing completions |
| All tasks | Median throughput | Confirmed drop >10% |

T5Gemma uses pinned model/dataset revisions, seed 42, and at most 512 input tokens.
Empty completions are allowed; text changes are informational. Invalid metrics
and runtime errors fail immediately.

Each task tests `default`, then `auto`, skipping `auto` only if both modes resolve
identically on both wheels. Models are converted once per wheel and reused.

Measurements use three alternating baseline/candidate pairs in fresh processes,
with warm-up, two CPU threads, and one worker. A median throughput drop >10%
requires confirmation in three additional pairs. Quality failures remain blocking.

## Running

CI runs on PRs, tag pushes, and manual dispatches using CPython 3.12 wheels:
Linux x86-64/ARM64 CPU, Windows x86-64 CPU, and macOS ARM64 CPU/MPS.
MPS requires Apple Silicon and MPS-enabled wheels.

```sh
python tools/release-gates/run_gate.py --candidate-wheel-pattern "dist/*.whl"
```

The pattern must match exactly one wheel; `--candidate-wheel` accepts a path or
pip spec. Omit both to compare PyPI against itself. Options:

- `--device cpu|cuda|mps` (default: `cpu`).
- `--limit N`: Whisper samples and T5Gemma prompts (default: 50; 0 uses all).
- `--bleu-limit N`: translation sentences (default: 500).

Direct translation task invocations through `tasks.py` require `--model-dir`.
