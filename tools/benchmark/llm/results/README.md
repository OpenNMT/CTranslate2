# Recorded M1 development measurements

These files support the tables in [model results](../../../../docs/mps_model_results.md).
They retain the public Qwen prompt token IDs, first-request measurements, warm
samples, source/model revisions, library hashes and sampling settings.
They are recorded development results, not new timings of the final PR commit.

- `apple-m1-development-results.json`: dense Qwen3 FP16 comparison, historical
  real Qwen3 MoE FP16/INT4 measurements, generic/vector decode A/B, and exploratory
  local Whisper/Marian latency samples. Process RSS is not total unified memory.
- `qwen-prompts-32-2048.json`: exact 32/128/512/2048-token inputs for the dense
  comparison and historical 32-output MoE baseline.
- `qwen-prompts-512-4096.json`: exact 512/2048/4096-token inputs for the separate
  16-output MoE generic/vector decode A/B.

After converting the pinned model into `$MODEL`, use the corresponding pinned
tokenizer directory as `$SOURCE`. Run on an idle machine with a native Release
build and no competing model processes:

```bash
python tools/benchmark/llm/benchmark.py --model "$MODEL" --tokenizer "$SOURCE" \
  --device mps --compute-type float16 --flash-attention --kv-cache-type float16 \
  --prompts tools/benchmark/llm/results/qwen-prompts-32-2048.json \
  --output-tokens 32 --repeats 3 --output ct2-dense.json
```

Replay these exact IDs with `llama_server.py` as described in the
[benchmark guide](../README.md). The historical MoE baseline uses two warm
repeats rather than three. For the long-context A/B, use the second prompt file,
16 outputs and two warm repeats; repeat in a fresh process with
`CT2_MPS_DISABLE_VECTOR_CACHE=1`. Retain output IDs and compare them before
interpreting timings.

The local Whisper/Marian checkpoints and private audio are not distributed.
Their small historical A/B results cannot serve as a reproducible public quality
benchmark. Use the [quality benchmark tools](../../quality/README.md) for public
WER/BLEU evaluation. No private input text or audio is included here.
