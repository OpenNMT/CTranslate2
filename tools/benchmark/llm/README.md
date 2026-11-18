# Decoder-only MPS validation and benchmarks

Use a native arm64 Release build, an idle Mac, and separate processes for CPU,
MPS, Transformers, and llama.cpp. Never time competing model processes. Preserve
the source snapshot revision and conversion settings. Do not use debug GEMM or
synchronization logging for reportable timing; collect a separate profile.

See [MPS decoder inference](../../../docs/mps_decoder_inference.md) for supported
attention, MoE and native INT4 configurations. Use `--flash-attention` to select
the compact KV cache and native prefill/decode kernels. The real-model regression
clears experimental GEMM/attention switches to check the default execution path.

`benchmark.py` writes prompt token IDs, effective compute type, sampling options,
module path, runtime controls, cold/first-per-shape and warm results, TTFT,
individual decode intervals, and process peak RSS. Its prompt-tokens/TTFT metric
includes the first decode and sampling; it is **not isolated prefill throughput**.
Process RSS is **not total peak unified memory**. KV memory is a geometry estimate;
the generic layer replicates GQA heads, while `--flash-attention` uses compact
KV heads with capacity rounded to 512-token blocks. No GPU utilization or
per-kernel timing is inferred from wall time.

Use `--validate-logits` for repeated-prefill correctness runs. This captures every
generated vocabulary logit, rejects nonfinite values and cold/warm greedy token
differences, and saves the failing run before exiting. Its extra GPU copies make
its timing unsuitable for comparison with ordinary timing runs. The opt-in model
regression runs short and 2048-token prompts without downloading models:

```bash
CT2_TEST_MPS=1 CT2_MPS_LLM_MODEL=/path/to/ct2-model \
CT2_MPS_LLM_SOURCE=/path/to/hf-snapshot \
python -m pytest -q python/tests/test_mps_llm_real.py
```

```bash
export PYTHONPATH="$PWD/python"
PYTHON=/path/to/python
MODEL=/path/to/ct2-qwen3-f16
SOURCE=/path/to/pinned/huggingface/snapshot

# Run logits/greedy/cache validation before timing; each export exits before the next.
$PYTHON tools/benchmark/llm/validate.py export --model "$MODEL" \
  --tokenizer "$SOURCE" --device cpu --compute-type float32 --output cpu.json
$PYTHON tools/benchmark/llm/validate.py export --model "$MODEL" \
  --tokenizer "$SOURCE" --device mps --compute-type float16 --output mps.json
$PYTHON tools/benchmark/llm/validate.py compare cpu.npz mps.npz --output parity.json

# Transformers checks the converter independently. It exports full-sequence logits.
$PYTHON tools/benchmark/llm/validate.py export --backend transformers \
  --model "$SOURCE" --tokenizer "$SOURCE" --output transformers.json

$PYTHON tools/benchmark/llm/benchmark.py --model "$MODEL" --tokenizer "$SOURCE" \
  --device mps --compute-type float16 --flash-attention --lengths 32 128 512 2048 \
  --output-tokens 32 --repeats 3 --output mps-timing.json

# Use a fresh process and a short request for shape/synchronization diagnostics.
CT2_MPS_PROFILE=1 CT2_MPS_LOG_GEMM=1 CT2_MPS_LOG_SYNC=1 \
$PYTHON tools/benchmark/llm/benchmark.py --model "$MODEL" --tokenizer "$SOURCE" \
  --device mps --lengths 32 --output-tokens 4 --repeats 1 \
  --output mps-profile.json 2>mps-profile.log
```

`validate.py` exports unscaled full-vocabulary logits, next-token perplexity on
the deterministic input, generated greedy IDs, and cached-step/full-sequence logit
differences for odd lengths. These short prompts are a regression smoke check,
not a corpus quality evaluation. Select tolerances before reviewing candidates;
defaults are max absolute logit error 0.2 and RMSE 0.025. Every value must be finite.
Greedy IDs must match exactly when comparing CPU and MPS exports. Inspect near-ties
if this fails; do not hide it by widening tolerances. Sampling must also be checked
with the existing primitive and decoding tests. Cross-backend seeded RNG identity
is not assumed.

`reference_dense.py` provides a memory-bounded Transformers reference for local
Gemma2, Gemma3 text and Phi3 safetensors snapshots. It loads one decoder layer at
a time and tiles the vocabulary projection. `--weight-reference int4_float16`
independently emulates group-64 weight quantization; it does not read converted
CT2 weights. Its default arithmetic is FP32. `--compute-dtype float16` is a separate
rounding diagnostic, with an FP32 vocabulary projection matching the MPS contract.
Both modes check the source model's cached/full agreement and retain failing
arrays. This reference performs repeated weight reads and is unsuitable for
throughput comparisons. Gemma2 requires `--attention eager` for its softcap.

```bash
$PYTHON tools/benchmark/llm/reference_dense.py --snapshot "$SOURCE" \
  --weight-reference int4_float16 --lengths 33 513 --output-tokens 4 \
  --output source-int4.json
$PYTHON tools/benchmark/llm/validate.py export --model "$MODEL" \
  --tokenizer "$SOURCE" --device mps --compute-type float16 \
  --flash-attention --cached-only --lengths 33 513 --output-tokens 4 \
  --output mps-int4.json
$PYTHON tools/benchmark/llm/validate.py compare source-int4.npz mps-int4.npz
```

For llama.cpp, convert the **same snapshot** to F16 GGUF, use F16 K/V, one slot,
and all layers on Metal. Do not compare AWQ to GGUF Q4 as if their quantization
schemes were identical. Retain build commit, server log, configuration and GGUF
conversion log. Start the server only after CT2 has exited:

```bash
llama-server -m qwen3-f16.gguf -ngl 99 -c 4096 -np 1 -ctk f16 -ctv f16 \
  --host 127.0.0.1 --port 18081 --no-warmup
$PYTHON tools/benchmark/llm/llama_server.py --prompts mps-timing.prompts.json \
  --output-tokens 32 --repeats 3 --output llama-timing.json
```

The replay disables prompt reuse, min-p and repetition penalties and sends the
exact CT2 token IDs. It rejects implicit prompt length changes and incomplete
outputs. TTFT includes localhost HTTP/SSE overhead; CT2 includes Python callback
overhead. llama.cpp server prompt throughput is recorded separately in the server
timings. Inspect output IDs before interpreting timing differences. Increasing
lengths beyond 2048 requires checking available unified memory first.

Complete normalized Qwen3 MoE/full-attention Mixtral conversion and execution,
the native GPU router/dispatch/expert kernels, and real source-reference checks
are described in [MPS decoder inference](../../../docs/mps_decoder_inference.md).
`reference_moe.py` provides an independent memory-bounded CPU FP32 source
reference. For long contexts use its `--last-logits-only` option and
`validate.py export --cached-only` for the candidate, then compare the NPZ files.
INT4 experts, shared Qwen2 experts, and explicit sliding-window MoE schedules
are covered by the extension tests. Window cache eviction and broad INT4 quality
qualification remain outstanding.

## Corpus perplexity

`prepare_corpus.py` requires `pyarrow`. It reads a local Parquet `text` column
and writes token IDs, source revision and content hashes. It joins all rows, including empty rows,
with two newlines and adds no special tokens. Prepare the manifest once and
reuse it across weight/cache formats. For the November qualification run, the
WikiText-2 raw test file is pinned to Salesforce/wikitext revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, file
`wikitext-2-raw-v1/test-00000-of-00001.parquet`.

```bash
$PYTHON tools/benchmark/llm/prepare_corpus.py --parquet "$PARQUET" \
  --tokenizer "$SOURCE" \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --output corpus-tokens.json
$PYTHON tools/benchmark/llm/perplexity.py --model "$MODEL" \
  --tokenizer "$SOURCE" --corpus corpus-tokens.json \
  --context 512 --stride 511 --kv-cache-type float16 --output quality.json
```

Every target after the corpus's first token is scored exactly once. Context
resets per window; only the new targets contribute to perplexity. Record both
context and stride when comparing results. No sampling or generated text enters
this metric. The tool rejects a tokenizer hash mismatch, changed sequence length,
and nonfinite probabilities, and saves progress after each window. Compare
weight formats with the same manifest, context, stride and cache. Evaluate cache
formats separately. `--max-tokens 16384` is a subset pilot, not a full-corpus
qualification. These quality runs are not throughput benchmarks.
