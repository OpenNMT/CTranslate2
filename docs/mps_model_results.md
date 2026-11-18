# MPS model results and implementation scope

These measurements use an Apple M1 MacBook Air with seven GPU cores, 8 GB
unified memory, macOS 27.0.1 and native arm64 Release builds. They are recorded
development measurements, not a benchmark of every supported checkpoint or of
the final draft integration commit. Source revisions and measured library
hashes are retained in the linked reports. Correctness checks run separately
from timing; large-model checks with constrained command submission do not
provide throughput figures.

## What this extension adds

- Native Metal attention: FP32 online softmax, tiled prefill, split-K/vector
  decode, compact GQA/MQA storage, causal and absolute sliding-window masks.
  Supported kernels avoid materializing the complete attention matrix.
- Weight-only `int4_float16`: group-64 conversion and persistent packed weights,
  with dequantization fused into dense and expert GEMV/GEMM. Full matrices are
  never dequantized for each token.
- Compatible Mixtral, Qwen3 MoE and Qwen2 MoE decoders: FP32 router projection,
  GPU TopK, grouping, indirect expert execution and weighted gathering, including
  empty/uneven expert loads, shared experts, INT4 experts and explicit windows.
- Compact cache growth, in-place append and GPU beam gathering. FP16 is the
  recommended storage format. INT8/INT5/INT4 cache options remain experimental
  and fail the real-model quality gates described below.
- Gemma2/Gemma3/Phi3 converter and numerical corrections, including attention
  scaling, layer schedules, softcaps, RMSNorm epsilon and LongRoPE state.
  Gemma2 uses FP32 activations with FP16/INT4 weights to limit decoding drift.
- Model/operator regressions, independent CPU source references, exact-token
  llama.cpp replay and corpus perplexity tools. The existing temperature,
  TopK, TopP, greedy and seeded sampling paths are exercised by decoder tests.

The extension builds on the already merged MPS backend. Its existing RMSNorm,
RoPE, stream batching and general elementwise kernels are not new features of
this PR. The separate T5Gemma2 numerical follow-up is a dependency.

## Model coverage

Errors below compare against independent source models. INT4 references quantize
source weights independently; matching an INT4 reference is different from
matching the original FP16 model's quality. Reported RMS errors are the largest
case-level values in each stated comparison.

| Model | Tested configuration | Correctness result | Performance evidence |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | FP16 weights/cache; native compact attention | All 32 output IDs match llama.cpp at each 32/128/512/2048-token shape and repetition | Matched table below; CT2 remains slower |
| Same dense Qwen3 | Group-64 INT4 weights, FP16 cache | All eight greedy IDs match independent CPU at 7/33 tokens; cached max error 0.063873, RMS 0.011473 | Full WikiText-2 quality table below; no matched llama.cpp Q4 speed claim |
| `DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2` | FP16 weights/cache | Full/cached max error 0.17673, RMS 0.01034 at 7/33 tokens; cached max 0.11044, RMS 0.01024 at 32/128/512/2048/2051; checked greedy IDs match | Historical FP16 and INT4 baselines and controlled decode A/B below |
| Same Qwen3 MoE | INT4 weights, FP16 cache | Full/cached max error 0.08717, RMS 0.01197 at 7/33 tokens; checked greedy IDs match independently quantized source | INT4 table is not a comparison with llama.cpp Q4 |
| `google/gemma-2-2b` | INT4 weights, explicit FP16 cache, FP32 activations | 4,097-token prompt crosses the 4,096 window; both greedy IDs match; cached max error 0.020956, RMS 0.004344 | Correctness only; softcapped attention still uses the generic GPU path |
| `google/gemma-3-1b-it` | FP16 weights/cache | 33/513/1025-token prompts; all 12 greedy IDs match; full/cached max error 0.096236, RMS 0.015781 | No isolated throughput benchmark recorded |
| `microsoft/Phi-3-mini-128k-instruct` | INT4 weights, FP16 cache | 33/4097-token prompts; all eight greedy IDs match; full/cached max error 0.109780, RMS 0.019547 | No isolated throughput benchmark recorded; 128K inference is not qualified |
| `jordimas/t5gemma-2-270m-270m` | Stored FP16 weights, FP32 MPS activations/KV | 32/32 source argmax positions; max error 0.009190, RMS 0.001808; 12 complete-generation requests match CPU tokens | Inherited from the separate numerical follow-up; no new throughput claim |
| GPT-2 | Small source-model regressions, FP32/FP16/BF16 | Large finite GELU inputs produce finite logits and valid greedy/beam outputs | Regression coverage, not a new real-checkpoint speed result |
| Mixtral / Qwen2 MoE | Small independent source models, FP16 and INT4 | Router semantics, full/cached logits, beam ordering, shared experts and explicit windows pass tested cases | No real Mixtral/Qwen2 MoE throughput benchmark recorded |
| Whisper / Marian | Local FP16 checkpoints | Exact saved Whisper tokens and Marian hypotheses retained | Exploratory self-attention A/B below; not corpus WER/BLEU qualification |

Pinned additional model revisions: Gemma2
`c5ebcd40d208330abc697524c919956e692655cf`, Gemma3
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`, Phi3
`f3c06aed622e14ca0abf5115094e4fc9a9948f36`, T5Gemma2
`dfb6d0d619d8e6d42f279a6197abb90dec0137c1`.

## Dense Qwen3: matched FP16 baseline

Source revision `c1899de289a04d12100db370d81485cdf75e47ca`. Both engines use
FP16 weights and K/V, identical prompt IDs, 32 output tokens, one worker/slot,
four CPU threads, greedy TopK 1, TopP 1, temperature 1 and seed 42. Processes
run separately. Values are three warm-request medians after one first request
per shape. No optimization speedup is inferred from this baseline.

| Prompt tokens | CT2 TTFT, s | llama.cpp TTFT, s | CT2 decode, tok/s | llama.cpp decode, tok/s |
|---:|---:|---:|---:|---:|
| 32 | 0.0805 | 0.0346 | 32.30 | 44.06 |
| 128 | 0.1611 | 0.0837 | 32.57 | 44.31 |
| 512 | 0.5875 | 0.3132 | 28.23 | 42.07 |
| 2048 | 3.3616 | 1.5747 | 27.61 | 36.87 |

CT2 reaches 67–75% of llama.cpp's decode throughput; TTFT is 1.88–2.33 times
longer. All compared 32-token outputs match exactly. The
[qualification report](mps_qualification.md#fresh-matched-dense-model-baseline)
retains first-request timing, engine settings and provenance. TTFT includes
Python callbacks for CT2 and localhost HTTP/SSE for llama.cpp; it is not isolated
prefill timing.

## Real Qwen3 MoE: historical baseline and decode improvement

Source revision `f85830ed241787c392660f3e064050ccacac9602`. Despite its name,
this checkpoint contains approximately 1.39 billion weights: 28 layers, width
1024, four experts, TopK 2, 16 query heads, eight KV heads and head depth 128.
The historical baseline uses identical prompt IDs and sampling, 32 outputs,
one first request and two warm repeats per shape. FP16 and INT4 use FP16 cache.

| Prompt tokens | CT2 FP16 TTFT, s | CT2 FP16 decode, tok/s | llama.cpp FP16 TTFT, s | llama.cpp FP16 decode, tok/s | CT2 INT4 TTFT, s | CT2 INT4 decode, tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.145 | 25.49 | 0.070 | 29.69 | 0.318 | 31.84 |
| 128 | 0.270 | 24.70 | 0.144 | 31.38 | 0.542 | 30.30 |
| 512 | 0.813 | 22.63 | 0.499 | 30.46 | 1.642 | 27.54 |
| 2048 | 4.224 | 19.41 | 2.307 | 27.05 | 7.007 | 22.78 |

All FP16 output IDs match the saved llama.cpp baseline. INT4 improves decode
throughput relative to this FP16 configuration but has slower prefill/TTFT;
its quality and format differ, so it is not a matching-quantization llama.cpp
comparison. These are historical measurements of the earlier 4.9.0 port, not
a new timing run on the draft branch.

A separate same-model A/B uses FP16 weights/cache, 16 outputs, one first
request and two warm repeats in fresh processes. Only the generic/vector decode
selection changes:

| Prompt tokens | Generic decode, tok/s | Vector decode, tok/s | Measured change |
|---:|---:|---:|---:|
| 512 | 21.99 | 22.48 | Same kernel; run variation |
| 2048 | 18.08 | 20.77 | +14.9% |
| 4096 | 14.18 | 16.84 | +18.8% |

All nine cold/warm sequences per variant match. At 4,096 tokens, TTFT stays
about 11.5 seconds: this is a decode gain, not a demonstrated prefill gain.
The 4,608-token allocated FP16 cache is approximately 504 MiB for this model.
The 16K-key/depth-128 attention microbenchmark measures 1.93 ms vector versus
3.41 ms generic over 200 samples; that ratio is not an end-to-end speedup.
See [original protocol and hashes](mps_decoder_inference.md#long-context-ab-measurement).

## Whisper and Marian self-attention

Earlier exploratory A/B runs used local checkpoints, one first request and
three warm repeats, four CPU threads and one worker. Whisper uses FP16,
Pashto audio, beam 5, temperature 0, no VAD and no previous-text conditioning.
Warm median transcription time is 6.8817 seconds with the new flash hook
disabled versus 6.5291 seconds for self-attention only, a 5.1% latency decrease.
All token hashes match. Enabling shared-KV cross-attention instead measured
8.4672 seconds, so that path remains opt-in. Encoder prefill is unchanged.

Marian uses FP16, at most 32 output tokens and these input/beam/batch settings:

| Input tokens | Beam | Batch | Hook disabled, s | Self-attention hook, s |
|---:|---:|---:|---:|---:|
| 32 | 1 | 1 | 0.1045 | 0.0966 |
| 32 | 1 | 4 | 0.2945 | 0.2806 |
| 128 | 1 | 1 | 0.1434 | 0.1329 |
| 128 | 1 | 4 | 0.2329 | 0.2111 |
| 32 | 4 | 1 | 0.2080 | 0.2013 |
| 32 | 4 | 4 | 0.4931 | 0.4714 |
| 128 | 4 | 1 | 0.2980 | 0.2937 |
| 128 | 4 | 4 | 0.7914 | 0.7787 |

Hypotheses match in all eight configurations. These small exploratory runs do
not establish general speedups or public WER/BLEU results. Later qualification
retains all 32 Marian requests and four Whisper token hashes; those later
concurrent runs are correctness evidence only.

## Quality and remaining work

The complete pinned WikiText-2 test stream has 299,077 scored targets, context
512 and stride 511. Qwen3-0.6B perplexity is **28.6077 with FP16 weights** versus
**37.2736 with group-64 INT4 weights**, both using FP16 cache: a **30.29% increase**.
Passing independently quantized-reference logits does not remove this quality
gap. See [corpus protocol](mps_qualification.md#weight-quantization-quality).

Real Qwen3 INT8/INT5/INT4 **cache** quality gates fail independently of weight
quantization. Those modes are not recommended. Gemma2 softcapped attention is
still materialized on the GPU; sliding-window caches do not evict old history.
The current integration still needs real FP16 MoE cold/warm requalification:
an earlier issue-validation snapshot diverged at a 128-token prompt without
logit capture, while the older qualified build and instrumented requests were
stable. The historical FP16 table does not close that qualification item.
Whole-model BF16, broader checkpoints, matching llama.cpp Q4 comparisons,
isolated prefill/routing profiles and total peak unified memory remain to be
qualified. Unsupported attention configurations retain guarded generic paths
or explicit errors as documented in [decoder inference](mps_decoder_inference.md).

## Reproduction and evidence

[Benchmark tools](../tools/benchmark/llm/README.md) describe source export,
conversion, independent references, exact-token replay and perplexity.
[Recorded inputs](../tools/benchmark/llm/results/README.md) include the public
Qwen prompt token IDs and compact measurement summaries. Historical results
must retain their own output lengths, repetitions, quantization and source
hashes; do not combine them into one claimed final-build benchmark.
