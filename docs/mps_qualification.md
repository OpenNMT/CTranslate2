# MPS decoder qualification: November 2026

These results use an Apple M1 MacBook Air, 7-core GPU, 8 GB unified memory and
an arm64 Release build. Operator parity, model regressions, quantization quality
and throughput are separate requirements.

## MoE discrepancies

The three Transformers 5.9 failures had two causes:

- Qwen3 windowed INT4 compared a 65-token full request with a 64-token scoring
  request. Generic attention rounded intermediate matrices to FP16. A router
  cutoff margin near 0.000014 changed expert selection after shape-dependent
  rounding, producing a 0.003778 score discrepancy against the 0.002 tolerance.
- Normalized Qwen2 INT4 compared FP16 execution with FP32 activations. At one
  token, router scores 0.050945517/0.051031854 became 0.050968841/0.050967120
  after input rounding. Routing recomputed on the actual GPU input agreed
  within 0.00000003 and selected the same experts. An independent mixed-precision
  reference reduced the full-logit difference to 0.000187. This was a precision
  discontinuity, not incorrect TopK, token grouping or expert aggregation.

Supported absolute-window decoder attention now uses existing native
online-softmax kernels for at least 64 prefill queries and incremental decoding.
An empty position-bias object permits this path. Active bias, returned attention
scores, softcapping, relative positions, incompatible shapes and unsupported
dtypes retain the generic path. Short prefills retain their established path.
The original random-router Qwen3 regression remains unchanged. Native adapter
tests check exact FP16 causal-prefix agreement at 64/65 tokens, masks, offsets,
head-dimension tails and FP32/oversized-head fallbacks.

The normalized Qwen2 cross-precision fixture uses signed power-of-two router
multiples: token-dependent expert pairs with clear cutoffs. Other architectures
retain their random routers. A separate primitive deliberately changes a
near-tied selection through FP16 input rounding and checks each precision on its
actual representable input. It detects accidental FP16 router-score rounding.
No logit tolerance or greedy-token assertion was relaxed.

Both original near-tied random Qwen2 models are also retained as separate
model-level regressions. Their independent CPU references use FP16 model
operations, FP32 routing and an FP32 vocabulary projection, matching the MPS
precision boundaries. No routes or logits are forced or replayed. Full logits,
greedy tokens, cached decoding and mixed-batch beam ordering are checked with
the original tolerances.

The final focused checks pass with each of Transformers 4.57.6, 5.9.0 and
5.22.2: 76 decoder cases, 12 cache cases and 45 benchmark checks, 133 in total.
The MPS-enabled native suite passes 641 tests with 28 skips; a CPU-only build
passes 373 with 16 skips. Skips include unsupported operators and unavailable
external fixtures. CUDA execution was not available locally.

## Gemma2 across its window boundary

`google/gemma-2-2b`, revision `c5ebcd40d208330abc697524c919956e692655cf`,
was checked at 4,097 prompt tokens, INT4 weights, FP16 cache and two greedy
outputs. The independent CPU source reference used FP32 arithmetic and its own
group-64 weight quantization.

| Check | Result |
|---|---:|
| Greedy IDs | Both match: 651, 20010 |
| Cached-logit maximum absolute error | 0.020956 |
| Cached-logit RMS error | 0.004344 |
| Source cached/full maximum difference | 0.00005341 |

This crosses the 4,096-token window. It does not qualify every longer context,
batch or precision. Softcapped Gemma2 still uses generic GPU attention; its
missing optimized attention path is not implemented by this change.

## Quantized KV caches remain unqualified

The 12 tiny dense/MoE cache tests pass, but real Qwen3-0.6B INT8/INT5/INT4 cache
gates still fail. INT4 **weights with FP16 cache** are a separate configuration.

A 33-token INT8-cache trace captured Q/K/V and attention outputs at all 28
layers. Independent CPU attention given those exact GPU inputs agreed within
0.001953125 maximum error and 0.00001735 maximum per-layer RMS error.
Differences existed before packing: first-layer CPU/GPU K/V inputs produced
405/1,234 different quantization codes; later layers amplified the differences.
This run does not indicate online-softmax or shared-memory corruption.

Matching FP16 activations, FP32 attention probabilities and the FP32 vocabulary
projection did not clear real-model gates. At 33 tokens, CPU/GPU cached-logit
maximum errors were 0.6011/5.8928/2.5845 for INT8/INT5/INT4. The independent
CPU implementation also showed cached/full sensitivity. Identical-input packing
and attention tests alone cannot qualify these lossy formats.

A CPU-only codec experiment used four 512-token WikiText windows starting at
0, 4096, 8192 and 12288: 2,044 targets, FP16 weights/activations and FP32 attention
probabilities/vocabulary projection.

| CPU cache experiment | Perplexity |
|---|---:|
| FP16 | 19.8663 |
| Current symmetric per-token INT8 | 20.0260 |
| Current symmetric per-token INT4 | 206.7058 |
| Asymmetric per-channel K/per-token V, 4 bits, causal FP16 tail | 19.8999 |

The last experiment keeps the current 32-token query block in FP16, compressing
only completed prior blocks. Statistics do not use future tokens from the
current block. It explores the cache axes studied in
[KIVI](https://proceedings.mlr.press/v235/liu24bz.html). It is a CPU prototype,
not native Metal support or a full-corpus claim. Native packing, causal prefill,
append, tail storage, beam gathering and fallback behavior still need
implementation and independent tests before adopting the format.

## Fresh matched dense-model baseline

Both engines use `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, FP16 weights/cache, identical saved
prompt IDs, one worker/slot, four CPU threads, greedy sampling (TopK 1, TopP 1,
temperature 1, seed 42) and 32 output tokens. Processes run separately, with
one first request per shape and three warm repeats. All 32 output IDs match
across engines for every shape/repeat. Values below are warm medians.

| Prompt tokens | CT2 TTFT, s | llama.cpp TTFT, s | CT2 decode, tok/s | llama.cpp decode, tok/s |
|---:|---:|---:|---:|---:|
| 32 | 0.0805 | 0.0346 | 32.30 | 44.06 |
| 128 | 0.1611 | 0.0837 | 32.57 | 44.31 |
| 512 | 0.5875 | 0.3132 | 28.23 | 42.07 |
| 2048 | 3.3616 | 1.5747 | 27.61 | 36.87 |

CT2 reaches 67–75% of llama.cpp's decode throughput, requiring approximately
1.34–1.49 times its present throughput to match. TTFT is 1.88–2.33 times longer.
This full-attention model does not exercise the window fix; this is a baseline,
not an optimization speedup. Historical MoE measurements are separate.

llama.cpp commit: `bfb4308b058b334c6e68085c661ec9eb7e3d59f4`, all layers on
Metal, context 4096, batch 2048, microbatch 512, automatic FlashAttention, prompt
reuse disabled. CT2: base `5f0630e1` plus the window patch, measured library
SHA256 `bf108678a89b92f917b2c59f23cd5235533c6a450e242cb4355f2e1c4283e5fa`.
The later defensive guards/short-prefill cutoff do not execute for this model
and were checked separately. Prompt-manifest SHA256:
`bd5e66e4e53d5304966f89eafffe19be6bc11b11df9a12f0612634b9eee9579c`.

CT2 TTFT includes Python callback delivery; llama.cpp includes local HTTP/SSE.
Prompt tokens divided by TTFT is not isolated prefill throughput. These wall
times do not measure GPU utilization or total peak unified memory.

## Weight quantization quality

WikiText-2 raw test data is pinned to Salesforce/wikitext revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Joining 4,358 text rows with two
newlines, retaining empty rows and adding no special tokens produces 299,078
Qwen3 tokens. SHA256 of little-endian INT32 IDs:
`b44603066a92719a20e2dc18d6c5f7f5342b1877c20c1e2bdd92deca662d3d56`.

A 16,384-token pilot used context 512, stride 256 and 16,383 unique targets.
Qwen3-0.6B FP16-weight perplexity was 20.0700; group-64 INT4-weight perplexity
was 25.5841, both with FP16 cache: a 27.47% increase. Operator/CPU parity does
not remove this quantization-quality gap. It is separate from lossy KV storage.
The separate full-corpus run uses context 512 and stride 511, scoring all
299,077 targets exactly once:

| Qwen3-0.6B weights, FP16 cache | Full-test perplexity |
|---|---:|
| FP16 | 28.6077 |
| Group-64 INT4 | 37.2736 |

The full-test increase is 30.29% (mean NLL increase 0.264612). Context/stride
and token IDs are identical between weight formats. These are measurements of
this quantizer and model; they do not qualify INT4 quality on other models.
Better weight quantization remains necessary for the intended quality target.

See [benchmark tools](../tools/benchmark/llm/README.md) for corpus preparation,
target accounting and reproduction. Broader models/precisions, longer Gemma2
contexts, optimized softcapped attention, quantizer quality and matched INT4
llama.cpp comparisons remain qualification items until measured.
