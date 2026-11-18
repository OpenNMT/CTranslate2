# Decoder inference on Apple Metal

See [model results](mps_model_results.md) for implementation scope, per-model
correctness, measured latency/throughput and remaining qualification gaps.

See the [November qualification report](mps_qualification.md) for the windowed
MoE fix, precision-sensitive routing tests, Gemma2 window-boundary validation,
fresh dense-model baselines and remaining quantization-quality limitations.

The experimental MPS backend supports dense decoder-only Transformers and
compatible Mixtral, Qwen3 MoE and Qwen2 MoE models. Select `device="mps"`
explicitly; automatic device selection continues to use CPU. Build with
`-DWITH_MPS=ON` or use an Apple Silicon wheel. FP16 is the recommended model
compute type. See [hardware support](hardware_support.md) for general limits.

```python
import ctranslate2

generator = ctranslate2.Generator(
    "/path/to/converted-model",
    device="mps",
    compute_type="float16",
    flash_attention=True,
)
results = generator.generate_batch([prompt_tokens], max_length=32)
```

## Attention and KV cache

`flash_attention=True` selects compact `[batch, capacity, kv_heads, depth]`
caches, retaining GQA/MQA KV heads without replication. Capacity grows in
512-token blocks. Decode appends in place; growth copies only valid rows and
beam reordering uses GPU gather. Cache storage defaults to FP16 for FP16 models,
including models with INT4 weights. Gemma2 uses a higher-precision activation
path; its explicit FP16 cache option is described below. Quantized cache formats are optional and
experimental; see below.

Supported prefills with at least 64 query rows use 32-query by 32-key tiled
attention. QK, online softmax and attention-value accumulation use FP32.
Running maxima rescale previous accumulators between tiles, avoiding a full
query-by-key allocation. Incremental decoding uses 256-key split-K blocks and
a numerically stable FP32 merge. Supported masks include causal positions,
per-row lengths, explicit sliding windows, and all-masked rows. Kernel tests
cover FP16/BF16 storage, GQA/MQA, depths through 256, odd tails, cache padding,
large scores, and long contexts.

Short prefills and unsupported kernel shapes retain materialized GPU attention.
Requests for attention weights use the existing implementation. Configurations
incompatible with compact attention, such as ALiBi, relative position bias or
tensor parallelism, must use `flash_attention=False`; unsupported combinations
raise an error rather than silently changing model semantics.

The shared standard-layout incremental attention hook also serves Whisper and
Marian self-attention. Their encoder/prefill path remains unchanged. Shared-KV
cross-attention is opt-in with `CT2_MPS_USE_FLASH_CROSS_ATTENTION=1`: the tested
Whisper workload was slower with this path, so it is not enabled by default.
Whole-model BF16 qualification remains outstanding.

## Gemma and Phi conversion

Reconvert Gemma2, Gemma3 text and Phi3 checkpoints to retain their attention
scales, layer window schedules and RMSNorm epsilon. Gemma2 additionally requires
both its attention and vocabulary logit softcaps. Its attention currently uses
the generic GPU implementation even when `flash_attention=True`: the compact
kernels do not implement softcapping. Explicit quantized KV cache requests are
rejected for these layers. This fallback retains replicated KV heads and
materialized scores, so long-context memory use differs from compact attention.

Gemma2 uses FP32 activations, normalization and residual additions on MPS even
with FP16 or native INT4 model weights. Metal GEMV/GEMM read the original compact
weights and produce FP32 results; complete matrices are never expanded per
token. This limits accumulated cached-decoding error. `kv_cache_type="float16"`
retains FP16 cache storage and promotes cache reads for the generic FP32
attention calculation. `"auto"` follows the FP32 activation format. This path
uses more activation memory and has not been qualified as a throughput win.

Phi3 LongRoPE factors remain FP32 when model weights are quantized. Selection
uses the live sequence length rather than the allocated rotary-table capacity;
generation supplies the complete initial prompt length before splitting prefill.
Static-prefix cache reuse is disabled for this context-dependent position scheme.
As in the Transformers implementation, changing factors during generation does
not recompute keys already in the cache. Full-sequence comparisons must account
for a request crossing the original context boundary.

`python/tests/test_mps_gemma_phi.py` checks these converters against independent
source models, including FP16/INT4, sliding windows, cached decoding, reused
rotary tables, static prompts and mixed-length batches. Passing these small
fixtures does not qualify every real checkpoint or quantization configuration.

## KV cache storage options

`Generator(..., kv_cache_type="float16")` explicitly keeps the FP16 cache.
`"fp16"` is an alias; `"auto"` preserves the activation format and existing
CPU/CUDA behavior. `generator.kv_cache_type` reports the resolved format.
Independent generator instances can select different formats without changing
global environment variables or model files.

Experimental `"int8"`, `"int5"` and `"int4"` options require an MPS decoder-only
Transformer, FP16 activations, `flash_attention=True`, and head depth at most
256. Unsupported combinations fail explicitly. Cache precision is independent
of weight quantization: an INT4-weight model can use an FP16 cache or any of
these experimental formats. FP16 remains the recommended default.

| Cache format | Bytes per 128-dimensional K or V row | Logical KV memory per token for the model below |
|---|---:|---:|
| FP16 | 256 | 112 KiB |
| INT8 | 144 | 63 KiB |
| INT5 | 96 | 42 KiB |
| INT4 | 80 | 35 KiB |

Each group of 32 values contains one FP32 scale and packed symmetric codes.
Signed code ranges are -127..127, -15..15 and -7..7, respectively. The packer
uses normalized nearest-even rounding with an FMA remainder correction at ties.
Dequantized values round to FP16 before attention accumulation. Tail groups
contain zero padding; very small heads may have little or no memory saving.
The table is logical storage, excluding capacity rounding and temporary buffers,
not a measurement of total unified memory.

Only new rows are quantized. Growth copies prior packed bytes, and scales travel
with the codes during GPU beam gathering. Native prefill/decode fuse cache reads
and dequantization; supported token decoding does not unpack the complete cache.
Short prefill, attention-weight requests and generic fallbacks can unpack valid
rows on the GPU. Appending and reordering introduce no host transfers.

These formats are lossy. Independent primitive and tiny dense/MoE decoder tests
pass, including FP16/INT4 weights, cache growth, masks, tails and reordering.
The real MoE checkpoint below does **not** yet pass the unchanged real-model
logit/cache qualification gates for these formats. INT4 cache also materially
increases short-prompt perplexity. Neither primitive parity nor a faster cache
kernel establishes acceptable model quality. Validate the chosen format against
an independent source and representative prompts before considering deployment.
The benchmark tools accept `--kv-cache-type`; `reference_moe.py --kv-cache-bits`
provides the independent CPU cache oracle. Validation retains failing arrays
and exits unsuccessfully when the configured gates fail.

On the pinned real MoE checkpoint, the 33-token regression prompt produces:

| Cache | Prompt perplexity | Full-logit max error vs matching CPU cache | Four greedy IDs match CPU at 7 / 33 tokens |
|---|---:|---:|---|
| FP16 | 197.80 | 0.17673 vs unquantized source | yes / yes |
| INT8 | 192.39 | 0.37134 | yes / yes |
| INT5 | 236.59 | 5.98859 | yes / no |
| INT4 | 501.60 | 4.26444 | no / no |

All quantized formats fail the existing maximum-error/RMSE or cached/full gates;
INT5/INT4 also fail exact generated-ID parity. Their CPU fake-quantized caches
also show cached/full sensitivity. These results are qualification failures, not
accepted accuracy tradeoffs or production recommendations. The prompt is a
short smoke metric, not corpus perplexity; the small INT8 perplexity difference
does not establish a quality benefit. Outlier-aware or mixed-precision cache
schemes and broader quality evaluation remain necessary before promotion.
Quantized-cache prefill uses FP32 attention even for short prompts, matching
decode's probability precision. Keeping the legacy FP16 probability rounding
there amplified subsequent quantization errors; consistent precision improves
the INT4-cache perplexity result but does not resolve its qualification failures.

## Long-context decode specialization

Compact FP16 decoding with one query, aligned buffers and head depth 128 uses
vectorized Q/K/V loads for contexts of at least 1024 tokens. Four adjacent values
share each load; SIMD groups partition value accumulation, then merge FP32
partials. Separate INT8/INT5/INT4 kernels specialize code extraction and reuse a
scale across four values. Other depths, layouts and alignments retain the
generic kernels. The shared Whisper/Marian layout remains unchanged.

Sliding-window decoding skips fully masked key blocks and sizes its temporary
workspace to the active blocks. Historical cache storage is still retained;
this is scheduling and workspace reduction, not cache eviction. Stable FP32
maxima/denominators and the existing split-K merge are preserved. Tests cover
16,385 keys, GQA/MQA, all-masked rows and poisoned padding.

`CT2_MPS_DISABLE_VECTOR_CACHE=1` selects the retained generic kernel for isolated
comparison. An earlier scalar parallel-reduction prototype measured slower and
was removed. End-to-end gains must be measured separately from kernel timing.

## Native weight-only INT4

Convert an unquantized floating-point decoder-only Transformer with
`ct2-transformers-converter --quantization int4_float16`. Existing AWQ/GPTQ
weights cannot be reinterpreted as this format. Whisper, Marian and other model
types do not accept this conversion mode.

Each output row stores symmetric signed values from -7 to 7 and one positive
FP32 scale per 64 input columns. Adjacent columns occupy low/high nibbles with
encoding `q + 8`; odd tails are padded with encoded zero. Conversion uses
nearest-even rounding, assigns scale 1 to zero groups and rejects nonfinite
weights. Embeddings, vocabulary projection, routers and scalar shared-expert
gates remain floating point; this is a mixed-precision format.

Packing happens once during conversion. Immutable packed weights/scales belong
to the model and follow its lifetime. GEMV dequantizes into registers; GEMM
dequantizes only the current tile into threadgroup memory, rounding weights to
FP16 before FP32 accumulation. Complete weights are never dequantized per token.
MPS requires FP16 activations. The CPU reference expands weights once at load;
unsupported devices and compute types fail explicitly.

For a memory-bounded conversion of an existing CT2 FP16 decoder, use:

```bash
python tools/benchmark/llm/pack_int4.py /path/to/ct2-f16 /path/to/new-ct2-int4
```

This tool streams one linear matrix at a time into a new directory. Source
conversion is required for aliased linear matrices; embedding/head aliases are
supported. The qualification report records full WikiText-2 perplexity for Qwen3-0.6B
and its remaining INT4 quality gap. Broader model quality evaluation and a
comparison with matching llama.cpp quantization remain outstanding.

## MoE execution

Compatible decoders use SwiGLU experts with CPU FP32 reference execution or MPS
FP16 execution. Both separate and stacked source expert tensors are supported.
Qwen3/Mixtral normalize selected routing weights; Qwen2 can retain probabilities
from the full expert softmax. TopK orders descending logits with ascending
expert index for ties. Shared Qwen2 experts compute
`sigmoid(shared_gate(x)) * shared_expert(x)` and support INT4 linear weights.

Router projections keep a model-owned FP32 weight copy and accumulate in FP32.
TopK, token grouping, offsets, inverse mapping, indirect expert dispatch and
weighted gathering remain on the GPU. Expert IDs/counts are not read back to
the CPU. Small batches use fused planning and GEMV; larger workloads use
grouped tiled gate/up and down GEMMs. FP16/INT4 expert tiles are selected from
16x64x32 and 64x64x32 variants according to token count, with generic fallbacks.
Empty experts perform no work, and aggregation preserves token order and
activation rounding. Per-call temporary storage uses the existing allocator.

Explicit full/window layer schedules are retained with absolute-position masks
in generic and compact attention. Windows currently retain historical KV
storage; cache eviction is not implemented. Ambiguous legacy Qwen2 schedules
are rejected when their eager/Flash behavior differs. Supply explicit compatible
`layer_types` rather than silently replacing windows with full attention.

Complete CUDA/HIP MoE decoders, tensor parallelism, biased routers/experts and
group-limited routing are not implemented by this extension. Existing default
`MoeTopK` behavior used by other architectures is preserved. Full-softmax routing
is supported on CPU/MPS and rejected on CUDA/HIP until implemented.

## Numerical behavior and validation

Repeated large FP16 MPSMatrix projections reproduced nonfinite outputs. The
backend uses native SIMD-group GEMM for large FP16 dense projections with
`beta=0`, with guarded generic Metal fallbacks for incompatible layouts.
Regression tests poison output storage and rows beyond the input extent; they
also cover strides, alignment and odd dimensions. This is a tested workaround,
not a diagnosis of the internal MPSMatrix implementation.

Decoder-only FP16 vocabulary projections retain FP32 accumulator outputs for
logits to avoid creating artificial ties by rounding to FP16. Hidden activations
and KV caches remain FP16. Encoder-decoder vocabulary behavior is preserved.

Focused native tests cover attention against double-precision references,
routing indices/weights, uneven and empty experts, grouping and gathering,
quantized GEMV/GEMM, shared gates, and poisoned-storage regressions. Tiny source
models in `python/tests/test_moe_decoder.py` check conversion and complete
decoding, full/cached logits, beam reordering, mixed window schedules, INT4,
TopK configurations and seeded temperature/TopK/TopP sampling. These tests
include independent source-weight quantization and CPU references.

The opt-in `python/tests/test_mps_llm_real.py` checks repeated short and long
dense requests, finite vocabulary logits and exact greedy IDs. It requires
local model paths and never downloads a fixture implicitly. Native test suites
also retain existing Marian/Whisper coverage.

See [benchmark and reference tools](../tools/benchmark/llm/README.md) for pinned
real-model exports, independent memory-bounded MoE references and exact-token
llama.cpp replay. Run correctness checks before timing and retain source
revision, conversion, build, prompt IDs and sampling settings. The default
real-model logit gates are maximum absolute error 0.2 and RMSE 0.025, with finite
values and exact generated greedy IDs. FP16 reductions can change near-tied
full/cached argmax decisions; inspect those differences explicitly rather than
widening tolerances.

TTFT includes prefill, the first decode and callback/sampling overhead; prompt
tokens divided by TTFT is not isolated prefill throughput. Process peak RSS is
not total peak unified memory. GPU utilization, kernel timings and
synchronization cost require separate profiles. This implementation does not
claim to outperform llama.cpp.

## Real-model validation

The source checkpoint is
`DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2` at revision
`f85830ed241787c392660f3e064050ccacac9602`. Its matrices contain approximately
1.39 billion weights: 28 layers, width 1024, expert width 3072, four experts,
TopK 2, 16 query heads, eight KV heads, and head depth 128. The model name is not
a parameter count. References use source weights on CPU FP32; the INT4 reference
independently quantizes those weights with PyTorch.

Validation of the upstream 4.9.0 port on an M1 MacBook Air (seven GPU cores,
8 GB unified memory, macOS 27.0.1, native arm64 Release build) produced:

| Comparison | Prompt lengths | Maximum logit error | Maximum RMSE |
|---|---|---:|---:|
| FP16 vs source FP32, full/cached logits | 7, 33 | 0.17673 | 0.01034 |
| INT4 vs independently quantized source, full/cached logits | 7, 33 | 0.08717 | 0.01197 |
| FP16 vs source FP32, cached logits | 32, 128, 512, 2048, 2051 | 0.11044 | 0.01024 |

All compared source argmax decisions and four-token greedy outputs match.
FP16 and INT4 cached/full decoding also agree on all checked argmax decisions
at 7/33 tokens. These are regression cases, not a general quality guarantee.
Native suites pass 248 MPS and 369 CPU/shared tests; 194 Python tests cover
conversion, operators, complete tiny decoders, sampling, translation, storage
and benchmark metrics. Unsupported operators and unavailable external fixtures
remain explicit skips.

Eight opt-in dense-model tests also pass with `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca`: repeated 33/2048/2051-token requests
with and without compact attention, plus mixed 33/65-token batch greedy parity
against CPU with and without returned logits.

## Earlier 32-token end-to-end baseline

The same M1 configuration uses one generator worker, four CPU threads, compact
FP16 KV storage, identical saved prompt IDs, greedy decoding (TopK 1, TopP 1,
temperature 1, seed 42), and exactly 32 output tokens. Each shape has one first
request and two warm repeats in a fresh process per model format. Loading and
correctness-logit capture are excluded. The table reports warm medians; this
small run is a baseline, not a statistical performance guarantee.

| Prompt tokens | CT2 FP16 TTFT s | CT2 FP16 decode tok/s | CT2 INT4 TTFT s | CT2 INT4 decode tok/s |
|---:|---:|---:|---:|---:|
| 32 | 0.145 | 25.49 | 0.318 | 31.84 |
| 128 | 0.270 | 24.70 | 0.542 | 30.30 |
| 512 | 0.813 | 22.63 | 1.642 | 27.54 |
| 2048 | 4.224 | 19.41 | 7.007 | 22.78 |

First-request FP16 TTFT is 0.376/0.272/0.816/4.266 seconds respectively; INT4
is 0.408/0.542/1.642/7.006 seconds. All 12 outputs per format match the previous
qualified build exactly. Whisper's saved beam-five token hash and Marian's
eight beam/batch/length hypothesis sets are also unchanged; these local-model
smoke checks do not replace corpus WER/BLEU evaluation.

The saved llama.cpp F16 baseline on the same machine, source, tokenizer, prompt
IDs and sampling uses commit `bfb4308b058b334c6e68085c661ec9eb7e3d59f4`, all
29 layers on Metal, one slot, context 4096, batch 2048, microbatch 512, four CPU
threads, F16 K/V, automatic FlashAttention and disabled prompt reuse. Its warm
decode rates are 29.69/31.38/30.46/27.05 tok/s and TTFT is
0.070/0.144/0.499/2.307 seconds. All 32 generated FP16 tokens match CT2 at every
length and repeat. This is an existing baseline, not a new llama.cpp rerun.
CT2 still trails it in matched FP16 inference; the INT4 column is not a matching
Q4 comparison. INT4 prefill is also slower than FP16 here. Remaining targets are
expert dispatch overhead, expert GEMM utilization, and quantized prefill.

Reproduce CT2 with the benchmark command in the linked tools guide using
`--repeats 2`. Prompt SHA256 is
`bd5e66e4e53d5304966f89eafffe19be6bc11b11df9a12f0612634b9eee9579c`.
The measured 4.9.0 library SHA256 is
`1537be1d37d4d0df4f7da94e6d1274150656fed835ccb2c7eedc4abdb76e6fa4`.
Raw timings, prompt IDs and CPU-reference exports are generated artifacts and
are intentionally kept outside the source tree. This table records the earlier
`18c5bb3a` baseline; it is separate from the long-context A/B run below.

## Long-context A/B measurement

The same pinned MoE checkpoint and M1 configuration use FP16 weights/cache, one
worker, four CPU threads, greedy decoding, **16** output tokens, one first request
and two warm repeats per shape. Each variant runs in its own fresh process with
no competing build/model. Warm medians compare the retained generic kernel
(`CT2_MPS_DISABLE_VECTOR_CACHE=1`) against default vector decoding:

| Prompt tokens | Generic decode tok/s | Vector decode tok/s | Measured change |
|---:|---:|---:|---:|
| 512 | 21.99 | 22.48 | Same kernel; run variation |
| 2048 | 18.08 | 20.77 | +14.9% |
| 4096 | 14.18 | 16.84 | +18.8% |

All nine cold/warm 16-token outputs per variant exactly match the pre-change
baseline. At 4096 tokens, warm TTFT remains about 11.5 seconds for both variants;
this is a decode optimization, not a demonstrated prefill improvement. The
FP16 KV capacity after generation is 4608 tokens, approximately 504 MiB of
logical cache storage. RSS is recorded separately and is not total unified
memory or evidence of a memory improvement between these two FP16 variants.

The isolated 128-dimensional attention benchmark at 16K keys measures 1.93 ms
for vector FP16 versus 3.41 ms for generic FP16 (200 samples). This kernel-only
result does not imply the same end-to-end multiplier. Experimental quantized
kernel timings do not establish a usable model speedup while their quality
gates fail.

Native INT4 **weights** with an FP16 cache produce 25.89 warm decode tok/s at
2048 tokens with 32 outputs and two warm repeats. All three 32-token sequences
match the earlier qualified INT4-weight build. No matching llama.cpp Q4 result
is claimed. Expert execution/dispatch and memory bandwidth remain bottlenecks.

Reproduce the A/B run with the benchmark tool using `--kv-cache-type float16
--lengths 512 2048 4096 --output-tokens 16 --repeats 2`, then repeat in a fresh
process with `CT2_MPS_DISABLE_VECTOR_CACHE=1`. The measured source commit is
`1670cdb5ee5a41b9adf9a05a8379206c5bf6a369`; the later short-prefill consistency
change affects experimental quantized caches only. Prompt SHA256 is
`3bd666fce870400b85f8084479b895ad835971eb56d912838b14cd8835b70dce` and the
measured library SHA256 is
`a4047467d56b726bfcef1463bace65aa63fd292e2f3c85bfa26cc9118f33c41c`.

Validation of that earlier long-context snapshot includes 620 native tests,
194 existing Python regressions,
12 new cache-model tests and eight real dense-model tests. Native coverage
includes independent 16,385-key attention, GQA/MQA, masks, poisoned padding,
odd dimensions, growth and K/V gathering. The CPU-only build passes. Whisper
and Marian retain their saved token/hypothesis results. These passes qualify
the FP16 path and primitive behavior; the quantized real-model gates described
above remain failures.
