# Drama 3

Drama 3 provides experimental offline text-to-speech on CPU, with optional
CUDA or ROCm acceleration for backbone and tower projections. The native runtime
generates audio codes and decodes them to a mono waveform.

## Supported scope

Supported: offline synthesis and waveform decoding. Unsupported: reference-audio
voice cloning, audio encoding, voice design, a dedicated multi-speaker API,
speech editing or history, automatic long-text splitting, public streaming,
and tensor batching. Independent requests can be submitted together and scheduled
across execution replicas. Transcription is a separate speech-recognition task.

Voice cloning requires a reference-audio encoder and reference-conditioned
prompt preparation; the waveform decoder cannot encode reference recordings.

## Model conversion

Convert the official Drama 3 checkpoint with the Transformers converter:

```bash
ct2-transformers-converter \
  --model /path/to/drama-3-official \
  --output_dir /path/to/drama-3-ct2 \
  --quantization float32 --dry-run
```

Remove `--dry-run` to convert. A dry run audits mappings and shapes without
loading weights or creating the output directory; `--report plan.json` saves
the audit. No files are downloaded. Use `--force` to replace an existing directory.

The Python API uses the standard CTranslate2 `Converter` interface:

```python
from ctranslate2.converters import TransformersConverter

converter = TransformersConverter("/path/to/drama-3-official")
converter.convert("/path/to/drama-3-ct2", quantization="float32")
```

Fish Speech is not required. Use `--quantization int8_float32` for INT8 weights
with FLOAT32 activations; codec weights remain FLOAT32. See the
[quantization guide](../quantization.md) for other options. Vocabulary mapping
is unsupported.

Conversion holds model and codec weights in RAM before saving. A dry run does
not verify speech quality.

## Text-to-speech

See the [official prompting guide](https://docs.fish.audio/features/text-to-speech/drama-3)
for speaker tags, bracketed performance directions, paired delivery tags, and
phoneme markers. Bracketed directions are ordinary text, not a fixed token
vocabulary. The guide describes the upstream model; CT2's supported scope is above.

Install `tokenizers` for this single-speaker example. It prepares text-only
prompt rows with the converted model's local tokenizer and saves a mono PCM16 WAV:

```python
from pathlib import Path
import wave

import numpy as np
from tokenizers import Tokenizer
from ctranslate2.models import Drama3

model_path = Path("/path/to/drama-3-ct2")
tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
tokenizer.no_padding()
tokenizer.no_truncation()
text = "Hello. This is a speech synthesis test."
segments = (
    "<|im_start|>system\n", "convert the provided text to speech",
    "<|im_end|>\n", "<|im_start|>user\n", "<|speaker:0|> " + text,
    "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>",
)
ids = [token for segment in segments for token in tokenizer.encode(segment).ids]
prompt_rows = np.zeros((len(ids), 11), dtype=np.int32)
prompt_rows[:, 0] = ids

model = Drama3(str(model_path), intra_threads=8, compute_type="default")
result = model.generate(prompt_rows, max_frames=256, seed=0)
print(result["sample_rate"], result["audio"].shape, result["finish_reason"])
pcm = np.rint(np.clip(result["audio"], -1, 1) * 32767).astype("<i2")
with wave.open("hello.wav", "wb") as wav:
    wav.setparams((1, 2, result["sample_rate"], 0, "NONE", "not compressed"))
    wav.writeframes(pcm.tobytes())
```

Encode segments separately to preserve token boundaries; the generic chat
template cannot replace this prompt. Input text must not contain reserved prompt
tokens or speaker tags. `prompt_rows` must be contiguous INT32 `[rows, 11]`:
tokenizer IDs in column 0 and zeros in the ten codec columns for text rows.
The native API does not accept plain text strings or encode reference audio.

Local paths trigger no downloads. Synthesis requires no PyTorch, Transformers,
audio.cpp, or GGML.

See the [Python API reference](../python/ctranslate2.models.Drama3.rst) for
constructor options, generation parameters and defaults, and result fields.

### Concurrent and asynchronous requests

Drama3 uses CTranslate2's replica pool. Set `inter_threads` to the number of
requests that can execute concurrently; `intra_threads` controls CPU threads per
replica. Replicas share immutable CPU checkpoint weights and own separate layers,
GPU weight caches, and sequence state. Additional GPU replicas require additional
VRAM; `CT2_DRAMA3_GPU_EXPERT_CACHE_GB` applies to each replica.

```python
model = Drama3(str(model_path), intra_threads=4, inter_threads=2)
pending = model.generate(prompt_rows, max_frames=256, asynchronous=True)
result = pending.result()  # Waits and propagates generation errors.
print(result.sample_rate, result.audio.shape)

results = model.generate_batch([prompt_rows, prompt_rows], max_frames=256)
```

`generate` returns a typed `Drama3Result`; existing dictionary-style access such
as `result["audio"]` remains supported, and `result.to_dict()` returns a dictionary.
`asynchronous=True` returns `Drama3ResultAsync` with `done()` and `result()` methods.
`generate_batch` returns results in input order, or a list of async results when
requested. Each prompt has independent sequence state and uses the supplied seed.
Kernels still process one sequence at a time within each replica.

`max_queued_batches` limits queued requests: `0` allows four per replica, and a
negative value removes the limit. Submitting to a full queue waits for space,
including asynchronous submissions. `num_replicas`, `num_queued_batches`, and
`num_active_batches` expose pool activity.

### CPU execution

The synthesis graph and codec use FLOAT32 activations. For FP8 storage with
FLOAT32 execution, selected experts are decoded on demand. INT8/INT16 execution
decodes and quantizes matrices at load time, requiring temporary memory.
All experts remain resident in RAM; sparse selection reduces computation,
not storage.

### CUDA and ROCm

Build with `WITH_CUDA=ON` for NVIDIA or `WITH_HIP=ON` for AMD (see
[installation](../installation.md)). Both use `device="cuda"`:

```python
model = Drama3(
    str(model_path), device="cuda", device_index=0, compute_type="float16"
)
```

`device="auto"` selects an available GPU. Projection types follow backend support:
FLOAT32, FLOAT16, BFLOAT16, and supported INT8 variants. Unsupported explicit
types raise an error; `default` and `auto` allow fallback.

Hybrid execution runs backbone and tower dense projections, including routed
experts, on GPU; attention, recurrent updates, routing, sampling, embeddings,
and the waveform codec run on CPU. Activations transfer per projection.
The graph is not entirely GPU-resident; tensor parallelism is unsupported.

By default, supported byte-sized host expert weights are memory-mapped on Linux,
with other parameters loaded
as FLOAT32 regardless of GPU precision. Weights upload on first use and stay
cached, so GPU memory grows as new experts are selected. FLOAT32 execution
caches FP8 expert bytes and decodes them on GPU per projection; FLOAT16/BFLOAT16
decode once on upload, while INT8 quantizes projections on first use. Existing converted checkpoints work unchanged;
output audio and codes remain CPU arrays.

### Loading options

CPU inference loads weights into allocated RAM. On Linux, the optional GPU
expert cache maps supported INT8 and FP8 expert weights from local checkpoints;
see `CT2_DRAMA3_GPU_EXPERT_CACHE_GB` above.

### Memory configuration

Memory use is controlled by checkpoint quantization and the sizes of individual
allocations, without a guaranteed total RAM ceiling:

- The default FP8 checkpoint is supported directly; no separate INT8 model is
  required. Compute type and checkpoint quantization affect weight allocations.
  Selecting INT8 compute for FP8 weights performs runtime conversion and allocates
  the converted tensors.
- Limit prompt and history length and `max_frames` to bound generated context
  and output allocations. The checkpoint's maximum context remains the upper bound.
- Lower `prefill_chunk_size` to reduce prompt-processing intermediate allocations.
  It does not reduce retained history or enable streaming. Generation accepts one
  sequence per replica at a time; independent calls use the replica pool.
- [`CT2_DRAMA3_GPU_EXPERT_CACHE_GB`](../environment_variables.md)
  bounds cached GPU expert weights, rather than total VRAM, on CUDA and ROCm.
  Other allocations require additional memory.

Drama3 currently has no independent KV-cache precision setting. Compute type does
not imply that its attention cache uses that precision. CT2 does not create
systemd scopes or configure process memory limits.
