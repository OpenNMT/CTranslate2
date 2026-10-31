# Drama 3

Drama 3 provides experimental offline text-to-speech on CPU, with optional
CUDA or ROCm acceleration for backbone and tower projections. The native runtime
generates audio codes and decodes them to a mono waveform.

## Supported scope

Supported: offline synthesis, waveform decoding, and CPU reference-audio encoding
for voice cloning. Unsupported: voice design, a dedicated multi-speaker API,
speech editing or history, automatic long-text splitting, public streaming,
and tensor batching. Independent requests can be submitted together and scheduled
across execution replicas. Transcription is a separate speech-recognition task.

Reference encoding requires a checkpoint converted with encoder weights. Existing
decoder-only checkpoints still generate speech and accept precomputed reference codes.

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
The native generation API accepts prepared prompt rows, not plain text strings.

## Voice cloning

The client handles file decoding, mono conversion, resampling to 44100 Hz, and
transcription. CT2's `encode_reference` takes contiguous mono FLOAT32 samples
and returns raw INT32 codec indices `[frames, 10]`. Encoding runs on CPU,
including when generation uses a GPU. No Fish Speech or PyTorch is required at
inference time. Reconverting the official checkpoint includes the encoder weights;
calling this method with an older decoder-only checkpoint raises an error.

Using `model` and `tokenizer` from the example above:

```python
import soundfile as sf  # Optional client-side file reader.

waveform, rate = sf.read("reference.wav", dtype="float32")
if rate != 44100 or waveform.ndim != 1:
    raise ValueError("Prepare a mono 44100 Hz reference first")
codes = model.encode_reference(np.ascontiguousarray(waveform))
np.save("voice-codes.npy", codes)  # Reuse without encoding the reference again.
transcript = "The words spoken in the reference recording."
text = "New words to speak in the reference voice."
segments = (
    "<|im_start|>system\n",
    "convert the provided text to speech reference to the following:",
    "\n\n[Reference 1]\nText: <|speaker:0|> " + transcript + "\nSpeech: ",
    codes,
    "<|im_end|>\n",
    "<|im_start|>user\n", "<|speaker:0|> " + text, "<|im_end|>\n",
    "<|im_start|>assistant\n<|voice|>",
)
blocks = []
for segment in segments:
    if isinstance(segment, str):
        ids = tokenizer.encode(segment).ids
        rows = np.zeros((len(ids), 11), dtype=np.int32)
        rows[:, 0] = ids
    else:
        semantic_start = tokenizer.token_to_id("<|semantic:0|>")
        rows = np.column_stack((segment[:, 0] + semantic_start, segment))
    blocks.append(rows)
prompt_rows = np.ascontiguousarray(np.concatenate(blocks), dtype=np.int32)
result = model.generate(prompt_rows, max_frames=256)
```

References belong in the system message; completed generation history uses a
different format. Match speaker IDs in reference transcripts and target text.
Reuse codes only with the same codec weights, and keep reference rows within the
model context budget. Generation returns only newly generated audio. Float32 backend differences can
select different indices when two codebook candidates are nearly tied.

Local paths trigger no downloads. Synthesis requires no PyTorch, Transformers,
audio.cpp, or GGML.

See the [Python API reference](../python/ctranslate2.models.Drama3.rst) for
constructor options, generation parameters and defaults, and result fields.

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

Use `device="auto"` to select an available GPU. Supported compute types depend
on the backend; `default` and `auto` allow fallback.

Read `model.device`, `model.device_index`, and `model.compute_type` to inspect
the resolved execution settings. As with Whisper, `device_index` is a list
(for example, `[0]`). The compute type describes execution precision, independently
of the CPU checkpoint storage; some stages still use FLOAT32 activations.

Backbone and tower projections, including experts, run on GPU. The remaining
operations and waveform decoder run on CPU. Tensor parallelism is unsupported.

Expert weights upload on first use and remain cached, so GPU memory use grows
as new experts are selected. Limit this cache with
[`CT2_DRAMA3_GPU_EXPERT_CACHE_GB`](../environment_variables.md).
Existing converted checkpoints work unchanged; output audio and codes are CPU arrays.

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
