# Drama 3 voice-cloning test plan

Low word error rate (WER) measures intelligibility, not voice identity. Evaluate
content, speaker similarity, numerical parity, and performance separately. Freeze
settings before generation; retain failures without retries or seed selection.

## Component and compatibility coverage

| Area | Cases | Acceptance |
|---|---|---|
| Codec parity | Silence, sine waves, speech, and sequences crossing the attention window | Official indices, except an alternate within `1e-5` normalized squared distance of the nearest codebook entry |
| Padding | 1, 2048, 2049, 8193, 270337 samples | `ceil(samples / 2048)` frames, ten codebooks |
| Invalid PCM | Empty, rank two, wrong dtype, NaN, infinity | Clear exception; pool remains usable |
| Conversion | Complete/missing/wrong-shaped weights and weight normalization | Complete graph mapped; incomplete graph rejected |
| Compatibility | Decoder-only checkpoints and existing synthesis tests | Generation still works; encoding reports missing weights |
| Client persistence | Encode, save `.npy`, reload without pickle | Identical codes and prompt rows |
| Determinism | Repeat the first cloned request with the same settings | Identical generated codes |
| Existing decoder | Native Fish DAC and Drama3 suites | Tests pass; document unavailable GPU/large-fixture checks |

Run component checks with a native build and its Python binding:

```bash
build/tests/ctranslate2_test tests/data --gtest_filter='FishDAC*:*Drama3*'
python -m unittest discover -s python/tests -p test_drama3_codec_spec.py
python -m unittest discover -s python/tests -p test_drama3_runtime.py
python -m unittest discover -s python/tests -p test_drama3_voice_evaluation.py
```

Generate the optional official parity fixture using Fish Speech and PyTorch:

```bash
PYTHONPATH=python:/path/to/fish-speech python tools/generate_drama3_reference_fixture.py \
  --checkpoint /path/to/codec.pth --output /path/to/new-fixture \
  --audio /path/to/mono-44100-speech.wav
CT2_DRAMA3_REFERENCE_FIXTURE=/path/to/new-fixture \
  build/tests/ctranslate2_test tests/data --gtest_filter='FishDACReferenceTest.*'
```

The fixture records only officially near-tied alternatives, not an unrestricted
matching-percentage tolerance. Native repeated encoding must be deterministic.

## End-to-end matrix and fixed acceptance gates

Use two distinct generated reference voices with accurate transcripts and clean
mono PCM16 WAV recordings at 44100 Hz. Each reference must last 1–30 seconds.
Use the committed English/Spanish narration/instruction corpus, seeds `0,17`,
temperature `0.7`, top-p `0.7`, top-k `30`, and a 320-frame generation limit.
This produces 16 cloned samples plus four unconditioned seed-zero controls.

- Aggregate cloned WER ≤5%; each voice/language group ≤10%; each utterance ≤15%.
- Reference transcripts have WER ≤10%, otherwise the setup fails.
- Every cloned output is finite, nonempty, RMS >0.001, and finishes with EOS.
- The entire matrix must be scored; interrupted studies cannot pass.
- Record unconditioned controls separately, including WER and audio health.
- Independently compare speaker embeddings with both references; the target
  must be preferred in at least 75% of cloned samples. Report all cosine scores.
- Human listening remains a separate gate for similarity, leakage of reference
  words, pronunciation, clipping, missing speech, and unnatural artifacts.

Create a references JSON file:

```json
[
  {"id":"voice-a", "audio":"/path/a.wav", "text":"Transcript A.", "language":"en"},
  {"id":"voice-b", "audio":"/path/b.wav", "text":"Transcript B.", "language":"es"}
]
```

Run generation and faster-whisper ASR in separate processes to release TTS weights:

```bash
python tools/evaluate_drama3_voice_cloning.py generate \
  --model /path/to/converted-drama3 --references references.json --output results
python tools/evaluate_drama3_voice_cloning.py asr \
  --asr-model /path/to/faster-whisper-medium --references references.json --output results
```

ASR uses CPU INT8, beam size five, an explicit language, no VAD filtering, and no
previous-text conditioning. Resample to 16 kHz. WER uses Unicode NFC, case folding,
punctuation removal, preserved diacritics, and word-level Levenshtein distance.
Report edit counts and word counts; aggregate by total words, not mean percentages.

For independent speaker verification, use the local
[WavLM Base Plus SV](https://huggingface.co/microsoft/wavlm-base-plus-sv) checkpoint:

```bash
python tools/evaluate_drama3_speakers.py \
  --model /path/to/wavlm-base-plus-sv --references references.json --output results
```

This phase requires PyTorch, Transformers, SoundFile, and SciPy. It uses 16 kHz
input and normalized speaker embeddings. The relative comparison is an engineering
gate, not a calibrated universal speaker-identification threshold.

## Evidence and broader release coverage

Preserve manifests, reference hashes, native/build identity, model precision,
ASR checkpoint identity, generated WAVs/codes, timings, errors, and all scores.
Resume only with an unchanged manifest. Never silently overwrite failed samples.
Record encode and generation time without claiming speedups absent a baseline.

Before broader release, extend to more voices, reference lengths, noisy recordings,
long-form reference reuse, concurrent encoding/generation, CPU backends, and
CUDA/ROCm generation with CPU encoding. Report these as pending when untested.
A passing local CPU matrix does not establish perceptual parity or production
readiness, and WER alone never establishes successful voice cloning.
