# MPS quality benchmarks

These host-native benchmarks compare CPU and MPS with the same model, input,
search settings, and sample order. Model loading and one warm-up inference are
reported separately and are not included in inference time.

Install the optional dependencies in the environment containing this
CTranslate2 build:

```bash
python -m pip install -r tools/benchmark/quality/requirements.txt
```

## Whisper WER

The Whisper benchmark follows the faster-whisper reference setup: the
LibriSpeech `clean` validation split, English normalization, and corpus WER.
The default model is `Systran/faster-whisper-small.en`, with beam size 5 on
both devices. The spelling normalizer is pinned to faster-whisper commit
`ed9a06cd89a93e47838f564998a6c09b655d7f43`, downloaded once, and verified by
SHA-256 before use.

Use a short run to verify the setup:

```bash
python tools/benchmark/quality/whisper_wer.py \
  --max-samples 25 \
  --output whisper-smoke.json
```

Omit `--max-samples` for the reportable 2,703-utterance result:

```bash
python tools/benchmark/quality/whisper_wer.py \
  --model Systran/faster-whisper-small.en \
  --beam-size 5 \
  --cpu-compute-type float32 \
  --mps-compute-type float16 \
  --output whisper-librispeech-clean.json
```

The model and dataset are downloaded on first use and then read from the local
Hugging Face cache. A subset is useful for correctness and estimating runtime,
but its WER should not be published as the LibriSpeech validation result.

## Translation BLEU

This extends the existing CTranslate2 translation benchmark with a native MPS
path. It uses SacreBLEU's WMT14 English-German files and the OPUS-MT model that
is already represented under `tools/benchmark/opus_mt_ende`.

Convert the model once without quantizing the stored weights:

```bash
ct2-transformers-converter \
  --model Helsinki-NLP/opus-mt-en-de \
  --output_dir opus-mt-en-de-ct2
```

Then run a smoke comparison:

```bash
python tools/benchmark/quality/translation_bleu.py \
  --model opus-mt-en-de-ct2 \
  --max-samples 64 \
  --output translation-smoke.json
```

Omit `--max-samples` for the reportable full WMT14 result:

```bash
python tools/benchmark/quality/translation_bleu.py \
  --model opus-mt-en-de-ct2 \
  --beam-size 4 \
  --batch-size 4 \
  --length-penalty 0 \
  --cpu-compute-type float32 \
  --mps-compute-type float16 \
  --output translation-wmt14-en-de.json
```

The default beam size (4), length penalty (0), and batch size (32) match the
existing CTranslate2 WMT14 benchmark. The measured MPS result below uses batch
size 4, which was selected before the full evaluation and is specified
explicitly in the reproduction command. The JSON output records the model,
compute types, settings, platform, timings, BLEU scores, output agreement, and
individual hypotheses needed to investigate any CPU/MPS difference.

## Measured Apple M1 results

These full-dataset measurements were collected with CTranslate2 4.8.1 from a
native arm64 Release build on a MacBook Air with an Apple M1 (7-core GPU), 8 GB
unified memory, and macOS 26.5.2. Model loading and the warm-up inference are
excluded from inference time. CPU and MPS used the same model, inputs, sample
order, and decoding settings; the compute types are reported separately.

### LibriSpeech clean validation

Model: `Systran/faster-whisper-small.en`; 2,703 utterances; beam size 5.

| Device | Compute type | WER | Inference time | Audio speed |
|---|---:|---:|---:|---:|
| CPU | FP32 | 2.9125% | 7,929.29 s | 2.45x realtime |
| MPS | FP16 | 2.9089% | 4,114.88 s | 4.71x realtime |

MPS was 1.93x faster, with a WER difference of -0.0036 percentage points and
99.63% exact normalized transcript agreement.

### WMT14 English-German

Model: `Helsinki-NLP/opus-mt-en-de`; 2,737 sentences; beam size 4; batch size
4; length penalty 0.

| Device | Compute type | BLEU | Inference time | Output tokens/s |
|---|---:|---:|---:|---:|
| CPU | FP32 | 27.9152 | 665.24 s | 107.69 |
| MPS | FP16 | 27.9332 | 312.97 s | 228.89 |

MPS was 2.13x faster, with a BLEU difference of +0.0180 and 96.60% exact
translation agreement. The SacreBLEU signature was
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`.
