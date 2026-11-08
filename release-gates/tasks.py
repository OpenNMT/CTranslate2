"""Release-gate tasks returning metrics and optional generated outputs."""

import tempfile
import time

CPU_THREADS = 2
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"


def run_and_measure(infer, inputs, warmup_size=1, **kwargs):
    """Consume warm-up and measured outputs, including lazy generators."""
    warmup_size = min(warmup_size, max(1, len(inputs) // 2))
    list(infer(inputs[:warmup_size], **kwargs))
    start = time.perf_counter()
    results = list(infer(inputs, **kwargs))
    return results, time.perf_counter() - start


def _load_librispeech(limit):
    import io
    import soundfile
    from datasets import Audio, load_dataset
    # Parquet-backed mirror maintained by HF for CI; the canonical
    # `librispeech_asr` loading script was removed from the Hub.
    # decode=False avoids pulling torchcodec/ffmpeg system libs; we decode
    # the raw FLAC bytes with soundfile instead.
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for x in list(ds.take(limit)):
        audio, sr = soundfile.read(io.BytesIO(x["audio"]["bytes"]))
        out.append((audio, sr, x["text"]))
    return out


def run_whisper_wer(limit=300, device="cpu"):
    import jiwer
    from faster_whisper import WhisperModel

    samples = _load_librispeech(limit)
    model = WhisperModel("tiny", device=device, compute_type="int8",
                         cpu_threads=CPU_THREADS, num_workers=1)

    def transcribe(batch):
        for audio, _, _ in batch:
            segments, _ = model.transcribe(audio, language="en", beam_size=1,
                                           temperature=0.0, vad_filter=False)
            yield " ".join(s.text for s in segments).strip().lower()

    hyps, elapsed = run_and_measure(transcribe, samples)
    refs = [text.strip().lower() for _, _, text in samples]
    audio_sec = sum(len(audio) / sr for audio, sr, _ in samples)

    return {"wer": jiwer.wer(refs, hyps),
            "rtf_inverse": audio_sec / elapsed,
            "num_samples": len(samples)}


def run_opus_bleu(limit=500, device="cpu"):
    import ctranslate2
    import sacrebleu
    import transformers
    from ctranslate2.converters.transformers import TransformersConverter

    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    src_path = sacrebleu.utils.get_source_file("wmt14", "en-de")
    ref_paths = sacrebleu.utils.get_reference_files("wmt14", "en-de")
    sources = open(src_path, encoding="utf-8").read().splitlines()[:limit]
    refs = [open(p, encoding="utf-8").read().splitlines()[:limit]
            for p in ref_paths]

    with tempfile.TemporaryDirectory() as tmp:
        TransformersConverter(model_name).convert(tmp, force=True)
        translator = ctranslate2.Translator(tmp, device=device,
                                             intra_threads=CPU_THREADS, inter_threads=1)
        tokenized = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s))
                     for s in sources]
        total_tokens = sum(len(t) for t in tokenized)
        results, elapsed = run_and_measure(
            translator.translate_batch, tokenized, warmup_size=16,
            beam_size=4, max_batch_size=16)

    hyps = [tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0]),
                             skip_special_tokens=True)
            for r in results]
    return {"bleu": sacrebleu.corpus_bleu(hyps, refs).score,
            "tokens_per_sec": total_tokens / elapsed,
            "num_sentences": len(sources)}


def run_t5gemma_generation(limit=50, device="cpu"):
    import ctranslate2
    import transformers
    from datasets import load_dataset
    from ctranslate2.converters.transformers import TransformersConverter

    model_name = "jordimas/t5gemma-2-270m-270m"
    revision = "dfb6d0d619d8e6d42f279a6197abb90dec0137c1"
    samples = load_dataset("rajpurkar/squad", split="validation",
                           revision=SQUAD_REVISION).shuffle(seed=42)[:limit]
    prompts = [f"Context: {context} Question: {question} Answer:"
               for context, question in zip(samples["context"], samples["question"])]
    if not prompts:
        raise ValueError("Generation requires at least one prompt")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, revision=revision)
    inputs = [tokenizer.convert_ids_to_tokens(tokenizer.encode(p, truncation=True, max_length=512))
              for p in prompts]
    with tempfile.TemporaryDirectory() as tmp:
        TransformersConverter(model_name, revision=revision).convert(
            tmp, force=True, quantization="float32")
        translator = ctranslate2.Translator(tmp, device=device, compute_type="float32",
                                             intra_threads=CPU_THREADS, inter_threads=1)
        results, elapsed = run_and_measure(
            translator.translate_batch, inputs, warmup_size=8,
            beam_size=1, sampling_topk=1, max_decoding_length=64, max_batch_size=8)
    outputs = [tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0]),
                                skip_special_tokens=True).strip() for r in results]
    if len(outputs) != len(prompts) or any(not output for output in outputs):
        raise ValueError("Generation returned missing or empty completions")
    return {"outputs": outputs, "num_prompts": len(prompts),
            "tokens_per_sec": sum(len(r.hypotheses[0]) for r in results) / elapsed}


TASKS = {"whisper_wer": run_whisper_wer, "opus_bleu": run_opus_bleu,
         "t5gemma_generation": run_t5gemma_generation}


if __name__ == "__main__":
    import argparse, json, sys
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=TASKS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    kw = {"device": a.device}
    if a.limit is not None:
        kw["limit"] = a.limit
    json.dump(TASKS[a.task](**kw), sys.stdout)
    sys.stdout.write("\n")
