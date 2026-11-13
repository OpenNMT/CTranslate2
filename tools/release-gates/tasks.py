"""Release-gate tasks returning metrics and optional generated outputs."""

import os
import tempfile
import time

CPU_THREADS = 2
COMPUTE_TYPE = os.environ.get("CT2_COMPUTE_TYPE", "default")
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


def load_task_model(task, device, compute_type, model_dir=None):
    if task == "whisper_wer":
        from faster_whisper import WhisperModel
        return WhisperModel("tiny", device=device, compute_type=compute_type,
                            cpu_threads=CPU_THREADS, num_workers=1)
    import ctranslate2
    return ctranslate2.Translator(model_dir, device=device, compute_type=compute_type,
                                 intra_threads=CPU_THREADS, inter_threads=1)


def effective_compute_type(task, model):
    return (model.model if task == "whisper_wer" else model).compute_type


def resolve_compute_types(task, device="cpu", model_dir=None):
    resolved = {}
    for requested in ("default", "auto"):
        model = load_task_model(task, device, requested, model_dir)
        resolved[requested] = effective_compute_type(task, model)
        del model
    return resolved


def run_whisper_wer(limit=300, device="cpu"):
    import jiwer

    samples = _load_librispeech(limit)
    model = load_task_model("whisper_wer", device, COMPUTE_TYPE)

    def transcribe(batch):
        for audio, _, _ in batch:
            segments, _ = model.transcribe(audio, language="en",
                                           temperature=0.0, vad_filter=False)
            yield " ".join(s.text for s in segments).strip().lower()

    hyps, elapsed = run_and_measure(transcribe, samples)
    refs = [text.strip().lower() for _, _, text in samples]
    audio_sec = sum(len(audio) / sr for audio, sr, _ in samples)

    return {"wer": jiwer.wer(refs, hyps),
            "rtf_inverse": audio_sec / elapsed,
            "num_samples": len(samples),
            "compute_type": effective_compute_type("whisper_wer", model)}


OPUS_MODEL = "Helsinki-NLP/opus-mt-en-de"
T5GEMMA_MODEL = "jordimas/t5gemma-2-270m-270m"
T5GEMMA_REVISION = "dfb6d0d619d8e6d42f279a6197abb90dec0137c1"


def _converted_dir(stack, model_dir, convert):
    if model_dir:
        return model_dir
    tmp = stack.enter_context(tempfile.TemporaryDirectory())
    convert(tmp)
    return tmp


def convert_opus_bleu(out_dir):
    from ctranslate2.converters.transformers import TransformersConverter
    TransformersConverter(OPUS_MODEL).convert(out_dir, force=True)


def convert_t5gemma_generation(out_dir):
    from ctranslate2.converters.transformers import TransformersConverter
    TransformersConverter(T5GEMMA_MODEL, revision=T5GEMMA_REVISION).convert(
        out_dir, force=True, quantization="float32")


def run_opus_bleu(limit=500, device="cpu", model_dir=None):
    import contextlib
    import sacrebleu
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(OPUS_MODEL)

    src_path = sacrebleu.utils.get_source_file("wmt14", "en-de")
    ref_paths = sacrebleu.utils.get_reference_files("wmt14", "en-de")
    sources = open(src_path, encoding="utf-8").read().splitlines()[:limit]
    refs = [open(p, encoding="utf-8").read().splitlines()[:limit]
            for p in ref_paths]

    with contextlib.ExitStack() as stack:
        model_path = _converted_dir(stack, model_dir, convert_opus_bleu)
        translator = load_task_model("opus_bleu", device, COMPUTE_TYPE, model_path)
        tokenized = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s))
                     for s in sources]
        total_tokens = sum(len(t) for t in tokenized)
        results, elapsed = run_and_measure(
            translator.translate_batch, tokenized, warmup_size=16,
            max_batch_size=16)

    hyps = [tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0]),
                             skip_special_tokens=True)
            for r in results]
    return {"bleu": sacrebleu.corpus_bleu(hyps, refs).score,
            "tokens_per_sec": total_tokens / elapsed,
            "num_sentences": len(sources), "compute_type": translator.compute_type}


def run_t5gemma_generation(limit=50, device="cpu", model_dir=None):
    import contextlib
    import transformers
    from datasets import load_dataset

    samples = load_dataset("rajpurkar/squad", split="validation",
                           revision=SQUAD_REVISION).shuffle(seed=42)[:limit]
    prompts = [f"Context: {context} Question: {question} Answer:"
               for context, question in zip(samples["context"], samples["question"])]
    if not prompts:
        raise ValueError("Generation requires at least one prompt")
    tokenizer = transformers.AutoTokenizer.from_pretrained(T5GEMMA_MODEL, revision=T5GEMMA_REVISION)
    inputs = [tokenizer.convert_ids_to_tokens(tokenizer.encode(p, truncation=True, max_length=512))
              for p in prompts]
    with contextlib.ExitStack() as stack:
        model_path = _converted_dir(stack, model_dir, convert_t5gemma_generation)
        translator = load_task_model("t5gemma_generation", device, COMPUTE_TYPE, model_path)
        results, elapsed = run_and_measure(
            translator.translate_batch, inputs, warmup_size=8,
            sampling_topk=1, max_decoding_length=64, max_batch_size=8)
    outputs = [tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0]),
                                skip_special_tokens=True).strip() for r in results]
    if len(outputs) != len(prompts):
        raise ValueError("Generation returned missing completions")
    return {"outputs": outputs, "num_prompts": len(prompts),
            "tokens_per_sec": sum(len(r.hypotheses[0]) for r in results) / elapsed,
            "compute_type": translator.compute_type}


TASKS = {"whisper_wer": run_whisper_wer, "opus_bleu": run_opus_bleu,
         "t5gemma_generation": run_t5gemma_generation}

CONVERTERS = {"opus_bleu": convert_opus_bleu,
              "t5gemma_generation": convert_t5gemma_generation}


if __name__ == "__main__":
    import argparse, json, sys
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("task", choices=TASKS)
    run_p.add_argument("--limit", type=int, default=None)
    run_p.add_argument("--device", default="cpu")
    run_p.add_argument("--model-dir", default=None)
    resolve_p = sub.add_parser("resolve")
    resolve_p.add_argument("task", choices=TASKS)
    resolve_p.add_argument("--device", default="cpu")
    resolve_p.add_argument("--model-dir", default=None)
    conv_p = sub.add_parser("convert")
    conv_p.add_argument("task", choices=CONVERTERS)
    conv_p.add_argument("out_dir")
    a = p.parse_args()
    if a.cmd == "convert":
        CONVERTERS[a.task](a.out_dir)
    elif a.cmd == "resolve":
        json.dump(resolve_compute_types(a.task, a.device, a.model_dir), sys.stdout)
        sys.stdout.write("\n")
    else:
        kw = {"device": a.device}
        if a.limit is not None:
            kw["limit"] = a.limit
        if a.model_dir is not None and a.task in CONVERTERS:
            kw["model_dir"] = a.model_dir
        json.dump(TASKS[a.task](**kw), sys.stdout)
        sys.stdout.write("\n")
