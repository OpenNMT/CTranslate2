"""Run generation and faster-whisper WER in separate processes; no retries.

References JSON: [{"id":"voice-a", "audio":"/path/a.wav", "text":"Transcript",
"language":"en"}, ...]. Use mono PCM16 WAV references at 44100 Hz.
"""

import argparse
import hashlib
import json
import re
import time
import unicodedata
import wave

from pathlib import Path

import numpy as np


def read_audio(path, rate=44100):
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16 WAV")
        source_rate = reader.getframerate()
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    if source_rate == rate:
        return pcm.astype(np.float32) / 32768
    import av

    frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = source_rate
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
    frames = resampler.resample(frame) + resampler.resample(None)
    return np.ascontiguousarray(
        np.concatenate([f.to_ndarray().ravel() for f in frames])
    )


def word_errors(reference, hypothesis):
    def words(text):
        return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())

    reference, hypothesis = words(reference), words(hypothesis)
    row = list(range(len(hypothesis) + 1))
    for i, token in enumerate(reference, 1):
        previous, row = row, [i]
        for j, other in enumerate(hypothesis, 1):
            row.append(
                min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other))
            )
    return {
        "errors": row[-1],
        "words": len(reference),
        "wer": row[-1] / max(1, len(reference)),
    }


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def prompt_rows(tokenizer, text, reference=None):
    segments = ["<|im_start|>system\n"]
    if reference is None:
        segments += ["convert the provided text to speech"]
    else:
        transcript, codes = reference
        segments += [
            "convert the provided text to speech reference to the following:",
            "\n\n[Reference 1]\nText: <|speaker:0|> " + transcript + "\nSpeech: ",
            codes,
        ]
    segments += [
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "<|speaker:0|> " + text,
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    blocks = []
    for segment in segments:
        if isinstance(segment, str):
            ids = tokenizer.encode(segment).ids
            block = np.zeros((len(ids), 11), dtype=np.int32)
            block[:, 0] = ids
        else:
            block = np.column_stack(
                (segment[:, 0] + tokenizer.token_to_id("<|semantic:0|>"), segment)
            )
        blocks.append(block)
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.int32)


def generate(args, references, cases):
    from tokenizers import Tokenizer

    import ctranslate2

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    manifest = {
        "references": references,
        "cases": cases,
        "seeds": args.seeds,
        "model": str(args.model.resolve()),
        "threads": args.threads,
        "sampling": {"temperature": 0.7, "top_p": 0.7, "top_k": 30},
        "max_frames": 320,
        "ctranslate2_version": ctranslate2.__version__,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(
            Path(ctranslate2._ext.__file__).read_bytes()
        ).hexdigest(),
    }
    manifest["reference_sha256"] = {
        r["id"]: hashlib.sha256(Path(r["audio"]).read_bytes()).hexdigest()
        for r in references
    }
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Resume manifest differs; use a fresh output directory")
    save_json(path, manifest)
    model = ctranslate2.models.Drama3(str(args.model), intra_threads=args.threads)
    prepared = {}
    for reference in references:
        waveform = read_audio(reference["audio"])
        if not 44100 <= len(waveform) <= 30 * 44100:
            raise ValueError("Use references between one and thirty seconds")
        codes = model.encode_reference(waveform)
        file = args.output / (reference["id"] + ".npy")
        np.save(file, codes)
        restored = np.load(file, allow_pickle=False)
        np.testing.assert_array_equal(codes, restored)
        prepared[reference["id"]] = (reference["text"], restored)
    records_path = args.output / "generation.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    done = {r["id"] for r in records}
    for voice in [*prepared, "unconditioned"]:
        for case in cases:
            for seed in args.seeds if voice != "unconditioned" else args.seeds[:1]:
                key = f"{voice}-{case['id']}-seed{seed}"
                if key in done:
                    continue
                record = dict(case, id=key, voice=voice, seed=seed)
                started = time.perf_counter()
                print("BEGIN " + key, flush=True)
                try:
                    prompt = prompt_rows(tokenizer, case["text"], prepared.get(voice))
                    result = model.generate(
                        prompt,
                        max_frames=320,
                        temperature=0.7,
                        top_p=0.7,
                        top_k=30,
                        seed=seed,
                    )
                    audio = result.audio
                    with wave.open(str(args.output / (key + ".wav")), "wb") as writer:
                        writer.setparams(
                            (1, 2, result.sample_rate, 0, "NONE", "not compressed")
                        )
                        writer.writeframes(
                            np.rint(np.clip(audio, -1, 1) * 32767)
                            .astype("<i2")
                            .tobytes()
                        )
                    np.save(args.output / (key + ".codes.npy"), result.codes)
                    record.update(
                        status="ok",
                        finish_reason=result.finish_reason,
                        duration=len(audio) / result.sample_rate,
                        finite=bool(np.isfinite(audio).all()),
                        rms=float(np.sqrt(np.mean(audio**2))) if len(audio) else 0,
                        codes_sha256=hashlib.sha256(result.codes.tobytes()).hexdigest(),
                    )
                    if not records:
                        repeat = model.generate(
                            prompt,
                            max_frames=320,
                            temperature=0.7,
                            top_p=0.7,
                            top_k=30,
                            seed=seed,
                        )
                        record["repeat_equal"] = bool(
                            np.array_equal(result.codes, repeat.codes)
                        )
                except Exception as error:
                    record.update(
                        status="error", error=f"{type(error).__name__}: {error}"
                    )
                record["seconds"] = time.perf_counter() - started
                records.append(record)
                save_json(records_path, records)
                print(json.dumps(record, ensure_ascii=False), flush=True)


def asr(args, references):
    from faster_whisper import WhisperModel

    recognizer = WhisperModel(
        str(args.asr_model),
        device="cpu",
        compute_type="int8",
        cpu_threads=args.threads,
        local_files_only=True,
    )
    records = json.loads((args.output / "generation.json").read_text())
    manifest = json.loads((args.output / "manifest.json").read_text())
    reference_results = []
    for record in [*references, *records]:
        if record.get("status", "ok") != "ok":
            continue
        path = record.get("audio", args.output / (record["id"] + ".wav"))
        segments, _ = recognizer.transcribe(
            read_audio(path, 16000),
            language=record["language"],
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        hypothesis = " ".join(segment.text.strip() for segment in segments)
        record["asr"] = dict(text=hypothesis, **word_errors(record["text"], hypothesis))
        if "voice" not in record:
            reference_results.append(record)
        print(record["id"], record["asr"], flush=True)
        save_json(
            args.output / "asr.json",
            {"references": reference_results, "cases": records},
        )
    cloned = [r for r in records if r["voice"] != "unconditioned"]
    scored = [r for r in cloned if "asr" in r]
    wer = sum(r["asr"]["errors"] for r in scored) / max(
        1, sum(r["asr"]["words"] for r in scored)
    )
    groups = {}
    for voice in {r["voice"] for r in cloned}:
        for language in {r["language"] for r in cloned}:
            subset = [
                r["asr"]
                for r in scored
                if r["voice"] == voice and r["language"] == language
            ]
            groups[voice + "/" + language] = sum(r["errors"] for r in subset) / max(
                1, sum(r["words"] for r in subset)
            )
    expected = len(references) * len(manifest["cases"]) * len(manifest["seeds"])
    passed = (
        len(scored) == len(cloned) == expected
        and expected > 0
        and wer <= 0.05
        and all(value <= 0.10 for value in groups.values())
        and all(
            r["asr"]["wer"] <= 0.15
            and r["finish_reason"] == "eos"
            and r["finite"]
            and r["rms"] > 0.001
            for r in scored
        )
        and len(reference_results) == len(references)
        and all(r["asr"]["wer"] <= 0.10 for r in reference_results)
        and any(r.get("repeat_equal") is True for r in records)
    )
    summary = {
        "content_pass": passed,
        "micro_wer": wer,
        "group_wer": groups,
        "cloned_cases": len(cloned),
        "scored_cases": len(scored),
        "asr_model": str(args.asr_model.resolve()),
        "beam_size": 5,
    }
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "asr"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/data/drama3/voice_cloning_cases.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17])
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if (args.phase == "generate" and args.model is None) or (
        args.phase == "asr" and args.asr_model is None
    ):
        parser.error("provide --model for generation or --asr-model for ASR")
    references = json.loads(args.references.read_text())
    if len(references) < 2 or len({r["id"] for r in references}) != len(references):
        parser.error("Provide at least two references with distinct IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "generate":
        generate(args, references, json.loads(args.cases.read_text()))
    else:
        asr(args, references)


if __name__ == "__main__":
    main()
