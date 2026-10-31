"""Compare cloned voices with independent WavLM speaker embeddings.

Run after evaluate_drama3_voice_cloning.py asr. Requires PyTorch, Transformers,
SoundFile, SciPy, and a local microsoft/wavlm-base-plus-sv checkpoint.
"""

import argparse
import json
import math

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from evaluate_drama3_voice_cloning import save_json
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, WavLMForXVector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    model = WavLMForXVector.from_pretrained(args.model, local_files_only=True).eval()

    def embedding(path):
        audio, rate = sf.read(path, dtype="float32")
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Expected finite, nonempty mono audio")
        common = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // common, rate // common)
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            return torch.nn.functional.normalize(model(**inputs).embeddings, dim=-1)[0]

    references = json.loads(args.references.read_text())
    reference_embeddings = {r["id"]: embedding(r["audio"]) for r in references}
    records = json.loads((args.output / "generation.json").read_text())
    scores = []
    for record in records:
        if record.get("status") != "ok" or record["voice"] == "unconditioned":
            continue
        vector = embedding(args.output / (record["id"] + ".wav"))
        similarities = {
            voice: float(vector @ value)
            for voice, value in reference_embeddings.items()
        }
        target = similarities[record["voice"]]
        other = max(
            value for voice, value in similarities.items() if voice != record["voice"]
        )
        scores.append(
            {"id": record["id"], "similarities": similarities, "margin": target - other}
        )
        save_json(args.output / "speakers.json", scores)
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text())
    preference = sum(r["margin"] > 0 for r in scores) / max(1, len(scores))
    summary.update(
        speaker_target_preference=preference,
        speaker_pass=bool(scores)
        and len(scores) == summary["cloned_cases"]
        and preference >= 0.75,
        speaker_reference_cosines={
            a: {b: float(x @ y) for b, y in reference_embeddings.items()}
            for a, x in reference_embeddings.items()
        },
    )
    summary["overall_automated_pass"] = (
        summary["content_pass"] and summary["speaker_pass"]
    )
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
