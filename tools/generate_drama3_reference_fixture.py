"""Generate an optional native parity fixture using Fish Speech's actual encoder.

Requires PyTorch, Fish Speech on PYTHONPATH, and a local official codec.pth.
No backbone weights or downloads are needed.
"""

import argparse
import json

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fish_speech.models.text2semantic.inference import load_codec_model

from ctranslate2.converters.drama3_codec import load_checkpoint, load_codec_tensors
from ctranslate2.specs.drama3_spec import Drama3Spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--audio", type=Path, help="Optional 44.1 kHz speech WAV for parity"
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=False)
    variables = load_codec_tensors(load_checkpoint(args.checkpoint))
    spec = Drama3Spec(variables, model_name="Drama3ReferenceFixture")
    spec.validate()
    (args.output / "model").mkdir()
    spec.save(str(args.output / "model"))
    del spec, variables
    model = load_codec_model(str(args.checkpoint), "cpu", precision=torch.float32)
    # Float32 backends can select different entries at a quantizer boundary.
    # Record an alternate only when the oracle's top two squared distances
    # differ by <= 1e-5; this does not allow arbitrary token mismatches.
    alternatives = []

    def quantizer_hook(module, inputs, output):
        latent = F.normalize(output[-1].transpose(1, 2), dim=-1)
        embedding = F.normalize(module.codebook.weight, dim=-1)
        distances = (
            latent.square().sum(-1, keepdim=True)
            - 2 * latent @ embedding.T
            + embedding.square().sum(-1)
        )
        values, indices = distances.topk(2, largest=False)
        primary = output[3]
        other = torch.where(
            indices[..., 0] == primary, indices[..., 1], indices[..., 0]
        )
        gap = values[..., 1] - values[..., 0]
        alternatives.append(torch.where(gap <= 1e-5, other, -1)[0].numpy())

    books = [
        *model.quantizer.semantic_quantizer.quantizers,
        *model.quantizer.quantizer.quantizers,
    ]
    for book in books:
        book.register_forward_hook(quantizer_hook)
    cases = []
    signals = [np.zeros(1, dtype=np.float32)]
    for length in (2048, 2049, 8193, 270337):
        time = np.arange(length, dtype=np.float32) / 44100
        signals.append((0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32))
    if args.audio:
        import soundfile as sf

        samples, rate = sf.read(args.audio, dtype="float32")
        if rate != 44100 or samples.ndim != 1:
            raise ValueError("--audio must be mono 44100 Hz")
        signals.append(samples[:44100])
    with torch.inference_mode():
        for i, waveform in enumerate(signals):
            alternatives.clear()
            codes, _ = model.encode(torch.from_numpy(waveform)[None, None])
            expected = codes[0].T.contiguous().numpy().astype(np.int32)
            waveform.tofile(args.output / f"{i}.pcm")
            expected.tofile(args.output / f"{i}.codes")
            np.stack(alternatives, axis=1).astype(np.int32).tofile(
                args.output / f"{i}.alternate"
            )
            cases.append(
                {
                    "samples": len(waveform),
                    "frames": len(expected),
                    "waveform": f"{i}.pcm",
                    "codes": f"{i}.codes",
                    "alternate": f"{i}.alternate",
                }
            )
    (args.output / "manifest.json").write_text(json.dumps({"cases": cases}, indent=2))


if __name__ == "__main__":
    main()
