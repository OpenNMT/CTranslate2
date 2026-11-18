"""Independent CPU reference for lossy group-32 MPS KV cache formats."""

import torch

from transformers import DynamicCache


def quantize_kv(value, bits):
    """Quantize each token/head independently; return FP16-representable values."""
    if bits not in (4, 5, 8):
        raise ValueError("KV reference supports 4, 5, or 8 bits")
    depth = value.shape[-1]
    # The native cache receives FP16 keys/values after normalization and RoPE.
    rounded = value.to(torch.float16).float()
    padded = torch.nn.functional.pad(rounded, (0, (-depth) % 32))
    groups = padded.reshape(*value.shape[:-1], -1, 32)
    limit = (1 << (bits - 1)) - 1
    maximum = groups.abs().amax(-1, keepdim=True)
    scale = maximum * (1.0 / limit)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    divisor = torch.where(maximum == 0, torch.ones_like(maximum), maximum)
    codes = torch.round((groups * limit) / divisor).clamp(-limit, limit)
    result = (codes * scale).reshape(*value.shape[:-1], -1)[..., :depth]
    return result.to(torch.float16).to(value.dtype)


class QuantizedReferenceCache(DynamicCache):
    """Fake-quantized source cache: no CT2 packing or kernels are called."""

    def __init__(self, config, bits):
        super().__init__(config=config)
        self.bits = bits

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return super().update(
            quantize_kv(key_states, self.bits),
            quantize_kv(value_states, self.bits),
            layer_idx,
            *args,
            **kwargs,
        )
