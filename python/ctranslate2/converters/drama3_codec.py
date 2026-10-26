"""Load Drama3 codec weights for standard CTranslate2 conversion.

Fold weight normalization in row chunks and retain FLOAT32 arrays. PyTorch is
needed only to safely load the tensor-only checkpoint with memory mapping.
"""

import numpy as np

from ctranslate2.specs.drama3_spec import codec_parameter_shapes

FLOAT_TYPES = {
    "float32",
    "float16",
    "float64",
    "torch.float32",
    "torch.float16",
    "torch.float64",
    "torch.bfloat16",
}


def conversion_plan(schema):
    """Resolve direct/legacy/parametrized tensors using only names and shapes."""
    keys = schema.get("keys", schema)
    plan = []
    for source, shape in codec_parameter_shapes().items():
        candidates = []
        if source in keys:
            candidates.append((source, None))
        if source.endswith(".weight"):
            stem = source[: -len(".weight")]
            for g, v in (
                (
                    stem + ".parametrizations.weight.original0",
                    stem + ".parametrizations.weight.original1",
                ),
                (stem + ".weight_g", stem + ".weight_v"),
            ):
                if g in keys or v in keys:
                    if g not in keys or v not in keys:
                        raise ValueError(
                            "incomplete weight-normalization pair: " + source
                        )
                    candidates.append((v, g))
        if len(candidates) != 1:
            raise ValueError("missing or ambiguous official tensor: " + source)
        value_key, magnitude_key = candidates[0]
        if tuple(keys[value_key]["shape"]) != shape:
            raise ValueError("unsupported codec weight shape: " + source)
        selected = [value_key]
        if magnitude_key:
            if list(keys[magnitude_key]["shape"]) != [shape[0]] + [1] * (
                len(shape) - 1
            ):
                raise ValueError(
                    "weight-normalization magnitude must retain axis0: " + source
                )
            selected.append(magnitude_key)
        if any(keys[key]["dtype"] not in FLOAT_TYPES for key in selected):
            raise ValueError("non-floating codec tensor: " + source)
        target = "codec/" + source.replace(".", "/")
        plan.append(
            {
                "source_name": source,
                "target": target,
                "shape": list(keys[value_key]["shape"]),
                "value_key": value_key,
                "magnitude_key": magnitude_key,
            }
        )
    return plan


def numpy_float_chunk(value):
    if isinstance(value, np.ndarray):
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(value, dtype=np.float32, order="C")
    else:
        # Only the supplied slice is converted, never a whole checkpoint bank.
        result = (
            value.detach()
            .to(device="cpu", dtype=__import__("torch").float32)
            .contiguous()
            .numpy()
        )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite codec weight or float32 conversion overflow")
    return result


def fold_axis0_chunk(direction, magnitude):
    """Independent axis-0 WN: each first-axis row has one magnitude."""
    direction = numpy_float_chunk(direction)
    magnitude = numpy_float_chunk(magnitude)
    if direction.ndim < 2 or magnitude.shape != (direction.shape[0],) + (1,) * (
        direction.ndim - 1
    ):
        raise ValueError("incompatible axis0 weight-normalization shapes")
    axes = tuple(range(1, direction.ndim))
    # Float64 norm avoids squaring overflow; product is checked again in F32.
    norm = np.sqrt(np.sum(direction.astype(np.float64) ** 2, axis=axes, keepdims=True))
    if not np.isfinite(norm).all() or (norm == 0).any():
        raise ValueError("zero or nonfinite weight-normalization norm")
    folded = np.asarray(
        direction * (magnitude.astype(np.float64) / norm), dtype=np.float32, order="C"
    )
    if not np.isfinite(folded).all():
        raise ValueError("weight-normalization output overflows float32")
    return folded


def tensor_chunks(state, item, chunk_rows):
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    value = state[item["value_key"]]
    magnitude = state[item["magnitude_key"]] if item["magnitude_key"] else None
    for start in range(0, item["shape"][0], chunk_rows):
        end = min(start + chunk_rows, item["shape"][0])
        if magnitude is None:
            yield numpy_float_chunk(value[start:end])
        else:
            yield fold_axis0_chunk(value[start:end], magnitude[start:end])


def state_schema(state):
    return {
        "keys": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state.items()
        }
    }


def load_codec_tensors(state, plan=None, chunk_rows=32):
    """Load decoder weights as whole FLOAT32 arrays for standard ModelSpec."""
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    plan = conversion_plan(state_schema(state)) if plan is None else plan
    tensors = {}
    for item in plan:
        value = np.empty(item["shape"], dtype=np.float32)
        start = 0
        for chunk in tensor_chunks(state, item, chunk_rows):
            value[start : start + len(chunk)] = chunk
            start += len(chunk)
        tensors[item["target"]] = value
    return tensors


def load_checkpoint(path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("expected a flat tensor-only codec state dictionary")
    return state
