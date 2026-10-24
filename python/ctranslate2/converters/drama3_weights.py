"""Bounded-memory safetensors matrix reads for an independent Drama 3 converter.

Rank-two matrices are supported.
No tensor-name conventions or automatic quantization-scale matching are assumed.
Files must remain immutable throughout reading. Safetensors validates and maps
shards; PyTorch converts individual FP8/BF16 chunks to float32.
"""

from pathlib import Path

import numpy as np

from safetensors import SafetensorError, safe_open

STORAGE_DTYPES = {
    "F32": "<f4",
    "F16": "<f2",
    "BF16": "<u2",
    "F8_E4M3FN": "u1",
    "F8_E4M3": "u1",
}


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate safetensors header key: {name}")
        result[name] = value
    return result


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _float32(raw, dtype):
    import torch

    tensor = torch.from_numpy(raw)
    if dtype == "BF16":
        tensor = tensor.view(torch.bfloat16)
    elif dtype.startswith("F8_"):
        tensor = tensor.view(torch.float8_e4m3fn)
    return tensor.to(dtype=torch.float32, copy=True).numpy()


class SafeTensorMatrixReader:
    """Inspect one immutable safetensors shard, mapping only requested tensors.

    Conversion buffers scale with chunk_rows * matrix columns plus relevant
    block-scale rows. The library maps the file without allocating a full dense
    tensor buffer. Consumers must process/discard chunks rather than collect them
    if bounded working memory is required.
    """

    def __init__(self, path):
        self.path = Path(path)
        stat = self.path.stat()
        self._identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        try:
            self._shard = safe_open(self.path, framework="pt", device="cpu")
            self.tensors = {}
            for name in self._shard.keys():
                tensor = self._shard.get_slice(name)
                dtype = tensor.get_dtype()
                if dtype not in STORAGE_DTYPES:
                    raise ValueError(
                        f"unsupported safetensors dtype for {name}: {dtype}"
                    )
                self.tensors[name] = {
                    "dtype": dtype,
                    "shape": tuple(tensor.get_shape()),
                }
        except SafetensorError as error:
            raise ValueError(f"invalid safetensors shard: {error}") from error

    def _map(self, name, index=None):
        import torch

        stat = self.path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != self._identity:
            raise ValueError("checkpoint shard changed after header inspection")
        tensor = (
            self._shard.get_tensor(name)
            if index is None
            else self._shard.get_slice(name)[index]
        )
        # Reinterpret unsupported NumPy dtypes without copying or decoding a bank.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.uint16)
        elif tensor.dtype == torch.float8_e4m3fn:
            tensor = tensor.view(torch.uint8)
        return tensor.numpy()

    def iter_matrix_rows(
        self,
        name,
        chunk_rows=128,
        inverse_scale=None,
        block_size=(128, 128),
        scale_reader=None,
    ):
        """Yield bounded FP32 row chunks of a rank-two matrix.

        scale_reader, when supplied, explicitly locates inverse_scale in another
        safetensors shard; scale names and layouts are never inferred.
        """
        if not _is_int(chunk_rows) or chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if (
            not isinstance(block_size, (tuple, list))
            or len(block_size) != 2
            or any(not _is_int(d) or d <= 0 for d in block_size)
        ):
            raise ValueError("block_size must contain two positive integers")
        tensor = self.tensors[name]
        if len(tensor["shape"]) != 2:
            raise ValueError("row conversion requires a rank-two matrix")
        rows, cols = tensor["shape"]
        if scale_reader is not None and not isinstance(
            scale_reader, SafeTensorMatrixReader
        ):
            raise ValueError("scale_reader must be a SafeTensorMatrixReader")
        if scale_reader is not None and inverse_scale is None:
            raise ValueError(
                "scale_reader requires an explicit inverse_scale tensor name"
            )
        scale_source = self if scale_reader is None else scale_reader
        br, bc = block_size
        scale_meta = None
        scales = None
        if inverse_scale is not None:
            if not tensor["dtype"].startswith("F8_"):
                raise ValueError("inverse_scale is supported only for FP8 matrices")
            scale_meta = scale_source.tensors[inverse_scale]
            expected_shape = ((rows + br - 1) // br, (cols + bc - 1) // bc)
            if scale_meta["shape"] != expected_shape:
                raise ValueError("inverse_scale shape does not match block grid")
            if scale_meta["dtype"] not in ("F32", "F16", "BF16"):
                raise ValueError("inverse_scale must be F32/F16/BF16")
            scales = scale_source._map(inverse_scale)
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            chunk = _float32(self._map(name, slice(start, stop)), tensor["dtype"])
            if scales is not None:
                first_block = start // br
                scale_rows = _float32(
                    scales[first_block : (stop - 1) // br + 1], scale_meta["dtype"]
                )
                if not np.all(np.isfinite(scale_rows)) or np.any(scale_rows <= 0):
                    raise ValueError("inverse_scale values must be positive and finite")
                # Avoid expanding an entire rows*columns scale matrix.
                for row in range(stop - start):
                    sr = scale_rows[(start + row) // br - first_block]
                    for col_block in range(len(sr)):
                        chunk[
                            row, col_block * bc : min((col_block + 1) * bc, cols)
                        ] *= sr[col_block]
            yield chunk
