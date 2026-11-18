#!/usr/bin/env python3
"""Stream an existing CT2 FP16 decoder checkpoint into native group-64 INT4.

Only decoder layer linear matrices are quantized. Embeddings, vocabulary head,
routers, and scalar shared gates stay FP16. The source directory is never edited.
"""

import argparse
import json
import shutil
import struct

from pathlib import Path

import numpy as np

from ctranslate2.specs.common_spec import LinearSpec
from ctranslate2.specs.model_spec import NumpyVariable


def read(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("Truncated CT2 model")
    return data


def integer(stream, fmt):
    return struct.unpack("<" + fmt, read(stream, struct.calcsize(fmt)))[0]


def string(stream):
    return read(stream, integer(stream, "H"))[:-1].decode()


def write_string(stream, value):
    data = value.encode() + b"\0"
    stream.write(struct.pack("<H", len(data)))
    stream.write(data)


def pack(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
        or destination.exists()
    ):
        raise ValueError("Use a new destination directory outside the source")
    destination.mkdir(parents=True)
    matrices, replaced, aliases = 0, set(), []
    try:
        with (source / "model.bin").open("rb") as src, (destination / "model.bin").open(
            "wb"
        ) as dst:
            binary = integer(src, "I")
            if binary != 6:
                raise ValueError("Only CT2 binary version 6 is supported")
            dst.write(struct.pack("<I", binary))
            name = string(src)
            if name != "TransformerDecoderSpec":
                raise ValueError("Only Transformer decoder checkpoints are supported")
            write_string(dst, name)
            dst.write(struct.pack("<I", integer(src, "I")))
            count = integer(src, "I")
            count_position = dst.tell()
            dst.write(struct.pack("<I", 0))
            output_count = 0
            for _ in range(count):
                key = string(src)
                rank = integer(src, "B")
                shape = tuple(integer(src, "I") for _ in range(rank))
                dtype, size = integer(src, "B"), integer(src, "I")
                eligible = (
                    key.startswith("decoder/layer_")
                    and "/linear_" in key
                    and key.endswith("/weight")
                )
                if eligible:
                    if rank != 2 or dtype != 4 or size != int(np.prod(shape)) * 2:
                        raise ValueError(
                            "Expected unquantized FP16 linear matrix: " + key
                        )
                    linear = LinearSpec()
                    linear.weight = NumpyVariable(
                        np.frombuffer(read(src, size), np.float16).reshape(shape)
                    )
                    linear.optimize("int4_float16")
                    base = key[:-7]
                    for suffix, value, type_id in (
                        ("weight_int4", linear.weight_int4, 1),
                        ("weight_int4_scale", linear.weight_int4_scale, 0),
                        ("int4_depth", linear.int4_depth, 3),
                    ):
                        write_string(dst, base + "/" + suffix)
                        dst.write(struct.pack("<B", len(value.shape)))
                        for dim in value.shape:
                            dst.write(struct.pack("<I", dim))
                        dst.write(struct.pack("<BI", type_id, value.num_bytes()))
                        dst.write(value.to_bytes())
                    matrices += 1
                    replaced.add(key)
                    output_count += 3
                else:
                    write_string(dst, key)
                    dst.write(struct.pack("<B", rank))
                    for dim in shape:
                        dst.write(struct.pack("<I", dim))
                    dst.write(struct.pack("<BI", dtype, size))
                    remaining = size
                    while remaining:
                        block = read(src, min(1024 * 1024, remaining))
                        dst.write(block)
                        remaining -= len(block)
                    output_count += 1
            for _ in range(integer(src, "I")):
                alias, target = string(src), string(src)
                if alias in replaced or target in replaced:
                    raise ValueError(
                        "Aliased linear matrices need conversion from the source model"
                    )
                aliases.append((alias, target))
            if src.read(1) or not matrices:
                raise ValueError("Unexpected model tail or no eligible linear matrices")
            dst.write(struct.pack("<I", len(aliases)))
            for alias, target in aliases:
                write_string(dst, alias)
                write_string(dst, target)
            dst.seek(count_position)
            dst.write(struct.pack("<I", output_count))
        for path in source.iterdir():
            if path.is_file() and path.name != "model.bin":
                shutil.copy2(path, destination / path.name)
        report = {
            "source": str(source),
            "matrices": matrices,
            "group_size": 64,
            "source_bytes": (source / "model.bin").stat().st_size,
            "packed_bytes": (destination / "model.bin").stat().st_size,
        }
        (destination / "int4_conversion.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report
    except BaseException:
        shutil.rmtree(destination)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(pack(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
