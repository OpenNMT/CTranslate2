"""Coverage of Drama3 checkpoint reads without the CT2 extension."""
import json
import struct
import tempfile
import unittest

from pathlib import Path

import numpy as np

from drama_test_utils import shard

from ctranslate2.converters import drama3_weights as weights


def write_tensors(path, tensors):
    shard(
        path,
        {
            name: (np.asarray(data, dtype=weights.STORAGE_DTYPES[dtype]), dtype)
            for name, (dtype, data) in tensors.items()
        },
    )


class Drama3WeightsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "matrix.safetensors"

    def test_fp8_bit_patterns(self):
        raw = np.array(
            [0, 128, 1, 129, 7, 8, 56, 184, 126, 254, 127, 255], dtype=np.uint8
        )
        write_tensors(self.path, {"matrix": ("F8_E4M3FN", raw.reshape(1, -1))})
        result = next(
            weights.SafeTensorMatrixReader(self.path).iter_matrix_rows("matrix")
        )[0]
        np.testing.assert_array_equal(
            result[:10], [0, -0.0, 1 / 512, -1 / 512, 7 / 512, 1 / 64, 1, -1, 448, -448]
        )
        self.assertFalse(np.signbit(result[0]))
        self.assertTrue(np.signbit(result[1]))
        self.assertTrue(np.all(np.isnan(result[10:])))
        self.assertEqual(result.dtype, np.float32)

    def test_float32_float16_and_independent_chunks(self):
        matrix = np.arange(15, dtype=np.float32).reshape(5, 3) / 4
        for dtype in ("F32", "F16"):
            with self.subTest(dtype=dtype):
                # A live reader maps its shard; Windows cannot overwrite it.
                path = self.path.with_name(f"matrix-{dtype}.safetensors")
                write_tensors(path, {"matrix": (dtype, matrix)})
                reader = weights.SafeTensorMatrixReader(path)
                chunks = list(reader.iter_matrix_rows("matrix", chunk_rows=2))
                self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
                self.assertTrue(all(chunk.dtype == np.float32 for chunk in chunks))
                np.testing.assert_array_equal(np.concatenate(chunks), matrix)
                chunks[0][0, 0] = 99
                np.testing.assert_array_equal(
                    np.concatenate(list(reader.iter_matrix_rows("matrix"))), matrix
                )

    def test_bfloat16(self):
        raw = np.array(
            [[0x0000, 0x8000, 0x3F80, 0xC020, 0x0001, 0x7F80, 0x7FC0]], dtype=np.uint16
        )
        write_tensors(self.path, {"matrix": ("BF16", raw)})
        result = next(
            weights.SafeTensorMatrixReader(self.path).iter_matrix_rows("matrix")
        )[0]
        np.testing.assert_array_equal(result[:5], [0, -0.0, 1, -2.5, 2.0**-133])
        self.assertTrue(np.signbit(result[1]))
        self.assertTrue(np.isposinf(result[5]))
        self.assertTrue(np.isnan(result[6]))

    def test_block_scales_tail_blocks_and_crossing_row_chunks(self):
        raw = np.full((5, 7), 56, dtype=np.uint8)  # all ones
        scales = np.array([[2, 3, 5], [7, 11, 13], [17, 19, 23]], dtype=np.float32)
        write_tensors(
            self.path, {"matrix": ("F8_E4M3FN", raw), "explicit": ("F32", scales)}
        )
        reader = weights.SafeTensorMatrixReader(self.path)
        result = np.concatenate(
            list(
                reader.iter_matrix_rows(
                    "matrix", chunk_rows=3, inverse_scale="explicit", block_size=(2, 3)
                )
            )
        )
        expected = np.array(
            [[2, 2, 2, 3, 3, 3, 5]] * 2
            + [[7, 7, 7, 11, 11, 11, 13]] * 2
            + [[17, 17, 17, 19, 19, 19, 23]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_library_written_fp8_and_bfloat16_preserve_storage_and_values(self):
        import torch

        from safetensors.torch import save_file

        tensors = (
            torch.arange(256, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .reshape(16, 16),
            torch.tensor(
                [0, 0x8000, 0x3F80, 0xC020, 1, 0x7F80, 0x7FC0], dtype=torch.uint16
            )
            .view(torch.bfloat16)
            .reshape(1, 7),
        )
        for tensor in tensors:
            with self.subTest(dtype=tensor.dtype):
                path = self.path.with_name(f"matrix-{tensor.dtype}.safetensors")
                save_file({"matrix": tensor}, str(path))
                reader = weights.SafeTensorMatrixReader(path)
                storage_dtype = (
                    torch.uint8 if tensor.element_size() == 1 else torch.uint16
                )
                np.testing.assert_array_equal(
                    reader._map("matrix"), tensor.view(storage_dtype).numpy()
                )
                chunks = list(reader.iter_matrix_rows("matrix", chunk_rows=3))
                actual = np.concatenate(chunks)
                expected = tensor.float().numpy()
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(
                    np.signbit(actual[actual == 0]), np.signbit(expected[expected == 0])
                )

    def test_invalid_scales_and_arguments(self):
        write_tensors(
            self.path,
            {
                "matrix": ("F8_E4M3FN", np.ones((2, 3), np.uint8)),
                "badshape": ("F32", [[1, 2]]),
                "nan": ("F32", [[np.nan]]),
                "negative": ("F32", [[-1]]),
                "zero": ("F32", [[0]]),
                "fp8scale": ("F8_E4M3FN", [[56]]),
                "rank3": ("F32", np.zeros((2, 2, 2))),
            },
        )
        reader = weights.SafeTensorMatrixReader(self.path)
        for name in ("badshape", "nan", "negative", "zero", "fp8scale"):
            with self.subTest(scale=name), self.assertRaises(ValueError):
                list(reader.iter_matrix_rows("matrix", inverse_scale=name))
        for kwargs in (
            {"chunk_rows": 0},
            {"chunk_rows": True},
            {"block_size": (0, 2)},
            {"block_size": (2,)},
            {"block_size": (True, 2)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                list(reader.iter_matrix_rows("matrix", **kwargs))
        with self.assertRaises(ValueError):
            list(reader.iter_matrix_rows("rank3"))

    def test_rejects_bad_ranges_and_headers(self):
        for entry in (
            {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 15]},
            {"dtype": "F32", "shape": [2, 2], "data_offsets": [-1, 15]},
            {"dtype": "F32", "shape": [2, -2], "data_offsets": [0, 16]},
            {"dtype": "F32", "shape": [2, True], "data_offsets": [0, 16]},
        ):
            encoded = json.dumps({"matrix": entry}).encode()
            self.path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(16))
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                weights.SafeTensorMatrixReader(self.path)
        for data in (
            b"short",
            struct.pack("<Q", 100_000_001),
            struct.pack("<Q", 13) + b'{"x":1,"x":2}',
        ):
            self.path.write_bytes(data)
            with self.assertRaises(ValueError):
                weights.SafeTensorMatrixReader(self.path)

    def test_detects_file_change_before_mapping(self):
        write_tensors(self.path, {"matrix": ("F32", [[1, 2]])})
        reader = weights.SafeTensorMatrixReader(self.path)
        with self.path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            list(reader.iter_matrix_rows("matrix"))

    def test_empty_matrix(self):
        write_tensors(self.path, {"matrix": ("F32", np.empty((0, 3)))})
        self.assertEqual(
            list(weights.SafeTensorMatrixReader(self.path).iter_matrix_rows("matrix")),
            [],
        )

    def test_cross_shard_matrix_scales(self):
        write_tensors(self.path, {"matrix": ("F8_E4M3FN", np.full((2, 3), 56))})
        scale_path = self.path.with_name("scales.safetensors")
        write_tensors(scale_path, {"matrix_scale": ("BF16", [[0x4000]])})
        reader = weights.SafeTensorMatrixReader(self.path)
        scale_reader = weights.SafeTensorMatrixReader(scale_path)
        result = next(
            reader.iter_matrix_rows(
                "matrix", inverse_scale="matrix_scale", scale_reader=scale_reader
            )
        )
        np.testing.assert_array_equal(result, np.full((2, 3), 2))
        with self.assertRaises(ValueError):
            list(reader.iter_matrix_rows("matrix", scale_reader=scale_reader))
        with self.assertRaises(ValueError):
            list(reader.iter_matrix_rows("matrix", scale_reader="path"))


if __name__ == "__main__":
    unittest.main()
