"""Exercise the native Drama3 replica pool with a small complete checkpoint."""

import json
import tempfile
import unittest

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import ctranslate2

from ctranslate2.specs.drama3_spec import Drama3Spec


def save_model(directory):
    variables = {}

    def put(name, shape, value=0.01):
        variables[name] = np.full(shape, value, dtype=np.float32)

    def norm(name, width=2):
        put(name + "/weight", (width,), 0)

    def attention(scope):
        for name, shape in {
            "q_proj": (4, 2),
            "k_proj": (2, 2),
            "v_proj": (2, 2),
            "o_proj": (2, 2),
        }.items():
            put(scope + "/" + name + "/weight", shape)
        norm(scope + "/q_norm")
        norm(scope + "/k_norm")

    put("backbone/embeddings/weight", (4098, 2))
    put("backbone/lm_head/weight", (4098, 2))
    put("backbone/lm_head/bias", (4098,), 0)
    variables["backbone/lm_head/bias"][2] = 1
    norm("backbone/norm")
    block = "backbone/layers_0"
    norm(block + "/input_layernorm")
    norm(block + "/post_attention_layernorm")
    attention(block + "/self_attn")
    for name in ("router", "shared_expert_gate"):
        put(block + "/mlp/" + name + "/weight", (1, 2))
    for expert in ("experts_0", "shared_expert"):
        for name, shape in {
            "gate_proj": (1, 2),
            "up_proj": (1, 2),
            "down_proj": (2, 1),
        }.items():
            put(block + "/mlp/" + expert + "/" + name + "/weight", shape)

    put("tower/codebook_embeddings/weight", (40960, 2))
    for index in range(10):
        shared = index == 0
        scope = "tower/" + (
            "shared_layers_0" if shared else f"residual_layers_{index - 1}"
        )
        name = "semantic_embedding" if shared else "codebook_embedding"
        put(scope + "/" + name + "/weight", (4096, 2))
        for name in (
            "enorm",
            "hnorm",
            "final_layernorm",
            "transformer_layer/attention_norm",
            "transformer_layer/ffn_norm",
        ):
            norm(scope + "/" + name)
        put(scope + "/eh_proj/weight", (2, 4))
        attention(scope + "/transformer_layer/attention")
        for name, shape in {
            "gate_proj": (1, 2),
            "up_proj": (1, 2),
            "down_proj": (2, 1),
        }.items():
            put(scope + "/transformer_layer/feed_forward/" + name + "/weight", shape)
        if not shared:
            put(f"tower/heads_{index - 1}/weight", (4096, 2))

    def conv(scope, inputs, outputs, kernel, transpose=False):
        shape = (inputs, outputs, kernel) if transpose else (outputs, inputs, kernel)
        put(scope + "/weight", shape)
        put(scope + "/bias", (outputs,), 0)

    for book in range(10):
        scope = "codec/quantizer/" + (
            "semantic_quantizer/quantizers/0"
            if book == 0
            else f"quantizer/quantizers/{book - 1}"
        )
        put(scope + "/codebook/weight", (4096 if book == 0 else 1024, 2))
        conv(scope + "/out_proj", 2, 32, 1)
    scope = "codec/quantizer/post_module"
    put(scope + "/norm/weight", (32,), 1)
    for index in range(8):
        layer = scope + f"/layers/{index}"
        for name in (
            "attention_norm/weight",
            "ffn_norm/weight",
            "attention_layer_scale/gamma",
            "ffn_layer_scale/gamma",
        ):
            put(layer + "/" + name, (32,), 1)
        for name, shape in {
            "attention/wqkv": (96, 32),
            "attention/wo": (32, 32),
            "feed_forward/w1": (32, 32),
            "feed_forward/w2": (32, 32),
            "feed_forward/w3": (32, 32),
        }.items():
            put(layer + "/" + name + "/weight", shape)
    for index in range(2):
        scope = f"codec/quantizer/upsample/{index}"
        conv(scope + "/0/conv", 32, 32, 2, True)
        conv(scope + "/1/dwconv/conv", 1, 32, 7)
        for name in ("norm/weight", "norm/bias", "gamma"):
            put(scope + "/1/" + name, (32,), 0.1)
        for name, shape in {"pwconv1": (128, 32), "pwconv2": (32, 128)}.items():
            put(scope + "/1/" + name + "/weight", shape)
            put(scope + "/1/" + name + "/bias", (shape[0],), 0)
    conv("codec/decoder/model/0/conv", 32, 1, 7)
    for index, stride in enumerate((8, 8, 4, 2), 1):
        scope = f"codec/decoder/model/{index}/block"
        put(scope + "/0/alpha", (1, 1, 1), 1)
        conv(scope + "/1/conv", 1, 1, 2 * stride, True)
        for unit in range(2, 5):
            for activation in (0, 2):
                put(f"{scope}/{unit}/block/{activation}/alpha", (1, 1, 1), 1)
            conv(f"{scope}/{unit}/block/1/conv", 1, 1, 7)
            conv(f"{scope}/{unit}/block/3/conv", 1, 1, 1)
    put("codec/decoder/model/5/alpha", (1, 1, 1), 1)
    conv("codec/decoder/model/6/conv", 1, 1, 7)

    common = dict(
        head_dim=2, rotary_interleaved=False, layernorm_zero_centered_gamma=True
    )
    text = dict(
        common,
        num_attention_heads=1,
        num_key_value_heads=1,
        rope_parameters=dict(partial_rotary_factor=1, rope_theta=100),
        rms_norm_eps=1e-5,
        attn_output_gate=True,
        max_position_embeddings=128,
        vocab_size=4098,
        layer_types=["full_attention"],
        num_hidden_layers=1,
        num_experts=1,
        num_experts_per_tok=1,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=1,
        linear_value_head_dim=1,
    )
    audio = dict(
        common,
        n_head=1,
        n_local_heads=1,
        rotary_percent=1,
        rope_base=100,
        norm_eps=1e-5,
        attention_output_gate=True,
        num_codebooks=10,
        vocab_size=4096,
        shared_enrichment_layers=1,
        vq_backbone_codebooks=1,
    )
    audio["rotary_interleaved"] = True
    config = dict(
        model_type="drama3",
        ct2_layout_version=1,
        quantization_policy="standard",
        source_config=dict(
            text_config=text,
            audio_decoder_config=audio,
            semantic_start_token_id=2,
            semantic_end_token_id=4097,
            eos_token_id=0,
        ),
    )
    spec = Drama3Spec(variables)
    spec.validate()
    spec.optimize(quantization="float32")
    spec.save(directory)
    Path(directory, "config.json").write_text(json.dumps(config))


@unittest.skipUnless(hasattr(ctranslate2.models, "Drama3"), "native Drama3 unavailable")
class Drama3RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        save_model(cls.directory.name)

    def setUp(self):
        self.model = ctranslate2.models.Drama3(
            self.directory.name, intra_threads=1, inter_threads=2
        )
        self.prompt = np.zeros((2, 11), dtype=np.int32)
        self.prompt[:, 0] = 1

    def test_resolved_execution_properties(self):
        self.assertEqual(self.model.device, "cpu")
        self.assertEqual(self.model.device_index, [0])
        self.assertEqual(self.model.compute_type, "float32")
        for name in ("device", "device_index", "compute_type"):
            with self.subTest(property=name), self.assertRaises(AttributeError):
                setattr(self.model, name, None)

    @unittest.skipUnless(
        "int8_float32" in ctranslate2.get_supported_compute_types("cpu"),
        "CPU INT8 unavailable",
    )
    def test_requested_precision_is_resolved(self):
        for requested in ("int8", "auto"):
            with self.subTest(compute_type=requested):
                model = ctranslate2.models.Drama3(
                    self.directory.name,
                    intra_threads=1,
                    device="cpu",
                    compute_type=requested,
                )
                self.assertEqual(model.compute_type, "int8_float32")
                self.assertEqual(
                    model.generate(self.prompt, max_frames=1, greedy=True).codes.shape,
                    (1, 10),
                )

    @unittest.skipUnless(ctranslate2.get_cuda_device_count(), "GPU unavailable")
    def test_auto_device_reports_execution_device(self):
        model = ctranslate2.models.Drama3(
            self.directory.name, intra_threads=1, device="auto", compute_type="float32"
        )
        self.assertEqual(model.device, "cuda")
        self.assertEqual(model.device_index, [0])
        self.assertEqual(model.compute_type, "float32")
        if "float16" in ctranslate2.get_supported_compute_types("cuda"):
            model = ctranslate2.models.Drama3(
                self.directory.name,
                intra_threads=1,
                device="cuda",
                compute_type="float16",
            )
            # Checkpoint storage remains float32 on CPU for this GPU model.
            self.assertEqual(model.compute_type, "float16")
            self.assertEqual(
                model.generate(self.prompt, max_frames=1, greedy=True).codes.shape,
                (1, 10),
            )

    def test_sync_async_and_batch_preserve_results(self):
        self.assertEqual(self.model.num_replicas, 2)
        expected = self.model.generate(self.prompt, max_frames=2, greedy=True)
        self.assertIsInstance(expected, ctranslate2.models.Drama3Result)
        self.assertEqual(expected.audio.shape, (4096,))
        self.assertEqual(expected.codes.shape, (2, 10))
        np.testing.assert_array_equal(expected["audio"], expected.audio)
        async_result = self.model.generate(
            self.prompt, max_frames=2, greedy=True, asynchronous=True
        )
        results = self.model.generate_batch(
            [self.prompt] * 4, max_frames=2, greedy=True
        )
        results += [async_result.result()]
        for result in results:
            np.testing.assert_array_equal(result.codes, expected.codes)
            np.testing.assert_allclose(result.audio, expected.audio)
        self.assertTrue(async_result.done())
        self.assertEqual(self.model.generate_batch([]), [])

    def test_async_owns_input_and_recovers_from_exception(self):
        pending = self.model.generate(
            self.prompt, max_frames=1, greedy=True, asynchronous=True
        )
        self.prompt.fill(-1)
        self.assertEqual(pending.result().codes.shape, (1, 10))
        bad = self.model.generate(self.prompt, max_frames=1, asynchronous=True)
        with self.assertRaises(ValueError):
            bad.result()
        self.prompt.fill(0)
        self.prompt[:, 0] = 1
        pending = self.model.generate_batch(
            [self.prompt] * 4, max_frames=1, greedy=True, asynchronous=True
        )
        for result in pending:
            self.assertEqual(result.result().finish_reason, "length")

    def test_concurrent_callers_have_independent_state(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _: self.model.generate(
                        self.prompt, max_frames=2, greedy=True
                    ),
                    range(8),
                )
            )
        for result in results[1:]:
            np.testing.assert_array_equal(result.codes, results[0].codes)
            np.testing.assert_allclose(result.audio, results[0].audio)

    def test_seeded_requests_do_not_share_rng_state(self):
        options = dict(max_frames=2, seed=42, top_k=3, top_p=1)
        expected = self.model.generate(self.prompt, **options)
        pending = self.model.generate_batch(
            [self.prompt] * 4, asynchronous=True, **options
        )
        for result in pending:
            np.testing.assert_array_equal(result.result().codes, expected.codes)

    def test_pending_results_survive_pool_destruction(self):
        pending = self.model.generate_batch(
            [self.prompt] * 4, max_frames=1, greedy=True, asynchronous=True
        )
        del self.model
        for result in pending:
            self.assertEqual(result.result().codes.shape, (1, 10))

    def test_static_prefix_cache_preserves_output(self):
        # Isolate the cache from the pool shared across other tests.
        model = ctranslate2.models.Drama3(
            self.directory.name, intra_threads=1, inter_threads=1
        )
        baseline = model.generate(
            self.prompt, max_frames=2, greedy=True, cache_static_prompt=False
        )
        first = model.generate(
            self.prompt,
            max_frames=2,
            greedy=True,
            static_prompt_rows=1,
            cache_static_prompt=True,
        )
        second = model.generate(
            self.prompt,
            max_frames=2,
            greedy=True,
            static_prompt_rows=1,
            cache_static_prompt=True,
        )
        np.testing.assert_array_equal(first.codes, baseline.codes)
        np.testing.assert_array_equal(first.audio, baseline.audio)
        np.testing.assert_array_equal(second.codes, baseline.codes)
        np.testing.assert_array_equal(second.audio, baseline.audio)
        stats = model.prefix_cache_stats
        # First call: miss + save. Second call: hit. Baseline bypasses the cache.
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_static_prefix_cache_uses_current_request_capacity(self):
        model = ctranslate2.models.Drama3(
            self.directory.name, intra_threads=1, inter_threads=1
        )
        model.generate(self.prompt, max_frames=1, greedy=True, static_prompt_rows=1)
        for prompt, max_frames in ((self.prompt, 5), (np.tile(self.prompt, (3, 1)), 2)):
            with self.subTest(rows=len(prompt), max_frames=max_frames):
                baseline = model.generate(prompt, max_frames=max_frames, greedy=True)
                cached = model.generate(
                    prompt, max_frames=max_frames, greedy=True, static_prompt_rows=1
                )
                np.testing.assert_array_equal(cached.codes, baseline.codes)
                np.testing.assert_array_equal(cached.audio, baseline.audio)
        self.assertEqual(model.prefix_cache_stats["hits"], 2)

    def test_static_prefix_cache_distinguishes_keys(self):
        model = ctranslate2.models.Drama3(
            self.directory.name, intra_threads=1, inter_threads=1
        )
        prompt_a = self.prompt.copy()
        prompt_b = self.prompt.copy()
        prompt_b[0, 0] = 2  # different static-prefix bytes
        baseline_a = model.generate(
            prompt_a, max_frames=2, greedy=True, cache_static_prompt=False
        )
        baseline_b = model.generate(
            prompt_b, max_frames=2, greedy=True, cache_static_prompt=False
        )
        cached_a = model.generate(
            prompt_a,
            max_frames=2,
            greedy=True,
            static_prompt_rows=1,
            cache_static_prompt=True,
        )
        cached_b = model.generate(
            prompt_b,
            max_frames=2,
            greedy=True,
            static_prompt_rows=1,
            cache_static_prompt=True,
        )
        np.testing.assert_array_equal(cached_a.codes, baseline_a.codes)
        np.testing.assert_array_equal(cached_b.codes, baseline_b.codes)
        stats = model.prefix_cache_stats
        self.assertEqual(stats["entries"], 2)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 2)

    def test_static_prefix_cache_restores_populated_tower_caches(self):
        model = ctranslate2.models.Drama3(
            self.directory.name, intra_threads=1, inter_threads=1
        )
        # Three prefix rows populate both backbone and tower caches. Warm with
        # a larger request, then reuse the prefix with smaller/larger capacities.
        for rows, max_frames, chunk_size in ((8, 5, 2), (4, 1, 1), (10, 6, 4)):
            with self.subTest(rows=rows, max_frames=max_frames):
                prompt = np.tile(self.prompt[:1], (rows, 1))
                baseline = model.generate(
                    prompt, max_frames=max_frames, greedy=True,
                    prefill_chunk_size=chunk_size,
                )
                cached = model.generate(
                    prompt, max_frames=max_frames, greedy=True,
                    prefill_chunk_size=chunk_size, static_prompt_rows=3,
                )
                np.testing.assert_array_equal(cached.codes, baseline.codes)
                np.testing.assert_array_equal(cached.audio, baseline.audio)
        self.assertEqual(model.prefix_cache_stats["hits"], 2)

    def test_static_prompt_rows_bounds_are_validated(self):
        model = ctranslate2.models.Drama3(
            self.directory.name, intra_threads=1, inter_threads=1
        )
        with self.assertRaises(ValueError):
            model.generate(
                self.prompt, max_frames=1, greedy=True, static_prompt_rows=-1
            )
        with self.assertRaises(ValueError):
            model.generate(
                self.prompt,
                max_frames=1,
                greedy=True,
                static_prompt_rows=self.prompt.shape[0],
            )
