"""Drama3 checkpoint layout for standard CTranslate2 conversion."""

from ctranslate2.specs import model_spec


class _ParameterSpec(model_spec.LayerSpec):
    def __init__(self, parameters):
        for name, value in parameters.items():
            setattr(
                self, name, _ParameterSpec(value) if isinstance(value, dict) else value
            )


class Drama3Spec(model_spec.ModelSpec):
    def __init__(self, variables, config=None, model_name="Drama3Spec"):
        super().__init__()
        self._model_name = model_name
        self._config = model_spec.ModelConfig(**config) if config is not None else None
        self._matrix_names = {
            name
            for name, value in variables.items()
            if not name.startswith("codec/")
            and name.endswith("/weight")
            and len(value.shape) == 2
        }
        parameters = {}
        for path, value in variables.items():
            parts = path.split("/")
            if any(not part or part.startswith("_") for part in parts):
                raise ValueError("Invalid Drama3 parameter path: %s" % path)
            parent = parameters
            for part in parts[:-1]:
                parent = parent.setdefault(part, {})
            parent[parts[-1]] = value
        for name, value in parameters.items():
            setattr(
                self, name, _ParameterSpec(value) if isinstance(value, dict) else value
            )

    @property
    def name(self):
        return self._model_name

    def _quantize(self, quantization):
        # The CPU synthesis graph and codec retain FLOAT32 state and parameters.
        super()._quantize(quantization, lambda name: name in self._matrix_names)
        super()._quantize("float32", lambda name: name not in self._matrix_names)


def codec_parameter_shapes():
    """Required logical weights for the Fish DAC graph supported by CT2."""
    shapes = {}

    def conv(name, inputs, outputs, kernel, transpose=False):
        shapes[name + ".weight"] = (
            (inputs, outputs, kernel) if transpose else (outputs, inputs, kernel)
        )
        shapes[name + ".bias"] = (outputs,)

    conv("decoder.model.0.conv", 1024, 1536, 7)
    channels = 1536
    for stage, stride in enumerate((8, 8, 4, 2), 1):
        prefix = f"decoder.model.{stage}.block"
        shapes[prefix + ".0.alpha"] = (1, channels, 1)
        conv(prefix + ".1.conv", channels, channels // 2, 2 * stride, transpose=True)
        channels //= 2
        for unit in range(2, 5):
            block = f"{prefix}.{unit}.block"
            for index in (0, 2):
                shapes[f"{block}.{index}.alpha"] = (1, channels, 1)
            conv(block + ".1.conv", channels, channels, 7)
            conv(block + ".3.conv", channels, channels, 1)
    shapes["decoder.model.5.alpha"] = (1, channels, 1)
    conv("decoder.model.6.conv", channels, 1, 7)

    for layer in range(8):
        prefix = f"quantizer.post_module.layers.{layer}."
        for name in (
            "attention_norm.weight",
            "ffn_norm.weight",
            "attention_layer_scale.gamma",
            "ffn_layer_scale.gamma",
        ):
            shapes[prefix + name] = (1024,)
        for name, shape in {
            "attention.wo.weight": (1024, 1024),
            "attention.wqkv.weight": (3072, 1024),
            "feed_forward.w1.weight": (3072, 1024),
            "feed_forward.w2.weight": (1024, 3072),
            "feed_forward.w3.weight": (3072, 1024),
        }.items():
            shapes[prefix + name] = shape
    shapes["quantizer.post_module.norm.weight"] = (1024,)
    for book in range(10):
        prefix = (
            "quantizer.semantic_quantizer.quantizers.0"
            if book == 0
            else f"quantizer.quantizer.quantizers.{book - 1}"
        )
        shapes[prefix + ".codebook.weight"] = (4096 if book == 0 else 1024, 8)
        conv(prefix + ".out_proj", 8, 1024, 1)
    for stage in range(2):
        prefix = f"quantizer.upsample.{stage}"
        conv(prefix + ".0.conv", 1024, 1024, 2, transpose=True)
        conv(prefix + ".1.dwconv.conv", 1, 1024, 7)
        for name in ("gamma", "norm.weight", "norm.bias", "pwconv2.bias"):
            shapes[prefix + ".1." + name] = (1024,)
        shapes[prefix + ".1.pwconv1.weight"] = (4096, 1024)
        shapes[prefix + ".1.pwconv1.bias"] = (4096,)
        shapes[prefix + ".1.pwconv2.weight"] = (1024, 4096)
    return dict(sorted(shapes.items()))
