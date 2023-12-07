import json
import os

from typing import Optional

import ctranslate2

from ctranslate2.converters.opennmt_py import OpenNMTPyConverter


def _get_converter(model_path: str, model_type: str):
    if model_type == "OpenNMTPy":
        if not os.path.exists(model_path):
            raise RuntimeError("No model opennmt-py found in %s" % model_path)

        converter = OpenNMTPyConverter(model_path=model_path)
        return converter
    else:
        raise NotImplementedError(
            "Converter on the fly for %s is not implemented." % model_type
        )


class GeneratorOnTheFly:
    """Initializes the generator on the fly.

    Arguments:
      model_path: Path to the CTranslate2 model directory.
      device: Device to use (possible values are: cpu, cuda, auto).
      device_index: Device IDs where to place this generator on.
      compute_type: Model computation type or a dictionary mapping
          a device name to the computation type (possible values are:
          default, auto, int8, int8_float32, int8_float16, int8_bfloat16,
          int16, float16, bfloat16, float32).
      inter_threads: Maximum number of parallel generations.
      intra_threads: Number of OpenMP threads per generator
          (0 to use a default value).
      max_queued_batches: Maximum numbers of batches in the queue
          (-1 for unlimited, 0 for an automatic value).
          When the queue is full, future requests will block
          until a free slot is available.
      model_type: type of converter to convert the model
      quantization: quantize the model
    """

    def __init__(
        self,
        model_path: str,
        device="cpu",
        device_index=0,
        compute_type="default",
        inter_threads=1,
        intra_threads=0,
        max_queued_batches=0,
        model_type="OpenNMTPy",
        quantization: Optional[str] = None,
    ):
        converter = _get_converter(model_path=model_path, model_type=model_type)
        model_spec = converter.convert_on_the_fly(quantization=quantization)

        variables = model_spec.variables(ordered=True)
        self.vocabularies = model_spec.get_vocabulary()
        self.config = json.dumps(model_spec.config.to_dict())
        aliases = {}

        spec = model_spec.name
        spec_revision = model_spec.revision
        binary_version = model_spec.binary_version
        variables_cpp = dict()

        for key, value in variables:
            if isinstance(value, str):
                aliases[key] = value
            else:
                variables_cpp[key] = ctranslate2.StorageView.from_array(value.numpy())

        self.generator = ctranslate2.Generator(
            spec=spec,
            spec_revision=spec_revision,
            binary_version=binary_version,
            aliases=aliases,
            vocabularies=self.vocabularies,
            variables=variables_cpp,
            config=self.config,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
            max_queued_batches=max_queued_batches,
        )

    def generate_batch(self, prompt, *args, **kwargs):
        return self.generator.generate_batch(prompt, *args, **kwargs)

    def score_batch(self, tokens, *args, **kwargs):
        return self.generator.score_batch(tokens, *args, **kwargs)
