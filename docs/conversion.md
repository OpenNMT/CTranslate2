# Model conversion

The core CTranslate2 implementation is framework agnostic. The logic that is specific to each framework is moved to a conversion step that loads supported models into a unified representation. The weights can then be quantized and saved into an optimized binary format.

## Supported frameworks

The Python module includes a [conversion API](python/ctranslate2.converters.rst) and conversion scripts for multiple frameworks:

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :glob:

   guides/*
```

## Model structure

The conversion produces a model directory containing a binary model file and one or more vocabulary files:

```text
model.bin
source_vocabulary.txt
target_vocabulary.txt
```

```{tip}
The Python API exposes the function [`ctranslate2.contains_model`](python/ctranslate2.contains_model.rst) to check if a directory is a CTranslate2 model.
```

## Quantization and reduced precision

The converters support reducing the weights precision to save on space and possibly accelerate the model execution. See the [Quantization](quantization.md) documentation.

## Backward compatibility

Converted models are backward compatibility. This compatibility is rarely broken, even for major versions.

```{attention}
Forward compatibility is not guaranteed, however. The version loading the model should not be older than the version that converted the model.
```

## Portability

Converted models are portable in the sense they can be loaded on another machine using a different operating system or CPU architecture. However, the 2 machines must use the same [endianness](https://en.wikipedia.org/wiki/Endianness) which is usually the case nowadays.

## Add a new converter

You can write your own converter as long as the model architecture is supported by CTranslate2. The converter should populate a model specification with trained weights.

```{tip}
See the [existing converters](https://github.com/OpenNMT/CTranslate2/tree/master/python/ctranslate2/converters) which could be used as templates.
```

### Model specification

A model specification defines the structures and names of the model weights. Converters should fill out this specification with weights coming from a trained model.

In the Python code, a model specification is represented as nested [`LayerSpec`](python/ctranslate2.specs.LayerSpec.rst) objects, where intermediate objects define weights scopes and leaf objects define the weights name and value. This is similar to how you would define a model in PyTorch (using `nn.Module`) or TensorFlow (using `tf.Module`).

The final structure defines the full name of each weight that the C++ code should read when building the model. For example, a weight that can be accessed with `root.encoder.embeddings.weight` (where `root` is the top-level `LayerSpec` object) will have for name `encoder/embeddings/weight` in the serialized model.

Changes in this structure are tracked by a revision number (see next section).

### Model serialization

The model serialization is defined in the Python file [`model_spec.py`](https://github.com/OpenNMT/CTranslate2/blob/master/python/ctranslate2/specs/model_spec.py). It is a simple binary serialization that is easy and fast to load from C++.

Converted models have 2 levels of versioning to manage backward compatibility:

1. Binary version: the structure of the binary file
2. Model specification revision: the variable names expected by each model.

For example, adding a new field in the binary file will increment (1), but changing a variable name will increment (2).
