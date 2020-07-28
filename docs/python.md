# Python

```python
import ctranslate2
```

## Model conversion API

```python
converter = ctranslate2.converters.OpenNMTTFConverter(
    model_path: str = None,  # Path to a OpenNMT-tf checkpoint or SavedModel (mutually exclusive with variables)
    src_vocab: str = None,   # Path to the source vocabulary (required for checkpoints).
    tgt_vocab: str = None,   # Path to the target vocabulary (required for checkpoints).
    variables: dict = None)  # Dict of variables name to value (mutually exclusive with model_path).

converter = ctranslate2.converters.OpenNMTPyConverter(
    model_path: str)         # Path to the OpenNMT-py model.

output_dir = converter.convert(
    output_dir: str,          # Path to the output directory.
    model_spec: ModelSpec,    # A model specification instance from ctranslate2.specs.
    vmap: str = None,         # Path to a vocabulary mapping file.
    quantization: str = None, # Weights quantization: "int8" or "int16".
    force: bool = False)      # Override output_dir if it exists.
```

## Translation API

```python
translator = ctranslate2.Translator(
    model_path: str                 # Path to the CTranslate2 model directory.
    device: str = "cpu",            # The device to use: "cpu", "cuda", or "auto".
    device_index: int = 0,          # The index of the device to place this translator on.
    compute_type: str = "default"   # The computation type: "default", "int8", "int16", "float16", or "float",
                                    # or a dict mapping a device to a computation type.
    inter_threads: int = 1,         # Maximum number of concurrent translations (CPU only).
    intra_threads: int = 4)         # Threads to use per translation (CPU only).

# Properties:
translator.device              # Device this translator is running on.
translator.device_index        # Device index this translator is running on.
translator.num_translators     # Number of translators backing this instance.
translator.num_queued_batches  # Number of batches waiting to be translated.

# output is a 2D list [batch x num_hypotheses] containing dict with keys:
# * "score"
# * "tokens"
# * "attention" (if return_attention is set to True)
output = translator.translate_batch(
    source: list,                      # A list of list of string.
    target_prefix: list = None,        # An optional list of list of string.
    max_batch_size: int = 0,           # Maximum batch size to run the model on.
    batch_type: str = "examples",      # Whether max_batch_size is the number of examples or tokens.
    beam_size: int = 2,                # Beam size (set 1 to run greedy search).
    num_hypotheses: int = 1,           # Number of hypotheses to return (should be <= beam_size
                                       # unless return_alternatives is set).
    length_penalty: float = 0,         # Length penalty constant to use during beam search.
    coverage_penalty: float = 0,       # Converage penalty constant to use during beam search.
    max_decoding_length: int = 250,    # Maximum prediction length.
    min_decoding_length: int = 1,      # Minimum prediction length.
    use_vmap: bool = False,            # Use the vocabulary mapping file saved in this model.
    return_scores: bool = True,        # Include the prediction scores in the output.
    return_attention: bool = False,    # Include the attention vectors in the output.
    return_alternatives: bool = False, # Return alternatives at the first unconstrained decoding position.
    sampling_topk: int = 1,            # Randomly sample predictions from the top K candidates (with beam_size=1).
    sampling_temperature: float = 1)   # Sampling temperature to generate more random samples.

# stats is a tuple of file statistics containing in order:
# 1. the number of generated target tokens
# 2. the number of translated examples
# 3. the total translation time in milliseconds
stats = translator.translate_file(
    input_path: str,                # Input file.
    output_path: str,               # Output file.
    max_batch_size: int,            # Maximum batch size to run the model on.
    read_batch_size: int = 0,       # Number of sentences to read at once.
    batch_type: str = "examples",   # Whether the batch size is the number of examples or tokens.
    beam_size: int = 2,
    num_hypotheses: int = 1,
    length_penalty: float = 0,
    coverage_penalty: float = 0,
    max_decoding_length: int = 250,
    min_decoding_length: int = 1,
    use_vmap: bool = False,
    with_scores: bool = False,
    sampling_topk: int = 1,
    sampling_temperature: float = 1,
    tokenize_fn: callable = None,   # Function with signature: string -> list of strings
    detokenize_fn: callable = None) # Function with signature: list of strings -> string
```

Also see the [`TranslationOptions`](../include/ctranslate2/translator.h) structure for more details about the options.

## Memory management API

* `translator.unload_model(to_cpu: bool = False)`<br/>Unload the model attached to this translator but keep enough runtime context to quickly resume translation on the initial device. When `to_cpu` is `True`, the model is moved to the CPU memory and not fully unloaded.
* `translator.load_model()`<br/>Load the model back to the initial device.
* `translator.model_is_loaded`<br/>Property set to `True` when the model is loaded on the initial device and ready to be used.
* `del translator`<br/>Release the translator resources.

When using multiple Python threads, the application should ensure that no translations are running before calling these functions.

## Utility API

* `ctranslate2.contains_model(path: str)`<br/>Helper function to check if a directory seems to contain a CTranslate2 model.
