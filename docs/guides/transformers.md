# Transformers

CTranslate2 supports selected models from Hugging Face's [Transformers](https://github.com/huggingface/transformers). The following models are currently supported:

* BART
* M2M100
* MarianMT
* MBART
* OpenAI GPT2

The converter takes as argument the pretrained model name or the path to a model directory:

```bash
pip install transformers[torch]
ct2-transformers-converter --model facebook/m2m100_418M --output_dir ct2_model
```

## Special source tokens

For other frameworks, CTranslate2 implicitly adds special tokens to the source input when required. For example, models converted from Fairseq or Marian will implicitly append `</s>` to the source tokens.

However, these special tokens are not implicitly added for Transformers models since they are already returned by the corresponding tokenizer:

```python
>>> import transformers
>>> tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
>>> tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world!"))
['▁Hello', '▁world', '!', '</s>']
```

## MarianMT

```bash
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir opus-mt-en-de
```

```python
import ctranslate2
import transformers

translator = ctranslate2.Translator("opus-mt-en-de")
tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

source = tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world!"))
results = translator.translate_batch([source])
target = results[0].hypotheses[0]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
```

## M2M-100

```bash
ct2-transformers-converter --model facebook/m2m100_418M --output_dir m2m100_418
```

```python
import ctranslate2
import transformers

translator = ctranslate2.Translator("m2m100_418")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/m2m100_418M")
tokenizer.src_lang = "en"

source = tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world!"))
target_prefix = [tokenizer.lang_code_to_token["de"]]
results = translator.translate_batch([source], target_prefix=[target_prefix])
target = results[0].hypotheses[0][1:]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
```

## GPT-2

```bash
ct2-transformers-converter --model gpt2 --output_dir gpt2_ct2
```

```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("gpt2_ct2")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

# Unconditional generation.
results = generator.generate_batch([[tokenizer.bos_token]], max_length=30, sampling_topk=10)
output = results[0].sequences[0]
print(tokenizer.decode(tokenizer.convert_tokens_to_ids(output)))

# Conditional generation.
start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode("It is"))
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
output = results[0].sequences[0]
print(tokenizer.decode(tokenizer.convert_tokens_to_ids(output)))
```
