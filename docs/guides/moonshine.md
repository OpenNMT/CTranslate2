# Marian

CTranslate2 supports [Moonshine](https://github.com/usefulsensors/moonshine) transcription models. The conversion requires the paths to the model and vocabularies:

See the following repos for moonshine model.safetensor and tokenizer.json files: [tiny](https://huggingface.co/UsefulSensors/moonshine-tiny/tree/main) [base](https://huggingface.co/UsefulSensors/moonshine-base/tree/main).

```bash
ct2-moonshine-converter --model_path model.safetensors --vocab_path tokenizer.json --moonshine_variant tiny \
    --output_dir ct2_model
```
