# Text generation with GPT-2

This example demonstrates how to quantize the large GPT-2 model to 8-bit and use it for text generation.

**1\. Install the requirements:**

```bash
pip install -r requirements.txt
```

**2\. Download the model:**

```bash
python3 download_model.py 1558M
```

**3\. Convert and quantize the model:**

```bash
ct2-openai-gpt2-converter --model_dir models/1558M --output_dir models/1558M/ct2 --quantization int8
```

Note that the converted model is almost 4 times smaller on disk thanks to quantization:

```bash
$ du -h models/1558M/ct2
1.5G	models/1558M/ct2
```

**4\. Start the interactive generation script:**

```bash
python3 interactive_generation.py models/ 1558M
```

The script starts a loop which reads a text input and writes the model prediction. The text input can be empty in which case the generation starts from nothing.

Use `Ctrl-D` or `Ctrl-C` to exit the script.
