import sys
import os
import ctranslate2

from encoder import get_encoder


models_dir = sys.argv[1]
model_name = sys.argv[2]

# Load the BPE encoder.
bpe = get_encoder(model_name, models_dir)

# If you don't have a GPU, change "cuda" to "cpu" below (the execution will be slower).
generator = ctranslate2.Generator(os.path.join(models_dir, model_name, "ct2"), device="cuda")

while True:
    input_text = input(">>> ")
    input_tokens = bpe.encode(input_text) if input_text else ["<|endoftext|>"]

    results = generator.generate_batch(
        [input_tokens],
        sampling_topk=20,
        sampling_temperature=1.0,
        max_length=128,
    )

    output_tokens = results[0].sequences[0]
    if input_text:
        # The first token is used as the decoder start token and is currently not included
        # in the output sequence, so we should add it back.
        output_tokens = [input_tokens[0]] + output_tokens

    output_text = bpe.decode(output_tokens)
    print(output_text)
