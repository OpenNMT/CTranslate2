import os
import os.path

import librosa
import numpy as np
import torch
import torchaudio

from torchaudio.utils import download_asset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    Wav2Vec2BaseModelOutput,
)

import ctranslate2

# Models Conversion & Preparation
compute_type = "int8"
if not os.path.isfile("ctranslate2_model/model.bin"):
    model_name = "facebook/wav2vec2-large-robust-ft-swbd-300h"
    converter = ctranslate2.converters.TransformersConverter(
        model_name,
        load_as_float16=compute_type,
    )
    output_dir = converter.convert("ctranslate2_model")
else:
    output_dir = "ctranslate2_model"

if not os.path.isfile("ctranslate2_model/wav2vec2_partial.bin"):
    w2v2_model = Wav2Vec2ForCTC.from_pretrained(model_name)
    del w2v2_model.wav2vec2.encoder.layers
    del w2v2_model.wav2vec2.encoder.layer_norm
    torch.save(w2v2_model, "ctranslate2_model/wav2vec2_partial.bin")
    w2v2_processor = Wav2Vec2Processor.from_pretrained(model_name)
    torch.save(w2v2_processor, "ctranslate2_model/wav2vec2_processor.bin")


# ASR inference
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
cpu_threads = int(os.environ.get("OMP_NUM_THREADS", 0))

w2v2_model = torch.load("ctranslate2_model/wav2vec2_partial.bin").to(device)
w2v2_processor = torch.load("ctranslate2_model/wav2vec2_processor.bin")

SAMPLE_WAV = download_asset(
    "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
)
resample_rate = 16000
waveform, sampling_rate = torchaudio.load(SAMPLE_WAV)
if sampling_rate != resample_rate:
    speech_array = librosa.resample(
        waveform[0].numpy(),
        orig_sr=sampling_rate,
        target_sr=resample_rate,
    )
else:
    speech_array = waveform[0].numpy()

input_values = w2v2_processor(
    speech_array.astype(np.float32),
    padding=True,
    return_tensors="pt",
    sampling_rate=resample_rate,
).input_values

with torch.no_grad():
    extract_features = w2v2_model.wav2vec2.feature_extractor(
        input_values.to(w2v2_model.device)
    ).transpose(1, 2)
    hidden_states, extract_features = w2v2_model.wav2vec2.feature_projection(
        extract_features
    )
    position_embeddings = w2v2_model.wav2vec2.encoder.pos_conv_embed(hidden_states)
    hidden_states = position_embeddings + hidden_states
    # hidden_states = w2v2_model.encoder.dropout(hidden_states)
    # Dropout(p=0.0, inplace=False) bypassed

ct2_w2v2_model = ctranslate2.models.Wav2Vec2(
    output_dir,
    device=device,
    device_index=[0],
    compute_type=compute_type,
    intra_threads=cpu_threads,
    inter_threads=1,
)

if ct2_w2v2_model.device == "cuda":
    hidden_states = hidden_states.cpu()
else:
    hidden_states.numpy()

hidden_states = np.ascontiguousarray(hidden_states)
hidden_states = ctranslate2.StorageView.from_array(hidden_states)
to_cpu = ct2_w2v2_model.device == "cuda" and len(ct2_w2v2_model.device_index) > 1
ct2_output = ct2_w2v2_model.encode(
    hidden_states,
    to_cpu=to_cpu,
)
# 24 x Wav2Vec2EncoderLayerStableLayerNorm processed
if ct2_w2v2_model.device == "cuda":
    hidden_states = torch.as_tensor(
        ct2_output,
        device=ct2_w2v2_model.device,
    )
else:
    hidden_states = torch.as_tensor(
        np.array(ct2_output),
        dtype=torch.float32,
        device=ct2_w2v2_model.device,
    )

encoder_outputs = BaseModelOutput(
    last_hidden_state=hidden_states,
    hidden_states=None,
    attentions=None,
)
hidden_states = encoder_outputs[0]
outputs = Wav2Vec2BaseModelOutput(
    last_hidden_state=hidden_states,
    extract_features=extract_features,
    hidden_states=encoder_outputs.hidden_states,
    attentions=encoder_outputs.attentions,
)
hidden_states = outputs[0]
# hidden_states = w2v2_model.dropout(hidden_states)
# Dropout(p=0.0, inplace=False) bypassed
with torch.no_grad():
    logits = w2v2_model.lm_head(hidden_states.to(torch.float32))[0]

predicted_ids = torch.argmax(logits, dim=-1)
output = w2v2_processor.decode(predicted_ids, output_word_offsets=True)

print(output["text"])
# should be: I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT
