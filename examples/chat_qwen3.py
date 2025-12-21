"""
Summary:

A command line chat with Qwen3-4B model, with memory, memory purging if the context gets too long.

Notes:

This script uses the "set_cuda_paths" function to add to temporarily add to the system's PATH
where the pip-installed CUDA libraries are.  If you install CUDA systemwide (as most do) no need.

Pip installing CUDA libraries always required compatible version of Torch & CUDA.

For example:

pip install https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=c97dc47a1f64745d439dd9471a96d216b728d528011029b4f9ae780e985529e0
pip install nvidia-cublas-cu12==12.8.4.1
pip install nvidia-cudnn-cu12==9.10.2.21
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict

def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path_runtime = nvidia_base_path / 'cuda_runtime' / 'bin'
    cuda_path_runtime_lib = nvidia_base_path / 'cuda_runtime' / 'lib' / 'x64'
    cuda_path_runtime_include = nvidia_base_path / 'cuda_runtime' / 'include'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    nvrtc_path = nvidia_base_path / 'cuda_nvrtc' / 'bin'
    nvcc_path = nvidia_base_path / 'cuda_nvcc' / 'bin'
    paths_to_add = [
        str(cuda_path_runtime),
        str(cuda_path_runtime_lib),
        str(cuda_path_runtime_include),
        str(cublas_path),
        str(cudnn_path),
        str(nvrtc_path),
        str(nvcc_path),
    ]
    current_value = os.environ.get('PATH', '')
    new_value = os.pathsep.join(paths_to_add + ([current_value] if current_value else []))
    os.environ['PATH'] = new_value

    triton_cuda_path = nvidia_base_path / 'cuda_runtime'
    current_cuda_path = os.environ.get('CUDA_PATH', '')
    new_cuda_path = os.pathsep.join([str(triton_cuda_path)] + ([current_cuda_path] if current_cuda_path else []))
    os.environ['CUDA_PATH'] = new_cuda_path

set_cuda_paths()

import ctranslate2
import torch
import gc
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

MODEL_REPO = "CTranslate2HQ/Qwen3-4b-ct2-int8"

CONTEXT_LENGTH = 8192
MAX_GENERATION_LENGTH = 4096
MAX_PROMPT_LENGTH = CONTEXT_LENGTH - MAX_GENERATION_LENGTH

SAMPLING_TEMPERATURE = 0.7
SAMPLING_TOP_K = 50
SAMPLING_TOP_P = 1.0
REPETITION_PENALTY = 1.0

END_TOKEN: Optional[str] = None


def download_model(repo_id: str) -> str:
    print(f"Downloading/verifying model: {repo_id}...")
    model_path = snapshot_download(repo_id=repo_id)
    print(f"Model ready at: {model_path}")
    return model_path


def create_generator(model_path: str) -> ctranslate2.Generator:
    generator = ctranslate2.Generator(
        model_path,
        device="cuda",
        compute_type="int8",
    )
    return generator


def format_prompt_from_dialog(
    tokenizer: AutoTokenizer,
    dialog: List[Dict[str, str]]
) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            dialog,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception as e:
        print(f"Warning: apply_chat_template failed ({e}). Using basic format.")
        prompt = ""
        for msg in dialog:
            role = msg["role"].capitalize()
            prompt += f"{role}: {msg['content']}\n\n"
        prompt += "Assistant:"
    return prompt


def get_prompt_tokens(tokenizer: AutoTokenizer, dialog: List[Dict[str, str]]) -> List[str]:
    prompt = format_prompt_from_dialog(tokenizer, dialog)
    token_ids = tokenizer.encode(prompt)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return tokens


def trim_dialog(dialog: List[Dict[str, str]], has_system: bool) -> List[Dict[str, str]]:
    if has_system:
        return [dialog[0]] + dialog[3:]
    else:
        return dialog[2:]


def generate_streaming(
    tokens: List[str],
    generator: ctranslate2.Generator,
    tokenizer: AutoTokenizer,
    end_token: Optional[str] = None,
) -> str:

    end_token_str = None
    if end_token:
        end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)
        if end_token_ids:
            end_token_str = tokenizer.convert_ids_to_tokens(end_token_ids)[0]

    gen_kwargs = {
        "max_length": MAX_GENERATION_LENGTH,
        "sampling_temperature": SAMPLING_TEMPERATURE,
        "sampling_topk": SAMPLING_TOP_K,
        "sampling_topp": SAMPLING_TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
    }

    if end_token_str:
        gen_kwargs["end_token"] = end_token_str

    full_response = ""

    for step in generator.generate_tokens([tokens], **gen_kwargs):
        token = step.token
        token_id = step.token_id

        if token_id == tokenizer.eos_token_id:
            break
        if end_token_str and token == end_token_str:
            break
        if token in tokenizer.all_special_tokens:
            break

        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        print(decoded, end="", flush=True)
        full_response += decoded

    print()
    return full_response


def cleanup(generator, tokenizer):
    del generator
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


def main():
    model_path = download_model(MODEL_REPO)
    
    print(f"Loading model...")
    generator = create_generator(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded.\n")

    system_message: Optional[str] = "You are a helpful AI assistant."
    
    dialog: List[Dict[str, str]] = []
    has_system = False
    
    if system_message:
        dialog.append({"role": "system", "content": system_message})
        has_system = True

    print("Chat started. Type 'quit' or 'exit' to end.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            dialog.append({"role": "user", "content": user_input})

            while True:
                prompt_tokens = get_prompt_tokens(tokenizer, dialog)
                if len(prompt_tokens) <= MAX_PROMPT_LENGTH:
                    break
                if len(dialog) <= (2 if has_system else 1):
                    print("Warning: Single message exceeds max prompt length.")
                    break
                dialog = trim_dialog(dialog, has_system)

            print("\nAssistant: ", end="", flush=True)
            response = generate_streaming(prompt_tokens, generator, tokenizer, END_TOKEN)
            print()

            dialog.append({"role": "assistant", "content": response.strip()})

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    finally:
        cleanup(generator, tokenizer)


if __name__ == "__main__":
    main()
