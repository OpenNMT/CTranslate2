"""
Summary:

A command line in-memory vectordb that allows for semantic seach of a single PDF.
Creates embeddings, an in-memory vectordb, basic semantic search.

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
from typing import List, Dict, Any, Optional, Union

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

import regex as re
import ctranslate2
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


def load_pdf(filepath: Path) -> str:
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

    pages_with_text = []
    total_pages = 0

    with fitz.open(filepath) as doc:
        total_pages = len(doc)
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages_with_text.append(text.strip())

    if not pages_with_text:
        raise ValueError(
            f"No text layer found in PDF '{filepath.name}'. "
            f"The document has {total_pages} page(s) but none contain extractable text. "
            "This PDF may be scanned images without OCR. "
            "Please run OCR on the document first (e.g., using Adobe Acrobat, ocrmypdf, or similar tools)."
        )

    return "\n\n".join(pages_with_text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    separator_regex = re.compile(r"(?<=[.!?])\s+|\n\n+")
    sentences_with_pos = []
    last_end = 0

    for match in separator_regex.finditer(text):
        segment = text[last_end:match.start()]
        stripped = segment.strip()
        if stripped:
            first_char_offset = segment.find(stripped[0])
            actual_start = last_end + first_char_offset
            sentences_with_pos.append((stripped, actual_start, actual_start + len(stripped)))
        last_end = match.end()

    if last_end < len(text):
        segment = text[last_end:]
        stripped = segment.strip()
        if stripped:
            first_char_offset = segment.find(stripped[0])
            actual_start = last_end + first_char_offset
            sentences_with_pos.append((stripped, actual_start, actual_start + len(stripped)))

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, sent_start, sent_end in sentences_with_pos:
        sentence_len = len(sentence)

        if sentence_len > chunk_size:
            if current_chunk:
                chunks.append({
                    "content": " ".join(s[0] for s in current_chunk),
                    "start_char": current_chunk[0][1],
                    "end_char": current_chunk[-1][2]
                })
                current_chunk = []
                current_length = 0

            start = 0
            sub_offset = sent_start
            while start < len(sentence):
                end = start + chunk_size
                sub_chunk = sentence[start:end]
                chunks.append({
                    "content": sub_chunk,
                    "start_char": sub_offset,
                    "end_char": sub_offset + len(sub_chunk)
                })
                sub_offset += len(sub_chunk) - overlap
                start = end - overlap
            continue

        space_needed = 1 if current_chunk else 0
        if current_length + sentence_len + space_needed > chunk_size:
            if current_chunk:
                chunks.append({
                    "content": " ".join(s[0] for s in current_chunk),
                    "start_char": current_chunk[0][1],
                    "end_char": current_chunk[-1][2]
                })

                overlap_sentences = []
                overlap_size = 0
                for sent_tuple in reversed(current_chunk):
                    if overlap_size + len(sent_tuple[0]) > overlap:
                        break
                    overlap_sentences.insert(0, sent_tuple)
                    overlap_size += len(sent_tuple[0]) + 1

                current_chunk = overlap_sentences
                current_length = sum(len(s[0]) for s in current_chunk)
                if current_chunk:
                    current_length += len(current_chunk) - 1

        current_chunk.append((sentence, sent_start, sent_end))
        current_length += sentence_len + space_needed

    if current_chunk:
        chunks.append({
            "content": " ".join(s[0] for s in current_chunk),
            "start_char": current_chunk[0][1],
            "end_char": current_chunk[-1][2]
        })

    return chunks


class EmbeddingEngine:
    def __init__(self, model_path: str, max_batch_size: int = 32, max_length: int = 512):
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self._encoder = None
        self._tokenizer = None

    def load(self):
        self._encoder = ctranslate2.Encoder(self.model_path, device="cuda", compute_type="bfloat16")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return torch.tensor([])

        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            all_embeddings.append(self._encode_batch(batch))

        embeddings = torch.cat(all_embeddings, dim=0)
        norms = torch.clamp(torch.linalg.norm(embeddings, dim=1, keepdim=True), min=1e-9)
        return embeddings / norms

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        encoded = self._tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors=None)
        output = self._encoder.forward_batch(encoded["input_ids"])

        if output.pooler_output is not None:
            storage = output.pooler_output.to(ctranslate2.DataType.float16)
            return torch.as_tensor(storage, device="cuda").to(torch.bfloat16)

        storage = output.last_hidden_state.to(ctranslate2.DataType.float16)
        last_hidden = torch.as_tensor(storage, device="cuda").to(torch.bfloat16)
        attention_mask = torch.tensor(encoded["attention_mask"], device="cuda")
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.shape).to(last_hidden.dtype)
        sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
        return sum_hidden / sum_mask


class VectorStore:
    def __init__(self):
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings: Optional[torch.Tensor] = None

    def add(self, chunks: List[Dict[str, Any]], embeddings: torch.Tensor):
        if self._embeddings is not None and self._embeddings.numel() > 0:
            self._embeddings = torch.cat([self._embeddings, embeddings], dim=0)
        else:
            self._embeddings = embeddings
        self._chunks.extend(chunks)

    def search(self, query_embedding: torch.Tensor, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._chunks:
            return []

        query = query_embedding.flatten()
        query_norm = torch.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        similarities = torch.matmul(self._embeddings, query)
        top_k = min(top_k, len(similarities))
        scores, indices = torch.topk(similarities, top_k)

        return [
            {"chunk": self._chunks[idx], "score": score, "rank": rank + 1}
            for rank, (idx, score) in enumerate(zip(indices.tolist(), scores.tolist()))
        ]


def main():
    model_repo = "CTranslate2HQ/all-MiniLM-L12-v2-ct2-float32"
    file_path = Path(r"[ENTER RAW STRING PATH TO A PDF FILE HERE]")
    chunk_size = 900
    overlap = 300
    top_k = 4

    print(f"Downloading/verifying model: {model_repo}...")
    model_path = snapshot_download(repo_id=model_repo)

    print("Loading embedding model...")
    engine = EmbeddingEngine(model_path)
    engine.load()

    print(f"Loading PDF: {file_path}")
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    if file_path.suffix.lower() != ".pdf":
        print(f"Error: Expected a PDF file, got: {file_path.suffix}")
        return

    try:
        content = load_pdf(file_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Extracted {len(content)} characters from PDF")

    print(f"Chunking text (size={chunk_size}, overlap={overlap})...")
    chunks = chunk_text(content, chunk_size, overlap)
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        print("No chunks created. Exiting.")
        return

    print("Creating embeddings...")
    embeddings = engine.encode([chunk["content"] for chunk in chunks])

    vector_store = VectorStore()
    vector_store.add(chunks, embeddings)

    print("\n" + "=" * 50)
    print("Vector database ready. Enter queries to search.")
    print("Type 'quit' or 'exit' to end.")
    print("=" * 50 + "\n")

    while True:
        query = input("Query: ").strip()

        if not query:
            continue

        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        query_embedding = engine.encode(query)[0]
        results = vector_store.search(query_embedding, top_k=top_k)

        print(f"\nTop {len(results)} results:\n")
        print("-" * 50)
        for result in results:
            print(f"#{result['rank']} (score: {result['score']:.4f})")
            print()
            print(result['chunk']['content'])
            print()
            print("-" * 50)
        print()


if __name__ == "__main__":
    main()
