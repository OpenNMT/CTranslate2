#!/usr/bin/env python3
"""Prepare a reproducible token manifest from a local Parquet text corpus."""

import argparse
import hashlib
import json

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import numpy as np
    import pyarrow.parquet as pq

    from transformers import AutoTokenizer

    texts = pq.read_table(args.parquet, columns=["text"])["text"].to_pylist()
    text = "\n\n".join(texts)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(ids) < 2:
        raise ValueError("Corpus has fewer than two tokens")
    manifest = {
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "config": args.config,
        "split": args.split,
        "source_file": args.parquet.name,
        "source_file_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
        "text_rows": len(texts),
        "text_join": "two newline characters, including empty rows",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "tokenizer_revision": args.tokenizer.resolve().name,
        "tokenizer_sha256": hashlib.sha256(
            (args.tokenizer / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "special_tokens_added": False,
        "tokens_sha256_le_i32": hashlib.sha256(
            np.asarray(ids, dtype="<i4").tobytes()
        ).hexdigest(),
        "tokens": ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest) + "\n")
    print(json.dumps({"tokens": len(ids), "sha256": manifest["tokens_sha256_le_i32"]}))


if __name__ == "__main__":
    main()
