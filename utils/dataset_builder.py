# utils/dataset_builder.py

import re
import os
from pathlib import Path
from datasets import Dataset
import sentencepiece as spm
import json

RAW_DOCS_PATH = "data/raw_docs/"
TOKENIZED_PATH = "data/tokenized/"
TOKENIZER_MODEL = "tokenizer/slm_tokenizer.model"

def extract_code_blocks(text):
    """
    Extracts fenced code blocks (```...```) from markdown-style documentation.
    """
    blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    cleaned = [block.strip() for block in blocks if block.strip()]
    return cleaned

def load_docs(path=RAW_DOCS_PATH):
    all_snippets = []
    for file in Path(path).rglob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        snippets = extract_code_blocks(text)
        all_snippets.extend(snippets)
    return all_snippets

def tokenize_snippets(snippets, model_path=TOKENIZER_MODEL):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    tokenized = [{"input_ids": sp.encode(snippet, out_type=int)} for snippet in snippets]
    return tokenized

def save_dataset(tokenized_snippets, out_dir=TOKENIZED_PATH):
    os.makedirs(out_dir, exist_ok=True)
    dataset = Dataset.from_list(tokenized_snippets)
    dataset.save_to_disk(out_dir)
    print(f"✅ Saved tokenized dataset with {len(dataset)} samples to {out_dir}")

def main():
    print("🔍 Extracting code examples from documentation...")
    snippets = load_docs()
    print(f"📚 Found {len(snippets)} code snippets.")

    print("🔤 Tokenizing...")
    tokenized = tokenize_snippets(snippets)

    print("💾 Saving dataset...")
    save_dataset(tokenized)

if __name__ == "__main__":
    main()
