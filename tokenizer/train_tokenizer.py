import os
import sentencepiece as spm
from pathlib import Path

# Configuration
DATA_DIR = Path("data/raw_docs/")
TOKENIZER_DIR = Path("tokenizer/")
CONFIG_PATH = TOKENIZER_DIR / "tokenizer_config.yaml"
OUTPUT_PREFIX = TOKENIZER_DIR / "slm_tokenizer"

def collect_training_text(output_file="tokenizer/corpus.txt"):
    """Combines all doc files into a single corpus."""
    with open(output_file, "w", encoding="utf-8") as fout:
        for doc_path in DATA_DIR.glob("**/*.txt"):
            with open(doc_path, encoding="utf-8") as fin:
                fout.write(fin.read().strip() + "\n")
    return output_file

def train_tokenizer(corpus_file):
    """Trains the tokenizer using SentencePiece."""
    spm.SentencePieceTrainer.Train(
        input=corpus_file,
        model_prefix=str(OUTPUT_PREFIX),
        config=str(CONFIG_PATH),
        input_format="text"
    )
    print("✅ Tokenizer trained successfully!")

if __name__ == "__main__":
    corpus = collect_training_text()
    train_tokenizer(corpus)
