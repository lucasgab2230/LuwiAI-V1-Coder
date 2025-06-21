# utils/eval.py

import torch
import math
import ast
from datasets import load_from_disk
from architecture import SmallCodeGenModel
import json
from pathlib import Path

TOKENIZED_PATH = "data/tokenized/"
MODEL_PATH = "model/slm.pt"
CONFIG_PATH = "model/config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(config_path, weights_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    model = SmallCodeGenModel(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        depth=config["depth"],
        heads=config["heads"],
        ff_dim=config["ff_dim"],
        max_len=config["max_position_embeddings"]
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def compute_perplexity(model, dataset, batch_size=8):
    total_loss = 0
    count = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]["input_ids"]
        input_tensor = torch.tensor(batch, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor, labels=input_tensor)
        total_loss += output.loss.item() * input_tensor.size(0)
        count += input_tensor.size(0)
    return math.exp(total_loss / count)

def check_python_syntax(code_str):
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

def main():
    print("📦 Loading model and dataset...")
    model = load_model(CONFIG_PATH, MODEL_PATH)
    dataset = load_from_disk(TOKENIZED_PATH)

    print("📏 Evaluating perplexity...")
    ppl = compute_perplexity(model, dataset)
    print(f"🧠 Perplexity: {ppl:.2f}")

    # Optional: check syntax on generated outputs (if decoded text is available)
    from tokenizer.spm_decode import decode_ids
    num_valid = 0
    for row in dataset.select(range(100)):
        code = decode_ids(row["input_ids"])
        if check_python_syntax(code):
             num_valid += 1
    print(f"✅ Syntactically valid Python: {num_valid}/100")

if __name__ == "__main__":
    main()
