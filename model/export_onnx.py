# model/export_onnx.py

import torch
import json
from architecture import SmallCodeGenModel

CONFIG_PATH = "model/config.json"
MODEL_PATH = "model/slm.pt"
EXPORT_PATH = "model/slm_ivybridge.onnx"
SEQ_LEN = 128  # or whatever your max input length is

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    config = load_config(CONFIG_PATH)
    model = SmallCodeGenModel(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        depth=config["depth"],
        heads=config["heads"],
        ff_dim=config["ff_dim"],
        max_len=config["max_position_embeddings"]
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    dummy_input = torch.randint(low=0, high=config["vocab_size"], size=(1, SEQ_LEN), dtype=torch.long)

    torch.onnx.export(
        model,
        dummy_input,
        EXPORT_PATH,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        opset_version=13
    )

    print(f"✅ Exported model to {EXPORT_PATH}")

if __name__ == "__main__":
    main()
