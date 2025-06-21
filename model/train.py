# model/train.py

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from datasets import load_from_disk
import json
from architecture import SmallCodeGenModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "model/config.json"
TOKENIZED_DATA_PATH = "data/tokenized/"

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def train(model, dataloader, optimizer, scheduler, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = input_ids.clone()
            output = model(input_ids, labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Loss = {avg:.4f}")

def main():
    # Load configuration and dataset
    config = load_config(CONFIG_PATH)
    model = SmallCodeGenModel(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        depth=config["depth"],
        heads=config["heads"],
        ff_dim=config["ff_dim"],
        max_len=config["max_position_embeddings"]
    ).to(DEVICE)

    dataset = load_from_disk(TOKENIZED_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 3)

    train(model, dataloader, optimizer, scheduler)

    torch.save(model.state_dict(), "model/slm.pt")
    print("✅ Model trained and saved as model/slm.pt")

if __name__ == "__main__":
    main()
