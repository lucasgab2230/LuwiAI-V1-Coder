# model/architecture.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class SmallCodeGenModel(nn.Module):
    def __init__(self, vocab_size, dim=256, depth=4, heads=4, ff_dim=512, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.position = PositionalEncoding(dim, max_len)
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, ff_dim) for _ in range(depth)
        ])
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.position(x)
        x = self.transformer(x)
        logits = self.lm_head(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return type('Output', (), {'loss': loss, 'logits': logits})
        return type('Output', (), {'logits': logits})
