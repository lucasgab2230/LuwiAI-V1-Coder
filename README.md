# 🧠 slm-codegen

A lightweight Small Language Model (SLM) designed specifically for code generation on legacy CPUs (e.g., Intel Ivy Bridge), trained exclusively on official documentation for popular languages and frameworks like Python, JavaScript, C++, React, Django, and more.

---

## 🎯 Project Goals

- Run efficiently on CPUs without AVX2/AVX512 support
- Train on a focused dataset built from curated official documentation
- Output accurate, clean, and idiomatic code snippets from prompts
- Preserve syntax-aware tokenization and fast inference via quantized ONNX

---

## 🏗️ Architecture

- Transformer-based: 4 layers, 256-dimensional embeddings, 4 heads
- Tokenizer: Byte Pair Encoding (BPE) via SentencePiece
- Output: Clean code generation with minimal formatting errors

---

## 📁 Project Structure

slm-codegen/
│
├── data/
│ ├── raw_docs/ # Collected official documentation (.md, .txt, .html)
│ ├── processed/ # Extracted + cleaned code blocks
│ └── tokenized/ # Tokenized samples for training
│
├── tokenizer/
│ ├── tokenizer_config.yaml # SentencePiece or BPE config
│ ├── vocab.model # Trained tokenizer model
│ └── train_tokenizer.py # Script to train tokenizer
│
├── model/
│ ├── architecture.py # Defines lightweight transformer model
│ ├── config.json # Model hyperparameters
│ ├── train.py # Training loop
│ └── export_onnx.py # ONNX export script
│
├── quantization/
│ ├── quantize.py # ONNX Runtime quantization script
│ └── calibration_data.py # Calibration sample loader
│
├── inference/
│ ├── run_onnx.py # ONNX-based inference script
│ ├── runner.cpp # C++ inference (optional)
│ └── build.sh # g++ or CMake build script for Ivy Bridge
│
├── utils/
│ ├── dataset_builder.py # Preprocessing and chunking helpers
│ └── eval.py # Code generation accuracy + lint checker
│
├── tests/
│ └── test_generation.py # Unit tests for model outputs
│
├── README.md
└── requirements.txt

---

## ⚙️ Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Prepare training data:**

```bash
python utils/dataset_builder.py
```

3. **Train the tokenizer:**

```bash
python tokenizer/train_tokenizer.py
```

4. **Train the model:**

```bash
python model/train.py
```

5. **Export to ONNX + Quantize**:

```bash
python model/export_onnx.py
python quantization/quantize.py
```

6. **Run Inference:**

```bash
python inference/run_onnx.py
```

---

## 📊 Evaluation

Evaluate perplexity and code validity:

```bash
python utils/eval.py
```

---

🧪 Tested On

Intel Ivy Bridge CPUs (no AVX2 required)

Python 3.8+

PyTorch, HuggingFace Datasets, ONNX Runtime, SentencePiece

---

💬 Chat with Your Model

You can now interact with your trained model in a conversational way using a simple command-line interface:

```bash
python chat/chat_cli.py
```

Type a prompt like:
`Prompt > create a python function to validate email addresses`

And receive generated code directly from your quantized ONNX model!

`> To exit the chat loop, type exit or quit.`

Make sure your tokenizer model (tokenizer/slm_tokenizer.model) and quantized ONNX model (model/slm_ivybridge_int8.onnx) are built before starting the chat.

---

📜 License

MIT License
