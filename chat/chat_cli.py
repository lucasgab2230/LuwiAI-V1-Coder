# chat/chat_cli.py

import numpy as np
from inference.run_onnx import load_session, prepare_input, run_inference, sample_next_token
from tokenizer.spm_decode import decode_ids  # You must implement this if not done yet
from sentencepiece import SentencePieceProcessor

MODEL_PATH = "model/slm_ivybridge_int8.onnx"
TOKENIZER_PATH = "tokenizer/slm_tokenizer.model"
MAX_GEN_TOKENS = 100
MAX_SEQ_LEN = 128

def chat():
    print("👋 Welcome to slm-codegen interactive mode! Type 'exit' to quit.\n")

    session = load_session(MODEL_PATH)
    sp = SentencePieceProcessor(model_file=TOKENIZER_PATH)

    while True:
        user_input = input("🧑‍💻 Prompt > ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        input_ids = sp.encode(user_input, out_type=int)
        generated = input_ids[:]

        for _ in range(MAX_GEN_TOKENS):
            input_ids_padded = prepare_input(generated)
            logits = run_inference(session, input_ids_padded)
            next_token = sample_next_token(logits)
            generated.append(next_token)
            if next_token == 3:  # eos_token_id
                break

        result = sp.decode(generated)
        print("\n🤖 Output:\n", result, "\n")

if __name__ == "__main__":
    chat()
