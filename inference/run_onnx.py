# inference/run_onnx.py

import onnxruntime as ort
import numpy as np
from tokenizer.spm_decode import decode_ids  # Optional: Your decoder if available

MODEL_PATH = "model/slm_ivybridge_int8.onnx"
VOCAB_SIZE = 16000
SEQ_LEN = 128

def load_session(model_path=MODEL_PATH):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def prepare_input(prompt_ids):
    # Ensure input shape = (1, seq_len)
    if len(prompt_ids) < SEQ_LEN:
        padded = prompt_ids + [0] * (SEQ_LEN - len(prompt_ids))
    else:
        padded = prompt_ids[:SEQ_LEN]
    return np.array([padded], dtype=np.int64)

def run_inference(session, input_ids):
    inputs = {"input_ids": input_ids}
    outputs = session.run(None, inputs)
    return outputs[0]  # logits

def sample_next_token(logits, temperature=1.0):
    logits = logits[0, -1] / temperature
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return int(np.random.choice(len(probs), p=probs))

def main():
    session = load_session()
    prompt_ids = [2]  # BOS token ID, replace with your tokenizer's output
    generated = prompt_ids[:]

    for _ in range(SEQ_LEN - 1):
        input_ids = prepare_input(generated)
        logits = run_inference(session, input_ids)
        next_token = sample_next_token(logits)
        generated.append(next_token)
        if next_token == 3:  # EOS
            break

    print("🧠 Generated token IDs:", generated)
    # Optionally decode:
    # print("🔤 Decoded output:", decode_ids(generated))

if __name__ == "__main__":
    main()
