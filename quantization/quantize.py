# quantization/quantize.py

import os
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from calibration_data import CodeDatasetCalibrationReader

MODEL_PATH = "model/slm_ivybridge.onnx"
QUANT_MODEL_PATH = "model/slm_ivybridge_int8.onnx"

class DummyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, seq_len=128, batch_size=1, vocab_size=16000, steps=10):
        self.steps = steps
        self.data_iter = iter([
            {"input_ids": np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)}
            for _ in range(steps)
        ])

    def get_next(self):
        return next(self.data_iter, None)

def run_quantization():
    print(f"📦 Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    dr = CodeDatasetCalibrationReader()
    quantize_static(
        model_input=MODEL_PATH,
        model_output=QUANT_MODEL_PATH,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        optimize_model=False  # Skip extra graph optimization passes
    )

    print(f"✅ Quantized model saved to: {QUANT_MODEL_PATH}")

if __name__ == "__main__":
    run_quantization()
