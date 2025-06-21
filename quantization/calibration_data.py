# quantization/calibration_data.py

import numpy as np
from datasets import load_from_disk
from onnxruntime.quantization import CalibrationDataReader

class CodeDatasetCalibrationReader(CalibrationDataReader):
    def __init__(self, dataset_path="data/tokenized/", field="input_ids", max_samples=100):
        dataset = load_from_disk(dataset_path)
        self.samples = []
        for row in dataset.select(range(min(max_samples, len(dataset)))):
            input_ids = np.array(row[field], dtype=np.int64)
            self.samples.append({"input_ids": input_ids[np.newaxis, :]})  # batch dim
        self._iterator = iter(self.samples)

    def get_next(self):
        return next(self._iterator, None)
