import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from src.utils.audio import AudioUtils

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


class Features:
    @staticmethod
    def extract_features(dataset) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        # Use ProcessPoolExecutor for CPU-bound audio processing
        with ProcessPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(Features.process_single_record, dataset.records),
                    total=len(dataset.records),
                    desc="Extracting features",
                )
            )

        for features, label in results:
            X.append(features)
            y.append(label)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.str_)

    @staticmethod
    def process_single_record(record):
        signal = AudioUtils.load_audio(record.path)
        features = AudioUtils.extract_features(signal)
        return features, record.label
