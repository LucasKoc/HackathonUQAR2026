import warnings
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
from tqdm import tqdm

from src.utils.audio import AudioUtils

from config import Config

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

    @staticmethod
    def build_window_dataset(wav_files, labels, window_size_sec=0.05, hop_size_sec=0.025):
        all_features = []
        all_labels = []
        all_group_ids = []

        window_samples = int(window_size_sec * Config.AUDIO_SAMPLE_RATE)
        hop_samples = int(hop_size_sec * Config.AUDIO_SAMPLE_RATE)

        for i, (wf, label) in enumerate(zip(wav_files, labels)):
            if i % 50 == 0:
                print(f"Traitement {i + 1}/{len(wav_files)}...")

            audio, Config.AUDIO_SAMPLE_RATE = librosa.load(wf, sr=Config.AUDIO_SAMPLE_RATE)
            group_id = wf  # pour éviter fuite train/test

            for start_sample in range(0, len(audio) - window_samples + 1, hop_samples):
                end_sample = start_sample + window_samples
                segment = audio[start_sample:end_sample]
                if len(segment) < window_samples:
                    continue

                features = AudioUtils.extract_features(segment)
                all_features.append(features)
                all_labels.append(label)
                all_group_ids.append(group_id)

        return np.array(all_features), np.array(all_labels), np.array(all_group_ids)
