import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from config import Config
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

        for feature_list, label in results:
            # feature_list peut contenir 1 ou plusieurs fenêtres
            for features in feature_list:
                X.append(features)
                y.append(label)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.str_)

    @staticmethod
    def process_single_record(record):
        """
        Charge le fichier audio COMPLET et extrait des fenêtres
        de AUDIO_DURATION secondes avec un hop de 50%.

        Si le clip est plus court que AUDIO_DURATION, on le pad.
        Si le clip est exactement AUDIO_DURATION, une seule fenêtre.
        Si le clip est plus long, plusieurs fenêtres chevauchantes.
        """
        signal = AudioUtils.load_audio_full(record.path)

        target_length = int(Config.AUDIO_SAMPLE_RATE * Config.AUDIO_DURATION)
        hop_length = target_length // 2  # 50% overlap

        feature_list = []

        if len(signal) <= target_length:
            # Clip court : pad et extraire une seule fenêtre
            if len(signal) < target_length:
                signal = np.pad(signal, (0, target_length - len(signal)))
            features = AudioUtils.extract_features(signal)
            feature_list.append(features)
        else:
            # Clip long : extraire plusieurs fenêtres chevauchantes
            for start in range(0, len(signal) - target_length + 1, hop_length):
                window = signal[start : start + target_length]
                features = AudioUtils.extract_features(window)
                feature_list.append(features)

            # S'assurer qu'on inclut la fin du clip
            last_start = len(signal) - target_length
            if last_start % hop_length != 0:
                window = signal[last_start:]
                features = AudioUtils.extract_features(window)
                feature_list.append(features)

        return feature_list, record.label
