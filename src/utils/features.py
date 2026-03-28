from sklearn.utils import deprecated
from tqdm import tqdm

from src.utils.audio import AudioUtils


class Features:
    @staticmethod
    @deprecated
    def extract_features(dataset) -> tuple[list, list]:
        X, y = [], []
        for i in tqdm(range(len(dataset.records))):
            signal, _ = AudioUtils.load_audio(dataset.records[i].path)
            features = AudioUtils.extract_features(signal)
            X.append(features)
            y.append(dataset.records[i].label)
        return X, y