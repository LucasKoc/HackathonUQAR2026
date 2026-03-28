"""
Hackathon IA'Hack 2026 (hackathon.uqar.ca)
"""
from tqdm import tqdm

from config import Config
from src.utils.audio import AudioUtils
from src.utils.dataset import AudioDataset

if __name__ == "__main__":
    train_dataset = AudioDataset(Config.DATASET_PATH_AUDIO + Config.PATH_TRAIN)
    test_dataset = AudioDataset(Config.DATASET_PATH_AUDIO + Config.PATH_TEST)

    X, y = [], []
    for i in tqdm(range(len(train_dataset.records))):
        signal, _ = AudioUtils.load_audio(train_dataset.records[i].path)
        features = AudioUtils.extract_features(signal)
        X.append(features)
        y.append(train_dataset.records[i].label)

    exit()