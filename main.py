"""
Hackathon IA'Hack 2026 (hackathon.uqar.ca)
"""

from config import Config
from src.utils.dataset import AudioDataset

if __name__ == "__main__":
    train_dataset = AudioDataset(Config.DATASET_PATH_AUDIO + Config.PATH_TRAIN)
    test_dataset = AudioDataset(Config.DATASET_PATH_AUDIO + Config.PATH_TEST)

    exit()