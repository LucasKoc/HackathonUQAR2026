"""
Hackathon IA'Hack 2026 (hackathon.uqar.ca)
"""

from config import Config
from src.utils.dataset import AudioDataset
from src.spectrogram.spectrogram import Spectrogram

if __name__ == "__main__":
    train_dataset = AudioDataset(Config.DATASET_PATH_AUDIO + Config.PATH_TRAIN)
    test_dataset = AudioDataset(Config.DATASET_PATH_AUDIO + Config.PATH_TEST)

    Spectrogram.wave_to_spectrogram(train_dataset, Config.PATH_TRAIN)
    Spectrogram.wave_to_spectrogram(test_dataset, Config.PATH_TEST)
