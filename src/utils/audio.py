import warnings

import librosa
import numpy as np

from config import Config

warnings.filterwarnings("ignore")


class AudioUtils:
    @staticmethod
    def load_audio(filepath, sample_rate=Config.AUDIO_SAMPLE_RATE) -> np.ndarray:
        target_length = int(Config.AUDIO_SAMPLE_RATE * Config.AUDIO_DURATION)

        signal, _ = librosa.load(filepath, sr=sample_rate, mono=True)

        if len(signal) > target_length:
            signal = signal[:target_length]

        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)))

        return signal

    @staticmethod
    def extract_features(signal):
        mfccs = np.mean(
            librosa.feature.mfcc(
                y=signal, sr=Config.AUDIO_SAMPLE_RATE, n_mfcc=Config.AUDIO_N_MFCC
            ),
            axis=1,
        )
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal), axis=1)
        contrast = np.mean(
            librosa.feature.spectral_contrast(y=signal, sr=Config.AUDIO_SAMPLE_RATE),
            axis=1,
        )
        chroma = np.mean(
            librosa.feature.chroma_stft(y=signal, sr=Config.AUDIO_SAMPLE_RATE), axis=1
        )
        features = np.concatenate((mfccs, chroma, contrast, zcr))
        return features
