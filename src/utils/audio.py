import librosa
import numpy as np

from config import Config


class AudioUtils:
    @staticmethod
    def load_audio(filepath, sample_rate=Config.AUDIO_SAMPLE_RATE):
        return librosa.load(filepath, sr=sample_rate)

    @staticmethod
    def extract_features(signal):
        mfcc = librosa.feature.mfcc(
            y=signal, sr=Config.AUDIO_SAMPLE_RATE, n_mfcc=Config.AUDIO_N_MFCC
        )
        zcr = librosa.feature.zero_crossing_rate(signal)
        rms = librosa.feature.rms(y=signal)
        centroid = librosa.feature.spectral_centroid(
            y=signal, sr=Config.AUDIO_SAMPLE_RATE
        )
        features = np.concatenate(
            [
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                [zcr.mean(), zcr.std()],
                [rms.mean(), rms.std()],
                [centroid.mean(), centroid.std()],
            ]
        )
        return features
