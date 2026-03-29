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
    def load_audio_full(filepath, sample_rate=Config.AUDIO_SAMPLE_RATE) -> np.ndarray:
        """Charge le fichier audio COMPLET sans troncature."""
        signal, _ = librosa.load(filepath, sr=sample_rate, mono=True)
        return signal

    @staticmethod
    def extract_features(signal):
        """
        Features :
        - MFCC : mean + std (13×2 = 26)
        - Delta MFCC : mean (13)
        - Chroma : mean + std (12×2 = 24)
        - Spectral contrast : mean (7)
        - ZCR : mean + std (2)
        - Spectral centroid : mean + std (2)
        - Spectral bandwidth : mean + std (2)
        - Spectral rolloff : mean + std (2)
        - RMS energy : mean + std (2)
        Total : ~80 features
        """
        sr = Config.AUDIO_SAMPLE_RATE
        n_mfcc = Config.AUDIO_N_MFCC

        # MFCC (mean + std)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Delta MFCC (mean)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mfccs_mean = np.mean(delta_mfccs, axis=1)

        # Chroma (mean + std)
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Spectral contrast (mean)
        contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # Zero crossing rate (mean + std)
        zcr = librosa.feature.zero_crossing_rate(y=signal)
        zcr_mean = np.mean(zcr, axis=1)
        zcr_std = np.std(zcr, axis=1)

        # Spectral centroid (mean + std)
        centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        centroid_mean = np.mean(centroid, axis=1)
        centroid_std = np.std(centroid, axis=1)

        # Spectral bandwidth (mean + std)
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        bandwidth_mean = np.mean(bandwidth, axis=1)
        bandwidth_std = np.std(bandwidth, axis=1)

        # Spectral rolloff (mean + std)
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
        rolloff_mean = np.mean(rolloff, axis=1)
        rolloff_std = np.std(rolloff, axis=1)

        # RMS energy (mean + std)
        rms = librosa.feature.rms(y=signal)
        rms_mean = np.mean(rms, axis=1)
        rms_std = np.std(rms, axis=1)

        features = np.concatenate(
            (
                mfccs_mean,
                mfccs_std,
                delta_mfccs_mean,
                chroma_mean,
                chroma_std,
                contrast_mean,
                zcr_mean,
                zcr_std,
                centroid_mean,
                centroid_std,
                bandwidth_mean,
                bandwidth_std,
                rolloff_mean,
                rolloff_std,
                rms_mean,
                rms_std,
            )
        )
        return features
