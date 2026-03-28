import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def create_spectrogram(file_path, output_path):

    y, sr = librosa.load(file_path)

    D = librosa.stft(y)

    S_db = librosa.amplitude_to_db(abs(D), ref=np.max)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
