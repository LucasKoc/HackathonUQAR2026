from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import Config
from src.utils.dataset import AudioDataset


class Spectrogram:
    @staticmethod
    def wave_to_spectrogram(dataset: AudioDataset, data_type: str) -> None:
        with ProcessPoolExecutor() as executor:
            train_futures = [
                executor.submit(
                    Spectrogram.create_spectrogram,
                    record.path,
                    record.label,
                    data_type,
                )
                for record in dataset.records
            ]

            for future in tqdm(
                as_completed(train_futures),
                total=len(train_futures),
                desc=f'Transform .wav to spectrogram | "{data_type}" dataset',
            ):
                future.result()

    @staticmethod
    def create_spectrogram(
        file_path: Path,
        label,
        data_type=Config.PATH_TRAIN,
        out_path: Path | None = None,
    ) -> None:
        if out_path is None:
            out_path = file_path.parent
        output_path = Path(out_path) + data_type + "/" + label + "/" + file_path.stem + ".png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        y, sr = librosa.load(file_path)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()
