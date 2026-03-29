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
            futures = [
                executor.submit(
                    Spectrogram.create_spectrogram,
                    record.path,
                    record.label,
                    data_type,
                )
                for record in dataset.records
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f'Transform .wav to spectrogram | "{data_type}" dataset',
            ):
                future.result()

    @staticmethod
    def create_spectrogram(
        file_path: Path,
        label: str,
        data_type: str = Config.PATH_TRAIN,
        out_path: Path | None = None,
    ) -> Path:
        if out_path is None:
            out_path = file_path.parent

        out_path = Path(out_path)
        output_dir = out_path / data_type / label
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{file_path.stem}.png"

        y, sr = librosa.load(file_path, sr=None)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close()

        return output_path

    @staticmethod
    def create_single_spectrogram(
        file_path: str | Path, output_path: str | Path
    ) -> Path:
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        y, sr = librosa.load(file_path, sr=None)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close()

        return output_path
