from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class AudioRecord:
    path: Path
    label: str


class AudioDataset:
    def __init__(self, path: str):
        self.path = Path(path)
        self.records = self._index_files()

    def _index_files(self) -> list[AudioRecord]:
        records = []
        for class_dir in sorted(self.path.iterdir()):
            if class_dir.is_dir():
                label = class_dir.name
                for wav_file in sorted(class_dir.glob("*.wav")):
                    records.append(AudioRecord(path=wav_file, label=label))
        return records
