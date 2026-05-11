import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class DDSPDataset(Dataset):
    """
    Dataset of preprocessed DDSP training samples.

    Each sample is a tuple of (audio, f0, loudness) loaded from .npy files.

    When return_class=True, also extracts the timbre class from the filename
    (e.g., "brass_A2_f" → class="brass") and returns it as "timbre_class".
    """

    TIMBRE_CLASSES = ["string", "brass", "voice", "electronic"]

    def __init__(self, data_dir: str, split: str = "train", return_class: bool = False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.return_class = return_class
        self.samples: list[dict] = []
        self._load_manifest()

    def _load_manifest(self):
        manifest_path = self.data_dir / f"{self.split}_manifest.txt"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                name = parts[0]
                sample = {
                    "name": name,
                    "audio_path": str(self.data_dir / f"{name}_audio.npy"),
                    "f0_path": str(self.data_dir / f"{name}_f0.npy"),
                    "loudness_path": str(self.data_dir / f"{name}_loudness.npy"),
                }
                if self.return_class:
                    sample["timbre_class"] = self._parse_class(name)
                self.samples.append(sample)

    @staticmethod
    def _parse_class(name: str) -> str:
        prefix = name.split("_")[0]
        if prefix in DDSPDataset.TIMBRE_CLASSES:
            return prefix
        raise ValueError(f"Unknown timbre class in '{name}': expected one of {DDSPDataset.TIMBRE_CLASSES}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        audio = torch.from_numpy(np.load(sample["audio_path"]))
        f0 = torch.from_numpy(np.load(sample["f0_path"]))
        loudness = torch.from_numpy(np.load(sample["loudness_path"]))
        item = {"audio": audio, "f0": f0, "loudness": loudness, "name": sample["name"]}
        if self.return_class:
            item["timbre_class"] = sample["timbre_class"]
        return item


def collate_variable_length(batch: list[dict]) -> dict:
    """Collate function for variable-length audio clips."""
    audio_list = [b["audio"] for b in batch]
    f0_list = [b["f0"] for b in batch]
    loudness_list = [b["loudness"] for b in batch]
    names = [b["name"] for b in batch]

    audio = torch.stack(audio_list)
    f0 = torch.stack(f0_list)
    loudness = torch.stack(loudness_list)

    result = {"audio": audio, "f0": f0, "loudness": loudness, "name": names}

    if "timbre_class" in batch[0]:
        result["timbre_class"] = [b["timbre_class"] for b in batch]

    return result
