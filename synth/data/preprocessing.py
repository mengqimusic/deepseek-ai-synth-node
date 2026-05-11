import torch
import numpy as np
from pathlib import Path
from scipy.io import wavfile

from synth.dsp.processors import (
    extract_loudness,
    pre_emphasis,
    midi_to_hz,
)


def load_audio_wav(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load WAV file and resample if needed. Returns [S] float32 tensor."""
    sr, data = wavfile.read(file_path)
    # Convert to float32 and mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio = torch.from_numpy(data.astype(np.float32) / 32767.0)
    # Resample if needed
    if sr != target_sr:
        # Simple linear resampling (works for integer ratio)
        from scipy import signal as sp_signal

        audio_np = audio.numpy()
        n_target = int(len(audio_np) * target_sr / sr)
        audio_np = sp_signal.resample(audio_np, n_target)
        audio = torch.from_numpy(audio_np.astype(np.float32))
    return audio


def process_file(
    input_path: str,
    midi_note: int,
    output_dir: str,
    sample_rate: int = 16000,
    hop_size: int = 64,
    block_size: int = 128,
    pre_emphasis_coef: float = 0.95,
) -> str:
    """
    Process a single audio file into f0, loudness, and pre-emphasized audio.

    Args:
        input_path: Path to raw WAV file
        midi_note: MIDI note number for ground-truth f0
        output_dir: Directory to save .npy files
        sample_rate: Target sample rate
        hop_size: Frame hop for features
        block_size: Frame size for loudness
        pre_emphasis_coef: Pre-emphasis filter coefficient

    Returns:
        base_name: used for output filenames
    """
    audio = load_audio_wav(input_path, target_sr=sample_rate)  # [S]
    name = Path(input_path).stem

    # Extract loudness
    loudness = extract_loudness(
        audio, sample_rate=sample_rate, hop_size=hop_size, block_size=block_size
    )

    # Generate f0 curve from MIDI note (constant pitch per clip)
    f0_hz_val = midi_to_hz(midi_note)
    T = loudness.shape[-1]
    f0 = torch.full((T,), f0_hz_val, dtype=torch.float32)

    # Pre-emphasis
    audio_emph = pre_emphasis(audio, coef=pre_emphasis_coef)

    # Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / f"{name}_audio.npy"), audio_emph.numpy())
    np.save(str(out_dir / f"{name}_f0.npy"), f0.numpy())
    np.save(str(out_dir / f"{name}_loudness.npy"), loudness.numpy())

    return name


def create_manifest(names: list[str], output_dir: str, split: str):
    """Create train/val manifest file."""
    manifest_path = Path(output_dir) / f"{split}_manifest.txt"
    with open(manifest_path, "w") as f:
        for name in names:
            f.write(f"{name}\n")
