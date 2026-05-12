#!/usr/bin/env python3
"""Run RichParamModel inference: reconstruction and synthesis."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from synth.nn.model import RichParamModel
from synth.dsp.processors import de_emphasis, midi_to_hz


def _save_audio(audio: torch.Tensor, path: str, sample_rate: int = 16000):
    """Save float32 audio tensor as 16-bit WAV."""
    audio_np = audio.numpy().astype(np.float32)
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(path, sample_rate, audio_int16)


def reconstruct(model, f0_path, loudness_path, audio_path, output_path, device):
    """Reconstruct audio from pre-extracted features."""
    f0 = torch.from_numpy(np.load(f0_path)).unsqueeze(0).to(device)
    loudness = torch.from_numpy(np.load(loudness_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        audio = model(f0, loudness).squeeze(0).cpu()
    audio = de_emphasis(audio)
    _save_audio(audio, str(output_path), 16000)
    print(f"  Reconstruction → {output_path}")


def synthesize(model, f0_hz, duration_sec, loudness_db, output_path, device):
    """Synthesize a sustained note at given pitch and loudness."""
    hop_size = 64
    sample_rate = 16000
    n_frames = int(duration_sec * sample_rate / hop_size)
    f0 = torch.full((1, n_frames), f0_hz, device=device)
    loudness = torch.full((1, n_frames), loudness_db, device=device)
    with torch.no_grad():
        audio = model(f0, loudness).squeeze(0).cpu()
    audio = de_emphasis(audio)
    _save_audio(audio, str(output_path), sample_rate)
    print(f"  Synthesis → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RichParamModel inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--config", default="configs/phase10a.yaml", help="Config file")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--data_dir", default="data/processed", help="Processed data dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["reconstruct", "synthesize", "both"], default="both")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = RichParamModel(
        sample_rate=config["data"]["sample_rate"],
        block_size=config["data"]["block_size"],
        table_size=config["model"].get("table_size", 2048),
        transformer_dim=config["model"]["transformer_dim"],
        transformer_heads=config["model"]["transformer_heads"],
        transformer_layers=config["model"]["transformer_layers"],
        gru_hidden=config["model"]["gru_hidden"],
        n_harmonics=config["model"]["n_harmonics"],
        n_noise_mel=config["model"]["n_noise_mel"],
        n_noise_grain=config["model"]["n_noise_grain"],
        beta_max=config["model"].get("beta_max", 0.02),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Read validation manifest for reconstruction examples
    val_manifest = Path(args.data_dir) / "val_manifest.txt"
    val_names = []
    if val_manifest.exists():
        with open(val_manifest) as f:
            val_names = [line.strip() for line in f if line.strip()]

    if args.mode in ("reconstruct", "both"):
        n_reconstruct = min(5, len(val_names))
        for name in val_names[:n_reconstruct]:
            f0_path = Path(args.data_dir) / f"{name}_f0.npy"
            loudness_path = Path(args.data_dir) / f"{name}_loudness.npy"
            audio_path = Path(args.data_dir) / f"{name}_audio.npy"
            output_path = out_dir / f"{name}_recon.wav"
            reconstruct(model, str(f0_path), str(loudness_path), str(audio_path),
                       str(output_path), args.device)

    if args.mode in ("synthesize", "both"):
        # Synthesize novel pitches at different loudness levels
        synths = [
            # (midi, note_name, loudness_db)
            (55, "G3", -10),   # G3 at mf
            (67, "G4", -15),   # G4 at p
            (72, "C5", -5),    # C5 at f
        ]
        for midi, name, loudness_db in synths:
            f0_hz = midi_to_hz(midi)
            output_path = out_dir / f"synth_{name}_midi{midi}.wav"
            synthesize(model, f0_hz, 4.0, loudness_db, str(output_path), args.device)

        # Sustained notes (10s) for temporal stability test
        sustained = [
            (60, "C4", -10),
            (72, "C5", -10),
        ]
        for midi, name, loudness_db in sustained:
            f0_hz = midi_to_hz(midi)
            output_path = out_dir / f"synth_{name}_midi{midi}_10s.wav"
            synthesize(model, f0_hz, 10.0, loudness_db, str(output_path), args.device)

    print("Done.")


if __name__ == "__main__":
    main()
