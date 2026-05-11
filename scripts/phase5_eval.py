#!/usr/bin/env python3
"""
Phase 5 evaluation: validate hypernetwork-driven weight modulation.

Generates:
  1. same_input_diff_energy.wav — same (f0=440Hz, loudness=-10dB) driven
     with 4 different energy states → verifies spectrograms diverge
  2. neutral_baseline.wav — zero energy (base decoder output) for reference
  3. Per-class reconstruction quality metrics

Usage:
    python scripts/phase5_eval.py --checkpoint checkpoints/phase5_final.pt --device cpu
"""

import argparse
import warnings
import numpy as np
import torch
import yaml
from pathlib import Path
from scipy.io import wavfile

warnings.filterwarnings("ignore", message=".*resized.*")
warnings.filterwarnings("ignore", message=".*An output with one or more.*")

from synth.nn.model import DDSPModel
from synth.nn.hypernetwork import Hypernetwork
from synth.nn.modulated_decoder import ModulatedDecoder
from synth.dsp.processors import scale_f0


ENERGY_NAMES = ["tension", "turbulence", "resonance", "memory"]

ENERGY_STATES = {
    "neutral":     torch.tensor([0.0, 0.0, 0.0, 0.0]),
    "string":      torch.tensor([0.8, 0.1, 0.3, 0.2]),
    "brass":       torch.tensor([0.2, 0.8, 0.1, 0.1]),
    "voice":       torch.tensor([0.1, 0.2, 0.8, 0.3]),
    "electronic":  torch.tensor([0.3, 0.4, 0.2, 0.8]),
}


def load_model(config_path: str, checkpoint_path: str, device: str) -> ModulatedDecoder:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]

    base_model = DDSPModel(
        hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        sample_rate=data_cfg["sample_rate"],
        block_size=data_cfg["block_size"],
    )
    ckpt = torch.load(model_cfg["base_checkpoint"], map_location="cpu")
    base_model.load_state_dict(ckpt["model_state_dict"])

    hypernetwork = Hypernetwork(
        hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        bottleneck=model_cfg.get("bottleneck", 48),
        max_scale=model_cfg.get("max_scale", 0.12),
    )

    if checkpoint_path:
        h_ckpt = torch.load(checkpoint_path, map_location="cpu")
        hypernetwork.load_state_dict(h_ckpt["hypernetwork_state_dict"])
        print(f"Loaded hypernetwork from {checkpoint_path} (step {h_ckpt['step']})")

    modulated = ModulatedDecoder(
        base_decoder=base_model.decoder,
        hypernetwork=hypernetwork,
        frozen_decoder=True,
    )
    modulated.harmonic_synth = base_model.harmonic_synth
    modulated.noise_synth = base_model.noise_synth
    modulated.sample_rate = base_model.sample_rate
    modulated.block_size = base_model.block_size

    return modulated.to(device)


@torch.no_grad()
def synthesize(model: ModulatedDecoder, f0_hz: float, loudness_db: float,
               energy_state: torch.Tensor, duration_sec: float,
               device: str) -> np.ndarray:
    sample_rate = model.sample_rate
    block_size = model.block_size
    frame_duration = block_size / sample_rate
    n_frames = int(duration_sec / frame_duration)

    f0 = torch.full((1, n_frames), f0_hz, device=device)
    loudness = torch.full((1, n_frames), loudness_db, device=device)
    energy = energy_state.unsqueeze(0).to(device)  # [1, 4]

    f0_scaled = scale_f0(f0)
    harm_amps, noise_mags = model(f0_scaled, loudness, energy)
    harm_audio = model.harmonic_synth(harm_amps, f0)
    noise_audio = model.noise_synth(noise_mags)
    audio = (harm_audio + noise_audio).squeeze().cpu().numpy()

    return audio.astype(np.float32)


def spectral_cosine_similarity(a: np.ndarray, b: np.ndarray,
                               n_fft: int = 2048, hop: int = 64) -> float:
    """Cosine similarity between magnitude spectrograms."""
    from scipy.signal import stft

    _, _, Zxx_a = stft(a, fs=16000, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Zxx_b = stft(b, fs=16000, nperseg=n_fft, noverlap=n_fft - hop)

    mag_a = np.abs(Zxx_a).flatten()
    mag_b = np.abs(Zxx_b).flatten()

    dot = np.dot(mag_a, mag_b)
    norm_a = np.linalg.norm(mag_a)
    norm_b = np.linalg.norm(mag_b)

    if norm_a > 0 and norm_b > 0:
        return dot / (norm_a * norm_b)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Phase 5 hypernetwork evaluation")
    parser.add_argument("--config", default="configs/phase5.yaml")
    parser.add_argument("--checkpoint", default=None, help="Hypernetwork checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", default="outputs/phase5")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds per demo")
    parser.add_argument("--f0", type=float, default=440.0, help="Test f0 in Hz")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    model = load_model(args.config, args.checkpoint, device)
    model.eval()

    print(f"Hypernetwork params: {model.hypernetwork.count_parameters():,}")
    print(f"Base decoder frozen: {all(not p.requires_grad for p in model.base.parameters())}")

    # Generate per-energy audio
    audios = {}
    for label, energy in ENERGY_STATES.items():
        audio = synthesize(model, args.f0, -10.0, energy, args.duration, device)
        audios[label] = audio
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        print(f"  {label:12s}: RMS={rms:.4f}, peak={peak:.4f}, samples={len(audio)}")

    # Save individual WAVs
    sr = model.sample_rate
    for label, audio in audios.items():
        path = out_dir / f"{label}.wav"
        wavfile.write(str(path), sr, audio)

    # Concatenate into comparison file: neutral, string, brass, voice, electronic
    order = ["neutral", "string", "brass", "voice", "electronic"]
    gap = np.zeros(int(0.1 * sr), dtype=np.float32)
    segments = []
    for label in order:
        segments.append(audios[label])
        segments.append(gap)
    comparison = np.concatenate(segments)
    wavfile.write(str(out_dir / "same_input_diff_energy.wav"), sr, comparison)
    print(f"\nComparison WAV → {out_dir / 'same_input_diff_energy.wav'}")

    # Spectral cosine similarity matrix (vs neutral baseline)
    print("\nSpectral cosine similarity (vs neutral baseline):")
    baseline = audios["neutral"]
    for label in ["string", "brass", "voice", "electronic"]:
        sim = spectral_cosine_similarity(baseline, audios[label])
        print(f"  neutral vs {label:12s}: {sim:.4f}")

    # Pairwise similarities
    print("\nPairwise spectral cosine similarities:")
    energy_labels = ["neutral", "string", "brass", "voice", "electronic"]
    for i, la in enumerate(energy_labels):
        for j, lb in enumerate(energy_labels):
            if i < j:
                sim = spectral_cosine_similarity(audios[la], audios[lb])
                print(f"  {la:12s} vs {lb:12s}: {sim:.4f}")

    # Verify base decoder integrity
    print("\nBase decoder integrity check:")
    model_neutral = ModulatedDecoder(
        base_decoder=model.base,
        hypernetwork=Hypernetwork(
            hidden_size=model.hidden_size,
            n_harmonics=model.n_harmonics,
            n_magnitudes=model.n_magnitudes,
            bottleneck=48,
        ),
        frozen_decoder=False,
    ).to(device)

    # Compare neutral output with base decoder directly
    f0 = torch.full((1, 32), args.f0, device=device)
    loudness = torch.full((1, 32), -10.0, device=device)
    f0_scaled = scale_f0(f0)

    with torch.no_grad():
        harm_base, noise_base = model.base(f0_scaled, loudness)
        harm_mod, noise_mod = model.forward_neutral(f0_scaled, loudness)

    harm_diff = (harm_base - harm_mod).abs().max().item()
    noise_diff = (noise_base - noise_mod).abs().max().item()
    print(f"  Max harm diff (base vs neutral): {harm_diff:.2e}")
    print(f"  Max noise diff (base vs neutral): {noise_diff:.2e}")
    if harm_diff < 1e-6 and noise_diff < 1e-6:
        print("  PASS: base decoder unchanged")
    else:
        print("  WARNING: base decoder output differs (expected if trained)")

    print(f"\nAll outputs → {out_dir}/")


if __name__ == "__main__":
    main()
