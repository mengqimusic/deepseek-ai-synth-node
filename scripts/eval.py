#!/usr/bin/env python3
"""Evaluate RichParamModel: latency benchmark and spectrogram comparison."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from synth.nn.model import RichParamModel
from synth.eval.latency import measure_latency
from synth.eval.metrics import compute_spectrogram, compute_multi_scale_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDSP model")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--config", default="configs/phase10a.yaml", help="Config file")
    parser.add_argument("--data_dir", default="data/processed", help="Processed data dir")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = out_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)

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

    print("=" * 50)
    print("RichParamModel Evaluation")
    print("=" * 50)

    # Parameter count
    param_info = model.count_parameters()
    print(f"\nParameters: decoder={param_info['decoder']:,}  total={param_info['total']:,}")

    # Latency benchmark
    print(f"\n--- Latency Benchmark ({args.device}) ---")
    lat = measure_latency(model, args.device)
    print(f"  Decoder:       {lat['decoder_ms']:.3f} ms")
    print(f"  Harmonic Synth: {lat['harmonic_ms']:.3f} ms ({lat['harmonic_pct']:.0f}%)")
    print(f"  Noise Synth:    {lat['noise_ms']:.3f} ms")
    print(f"  Total per-frame: {lat['total_ms']:.3f} ms")
    print(f"  Frame duration:  {lat['frame_duration_ms']:.1f} ms")
    print(f"  Real-time factor: {lat['rtf']:.4f}")

    rtf_pass = lat["rtf"] < 0.5
    params_pass = param_info["decoder"] < 6_000_000
    print(f"\n  RTF < 0.5: {'PASS' if rtf_pass else 'FAIL'}")
    print(f"  Params < 6M: {'PASS' if params_pass else 'FAIL'}")

    # Spectrogram comparison on validation samples
    val_manifest = Path(args.data_dir) / "val_manifest.txt"
    if val_manifest.exists():
        with open(val_manifest) as f:
            val_names = [line.strip() for line in f if line.strip()]

        print(f"\n--- Spectrogram Comparison ({min(3, len(val_names))} samples) ---")

        for name in val_names[:3]:
            f0 = torch.from_numpy(np.load(str(Path(args.data_dir) / f"{name}_f0.npy"))).unsqueeze(0).to(args.device)
            loudness = torch.from_numpy(np.load(str(Path(args.data_dir) / f"{name}_loudness.npy"))).unsqueeze(0).to(args.device)
            target = torch.from_numpy(np.load(str(Path(args.data_dir) / f"{name}_audio.npy"))).unsqueeze(0).to(args.device)

            with torch.no_grad():
                pred = model(f0, loudness)

            # Per-scale losses
            losses = compute_multi_scale_loss(pred, target)
            loss_str = "  ".join(f"{k}:{v:.3f}" for k, v in losses.items())
            print(f"  {name}: {loss_str}")

            # Spectrogram plots
            pred_spec = compute_spectrogram(pred)
            target_spec = compute_spectrogram(target)

            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            for ax, data, title in [
                (axes[0], target_spec, f"{name} — Target"),
                (axes[1], pred_spec, f"{name} — Predicted"),
                (axes[2], target_spec - pred_spec, f"{name} — Difference"),
            ]:
                im = ax.imshow(data, aspect="auto", origin="lower",
                              cmap="magma" if "Diff" not in title else "RdBu")
                ax.set_title(title)
                plt.colorbar(im, ax=ax)

            plt.tight_layout()
            fig_path = spec_dir / f"{name}_spectrogram.png"
            plt.savefig(fig_path, dpi=100)
            plt.close(fig)
            print(f"    Plot → {fig_path}")

    print(f"\nTraining steps completed: {ckpt.get('step', 'unknown')}")
    print("Done.")


if __name__ == "__main__":
    main()
