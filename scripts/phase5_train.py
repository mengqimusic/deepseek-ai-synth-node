#!/usr/bin/env python3
"""
Train Phase 5 hypernetwork on multi-timbral data.

Freezes the base DDSPDecoder (Phase 4 checkpoint) and trains only the
Hypernetwork. The hypernetwork learns to map energy states to ΔW such
that the decoder produces timbre-appropriate output for each class.

Training signal: per-class reconstruction loss — the hypernetwork must
learn that energy_state[string] → ΔW that makes the decoder sound more
string-like, etc.

Usage:
    python scripts/phase5_train.py --device cpu     # or mps / cuda
    python scripts/phase5_train.py --device mps --resume checkpoints/phase5_step_001000.pt
"""

import argparse
import warnings
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message=".*resized.*")
warnings.filterwarnings("ignore", message=".*An output with one or more.*")

from synth.nn.model import DDSPModel
from synth.nn.hypernetwork import Hypernetwork
from synth.nn.modulated_decoder import ModulatedDecoder
from synth.train.losses import MultiScaleSpectralLoss
from synth.data.dataset import DDSPDataset, collate_variable_length
from synth.dsp.processors import scale_f0


ENERGY_MAP = {
    "string":     torch.tensor([0.8, 0.1, 0.3, 0.2]),
    "brass":      torch.tensor([0.2, 0.8, 0.1, 0.1]),
    "voice":      torch.tensor([0.1, 0.2, 0.8, 0.3]),
    "electronic": torch.tensor([0.3, 0.4, 0.2, 0.8]),
}


def build_model(config: dict, device: str) -> ModulatedDecoder:
    model_cfg = config["model"]
    data_cfg = config["data"]

    # Load base DDSPModel (Phase 4 checkpoint)
    base_model = DDSPModel(
        hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        sample_rate=data_cfg["sample_rate"],
        block_size=data_cfg["block_size"],
    )
    ckpt_path = model_cfg["base_checkpoint"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    base_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded base decoder from {ckpt_path} (step {ckpt['step']})")

    # Create hypernetwork
    hypernetwork = Hypernetwork(
        hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        bottleneck=model_cfg.get("bottleneck", 48),
        max_scale=model_cfg.get("max_scale", 0.12),
    )

    # Wrap in ModulatedDecoder
    modulated = ModulatedDecoder(
        base_decoder=base_model.decoder,
        hypernetwork=hypernetwork,
        frozen_decoder=True,
    )

    # Keep synth components for audio generation
    modulated.harmonic_synth = base_model.harmonic_synth
    modulated.noise_synth = base_model.noise_synth
    modulated.sample_rate = base_model.sample_rate
    modulated.block_size = base_model.block_size

    return modulated.to(device)


def train_step(
    model: ModulatedDecoder,
    batch: dict,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()

    f0 = batch["f0"].to(device)
    loudness = batch["loudness"].to(device)
    target = batch["audio"].to(device)
    classes = batch["timbre_class"]

    # Map class → energy state
    energy = torch.stack([ENERGY_MAP[c].to(device) for c in classes])  # [B, 4]

    optimizer.zero_grad()

    if scaler is not None:
        with torch.cuda.amp.autocast():
            loss = _forward_and_loss(model, f0, loudness, energy, target, loss_fn)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss = _forward_and_loss(model, f0, loudness, energy, target, loss_fn)
        loss.backward()
        optimizer.step()

    return loss.item()


def _forward_and_loss(model, f0, loudness, energy, target, loss_fn):
    f0_scaled = scale_f0(f0)
    harm_amps, noise_mags = model(f0_scaled, loudness, energy)
    harm_audio = model.harmonic_synth(harm_amps, f0)
    noise_audio = model.noise_synth(noise_mags)
    pred = harm_audio + noise_audio

    min_len = min(pred.shape[-1], target.shape[-1])
    assert min_len > 0, f"Zero-length output (T={f0.shape[-1]})"
    pred = pred[..., :min_len]
    target = target[..., :min_len]
    return loss_fn(pred, target)


@torch.no_grad()
def validate(model: ModulatedDecoder, val_loader: DataLoader, loss_fn: nn.Module,
             device: str, max_batches: int = 20) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    per_class_loss = {c: 0.0 for c in ENERGY_MAP}
    per_class_count = {c: 0 for c in ENERGY_MAP}
    n = 0

    for batch in val_loader:
        if n >= max_batches:
            break
        f0 = batch["f0"].to(device)
        loudness = batch["loudness"].to(device)
        target = batch["audio"].to(device)
        classes = batch["timbre_class"]
        energy = torch.stack([ENERGY_MAP[c].to(device) for c in classes])

        f0_scaled = scale_f0(f0)
        harm_amps, noise_mags = model(f0_scaled, loudness, energy)
        harm_audio = model.harmonic_synth(harm_amps, f0)
        noise_audio = model.noise_synth(noise_mags)
        pred = harm_audio + noise_audio

        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        # Per-sample losses for per-class breakdown
        for i, c in enumerate(classes):
            sample_loss = loss_fn(pred[i:i+1], target[i:i+1]).item()
            per_class_loss[c] += sample_loss
            per_class_count[c] += 1
            total_loss += sample_loss

        n += 1

    results = {"val_loss": total_loss / sum(per_class_count.values()) if n > 0 else 0.0}
    for c in ENERGY_MAP:
        if per_class_count[c] > 0:
            results[f"val_{c}"] = per_class_loss[c] / per_class_count[c]

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Phase 5 hypernetwork")
    parser.add_argument("--config", default="configs/phase5.yaml")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    train_cfg = config["training"]

    # Datasets (with class labels)
    train_dataset = DDSPDataset(data_cfg["data_dir"], split="train", return_class=True)
    val_dataset = DDSPDataset(data_cfg["data_dir"], split="val", return_class=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=2,
    )

    print(f"Train: {len(train_dataset)} clips, Val: {len(val_dataset)} clips")

    # Model
    model = build_model(config, args.device)
    param_info = model.count_parameters()
    print(f"Model: base_decoder={param_info['base_decoder']:,} (frozen), "
          f"hypernetwork={param_info['hypernetwork']:,} (trainable), "
          f"total={param_info['total']:,}")

    # Loss
    loss_fn = MultiScaleSpectralLoss(
        fft_sizes=train_cfg["fft_sizes"],
        hop_size=data_cfg["hop_size"],
    ).to(args.device)

    # Optimizer (only hypernetwork params)
    optimizer = torch.optim.Adam(
        model.hypernetwork.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(0.9, 0.999),
    )

    # AMP
    use_amp = args.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.hypernetwork.load_state_dict(ckpt["hypernetwork_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    # Training loop
    import time

    def cycle(dl):
        while True:
            for batch in dl:
                yield batch

    train_iter = cycle(train_loader)
    t0 = time.time()
    total_steps = train_cfg["total_steps"]
    ckpt_interval = train_cfg["ckpt_interval"]
    log_interval = train_cfg["log_interval"]

    for step in range(start_step + 1, start_step + total_steps + 1):
        batch = next(train_iter)
        loss = train_step(model, batch, loss_fn, optimizer, args.device, scaler)

        if step % log_interval == 0:
            val_results = validate(model, val_loader, loss_fn, args.device)
            elapsed = time.time() - t0
            steps_done = step - start_step
            rate = steps_done / elapsed if elapsed > 0 else 0
            parts = [f"val_loss={val_results['val_loss']:.4f}"]
            for c in ["string", "brass", "voice", "electronic"]:
                k = f"val_{c}"
                if k in val_results:
                    parts.append(f"{c[:4]}={val_results[k]:.3f}")
            print(
                f"[{step:6d}/{start_step + total_steps}] "
                f"train_loss={loss:.4f}  " + "  ".join(parts) +
                f"  rate={rate:.1f} steps/s",
                flush=True,
            )

        if step % ckpt_interval == 0:
            ckpt_path = f"checkpoints/phase5_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "hypernetwork_state_dict": model.hypernetwork.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"  checkpoint → {ckpt_path}")

    # Final checkpoint
    final_path = "checkpoints/phase5_final.pt"
    torch.save(
        {
            "step": start_step + total_steps,
            "hypernetwork_state_dict": model.hypernetwork.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_path,
    )
    print(f"Done. Final checkpoint → {final_path}")


if __name__ == "__main__":
    main()
