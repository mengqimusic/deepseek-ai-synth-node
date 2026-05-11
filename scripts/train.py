#!/usr/bin/env python3
"""Train a DDSP model on preprocessed data."""

import argparse
import warnings
import yaml
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message=".*resized.*")
warnings.filterwarnings("ignore", message=".*An output with one or more.*")

from synth.nn.model import DDSPModel
from synth.train.losses import MultiScaleSpectralLoss
from synth.train.trainer import Trainer
from synth.data.dataset import DDSPDataset, collate_variable_length


def main():
    parser = argparse.ArgumentParser(description="Train DDSP model")
    parser.add_argument("--config", default="configs/phase1.yaml", help="Config file")
    parser.add_argument("--data_dir", default="data/processed", help="Preprocessed data")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Datasets
    train_dataset = DDSPDataset(args.data_dir, split="train")
    val_dataset = DDSPDataset(args.data_dir, split="val")

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
    model = DDSPModel(
        hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        sample_rate=data_cfg["sample_rate"],
        block_size=data_cfg["block_size"],
        table_size=model_cfg["table_size"],
    )
    param_info = model.count_parameters()
    print(f"Model: decoder={param_info['decoder']:,} params, total={param_info['total']:,}")

    # Loss
    loss_fn = MultiScaleSpectralLoss(
        fft_sizes=train_cfg["fft_sizes"],
        hop_size=data_cfg["hop_size"],
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(0.9, 0.999),
    )

    # Resume if requested
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        use_amp=True,
        log_interval=train_cfg["log_interval"],
        ckpt_interval=train_cfg["ckpt_interval"],
    )

    remaining_steps = train_cfg["total_steps"] - start_step
    if remaining_steps <= 0:
        print("Already trained for required steps.")
        return

    print(f"Training for {remaining_steps} steps...")
    trainer.train(train_loader, val_loader, remaining_steps)


if __name__ == "__main__":
    main()
