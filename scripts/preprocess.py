#!/usr/bin/env python3
"""
Preprocess raw audio files into training-ready .npy features.

Expected input:
    data/raw/*.wav + data/raw/manifest.csv

manifest.csv format:
    filename,midi_note,velocity,instrument
    flute_C4_p.wav,60,p,flute
    ...
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from synth.data.preprocessing import process_file, create_manifest


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for DDSP training")
    parser.add_argument("--raw_dir", default="data/raw", help="Raw audio directory")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory")
    parser.add_argument("--manifest", default="data/raw/manifest.csv", help="CSV manifest")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--hop_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--pre_emphasis", type=float, default=0.95)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest
    entries = []
    with open(args.manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)

    print(f"Found {len(entries)} entries in manifest")

    # Process each file
    names = []
    for entry in entries:
        fname = entry["filename"]
        midi = int(entry["midi_note"])
        input_path = raw_dir / fname
        if not input_path.exists():
            print(f"  SKIP: {fname} not found")
            continue
        name = process_file(
            str(input_path),
            midi_note=midi,
            output_dir=str(out_dir),
            sample_rate=args.sample_rate,
            hop_size=args.hop_size,
            block_size=args.block_size,
            pre_emphasis_coef=args.pre_emphasis,
        )
        names.append(name)
        print(f"  OK: {name}")

    # Create train/val split (stratified by pitch)
    rng = np.random.RandomState(42)
    indices = np.arange(len(names))
    rng.shuffle(indices)
    split = int(len(names) * (1 - args.val_fraction))
    train_names = [names[i] for i in indices[:split]]
    val_names = [names[i] for i in indices[split:]]

    create_manifest(train_names, str(out_dir), "train")
    create_manifest(val_names, str(out_dir), "val")

    print(f"\nDone. Train: {len(train_names)} clips, Val: {len(val_names)} clips")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
