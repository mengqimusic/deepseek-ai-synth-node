#!/usr/bin/env python3
"""Render isolated energy direction demos for listening tests."""
import torch, yaml, numpy as np, os, math
from synth.nn.model import DDSPModel
from synth.energy import EnergyBiasModule
from synth.dsp.processors import scale_f0, midi_to_hz
import soundfile as sf

with open("configs/phase1.yaml") as f:
    config = yaml.safe_load(f)
model_cfg, data_cfg = config["model"], config["data"]
sample_rate, block_size = data_cfg["sample_rate"], data_cfg["block_size"]

ckpt = torch.load("checkpoints/phase1_final.pt", map_location="cpu", weights_only=False)
model = DDSPModel(
    hidden_size=model_cfg["hidden_size"], n_harmonics=model_cfg["n_harmonics"],
    n_magnitudes=model_cfg["n_magnitudes"], sample_rate=sample_rate,
    block_size=block_size, table_size=model_cfg["table_size"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

energy = EnergyBiasModule(
    n_harmonics=model_cfg["n_harmonics"], n_magnitudes=model_cfg["n_magnitudes"],
    sample_rate=sample_rate, block_size=block_size,
)
energy.eval()

os.makedirs("outputs/phase2", exist_ok=True)

base_f0 = midi_to_hz(67)  # G4
loudness_db = -10.0

demos = [
    # (name, direction, use_pitch_sweep)
    ("baseline", None, False),
    ("tension", "tension", False),
    ("turbulence", "turbulence", False),
    ("resonance", "resonance", False),
    ("memory", "memory", True),   # memory needs pitch change to be audible
]

for name, direction, use_sweep in demos:
    duration = 4.0
    n_frames = int(duration * sample_rate / block_size)

    # Build f0 trajectory
    if use_sweep:
        # Pitch sweep up then down (G4 → C5 → G4)
        f0_start = base_f0
        f0_peak = midi_to_hz(72)  # C5
        t_arr = np.arange(n_frames) * block_size / sample_rate
        # 0-1s: hold G4, 1-2.5s: sweep to C5, 2.5-3.5s: sweep back to G4, 3.5-4s: hold
        f0_arr = np.full(n_frames, f0_start)
        for i, t in enumerate(t_arr):
            if 1.0 <= t < 2.5:
                frac = (t - 1.0) / 1.5
                f0_arr[i] = f0_start + (f0_peak - f0_start) * frac
            elif 2.5 <= t < 3.5:
                frac = (t - 2.5) / 1.0
                f0_arr[i] = f0_peak + (f0_start - f0_peak) * frac
        f0 = torch.from_numpy(f0_arr).float().unsqueeze(0)  # [1, n_frames]
    else:
        f0 = torch.full((1, n_frames), base_f0)

    loudness = torch.full((1, n_frames), loudness_db)

    with torch.no_grad():
        harm, noise = model.decoder(scale_f0(f0), loudness)

    energy.reset_state()

    # Build per-frame energy levels (ramp 0→1→0)
    levels_seq = []
    for t in range(n_frames):
        sec = t * block_size / sample_rate
        lvl = 0.0
        if direction is not None:
            if sec < 0.5:
                lvl = 0.0
            elif sec < 2.0:
                lvl = (sec - 0.5) / 1.5
            elif sec < 3.5:
                lvl = 1.0
            else:
                lvl = max(0.0, 1.0 - (sec - 3.5) / 0.5)
        levels_seq.append({direction: lvl} if direction else {})

    harm_frames, noise_frames = [], []
    for t in range(n_frames):
        lv = levels_seq[t]
        h_t, n_t = energy(harm[:, t:t+1, :], noise[:, t:t+1, :], lv)
        harm_frames.append(h_t)
        noise_frames.append(n_t)

    harm_b = torch.cat(harm_frames, dim=1)
    noise_b = torch.cat(noise_frames, dim=1)
    audio = (model.harmonic_synth(harm_b, f0) + model.noise_synth(noise_b)).squeeze(0).cpu().numpy().astype(np.float32)

    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio / peak * 0.95

    path = f"outputs/phase2/{name}.wav"
    sf.write(path, audio, sample_rate)
    print(f"  {path}  rms={audio.std():.3f}  peak={np.abs(audio).max():.3f}")

print("\nDone. Listen to outputs/phase2/")
print("  memory.wav uses a pitch sweep — listen for old texture blending during the sweep")
