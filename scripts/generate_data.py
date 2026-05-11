#!/usr/bin/env python3
"""
Generate synthetic flute-like training data using additive synthesis.

Creates monophonic audio clips with controlled harmonic structure,
natural envelopes, and known f0 from MIDI notes.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def flute_harmonic_amps(n_harmonics: int, velocity: float) -> np.ndarray:
    """
    Model flute-like harmonic amplitudes.

    Flute has strong fundamental, moderate 2nd-4th harmonics, rapid rolloff above.
    Higher velocity → brighter sound (more high harmonics).
    """
    amps = np.zeros(n_harmonics)
    for k in range(n_harmonics):
        h = k + 1  # harmonic number (1-indexed)
        # Base rolloff: 1/h with breath noise emphasis
        base = 1.0 / h
        # Velocity-dependent brightness: more high harmonics at high velocity
        brightness = 1.0 + velocity * (h - 1) * 0.3
        # Flute has weak even harmonics relative to odd at low velocity
        even_odd_ratio = 0.7 if h % 2 == 0 else 1.0
        amps[k] = base * min(brightness, 5.0) * even_odd_ratio
    # Normalize
    amps = amps / np.sqrt(np.sum(amps**2))
    return amps.astype(np.float32)


def synthesize_note(
    midi_note: int,
    velocity: float,
    duration_sec: float = 4.0,
    sample_rate: int = 16000,
    n_harmonics: int = 100,
) -> np.ndarray:
    """
    Synthesize a single note with additive synthesis and natural envelope.

    Args:
        midi_note: MIDI note number (40-84 for flute range)
        velocity: 0.0 to 1.0 (p, mf, f)
        duration_sec: total duration including release
        sample_rate: audio sample rate
        n_harmonics: number of harmonic partials

    Returns:
        audio: [num_samples] float32 array
    """
    f0_hz = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate

    # ADSR envelope
    attack = 0.05   # 50ms
    decay = 0.15    # 150ms
    sustain_level = 0.7
    release = 0.5   # 500ms
    sustain_start = attack + decay
    release_start = duration_sec - release

    env = np.ones(n_samples, dtype=np.float32)
    for i in range(n_samples):
        ti = t[i]
        if ti < attack:
            env[i] = ti / attack
        elif ti < sustain_start:
            env[i] = 1.0 - (1.0 - sustain_level) * (ti - attack) / decay
        elif ti < release_start:
            env[i] = sustain_level
        else:
            env[i] = sustain_level * max(0.0, 1.0 - (ti - release_start) / release)

    # Velocity scales amplitude and brightness
    base_amplitude = 0.3 + velocity * 0.7  # 0.3 to 1.0

    # Generate audio via additive synthesis
    harmonic_amps = flute_harmonic_amps(n_harmonics, velocity)
    audio = np.zeros(n_samples, dtype=np.float32)
    phase = np.zeros(n_harmonics, dtype=np.float32)

    for k in range(n_harmonics):
        h = k + 1
        freq = f0_hz * h
        if freq >= sample_rate / 2:
            break
        omega = 2.0 * np.pi * freq / sample_rate
        phase_k = np.random.uniform(0, 2 * np.pi)  # random initial phase
        audio += harmonic_amps[k] * np.sin(omega * np.arange(n_samples) + phase_k)

    # Apply envelope and velocity scaling
    audio = audio * env * base_amplitude

    # Add subtle breath noise during attack
    breath_noise = np.random.randn(n_samples).astype(np.float32) * 0.008
    breath_env = np.exp(-t / 0.3) * velocity  # decays over 300ms
    audio += breath_noise * breath_env

    # Add very subtle vibrato (5-6 Hz, shallow)
    vibrato_rate = 5.0 + velocity * 1.5
    vibrato_depth = 0.002 + velocity * 0.003
    vibrato = 1.0 + vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * t)
    audio = audio * vibrato

    # Normalize to prevent clipping
    peak = np.abs(audio).max()
    if peak > 0.95:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic flute training data")
    parser.add_argument("--output_dir", default="data/raw", help="Output directory")
    parser.add_argument("--midi_min", type=int, default=40, help="Lowest MIDI note (E2=40)")
    parser.add_argument("--midi_max", type=int, default=84, help="Highest MIDI note (C6=84)")
    parser.add_argument("--duration", type=float, default=4.0, help="Note duration in seconds")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    velocities = {"p": 0.25, "mf": 0.55, "f": 0.85}
    note_names = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

    manifest_path = out_dir / "manifest.csv"
    n_generated = 0

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "midi_note", "velocity", "instrument"])

        for midi in range(args.midi_min, args.midi_max + 1):
            octave = midi // 12 - 1
            note_name = f"{note_names[midi % 12]}{octave}"

            for vel_name, vel_val in velocities.items():
                filename = f"flute_{note_name}_{vel_name}.wav"
                audio = synthesize_note(
                    midi_note=midi,
                    velocity=vel_val,
                    duration_sec=args.duration,
                    sample_rate=args.sample_rate,
                )
                # Save as 16-bit WAV via scipy
                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(str(out_dir / filename), args.sample_rate, audio_int16)
                writer.writerow([filename, midi, vel_name, "flute"])
                n_generated += 1

    print(f"Generated {n_generated} clips ({args.midi_min}-{args.midi_max}, 3 velocities)")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
