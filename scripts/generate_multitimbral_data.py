#!/usr/bin/env python3
"""
Generate multi-timbral training data for DDSP decoder.

Four timbre classes with randomized synthesis parameters:
  - string:  slow decay, rich even harmonics, bow noise
  - brass:   odd-harmonic dominant, fast attack, brightness-velocity coupling
  - voice:   formant structure, vowel switching, f0 perturbation
  - electronic: inharmonicity, sharp transients, sub-bass emphasis

Each class generates 135 clips (45 MIDI notes x 3 velocities) covering E2-C6.
Parameter randomization creates "intermediate states" between archetypes.
Deterministic: same base_seed -> identical output.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
VELOCITY_LAYERS = {"p": 0.25, "mf": 0.55, "f": 0.85}

# Approximate vowel formants: list of (center_hz, bandwidth_hz, gain)
VOWEL_FORMANTS = {
    "a": [(700, 90, 1.0), (1200, 80, 0.7), (2500, 100, 0.4)],
    "e": [(400, 70, 1.0), (2000, 70, 0.8), (2800, 100, 0.3)],
    "i": [(250, 60, 1.0), (2200, 60, 0.9), (3000, 80, 0.3)],
    "o": [(450, 80, 1.0), (800, 80, 0.7), (2600, 100, 0.3)],
    "u": [(300, 70, 1.0), (600, 80, 0.4), (2400, 100, 0.2)],
}
VOWEL_NAMES = list(VOWEL_FORMANTS.keys())


# ---------------------------------------------------------------------------
# Harmonic amplitude models
# ---------------------------------------------------------------------------

def string_harmonic_amps(n_harmonics, velocity, params):
    """String-like: slower rolloff (1/h^alpha, alpha<1), rich even harmonics."""
    h = np.arange(1, n_harmonics + 1, dtype=np.float32)
    decay_exp = params["harmonic_decay"]
    even_boost = params["even_odd_ratio"]

    amps = 1.0 / (h ** decay_exp)
    even_mask = (h % 2 == 0).astype(np.float32)
    amps *= 1.0 + (even_boost - 1.0) * even_mask
    brightness = 1.0 + velocity * h * 0.2
    amps *= np.minimum(brightness, 4.0)

    norm = np.sqrt(np.sum(amps ** 2))
    if norm > 0:
        amps /= norm
    return amps.astype(np.float32)


def brass_harmonic_amps(n_harmonics, velocity, params):
    """Brass-like: odd-harmonic dominant, strong velocity-brightness coupling."""
    h = np.arange(1, n_harmonics + 1, dtype=np.float32)
    decay_exp = params["harmonic_decay"]
    odd_ratio = params["even_odd_ratio"]  # <1 means even harmonics suppressed
    brightness_coupling = params["brightness_velocity_coupling"]

    amps = 1.0 / (h ** decay_exp)
    even_mask = (h % 2 == 0).astype(np.float32)
    amps *= 1.0 - (1.0 - odd_ratio) * even_mask
    brightness = 1.0 + velocity * brightness_coupling * h * 0.6
    amps *= np.minimum(brightness, 10.0)

    norm = np.sqrt(np.sum(amps ** 2))
    if norm > 0:
        amps /= norm
    return amps.astype(np.float32)


def voice_harmonic_amps(n_harmonics, velocity, params, f0_hz):
    """Voice-like: base 1/h rolloff shaped by formant spectral envelope."""
    h = np.arange(1, n_harmonics + 1, dtype=np.float32)
    freqs = f0_hz * h
    amps = 1.0 / h

    formant_env = np.ones(n_harmonics, dtype=np.float32)
    for f_center, bw, gain in params["formants"]:
        formant_env += gain * np.exp(-0.5 * ((freqs - f_center) / bw) ** 2)

    amps *= formant_env
    brightness = 1.0 + velocity * h * 0.15
    amps *= np.minimum(brightness, 3.0)

    norm = np.sqrt(np.sum(amps ** 2))
    if norm > 0:
        amps /= norm
    return amps.astype(np.float32)


def electronic_harmonic_amps(n_harmonics, velocity, params):
    """Electronic: variable rolloff, sub-bass boost, optional spectral notches."""
    h = np.arange(1, n_harmonics + 1, dtype=np.float32)
    decay_exp = params["harmonic_decay"]
    sub_bass_boost = params.get("sub_bass_boost", 1.0)

    amps = 1.0 / (h ** decay_exp)
    amps[0] *= sub_bass_boost

    if params.get("spectral_notch", False):
        nc = params.get("notch_center", 5)
        nw = params.get("notch_width", 3)
        lo = max(1, nc - nw)
        hi = min(n_harmonics, nc + nw)
        amps[lo - 1 : hi] *= 0.15

    norm = np.sqrt(np.sum(amps ** 2))
    if norm > 0:
        amps /= norm
    return amps.astype(np.float32)


# ---------------------------------------------------------------------------
# Frequency computation
# ---------------------------------------------------------------------------

def midi_to_f0(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def compute_partial_freqs(f0_hz, n_harmonics, inharmonicity_beta=0.0):
    """Partial frequencies with optional inharmonicity (string stiffness model)."""
    h = np.arange(1, n_harmonics + 1, dtype=np.float32)
    if inharmonicity_beta == 0.0:
        return f0_hz * h
    return f0_hz * h * np.sqrt(1.0 + inharmonicity_beta * (h ** 2))


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def adsr_envelope(n_samples, attack, decay, sustain_level, release,
                  duration_sec, sample_rate):
    """Generate ADSR amplitude envelope (vectorized)."""
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    env = np.ones(n_samples, dtype=np.float32)
    sustain_start = attack + decay
    release_start = duration_sec - release

    # Attack phase
    attack_mask = t < attack
    if attack > 0:
        env[attack_mask] = t[attack_mask] / attack
    else:
        env[attack_mask] = 1.0

    # Decay phase
    decay_mask = (t >= attack) & (t < sustain_start)
    if decay > 0:
        env[decay_mask] = 1.0 - (1.0 - sustain_level) * (t[decay_mask] - attack) / decay

    # Sustain phase
    sustain_mask = (t >= sustain_start) & (t < release_start)
    env[sustain_mask] = sustain_level

    # Release phase
    release_mask = t >= release_start
    if release > 0:
        rel_progress = (t[release_mask] - release_start) / release
        env[release_mask] = sustain_level * np.maximum(0.0, 1.0 - rel_progress)

    return env


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

def _one_pole_lpf(signal, cutoff, sample_rate):
    """Single-pole lowpass filter."""
    alpha = np.exp(-2.0 * np.pi * cutoff / sample_rate)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * signal[i]
    return out


def _one_pole_hpf(signal, cutoff, sample_rate):
    """Single-pole highpass filter (bilinear transform)."""
    K = np.pi * cutoff / sample_rate
    a1 = (1.0 - K) / (1.0 + K)
    b0 = 1.0 / (1.0 + K)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = a1 * out[i - 1] + b0 * signal[i] - b0 * signal[i - 1]
    return out


def generate_bow_noise(n_samples, level, sample_rate, rng):
    """Bow noise: lowpass-filtered noise for bowed-string character."""
    raw = rng.randn(n_samples).astype(np.float32)
    filtered = _one_pole_lpf(raw, 2000.0, sample_rate)
    return filtered * level


def generate_breath_noise(n_samples, level, sample_rate, rng):
    """Breath noise: highpass-filtered noise (envelope applied by caller)."""
    raw = rng.randn(n_samples).astype(np.float32)
    filtered = _one_pole_hpf(raw, 1000.0, sample_rate)
    return filtered * level


def generate_aspiration_noise(n_samples, level, sample_rate, rng):
    """Aspiration noise: soft bandpass-like filtered noise for voice."""
    raw = rng.randn(n_samples).astype(np.float32)
    lp = _one_pole_lpf(raw, 3000.0, sample_rate)
    hp = _one_pole_hpf(lp, 200.0, sample_rate)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    mod = 1.0 + 0.3 * np.sin(2.0 * np.pi * 3.0 * t)
    return hp * level * mod


def generate_electronic_noise(n_samples, level, sample_rate, rng, noise_color="white"):
    """Electronic noise: white or approximate pink."""
    raw = rng.randn(n_samples).astype(np.float32)
    if noise_color == "pink":
        return _one_pole_lpf(raw, 500.0, sample_rate) * level
    return raw * level


# ---------------------------------------------------------------------------
# Vibrato
# ---------------------------------------------------------------------------

def apply_vibrato(audio, n_samples, rate, depth, sample_rate, rng):
    """Amplitude vibrato via sinusoidal modulation."""
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    phase = rng.uniform(0, 2.0 * np.pi)
    mod = 1.0 + depth * np.sin(2.0 * np.pi * rate * t + phase)
    return audio * mod.astype(np.float32)


# ---------------------------------------------------------------------------
# Parameter randomization
# ---------------------------------------------------------------------------

def randomize_params(rng, timbre_class):
    """Generate randomized synthesis parameters for a given timbre class.

    Each call draws from class-specific ranges, creating intra-class variation
    that spans "intermediate states" between pure archetypes.
    """

    if timbre_class == "string":
        return {
            "timbre_class": "string",
            "harmonic_decay": rng.uniform(0.5, 0.9),
            "even_odd_ratio": rng.uniform(1.0, 1.8),
            "attack": rng.uniform(0.03, 0.12),
            "decay": rng.uniform(0.2, 0.5),
            "sustain_level": rng.uniform(0.6, 0.9),
            "release": rng.uniform(0.3, 0.8),
            "vibrato_rate": rng.uniform(4.0, 7.0),
            "vibrato_depth": rng.uniform(0.003, 0.012),
            "noise_type": "bow",
            "noise_level": rng.uniform(0.005, 0.02),
            "inharmonicity_beta": 0.0,
        }

    elif timbre_class == "brass":
        return {
            "timbre_class": "brass",
            "harmonic_decay": rng.uniform(0.8, 1.2),
            "even_odd_ratio": rng.uniform(0.2, 0.5),
            "brightness_velocity_coupling": rng.uniform(0.5, 1.0),
            "attack": rng.uniform(0.01, 0.04),
            "decay": rng.uniform(0.08, 0.2),
            "sustain_level": rng.uniform(0.5, 0.8),
            "release": rng.uniform(0.1, 0.3),
            "vibrato_rate": rng.uniform(4.0, 8.0),
            "vibrato_depth": rng.uniform(0.002, 0.008),
            "noise_type": "breath",
            "noise_level": rng.uniform(0.003, 0.015),
            "inharmonicity_beta": 0.0,
        }

    elif timbre_class == "voice":
        vowel_a = VOWEL_NAMES[rng.randint(0, len(VOWEL_NAMES))]
        vowel_b = VOWEL_NAMES[rng.randint(0, len(VOWEL_NAMES))]
        # Slightly jitter formant positions for natural variation
        f1_jitter = rng.uniform(-30, 30)
        f2_jitter = rng.uniform(-60, 60)
        formants = []
        for fc, bw, gain in VOWEL_FORMANTS[vowel_a]:
            jitter = f1_jitter if fc < 1000 else f2_jitter
            formants.append((
                fc + jitter,
                bw * rng.uniform(0.8, 1.2),
                gain * rng.uniform(0.8, 1.2),
            ))

        return {
            "timbre_class": "voice",
            "formants": formants,
            "vowel_start": vowel_a,
            "vowel_end": vowel_b,
            "attack": rng.uniform(0.02, 0.08),
            "decay": rng.uniform(0.1, 0.3),
            "sustain_level": rng.uniform(0.5, 0.8),
            "release": rng.uniform(0.2, 0.5),
            "vibrato_rate": rng.uniform(5.0, 7.0),
            "vibrato_depth": rng.uniform(0.005, 0.015),
            "jitter_amount": rng.uniform(0.001, 0.004),
            "noise_type": "aspiration",
            "noise_level": rng.uniform(0.002, 0.01),
            "inharmonicity_beta": 0.0,
        }

    elif timbre_class == "electronic":
        has_notch = rng.uniform() < 0.4
        return {
            "timbre_class": "electronic",
            "harmonic_decay": rng.uniform(0.3, 1.5),
            "sub_bass_boost": rng.uniform(1.0, 3.0),
            "spectral_notch": has_notch,
            "notch_center": int(rng.randint(3, 12)) if has_notch else 5,
            "notch_width": int(rng.randint(1, 5)) if has_notch else 2,
            "inharmonicity_beta": rng.uniform(0.0, 0.003),
            "attack": rng.uniform(0.0, 0.005),
            "decay": rng.uniform(0.05, 0.5),
            "sustain_level": rng.uniform(0.3, 1.0),
            "release": rng.uniform(0.05, 1.0),
            "vibrato_rate": rng.uniform(0.0, 10.0),
            "vibrato_depth": rng.uniform(0.0, 0.03),
            "noise_type": "white" if rng.uniform() < 0.6 else "pink",
            "noise_level": rng.uniform(0.0, 0.03),
        }

    else:
        raise ValueError(f"Unknown timbre class: {timbre_class}")


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _additive_synth(f0_hz, harm_amps, partial_freqs, n_samples, sample_rate, rng):
    """Core additive synthesis: sum of harmonically-related sinusoids."""
    audio = np.zeros(n_samples, dtype=np.float32)
    n_harmonics = len(harm_amps)
    for k in range(n_harmonics):
        freq = partial_freqs[k]
        if freq >= sample_rate / 2:
            break
        omega = 2.0 * np.pi * freq / sample_rate
        phase = rng.uniform(0, 2.0 * np.pi)
        audio += harm_amps[k] * np.sin(omega * np.arange(n_samples) + phase)
    return audio


def synthesize_note(midi_note, velocity, params, duration_sec=4.0,
                    sample_rate=16000, n_harmonics=100, rng=None):
    """Synthesize a note with given timbre parameters.

    Returns audio: [n_samples] float32
    """
    if rng is None:
        rng = np.random.RandomState(42)

    f0_hz = midi_to_f0(midi_note)
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    timbre_class = params["timbre_class"]

    # --- Envelope ---
    env = adsr_envelope(n_samples, params["attack"], params["decay"],
                        params["sustain_level"], params["release"],
                        duration_sec, sample_rate)

    # --- Harmonic amplitudes ---
    _DISPATCH = {
        "string": string_harmonic_amps,
        "brass": brass_harmonic_amps,
        "electronic": electronic_harmonic_amps,
    }
    if timbre_class in _DISPATCH:
        harm_amps = _DISPATCH[timbre_class](n_harmonics, velocity, params)
    elif timbre_class == "voice":
        harm_amps = voice_harmonic_amps(n_harmonics, velocity, params, f0_hz)
    else:
        raise ValueError(f"Unknown timbre class: {timbre_class}")

    # --- Partial frequencies ---
    beta = params.get("inharmonicity_beta", 0.0)
    partial_freqs = compute_partial_freqs(f0_hz, n_harmonics, beta)

    # --- Additive synthesis ---
    audio = _additive_synth(f0_hz, harm_amps, partial_freqs, n_samples, sample_rate, rng)

    # --- Jitter (voice only) ---
    if timbre_class == "voice" and params.get("jitter_amount", 0) > 0:
        jitter_raw = rng.randn(n_samples).astype(np.float32)
        jitter_smooth = _one_pole_lpf(jitter_raw, 12.0, sample_rate)
        # Subtle amplitude modulation from jitter
        audio *= (1.0 + jitter_smooth * params["jitter_amount"] * 3.0)

    # --- Envelope + velocity scaling ---
    base_amplitude = 0.3 + velocity * 0.7
    audio = audio * env * base_amplitude

    # --- Noise ---
    noise_type = params.get("noise_type", "breath")
    noise_level = params.get("noise_level", 0.005)

    if noise_type == "bow":
        noise = generate_bow_noise(n_samples, noise_level, sample_rate, rng)
        audio += noise * env
    elif noise_type == "breath":
        noise = generate_breath_noise(n_samples, noise_level, sample_rate, rng)
        breath_env = np.exp(-t / 0.3) * velocity
        audio += noise * breath_env
    elif noise_type == "aspiration":
        noise = generate_aspiration_noise(n_samples, noise_level, sample_rate, rng)
        audio += noise * env * velocity
    elif noise_type in ("white", "pink"):
        noise = generate_electronic_noise(n_samples, noise_level, sample_rate, rng, noise_type)
        audio += noise * env

    # --- Vibrato ---
    vib_rate = params.get("vibrato_rate", 5.0)
    vib_depth = params.get("vibrato_depth", 0.005)
    if vib_depth > 0 and vib_rate > 0:
        audio = apply_vibrato(audio, n_samples, vib_rate, vib_depth, sample_rate, rng)

    # --- Vowel crossfade (voice only) ---
    if timbre_class == "voice":
        vowel_start = params.get("vowel_start", "a")
        vowel_end = params.get("vowel_end", "a")
        if vowel_start != vowel_end:
            # Build end-vowel formants with separate jitter
            f1_jitter = rng.uniform(-30, 30)
            f2_jitter = rng.uniform(-60, 60)
            end_formants = []
            for fc, bw, gain in VOWEL_FORMANTS[vowel_end]:
                jitter = f1_jitter if fc < 1000 else f2_jitter
                end_formants.append((
                    fc + jitter,
                    bw * rng.uniform(0.8, 1.2),
                    gain * rng.uniform(0.8, 1.2),
                ))

            end_params = dict(params)
            end_params["formants"] = end_formants
            end_params["jitter_amount"] = params.get("jitter_amount", 0)

            end_amps = voice_harmonic_amps(n_harmonics, velocity, end_params, f0_hz)
            audio_end = _additive_synth(f0_hz, end_amps, partial_freqs,
                                        n_samples, sample_rate, rng)
            audio_end = audio_end * env * base_amplitude

            if vib_depth > 0 and vib_rate > 0:
                audio_end = apply_vibrato(audio_end, n_samples, vib_rate,
                                          vib_depth, sample_rate, rng)

            # Crossfade over 0.8-1.5s starting at 0.5-1.5s
            xf_start = rng.uniform(0.5, 1.5)
            xf_dur = rng.uniform(0.8, 1.5)
            xf = np.clip((t - xf_start) / xf_dur, 0.0, 1.0).astype(np.float32)
            audio = audio * (1.0 - xf) + audio_end * xf

    # --- Normalize ---
    peak = np.abs(audio).max()
    if peak > 0.95:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-timbral synthetic training data")
    parser.add_argument("--output_dir", default="data/multitimbral_raw",
                        help="Output directory for WAV files")
    parser.add_argument("--midi_min", type=int, default=40,
                        help="Lowest MIDI note (E2=40)")
    parser.add_argument("--midi_max", type=int, default=84,
                        help="Highest MIDI note (C6=84)")
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Note duration in seconds")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base random seed for reproducibility")
    parser.add_argument("--classes", default="string,brass,voice,electronic",
                        help="Comma-separated timbre classes to generate")
    args = parser.parse_args()

    timbre_classes = [c.strip() for c in args.classes.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    velocities = VELOCITY_LAYERS
    clip_index = 0
    n_generated = 0

    manifest_path = out_dir / "manifest.csv"
    n_per_class = (args.midi_max - args.midi_min + 1) * len(velocities)

    print(f"MIDI range: {args.midi_min}-{args.midi_max} ({n_per_class} clips/class)")
    print(f"Timbre classes: {timbre_classes}")
    print(f"Base seed: {args.base_seed}")
    print(f"Output: {out_dir}")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "midi_note", "velocity", "instrument"])

        for timbre_class in timbre_classes:
            class_start = n_generated
            print(f"\n  {timbre_class} ...", end=" ", flush=True)

            for midi in range(args.midi_min, args.midi_max + 1):
                octave = midi // 12 - 1
                note_name = f"{NOTE_NAMES[midi % 12]}{octave}"

                for vel_name, vel_val in velocities.items():
                    seed = args.base_seed + clip_index
                    rng = np.random.RandomState(seed)

                    params = randomize_params(rng, timbre_class)
                    audio = synthesize_note(
                        midi_note=midi,
                        velocity=vel_val,
                        params=params,
                        duration_sec=args.duration,
                        sample_rate=args.sample_rate,
                        rng=rng,
                    )

                    filename = f"{timbre_class}_{note_name}_{vel_name}.wav"
                    audio_int16 = (audio * 32767).astype(np.int16)
                    wavfile.write(str(out_dir / filename), args.sample_rate, audio_int16)
                    writer.writerow([filename, midi, vel_name, timbre_class])

                    n_generated += 1
                    clip_index += 1

            print(f"{n_generated - class_start} clips")

    print(f"\nTotal: {n_generated} clips")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
