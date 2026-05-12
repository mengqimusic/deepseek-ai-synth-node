#!/usr/bin/env python3
"""Render Phase 9 feature demos: inharmonicity, formant filter, expanded modulation.

Generates WAV files demonstrating each new feature in isolation and combined.
Uses VoiceModule to exercise the full pipeline (decoder → energy bias → synth).
"""

import torch, yaml, numpy as np, os, math
from synth.nn.model import DDSPModel
from synth.nn.hypernetwork import Hypernetwork
from synth.nn.modulated_decoder import ModulatedDecoder
from synth.voice import VoiceModule, ENERGY_NAMES
from synth.dsp.processors import midi_to_hz
import scipy.io.wavfile as _wav


def _write_wav(path, audio, sr):
    """Write float32 audio [-1, 1] as 16-bit WAV via scipy."""
    audio = np.clip(audio, -1.0, 1.0)
    _wav.write(path, sr, (audio * 32767.0).astype(np.int16))

# ---------------------------------------------------------------------------
# Config & checkpoint loading
# ---------------------------------------------------------------------------
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

# Try loading hypernetwork for richer timbre
hypernetwork = None
modulated_decoder = None
try:
    h_ckpt = torch.load("checkpoints/phase5_final.pt", map_location="cpu", weights_only=False)
    hypernetwork = Hypernetwork(
        input_dim=4, hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"], n_magnitudes=model_cfg["n_magnitudes"],
        bottleneck=48, max_scale=0.30,  # Phase 9: expanded modulation range
    )
    hypernetwork.load_state_dict(h_ckpt["hypernetwork_state_dict"])
    hypernetwork.eval()
    modulated_decoder = ModulatedDecoder(
        base_decoder=model.decoder, hypernetwork=hypernetwork, frozen_decoder=True,
    )
    modulated_decoder.eval()
    print("Loaded hypernetwork checkpoint (phase5_final.pt)")
except FileNotFoundError:
    print("No hypernetwork checkpoint found, using base decoder only")

# ---------------------------------------------------------------------------
# Create a single VoiceModule (voice_id=0)
# ---------------------------------------------------------------------------
voice = VoiceModule(
    voice_id=0,
    decoder=model.decoder,
    harmonic_synth=model.harmonic_synth,
    noise_synth=model.noise_synth,
    n_harmonics=model_cfg["n_harmonics"],
    n_magnitudes=model_cfg["n_magnitudes"],
    sample_rate=sample_rate,
    block_size=block_size,
    modulated_decoder=modulated_decoder,
)
voice.eval()

os.makedirs("outputs/phase9", exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: render a note with a per-frame energy callback
# ---------------------------------------------------------------------------
def render_note(
    midi_note: int,
    duration: float,
    loudness_db: float,
    energy_cb,
) -> np.ndarray:
    """
    Render a sustained note using VoiceModule.process_frame().

    Args:
        midi_note: MIDI note number
        duration: seconds
        loudness_db: loudness in dB
        energy_cb: callable(frame_idx, total_frames, time_sec) → dict[name, level]

    Returns:
        audio: [total_samples] float32 numpy array
    """
    f0_hz = midi_to_hz(midi_note)
    n_frames = int(duration * sample_rate / block_size)

    voice.reset()
    frames = []
    for t in range(n_frames):
        sec = t * block_size / sample_rate
        levels = energy_cb(t, n_frames, sec)
        audio, harm, noise = voice.process_frame(f0_hz, loudness_db, levels)
        frames.append(audio)

    audio = np.concatenate(frames)

    # Peak normalize
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


# ===================================================================
# Demo 1: Inharmonicity (tension → β)
# ===================================================================
print("Rendering Demo 1: Inharmonicity...")

def inharmonic_ramp(t, total, sec):
    """Ramp tension 0→1→0 to demonstrate inharmonicity sweep."""
    lvl = 0.0
    if sec < 0.5:
        lvl = 0.0
    elif sec < 2.5:
        lvl = (sec - 0.5) / 2.0  # 0→1 over 2s
    elif sec < 3.5:
        lvl = 1.0
    else:
        lvl = max(0.0, 1.0 - (sec - 3.5) / 0.5)
    return {"tension": lvl}

audio_inharm = render_note(midi_note=55, duration=4.0, loudness_db=-10.0, energy_cb=inharmonic_ramp)
_write_wav("outputs/phase9/01_inharmonicity_tension_sweep.wav", audio_inharm, sample_rate)
print("  -> outputs/phase9/01_inharmonicity_tension_sweep.wav")


# ===================================================================
# Demo 2: Formant filter (resonance → formant shaping)
# ===================================================================
print("Rendering Demo 2: Formant filter...")

def formant_ramp(t, total, sec):
    """Ramp resonance 0→1→0 to sweep formant frequencies."""
    lvl = 0.0
    if sec < 0.5:
        lvl = 0.0
    elif sec < 2.5:
        lvl = (sec - 0.5) / 2.0
    elif sec < 3.5:
        lvl = 1.0
    else:
        lvl = max(0.0, 1.0 - (sec - 3.5) / 0.5)
    return {"resonance": lvl}

audio_formant = render_note(midi_note=55, duration=4.0, loudness_db=-10.0, energy_cb=formant_ramp)
_write_wav("outputs/phase9/02_formant_resonance_sweep.wav", audio_formant, sample_rate)
print("  -> outputs/phase9/02_formant_resonance_sweep.wav")


# ===================================================================
# Demo 3: Formant steps (dark / neutral / bright)
# ===================================================================
print("Rendering Demo 3: Formant steps...")

def formant_steps(t, total, sec):
    """Three discrete formant positions."""
    if sec < 1.0:
        return {"resonance": 0.0}   # dark /u/
    elif sec < 2.0:
        return {"resonance": 0.5}   # neutral /a/
    elif sec < 3.0:
        return {"resonance": 1.0}   # bright /i/
    else:
        return {"resonance": 0.5}

audio_steps = render_note(midi_note=55, duration=4.0, loudness_db=-10.0, energy_cb=formant_steps)
_write_wav("outputs/phase9/03_formant_steps_dark_neutral_bright.wav", audio_steps, sample_rate)
print("  -> outputs/phase9/03_formant_steps_dark_neutral_bright.wav")


# ===================================================================
# Demo 4: Combined inharmonicity + formant (tension + resonance)
# ===================================================================
print("Rendering Demo 4: Combined inharmonicity + formant...")

def combined_ramp(t, total, sec):
    """Cross-fade between tension and resonance dominance."""
    lvls = {"tension": 0.0, "resonance": 0.0}
    if sec < 0.5:
        pass
    elif sec < 2.0:
        # Tension dominant (inharmonicity + sharpening)
        lvls["tension"] = (sec - 0.5) / 1.5
    elif sec < 3.0:
        # Both at max
        lvls["tension"] = 1.0
        lvls["resonance"] = 1.0
    elif sec < 4.5:
        # Resonance dominant (formant shaping)
        lvls["tension"] = max(0.0, 1.0 - (sec - 3.0) / 1.5)
        lvls["resonance"] = 1.0
    else:
        lvls["resonance"] = max(0.0, 1.0 - (sec - 4.5) / 0.5)
    return lvls

audio_combined = render_note(midi_note=55, duration=5.0, loudness_db=-10.0, energy_cb=combined_ramp)
_write_wav("outputs/phase9/04_combined_inharmonic_formant.wav", audio_combined, sample_rate)
print("  -> outputs/phase9/04_combined_inharmonic_formant.wav")


# ===================================================================
# Demo 5: Baseline (no energy, for A/B comparison)
# ===================================================================
print("Rendering Demo 5: Baseline (no energy)...")

def baseline(t, total, sec):
    return {}

audio_baseline = render_note(midi_note=55, duration=2.0, loudness_db=-10.0, energy_cb=baseline)
_write_wav("outputs/phase9/00_baseline_no_energy.wav", audio_baseline, sample_rate)
print("  -> outputs/phase9/00_baseline_no_energy.wav")


# ===================================================================
# Demo 6: Pitch sweep with inharmonicity (more audible at higher harmonics)
# ===================================================================
print("Rendering Demo 6: Pitch sweep + inharmonicity...")

def pitch_sweep_inharm(t, total, sec):
    """Constant high tension so inharmonicity is always on."""
    return {"tension": 0.8}

# G3 → C5 sweep with inharmonicity on
f0_start = midi_to_hz(55)  # G3
f0_end = midi_to_hz(72)    # C5
n_frames = int(4.0 * sample_rate / block_size)
voice.reset()
frames = []
for t in range(n_frames):
    sec = t * block_size / sample_rate
    frac = sec / 4.0
    f0_hz = f0_start + (f0_end - f0_start) * frac
    levels = {"tension": 0.8}
    audio, harm, noise = voice.process_frame(f0_hz, -10.0, levels)
    frames.append(audio)
audio_ps = np.concatenate(frames)
peak = np.abs(audio_ps).max()
if peak > 0.99:
    audio_ps = audio_ps / peak * 0.95
_write_wav("outputs/phase9/05_inharmonic_pitch_sweep.wav", audio_ps.astype(np.float32), sample_rate)
print("  -> outputs/phase9/05_inharmonic_pitch_sweep.wav")


# ===================================================================
# Summary
# ===================================================================
print()
print("=" * 60)
print("Phase 9 demos rendered to outputs/phase9/")
print()
print("  00_baseline_no_energy.wav          — reference: no energy injection")
print("  01_inharmonicity_tension_sweep.wav — tension 0→1→0, hear frequency stretching")
print("  02_formant_resonance_sweep.wav     — resonance 0→1→0, hear vowel sweep")
print("  03_formant_steps_dark_neutral_bright.wav — 3 discrete formant positions")
print("  04_combined_inharmonic_formant.wav — tension + resonance cross-fade")
print("  05_inharmonic_pitch_sweep.wav      — pitch sweep with inharmonicity on")
print()
print("Listening guide:")
print("  - 01 vs 00: tension now adds inharmonicity (higher partials stretch sharp)")
print("  - 02 vs 00: resonance now applies formant filter (dark↔bright vowel shift)")
print("  - 03: step through /u/ (dark) → /a/ (neutral) → /i/ (bright)")
print("=" * 60)
