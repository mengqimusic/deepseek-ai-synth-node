#!/usr/bin/env python3
"""
Phase 3 offline render demos: multi-Voice independent development,
spectral competition, and phase transitions.

Generates WAV files for listening verification without requiring a sound card.
Deterministic: same inputs → same outputs (no random elements in pipeline).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synth.nn.model import DDSPModel
from synth.poly import PolyphonicSynth
from synth.voice import ENERGY_NAMES
from synth.dsp.processors import midi_to_hz


def make_synth(config_path="configs/phase1.yaml", checkpoint_path="checkpoints/phase1_final.pt", device="cpu"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]
    sample_rate = data_cfg["sample_rate"]
    block_size = data_cfg["block_size"]

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DDSPModel(
        hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        sample_rate=sample_rate,
        block_size=block_size,
        table_size=model_cfg["table_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    synth = PolyphonicSynth(
        decoder=model.decoder,
        harmonic_synth=model.harmonic_synth,
        noise_synth=model.noise_synth,
        num_voices=5,
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        sample_rate=sample_rate,
        block_size=block_size,
    )
    synth.to(device)
    synth.eval()

    return synth, sample_rate, block_size


def render_frames(synth, total_frames, block_size, sample_rate, callback=None):
    """
    Run synth.process_frame() for total_frames, collecting audio.
    callback(frame_idx) is called before each frame to script events.
    """
    frames = []
    for t in range(total_frames):
        if callback:
            callback(t, block_size, sample_rate)
        audio = synth.process_frame()
        frames.append(audio)

    audio = np.concatenate(frames)
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio / peak * 0.95
    return audio


# ---------------------------------------------------------------------------
# Demo 1: Independent Voice Development
# ---------------------------------------------------------------------------
def demo_independent_voices(synth, sample_rate, block_size, out_dir):
    """
    5 Voices each cultivated with a different energy direction.
    Each Voice plays a simple ascending pattern while receiving its
    designated energy type. After ~30s of accumulation, the voices
    diverge audibly through phase baselines and competition profiles.
    """
    synth.reset_all()
    total_duration = 36.0
    total_frames = int(total_duration * sample_rate / block_size)

    # Voice assignment: each voice gets a dedicated energy direction
    voice_direction = {
        0: "tension",
        1: "turbulence",
        2: "resonance",
        3: "memory",
        4: None,  # control
    }

    # Note sequence: staggered entries
    base_notes = [60, 64, 67, 72]  # C4, E4, G4, C5
    note_duration_frames = int(1.5 * sample_rate / block_size)
    cycle_frames = note_duration_frames * 4 + int(1.0 * sample_rate / block_size)

    def callback(t, bs, sr):
        sec = t * bs / sr
        cycle_pos = t % cycle_frames

        # Energy injection: ramp up over first 10s, then hold
        energy_level = min(1.0, sec / 10.0) if sec < 30.0 else max(0.0, 1.0 - (sec - 30.0) / 6.0)

        for vid, direction in voice_direction.items():
            if direction:
                synth.set_energy(vid, direction, energy_level)

        # Note triggering: each voice plays one note in the sequence
        note_idx = (t // note_duration_frames) % 4
        if cycle_pos % note_duration_frames == 0:
            # Release previous note for this voice slot
            for vid in range(5):
                active = synth.active_notes()
                for midi, v in list(active.items()):
                    if v == vid:
                        synth.note_off(midi)

            # Assign new note — voice plays its designated note
            note = base_notes[note_idx]
            synth.note_on(note, -10.0)

    audio = render_frames(synth, total_frames, block_size, sample_rate, callback)
    path = os.path.join(out_dir, "independent_voices.wav")
    sf.write(path, audio, sample_rate)
    print(f"  {path}  ({total_duration:.0f}s, rms={audio.std():.3f})")

    # Print final Voice states
    print("  Final Voice states:")
    for vid in range(5):
        st = synth.get_voice_state(vid)
        print(f"    Voice {vid}: accum={ {k: f'{v:.1f}' for k,v in st.energy_accumulation.items()} }  "
              f"phases={st.phase_count}  weight={st.competition_weight:.2f}  "
              f"wd={st.withdrawal_low:.2f}/{st.withdrawal_mid:.2f}/{st.withdrawal_high:.2f}  "
              f"dominant={st.dominant_direction}")


# ---------------------------------------------------------------------------
# Demo 2: Spectral Competition
# ---------------------------------------------------------------------------
def demo_spectral_competition(synth, sample_rate, block_size, out_dir):
    """
    3 Voices playing overlapping notes in the same frequency range.
    Demonstrates spectral competition scheduling with differentiated
    withdrawal styles.
    """
    synth.reset_all()

    # Pre-cultivate Voices to establish different competition profiles
    # Voice 0: tension-heavy → low withdrawal_mid (preserves harmonics)
    # Voice 1: turbulence-heavy → high withdrawal_mid (yields mids)
    # Voice 2: resonance-heavy → high withdrawal_high (yields highs)
    cultivation_frames = int(5.0 * sample_rate / block_size)  # 5s cultivation at level=1.0

    # Phase 0: cultivation (silent, energy-only, no notes)
    for t in range(cultivation_frames):
        synth.set_energy(0, "tension", 1.0)
        synth.set_energy(1, "turbulence", 1.0)
        synth.set_energy(2, "resonance", 1.0)
        synth.process_frame()

    # Clear energy for the actual performance
    for vid in range(3):
        for d in ENERGY_NAMES:
            synth.set_energy(vid, d, 0.0)

    print("  After cultivation:")
    for vid in range(3):
        st = synth.get_voice_state(vid)
        print(f"    Voice {vid}: accum={ {k: f'{v:.1f}' for k,v in st.energy_accumulation.items()} }  "
              f"weight={st.competition_weight:.2f}  "
              f"wd={st.withdrawal_low:.2f}/{st.withdrawal_mid:.2f}/{st.withdrawal_high:.2f}")

    # Phase 1: 3 voices playing the same note (dense competition)
    synth.all_notes_off()
    for vid in range(3):
        synth.note_on(67, -10.0)  # all play G4

    phase1_frames = int(6.0 * sample_rate / block_size)
    phase1_audio = render_frames(synth, phase1_frames, block_size, sample_rate)

    # Phase 2: spread out to different octaves (less competition)
    synth.all_notes_off()
    synth.note_on(55, -10.0)  # G3
    synth.note_on(67, -10.0)  # G4
    synth.note_on(79, -10.0)  # G5

    phase2_frames = int(6.0 * sample_rate / block_size)
    phase2_audio = render_frames(synth, phase2_frames, block_size, sample_rate)

    # Phase 3: competing again with tension energy injected to Voice 0
    synth.all_notes_off()
    for vid in range(3):
        synth.note_on(67, -10.0)

    phase3_frames = int(6.0 * sample_rate / block_size)
    def phase3_callback(t, bs, sr):
        synth.set_energy(0, "tension", 1.0)
    phase3_audio = render_frames(synth, phase3_frames, block_size, sample_rate, phase3_callback)

    audio = np.concatenate([phase1_audio, phase2_audio, phase3_audio])
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio / peak * 0.95

    total_duration = len(audio) / sample_rate
    path = os.path.join(out_dir, "spectral_competition.wav")
    sf.write(path, audio, sample_rate)
    print(f"  {path}  ({total_duration:.0f}s, rms={audio.std():.3f})")


# ---------------------------------------------------------------------------
# Demo 3: Phase Transition
# ---------------------------------------------------------------------------
def demo_phase_transition(synth, sample_rate, block_size, out_dir):
    """
    Single Voice accumulating tension energy until phase transition.
    Voice plays a sustained G4 while tension energy is continuously
    injected. The phase transition occurs at ~5s of accumulation.
    """
    synth.reset_all()
    total_duration = 12.0
    total_frames = int(total_duration * sample_rate / block_size)

    synth.note_on(67, -10.0)  # G4

    phase_triggered_frame = None

    def callback(t, bs, sr):
        nonlocal phase_triggered_frame
        synth.set_energy(0, "tension", 1.0)

        # Check if phase just triggered
        if phase_triggered_frame is None and synth.get_voice_state(0).phase_tension:
            phase_triggered_frame = t

    audio = render_frames(synth, total_frames, block_size, sample_rate, callback)

    phase_sec = phase_triggered_frame * block_size / sample_rate if phase_triggered_frame else -1
    print(f"  Phase transition frame: {phase_triggered_frame} ({phase_sec:.1f}s)")

    path = os.path.join(out_dir, "phase_transition.wav")
    sf.write(path, audio, sample_rate)
    print(f"  {path}  ({total_duration:.0f}s, rms={audio.std():.3f})")

    # Verify: after phase, tension baseline should be active
    st = synth.get_voice_state(0)
    print(f"  Final Voice 0: phase_tension={st.phase_tension}  "
          f"accum_tension={st.energy_accumulation['tension']:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 3 render demos")
    parser.add_argument("--checkpoint", default="checkpoints/phase1_final.pt")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="outputs/phase3")
    parser.add_argument("--demo", choices=["all", "independent", "competition", "transition"],
                        default="all")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(42)  # deterministic noise synth

    synth, sample_rate, block_size = make_synth(args.config, args.checkpoint, args.device)
    print(f"Phase 3 Render Demos → {args.out}/")
    print(f"  sample_rate={sample_rate}  block_size={block_size}  device={args.device}")

    if args.demo in ("all", "independent"):
        print("\n[1/3] Independent Voice Development")
        demo_independent_voices(synth, sample_rate, block_size, args.out)

    if args.demo in ("all", "competition"):
        print("\n[2/3] Spectral Competition")
        demo_spectral_competition(synth, sample_rate, block_size, args.out)

    if args.demo in ("all", "transition"):
        print("\n[3/3] Phase Transition")
        demo_phase_transition(synth, sample_rate, block_size, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
