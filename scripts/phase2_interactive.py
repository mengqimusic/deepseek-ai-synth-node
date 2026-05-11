#!/usr/bin/env python3
"""
Phase 2 interactive test: real-time energy direction validation.

Hold/toggle keys 1-4 to inject energy into the DDSP model and hear
the perceptual signature of each direction in real time.

Keys:
  1 — 张 (Tension):  harmonic sharpening
  2 — 扰 (Turbulence): sideband splitting + noise texture
  3 — 吟 (Resonance):  harmonic neighbor coupling dynamics
  4 — 忆 (Memory):     historical harmonic blending
  ↑/↓ — adjust pitch ±1 semitone
  ←/→ — adjust loudness ±3 dB
  q — quit

Requires: sounddevice (pip install sounddevice)
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synth.nn.model import DDSPModel
from synth.dsp.processors import scale_f0, midi_to_hz
from synth.energy import EnergyBiasModule

# ---------------------------------------------------------------------------
# Shared state between curses thread and audio callback
# ---------------------------------------------------------------------------
ENERGY_NAMES = ["tension", "turbulence", "resonance", "memory"]
ENERGY_LABELS = ["张", "扰", "吟", "忆"]

_energy_lock = threading.Lock()
_energy_toggle = {k: 0.0 for k in ENERGY_NAMES}   # user intent: set ONLY by key press, never smoothed
_energy_levels = {k: 0.0 for k in ENERGY_NAMES}   # smoothed output: read by audio callback and TUI
_running = True
_current_midi = 67  # G4
_current_loudness = -10.0


class RealtimeSynth:
    """Per-frame DDSP inference with GRU state tracking and energy biases."""

    def __init__(self, model: DDSPModel, energy: EnergyBiasModule):
        self.model = model
        self.energy = energy
        self.gru_hidden = None  # GRU state [1, 1, hidden]
        self.block_size = model.block_size
        self.device = next(model.parameters()).device

    def process_frame(
        self, f0_hz: float, loudness_db: float, levels: dict[str, float]
    ) -> np.ndarray:
        """Generate one frame of audio [block_size]."""
        with torch.no_grad():
            f0 = torch.tensor([[f0_hz]], device=self.device)
            loudness = torch.tensor([[loudness_db]], device=self.device)
            f0_scaled = scale_f0(f0)

            x = torch.stack([f0_scaled, loudness], dim=-1)  # [1, 1, 2]
            x = self.model.decoder.pre_mlp(x)
            x, h = self.model.decoder.gru(x, self.gru_hidden)
            self.gru_hidden = h.detach()
            x = self.model.decoder.post_mlp(x)
            harm = torch.sigmoid(self.model.decoder.harm_head(x))
            noise = torch.sigmoid(self.model.decoder.noise_head(x))

            harm, noise = self.energy(harm, noise, levels)
            audio = self.model.harmonic_synth(harm, f0) + self.model.noise_synth(noise)
            return audio.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    def reset(self):
        self.gru_hidden = None
        self.energy.reset_state()


def audio_callback_factory(synth: RealtimeSynth, block_size: int):
    """Return a sounddevice callback that pulls from RealtimeSynth."""

    def callback(outdata, frames, _time, status):
        if status:
            print(f"audio status: {status}", file=sys.stderr)

        with _energy_lock:
            levels = dict(_energy_levels)
        f0 = midi_to_hz(_current_midi)
        loudness = _current_loudness

        offset = 0
        remaining = frames
        while remaining > 0:
            chunk = synth.process_frame(f0, loudness, levels)
            n = min(block_size, remaining)
            outdata[offset : offset + n, 0] = chunk[:n]
            offset += n
            remaining -= n

    return callback


# ---------------------------------------------------------------------------
# Curses TUI
# ---------------------------------------------------------------------------
def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def run_interactive(synth: RealtimeSynth, block_size: int, sample_rate: int):
    import curses

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)  # non-blocking getch

    # Smoothing state (main thread)
    smooth = {k: 0.0 for k in ENERGY_NAMES}
    attack_alpha = 1.0 - math.exp(-0.004 / 0.030)  # 30ms attack
    release_alpha = 1.0 - math.exp(-0.004 / 0.150)  # 150ms release

    global _current_midi, _current_loudness

    stream = sd.OutputStream(
        samplerate=sample_rate,
        blocksize=block_size * 8,  # 512 samples = 32ms buffer
        channels=1,
        callback=audio_callback_factory(synth, block_size),
        dtype=np.float32,
    )

    try:
        stream.start()
        last_tui = 0.0

        while _running:
            key = stdscr.getch()
            now = time.time()

            # — Keyboard dispatch —
            if key == ord("1"):
                _energy_toggle["tension"] = 1.0 if _energy_levels["tension"] < 0.5 else 0.0
            elif key == ord("2"):
                _energy_toggle["turbulence"] = 1.0 if _energy_levels["turbulence"] < 0.5 else 0.0
            elif key == ord("3"):
                _energy_toggle["resonance"] = 1.0 if _energy_levels["resonance"] < 0.5 else 0.0
            elif key == ord("4"):
                _energy_toggle["memory"] = 1.0 if _energy_levels["memory"] < 0.5 else 0.0
            elif key == ord("q"):
                break
            elif key == curses.KEY_UP:
                _current_midi = min(96, _current_midi + 1)
                synth.reset()
            elif key == curses.KEY_DOWN:
                _current_midi = max(21, _current_midi - 1)
                synth.reset()
            elif key == curses.KEY_RIGHT:
                _current_loudness = min(0.0, _current_loudness + 3.0)
            elif key == curses.KEY_LEFT:
                _current_loudness = max(-40.0, _current_loudness - 3.0)

            # — Energy smoothing (attack / release) —
            for k in ENERGY_NAMES:
                target = _energy_toggle[k]
                alpha = attack_alpha if target > smooth[k] else release_alpha
                smooth[k] += alpha * (target - smooth[k])

            with _energy_lock:
                _energy_levels.update(smooth)

            # — TUI rendering (throttled to ~20 Hz) —
            if now - last_tui > 0.05:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                note_name = _midi_note_name(_current_midi)

                title = " Phase 2 — Energy Direction Test "
                stdscr.addstr(0, max(0, (w - len(title)) // 2), title, curses.A_REVERSE)
                stdscr.addstr(2, 2, f"Note: {note_name} ({midi_to_hz(_current_midi):.0f} Hz), {_current_loudness:+.0f} dB")
                stdscr.addstr(3, 2, "─" * 50)

                for i, k in enumerate(ENERGY_NAMES):
                    y = 4 + i
                    active = "●" if _energy_levels[k] > 0.5 else "○"
                    label = ENERGY_LABELS[i]
                    bar = _bar(smooth[k])
                    stdscr.addstr(y, 2, f"{active} {i+1} {label} [{bar}] {smooth[k]:.2f}")

                stdscr.addstr(9, 2, "─" * 50)
                stdscr.addstr(10, 2, "1-4: toggle energy  ↑↓: pitch  ←→: loudness  q: quit")

                if h > 12:
                    hints = [
                        "张 — harmonic sharpening (spectral peaks tighten)",
                        "扰 — sideband splitting + noise texture (clean→grainy)",
                        "吟 — neighbor coupling dynamics (static→breathing)",
                        "忆 — delay-line snapshot blending (change pitch to hear echo)",
                    ]
                    for i, hint in enumerate(hints):
                        if 12 + i < h - 1:
                            stdscr.addstr(12 + i, 2, hint, curses.A_DIM)

                stdscr.refresh()
                last_tui = now

            time.sleep(0.004)  # ~250 Hz polling, fine-grained

    finally:
        stream.stop()
        stream.close()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()


def _midi_note_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


# ---------------------------------------------------------------------------
# Offline render mode
# ---------------------------------------------------------------------------
def render_offline(
    model: DDSPModel,
    energy: EnergyBiasModule,
    f0_hz: float,
    loudness_db: float,
    duration_sec: float,
    output_path: str,
    device: str,
):
    """
    Render a sequence with scripted energy gestures for careful listening.

    Timeline (4 seconds per direction, 1s gap between):
      0-1s:  baseline (no energy)
      1-3s:  张 ramp 0→1→0
      5-7s:  扰 ramp 0→1→0
      9-11s: 吟 ramp 0→1→0
      13-15s: 忆 ramp 0→1→0
    """
    sample_rate = model.sample_rate
    block_size = model.block_size
    hop_size = block_size
    total_frames = int(duration_sec * sample_rate / hop_size)

    f0 = torch.full((1, total_frames), f0_hz, device=device)
    loudness = torch.full((1, total_frames), loudness_db, device=device)

    with torch.no_grad():
        f0_scaled = scale_f0(f0)
        harm, noise = model.decoder(f0_scaled, loudness)  # [1, T, *]

    # Build energy level trajectories
    levels_seq = []
    for t in range(total_frames):
        sec = t * hop_size / sample_rate
        levels = {"tension": 0.0, "turbulence": 0.0, "resonance": 0.0, "memory": 0.0}

        def _ramp(t_sec, start, center, end):
            if t_sec < start or t_sec > end:
                return 0.0
            mid = (start + end) / 2
            dur = (end - start) / 2
            dist = abs(t_sec - mid) / dur
            return max(0.0, 1.0 - dist)

        levels["tension"] = _ramp(sec, 1.0, 2.0, 3.0)
        levels["turbulence"] = _ramp(sec, 5.0, 6.0, 7.0)
        levels["resonance"] = _ramp(sec, 9.0, 10.0, 11.0)
        levels["memory"] = _ramp(sec, 13.0, 14.0, 15.0)
        levels_seq.append(levels)

    # Apply energy biases frame by frame
    harm_out = harm.clone()
    noise_out = noise.clone()

    # Stateless ops first (can use subset of frames)
    # We'll process frame by frame for clean state handling
    harm_frames = []
    noise_frames = []
    for t in range(total_frames):
        h_t = harm[:, t : t + 1, :]
        n_t = noise[:, t : t + 1, :]
        h_t, n_t = energy(h_t, n_t, levels_seq[t])
        harm_frames.append(h_t)
        noise_frames.append(n_t)

    harm_biased = torch.cat(harm_frames, dim=1)
    noise_biased = torch.cat(noise_frames, dim=1)

    # Synthesize
    harm_audio = model.harmonic_synth(harm_biased, f0)
    noise_audio = model.noise_synth(noise_biased)
    audio = (harm_audio + noise_audio).squeeze(0).cpu().numpy().astype(np.float32)

    # Normalize
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio / peak * 0.95

    import soundfile as sf

    sf.write(output_path, audio, sample_rate)
    print(f"Rendered {duration_sec:.0f}s → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 2 energy direction test")
    parser.add_argument("--checkpoint", default="checkpoints/phase1_final.pt")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--midi", type=int, default=67, help="MIDI note (default: G4=67)")
    parser.add_argument("--loudness", type=float, default=-10.0, help="Loudness in dB")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--render",
        default=None,
        metavar="PATH",
        help="Render scripted energy sequence to WAV instead of interactive mode",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]
    sample_rate = data_cfg["sample_rate"]
    block_size = data_cfg["block_size"]

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DDSPModel(
        hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        sample_rate=sample_rate,
        block_size=block_size,
        table_size=model_cfg["table_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()
    print(f"Loaded checkpoint step {ckpt['step']}")

    energy_module = EnergyBiasModule(
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        sample_rate=sample_rate,
        block_size=block_size,
    )
    energy_module.to(args.device)
    energy_module.eval()

    if args.render:
        render_offline(
            model,
            energy_module,
            midi_to_hz(args.midi),
            args.loudness,
            duration_sec=18.0,
            output_path=args.render,
            device=args.device,
        )
    else:
        synth = RealtimeSynth(model, energy_module)
        synth.reset()
        print(f"Starting interactive mode — note: {_midi_note_name(args.midi)}")
        print("Keys: 1=张 2=扰 3=吟 4=忆  ↑↓=pitch  ←→=loudness  q=quit")
        print("Press keys to toggle energy directions, listen for changes.")
        try:
            run_interactive(synth, block_size, sample_rate)
        except KeyboardInterrupt:
            pass
        print("Done.")


if __name__ == "__main__":
    main()
