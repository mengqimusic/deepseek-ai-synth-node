#!/usr/bin/env python3
"""
Phase 6 interactive test: Hypernetwork + 5 Voice integrated performance pipeline.

Each Voice's energy state drives per-Voice ΔW via hypernetwork, producing
independent timbre directions.  Same (f0, loudness) + different energy →
different harmonic structure.

Keys:
  1-5      — select Voice for energy injection
  q w e r  — toggle Tension / Turbulence / Resonance / Memory on selected Voice
  Space    — trigger note on selected Voice (current pitch)
  Up/Down  — adjust pitch ±1 semitone for selected Voice
  Left/Right — adjust loudness ±3 dB for selected Voice
  z        — release all notes on selected Voice
  a        — release all notes (all Voices)
  x        — reset selected Voice (destructive — clears developmental state)
  m        — toggle hypernetwork modulation bypass (A/B comparison)
  Esc      — quit

Requires: sounddevice (pip install sounddevice)
"""

import argparse
import math
import queue
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
from synth.nn.hypernetwork import Hypernetwork
from synth.poly import PolyphonicSynth
from synth.voice import ENERGY_NAMES, PHASE_THRESHOLDS, PHASE_BASELINE
from synth.dsp.processors import midi_to_hz

ENERGY_LABELS = {"tension": "张", "turbulence": "扰", "resonance": "吟", "memory": "忆"}
ENERGY_KEYS = {"tension": "q", "turbulence": "w", "resonance": "e", "memory": "r"}

AUDIO_QUEUE_DEPTH = 64  # ~256ms buffer at 4ms frame

# ---------------------------------------------------------------------------
# Shared state (main thread writes, inference thread reads)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_running = True
_selected_voice = 0  # 0-4
_voice_pitch = [67, 67, 67, 67, 67]
_voice_loudness = [-10.0, -10.0, -10.0, -10.0, -10.0]
_voice_energy_toggle = [{k: 0.0 for k in ENERGY_NAMES} for _ in range(5)]
_voice_energy_smooth = [{k: 0.0 for k in ENERGY_NAMES} for _ in range(5)]
_voice_trigger = [False] * 5
_modulation_bypass = False  # set True to send zero energy to hypernetwork

# Feedback coupling state (Phase 7 D3)
_feedback_bypass = False
_feedback_self_enabled = True
_feedback_phase_lock_enabled = True
_feedback_diffusion_enabled = True
_feedback_self_gain = 0.008
_feedback_phase_lock_gain = 0.10
_feedback_diffusion_rate = 0.005

# Stats (inference thread writes, main thread reads)
_underrun_count = 0
_overrun_count = 0


def inference_loop(synth: PolyphonicSynth, audio_queue: queue.Queue, block_size: int):
    """
    Background thread: run process_frame() in a tight loop, push to queue.
    """
    global _underrun_count, _overrun_count

    while _running:
        with _lock:
            for vid in range(5):
                if _voice_trigger[vid]:
                    synth.note_on(_voice_pitch[vid], _voice_loudness[vid])
                    _voice_trigger[vid] = False
            for vid in range(5):
                levels = dict(_voice_energy_smooth[vid])
                if _modulation_bypass:
                    levels = {k: 0.0 for k in ENERGY_NAMES}
                synth.set_all_energy(vid, levels)

        # Feedback sync (read-only globals, no lock needed)
        synth.set_feedback_bypass(_feedback_bypass)
        synth.set_feedback_self_enabled(_feedback_self_enabled)
        synth.set_feedback_phase_lock_enabled(_feedback_phase_lock_enabled)
        synth.set_feedback_diffusion_enabled(_feedback_diffusion_enabled)
        synth.set_feedback_self_gain(_feedback_self_gain)
        synth.set_feedback_phase_lock_gain(_feedback_phase_lock_gain)
        synth.set_feedback_diffusion_rate(_feedback_diffusion_rate)

        audio = synth.process_frame()

        # Throttle: if queue is filling up, skip this frame to let audio drain.
        # This prevents runaway overruns when inference is faster than playback.
        if audio_queue.qsize() >= AUDIO_QUEUE_DEPTH - 4:
            _overrun_count += 1
        else:
            try:
                audio_queue.put_nowait(audio)
            except queue.Full:
                _overrun_count += 1


def audio_callback_factory(audio_queue: queue.Queue, block_size: int):
    """Return a sounddevice callback that reads pre-computed audio from queue."""

    last_chunk = np.zeros(block_size, dtype=np.float32)

    def callback(outdata, frames, _time, status):
        nonlocal last_chunk
        if status:
            print(f"audio status: {status}", file=sys.stderr)

        global _underrun_count

        offset = 0
        remaining = frames
        while remaining > 0:
            try:
                chunk = audio_queue.get_nowait()
                last_chunk = chunk
            except queue.Empty:
                chunk = last_chunk
                _underrun_count += 1

            n = min(block_size, remaining)
            outdata[offset : offset + n, 0] = chunk[:n]
            offset += n
            remaining -= n

    return callback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bar(value: float, width: int = 12) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def _phase_marker(triggered: bool) -> str:
    return "◆" if triggered else "◇"


def _midi_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def _timbre_label(levels: dict[str, float]) -> str:
    """Heuristic timbre label from energy direction mix."""
    total = sum(levels.values())
    if total < 0.01:
        return "neutral"
    dominant = max(levels, key=levels.get)
    fraction = levels[dominant] / total
    if fraction > 0.6:
        return ENERGY_LABELS[dominant]
    # Blend: show top two directions
    sorted_dirs = sorted(levels, key=levels.get, reverse=True)
    return f"{ENERGY_LABELS[sorted_dirs[0]]}+{ENERGY_LABELS[sorted_dirs[1]]}"


# ---------------------------------------------------------------------------
# Curses TUI
# ---------------------------------------------------------------------------
def run_interactive(synth: PolyphonicSynth, block_size: int, sample_rate: int,
                    has_modulation: bool = False):
    import curses

    global _selected_voice, _voice_pitch, _voice_loudness, _running
    global _voice_energy_toggle, _voice_energy_smooth, _voice_trigger
    global _underrun_count, _overrun_count, _modulation_bypass
    global _feedback_bypass, _feedback_self_enabled, _feedback_phase_lock_enabled
    global _feedback_diffusion_enabled, _feedback_self_gain
    global _feedback_phase_lock_gain, _feedback_diffusion_rate

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    curses.curs_set(0)

    # Smoothing
    smooth = [{k: 0.0 for k in ENERGY_NAMES} for _ in range(5)]
    attack_alpha = 1.0 - math.exp(-0.004 / 0.030)
    release_alpha = 1.0 - math.exp(-0.004 / 0.150)

    # Audio queue + inference thread
    audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_DEPTH)

    # Pre-fill before starting daemon thread to avoid concurrent state mutation
    for _ in range(8):
        audio_queue.put(synth.process_frame())

    infer_thread = threading.Thread(
        target=inference_loop,
        args=(synth, audio_queue, block_size),
        daemon=True,
    )
    infer_thread.start()

    stream = sd.OutputStream(
        samplerate=sample_rate,
        blocksize=block_size * 8,
        channels=1,
        callback=audio_callback_factory(audio_queue, block_size),
        dtype=np.float32,
    )

    try:
        stream.start()
        last_tui = 0.0

        while _running:
            key = stdscr.getch()
            now = time.time()
            vid = _selected_voice

            # — Keyboard dispatch —
            if key == ord("1"): _selected_voice = 0
            elif key == ord("2"): _selected_voice = 1
            elif key == ord("3"): _selected_voice = 2
            elif key == ord("4"): _selected_voice = 3
            elif key == ord("5"): _selected_voice = 4

            elif key == ord("q"):
                _voice_energy_toggle[vid]["tension"] = (
                    1.0 if _voice_energy_smooth[vid]["tension"] < 0.5 else 0.0
                )
            elif key == ord("w"):
                _voice_energy_toggle[vid]["turbulence"] = (
                    1.0 if _voice_energy_smooth[vid]["turbulence"] < 0.5 else 0.0
                )
            elif key == ord("e"):
                _voice_energy_toggle[vid]["resonance"] = (
                    1.0 if _voice_energy_smooth[vid]["resonance"] < 0.5 else 0.0
                )
            elif key == ord("r"):
                _voice_energy_toggle[vid]["memory"] = (
                    1.0 if _voice_energy_smooth[vid]["memory"] < 0.5 else 0.0
                )

            elif key == ord("m"):
                _modulation_bypass = not _modulation_bypass

            # — Feedback controls (Phase 7 D3) —
            elif key == ord("f"):
                _feedback_self_enabled = not _feedback_self_enabled
            elif key == ord("g"):
                _feedback_phase_lock_enabled = not _feedback_phase_lock_enabled
            elif key == ord("h"):
                _feedback_diffusion_enabled = not _feedback_diffusion_enabled
            elif key == ord("F"):  # Shift+F = global feedback bypass
                _feedback_bypass = not _feedback_bypass
            elif key == ord("G"):  # Shift+G = increase self-feedback gain
                _feedback_self_gain = min(0.5, _feedback_self_gain + 0.01)
            elif key == ord("H"):  # Shift+H = increase diffusion rate
                _feedback_diffusion_rate = min(0.2, _feedback_diffusion_rate + 0.005)

            elif key == ord(" "):  # Space: trigger note
                with _lock:
                    _voice_trigger[vid] = True

            elif key == curses.KEY_UP:
                _voice_pitch[vid] = min(96, _voice_pitch[vid] + 1)
            elif key == curses.KEY_DOWN:
                _voice_pitch[vid] = max(21, _voice_pitch[vid] - 1)
            elif key == curses.KEY_RIGHT:
                _voice_loudness[vid] = min(0.0, _voice_loudness[vid] + 3.0)
            elif key == curses.KEY_LEFT:
                _voice_loudness[vid] = max(-40.0, _voice_loudness[vid] - 3.0)

            elif key == ord("z"):  # release selected Voice's notes
                with _lock:
                    active = synth.active_notes()
                    for midi, v in list(active.items()):
                        if v == vid:
                            synth.note_off(midi)
            elif key == ord("a"):  # release all notes
                with _lock:
                    synth.all_notes_off()
            elif key == ord("x"):  # reset selected Voice
                with _lock:
                    synth.reset_voice(vid)
                _voice_energy_toggle[vid] = {k: 0.0 for k in ENERGY_NAMES}
                smooth[vid] = {k: 0.0 for k in ENERGY_NAMES}

            elif key == 27:  # Esc
                break

            # — Energy smoothing —
            for v in range(5):
                for k in ENERGY_NAMES:
                    target = _voice_energy_toggle[v][k]
                    alpha = attack_alpha if target > smooth[v][k] else release_alpha
                    smooth[v][k] += alpha * (target - smooth[v][k])

            with _lock:
                for v in range(5):
                    _voice_energy_smooth[v] = dict(smooth[v])

            # — TUI rendering (throttled to ~20 Hz) —
            if now - last_tui > 0.05:
                stdscr.erase()
                h, w = stdscr.getmaxyx()

                if has_modulation:
                    mod_status = "BYPASSED" if _modulation_bypass else "ACTIVE"
                    title = f" Phase 6 — Hypernetwork + 5 Voice [{mod_status}] "
                else:
                    title = " Phase 6 — 5 Voice (no hypernetwork) "
                stdscr.addstr(0, max(0, (w - len(title)) // 2), title, curses.A_REVERSE)

                # Voice selection header
                header = "  "
                for v in range(5):
                    marker = "▶" if v == vid else " "
                    header += f" {marker}Voice {v+1} {marker}  "
                stdscr.addstr(2, 2, header)

                # Per-Voice state display
                active_notes = synth.active_notes()
                row = 4
                col_labels = "ID  Note  L(dB)  Timbre        张          扰          吟          忆         Phases  Weight"
                stdscr.addstr(row, 2, col_labels)
                row += 1
                stdscr.addstr(row, 2, "─" * min(82, w - 4))
                row += 1

                for v in range(5):
                    if row >= h - 6:
                        break

                    st = synth.get_voice_state(v)
                    if st is None:
                        continue

                    # Find active note for this voice
                    note_str = "---"
                    for midi, voice_id in active_notes.items():
                        if voice_id == v:
                            note_str = _midi_name(midi)
                            break

                    phases = (
                        f"{_phase_marker(st.phase_tension)}"
                        f"{_phase_marker(st.phase_turbulence)}"
                        f"{_phase_marker(st.phase_resonance)}"
                        f"{_phase_marker(st.phase_memory)}"
                    )

                    # Energy bars per direction
                    bars = ""
                    for k in ENERGY_NAMES:
                        bars += f" {_bar(smooth[v][k], 8)} "

                    timbre = _timbre_label(smooth[v])

                    sel = ">" if v == vid else " "
                    line = (
                        f"{sel}{v+1}  {note_str:4s} {_voice_loudness[v]:+5.0f} "
                        f"{timbre:12s} {bars} {phases}   {st.competition_weight:.2f}"
                    )
                    stdscr.addstr(row, 2, line)
                    row += 1

                # Accumulation details for selected Voice
                row += 1
                st = synth.get_voice_state(vid)
                if st:
                    stdscr.addstr(row, 2, f"Voice {vid+1} accum (s): ", curses.A_BOLD)
                    accum_str = " | ".join(
                        f"{ENERGY_LABELS[k]}={st.energy_accumulation[k]:.1f}"
                        for k in ENERGY_NAMES
                    )
                    stdscr.addstr(row, 26, accum_str)
                    row += 1

                    # Hypernetwork energy vector
                    hn_str = " | ".join(
                        f"{ENERGY_LABELS[k]}={smooth[vid][k]:.3f}"
                        for k in ENERGY_NAMES
                    )
                    bypass_note = " (ZERO)" if _modulation_bypass else ""
                    stdscr.addstr(row, 2, f"HN input:{bypass_note} [{hn_str}]")
                    row += 1

                    wd_str = (
                        f"withdrawal: L={st.withdrawal_low:.2f} "
                        f"M={st.withdrawal_mid:.2f} H={st.withdrawal_high:.2f}"
                    )
                    stdscr.addstr(row, 2, wd_str)

                # Controls
                row = h - 8
                qsz = audio_queue.qsize()
                qbar = _bar(min(qsz / AUDIO_QUEUE_DEPTH, 1.0), 20)
                health = "OK" if _underrun_count == 0 else f"UNDERRUN x{_underrun_count}"
                stdscr.addstr(row, 2, f"Queue: [{qbar}] {qsz}/{AUDIO_QUEUE_DEPTH}  {health}")
                row += 1
                stdscr.addstr(row, 2, "─" * min(82, w - 4))
                row += 1
                stdscr.addstr(row, 2, "1-5:select Voice  qwer:toggle energy  Space:note on  z:release voice  a:all off  x:reset")
                row += 1
                emap = ", ".join(f"{v}={ENERGY_LABELS[k]}" for k, v in ENERGY_KEYS.items())
                stdscr.addstr(row, 2, f"Directions: {emap}")
                row += 1
                stdscr.addstr(row, 2, "↑↓:pitch  ←→:loudness  m:mod bypass  fgh:feedback toggle  F:fb bypass  GH:fb gain±")
                row += 1
                if has_modulation:
                    bypass_str = "ON (zero energy to HN)" if _modulation_bypass else "OFF (energy drives ΔW)"
                    stdscr.addstr(row, 2, f"Modulation: {bypass_str}")
                else:
                    stdscr.addstr(row, 2, "Modulation: N/A (no hypernetwork loaded)")
                row += 1

                # Feedback status (Phase 7 D3)
                if _feedback_bypass:
                    fb_status = "FEEDBACK BYPASSED (all mechanisms disabled)"
                else:
                    parts = []
                    parts.append(f"self={'ON' if _feedback_self_enabled else 'OFF'}")
                    parts.append(f"phase-lock={'ON' if _feedback_phase_lock_enabled else 'OFF'}")
                    parts.append(f"diffusion={'ON' if _feedback_diffusion_enabled else 'OFF'}")
                    parts.append(f"gain={_feedback_self_gain:.2f}")
                    parts.append(f"rate={_feedback_diffusion_rate:.3f}")
                    fb_status = "Feedback: " + "  ".join(parts)
                stdscr.addstr(row, 2, fb_status)
                row += 1

                stdscr.addstr(row, 2, f"Phase thresholds: " + ", ".join(
                    f"{ENERGY_LABELS[k]}={PHASE_THRESHOLDS[k]:.0f}s" for k in ENERGY_NAMES
                ))

                stdscr.refresh()
                last_tui = now

            time.sleep(0.004)

    finally:
        _running = False
        infer_thread.join(timeout=1.0)
        stream.stop()
        stream.close()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        if _underrun_count or _overrun_count:
            print(f"Audio stats: {_underrun_count} underruns, {_overrun_count} overruns")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_model_with_hypernetwork(
    config_path: str,
    base_checkpoint: str,
    hypernetwork_checkpoint: str,
    device: str,
) -> tuple[PolyphonicSynth, DDSPModel]:
    """Load base model + hypernetwork, wire into PolyphonicSynth."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]
    sample_rate = data_cfg["sample_rate"]
    block_size = data_cfg["block_size"]

    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    model = DDSPModel(
        hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        sample_rate=sample_rate,
        block_size=block_size,
        table_size=model_cfg.get("table_size", 2048),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    hypernetwork = Hypernetwork(
        hidden_size=model_cfg["hidden_size"],
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        bottleneck=model_cfg.get("bottleneck", 48),
        max_scale=model_cfg.get("max_scale", 0.12),
    )
    h_ckpt = torch.load(hypernetwork_checkpoint, map_location="cpu", weights_only=False)
    hypernetwork.load_state_dict(h_ckpt["hypernetwork_state_dict"])
    hypernetwork.to(device)
    hypernetwork.eval()

    synth = PolyphonicSynth(
        decoder=model.decoder,
        harmonic_synth=model.harmonic_synth,
        noise_synth=model.noise_synth,
        num_voices=5,
        n_harmonics=model_cfg["n_harmonics"],
        n_magnitudes=model_cfg["n_magnitudes"],
        sample_rate=sample_rate,
        block_size=block_size,
        hypernetwork=hypernetwork,
    )
    synth.to(device)
    synth.eval()

    return synth, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 6 interactive hypernetwork + 5-Voice test")
    parser.add_argument("--base-checkpoint", default="checkpoints/phase1_final.pt")
    parser.add_argument("--hypernetwork-checkpoint", default="checkpoints/phase5_final.pt")
    parser.add_argument("--config", default="configs/phase5.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-hypernetwork", action="store_true",
                        help="Run without hypernetwork for baseline comparison")
    args = parser.parse_args()

    has_modulation = not args.no_hypernetwork

    if has_modulation:
        if not Path(args.hypernetwork_checkpoint).exists():
            print(f"Hypernetwork checkpoint not found: {args.hypernetwork_checkpoint}")
            print("Falling back to no hypernetwork. Use --no-hypernetwork to suppress this warning.")
            has_modulation = False

    if has_modulation:
        synth, model = load_model_with_hypernetwork(
            args.config, args.base_checkpoint, args.hypernetwork_checkpoint, args.device
        )
        hn_params = sum(p.numel() for p in synth.voices[0].modulated_decoder.hypernetwork.parameters())
        print(f"Loaded base model (step {torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)['step']})")
        print(f"Loaded hypernetwork ({hn_params:,} params)")
    else:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        model_cfg = config["model"]
        data_cfg = config["data"]
        sample_rate = data_cfg["sample_rate"]
        block_size = data_cfg["block_size"]

        ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        model = DDSPModel(
            hidden_size=model_cfg["hidden_size"],
            n_harmonics=model_cfg["n_harmonics"],
            n_magnitudes=model_cfg["n_magnitudes"],
            sample_rate=sample_rate,
            block_size=block_size,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(args.device)
        model.eval()

        synth = PolyphonicSynth(
            decoder=model.decoder,
            harmonic_synth=model.harmonic_synth,
            noise_synth=model.noise_synth,
            num_voices=5,
            sample_rate=sample_rate,
            block_size=block_size,
        )
        synth.to(args.device)
        synth.eval()
        print(f"Loaded base model (step {ckpt['step']}) — no hypernetwork")

    print("Phase 6 Interactive — Hypernetwork + 5 Voice Performance Pipeline")
    print("Keys: 1-5=select voice  qwer=toggle energy  Space=note  m=bypass  Esc=quit")
    print("Starting...")

    try:
        run_interactive(synth, synth.block_size, synth.sample_rate, has_modulation)
    except KeyboardInterrupt:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
