#!/usr/bin/env python3
"""
Phase 3 interactive test: multi-Voice development and spectral competition.

Real-time 5-Voice polyphonic synthesis with independent energy injection
per Voice.  Monitor Voice states, phase transitions, and competition
dynamics live.

Keys:
  1-5     — select Voice for energy injection
  q w e r — toggle Tension / Turbulence / Resonance / Memory on selected Voice
  Space   — trigger note on selected Voice (current pitch)
  Up/Down — adjust pitch ±1 semitone for selected Voice
  Left/Right — adjust loudness ±3 dB for selected Voice
  z       — release all notes on selected Voice
  a       — release all notes (all Voices)
  x       — reset selected Voice (destructive — clears developmental state)
  Esc     — quit

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

# Stats (inference thread writes, main thread reads)
_underrun_count = 0
_overrun_count = 0


def inference_loop(synth: PolyphonicSynth, audio_queue: queue.Queue, block_size: int):
    """
    Background thread: run process_frame() in a tight loop, push to queue.
    This decouples inference timing from the audio callback deadline —
    if inference is slower than real-time the queue drains gracefully
    (callback repeats last frame); if faster, the queue fills and put() blocks.
    """
    global _underrun_count, _overrun_count

    while _running:
        # Apply note triggers and energy levels under lock
        with _lock:
            for vid in range(5):
                if _voice_trigger[vid]:
                    synth.note_on(_voice_pitch[vid], _voice_loudness[vid])
                    _voice_trigger[vid] = False
            for vid in range(5):
                synth.set_all_energy(vid, _voice_energy_smooth[vid])

        audio = synth.process_frame()

        try:
            audio_queue.put(audio, timeout=0.1)
        except queue.Full:
            _overrun_count += 1
            # Drop oldest to make room (queue is full → inference is ahead)
            try:
                audio_queue.get_nowait()
                audio_queue.put_nowait(audio)
            except queue.Empty:
                pass


def audio_callback_factory(audio_queue: queue.Queue, block_size: int):
    """Return a sounddevice callback that reads pre-computed audio from queue."""

    last_chunk = np.zeros(block_size, dtype=np.float32)

    def callback(outdata, frames, _time, status):
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


# ---------------------------------------------------------------------------
# Curses TUI
# ---------------------------------------------------------------------------
def run_interactive(synth: PolyphonicSynth, block_size: int, sample_rate: int):
    import curses

    global _selected_voice, _voice_pitch, _voice_loudness, _running
    global _voice_energy_toggle, _voice_energy_smooth, _voice_trigger
    global _underrun_count, _overrun_count

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

    # Audio queue + inference thread (decoupled from callback deadline)
    audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_DEPTH)
    infer_thread = threading.Thread(
        target=inference_loop,
        args=(synth, audio_queue, block_size),
        daemon=True,
    )
    infer_thread.start()

    # Pre-fill a few frames so the callback doesn't underrun on start
    for _ in range(8):
        audio_queue.put(synth.process_frame())

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
                active = synth.active_notes()
                for midi, v in list(active.items()):
                    if v == vid:
                        synth.note_off(midi)
            elif key == ord("a"):  # release all notes
                synth.all_notes_off()
            elif key == ord("x"):  # reset selected Voice
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

                title = " Phase 3 — Multi-Voice Development "
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
                stdscr.addstr(row, 2, "ID  Note  L(dB)  张         扰         吟         忆         Phases  Weight")
                row += 1
                stdscr.addstr(row, 2, "─" * min(78, w - 4))
                row += 1

                for v in range(5):
                    if row >= h - 4:
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

                    sel = ">" if v == vid else " "
                    line = (
                        f"{sel}{v+1}  {note_str:4s} {_voice_loudness[v]:+5.0f} "
                        f"{bars} {phases}   {st.competition_weight:.2f}"
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
                    wd_str = (
                        f"withdrawal: L={st.withdrawal_low:.2f} "
                        f"M={st.withdrawal_mid:.2f} H={st.withdrawal_high:.2f}"
                    )
                    stdscr.addstr(row, 2, wd_str)

                # Controls
                row = h - 7
                # Queue health
                qsz = audio_queue.qsize()
                qbar = _bar(min(qsz / AUDIO_QUEUE_DEPTH, 1.0), 20)
                health = "OK" if _underrun_count == 0 else f"UNDERRUN x{_underrun_count}"
                stdscr.addstr(row, 2, f"Queue: [{qbar}] {qsz}/{AUDIO_QUEUE_DEPTH}  {health}")
                row += 1
                stdscr.addstr(row, 2, "─" * min(78, w - 4))
                row += 1
                stdscr.addstr(row, 2, "1-5:select Voice  qwer:toggle energy  Space:note on  z:release voice  a:all off  x:reset  Esc:quit")
                row += 1
                emap = ", ".join(f"{v}={ENERGY_LABELS[k]}" for k, v in ENERGY_KEYS.items())
                stdscr.addstr(row, 2, f"Directions: {emap}")
                row += 1
                stdscr.addstr(row, 2, "↑↓:pitch  ←→:loudness")
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 3 interactive multi-voice test")
    parser.add_argument("--checkpoint", default="checkpoints/phase1_final.pt")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]
    sample_rate = data_cfg["sample_rate"]
    block_size = data_cfg["block_size"]

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
    synth.to(args.device)
    synth.eval()

    print(f"Loaded checkpoint step {ckpt['step']}")
    print("Phase 3 Interactive — Multi-Voice Development & Competition")
    print("Keys: 1-5=select voice  qwer=toggle energy  Space=note  z=release  a=all off  x=reset  Esc=quit")
    print("Starting...")

    try:
        run_interactive(synth, block_size, sample_rate)
    except KeyboardInterrupt:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
