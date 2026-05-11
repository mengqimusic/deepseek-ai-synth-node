"""
PolyphonicSynth — top-level orchestrator for 5 independent Voices.

Wires together:
  - 5 VoiceModules (shared DDSPDecoder weights, independent state)
  - VoiceAllocator (round-robin)
  - SpectralCompetitionScheduler (soft gain scheduling)
  - Mixer (sum all Voice outputs)

Deterministic: same input sequence + same Voice states → same output.
"""

import numpy as np
import torch
import torch.nn as nn

from synth.dsp.processors import midi_to_hz
from synth.nn.decoder import DDSPDecoder
from synth.dsp.harmonic import WavetableHarmonicSynth
from synth.dsp.noise import FilteredNoiseSynth
from synth.voice import VoiceModule, VoiceState, ENERGY_NAMES
from synth.competition import SpectralCompetitionScheduler


class VoiceAllocator:
    """Round-robin Voice allocation. Not bound to pitch."""

    def __init__(self, num_voices: int = 5):
        self.num_voices = num_voices
        self._next = 0
        self._active_notes: dict[int, int] = {}  # midi_note → voice_id

    def allocate(self, midi_note: int) -> int:
        """Assign a note to the next Voice in round-robin order."""
        # Steal if already active
        if midi_note in self._active_notes:
            return self._active_notes[midi_note]

        voice_id = self._next
        self._next = (self._next + 1) % self.num_voices

        # If target voice was playing a different note, release it
        for note, vid in list(self._active_notes.items()):
            if vid == voice_id:
                del self._active_notes[note]
                break

        self._active_notes[midi_note] = voice_id
        return voice_id

    def release(self, midi_note: int) -> int | None:
        """Release a note. Returns the voice_id that was playing it."""
        return self._active_notes.pop(midi_note, None)

    def voice_for_note(self, midi_note: int) -> int | None:
        return self._active_notes.get(midi_note)

    def active_voices(self) -> set[int]:
        return set(self._active_notes.values())

    def notes_for_voice(self, voice_id: int) -> list[int]:
        return [n for n, v in self._active_notes.items() if v == voice_id]


class PolyphonicSynth(nn.Module):
    """
    5-Voice polyphonic synthesizer with independent Voice development
    and spectral competition scheduling.
    """

    def __init__(
        self,
        decoder: DDSPDecoder,
        harmonic_synth: WavetableHarmonicSynth,
        noise_synth: FilteredNoiseSynth,
        num_voices: int = 5,
        n_harmonics: int = 100,
        n_magnitudes: int = 65,
        sample_rate: int = 16000,
        block_size: int = 64,
    ):
        super().__init__()
        self.num_voices = num_voices
        self.block_size = block_size
        self.sample_rate = sample_rate

        self.voices = nn.ModuleList([
            VoiceModule(
                voice_id=i,
                decoder=decoder,
                harmonic_synth=harmonic_synth,
                noise_synth=noise_synth,
                n_harmonics=n_harmonics,
                n_magnitudes=n_magnitudes,
                sample_rate=sample_rate,
                block_size=block_size,
            )
            for i in range(num_voices)
        ])

        self.allocator = VoiceAllocator(num_voices=num_voices)
        self.competition = SpectralCompetitionScheduler()

        # Per-voice energy injection levels (set externally by performer)
        self._energy_levels: list[dict[str, float]] = [
            {k: 0.0 for k in ENERGY_NAMES} for _ in range(num_voices)
        ]

    # ------------------------------------------------------------------
    # Note interface
    # ------------------------------------------------------------------
    def note_on(self, midi_note: int, loudness_db: float = -10.0) -> int:
        """Trigger a note. Returns the assigned voice_id."""
        voice_id = self.allocator.allocate(midi_note)
        self.voices[voice_id].set_note(midi_note, loudness_db)
        return voice_id

    def note_off(self, midi_note: int) -> int | None:
        """Release a note. Returns the voice_id that was playing it."""
        voice_id = self.allocator.release(midi_note)
        if voice_id is not None:
            self.voices[voice_id].set_note(None)
        return voice_id

    def all_notes_off(self):
        """Silence all Voices."""
        for midi in list(self.allocator._active_notes.keys()):
            self.note_off(midi)

    # ------------------------------------------------------------------
    # Energy injection interface
    # ------------------------------------------------------------------
    def set_energy(self, voice_id: int, direction: str, level: float):
        """Set energy injection level for a specific Voice."""
        if 0 <= voice_id < self.num_voices and direction in ENERGY_NAMES:
            self._energy_levels[voice_id][direction] = max(0.0, min(1.0, level))

    def set_all_energy(self, voice_id: int, levels: dict[str, float]):
        """Set all four energy levels for a specific Voice at once."""
        if 0 <= voice_id < self.num_voices:
            for k in ENERGY_NAMES:
                self._energy_levels[voice_id][k] = max(0.0, min(1.0, levels.get(k, 0.0)))

    def get_energy(self, voice_id: int) -> dict[str, float]:
        if 0 <= voice_id < self.num_voices:
            return dict(self._energy_levels[voice_id])
        return {}

    # ------------------------------------------------------------------
    # Audio generation
    # ------------------------------------------------------------------
    def process_frame(self) -> np.ndarray:
        """
        Generate one frame of mixed audio [block_size].

        Pipeline:
          1. Each active Voice: process_params → (harm, noise, f0)
          2. Spectral competition: adjust harm/noise per Voice
          3. Each Voice: synthesize_from modified params
          4. Mix (sum) all Voice outputs
        """
        active_voices = self.allocator.active_voices()
        active_notes = dict(self.allocator._active_notes)  # snapshot

        # Step 1: get biased params from each Voice
        voice_params = []
        voice_audio = {}  # voice_id → audio for non-competing voices
        param_list = []  # for competition

        for voice_id in range(self.num_voices):
            voice = self.voices[voice_id]

            # Find the note this voice is playing
            playing_note = None
            for midi, vid in active_notes.items():
                if vid == voice_id:
                    playing_note = midi
                    break

            is_active = playing_note is not None

            if not is_active:
                # Energy still accumulates even when voice is silent
                levels = self._energy_levels[voice_id]
                voice.process_params(
                    f0_hz=midi_to_hz(60),  # middle C as neutral
                    loudness_db=-40.0,
                    levels=levels,
                )
                param_list.append({
                    "voice_id": voice_id,
                    "harm_amps": torch.zeros(1, 1, voice.energy_module.n_harmonics),
                    "noise_mags": torch.zeros(1, 1, voice.energy_module.n_magnitudes),
                    "f0_hz": 0.0,
                    "competition_weight": voice.state.competition_weight,
                    "withdrawal_low": voice.state.withdrawal_low,
                    "withdrawal_mid": voice.state.withdrawal_mid,
                    "withdrawal_high": voice.state.withdrawal_high,
                    "is_active": False,
                })
                voice_audio[voice_id] = np.zeros(self.block_size, dtype=np.float32)
                continue

            f0_hz = midi_to_hz(playing_note)
            loudness = voice.state.active_loudness
            levels = self._energy_levels[voice_id]

            harm, noise, f0_tensor = voice.process_params(f0_hz, loudness, levels)

            vp = {
                "voice_id": voice_id,
                "harm_amps": harm,
                "noise_mags": noise,
                "f0_hz": f0_hz,
                "competition_weight": voice.state.competition_weight,
                "withdrawal_low": voice.state.withdrawal_low,
                "withdrawal_mid": voice.state.withdrawal_mid,
                "withdrawal_high": voice.state.withdrawal_high,
                "is_active": True,
            }
            param_list.append(vp)
            voice_params.append(vp)

        # Step 2: spectral competition
        param_list = self.competition(param_list)

        # Step 3: synthesize from (possibly modified) params
        mixed = np.zeros(self.block_size, dtype=np.float32)
        for vp in param_list:
            vid = vp["voice_id"]
            if vp.get("is_active", False):
                f0_tensor = torch.tensor([[vp["f0_hz"]]], device=vp["harm_amps"].device)
                audio = self.voices[vid].synthesize_from(
                    vp["harm_amps"], vp["noise_mags"], f0_tensor
                )
                mixed += audio
            else:
                mixed += voice_audio.get(vid, np.zeros(self.block_size, dtype=np.float32))

        # Soft clip to prevent hard clipping with multi-voice mixing
        mixed = np.tanh(mixed)
        return mixed

    def process_frame_simple(self) -> np.ndarray:
        """
        Generate one frame WITHOUT spectral competition.
        Useful for A/B comparison and debugging.
        """
        active_notes = dict(self.allocator._active_notes)
        mixed = np.zeros(self.block_size, dtype=np.float32)

        for voice_id in range(self.num_voices):
            voice = self.voices[voice_id]

            playing_note = None
            for midi, vid in active_notes.items():
                if vid == voice_id:
                    playing_note = midi
                    break

            if playing_note is None:
                levels = self._energy_levels[voice_id]
                voice.process_params(
                    f0_hz=midi_to_hz(60),
                    loudness_db=-40.0,
                    levels=levels,
                )
                continue

            f0_hz = midi_to_hz(playing_note)
            loudness = voice.state.active_loudness
            levels = self._energy_levels[voice_id]

            audio, _, _ = voice.process_frame(f0_hz, loudness, levels)
            mixed += audio

        mixed = np.tanh(mixed)
        return mixed

    # ------------------------------------------------------------------
    # State query interface
    # ------------------------------------------------------------------
    def get_voice_state(self, voice_id: int) -> VoiceState | None:
        if 0 <= voice_id < self.num_voices:
            return self.voices[voice_id].state
        return None

    def get_all_voice_states(self) -> list[VoiceState]:
        return [self.voices[i].state for i in range(self.num_voices)]

    def active_notes(self) -> dict[int, int]:
        """Return {midi_note: voice_id} for currently active notes."""
        return dict(self.allocator._active_notes)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_all(self):
        """Reset all Voice states (destructive — clears developmental history)."""
        self.all_notes_off()
        for voice in self.voices:
            voice.reset_full()
        self._energy_levels = [
            {k: 0.0 for k in ENERGY_NAMES} for _ in range(self.num_voices)
        ]

    def reset_voice(self, voice_id: int):
        """Reset a single Voice (destructive)."""
        if 0 <= voice_id < self.num_voices:
            for midi, vid in list(self.allocator._active_notes.items()):
                if vid == voice_id:
                    self.note_off(midi)
            self.voices[voice_id].reset_full()
            self._energy_levels[voice_id] = {k: 0.0 for k in ENERGY_NAMES}
