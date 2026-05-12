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
from synth.nn.hypernetwork import HypernetworkV2
from synth.nn.modulated_decoder import ModulatedRichDecoder
from synth.nn.transformer_decoder import RichParamDecoder
from synth.dsp.harmonic import WavetableHarmonicSynth
from synth.dsp.noise import FilteredNoiseSynth, GrainNoiseSynth
from synth.dsp.fm import FMSynth
from synth.dsp.transient import TransientCombNoise
from synth.voice import VoiceModule, VoiceState, ENERGY_NAMES
from synth.competition import SpectralCompetitionScheduler
from synth.feedback import FeedbackCoupler

# Harmonic overlap proximity per semitone interval (1% tolerance, H=20 harmonics)
# Consonant intervals (octave/fifth) → higher coupling; dissonant (tritone) → low.
_HARMONIC_PROXIMITY = {
    0: 1.00,   # Unison
    1: 0.25,   # Minor 2nd
    2: 0.20,   # Major 2nd
    3: 0.25,   # Minor 3rd
    4: 0.25,   # Major 3rd
    5: 0.25,   # Perfect 4th
    6: 0.05,   # Tritone
    7: 0.30,   # Perfect 5th
    8: 0.15,   # Minor 6th
    9: 0.20,   # Major 6th
    10: 0.05,  # Minor 7th
    11: 0.15,  # Major 7th
}


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
        decoder: RichParamDecoder,
        harmonic_synth: WavetableHarmonicSynth,
        noise_synth: FilteredNoiseSynth,
        num_voices: int = 5,
        n_harmonics: int = 256,
        n_magnitudes: int = 65,
        sample_rate: int = 16000,
        block_size: int = 64,
        hypernetwork_v2: HypernetworkV2 | None = None,
        fm_synth: FMSynth | None = None,
        grain_synth: GrainNoiseSynth | None = None,
        transient_synth: TransientCombNoise | None = None,
    ):
        super().__init__()
        self.num_voices = num_voices
        self.block_size = block_size
        self.sample_rate = sample_rate

        if hasattr(decoder, 'n_harmonics'):
            n_harmonics = decoder.n_harmonics
        if hasattr(decoder, 'n_noise_mel'):
            n_magnitudes = decoder.n_noise_mel

        if hypernetwork_v2 is not None:
            modulated_decoder = ModulatedRichDecoder(
                base_decoder=decoder,
                hypernetwork=hypernetwork_v2,
                frozen_decoder=True,
            )
        else:
            modulated_decoder = None

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
                modulated_decoder=modulated_decoder,
                fm_synth=fm_synth,
                grain_synth=grain_synth,
                transient_synth=transient_synth,
            )
            for i in range(num_voices)
        ])

        self.allocator = VoiceAllocator(num_voices=num_voices)
        self.competition = SpectralCompetitionScheduler()
        self.feedback = FeedbackCoupler(
            num_voices=num_voices,
            n_harmonics=n_harmonics,
            sample_rate=sample_rate,
            block_size=block_size,
        )

        # Per-voice energy injection levels (set externally by performer)
        self._energy_levels: list[dict[str, float]] = [
            {k: 0.0 for k in ENERGY_NAMES} for _ in range(num_voices)
        ]

        # Per-voice continuous loudness (dB) — updated each frame by performer
        self._voice_loudness: list[float] = [-10.0] * num_voices

    # ------------------------------------------------------------------
    # Note interface
    # ------------------------------------------------------------------
    def note_on(self, midi_note: int, loudness_db: float = -10.0) -> int:
        """Trigger a note. Returns the assigned voice_id."""
        voice_id = self.allocator.allocate(midi_note)
        self.voices[voice_id].set_note(midi_note, loudness_db)
        self.voices[voice_id].inject_burst(self._energy_levels[voice_id])
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

    def set_loudness(self, voice_id: int, db: float):
        """Set continuous loudness for a specific Voice (dB, -40 to 0)."""
        if 0 <= voice_id < self.num_voices:
            self._voice_loudness[voice_id] = max(-40.0, min(0.0, db))

    def set_all_loudness(self, db: float):
        """Set continuous loudness for all Voices."""
        for i in range(self.num_voices):
            self._voice_loudness[i] = max(-40.0, min(0.0, db))

    def set_energy_gain(self, voice_id: int, gain: float):
        """Set hypernetwork energy gain for a specific Voice (1.0 = default)."""
        if 0 <= voice_id < self.num_voices:
            self.voices[voice_id].set_energy_gain(gain)

    def set_all_energy_gain(self, gain: float):
        """Set hypernetwork energy gain for all Voices."""
        for voice in self.voices:
            voice.set_energy_gain(gain)

    def get_energy(self, voice_id: int) -> dict[str, float]:
        if 0 <= voice_id < self.num_voices:
            return dict(self._energy_levels[voice_id])
        return {}

    # ------------------------------------------------------------------
    # Feedback control interface
    # ------------------------------------------------------------------
    def set_feedback_bypass(self, bypass: bool):
        """Global kill-switch: True = all feedback mechanisms disabled."""
        self.feedback.global_bypass = bypass

    def set_feedback_self_enabled(self, enabled: bool):
        self.feedback.self_feedback_enabled = enabled

    def set_feedback_phase_lock_enabled(self, enabled: bool):
        self.feedback.phase_lock_enabled = enabled

    def set_feedback_diffusion_enabled(self, enabled: bool):
        self.feedback.diffusion_enabled = enabled

    def set_feedback_self_gain(self, gain: float):
        self.feedback.set_self_feedback_gain(gain)

    def set_feedback_phase_lock_gain(self, gain: float):
        self.feedback.set_phase_lock_gain(gain)

    def set_feedback_diffusion_rate(self, rate: float):
        self.feedback.set_diffusion_rate(rate)

    def get_feedback_state(self) -> dict:
        """Return feedback status for TUI display."""
        f = self.feedback
        return {
            "bypass": f.global_bypass,
            "self_enabled": f.self_feedback_enabled,
            "phase_lock_enabled": f.phase_lock_enabled,
            "diffusion_enabled": f.diffusion_enabled,
            "self_gain": f.self_feedback_gain,
            "phase_lock_gain": f.phase_lock_gain,
            "diffusion_rate": f.diffusion_rate,
            "co_activation": f.get_co_activation_matrix(),
        }

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

        # Step 0: energy exchange — crosstalk + phase-lock coupling + diffusion
        f0_active: dict[int, float] = {}  # voice_id → f0_hz (for diffusion)
        if len(active_voices) >= 2:
            active_list = sorted(active_voices)
            for i, vid_a in enumerate(active_list):
                for vid_b in active_list[i + 1:]:
                    note_a = None
                    note_b = None
                    for midi, v in active_notes.items():
                        if v == vid_a:
                            note_a = midi
                        elif v == vid_b:
                            note_b = midi
                    if note_a is not None and note_b is not None:
                        f0_a = midi_to_hz(note_a)
                        f0_b = midi_to_hz(note_b)
                        f0_active[vid_a] = f0_a
                        f0_active[vid_b] = f0_b

                        semitone_dist = abs(note_a - note_b)
                        octave_shift = semitone_dist // 12
                        semitone_mod = semitone_dist % 12
                        base_proximity = _HARMONIC_PROXIMITY.get(semitone_mod, 0.1)
                        proximity = base_proximity * (0.7 ** octave_shift)

                        # Phase-lock bonus: consonant f0 ratios get stronger coupling
                        lock_strength = self.feedback.compute_phase_lock_strength(f0_a, f0_b)
                        effective_proximity = proximity * (1.0 + lock_strength)

                        if effective_proximity > 0.01:
                            voice_a = self.voices[vid_a]
                            voice_b = self.voices[vid_b]
                            voice_a.apply_energy_crosstalk(voice_b._energy_smooth, effective_proximity)
                            voice_b.apply_energy_crosstalk(voice_a._energy_smooth, effective_proximity)

        # Energy field diffusion: all 5 Voices participate
        energy_smooth_list = [
            dict(self.voices[vid]._energy_smooth) for vid in range(self.num_voices)
        ]
        diffusion_deltas = self.feedback.step_diffusion(
            energy_smooth_list, active_notes, f0_active
        )
        for vid in range(self.num_voices):
            self.voices[vid].apply_feedback_energy(diffusion_deltas[vid])

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
                silent_params = {
                    "voice_id": voice_id,
                    "harm_amps": torch.zeros(1, 1, voice.energy_module.n_harmonics),
                    "noise_mags": torch.zeros(1, 1, voice.energy_module.n_magnitudes),
                    "f0_hz": 0.0,
                    "competition_weight": voice.state.competition_weight,
                    "withdrawal_low": voice.state.withdrawal_low,
                    "withdrawal_mid": voice.state.withdrawal_mid,
                    "withdrawal_high": voice.state.withdrawal_high,
                    "is_active": False,
                }
                param_list.append(silent_params)
                voice_audio[voice_id] = np.zeros(self.block_size, dtype=np.float32)
                continue

            f0_hz = midi_to_hz(playing_note)
            loudness = self._voice_loudness[voice_id]
            levels = self._energy_levels[voice_id]

            vp = voice.process_params(f0_hz, loudness, levels)
            vp["voice_id"] = voice_id

            param_list.append(vp)
            voice_params.append(vp)

        # Step 2: spectral competition
        param_list = self.competition(param_list)

        # Step 3: synthesize from (possibly modified) params
        mixed = np.zeros(self.block_size, dtype=np.float32)
        for vp in param_list:
            vid = vp["voice_id"]
            if vp.get("is_active", False):
                audio = self.voices[vid].synthesize_from(vp)
                mixed += audio
            else:
                mixed += voice_audio.get(vid, np.zeros(self.block_size, dtype=np.float32))

        # Step 4: self-feedback — harmonic output features → own energy state
        # Uses post-competition harm_amps (what was actually synthesized).
        # Gated by the performer's target levels so feedback amplifies intent
        # rather than creating energy from nothing.
        # Feedback affects the NEXT frame, not the current one.
        for vp in param_list:
            vid = vp["voice_id"]
            if vp.get("is_active", False):
                target_levels = self._energy_levels[vid]
                deltas = self.feedback.compute_self_feedback(
                    vid, vp["harm_amps"], target_levels
                )
                self.voices[vid].apply_feedback_energy(deltas)

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
            loudness = self._voice_loudness[voice_id]
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
        self.allocator._next = 0
        self.allocator._active_notes.clear()
        for voice in self.voices:
            voice.reset_full()
        self._energy_levels = [
            {k: 0.0 for k in ENERGY_NAMES} for _ in range(self.num_voices)
        ]
        self._voice_loudness = [-10.0] * self.num_voices
        self.feedback.reset_all()

    def reset_voice(self, voice_id: int):
        """Reset a single Voice (destructive)."""
        if 0 <= voice_id < self.num_voices:
            for midi, vid in list(self.allocator._active_notes.items()):
                if vid == voice_id:
                    del self.allocator._active_notes[midi]
            self.voices[voice_id].reset_full()
            self._energy_levels[voice_id] = {k: 0.0 for k in ENERGY_NAMES}
            self._voice_loudness[voice_id] = -10.0
            self.feedback.reset_voice(voice_id)
