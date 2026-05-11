from dataclasses import dataclass, field
import torch
import torch.nn as nn

from synth.energy import EnergyBiasModule
from synth.dsp.processors import scale_f0

ENERGY_NAMES = ["tension", "turbulence", "resonance", "memory"]

# Phase transition thresholds (cumulative seconds at level=1.0)
PHASE_THRESHOLDS = {
    "tension": 5.0,
    "turbulence": 5.0,
    "resonance": 5.0,
    "memory": 5.0,
}

PHASE_BASELINE = 0.08  # floor level after phase transition


@dataclass
class VoiceState:
    """Per-Voice developmental state. Deterministic given same inputs."""

    voice_id: int

    # Cumulative energy injection (seconds at level=1.0 equivalent)
    energy_accumulation: dict[str, float] = field(default_factory=lambda: {
        k: 0.0 for k in ENERGY_NAMES
    })

    # Irreversible phase transition flags
    phase_tension: bool = False
    phase_turbulence: bool = False
    phase_resonance: bool = False
    phase_memory: bool = False

    # Spectral competition weight (derived from accumulation history)
    competition_weight: float = 1.0

    # Withdrawal style: concession factors per frequency band [low, mid, high]
    withdrawal_low: float = 0.5
    withdrawal_mid: float = 0.5
    withdrawal_high: float = 0.5

    # Current note state
    active_note: int | None = None
    active_loudness: float = -10.0

    @property
    def phase_count(self) -> int:
        return sum([self.phase_tension, self.phase_turbulence,
                     self.phase_resonance, self.phase_memory])

    @property
    def dominant_direction(self) -> str | None:
        """Return the direction with highest accumulation, or None if all zero."""
        if all(v == 0.0 for v in self.energy_accumulation.values()):
            return None
        return max(self.energy_accumulation, key=self.energy_accumulation.get)


class VoiceModule(nn.Module):
    """
    A single Voice with independent developmental history.

    Shares the DDSPDecoder weights with other Voices but maintains
    its own GRU hidden state, EnergyBiasModule state buffers, and
    developmental state (energy accumulation, phase transitions,
    competition profile).
    """

    def __init__(
        self,
        voice_id: int,
        decoder: nn.Module,  # shared DDSPDecoder
        harmonic_synth: nn.Module,  # shared WavetableHarmonicSynth
        noise_synth: nn.Module,  # shared FilteredNoiseSynth
        n_harmonics: int = 100,
        n_magnitudes: int = 65,
        sample_rate: int = 16000,
        block_size: int = 64,
        modulated_decoder: nn.Module | None = None,  # shared ModulatedDecoder
    ):
        super().__init__()
        self.voice_id = voice_id
        self.decoder = decoder
        self.modulated_decoder = modulated_decoder
        self.harmonic_synth = harmonic_synth
        self.noise_synth = noise_synth
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.frame_duration = block_size / sample_rate

        self.energy_module = EnergyBiasModule(
            n_harmonics=n_harmonics,
            n_magnitudes=n_magnitudes,
            sample_rate=sample_rate,
            block_size=block_size,
        )

        self.state = VoiceState(voice_id=voice_id)
        self._gru_hidden = None  # [1, 1, hidden_size] or None
        self._noise_gen = torch.Generator()
        self._noise_gen.manual_seed(voice_id + 1)  # per-voice deterministic noise

    def _apply_phase_baseline(self, levels: dict[str, float]) -> dict[str, float]:
        """Apply phase-based baseline floors to energy levels."""
        result = dict(levels)
        if self.state.phase_tension:
            result["tension"] = max(result.get("tension", 0.0), PHASE_BASELINE)
        if self.state.phase_turbulence:
            result["turbulence"] = max(result.get("turbulence", 0.0), PHASE_BASELINE)
        if self.state.phase_resonance:
            result["resonance"] = max(result.get("resonance", 0.0), PHASE_BASELINE)
        if self.state.phase_memory:
            result["memory"] = max(result.get("memory", 0.0), PHASE_BASELINE)
        return result

    def _check_phase_transitions(self):
        """Check and apply irreversible phase transitions based on accumulation."""
        for direction in ENERGY_NAMES:
            accum = self.state.energy_accumulation[direction]
            threshold = PHASE_THRESHOLDS[direction]

            if direction == "tension" and not self.state.phase_tension and accum >= threshold:
                self.state.phase_tension = True
            elif direction == "turbulence" and not self.state.phase_turbulence and accum >= threshold:
                self.state.phase_turbulence = True
            elif direction == "resonance" and not self.state.phase_resonance and accum >= threshold:
                self.state.phase_resonance = True
            elif direction == "memory" and not self.state.phase_memory and accum >= threshold:
                self.state.phase_memory = True

    def _update_competition_profile(self):
        """Update competition weight and withdrawal style from accumulation history."""
        acc = self.state.energy_accumulation
        total = sum(acc.values())

        # Competition weight: accumulated voices have more say
        self.state.competition_weight = 1.0 + total * 0.1 + self.state.phase_count * 0.3

        # Withdrawal style: blend of direction-specific profiles
        if total > 0:
            t_frac = acc["tension"] / total
            u_frac = acc["turbulence"] / total
            r_frac = acc["resonance"] / total
            m_frac = acc["memory"] / total

            # Each direction's preferred withdrawal profile
            # Tension: preserve harmonics (yield mids less)
            # Turbulence: preserve sideband texture (yield mids more)
            # Resonance: preserve lows (yield highs more)
            # Memory: balanced
            self.state.withdrawal_low = (
                0.5 * t_frac + 0.5 * u_frac + 0.3 * r_frac + 0.5 * m_frac
            ) / (t_frac + u_frac + r_frac + m_frac)
            self.state.withdrawal_mid = (
                0.3 * t_frac + 0.7 * u_frac + 0.5 * r_frac + 0.5 * m_frac
            ) / (t_frac + u_frac + r_frac + m_frac)
            self.state.withdrawal_high = (
                0.3 * t_frac + 0.5 * u_frac + 0.7 * r_frac + 0.5 * m_frac
            ) / (t_frac + u_frac + r_frac + m_frac)

    def process_params(
        self,
        f0_hz: float,
        loudness_db: float,
        levels: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate biased synthesis parameters for this Voice (step 1/2).

        Runs decoder + energy bias + state updates. Returns parameters
        that can be modified by the competition scheduler before synthesis.

        Args:
            f0_hz: fundamental frequency in Hz
            loudness_db: loudness in dB
            levels: per-direction energy injection levels {name: float}

        Returns:
            harm_amps: [1, 1, n_harmonics] — biased harmonic amplitudes
            noise_mags: [1, 1, n_magnitudes] — biased noise magnitudes
            f0_tensor: [1, 1] — f0_hz tensor for synthesis
        """
        device = next(self.decoder.parameters()).device

        with torch.no_grad():
            # Apply phase baseline to effective levels
            effective_levels = self._apply_phase_baseline(levels)

            # Accumulate energy
            for k in ENERGY_NAMES:
                self.state.energy_accumulation[k] += effective_levels[k] * self.frame_duration

            # Check phase transitions
            self._check_phase_transitions()

            # Update competition profile
            self._update_competition_profile()

            # Decoder inference — hypernetwork-modulated or vanilla
            f0 = torch.tensor([[f0_hz]], device=device)
            loudness = torch.tensor([[loudness_db]], device=device)
            f0_scaled = scale_f0(f0)

            if self.modulated_decoder is not None:
                energy_tensor = torch.tensor([[
                    effective_levels["tension"],
                    effective_levels["turbulence"],
                    effective_levels["resonance"],
                    effective_levels["memory"]
                ]], device=device)
                harm, noise, self._gru_hidden = self.modulated_decoder.forward_step(
                    f0_scaled, loudness, energy_tensor, self._gru_hidden
                )
            else:
                x = torch.stack([f0_scaled, loudness], dim=-1)  # [1, 1, 2]
                x = self.decoder.pre_mlp(x)
                x, h = self.decoder.gru(x, self._gru_hidden)
                self._gru_hidden = h.detach()
                x = self.decoder.post_mlp(x)
                harm = torch.sigmoid(self.decoder.harm_head(x))
                noise = torch.sigmoid(self.decoder.noise_head(x))

            # Apply energy bias
            harm, noise = self.energy_module(harm, noise, effective_levels)

            return harm, noise, f0

    def synthesize_from(
        self,
        harm_amps: torch.Tensor,
        noise_mags: torch.Tensor,
        f0_hz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Synthesize audio from (possibly competition-modified) parameters (step 2/2).

        Args:
            harm_amps: [1, 1, n_harmonics]
            noise_mags: [1, 1, n_magnitudes]
            f0_hz: [1, 1]

        Returns:
            audio: [block_size] numpy array
        """
        with torch.no_grad():
            audio = self.harmonic_synth(harm_amps, f0_hz) + self.noise_synth(
                noise_mags, generator=self._noise_gen
            )
            return audio.squeeze(0).squeeze(0).cpu().numpy().astype("float32")

    def process_frame(
        self,
        f0_hz: float,
        loudness_db: float,
        levels: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate one frame of audio for this Voice (combined 1+2, no competition).

        Returns:
            audio: [block_size] numpy array
            harm_amps: [n_harmonics] tensor (for external analysis)
            noise_mags: [n_magnitudes] tensor (for external analysis)
        """
        harm, noise, f0 = self.process_params(f0_hz, loudness_db, levels)
        audio = self.synthesize_from(harm, noise, f0)
        harm_out = harm.squeeze(0).squeeze(0)  # [n_harmonics]
        noise_out = noise.squeeze(0).squeeze(0)  # [n_magnitudes]
        return audio, harm_out, noise_out

    def reset(self):
        """Reset GRU hidden state, energy module state buffers, and noise generator."""
        self._gru_hidden = None
        self.energy_module.reset_state()
        self._noise_gen.manual_seed(self.voice_id + 1)

    def reset_full(self):
        """Full reset including developmental state (destructive)."""
        self.reset()
        self.state = VoiceState(voice_id=self.voice_id)

    def set_note(self, midi_note: int | None, loudness_db: float = -10.0):
        """Set the currently active note for this Voice."""
        self.state.active_note = midi_note
        if midi_note is not None:
            self.state.active_loudness = loudness_db
            self.reset()  # fresh start for new note
