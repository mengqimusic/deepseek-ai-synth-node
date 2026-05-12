import math
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
        energy_gain: float = 1.0,  # multiplier for hypernetwork energy input
    ):
        super().__init__()
        self.voice_id = voice_id
        self.decoder = decoder
        self.modulated_decoder = modulated_decoder
        self.energy_gain = energy_gain
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

        # Energy dynamics
        self.energy_attack_ms = 30.0   # attack smoothing time constant
        self.energy_release_ms = 150.0  # release smoothing time constant
        self.energy_burst_factor = 0.3  # burst as fraction of current level on note-on
        self.energy_burst_decay_ms = 100.0  # burst decay time constant
        self._energy_smooth = {k: 0.0 for k in ENERGY_NAMES}
        self._burst_energy = {k: 0.0 for k in ENERGY_NAMES}
        self._attack_alpha = 1.0 - math.exp(-(block_size / sample_rate) / (self.energy_attack_ms / 1000.0))
        self._release_alpha = 1.0 - math.exp(-(block_size / sample_rate) / (self.energy_release_ms / 1000.0))
        self._burst_decay = math.exp(-(block_size / sample_rate) / (self.energy_burst_decay_ms / 1000.0))
        self._harmonic_phase: float = 0.0  # continuous phase across frames [0, harmonic_synth.table_size)

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

    def _apply_energy_dynamics(self, levels: dict[str, float]) -> dict[str, float]:
        """Smooth energy levels toward target and apply burst."""
        result = {}
        for k in ENERGY_NAMES:
            target = levels.get(k, 0.0)
            current = self._energy_smooth[k]
            alpha = self._attack_alpha if target > current else self._release_alpha
            smoothed = current + alpha * (target - current)
            result[k] = max(0.0, min(1.0, smoothed + self._burst_energy[k]))
            self._energy_smooth[k] = smoothed
            # Decay burst
            self._burst_energy[k] *= self._burst_decay
        return result

    def inject_burst(self, target_levels: dict[str, float] | None = None):
        """Inject a per-note energy burst based on target levels or accumulation history.

        If target_levels provided, uses those (current performance intent).
        Otherwise falls back to normalized energy_accumulation (developmental history).
        """
        if target_levels is not None:
            for k in ENERGY_NAMES:
                self._burst_energy[k] += target_levels.get(k, 0.0) * self.energy_burst_factor
                self._burst_energy[k] = min(self._burst_energy[k], 1.0)
        else:
            for k in ENERGY_NAMES:
                accum_norm = min(1.0, self.state.energy_accumulation[k] / 10.0)
                self._burst_energy[k] += accum_norm * self.energy_burst_factor
                self._burst_energy[k] = min(self._burst_energy[k], 1.0)

    def apply_energy_crosstalk(self, source_levels: dict[str, float], proximity: float):
        """Leak energy from another Voice into this one's smoothed levels.

        Args:
            source_levels: smoothed energy levels from the source Voice
            proximity: 0.0-1.0 frequency proximity (1.0 = same pitch)
        """
        for k in ENERGY_NAMES:
            leak = source_levels.get(k, 0.0) * proximity * 0.02  # 2% base × proximity
            self._energy_smooth[k] = min(1.0, self._energy_smooth[k] + leak)

    def apply_feedback_energy(self, deltas: dict[str, float]):
        """Apply feedback-derived energy deltas to smoothed energy state.

        Distinct from performer injection (which sets target levels) and
        crosstalk (which leaks between Voices based on frequency proximity).
        Feedback closes the loop from audio output back to internal state.

        Args:
            deltas: per-direction energy increments from FeedbackCoupler
        """
        for k in ENERGY_NAMES:
            d = deltas.get(k, 0.0)
            if d != 0.0:
                # Phase unlock: memory feedback strengthened (Phase 8c)
                if k == "memory" and self.state.phase_memory:
                    accum = self.state.energy_accumulation["memory"]
                    threshold = PHASE_THRESHOLDS["memory"]
                    boost = 1.0 + min(1.0, max(0.0, (accum - threshold) / 10.0))
                    d *= boost  # up to 2× for phase_memory
                self._energy_smooth[k] = max(0.0, min(1.0, self._energy_smooth[k] + d))

    def _compute_phase_boosts(self) -> dict[str, float]:
        """Compute per-direction energy gain boosts from phase unlocks.

        Post-transition, each direction's effective energy gain scales
        progressively with accumulation beyond the threshold,
        from 1.0 (at transition) up to 5.0 (at transition + 20s).

        Returns:
            dict of {direction: boost_multiplier} — 1.0 = no boost.
        """
        boosts = {k: 1.0 for k in ENERGY_NAMES}
        phase_map = {
            "tension": self.state.phase_tension,
            "turbulence": self.state.phase_turbulence,
            "resonance": self.state.phase_resonance,
            "memory": self.state.phase_memory,
        }
        for k in ENERGY_NAMES:
            if phase_map[k]:
                accum = self.state.energy_accumulation[k]
                threshold = PHASE_THRESHOLDS[k]
                extra = (accum - threshold) * 0.2  # +1.0 per 5s beyond threshold
                boosts[k] = 1.0 + min(4.0, max(0.0, extra))
        return boosts

    def _check_phase_transitions(self):
        """Check and apply irreversible phase transitions based on accumulation.

        On transition, injects an audible burst cue (~200ms harmonic shimmer)
        so the performer perceives the developmental milestone.
        """
        for direction in ENERGY_NAMES:
            accum = self.state.energy_accumulation[direction]
            threshold = PHASE_THRESHOLDS[direction]
            triggered = False

            if direction == "tension" and not self.state.phase_tension and accum >= threshold:
                self.state.phase_tension = True
                triggered = True
            elif direction == "turbulence" and not self.state.phase_turbulence and accum >= threshold:
                self.state.phase_turbulence = True
                triggered = True
            elif direction == "resonance" and not self.state.phase_resonance and accum >= threshold:
                self.state.phase_resonance = True
                triggered = True
            elif direction == "memory" and not self.state.phase_memory and accum >= threshold:
                self.state.phase_memory = True
                triggered = True

            if triggered:
                # Audible cue: brief harmonic burst on the triggering direction
                self._burst_energy[direction] = min(1.0, self._burst_energy[direction] + 0.5)

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
            # Phase baseline → target; energy dynamics → effective
            target_levels = self._apply_phase_baseline(levels)
            effective_levels = self._apply_energy_dynamics(target_levels)

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
                # Phase-based per-direction gain boosts (Phase 8c)
                boosts = self._compute_phase_boosts()
                energy_tensor = torch.tensor([[
                    effective_levels["tension"] * boosts["tension"],
                    effective_levels["turbulence"] * boosts["turbulence"],
                    effective_levels["resonance"] * boosts["resonance"],
                    effective_levels["memory"] * boosts["memory"],
                ]], device=device) * self.energy_gain
                # Squash to [-1, 1] before hypernetwork to prevent tanh saturation.
                # Effective range [0, ~5] maps to [0, ~0.987]; beyond 5 saturates gracefully.
                energy_tensor = torch.tanh(energy_tensor * 0.5)
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

    def _apply_turbulence_fm(
        self,
        noise_mags: torch.Tensor,
        harm_amps: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate noise magnitudes by harmonic structure (turbulence FM).

        When phase_turbulence is active, noise energy concentrates near
        harmonic peaks — the "边带分叉" (sideband bifurcation) effect.
        Modulation depth scales with turbulence accumulation beyond threshold.

        Args:
            noise_mags: [1, 1, n_magnitudes] — mel-band noise magnitudes
            harm_amps: [1, 1, n_harmonics] — harmonic amplitudes

        Returns:
            noise_mags: [1, 1, n_magnitudes] — FM-modulated
        """
        if not self.state.phase_turbulence:
            return noise_mags

        accum = self.state.energy_accumulation["turbulence"]
        threshold = PHASE_THRESHOLDS["turbulence"]
        mix = min(1.0, max(0.0, (accum - threshold) / 10.0))  # 0→1 over 10s

        if mix < 0.01:
            return noise_mags

        n_h = harm_amps.shape[-1]
        n_m = noise_mags.shape[-1]
        h_split1 = n_h // 3
        h_split2 = 2 * n_h // 3
        m_split1 = n_m // 3
        m_split2 = 2 * n_m // 3

        # 3-band harmonic energy
        h_low = harm_amps[:, :, :h_split1].mean(dim=-1, keepdim=True)
        h_mid = harm_amps[:, :, h_split1:h_split2].mean(dim=-1, keepdim=True)
        h_high = harm_amps[:, :, h_split2:].mean(dim=-1, keepdim=True)

        # Map to rough mel-band regions (split into 3 proportional bands)
        mod = torch.ones_like(noise_mags)
        mod[:, :, :m_split1] = 1.0 + h_low * 2.0 * mix
        mod[:, :, m_split1:m_split2] = 1.0 + h_mid * 2.0 * mix
        mod[:, :, m_split2:] = 1.0 + h_high * 2.0 * mix

        return noise_mags * mod

    def synthesize_from(
        self,
        harm_amps: torch.Tensor,
        noise_mags: torch.Tensor,
        f0_hz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Synthesize audio from (possibly competition-modified) parameters (step 2/2).

        Maintains continuous phase across frames via _harmonic_phase state,
        preventing 250 Hz frame-rate artifacts.

        Args:
            harm_amps: [1, 1, n_harmonics]
            noise_mags: [1, 1, n_magnitudes]
            f0_hz: [1, 1]

        Returns:
            audio: [block_size] numpy array
        """
        with torch.no_grad():
            # Turbulence FM: noise tracks harmonic structure (Phase 8c)
            noise_mags = self._apply_turbulence_fm(noise_mags, harm_amps)

            phase_start = torch.tensor([self._harmonic_phase], device=harm_amps.device, dtype=harm_amps.dtype)
            harm_audio, phase_end = self.harmonic_synth(harm_amps, f0_hz, phase_start=phase_start)
            self._harmonic_phase = phase_end[0].item()
            audio = harm_audio + self.noise_synth(
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

    def set_energy_gain(self, gain: float):
        """Set hypernetwork energy gain multiplier (1.0 = default, 3.0-5.0 = aggressive)."""
        self.energy_gain = max(0.0, min(10.0, gain))

    def reset(self):
        """Reset GRU hidden state, energy module state buffers, noise generator, and dynamics."""
        self._gru_hidden = None
        self.energy_module.reset_state()
        self._noise_gen.manual_seed(self.voice_id + 1)
        self._energy_smooth = {k: 0.0 for k in ENERGY_NAMES}
        self._burst_energy = {k: 0.0 for k in ENERGY_NAMES}
        self._harmonic_phase = 0.0

    def reset_full(self):
        """Full reset including developmental state (destructive)."""
        self.reset()
        self.state = VoiceState(voice_id=self.voice_id)

    def set_note(self, midi_note: int | None, loudness_db: float = -10.0):
        """Set the currently active note for this Voice."""
        self.state.active_note = midi_note
        if midi_note is not None:
            self.state.active_loudness = loudness_db
            self.reset()  # fresh start for new note (clears GRU, smooth, burst)
