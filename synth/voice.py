import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn

from synth.energy import EnergyBiasModule
from synth.dsp.processors import scale_f0
from synth.dsp.formant import FormantFilter

ENERGY_NAMES = ["tension", "turbulence", "resonance", "memory"]

PHASE_THRESHOLDS = {
    "tension": 5.0,
    "turbulence": 5.0,
    "resonance": 5.0,
    "memory": 5.0,
}

PHASE_BASELINE = 0.08


@dataclass
class VoiceState:
    """Per-Voice developmental state. Deterministic given same inputs."""

    voice_id: int

    energy_accumulation: dict[str, float] = field(default_factory=lambda: {
        k: 0.0 for k in ENERGY_NAMES
    })

    phase_tension: bool = False
    phase_turbulence: bool = False
    phase_resonance: bool = False
    phase_memory: bool = False

    competition_weight: float = 1.0
    withdrawal_low: float = 0.5
    withdrawal_mid: float = 0.5
    withdrawal_high: float = 0.5

    active_note: int | None = None
    active_loudness: float = -10.0

    @property
    def phase_count(self) -> int:
        return sum([self.phase_tension, self.phase_turbulence,
                     self.phase_resonance, self.phase_memory])

    @property
    def dominant_direction(self) -> str | None:
        if all(v == 0.0 for v in self.energy_accumulation.values()):
            return None
        return max(self.energy_accumulation, key=self.energy_accumulation.get)


class VoiceModule(nn.Module):
    """
    A single Voice with independent developmental history.

    Shares decoder and DSP synth weights with other Voices but maintains
    its own transformer hidden state, EnergyBiasModule state buffers,
    and developmental state.

    Uses RichParamDecoder (256 harm + per-harm β + formant + transient +
    FM + 65 mel noise + 65 grain noise) with optional ModulatedRichDecoder
    for hypernetwork-driven weight injection.
    """

    def __init__(
        self,
        voice_id: int,
        decoder: nn.Module,
        harmonic_synth: nn.Module,
        noise_synth: nn.Module,
        n_harmonics: int = 256,
        n_magnitudes: int = 65,
        sample_rate: int = 16000,
        block_size: int = 64,
        modulated_decoder: nn.Module | None = None,
        energy_gain: float = 1.0,
        fm_synth: nn.Module | None = None,
        grain_synth: nn.Module | None = None,
        transient_synth: nn.Module | None = None,
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

        # Rich mode synths
        self.fm_synth = fm_synth
        self.grain_synth = grain_synth
        self.transient_synth = transient_synth

        self.energy_module = EnergyBiasModule(
            n_harmonics=n_harmonics,
            n_magnitudes=n_magnitudes,
            sample_rate=sample_rate,
            block_size=block_size,
        )

        if hasattr(decoder, 'n_harmonics'):
            if decoder.n_harmonics != n_harmonics:
                raise ValueError(
                    f"Decoder outputs {decoder.n_harmonics} harmonics "
                    f"but VoiceModule got n_harmonics={n_harmonics}"
                )

        self.state = VoiceState(voice_id=voice_id)
        self._gru_hidden = None  # [1, 1, hidden_size] or None
        self._xf_buffer = None   # [1, ctx, D] — transformer context (rich mode)
        self._noise_gen = torch.Generator()
        self._noise_gen.manual_seed(voice_id + 1)
        self._grain_gen = torch.Generator()
        self._grain_gen.manual_seed(voice_id + 100)

        # Energy dynamics
        self.energy_attack_ms = 30.0
        self.energy_release_ms = 150.0
        self.energy_burst_factor = 0.3
        self.energy_burst_decay_ms = 100.0
        self._energy_smooth = {k: 0.0 for k in ENERGY_NAMES}
        self._burst_energy = {k: 0.0 for k in ENERGY_NAMES}
        self._attack_alpha = 1.0 - math.exp(-(block_size / sample_rate) / (self.energy_attack_ms / 1000.0))
        self._release_alpha = 1.0 - math.exp(-(block_size / sample_rate) / (self.energy_release_ms / 1000.0))
        self._burst_decay = math.exp(-(block_size / sample_rate) / (self.energy_burst_decay_ms / 1000.0))
        self._harmonic_phase: float = 0.0

        # Inharmonic synthesis state
        self._harmonic_phase_inharmonic: torch.Tensor | None = None  # [n_harmonics] radians

        # Formant filter
        self.formant_filter = FormantFilter(
            n_bands=3, sample_rate=sample_rate,
        )

        # Decoder-generated params (rich mode — stored between process_params and synthesize_from)
        self._decoder_inharm_beta: torch.Tensor | None = None
        self._decoder_formant: torch.Tensor | None = None
        self._decoder_transient: torch.Tensor | None = None
        self._decoder_fm: torch.Tensor | None = None
        self._decoder_noise_grain: torch.Tensor | None = None

    def _apply_phase_baseline(self, levels: dict[str, float]) -> dict[str, float]:
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
        result = {}
        for k in ENERGY_NAMES:
            target = levels.get(k, 0.0)
            current = self._energy_smooth[k]
            alpha = self._attack_alpha if target > current else self._release_alpha
            smoothed = current + alpha * (target - current)
            result[k] = max(0.0, min(1.0, smoothed + self._burst_energy[k]))
            self._energy_smooth[k] = smoothed
            self._burst_energy[k] *= self._burst_decay
        return result

    def inject_burst(self, target_levels: dict[str, float] | None = None):
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
        for k in ENERGY_NAMES:
            leak = source_levels.get(k, 0.0) * proximity * 0.02
            self._energy_smooth[k] = min(1.0, self._energy_smooth[k] + leak)

    def apply_feedback_energy(self, deltas: dict[str, float]):
        for k in ENERGY_NAMES:
            d = deltas.get(k, 0.0)
            if d != 0.0:
                if k == "memory" and self.state.phase_memory:
                    accum = self.state.energy_accumulation["memory"]
                    threshold = PHASE_THRESHOLDS["memory"]
                    boost = 1.0 + min(1.0, max(0.0, (accum - threshold) / 10.0))
                    d *= boost
                self._energy_smooth[k] = max(0.0, min(1.0, self._energy_smooth[k] + d))

    def _compute_phase_boosts(self) -> dict[str, float]:
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
                extra = (accum - threshold) * 0.2
                boosts[k] = 1.0 + min(4.0, max(0.0, extra))
        return boosts

    def _check_phase_transitions(self):
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
                self._burst_energy[direction] = min(1.0, self._burst_energy[direction] + 0.5)

    def _update_competition_profile(self):
        acc = self.state.energy_accumulation
        total = sum(acc.values())

        self.state.competition_weight = 1.0 + total * 0.1 + self.state.phase_count * 0.3

        if total > 0:
            t_frac = acc["tension"] / total
            u_frac = acc["turbulence"] / total
            r_frac = acc["resonance"] / total
            m_frac = acc["memory"] / total

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
    ) -> dict:
        """
        Generate synthesis parameters for this Voice.

        Returns dict with keys: harm_amps, noise_mags, f0_hz, inharm_beta,
        formant, transient, fm, noise_grain, plus competition/withdrawal fields.

        Args:
            f0_hz: fundamental frequency in Hz
            loudness_db: loudness in dB
            levels: per-direction energy injection levels {name: float}

        Returns:
            dict with synthesis parameters
        """
        device = next(self.decoder.parameters()).device

        with torch.no_grad():
            target_levels = self._apply_phase_baseline(levels)
            effective_levels = self._apply_energy_dynamics(target_levels)

            for k in ENERGY_NAMES:
                self.state.energy_accumulation[k] += effective_levels[k] * self.frame_duration

            self._check_phase_transitions()
            self._update_competition_profile()

            f0 = torch.tensor([[f0_hz]], device=device)
            loudness = torch.tensor([[loudness_db]], device=device)
            f0_scaled = scale_f0(f0)

            if self.modulated_decoder is not None:
                boosts = self._compute_phase_boosts()
                energy_tensor = torch.tensor([[
                    effective_levels["tension"] * boosts["tension"],
                    effective_levels["turbulence"] * boosts["turbulence"],
                    effective_levels["resonance"] * boosts["resonance"],
                    effective_levels["memory"] * boosts["memory"],
                ]], device=device) * self.energy_gain
                energy_tensor = torch.tanh(energy_tensor * 0.5)

                outputs, self._xf_buffer, self._gru_hidden = self.modulated_decoder.forward_step(
                    f0_scaled, loudness, energy_tensor,
                    xf_buffer=self._xf_buffer,
                    gru_hidden=self._gru_hidden,
                )
            else:
                outputs, self._xf_buffer, self._gru_hidden = self.decoder.forward_step(
                    f0_scaled, loudness,
                    xf_buffer=self._xf_buffer,
                    gru_hidden=self._gru_hidden,
                )
            harm = outputs["harm_amps"]
            noise_mel = outputs["noise_mel"]

            self._decoder_inharm_beta = outputs["inharm_beta"]
            self._decoder_formant = outputs["formant"]
            self._decoder_transient = outputs["transient"]
            self._decoder_fm = outputs["fm"]
            self._decoder_noise_grain = outputs["noise_grain"]

            # Energy bias post-processing
            harm, noise_mel = self.energy_module(harm, noise_mel, effective_levels)

            return {
                "harm_amps": harm,
                "noise_mags": noise_mel,
                "f0_hz": f0_hz,
                "inharm_beta": self._decoder_inharm_beta,
                "formant": self._decoder_formant,
                "transient": self._decoder_transient,
                "fm": self._decoder_fm,
                "noise_grain": self._decoder_noise_grain,
                "competition_weight": self.state.competition_weight,
                "withdrawal_low": self.state.withdrawal_low,
                "withdrawal_mid": self.state.withdrawal_mid,
                "withdrawal_high": self.state.withdrawal_high,
                "is_active": True,
            }

    def _apply_turbulence_fm(
        self,
        noise_mags: torch.Tensor,
        harm_amps: torch.Tensor,
    ) -> torch.Tensor:
        if not self.state.phase_turbulence:
            return noise_mags

        accum = self.state.energy_accumulation["turbulence"]
        threshold = PHASE_THRESHOLDS["turbulence"]
        mix = min(1.0, max(0.0, (accum - threshold) / 10.0))

        if mix < 0.01:
            return noise_mags

        n_h = harm_amps.shape[-1]
        n_m = noise_mags.shape[-1]
        h_split1 = n_h // 3
        h_split2 = 2 * n_h // 3
        m_split1 = n_m // 3
        m_split2 = 2 * n_m // 3

        h_low = harm_amps[:, :, :h_split1].mean(dim=-1, keepdim=True)
        h_mid = harm_amps[:, :, h_split1:h_split2].mean(dim=-1, keepdim=True)
        h_high = harm_amps[:, :, h_split2:].mean(dim=-1, keepdim=True)

        mod = torch.ones_like(noise_mags)
        mod[:, :, :m_split1] = 1.0 + h_low * 2.0 * mix
        mod[:, :, m_split1:m_split2] = 1.0 + h_mid * 2.0 * mix
        mod[:, :, m_split2:] = 1.0 + h_high * 2.0 * mix

        return noise_mags * mod

    def _map_formant_freqs(self, raw: torch.Tensor) -> torch.Tensor:
        """Map sigmoid [0,1] formant params to Hz ranges."""
        # raw: [1, 1, 3] — sigmoid outputs for f1, f2, f3
        # f1: 200-800, f2: 500-2500, f3: 1500-4000
        lo = torch.tensor([200., 500., 1500.], device=raw.device, dtype=raw.dtype)
        hi = torch.tensor([800., 2500., 4000.], device=raw.device, dtype=raw.dtype)
        return lo + raw * (hi - lo)

    def _map_formant_qs(self, raw: torch.Tensor) -> torch.Tensor:
        """Map sigmoid [0,1] to Q range [2, 20]."""
        return 2.0 + raw * 18.0

    def synthesize_from(self, params: dict) -> torch.Tensor:
        """
        Synthesize audio from (possibly competition-modified) parameters.

        Args:
            params: dict from process_params with keys:
                harm_amps, noise_mags, f0_hz, inharm_beta, formant,
                transient, fm, noise_grain

        Returns:
            audio: [block_size] numpy array
        """
        with torch.no_grad():
            harm_amps = params["harm_amps"]
            noise_mags = params["noise_mags"]
            f0 = params["f0_hz"]
            if isinstance(f0, (int, float)):
                f0 = torch.tensor([[f0]], device=harm_amps.device, dtype=harm_amps.dtype)

            # Turbulence FM
            noise_mags = self._apply_turbulence_fm(noise_mags, harm_amps)

            # Harmonic synthesis
            inharm_beta = params.get("inharm_beta", None)
            use_inharmonic = inharm_beta is not None and inharm_beta.abs().max() > 0

            if use_inharmonic:
                n_h = harm_amps.shape[-1]
                device = harm_amps.device
                dtype = harm_amps.dtype

                if self._harmonic_phase_inharmonic is None:
                    bridge_rad = self._harmonic_phase / self.harmonic_synth.table_size * (2.0 * math.pi)
                    self._harmonic_phase_inharmonic = torch.full(
                        (n_h,), bridge_rad, device=device, dtype=dtype
                    )

                phase_start = self._harmonic_phase_inharmonic.unsqueeze(0)
                harm_audio, phase_end = self.harmonic_synth(
                    harm_amps, f0,
                    phase_start=phase_start,
                    inharmonicity=inharm_beta,
                )
                self._harmonic_phase_inharmonic = phase_end[0]
                self._harmonic_phase = (
                    phase_end[0, 0].item() / (2.0 * math.pi) * self.harmonic_synth.table_size
                ) % self.harmonic_synth.table_size
            else:
                if self._harmonic_phase_inharmonic is not None:
                    self._harmonic_phase = (
                        self._harmonic_phase_inharmonic[0].item() / (2.0 * math.pi) * self.harmonic_synth.table_size
                    ) % self.harmonic_synth.table_size
                    self._harmonic_phase_inharmonic = None

                phase_start = torch.tensor(
                    [self._harmonic_phase], device=harm_amps.device, dtype=harm_amps.dtype
                )
                harm_audio, phase_end = self.harmonic_synth(harm_amps, f0, phase_start=phase_start)
                self._harmonic_phase = phase_end[0].item()

            # Formant filter (decoder-driven)
            formant_params = params.get("formant", None)
            if formant_params is not None:
                raw = formant_params  # [1, 1, 6]
                freqs = self._map_formant_freqs(raw[..., :3]).squeeze(0)  # [1, 3]
                qs = self._map_formant_qs(raw[..., 3:]).squeeze(0)
                harm_audio = self.formant_filter.forward_explicit(harm_audio, freqs, qs)

            # FM synth
            fm_params = params.get("fm", None)
            fm_audio = None
            if fm_params is not None and self.fm_synth is not None:
                fm_audio = self.fm_synth(fm_params, f0)

            # Mel noise
            noise_audio = self.noise_synth(noise_mags, generator=self._noise_gen)

            # Grain noise
            grain_params = params.get("noise_grain", None)
            grain_audio = None
            if grain_params is not None and self.grain_synth is not None:
                grain_audio = self.grain_synth(grain_params, generator=self._grain_gen)

            # Transient comb noise
            transient_params = params.get("transient", None)
            transient_audio = None
            if transient_params is not None and self.transient_synth is not None:
                transient_audio = self.transient_synth(transient_params)

            # Sum
            audio = harm_audio
            if fm_audio is not None:
                audio = audio + fm_audio
            audio = audio + noise_audio
            if grain_audio is not None:
                audio = audio + grain_audio
            if transient_audio is not None:
                audio = audio + transient_audio

            return audio.squeeze(0).squeeze(0).cpu().numpy().astype("float32")

    def process_frame(
        self,
        f0_hz: float,
        loudness_db: float,
        levels: dict[str, float],
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Generate one frame of audio for this Voice (combined 1+2, no competition)."""
        params = self.process_params(f0_hz, loudness_db, levels)
        audio = self.synthesize_from(params)
        harm_out = params["harm_amps"].squeeze(0).squeeze(0)
        noise_out = params["noise_mags"].squeeze(0).squeeze(0)
        return audio, harm_out, noise_out

    def set_energy_gain(self, gain: float):
        self.energy_gain = max(0.0, min(10.0, gain))

    def reset(self):
        self._gru_hidden = None
        self._xf_buffer = None
        self.energy_module.reset_state()
        self._noise_gen.manual_seed(self.voice_id + 1)
        self._grain_gen.manual_seed(self.voice_id + 100)
        self._energy_smooth = {k: 0.0 for k in ENERGY_NAMES}
        self._burst_energy = {k: 0.0 for k in ENERGY_NAMES}
        self._harmonic_phase = 0.0
        self._harmonic_phase_inharmonic = None
        self._decoder_inharm_beta = None
        self._decoder_formant = None
        self._decoder_transient = None
        self._decoder_fm = None
        self._decoder_noise_grain = None

    def reset_full(self):
        self.reset()
        self.state = VoiceState(voice_id=self.voice_id)

    def set_note(self, midi_note: int | None, loudness_db: float = -10.0):
        self.state.active_note = midi_note
        if midi_note is not None:
            self.state.active_loudness = loudness_db
            self.reset()
