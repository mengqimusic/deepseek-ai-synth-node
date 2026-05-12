import torch
import torch.nn as nn

from synth.dsp.harmonic import WavetableHarmonicSynth
from synth.dsp.noise import FilteredNoiseSynth, GrainNoiseSynth
from synth.dsp.fm import FMSynth
from synth.dsp.transient import TransientCombNoise
from synth.dsp.formant import FormantFilter
from synth.dsp.processors import scale_f0
from synth.nn.decoder import DDSPDecoder
from synth.nn.transformer_decoder import RichParamDecoder


class DDSPModel(nn.Module):
    """
    Full DDSP autoencoder (decoder-only for Phase 1).

    Wires together:
        DDSPDecoder (f0 + loudness → control params)
        WavetableHarmonicSynth (harmonic_amps → harmonic audio)
        FilteredNoiseSynth (noise_mags → noise audio)
    """

    def __init__(
        self,
        hidden_size: int = 180,
        n_harmonics: int = 100,
        n_magnitudes: int = 65,
        sample_rate: int = 16000,
        block_size: int = 64,
        table_size: int = 2048,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.decoder = DDSPDecoder(
            input_size=2,
            hidden_size=hidden_size,
            n_harmonics=n_harmonics,
            n_magnitudes=n_magnitudes,
        )
        self.harmonic_synth = WavetableHarmonicSynth(
            n_harmonics=n_harmonics,
            table_size=table_size,
            sample_rate=sample_rate,
            block_size=block_size,
        )
        self.noise_synth = FilteredNoiseSynth(
            n_magnitudes=n_magnitudes,
            sample_rate=sample_rate,
            block_size=block_size,
        )

    def forward(self, f0_hz: torch.Tensor, loudness: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0_hz:   [B, T] — fundamental frequency in Hz per frame
            loudness: [B, T] — loudness in dB per frame

        Returns:
            audio: [B, T * block_size] — synthesized audio waveform
        """
        f0_scaled = scale_f0(f0_hz)
        harm_amps, noise_mags = self.decoder(f0_scaled, loudness)
        harm_audio = self.harmonic_synth(harm_amps, f0_hz)
        noise_audio = self.noise_synth(noise_mags)
        return harm_audio + noise_audio

    @torch.no_grad()
    def synthesize(
        self,
        f0_hz: torch.Tensor,
        loudness: torch.Tensor,
        harmonic_gain: float = 1.0,
        noise_gain: float = 1.0,
    ) -> torch.Tensor:
        """
        Synthesize audio with controllable harmonic/noise mix.

        Useful for evaluation and listening tests.
        """
        f0_scaled = scale_f0(f0_hz)
        harm_amps, noise_mags = self.decoder(f0_scaled, loudness)
        harm_audio = self.harmonic_synth(harm_amps, f0_hz) * harmonic_gain
        noise_audio = self.noise_synth(noise_mags) * noise_gain
        return harm_audio + noise_audio

    def count_parameters(self) -> dict[str, int]:
        decoder_params = self.decoder.count_parameters()
        total = decoder_params  # harmonic/noise synths have no learnable parameters
        return {"decoder": decoder_params, "total": total}


class RichParamModel(nn.Module):
    """
    Phase 10a full rich-parameter model.

    Wires RichParamDecoder to all DSP synths for differentiable end-to-end
    training. Gradient flows from audio loss through all synthesis buses
    back to the decoder.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 64,
        table_size: int = 2048,
        transformer_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 3,
        gru_hidden: int = 512,
        n_harmonics: int = 256,
        n_noise_mel: int = 65,
        n_noise_grain: int = 65,
        beta_max: float = 0.02,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.n_harmonics = n_harmonics

        self.decoder = RichParamDecoder(
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            gru_hidden=gru_hidden,
            n_harmonics=n_harmonics,
            n_noise_mel=n_noise_mel,
            n_noise_grain=n_noise_grain,
            beta_max=beta_max,
        )
        self.harmonic_synth = WavetableHarmonicSynth(
            n_harmonics=n_harmonics,
            table_size=table_size,
            sample_rate=sample_rate,
            block_size=block_size,
        )
        self.formant_filter = FormantFilter(
            n_bands=3, sample_rate=sample_rate,
        )
        self.fm_synth = FMSynth(
            sample_rate=sample_rate, block_size=block_size,
        )
        self.noise_synth = FilteredNoiseSynth(
            n_magnitudes=n_noise_mel,
            sample_rate=sample_rate,
            block_size=block_size,
        )
        self.grain_synth = GrainNoiseSynth(
            n_bands=n_noise_grain,
            sample_rate=sample_rate,
            block_size=block_size,
        )
        self.transient_synth = TransientCombNoise(
            sample_rate=sample_rate, block_size=block_size,
        )

    def forward(self, f0_hz: torch.Tensor, loudness: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0_hz:   [B, T] fundamental frequency in Hz
            loudness: [B, T] loudness in dB

        Returns:
            audio: [B, T * block_size]
        """
        f0_scaled = scale_f0(f0_hz)
        outputs = self.decoder(f0_scaled, loudness)

        # Harmonic synthesis (wavetable or inharmonic)
        inharm_beta = outputs["inharm_beta"]
        harm_audio = self.harmonic_synth(
            outputs["harm_amps"], f0_hz, inharmonicity=inharm_beta
        )

        # Formant filter (decoder-driven)
        formant = outputs["formant"]  # [B, T, 6]
        B, T = f0_hz.shape
        freqs_raw = self._map_formant_freqs(formant[..., :3])  # [B, T, 3]
        qs_raw = self._map_formant_qs(formant[..., 3:])        # [B, T, 3]

        # Reshape for formant filter: [B, T*S]
        audio_2d = harm_audio  # [B, T * block_size]
        freqs_2d = freqs_raw.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        freqs_2d = freqs_2d.reshape(B, T * self.block_size, 3).mean(dim=1)  # [B, 3]
        qs_2d = qs_raw.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        qs_2d = qs_2d.reshape(B, T * self.block_size, 3).mean(dim=1)  # [B, 3]

        self.formant_filter.reset()
        harm_filtered = self.formant_filter.forward_explicit(audio_2d, freqs_2d, qs_2d)

        # FM synthesis
        fm_audio = self.fm_synth(outputs["fm"], f0_hz)

        # Noise buses
        noise_audio = self.noise_synth(outputs["noise_mel"])
        grain_audio = self.grain_synth(outputs["noise_grain"])
        transient_audio = self.transient_synth(outputs["transient"])

        return harm_filtered + fm_audio + noise_audio + grain_audio + transient_audio

    def _map_formant_freqs(self, raw: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor([200., 500., 1500.], device=raw.device, dtype=raw.dtype)
        hi = torch.tensor([800., 2500., 4000.], device=raw.device, dtype=raw.dtype)
        return lo + raw * (hi - lo)

    def _map_formant_qs(self, raw: torch.Tensor) -> torch.Tensor:
        return 2.0 + raw * 18.0

    def count_parameters(self) -> dict[str, int]:
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        return {"decoder": decoder_params, "total": decoder_params}
