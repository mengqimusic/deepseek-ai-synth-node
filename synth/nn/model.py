import torch
import torch.nn as nn

from synth.dsp.harmonic import WavetableHarmonicSynth
from synth.dsp.noise import FilteredNoiseSynth
from synth.dsp.processors import scale_f0
from synth.nn.decoder import DDSPDecoder


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
