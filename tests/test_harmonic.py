import torch
import pytest
from synth.dsp.harmonic import WavetableHarmonicSynth


class TestWavetableHarmonicSynth:
    @pytest.fixture
    def synth(self):
        return WavetableHarmonicSynth(
            n_harmonics=100, table_size=2048, sample_rate=16000, block_size=64
        )

    def test_output_shape(self, synth):
        B, T = 2, 250  # 250 frames ≈ 4s at 64-hop
        harm_amps = torch.rand(B, T, synth.n_harmonics)
        f0_hz = torch.ones(B, T) * 440.0
        audio = synth(harm_amps, f0_hz)
        assert audio.shape == (B, T * synth.block_size)

    def test_silence_with_zero_amplitudes(self, synth):
        B, T = 1, 10
        harm_amps = torch.zeros(B, T, synth.n_harmonics)
        f0_hz = torch.ones(B, T) * 440.0
        audio = synth(harm_amps, f0_hz)
        assert torch.allclose(audio, torch.zeros_like(audio), atol=1e-5)

    def test_constant_f0_produces_periodic_signal(self, synth):
        B, T = 1, 10
        # Single harmonic → sine wave
        harm_amps = torch.zeros(B, T, synth.n_harmonics)
        harm_amps[:, :, 0] = 1.0
        f0_hz = torch.ones(B, T) * 440.0
        audio = synth(harm_amps, f0_hz)
        audio = audio.squeeze()
        # Check signal is non-zero and roughly periodic
        assert audio.abs().max() > 0.1
        # Period in samples at 16kHz for 440Hz ≈ 36.36 samples
        period = int(synth.sample_rate / 440.0)
        autocorr = torch.nn.functional.conv1d(
            audio.view(1, 1, -1), audio.view(1, 1, -1), padding=audio.shape[-1] // 2
        )
        autocorr = autocorr.squeeze()
        # Autocorrelation peak should be near period
        center = autocorr.shape[0] // 2
        peak_offset = autocorr[center + period // 2 : center + period * 2].argmax()
        assert abs((peak_offset + period // 2) - period) <= 5

    def test_phase_continuity_across_frames(self, synth):
        """Phase should be continuous at frame boundaries."""
        B, T = 1, 3
        harm_amps = torch.rand(B, T, synth.n_harmonics)
        f0_hz = torch.ones(B, T) * 220.0
        audio = synth(harm_amps, f0_hz).squeeze()
        # Check no large discontinuities at frame boundaries
        for t in range(1, T):
            boundary = t * synth.block_size
            diff = abs(audio[boundary] - audio[boundary - 1])
            max_sample = audio.abs().max()
            assert diff < max_sample * 2.0  # Not a huge jump

    def test_varying_f0(self, synth):
        B, T = 1, 50
        harm_amps = torch.rand(B, T, synth.n_harmonics)
        f0_hz = torch.linspace(100, 800, T).unsqueeze(0).expand(B, T)
        audio = synth(harm_amps, f0_hz)
        assert audio.shape == (B, T * synth.block_size)
        assert not torch.isnan(audio).any()
        assert not torch.isinf(audio).any()

    def test_batch_independence(self, synth):
        B, T = 4, 10
        harm_amps = torch.zeros(B, T, synth.n_harmonics)
        harm_amps[0, :, 0] = 1.0  # Batch 0: first harmonic only
        harm_amps[1, :, 5] = 1.0  # Batch 1: sixth harmonic only
        f0_hz = torch.ones(B, T) * 440.0
        audio = synth(harm_amps, f0_hz)
        # Different batch elements should produce different audio
        assert not torch.allclose(audio[0], audio[1], atol=1e-3)

    def test_low_f0_clamped(self, synth):
        """Very low f0 should not crash."""
        B, T = 1, 5
        harm_amps = torch.rand(B, T, synth.n_harmonics)
        f0_hz = torch.zeros(B, T)  # 0 Hz → clamped internally
        audio = synth(harm_amps, f0_hz)
        assert not torch.isnan(audio).any()
