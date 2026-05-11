import torch
import pytest
from synth.dsp.noise import FilteredNoiseSynth


class TestFilteredNoiseSynth:
    @pytest.fixture
    def synth(self):
        return FilteredNoiseSynth(
            n_magnitudes=65, sample_rate=16000, block_size=64
        )

    def test_output_shape(self, synth):
        B, T = 2, 250
        noise_mags = torch.rand(B, T, synth.n_magnitudes)
        audio = synth(noise_mags)
        assert audio.shape == (B, T * synth.block_size)

    def test_zero_magnitudes_produce_silence(self, synth):
        B, T = 1, 10
        noise_mags = torch.zeros(B, T, synth.n_magnitudes)
        audio = synth(noise_mags)
        assert torch.allclose(audio, torch.zeros_like(audio), atol=1e-5)

    def test_output_is_finite(self, synth):
        B, T = 2, 50
        noise_mags = torch.rand(B, T, synth.n_magnitudes)
        audio = synth(noise_mags)
        assert not torch.isnan(audio).any()
        assert not torch.isinf(audio).any()

    def test_same_input_same_output(self, synth):
        """Deterministic with same random seed."""
        B, T = 1, 10
        noise_mags = torch.ones(B, T, synth.n_magnitudes) * 0.5
        torch.manual_seed(42)
        audio1 = synth(noise_mags)
        torch.manual_seed(42)
        audio2 = synth(noise_mags)
        assert torch.allclose(audio1, audio2)

    def test_flat_magnitudes_produce_broadband(self, synth):
        """Flat magnitudes should produce noise with energy across spectrum."""
        B, T = 1, 10
        noise_mags = torch.ones(B, T, synth.n_magnitudes) * 0.5
        torch.manual_seed(0)
        audio = synth(noise_mags).squeeze()
        # Compute spectrum
        spec = torch.abs(torch.fft.rfft(audio))
        # Should have energy distributed across bins (not just DC)
        nonzero_bins = (spec > 0.001).sum()
        assert nonzero_bins > len(spec) * 0.5  # At least half of bins

    def test_batch_independence(self, synth):
        B, T = 4, 10
        noise_mags = torch.zeros(B, T, synth.n_magnitudes)
        noise_mags[0, :, :] = 1.0
        noise_mags[1, :, :] = 0.0
        torch.manual_seed(0)
        audio = synth(noise_mags)
        assert not torch.allclose(audio[0], audio[1], atol=1e-3)

    def test_mel_filterbank_shape(self, synth):
        """mel_to_linear matrix should have correct shape."""
        assert synth.mel_to_linear.shape == (synth.n_bins, synth.n_magnitudes)
        # Most rows should sum to ~1.0 (Nyquist bin may be 0)
        row_sums = synth.mel_to_linear.sum(dim=1)
        nonzero_rows = row_sums > 0.01
        assert nonzero_rows.float().mean() > 0.95  # >95% of rows non-empty
        row_sums_nz = row_sums[nonzero_rows]
        assert torch.allclose(row_sums_nz, torch.ones_like(row_sums_nz), atol=1e-1)
