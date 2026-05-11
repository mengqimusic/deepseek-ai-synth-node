import torch
import pytest
from synth.nn.decoder import DDSPDecoder


class TestDDSPDecoder:
    @pytest.fixture
    def decoder(self):
        return DDSPDecoder(
            input_size=2, hidden_size=180, n_harmonics=100, n_magnitudes=65
        )

    def test_output_shapes(self, decoder):
        B, T = 4, 250
        f0_scaled = torch.rand(B, T)
        loudness = torch.randn(B, T) * 20 - 30
        harm_amps, noise_mags = decoder(f0_scaled, loudness)
        assert harm_amps.shape == (B, T, 100)
        assert noise_mags.shape == (B, T, 65)

    def test_output_range(self, decoder):
        B, T = 2, 50
        f0_scaled = torch.rand(B, T)
        loudness = torch.randn(B, T)
        harm_amps, noise_mags = decoder(f0_scaled, loudness)
        assert (harm_amps >= 0).all() and (harm_amps <= 1).all()
        assert (noise_mags >= 0).all() and (noise_mags <= 1).all()

    def test_parameter_count(self, decoder):
        n = decoder.count_parameters()
        assert 200_000 < n < 300_000, f"Expected ~258K params, got {n}"

    def test_gradient_flow(self, decoder):
        B, T = 4, 100
        f0_scaled = torch.rand(B, T, requires_grad=True)
        loudness = torch.randn(B, T, requires_grad=True)
        harm_amps, noise_mags = decoder(f0_scaled, loudness)
        loss = harm_amps.sum() + noise_mags.sum()
        loss.backward()
        for name, param in decoder.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"

    def test_deterministic_in_eval(self, decoder):
        decoder.eval()
        B, T = 1, 10
        f0_scaled = torch.ones(B, T) * 0.5
        loudness = torch.zeros(B, T)
        with torch.no_grad():
            out1 = decoder(f0_scaled, loudness)
            out2 = decoder(f0_scaled, loudness)
        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])
