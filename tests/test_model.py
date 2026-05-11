import torch
import pytest
from synth.nn.model import DDSPModel


class TestDDSPModel:
    @pytest.fixture
    def model(self):
        return DDSPModel(
            hidden_size=180,
            n_harmonics=100,
            n_magnitudes=65,
            sample_rate=16000,
            block_size=64,
            table_size=2048,
        )

    def test_output_shape(self, model):
        B, T = 2, 250
        f0_hz = torch.ones(B, T) * 440.0
        loudness = torch.randn(B, T) * 10 - 20
        audio = model(f0_hz, loudness)
        expected_samples = T * model.block_size
        assert audio.shape == (B, expected_samples)

    def test_output_finite(self, model):
        B, T = 2, 50
        f0_hz = torch.ones(B, T) * 440.0
        loudness = torch.randn(B, T) * 10 - 20
        audio = model(f0_hz, loudness)
        assert not torch.isnan(audio).any()
        assert not torch.isinf(audio).any()

    def test_gradient_flow(self, model):
        B, T = 2, 20
        f0_hz = torch.ones(B, T) * 440.0
        loudness = torch.randn(B, T) * 10 - 20
        audio = model(f0_hz, loudness)
        loss = audio.abs().mean()
        loss.backward()
        for name, param in model.decoder.named_parameters():
            assert param.grad is not None, f"decoder.{name} has no gradient"

    def test_count_parameters(self, model):
        info = model.count_parameters()
        assert 200_000 < info["decoder"] < 300_000
        assert info["total"] == info["decoder"]

    def test_synthesize_with_gain(self, model):
        B, T = 1, 10
        f0_hz = torch.ones(B, T) * 440.0
        loudness = torch.zeros(B, T)
        # Seed for deterministic noise
        torch.manual_seed(42)
        full = model.synthesize(f0_hz, loudness, harmonic_gain=1.0, noise_gain=1.0)
        torch.manual_seed(42)
        harm_only = model.synthesize(f0_hz, loudness, harmonic_gain=1.0, noise_gain=0.0)
        torch.manual_seed(42)
        noise_only = model.synthesize(f0_hz, loudness, harmonic_gain=0.0, noise_gain=1.0)
        assert not torch.allclose(harm_only, full)
        assert not torch.allclose(noise_only, full)
        recon = harm_only + noise_only
        assert torch.allclose(recon, full, atol=1e-5)

    def test_varying_f0(self, model):
        B, T = 1, 50
        f0_hz = torch.linspace(100, 800, T).unsqueeze(0)
        loudness = torch.randn(B, T) * 10 - 20
        audio = model(f0_hz, loudness)
        assert not torch.isnan(audio).any()
