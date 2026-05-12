import torch
import pytest
from synth.nn.model import RichParamModel


class TestRichParamModel:
    @pytest.fixture
    def model(self):
        return RichParamModel(
            sample_rate=16000,
            block_size=64,
            table_size=2048,
            transformer_dim=256,
            transformer_heads=4,
            transformer_layers=3,
            gru_hidden=512,
            n_harmonics=256,
            n_noise_mel=65,
            n_noise_grain=65,
            beta_max=0.02,
        )

    def test_output_shape(self, model):
        B, T = 2, 1
        f0_hz = torch.ones(B, T) * 440.0
        loudness = torch.randn(B, T) * 10 - 20
        audio = model(f0_hz, loudness)
        expected_samples = T * model.block_size
        assert audio.shape == (B, expected_samples)

    def test_output_finite(self, model):
        B, T = 2, 1
        f0_hz = torch.ones(B, T) * 440.0
        loudness = torch.randn(B, T) * 10 - 20
        audio = model(f0_hz, loudness)
        assert not torch.isnan(audio).any()
        assert not torch.isinf(audio).any()

    def test_gradient_flow(self, model):
        B, T = 2, 1
        f0_hz = torch.ones(B, T) * 440.0
        loudness = torch.randn(B, T) * 10 - 20
        audio = model(f0_hz, loudness)
        loss = audio.abs().mean()
        loss.backward()
        for name, param in model.decoder.named_parameters():
            assert param.grad is not None, f"decoder.{name} has no gradient"

    def test_count_parameters(self, model):
        info = model.count_parameters()
        assert 3_000_000 < info["decoder"] < 7_000_000, (
            f"Expected 5M params, got {info['decoder']}"
        )
        assert info["total"] == info["decoder"]

    def test_output_not_constant(self, model):
        model.eval()
        B, T = 1, 1
        f0_hz = torch.ones(B, T) * 440.0
        loudness = torch.zeros(B, T)
        with torch.no_grad():
            audio = model(f0_hz, loudness)
        assert audio.abs().max() > 0.01, "Expected non-silent output"
