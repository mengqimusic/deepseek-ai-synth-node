import time
import torch
import numpy as np


@torch.no_grad()
def measure_latency(model, device: str = "cpu", n_warmup: int = 50, n_timed: int = 500) -> dict:
    """
    Measure per-frame inference latency breakdown.

    Args:
        model: RichParamModel
        device: "cpu" or "cuda"
        n_warmup: warmup frames (discarded)
        n_timed: timed frames

    Returns:
        dict with per-component timing in ms and real-time factor
    """
    model.eval()
    model.to(device)

    total_frames = n_warmup + n_timed
    f0_hz = torch.randn(1, total_frames).abs() * 500 + 100
    f0_hz = f0_hz.clamp(min=20.0, max=2000.0).to(device)
    loudness = (torch.randn(1, total_frames) * 10 - 20).to(device)

    from synth.dsp.processors import scale_f0
    f0_scaled = scale_f0(f0_hz)

    # Warmup decoder
    for i in range(n_warmup):
        _ = model.decoder(
            f0_scaled[:, i : i + 1], loudness[:, i : i + 1]
        )

    # Timed: decoder
    times_decoder = []
    for i in range(n_timed):
        t0 = time.perf_counter()
        _ = model.decoder(
            f0_scaled[:, n_warmup + i : n_warmup + i + 1],
            loudness[:, n_warmup + i : n_warmup + i + 1],
        )
        if device == "cuda":
            torch.cuda.synchronize()
        times_decoder.append(time.perf_counter() - t0)

    # Warmup harmonic synth
    for i in range(n_warmup):
        _ = model.harmonic_synth(
            torch.rand(1, 1, model.n_harmonics, device=device),
            torch.tensor([[440.0]], device=device),
        )

    # Timed: harmonic synth
    times_harmonic = []
    for i in range(n_timed):
        harm_amps = torch.rand(1, 1, model.n_harmonics, device=device)
        t0 = time.perf_counter()
        _ = model.harmonic_synth(
            harm_amps,
            f0_hz[:, n_warmup + i : n_warmup + i + 1],
        )
        if device == "cuda":
            torch.cuda.synchronize()
        times_harmonic.append(time.perf_counter() - t0)

    # Warmup noise synth
    for i in range(n_warmup):
        _ = model.noise_synth(
            torch.rand(1, 1, 65, device=device),
        )

    # Timed: noise synth
    times_noise = []
    for i in range(n_timed):
        noise_mags = torch.rand(1, 1, 65, device=device)
        t0 = time.perf_counter()
        _ = model.noise_synth(noise_mags)
        if device == "cuda":
            torch.cuda.synchronize()
        times_noise.append(time.perf_counter() - t0)

    # End-to-end per-frame (decoder + both synths)
    times_total = [d + h + n for d, h, n in zip(times_decoder, times_harmonic, times_noise)]

    decoder_ms = np.mean(times_decoder) * 1000
    harmonic_ms = np.mean(times_harmonic) * 1000
    noise_ms = np.mean(times_noise) * 1000
    total_ms = np.mean(times_total) * 1000

    frame_duration_ms = model.block_size / model.sample_rate * 1000
    rtf = total_ms / frame_duration_ms

    harmonic_pct = harmonic_ms / total_ms * 100 if total_ms > 0 else 0

    return {
        "decoder_ms": decoder_ms,
        "harmonic_ms": harmonic_ms,
        "noise_ms": noise_ms,
        "total_ms": total_ms,
        "frame_duration_ms": frame_duration_ms,
        "rtf": rtf,
        "harmonic_pct": harmonic_pct,
    }
