import math
import torch
import torch.nn as nn


def _biquad_bandpass_coeffs(
    freq_hz: torch.Tensor,
    q: torch.Tensor,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute biquad bandpass coefficients (constant 0dB peak gain, RBJ cookbook).

    Args:
        freq_hz: [B] center frequency in Hz
        q:       [B] quality factor
        sample_rate: sample rate in Hz

    Returns:
        b0, b1, b2, a1, a2 — each [B]
    """
    omega = 2.0 * math.pi * freq_hz / sample_rate
    sin_omega = torch.sin(omega)
    cos_omega = torch.cos(omega)
    alpha = sin_omega / (2.0 * q.clamp(min=0.5))

    b0 = alpha
    b1 = torch.zeros_like(b0)
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha

    b0 = b0 / a0
    b1 = b1 / a0
    b2 = b2 / a0
    a1 = a1 / a0
    a2 = a2 / a0

    return b0, b1, b2, a1, a2


class FormantFilter(nn.Module):
    """
    Parallel stateful biquad bandpass filters for formant synthesis.

    3 bands emulate vowel-like spectral shaping. The resonance direction
    controls formant frequency positions (dark ↔ bright) and Q (sharpness).

    IIR state is carried across frames to avoid 250Hz filter-reset transients.
    State is reset on voice note-on (via VoiceModule.reset()).
    """

    _F_DARK = (270.0, 800.0, 2300.0)
    _F_NEUTRAL = (500.0, 1500.0, 2500.0)
    _F_BRIGHT = (730.0, 2100.0, 3000.0)

    _Q_DARK = (5.0, 5.0, 5.0)
    _Q_NEUTRAL = (8.0, 8.0, 8.0)
    _Q_BRIGHT = (12.0, 12.0, 12.0)

    def __init__(
        self,
        n_bands: int = 3,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.sample_rate = sample_rate

        self.register_buffer("_x1", torch.zeros(1, 1), persistent=False)
        self.register_buffer("_x2", torch.zeros(1, 1), persistent=False)
        self.register_buffer("_y1", torch.zeros(1, 1), persistent=False)
        self.register_buffer("_y2", torch.zeros(1, 1), persistent=False)
        self._initialized = False

    def _ensure_state(self, B: int, device: torch.device, dtype: torch.dtype):
        if not self._initialized or self._x1.shape[1] != B:
            self._x1 = torch.zeros(self.n_bands, B, device=device, dtype=dtype)
            self._x2 = torch.zeros(self.n_bands, B, device=device, dtype=dtype)
            self._y1 = torch.zeros(self.n_bands, B, device=device, dtype=dtype)
            self._y2 = torch.zeros(self.n_bands, B, device=device, dtype=dtype)
            self._initialized = True

    def reset(self):
        self._initialized = False

    def _interpolate_formants(
        self,
        level: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate formant frequencies and Q values from resonance level.

        Piecewise: level in [0, 0.5] → dark→neutral, [0.5, 1] → neutral→bright.

        Args:
            level: [B] resonance level in [0, 1]

        Returns:
            freqs_hz: [B, n_bands] center frequencies
            qs:       [B, n_bands] quality factors
        """
        device = level.device
        dtype = level.dtype

        dark_f = torch.tensor(self._F_DARK[: self.n_bands], device=device, dtype=dtype)
        neutral_f = torch.tensor(self._F_NEUTRAL[: self.n_bands], device=device, dtype=dtype)
        bright_f = torch.tensor(self._F_BRIGHT[: self.n_bands], device=device, dtype=dtype)

        dark_q = torch.tensor(self._Q_DARK[: self.n_bands], device=device, dtype=dtype)
        neutral_q = torch.tensor(self._Q_NEUTRAL[: self.n_bands], device=device, dtype=dtype)
        bright_q = torch.tensor(self._Q_BRIGHT[: self.n_bands], device=device, dtype=dtype)

        t_low = (level * 2.0).clamp(0.0, 1.0)
        t_high = ((level - 0.5) * 2.0).clamp(0.0, 1.0)

        use_high = (level >= 0.5).unsqueeze(-1)

        t_safe = t_low.unsqueeze(-1)
        freqs = (1.0 - t_safe) * dark_f + t_safe * neutral_f
        qs = (1.0 - t_safe) * dark_q + t_safe * neutral_q

        t_high_unsq = t_high.unsqueeze(-1)
        freqs_high = (1.0 - t_high_unsq) * neutral_f + t_high_unsq * bright_f
        qs_high = (1.0 - t_high_unsq) * neutral_q + t_high_unsq * bright_q

        freqs = torch.where(use_high, freqs_high, freqs)
        qs = torch.where(use_high, qs_high, qs)

        return freqs, qs

    def _apply_filter_bands(
        self,
        audio: torch.Tensor,
        freqs: torch.Tensor,
        qs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply stateful parallel biquad bandpass filters.

        Args:
            audio: [B, S]
            freqs: [B, n_bands]
            qs:    [B, n_bands]

        Returns:
            filtered: [B, S]
        """
        B, S = audio.shape
        out = torch.zeros_like(audio)

        for band in range(self.n_bands):
            f = freqs[:, band]
            q = qs[:, band]

            b0, b1, b2, a1, a2 = _biquad_bandpass_coeffs(f, q, self.sample_rate)

            x1 = self._x1[band].clone()
            x2 = self._x2[band].clone()
            y1 = self._y1[band].clone()
            y2 = self._y2[band].clone()

            for n in range(S):
                x_n = audio[:, n]
                y_n = b0 * x_n + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                out[:, n] = out[:, n] + y_n
                x2, x1 = x1, x_n
                y2, y1 = y1, y_n

            self._x1[band] = x1
            self._x2[band] = x2
            self._y1[band] = y1
            self._y2[band] = y2

        out = out / math.sqrt(self.n_bands)
        return out

    def forward(self, audio: torch.Tensor, resonance_level: torch.Tensor) -> torch.Tensor:
        """
        Apply formant filtering using resonance level (Phase 9 compat).

        Args:
            audio: [B, S] input audio samples
            resonance_level: [B] level in [0, 1] controlling formant shape

        Returns:
            filtered: [B, S]
        """
        B, S = audio.shape
        device = audio.device
        dtype = audio.dtype

        level = resonance_level.clamp(0.0, 1.0).to(device=device, dtype=dtype)
        if level.dim() == 0:
            level = level.unsqueeze(0)

        self._ensure_state(B, device, dtype)

        freqs, qs = self._interpolate_formants(level)
        return self._apply_filter_bands(audio, freqs, qs)

    def forward_explicit(
        self,
        audio: torch.Tensor,
        formant_freqs: torch.Tensor,
        formant_qs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply formant filtering with explicit per-band parameters (Phase 10a).

        Args:
            audio:         [B, S]
            formant_freqs: [B, n_bands] center frequencies in Hz
            formant_qs:    [B, n_bands] quality factors

        Returns:
            filtered: [B, S]
        """
        B, S = audio.shape
        device = audio.device
        dtype = audio.dtype

        self._ensure_state(B, device, dtype)

        freqs = formant_freqs.to(device=device, dtype=dtype)
        qs = formant_qs.to(device=device, dtype=dtype)
        if freqs.dim() == 1:
            freqs = freqs.unsqueeze(0)
        if qs.dim() == 1:
            qs = qs.unsqueeze(0)

        return self._apply_filter_bands(audio, freqs, qs)
