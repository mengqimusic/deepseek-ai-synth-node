"""
Spectral competition scheduler — simplest viable version.

Operates in the harmonic amplitude domain (pre-synthesis) to avoid
audio-domain artifacts. Soft gain reduction per frequency band based
on competition weights and withdrawal styles.

Deterministic: same inputs → same outputs (no random elements).
"""

import torch
import torch.nn as nn

# 3-band frequency split points
BAND_LOW_MAX = 500.0
BAND_MID_MAX = 2000.0

# Per-band energy threshold above which competition activates
ENERGY_THRESHOLD = 0.5


class SpectralCompetitionScheduler(nn.Module):
    """
    Soft gain scheduler for Voice-to-Voice spectral competition.

    Analyzes harmonic amplitudes across 3 frequency bands (low/mid/high).
    When multiple Voices crowd the same band, applies differentiated gain
    reduction: Voices with higher competition weight claim more resources;
    Voices with higher withdrawal factors in a band yield more.

    Stateless — all decisions are deterministic from current frame inputs.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        voice_params: list[dict],
    ) -> list[dict]:
        """
        Apply spectral competition to a list of Voice parameter dicts.

        Each dict must contain:
            harm_amps: [1, 1, n_harmonics] — biased harmonic amplitudes
            noise_mags: [1, 1, n_magnitudes] — biased noise magnitudes
            f0_hz: float — fundamental frequency
            competition_weight: float
            withdrawal_low: float
            withdrawal_mid: float
            withdrawal_high: float
            is_active: bool — whether this Voice is producing sound

        Returns modified copies of the input dicts with adjusted
        harm_amps and noise_mags.
        """
        active = [vp for vp in voice_params if vp.get("is_active", False)]
        if len(active) < 2:
            return voice_params  # no competition needed

        n_h = voice_params[0]["harm_amps"].shape[-1]

        # Step 1: compute per-Voice per-band energy
        per_voice_energy = []  # list of {low, mid, high}
        per_voice_mask = []  # list of {low, mid, high} boolean masks over harmonics

        for vp in voice_params:
            if not vp.get("is_active", False):
                per_voice_energy.append({"low": 0.0, "mid": 0.0, "high": 0.0})
                per_voice_mask.append(None)
                continue

            f0 = vp["f0_hz"]
            harm = vp["harm_amps"]  # [1, 1, n_h]
            device = harm.device

            harmonics = torch.arange(1, n_h + 1, device=device, dtype=torch.float32)
            freqs = harmonics * f0  # [n_h]

            low_mask = freqs < BAND_LOW_MAX
            mid_mask = (freqs >= BAND_LOW_MAX) & (freqs < BAND_MID_MAX)
            high_mask = freqs >= BAND_MID_MAX

            energy_low = harm[:, :, low_mask].sum().item()
            energy_mid = harm[:, :, mid_mask].sum().item()
            energy_high = harm[:, :, high_mask].sum().item()

            per_voice_energy.append({
                "low": energy_low,
                "mid": energy_mid,
                "high": energy_high,
            })
            per_voice_mask.append({
                "low": low_mask,
                "mid": mid_mask,
                "high": high_mask,
            })

        # Step 2: for each band, compute competition gain reductions
        bands = ["low", "mid", "high"]
        result = []

        for i, vp in enumerate(voice_params):
            if not vp.get("is_active", False):
                result.append(vp)
                continue

            harm = vp["harm_amps"].clone()
            noise = vp["noise_mags"].clone()

            cw_i = vp["competition_weight"]
            w_style = {
                "low": vp.get("withdrawal_low", 0.5),
                "mid": vp.get("withdrawal_mid", 0.5),
                "high": vp.get("withdrawal_high", 0.5),
            }

            total_weight = sum(
                vp2.get("competition_weight", 1.0)
                for vp2 in voice_params
                if vp2.get("is_active", False)
            )

            for band in bands:
                total_energy = sum(pe[band] for pe in per_voice_energy)
                my_energy = per_voice_energy[i][band]

                if total_energy <= ENERGY_THRESHOLD or my_energy <= 0:
                    continue

                # My claim to this band based on weight proportion
                claim_ratio = cw_i / max(total_weight, 1e-6)
                my_share = claim_ratio * ENERGY_THRESHOLD
                excess = my_energy - my_share

                if excess > 0:
                    # Soft reduction: scale excess by withdrawal factor
                    withdrawal = w_style[band]
                    excess_ratio = excess / max(my_energy, 1e-6)
                    reduction = 1.0 - withdrawal * excess_ratio * 0.5
                    reduction = max(0.3, reduction)  # floor at 0.3x

                    mask = per_voice_mask[i][band]
                    harm[:, :, mask] *= reduction

            # Noise: apply average reduction across bands
            harm_before = vp["harm_amps"]
            total_before = harm_before.sum().item()
            total_after = harm.sum().item()
            if total_before > 1e-6:
                noise_scale = total_after / total_before
                noise = noise * noise_scale

            result.append({
                **vp,
                "harm_amps": harm,
                "noise_mags": noise,
            })

        return result
