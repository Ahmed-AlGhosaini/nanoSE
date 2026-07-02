# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import fast_bss_eval
import numpy as np
import torch
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from torchmetrics.functional.audio.dnsmos import (
    deep_noise_suppression_mean_opinion_score,
)


class SpeechMetrics:
    def __init__(self, fs=16000, device="cpu"):
        self.fs = fs
        self.device = device

        self.pesq = PerceptualEvaluationSpeechQuality(fs, "wb")
        self.estoi = ShortTimeObjectiveIntelligibility(fs, extended=True)

    def compute_si_sdr(self, ref, est):
        """
        Fast and differentiable SI-SDR using fast_bss_eval.
        ref, est shape: [B, T] or [T]
        """
        # Save original shape to restore later
        is_1d = ref.ndim == 1
        if is_1d:
            ref = ref.unsqueeze(0)
            est = est.unsqueeze(0)

        # Add channel dimension: [B, 1, T] as expected by fast_bss_eval
        ref_ch = ref.unsqueeze(1)
        est_ch = est.unsqueeze(1)

        # Compute SI-SDR
        # zero_mean=True standardizes signals before computing SDR
        neg_sdr = fast_bss_eval.sdr_loss(est_ch, ref_ch, filter_length=1, zero_mean=True, clamp_db=30, pairwise=False)  # shape: [B, 1]
        sdr = -neg_sdr

        # Squeeze out channel dimension
        sdr = sdr.squeeze(1)  # shape: [B]

        if is_1d:
            return sdr[0]
        return sdr

    def compute_eval_metrics(self, ref, est):
        """
        Computes PESQ, STOI, and DNSMOS on the given tensors.
        ref, est shape: [B, T] or [T] (expects float32)
        Note: These metrics are not differentiable and are evaluated on CPU.
        """
        # Ensure we are on CPU and detached
        ref = ref.detach().cpu()
        est = est.detach().cpu()

        if ref.ndim == 1:
            ref = ref.unsqueeze(0)
            est = est.unsqueeze(0)

        pesq_scores = []
        estoi_scores = []
        dnsmos_scores = []

        for r, e in zip(ref, est):
            # Calculate PESQ
            try:
                p_val = self.pesq(e, r).item()
                pesq_scores.append(p_val)
            except Exception:
                # PESQ can fail if the signal has zero energy or clipping issues
                pesq_scores.append(float("nan"))

            # Calculate STOI
            try:
                s_val = self.estoi(e, r).item()
                estoi_scores.append(s_val)
            except Exception:
                estoi_scores.append(float("nan"))

            # Calculate DNSMOS
            try:
                # deep_noise_suppression_mean_opinion_score returns a dict or tensor
                # with personalized=False, it returns a tensor of shape (3,): [OVRL, SIG, BAK]
                # Let's extract overall score (first element)
                dns_val = deep_noise_suppression_mean_opinion_score(
                    e, self.fs, personalized=False
                )
                # dns_val: [OVRL, SIG, BAK] -> we want overall (index 0)
                dnsmos_scores.append(dns_val[0].item())
            except Exception:
                dnsmos_scores.append(float("nan"))

        # Return average values across the batch (ignoring NaNs)
        return {
            "pesq": np.nanmean(pesq_scores) if pesq_scores else float("nan"),
            "estoi": np.nanmean(estoi_scores) if estoi_scores else float("nan"),
            "dnsmos": np.nanmean(dnsmos_scores) if dnsmos_scores else float("nan"),
        }
