# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Spectrogram / Waveform Utility Pipeline (16kHz) ---


def waveform_to_spectrogram(waveform, n_fft=512, hop_length=256):
    """
    Computes complex STFT of a 1D waveform tensor.
    waveform shape: [B, num_samples]
    """
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=waveform.device),
        return_complex=True,
    )
    return stft


def spectrogram_to_waveform(stft, length, n_fft=512, hop_length=256):
    """
    Computes iSTFT to reconstruct 1D waveform.
    stft shape: [B, F, T]
    """
    waveform = torch.istft(
        stft,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=stft.device),
        length=length,
    )
    return waveform


def enhance_waveform(model, noisy_waveform, n_fft=512, hop_length=256):
    """
    Unified/backward-compatible helper to clean a noisy waveform.
    Uses the model's standardized enhance method if available.
    """
    if hasattr(model, "enhance"):
        return model.enhance(noisy_waveform)

    # Fallback for raw low-level models if they don't have the enhance method
    stft = waveform_to_spectrogram(noisy_waveform, n_fft, hop_length)  # [B, 257, T]
    mag = torch.abs(stft)  # [B, 257, T]
    phase = torch.angle(stft)  # [B, 257, T]

    # Slice the Nyquist bin to obtain a 256-bin shape for Conv2d encoder
    mag_cropped = mag[:, :256, :]  # [B, 256, T]

    # Forward pass through CRN (expects [B, 1, F, T], outputs [B, 1, F, T])
    mask = model(mag_cropped.unsqueeze(1)).squeeze(1)  # [B, 256, T]
    mag_enhanced_cropped = mag_cropped * mask  # [B, 256, T]

    # Re-insert the Nyquist frequency bin filled with zeros
    nyquist_bin = torch.zeros(mag.size(0), 1, mag.size(2), device=mag.device)
    mag_enhanced = torch.cat([mag_enhanced_cropped, nyquist_bin], dim=1)  # [B, 257, T]

    # Reconstruct waveform using polar representation
    stft_enhanced = torch.polar(mag_enhanced, phase)
    enhanced_waveform = spectrogram_to_waveform(
        stft_enhanced, noisy_waveform.size(1), n_fft, hop_length
    )
    return enhanced_waveform


def get_model(model_name: str, **kwargs):
    """
    Factory function to retrieve a standardized speech enhancement model.
    """
    model_name = model_name.lower()
    if model_name == "crn":
        from models.crn import CRN

        return CRN(**kwargs)
    elif model_name == "crntiny":
        from models.crn import CRNTiny

        return CRNTiny(**kwargs)
    elif model_name == "fastenhancer" or model_name == "fastenhancercompact":
        from models.fastenhancer import FastEnhancerCompact

        return FastEnhancerCompact(**kwargs)
    elif model_name == "glumaskd" or model_name == "glumasked":
        from models.glumaskd import GLUMaskd

        return GLUMaskd(**kwargs)
    elif model_name == "comfi_fastgrnn":
        from models.comfi_fastgrnn import ComfiFastGRNNModel

        return ComfiFastGRNNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

