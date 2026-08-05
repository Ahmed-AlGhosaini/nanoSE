# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import copy

import fast_bss_eval
import torch
import torch.nn as nn
import torch.nn.functional as F

# Activations selectable from a config file, e.g. CRNTiny(activation="prelu").
# The default "leaky_relu" reproduces the original baseline exactly.
ACTIVATIONS = {
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "hardswish": nn.Hardswish,
    "tanh": nn.Tanh,
}


def get_activation(activation):
    """
    Resolves an activation specification into a *fresh* nn.Module instance.

    Accepts a name from ACTIVATIONS, a callable returning a module (e.g. nn.ReLU),
    or a module instance (which is deep-copied so blocks never share parameters).
    """
    if isinstance(activation, nn.Module):
        return copy.deepcopy(activation)
    if callable(activation):
        return activation()
    key = str(activation).lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{activation}'. Available: {sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[key]()


def compressed_mse(pred_mag, target_mag, compression=1.0, eps=1e-6):
    """
    MSE between magnitude spectrograms, optionally with power-law compression.

    compression=1.0 gives the plain magnitude MSE of the baseline. Values below 1
    (e.g. 0.3) de-emphasize high-energy bins and are the standard choice in the
    speech enhancement literature.
    """
    if compression != 1.0:
        pred_mag = pred_mag.clamp_min(eps).pow(compression)
        target_mag = target_mag.clamp_min(eps).pow(compression)
    return F.mse_loss(pred_mag, target_mag)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(5, 3),
        stride=(2, 1),
        padding=(2, 1),
        activation="leaky_relu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(5, 3),
        stride=(2, 1),
        padding=(2, 1),
        output_padding=(1, 0),
        activation="leaky_relu",
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class CRN(nn.Module):
    """
    Lightweight CRN (Convolutional Recurrent Network) for Speech Enhancement.
    Input: [B, 1, 256, T] (Magnitude Spectrogram)
    Output: [B, 1, 256, T] (Predicted Ideal Ratio Mask)
    """

    def __init__(
        self,
        activation="leaky_relu",
        compression=1.0,
        spec_weight=1.0,
        sdr_weight=0.01,
    ):
        super().__init__()
        self.activation_name = activation if isinstance(activation, str) else "custom"
        self.compression = compression
        self.spec_weight = spec_weight
        self.sdr_weight = sdr_weight

        # Encoder: compresses spectral bins (height) from 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = EncoderBlock(1, 8, activation=activation)
        self.enc2 = EncoderBlock(8, 16, activation=activation)
        self.enc3 = EncoderBlock(16, 32, activation=activation)
        self.enc4 = EncoderBlock(32, 64, activation=activation)
        self.enc5 = EncoderBlock(64, 128, activation=activation)

        # Recurrent bottleneck GRU
        self.gru = nn.GRU(
            input_size=1024, hidden_size=256, num_layers=1, batch_first=True
        )
        self.linear = nn.Linear(256, 1024)

        # Decoder: channels are doubled because of skip-connections (concat with encoder outputs)
        self.dec5 = DecoderBlock(256, 64, activation=activation)
        self.dec4 = DecoderBlock(128, 32, activation=activation)
        self.dec3 = DecoderBlock(64, 16, activation=activation)
        self.dec2 = DecoderBlock(32, 8, activation=activation)

        # Final output layer mapping back to 1 channel (the predicted mask)
        self.dec1_deconv = nn.ConvTranspose2d(
            16,
            1,
            kernel_size=(5, 3),
            stride=(2, 1),
            padding=(2, 1),
            output_padding=(1, 0),
        )

    def forward(self, x):
        # x shape: [B, 1, 256, T]
        e1 = self.enc1(x)  # [B, 8, 128, T]
        e2 = self.enc2(e1)  # [B, 16, 64, T]
        e3 = self.enc3(e2)  # [B, 32, 32, T]
        e4 = self.enc4(e3)  # [B, 64, 16, T]
        e5 = self.enc5(e4)  # [B, 128, 8, T]

        # Reshape for GRU: [B, T, Channels * Freq_Bins]
        b, c, f, t = e5.size()
        g_in = e5.permute(0, 3, 1, 2).reshape(b, t, c * f)  # [B, T, 1024]

        # Recurrent processing
        g_out, _ = self.gru(g_in)  # [B, T, 256]
        g_out = self.linear(g_out)  # [B, T, 1024]

        # Reshape back to [B, 128, 8, T] for decoding
        d5_in = g_out.reshape(b, t, c, f).permute(0, 2, 3, 1)

        # Decoder passes with skip connections (Concatenating along channels)
        d5 = self.dec5(torch.cat([d5_in, e5], dim=1))  # [B, 64, 16, T]
        d4 = self.dec4(torch.cat([d5, e4], dim=1))  # [B, 32, 32, T]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # [B, 16, 64, T]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # [B, 8, 128, T]

        # Final deconvolution to mask shape
        d1_in = torch.cat([d2, e1], dim=1)  # [B, 16, 128, T]
        mask = torch.sigmoid(self.dec1_deconv(d1_in))  # [B, 1, 256, T]

        return mask

    def enhance(self, noisy):
        """
        Enhances a noisy waveform.
        noisy: [B, T] tensor
        """
        from model import spectrogram_to_waveform, waveform_to_spectrogram

        stft = waveform_to_spectrogram(noisy)  # [B, 257, T]
        mag = torch.abs(stft)  # [B, 257, T]
        phase = torch.angle(stft)  # [B, 257, T]

        # Slice the Nyquist bin to obtain a 256-bin shape for Conv2d encoder
        mag_cropped = mag[:, :256, :]  # [B, 256, T]

        # Forward pass through CRN
        mask = self.forward(mag_cropped.unsqueeze(1)).squeeze(1)  # [B, 256, T]
        mag_enhanced_cropped = mag_cropped * mask  # [B, 256, T]

        # Re-insert the Nyquist frequency bin filled with zeros
        nyquist_bin = torch.zeros(mag.size(0), 1, mag.size(2), device=mag.device)
        mag_enhanced = torch.cat(
            [mag_enhanced_cropped, nyquist_bin], dim=1
        )  # [B, 257, T]

        # Reconstruct waveform using polar representation
        stft_enhanced = torch.polar(mag_enhanced, phase)
        enhanced_waveform = spectrogram_to_waveform(stft_enhanced, noisy.size(1))
        return enhanced_waveform

    def compute_loss(self, noisy, clean):
        """
        Computes the combined loss, and returns (estimate, loss, metrics_dict).
        noisy, clean: [B, T] tensors
        """
        from model import waveform_to_spectrogram

        stft_noisy = waveform_to_spectrogram(noisy)  # [B, 257, T]
        mag_noisy = torch.abs(stft_noisy)[:, :256, :]  # [B, 256, T]

        stft_clean = waveform_to_spectrogram(clean)
        mag_clean = torch.abs(stft_clean)[:, :256, :]

        # Forward pass to predict mask
        pred_mask = self.forward(mag_noisy.unsqueeze(1)).squeeze(1)  # [B, 256, T]

        # Apply mask to reconstruct audio
        enhanced = self.enhance(noisy)

        # Compute spectrogram MSE loss (optionally power-law compressed)
        pred_mag = mag_noisy * pred_mask
        spec_loss = compressed_mse(pred_mag, mag_clean, self.compression)

        # Waveform SI-SDR loss (differentiable negative SI-SDR)
        sdr_loss = fast_bss_eval.si_sdr_loss(
            enhanced.unsqueeze(1), clean.unsqueeze(1), zero_mean=True, clamp_db=30
        ).mean()

        # Combined Loss
        loss = self.spec_weight * spec_loss + self.sdr_weight * sdr_loss

        metrics_dict = {
            "spec_loss": spec_loss.item(),
            "sdr_loss": sdr_loss.item(),
        }

        return enhanced, loss, metrics_dict


class CRNTiny(nn.Module):
    """
    Tiny CRN (Convolutional Recurrent Network) for fast speech enhancement baseline.
    Input: [B, 1, 256, T] (Magnitude Spectrogram)
    Output: [B, 1, 256, T] (Predicted Ideal Ratio Mask)
    """

    def __init__(
        self,
        activation="leaky_relu",
        compression=1.0,
        spec_weight=1.0,
        sdr_weight=0.01,
    ):
        super().__init__()
        self.activation_name = activation if isinstance(activation, str) else "custom"
        self.compression = compression
        self.spec_weight = spec_weight
        self.sdr_weight = sdr_weight

        # Encoder: compresses spectral bins (height) from 256 -> 128 -> 64 -> 32 -> 16
        self.enc1 = EncoderBlock(1, 4, activation=activation)
        self.enc2 = EncoderBlock(4, 8, activation=activation)
        self.enc3 = EncoderBlock(8, 16, activation=activation)
        self.enc4 = EncoderBlock(16, 32, activation=activation)

        # Recurrent bottleneck GRU (input_size: 32 channels * 16 bins = 512)
        self.gru = nn.GRU(
            input_size=512, hidden_size=64, num_layers=1, batch_first=True
        )
        self.linear = nn.Linear(64, 512)

        # Decoder: channels are doubled because of skip-connections (concat with encoder outputs)
        self.dec4 = DecoderBlock(64, 16, activation=activation)
        self.dec3 = DecoderBlock(32, 8, activation=activation)
        self.dec2 = DecoderBlock(16, 4, activation=activation)

        # Final output layer mapping back to 1 channel (the predicted mask)
        self.dec1_deconv = nn.ConvTranspose2d(
            8,
            1,
            kernel_size=(5, 3),
            stride=(2, 1),
            padding=(2, 1),
            output_padding=(1, 0),
        )

    def forward(self, x):
        # x shape: [B, 1, 256, T]
        e1 = self.enc1(x)  # [B, 4, 128, T]
        e2 = self.enc2(e1)  # [B, 8, 64, T]
        e3 = self.enc3(e2)  # [B, 16, 32, T]
        e4 = self.enc4(e3)  # [B, 32, 16, T]

        # Reshape for GRU: [B, T, Channels * Freq_Bins]
        b, c, f, t = e4.size()
        g_in = e4.permute(0, 3, 1, 2).reshape(b, t, c * f)  # [B, T, 512]

        # Recurrent processing
        g_out, _ = self.gru(g_in)  # [B, T, 64]
        g_out = self.linear(g_out)  # [B, T, 512]

        # Reshape back for decoding
        d4_in = g_out.reshape(b, t, c, f).permute(0, 2, 3, 1)

        # Decoder passes with skip connections
        d4 = self.dec4(torch.cat([d4_in, e4], dim=1))  # [B, 16, 32, T]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # [B, 8, 64, T]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # [B, 4, 128, T]

        # Final deconvolution to mask shape
        d1_in = torch.cat([d2, e1], dim=1)  # [B, 8, 128, T]
        mask = torch.sigmoid(self.dec1_deconv(d1_in))  # [B, 1, 256, T]

        return mask

    def enhance(self, noisy):
        from model import spectrogram_to_waveform, waveform_to_spectrogram

        stft = waveform_to_spectrogram(noisy)  # [B, 257, T]
        mag = torch.abs(stft)  # [B, 257, T]
        phase = torch.angle(stft)  # [B, 257, T]

        # Slice the Nyquist bin
        mag_cropped = mag[:, :256, :]  # [B, 256, T]

        # Forward pass through CRN
        mask = self.forward(mag_cropped.unsqueeze(1)).squeeze(1)  # [B, 256, T]
        mag_enhanced_cropped = mag_cropped * mask  # [B, 256, T]

        # Re-insert Nyquist bin
        nyquist_bin = torch.zeros(mag.size(0), 1, mag.size(2), device=mag.device)
        mag_enhanced = torch.cat([mag_enhanced_cropped, nyquist_bin], dim=1)

        # Reconstruct
        stft_enhanced = torch.polar(mag_enhanced, phase)
        enhanced_waveform = spectrogram_to_waveform(stft_enhanced, noisy.size(1))
        return enhanced_waveform

    def compute_loss(self, noisy, clean):
        from model import waveform_to_spectrogram

        stft_noisy = waveform_to_spectrogram(noisy)  # [B, 257, T]
        mag_noisy = torch.abs(stft_noisy)[:, :256, :]  # [B, 256, T]

        stft_clean = waveform_to_spectrogram(clean)
        mag_clean = torch.abs(stft_clean)[:, :256, :]

        # Forward pass to predict mask
        pred_mask = self.forward(mag_noisy.unsqueeze(1)).squeeze(1)  # [B, 256, T]

        # Apply mask to reconstruct audio
        enhanced = self.enhance(noisy)

        # Compute spectrogram MSE loss (optionally power-law compressed)
        pred_mag = mag_noisy * pred_mask
        spec_loss = compressed_mse(pred_mag, mag_clean, self.compression)

        # Waveform SI-SDR loss (differentiable negative SI-SDR)
        sdr_loss = fast_bss_eval.si_sdr_loss(
            enhanced.unsqueeze(1), clean.unsqueeze(1), zero_mean=True, clamp_db=30
        ).mean()

        # Combined Loss
        loss = self.spec_weight * spec_loss + self.sdr_weight * sdr_loss

        metrics_dict = {
            "spec_loss": spec_loss.item(),
            "sdr_loss": sdr_loss.item(),
        }

        return enhanced, loss, metrics_dict

