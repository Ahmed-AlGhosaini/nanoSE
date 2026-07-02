import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUMaskd(nn.Module):
    """
    Very simple 3-layer convolutional model with GLU activations to predict a mask.
    Loss is negative SNR.
    """

    def __init__(self):
        super().__init__()
        # GLU splits channels in half. To output C channels, conv must output 2C channels.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(5, 5), padding=(2, 2))
        self.conv3 = nn.Conv2d(16, 2, kernel_size=(5, 5), padding=(2, 2))

    def forward(self, x):
        # x shape: [B, 1, 256, T]
        x = F.glu(self.conv1(x), dim=1)  # [B, 8, 256, T]
        x = F.glu(self.conv2(x), dim=1)  # [B, 16, 256, T]
        mask = torch.sigmoid(F.glu(self.conv3(x), dim=1))  # [B, 1, 256, T]
        return mask

    def enhance(self, noisy):
        """
        Enhances a noisy waveform.
        noisy: [B, T] tensor
        """
        from model import spectrogram_to_waveform, waveform_to_spectrogram

        # Calculate mean and std per sample in batch
        m = noisy.mean(dim=-1, keepdim=True)
        s = noisy.std(dim=-1, keepdim=True).clamp(min=1e-8)

        # Normalize input waveform
        norm_noisy = (noisy - m) / s

        stft = waveform_to_spectrogram(norm_noisy)  # [B, 257, T]
        mag = torch.abs(stft)  # [B, 257, T]
        phase = torch.angle(stft)  # [B, 257, T]

        # Slice the Nyquist bin to obtain a 256-bin shape for Conv2d
        mag_cropped = mag[:, :256, :]  # [B, 256, T]

        # Log operation (epsilon protected) on the input magnitude feature
        mag_compressed = mag_cropped**0.3

        # Forward pass to predict mask
        mask = self.forward(mag_compressed.unsqueeze(1)).squeeze(1)  # [B, 256, T]
        mag_enhanced_cropped = mag_cropped * mask  # [B, 256, T]

        # Re-insert the Nyquist frequency bin filled with zeros
        nyquist_bin = torch.zeros(mag.size(0), 1, mag.size(2), device=mag.device)
        mag_enhanced = torch.cat(
            [mag_enhanced_cropped, nyquist_bin], dim=1
        )  # [B, 257, T]

        # Reconstruct waveform using polar representation
        stft_enhanced = torch.polar(mag_enhanced, phase)
        enhanced_waveform_norm = spectrogram_to_waveform(stft_enhanced, noisy.size(1))

        # Denormalize output waveform
        enhanced_waveform = s * enhanced_waveform_norm + m
        return enhanced_waveform

    def compute_loss(self, noisy, clean):
        """
        Computes the negative SNR loss, and returns (estimate, loss, metrics_dict).
        noisy, clean: [B, T] tensors
        """
        # Enhance audio
        enhanced = self.enhance(noisy)

        # Compute SNR Loss (negative SNR)
        eps = 1e-8
        power_target = torch.sum(clean**2, dim=-1)
        power_noise = torch.sum((clean - enhanced) ** 2, dim=-1)
        snr = 10 * torch.log10(power_target / (power_noise + eps) + eps)
        loss = -torch.mean(snr)

        metrics_dict = {
            "snr": torch.mean(snr).item(),
            "total_loss": loss.item(),
        }

        return enhanced, loss, metrics_dict
