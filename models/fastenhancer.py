# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelsLastBatchNorm(nn.Module):
    """
    Applies 1D Batch Normalization over the channel dimension of a [T, B, F, C] tensor.
    """
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels, eps=eps)

    def forward(self, x):
        T, B, F, C = x.shape
        x_flat = x.reshape(T * B * F, C)
        x_norm = self.bn(x_flat)
        return x_norm.reshape(T, B, F, C)


class Attention(nn.Module):
    """
    Multi-head Self-Attention along the frequency dimension.
    Uses PyTorch's native memory-efficient Scaled Dot Product Attention (SDPA).
    """
    def __init__(self, channels: int, num_heads: int, attn_bias: bool = False):
        super().__init__()
        self.channels = channels // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(channels, channels * 3, bias=attn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [T*B, Freq, C]
        TB, Freq, C = x.shape
        qkv = self.qkv(x)  # [TB, Freq, C*3]
        qkv = qkv.reshape(TB, Freq, self.num_heads, -1).transpose(1, 2)  # [TB, NH, Freq, C']
        
        q = qkv[:, :, :, :self.channels]
        k = qkv[:, :, :, self.channels:self.channels*2]
        v = qkv[:, :, :, self.channels*2:]
        
        # SDPA automatically uses FlashAttention/memory-efficient kernels where available
        out = F.scaled_dot_product_attention(q, k, v, scale=None)  # [TB, NH, Freq, C'']
        out = out.transpose(1, 2).reshape(TB, Freq, C)
        return out


def calculate_positional_embedding(channels: int, freq: int) -> torch.Tensor:
    f = torch.arange(1, freq + 1, dtype=torch.float32) * (math.pi / freq)
    c = torch.linspace(
        start=math.log(1),
        end=math.log(freq - 1),
        steps=channels // 2,
        dtype=torch.float32
    ).exp()
    grid = f.view(-1, 1) * c.view(1, -1)            # [F, C//2]
    pe = torch.cat((grid.sin(), grid.cos()), dim=1) # [F, C]
    return pe


class RNNFormerBlock(nn.Module):
    """
    Core dual-path building block: Causal time GRU and Frequency Self-Attention.
    """
    def __init__(
        self,
        channels: int,
        freq: int,
        num_heads: int,
        eps: float = 1e-5,
        positional_embedding: str = "train",  # None | "fixed" | "train"
        attn_bias: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.freq = freq

        # Time path (GRU)
        self.rnn = nn.GRU(channels, channels, batch_first=False)
        self.rnn_fc = nn.Linear(channels, channels, bias=False)
        self.rnn_norm = ChannelsLastBatchNorm(channels, eps=eps)

        # Frequency path (Self-Attention)
        self.attn = Attention(channels, num_heads, attn_bias=attn_bias)
        self.attn_fc = nn.Linear(channels, channels, bias=False)
        self.attn_norm = ChannelsLastBatchNorm(channels, eps=eps)

        # Positional Embedding
        self.pe = None
        if positional_embedding is not None:
            pe_val = calculate_positional_embedding(channels, freq)
            if positional_embedding == "fixed":
                self.register_buffer("pe", pe_val)
            elif positional_embedding == "train":
                self.pe = nn.Parameter(pe_val)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [T, B, num_freqs, C]
        T, B, num_freqs, C = x.shape
        x_in = x

        # 1. Time Processing
        # Flatten [T, B, num_freqs, C] -> [T, B*num_freqs, C] for GRU
        x_time = x.view(T, B * num_freqs, C)
        
        # MPS workaround for GRU in autocast (float16)
        orig_dtype = x_time.dtype
        if x_time.device.type == "mps":
            x_time = x_time.float()
            if h is not None:
                h = h.float()
            x_time, h = self.rnn(x_time, h)
            x_time = x_time.to(orig_dtype)
            if h is not None:
                h = h.to(orig_dtype)
        else:
            x_time, h = self.rnn(x_time, h)

        x_time = x_time.view(T, B, num_freqs, C)
        x_time = self.rnn_fc(x_time)
        x_time = self.rnn_norm(x_time)
        x = x_in + F.silu(x_time)

        # Add positional embedding before attention
        if self.pe is not None:
            x = x + self.pe.view(1, 1, num_freqs, C)

        x_in_attn = x

        # 2. Frequency Processing
        # Flatten [T, B, num_freqs, C] -> [T*B, num_freqs, C] for Attention
        x_freq = x.view(T * B, num_freqs, C)
        x_freq = self.attn(x_freq)
        x_freq = x_freq.view(T, B, num_freqs, C)
        x_freq = self.attn_fc(x_freq)
        x_freq = self.attn_norm(x_freq)
        x = x_in_attn + F.silu(x_freq)

        return x, h


def rf_pre_post_lin(n_freq: int, n_filter: int, init: str = "linear_fixed") -> tuple[nn.Module, nn.Module]:
    pre = nn.Linear(n_freq, n_filter, bias=False)
    post = nn.Linear(n_filter, n_freq, bias=False)

    if init.startswith("linear"):
        f_filter = torch.linspace(0, n_freq - 1, n_filter)
        f_freqs = torch.linspace(0, n_freq - 1, n_freq)
        delta = (n_freq - 1) / (n_filter - 1)
        down = (f_filter[1:, None] - f_freqs[None, :]) / delta
        up = (f_freqs[None, :] - f_filter[:-1, None]) / delta
        down = F.pad(down, (0, 0, 0, 1), value=1.0)
        up = F.pad(up, (0, 0, 1, 0), value=1.0)
        pre_weight = torch.max(torch.zeros_like(up), torch.min(down, up))
        pre_weight = pre_weight / pre_weight.sum(dim=1, keepdim=True)
        post_weight = pre_weight.transpose(0, 1)
        post_weight = post_weight / post_weight.sum(dim=1, keepdim=True)

        if init.endswith("_fixed"):
            delattr(pre, "weight")
            delattr(post, "weight")
            pre.register_buffer("weight", pre_weight.contiguous())
            post.register_buffer("weight", post_weight.contiguous())
        else:
            pre.weight.data.copy_(pre_weight)
            post.weight.data.copy_(post_weight)

    return pre, post


class ScaledConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, normalize: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            weight = F.normalize(self.weight, dim=(0, 1, 2)).mul(self.scale)
        else:
            weight = self.weight * self.scale
        return F.conv_transpose1d(
            x, weight, self.bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, dilation=self.dilation,
        )


class FastEnhancerCompact(nn.Module):
    """
    Aligned and parameterizable FastEnhancer model implementation.
    """
    def __init__(
        self,
        channels: int = 24,
        kernel_size: list[int] = [8, 3, 3],
        stride: int = 4,
        rnnformer_kwargs: dict = None,
        pre_post_init: str = "linear_fixed",
        n_fft: int = 512,
        hop_size: int = 256,
        win_size: int = 512,
        window: str = "hann",
        stft_normalized: bool = False,
        mask: str = None,
        activation: str = "SiLU",
        activation_kwargs: dict = None,
        input_compression: float = 0.3,
        normalize_final_conv: bool = True,
        weight_norm: bool = True,
        resnet: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.input_compression = input_compression

        # Window creation
        self.register_buffer("window", torch.hann_window(win_size))

        # Activation creation
        Act = getattr(nn, activation)
        act_kwargs = activation_kwargs or {"inplace": True}

        # 1. Encoder Pre-Net (Strided frequency reduction)
        self.enc_pre = nn.Sequential(
            nn.Conv1d(
                2, channels, kernel_size[0], stride=stride,
                padding=(kernel_size[0] - stride) // 2, bias=False
            ),
            nn.BatchNorm1d(channels),
            Act(**act_kwargs)
        )

        # 2. Encoder blocks
        self.encoder = nn.ModuleList()
        for idx in range(1, len(kernel_size)):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size[idx], padding=(kernel_size[idx] - 1) // 2, bias=False),
                    nn.BatchNorm1d(channels),
                    Act(**act_kwargs)
                )
            )

        # 3. RNNFormer Pre-Net & Blocks
        rnn_cfg = rnnformer_kwargs or {
            "num_blocks": 2,
            "channels": 20,
            "freq": 16,
            "num_heads": 4,
            "eps": 1e-5,
            "positional_embedding": "train",
            "attn_bias": False
        }
        self.rf_ch = rnn_cfg["channels"]
        self.rf_freq = rnn_cfg["freq"]

        # Map spatial frequency resolution from n_fft//2//stride down to self.rf_freq
        enc_out_freq = n_fft // 2 // stride
        rf_pre, rf_post = rf_pre_post_lin(enc_out_freq, self.rf_freq, pre_post_init)
        
        self.rf_pre = nn.Sequential(
            rf_pre,
            nn.Conv1d(channels, self.rf_ch, 1, bias=False),
            nn.BatchNorm1d(self.rf_ch)
        )

        # Blocks
        rf_list = []
        for i in range(rnn_cfg["num_blocks"]):
            # Positional embedding is only added to the first block
            pe = rnn_cfg["positional_embedding"] if i == 0 else None
            rf_list.append(
                RNNFormerBlock(
                    channels=self.rf_ch,
                    freq=self.rf_freq,
                    num_heads=rnn_cfg["num_heads"],
                    eps=rnn_cfg["eps"],
                    positional_embedding=pe,
                    attn_bias=rnn_cfg["attn_bias"]
                )
            )
        self.rf_blocks = nn.ModuleList(rf_list)

        # RNNFormer Post-Net
        self.rf_post = nn.Sequential(
            rf_post,
            nn.Conv1d(self.rf_ch, channels, 1, bias=False),
            nn.BatchNorm1d(channels)
        )

        # 4. Decoder blocks
        self.decoder = nn.ModuleList()
        for idx in range(len(kernel_size) - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(channels * 2, channels, 1, bias=False),
                    nn.BatchNorm1d(channels),
                    Act(**act_kwargs),
                    nn.Conv1d(channels, channels, kernel_size[idx], padding=(kernel_size[idx] - 1) // 2, bias=False),
                    nn.BatchNorm1d(channels),
                    Act(**act_kwargs)
                )
            )

        # 5. Decoder Post-Net
        self.dec_post = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            Act(**act_kwargs),
            ScaledConvTranspose1d(
                channels, 2, kernel_size[0], stride=stride,
                padding=(kernel_size[0] - stride) // 2, bias=True,
                normalize=normalize_final_conv
            )
        )

    def forward(self, noisy_waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # noisy_waveform: [B, T_wav]
        B, T_wav = noisy_waveform.shape

        # Calculate mean and std per sample in batch
        m = noisy_waveform.mean(dim=-1, keepdim=True)
        s = noisy_waveform.std(dim=-1, keepdim=True).clamp(min=1e-8)

        # Normalize input waveform
        norm_noisy_waveform = (noisy_waveform - m) / s

        # STFT
        stft = torch.stft(
            norm_noisy_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True,
            center=True,
        )  # [B, F_fft, T_spec]
        
        # Discard the highest Nyquist frequency bin
        spec = stft[:, :-1, :]  # [B, F, T_spec]
        spec_real_imag = torch.view_as_real(spec)  # [B, F, T_spec, 2]

        # Power Compression: Y_c = Y * |Y|^(c-1)
        mag = spec.abs().unsqueeze(-1).clamp(min=1e-12)
        spec_compressed = spec_real_imag * mag.pow(self.input_compression - 1.0) # [B, F, T_spec, 2]

        # Reshape to [B*T_spec, 2, F] for Encoder
        T_spec = spec_compressed.shape[2]
        x = spec_compressed.permute(0, 2, 3, 1).reshape(B * T_spec, 2, -1)  # [B*T_spec, 2, F]

        # Encoder Pre-net
        x = self.enc_pre(x)
        enc_outs = [x]

        # Encoder blocks
        for block in self.encoder:
            x = block(x)
            enc_outs.append(x)

        # RNNFormer Pre-net
        x = self.rf_pre(x)  # [B*T_spec, rf_ch, rf_freq]
        
        # Reshape for RNNFormer blocks: [T_spec, B, rf_freq, rf_ch]
        x = x.view(B, T_spec, self.rf_ch, self.rf_freq).permute(1, 0, 3, 2).contiguous()

        # RNNFormer blocks
        for block in self.rf_blocks:
            x, _ = block(x)

        # Reshape back to [B*T_spec, rf_ch, rf_freq]
        x = x.permute(1, 0, 3, 2).reshape(B * T_spec, self.rf_ch, self.rf_freq)

        # RNNFormer Post-net
        x = self.rf_post(x)

        # Decoder blocks
        for block in self.decoder:
            x = torch.cat([x, enc_outs.pop()], dim=1)
            x = block(x)

        # Decoder Post-net (final mask prediction)
        x = torch.cat([x, enc_outs.pop()], dim=1)
        mask = self.dec_post(x)  # [B*T_spec, 2, F]

        # Reshape mask back to [B, F, T_spec, 2]
        mask = mask.view(B, T_spec, 2, -1).permute(0, 3, 1, 2).contiguous()

        # Complex Mask Multiplication: est_compressed = noisy_compressed * mask
        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        spec_comp_real = spec_compressed[..., 0]
        spec_comp_imag = spec_compressed[..., 1]
        mask_real = mask[..., 0]
        mask_imag = mask[..., 1]

        est_comp_real = spec_comp_real * mask_real - spec_comp_imag * mask_imag
        est_comp_imag = spec_comp_real * mask_imag + spec_comp_imag * mask_real
        est_compressed = torch.stack([est_comp_real, est_comp_imag], dim=-1)  # [B, F, T_spec, 2]

        # Power Decompression: Y = Y_c * |Y_c|^(1/c - 1)
        est_comp_complex = torch.view_as_complex(est_compressed)
        est_mag = est_comp_complex.abs().clamp(min=1e-12)
        est_spec = est_comp_complex * est_mag.pow(1.0 / self.input_compression - 1.0)  # [B, F, T_spec]

        # Restore the Nyquist bin filled with zeros
        est_spec = F.pad(est_spec, (0, 0, 0, 1))  # [B, F_fft, T_spec]

        # Inverse STFT
        est_wav_norm = torch.istft(
            est_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=True,
            length=T_wav,
        )

        # Denormalize output waveform
        est_wav = s * est_wav_norm + m

        return est_wav, est_compressed

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        est_wav, _ = self.forward(noisy)
        return est_wav

    def compute_loss(self, noisy: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # Get mean and std from noisy to normalize targets consistently
        m = noisy.mean(dim=-1, keepdim=True)
        s = noisy.std(dim=-1, keepdim=True).clamp(min=1e-8)

        # Forward pass to get estimate and compressed estimate spectrogram (in normalized space)
        estimate, est_compressed = self.forward(noisy)

        # Normalize clean target
        clean_norm = (clean - m) / s

        # 1. Target compressed spectrogram (normalized)
        clean_stft = torch.stft(
            clean_norm,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True,
            center=True,
        )[:, :-1, :]  # Discard Nyquist bin
        clean_compressed = torch.view_as_real(clean_stft) * clean_stft.abs().unsqueeze(-1).clamp(min=1e-12).pow(self.input_compression - 1.0)

        # 2. Magnitude Loss: MSE between compressed magnitudes
        mag_clean = clean_compressed.norm(dim=-1)
        mag_est = est_compressed.norm(dim=-1)
        loss_mag = F.mse_loss(mag_est, mag_clean)

        # 3. Complex Loss: MSE between compressed complex spectrograms
        loss_comp = F.mse_loss(est_compressed, clean_compressed)

        # 4. Consistency Loss: MSE between clean compressed spec and the compressed STFT of reconstructed waveform
        est_wav_norm = (estimate - m) / s
        est_rec_stft = torch.stft(
            est_wav_norm,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True,
            center=True,
        )[:, :-1, :]
        est_rec_compressed = torch.view_as_real(est_rec_stft) * est_rec_stft.abs().unsqueeze(-1).clamp(min=1e-12).pow(self.input_compression - 1.0)
        loss_con = F.mse_loss(est_rec_compressed, clean_compressed)

        # 5. Waveform Loss: L1 distance in raw waveform domain
        loss_wav = F.l1_loss(estimate, clean)

        # Combined loss: 0.3 * mag + 0.2 * comp + 0.3 * con + 0.2 * wav
        total_loss = 0.3 * loss_mag + 0.2 * loss_comp + 0.3 * loss_con + 0.2 * loss_wav

        metrics_dict = {
            "loss_mag": loss_mag.item(),
            "loss_comp": loss_comp.item(),
            "loss_con": loss_con.item(),
            "loss_wav": loss_wav.item(),
            "total_loss": total_loss.item(),
        }

        return estimate, total_loss, metrics_dict

