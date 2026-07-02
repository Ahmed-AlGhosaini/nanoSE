# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import fast_bss_eval


# -------------------------------
# Auxiliar non-linear generation function
# -------------------------------
def gen_non_linearity(A, non_linearity):
    """
    Returns required activation for a tensor based on the inputs
    """
    if non_linearity == "tanh":
        return torch.tanh(A)
    elif non_linearity == "sigmoid":
        return torch.sigmoid(A)
    elif non_linearity == "relu":
        return torch.relu(A)
    elif non_linearity == "quantTanh":
        return torch.clamp(A, -1.0, 1.0)
    elif non_linearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return torch.clamp(A, 0.0, 1.0)
    elif non_linearity == "quantSigm4":
        A = (A + 2.0) / 4.0
        return torch.clamp(A, 0.0, 1.0)
    elif callable(non_linearity):
        return non_linearity(A)
    else:
        raise ValueError(
            "non_linearity must be one of ['tanh', 'sigmoid', 'relu', 'quantTanh', 'quantSigm', 'quantSigm4'] or callable"
        )


# -------------------------------
# Comfi-FastGRNN cell
# -------------------------------
class ComfiFastGRNNCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        gate_non_linearity="sigmoid",
        update_non_linearity="tanh",
        w_rank=None,
        u_rank=None,
        zeta_init=1.0,
        nu_init=-4.0,
        lambda_init=0.0,
        gamma_init=0.999,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_non_linearity = gate_non_linearity
        self.update_non_linearity = update_non_linearity
        self.w_rank = w_rank
        self.u_rank = u_rank

        # --- Weight definitions ---
        if w_rank is None:
            self.w_matrix = nn.Parameter(torch.empty(input_size, hidden_size))
            nn.init.xavier_uniform_(self.w_matrix)
        else:
            self.w_matrix_1 = nn.Parameter(torch.empty(input_size, w_rank))
            self.w_matrix_2 = nn.Parameter(torch.empty(w_rank, hidden_size))
            nn.init.xavier_uniform_(self.w_matrix_1)
            nn.init.xavier_uniform_(self.w_matrix_2)

        if u_rank is None:
            self.u_matrix = nn.Parameter(torch.empty(hidden_size, hidden_size))
            nn.init.xavier_uniform_(self.u_matrix)
        else:
            self.u_matrix_1 = nn.Parameter(torch.empty(hidden_size, u_rank))
            self.u_matrix_2 = nn.Parameter(torch.empty(u_rank, hidden_size))
            nn.init.xavier_uniform_(self.u_matrix_1)
            nn.init.xavier_uniform_(self.u_matrix_2)

        # --- Biases ---
        self.bias_gate = nn.Parameter(torch.ones(1, hidden_size))
        self.bias_update = nn.Parameter(torch.ones(1, hidden_size))

        # --- Scalars ---
        self.zeta = nn.Parameter(torch.tensor([[zeta_init]], dtype=torch.float32))
        self.nu = nn.Parameter(torch.tensor([[nu_init]], dtype=torch.float32))
        self.lambd = nn.Parameter(torch.tensor([lambda_init], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float32))

    def forward(self, x, h_prev):
        # Compute W*x
        if self.w_rank is None:
            W = x @ self.w_matrix
        else:
            W = x @ self.w_matrix_1 @ self.w_matrix_2

        # Compute U*h_prev
        if self.u_rank is None:
            U = h_prev @ self.u_matrix
        else:
            U = h_prev @ self.u_matrix_1 @ self.u_matrix_2

        # Gates
        z = gen_non_linearity(W + U + self.bias_gate, self.gate_non_linearity)
        h_hat = gen_non_linearity(W + U + self.bias_update, self.update_non_linearity)

        # FastGRNN update
        h = z * h_prev + (torch.sigmoid(self.zeta) * (1 - z) + torch.sigmoid(self.nu)) * h_hat

        # Comfi-FastGRNN update
        gamma_clamped = torch.clamp(self.gamma, 0.0, 1.0)
        h_t_comfi = gamma_clamped * h + (1 - gamma_clamped) * self.lambd

        return h_t_comfi


# -------------------------------
# Comfi-FastGRNN layer
# -------------------------------
class ComfiFastGRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        gate_non_linearity: str = "sigmoid",
        update_non_linearity: str = "tanh",
        w_rank: int | None = None,
        u_rank: int | None = None,
        zeta_init: float = 1.0,
        nu_init: float = -4.0,
        lambda_init: float = 0.0,
        gamma_init: float = 0.999,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.cells_fwd = nn.ModuleList()
        self.cells_bwd = nn.ModuleList() if bidirectional else None

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * self.num_directions

            self.cells_fwd.append(
                ComfiFastGRNNCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    gate_non_linearity=gate_non_linearity,
                    update_non_linearity=update_non_linearity,
                    w_rank=w_rank,
                    u_rank=u_rank,
                    zeta_init=zeta_init,
                    nu_init=nu_init,
                    lambda_init=lambda_init,
                    gamma_init=gamma_init,
                )
            )

            if bidirectional:
                self.cells_bwd.append(
                    ComfiFastGRNNCell(
                        input_size=in_size,
                        hidden_size=hidden_size,
                        gate_non_linearity=gate_non_linearity,
                        update_non_linearity=update_non_linearity,
                        w_rank=w_rank,
                        u_rank=u_rank,
                        zeta_init=zeta_init,
                        nu_init=nu_init,
                        lambda_init=lambda_init,
                        gamma_init=gamma_init,
                    )
                )

    def forward(self, x, h0=None):
        if not self.batch_first:
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.size()

        if h0 is None:
            h0 = x.new_zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
            )

        layer_input = x
        h_n = []

        for layer in range(self.num_layers):
            fw_cell = self.cells_fwd[layer]
            h_fw = h0[layer * self.num_directions + 0]
            fw_outs = []

            for t in range(seq_len):
                h_fw = fw_cell(layer_input[:, t, :], h_fw)
                fw_outs.append(h_fw.unsqueeze(1))

            fw_out = torch.cat(fw_outs, dim=1)

            if self.bidirectional:
                bw_cell = self.cells_bwd[layer]
                h_bw = h0[layer * self.num_directions + 1]
                bw_outs = []

                for t in reversed(range(seq_len)):
                    h_bw = bw_cell(layer_input[:, t, :], h_bw)
                    bw_outs.append(h_bw.unsqueeze(1))

                bw_outs.reverse()
                bw_out = torch.cat(bw_outs, dim=1)

                layer_out = torch.cat([fw_out, bw_out], dim=2)
                h_n.extend([h_fw, h_bw])
            else:
                layer_out = fw_out
                h_n.append(h_fw)

            if self.dropout > 0.0 and layer < self.num_layers - 1:
                layer_out = F.dropout(layer_out, p=self.dropout, training=self.training)

            layer_input = layer_out

        output = layer_input
        h_n = torch.stack(h_n, dim=0)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n


# -------------------------------
# STFT and Feature Engineering
# -------------------------------
class STFTLayer(nn.Module):
    def __init__(self, block_len, block_shift, window=None):
        super().__init__()
        self.block_len = block_len
        self.block_shift = block_shift
        if window is not None:
            self.register_buffer("window", window)
        else:
            self.window = None

    def forward(self, x):
        stft = torch.stft(
            x,
            n_fft=self.block_len,
            hop_length=self.block_shift,
            win_length=self.block_len,
            window=self.window,
            center=True,
            return_complex=True,
        )
        return stft.transpose(1, 2)


class ChannelWiseFeatureReorientation(nn.Module):
    def __init__(self, input_freq_dim=257):
        super().__init__()
        self.input_freq_dim = int(input_freq_dim)
        self.window_size = 48
        overlap = 0.33
        self.hop_size = math.ceil(self.window_size * (1 - overlap))
        self.n_bands = math.ceil(((self.input_freq_dim - self.window_size) / self.hop_size) + 1)

    def forward(self, x):
        batch_size, time_dim, freq_dim = x.shape
        subbands = []
        for i in range(self.n_bands):
            start = i * self.hop_size
            end = start + self.window_size
            if end > self.input_freq_dim:
                subband = x[:, :, start:self.input_freq_dim]
                padding = torch.zeros(
                    (batch_size, time_dim, end - self.input_freq_dim),
                    device=x.device,
                    dtype=x.dtype,
                )
                subband = torch.cat([subband, padding], dim=-1)
            else:
                subband = x[:, :, start:end]
            subbands.append(subband)

        return torch.stack(subbands, dim=2)


# -------------------------------
# Convolution Blocks
# -------------------------------
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sepconv1 = SeparableConv2d(in_channels, 32)
        self.sepconv2 = SeparableConv2d(32, 64)
        self.sepconv3 = SeparableConv2d(64, 96)
        self.sepconv4 = SeparableConv2d(96, 128)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(self.sepconv1(x))
        x = self.relu(self.sepconv2(x))
        x = self.pool(x)
        x = self.relu(self.sepconv3(x))
        x = self.pool(x)
        x = self.relu(self.sepconv4(x))
        x = self.pool(x)
        return x


# -------------------------------
# Complex Ratio Masking
# -------------------------------
class ComplexRatioMask(nn.Module):
    def __init__(self, masking_mode="E"):
        super().__init__()
        self.masking_mode = masking_mode

    def forward(self, real, imag, m_real, m_imag):
        if self.masking_mode == "E":
            m_norm = torch.sqrt(m_real**2 + m_imag**2 + 1e-12)
            scale = torch.tanh(m_norm) / m_norm
            est_real = scale * (real * m_real - imag * m_imag)
            est_imag = scale * (imag * m_real + real * m_imag)
            return torch.complex(est_real, est_imag)
        else:
            est_real = real * m_real - imag * m_imag
            est_imag = real * m_imag + imag * m_real
            return torch.complex(est_real, est_imag)


# -------------------------------
# Fast-ULCNet Model (ComfiFastGRNN architecture)
# -------------------------------
class ComfiFastGRNNModel(nn.Module):
    def __init__(
        self,
        block_len: int = 512,
        block_shift: int = 128,
        compression_factor: float = 0.3,
        bidirectional_frnn_units: int = 64,
        sub_band_rnn_units: int = 128,
        CRM_type: str = "E",
    ):
        super().__init__()
        self.block_len = block_len
        self.block_shift = block_shift
        self.compression_factor = compression_factor
        self.freq_dim = int(self.block_len // 2 + 1)

        self.bidirectional_frnn_units = bidirectional_frnn_units
        self.sub_band_rnn_units = sub_band_rnn_units

        # Analysis Window Setup (defaulting to standard Hann window)
        window = torch.hann_window(self.block_len)
        self.stft_layer = STFTLayer(self.block_len, self.block_shift, window)
        self.reorientation = ChannelWiseFeatureReorientation(input_freq_dim=self.freq_dim)

        self.conv_block = ConvBlock(in_channels=8)

        self.freq_rnn = ComfiFastGRNN(
            input_size=128,
            hidden_size=self.bidirectional_frnn_units,
            bidirectional=True,
            batch_first=True,
        )

        self.pointwise_conv = nn.Conv2d(
            2 * self.bidirectional_frnn_units, 64, kernel_size=(1, 1), bias=False
        )

        self.sub_band_rnn1 = ComfiFastGRNN(
            input_size=192,
            hidden_size=self.sub_band_rnn_units,
            num_layers=2,
            batch_first=True,
        )

        self.sub_band_rnn2 = ComfiFastGRNN(
            input_size=192,
            hidden_size=self.sub_band_rnn_units,
            num_layers=2,
            batch_first=True,
        )

        self.fc1 = nn.Linear(2 * self.sub_band_rnn_units, self.freq_dim)
        self.fc2 = nn.Linear(self.freq_dim, self.freq_dim)

        self.cnn_block = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.ReLU(),
        )
        self.complex_mask_conv = nn.Conv2d(32, 2, kernel_size=(1, 1), bias=False)
        self.crm_layer = ComplexRatioMask(masking_mode=CRM_type)

    def feature_preprocessing(self, x):
        real, imag = x.real, x.imag
        c = self.compression_factor
        comp_real = torch.sign(real) * torch.pow(torch.abs(real) + 1e-12, c)
        comp_imag = torch.sign(imag) * torch.pow(torch.abs(imag) + 1e-12, c)
        mag = torch.sqrt(comp_real**2 + comp_imag**2 + 1e-12)
        # Compute cos and sin of phase directly to avoid atan2/cos/sin gradient instability
        cos_phase = comp_real / mag
        sin_phase = comp_imag / mag
        return mag, cos_phase, sin_phase, comp_real, comp_imag

    def intermediate_feature_computation(self, intermediate_mask, cos_phase, sin_phase):
        inter_r = intermediate_mask * cos_phase
        inter_i = intermediate_mask * sin_phase
        inter_feat = torch.stack([inter_r, inter_i], dim=1)
        return inter_feat

    def power_law_decompression(self, x):
        final_real, final_imag = x.real, x.imag
        inv_c = 1.0 / self.compression_factor
        dec_real = torch.sign(final_real) * torch.pow(torch.abs(final_real) + 1e-12, inv_c)
        dec_imag = torch.sign(final_imag) * torch.pow(torch.abs(final_imag) + 1e-12, inv_c)
        return torch.complex(dec_real, dec_imag)

    def forward(self, x):
        # x shape: [B, T_wav]
        B, T_wav = x.shape

        # Calculate mean and std per sample in batch
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True).clamp(min=1e-8)

        # Normalize input waveform
        norm_x = (x - m) / s

        # 1. STFT and Feature Preprocessing
        stft_data = self.stft_layer(norm_x)
        mag, cos_phase, sin_phase, real, imag = self.feature_preprocessing(stft_data)

        # 2. Reorientation: [B, T, F] -> [B, T, n_bands, window_size]
        features = self.reorientation(mag)
        # To [B, C, T, F] for PyTorch Conv2d
        features = features.permute(0, 2, 1, 3)

        # 3. Conv Block with MaxPools
        x_conv = self.conv_block(features)

        # 4. Frequency RNN
        B_conv, C_conv, T_conv, F_red = x_conv.shape
        x_rnn = x_conv.permute(0, 2, 3, 1).reshape(B_conv * T_conv, F_red, C_conv)
        frnn_out, _ = self.freq_rnn(x_rnn)
        frnn_out = frnn_out.view(B_conv, T_conv, F_red, -1).permute(0, 3, 1, 2)

        # 5. Temporal Sub-band RNNs
        x_pt = F.relu(self.pointwise_conv(frnn_out))
        x_flat = x_pt.permute(0, 2, 3, 1).reshape(B_conv, T_conv, -1)

        sub1, sub2 = torch.chunk(x_flat, 2, dim=-1)
        r1, _ = self.sub_band_rnn1(sub1)
        r2, _ = self.sub_band_rnn2(sub2)
        concatenated = torch.cat([r1, r2], dim=-1)

        # 6. Mask Computation
        mask = F.relu(self.fc1(concatenated))
        mask = F.relu(self.fc2(mask))

        # 7. Intermediate Features for Stage 2
        inter_feat = self.intermediate_feature_computation(mask, cos_phase, sin_phase)

        # 8. CNN block for final mask
        cnn_out = self.cnn_block(inter_feat)
        c_mask = F.relu(self.complex_mask_conv(cnn_out))

        # 9. Complex Ratio Mask (CRM) and Decompression
        m_real, m_imag = c_mask[:, 0, :, :], c_mask[:, 1, :, :]
        est_speech_comp = self.crm_layer(real, imag, m_real, m_imag)
        estimated_speech = self.power_law_decompression(est_speech_comp)

        # 10. Inverse STFT to return waveform
        estimated_speech_trans = estimated_speech.transpose(1, 2)
        est_wav_norm = torch.istft(
            estimated_speech_trans,
            n_fft=self.block_len,
            hop_length=self.block_shift,
            win_length=self.block_len,
            window=self.stft_layer.window,
            center=True,
            length=T_wav,
        )

        # Denormalize output waveform
        est_wav = s * est_wav_norm + m

        return est_wav, estimated_speech

    def enhance(self, noisy):
        est_wav, _ = self.forward(noisy)
        return est_wav

    def compute_loss(self, noisy, clean):
        # Normalize clean targets consistently using noisy statistics
        m = noisy.mean(dim=-1, keepdim=True)
        s = noisy.std(dim=-1, keepdim=True).clamp(min=1e-8)

        estimate, estimated_speech = self.forward(noisy)
        
        # Normalize clean waveform for spec loss
        clean_norm = (clean - m) / s
        stft_clean = self.stft_layer(clean_norm)

        spec_loss = F.mse_loss(
            torch.view_as_real(estimated_speech), torch.view_as_real(stft_clean)
        )

        sdr_loss = fast_bss_eval.si_sdr_loss(
            estimate.unsqueeze(1), clean.unsqueeze(1), zero_mean=True, clamp_db=30
        ).mean()

        loss = spec_loss + 0.01 * sdr_loss

        metrics_dict = {
            "spec_loss": spec_loss.item(),
            "sdr_loss": sdr_loss.item(),
        }

        return estimate, loss, metrics_dict
