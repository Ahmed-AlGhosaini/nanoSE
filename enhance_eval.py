# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import argparse
import os
import time

import matplotlib
import numpy as np
import torch
import torchaudio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import VoiceBankDemandDataset
from metrics import SpeechMetrics
from model import enhance_waveform, get_model, waveform_to_spectrogram


def save_spec_plot(noisy, clean, enhanced, save_path):
    """
    Saves a plot comparing noisy, clean, and enhanced spectrograms.
    """
    # Convert to CPU numpy magnitudes for plotting
    noisy_spec = (
        torch.abs(waveform_to_spectrogram(noisy.unsqueeze(0))[:, :256, :])
        .squeeze(0)
        .cpu()
        .numpy()
    )
    clean_spec = (
        torch.abs(waveform_to_spectrogram(clean.unsqueeze(0))[:, :256, :])
        .squeeze(0)
        .cpu()
        .numpy()
    )
    enhanced_spec = (
        torch.abs(waveform_to_spectrogram(enhanced.unsqueeze(0))[:, :256, :])
        .squeeze(0)
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Log scale magnitudes for visual contrast
    axes[0].imshow(np.log1p(noisy_spec), aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Noisy Input", color="#ef4444", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.log1p(enhanced_spec), aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Model Enhanced", color="#3b82f6", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(np.log1p(clean_spec), aspect="auto", origin="lower", cmap="magma")
    axes[2].set_title(
        "Clean Reference", color="#10b981", fontsize=12, fontweight="bold"
    )
    axes[2].axis("off")

    fig.patch.set_facecolor("#1e293b")
    for ax in axes:
        ax.set_facecolor("#1e293b")

    plt.tight_layout()
    plt.savefig(save_path, facecolor="#1e293b", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Enhance a subset of the test set and build an interactive webpage."
    )
    parser.add_argument(
        "--num-samples", "-n", type=int, default=5, help="Number of samples to process"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None, help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="eval_results", help="Output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="crn",
        choices=["crn", "crntiny", "fastenhancer", "fastenhancercompact", "glumaskd", "glumasked", "comfi_fastgrnn"],
        help="Fallback model architecture type if it cannot be auto-detected from filename",
    )
    args = parser.parse_args()

    # Find the latest checkpoint if not provided
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        pt_files = []
        # Search recursively in runs/
        if os.path.exists("runs"):
            for root, _, files in os.walk("runs"):
                for f in files:
                    if f.endswith(".pt"):
                        pt_files.append(os.path.join(root, f))
        # Search recursively in checkpoints/
        if os.path.exists("checkpoints"):
            for root, _, files in os.walk("checkpoints"):
                for f in files:
                    if f.endswith(".pt"):
                        pt_files.append(os.path.join(root, f))

        if pt_files:
            # Sort by path / name (latest timestamp / epoch)
            pt_files.sort()
            checkpoint_path = pt_files[-1]
            print(f"No checkpoint specified. Selected latest: {checkpoint_path}")
        else:
            print("Error: No checkpoints found in 'runs/' or 'checkpoints/' folders.")
            return

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Choose device (CUDA GPU preferred, then Metal GPU, then CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running evaluation on device: {device.type.upper()}")

    # Determine model type (auto-detect based on checkpoint filename, fallback to --model)
    model_name = args.model
    filename_lower = os.path.basename(checkpoint_path).lower()
    if "fastenhancer" in filename_lower:
        model_name = "fastenhancer"
    elif "crn" in filename_lower:
        model_name = "crn"

    # 1. Load model
    print(f"Loading {model_name} model...")
    model = get_model(model_name).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 2. Initialize metrics
    metrics = SpeechMetrics(device=device)

    # 3. Load dataset (mock_size speeds up caching significantly)
    print("Loading dataset...")
    # For CUDA and MPS devices, load the whole dataset onto the GPU device
    ds_device = device if device.type in ("cuda", "mps") else None
    dataset = VoiceBankDemandDataset(split="test", mock_size=args.num_samples, device=ds_device)

    samples_html_list = []

    total_delta_sdr = 0.0
    total_noisy_sdr = 0.0
    total_enhanced_sdr = 0.0
    total_noisy_pesq = 0.0
    total_enhanced_pesq = 0.0
    total_noisy_estoi = 0.0
    total_enhanced_estoi = 0.0
    total_enhanced_dnsmos = 0.0

    actual_samples_processed = min(args.num_samples, len(dataset))

    print(f"\nProcessing {actual_samples_processed} samples...")
    for idx in range(actual_samples_processed):
        sample_id = dataset.ids[idx]
        print(f"[{idx+1}/{actual_samples_processed}] Enhancing sample: {sample_id}")

        # Extract full length waveform from cache (scaled from int16 to float32)
        noisy_wav = dataset.noisy_cached[idx].to(torch.float32) / 32767.0
        clean_wav = dataset.clean_cached[idx].to(torch.float32) / 32767.0

        # Enhance audio
        noisy_wav_device = noisy_wav.unsqueeze(0).to(device)
        with torch.no_grad():
            enhanced_wav = enhance_waveform(model, noisy_wav_device).squeeze(0).cpu()

        # Compute metrics
        noisy_sdr = metrics.compute_si_sdr(clean_wav, noisy_wav).item()
        enhanced_sdr = metrics.compute_si_sdr(clean_wav, enhanced_wav).item()
        delta_sdr = enhanced_sdr - noisy_sdr

        noisy_eval = metrics.compute_eval_metrics(clean_wav, noisy_wav)
        enhanced_eval = metrics.compute_eval_metrics(clean_wav, enhanced_wav)

        noisy_pesq = noisy_eval["pesq"]
        enhanced_pesq = enhanced_eval["pesq"]
        noisy_estoi = noisy_eval["estoi"]
        enhanced_estoi = enhanced_eval["estoi"]
        enhanced_dnsmos = enhanced_eval["dnsmos"]

        # Track global averages
        total_delta_sdr += delta_sdr
        total_noisy_sdr += noisy_sdr
        total_enhanced_sdr += enhanced_sdr
        total_noisy_pesq += noisy_pesq
        total_enhanced_pesq += enhanced_pesq
        total_noisy_estoi += noisy_estoi
        total_enhanced_estoi += enhanced_estoi
        total_enhanced_dnsmos += enhanced_dnsmos

        # File paths relative to output directory (for index.html links)
        noisy_filename = f"sample_{sample_id}_noisy.wav"
        clean_filename = f"sample_{sample_id}_clean.wav"
        enhanced_filename = f"sample_{sample_id}_enhanced.wav"
        spec_filename = f"sample_{sample_id}_spec.png"

        # Save audio files
        torchaudio.save(
            os.path.join(args.output_dir, noisy_filename), noisy_wav.unsqueeze(0).cpu(), 16000
        )
        torchaudio.save(
            os.path.join(args.output_dir, clean_filename), clean_wav.unsqueeze(0).cpu(), 16000
        )
        torchaudio.save(
            os.path.join(args.output_dir, enhanced_filename),
            enhanced_wav.unsqueeze(0).cpu(),
            16000,
        )

        # Save spectrogram plot
        save_spec_plot(
            noisy_wav,
            clean_wav,
            enhanced_wav,
            os.path.join(args.output_dir, spec_filename),
        )

        # Generate HTML snippet for this sample
        sample_html = f"""
        <div class="sample-card">
            <div class="sample-header">
                <span class="sample-id">🎵 Sample ID: {sample_id}</span>
                <div class="metric-badges">
                    <span class="badge badge-sdr">SI-SDR Δ: <span class="improvement">+{delta_sdr:.2f} dB</span> (Noisy: {noisy_sdr:.1f} dB → Enhanced: {enhanced_sdr:.1f} dB)</span>
                    <span class="badge badge-pesq">PESQ: {enhanced_pesq:.2f} (Noisy: {noisy_pesq:.2f})</span>
                    <span class="badge badge-estoi">STOI: {enhanced_estoi:.3f} (Noisy: {noisy_estoi:.3f})</span>
                    <span class="badge badge-dns">DNSMOS: {enhanced_dnsmos:.2f}</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="spec-container">
                    <img class="spec-image" src="{spec_filename}" alt="Spectrogram for sample {sample_id}">
                </div>
                <div class="audio-container">
                    <div class="audio-row">
                        <span class="audio-label label-noisy">🔴 Noisy Input Waveform</span>
                        <audio controls src="{noisy_filename}"></audio>
                    </div>
                    <div class="audio-row">
                        <span class="audio-label label-enhanced">🔵 Model Enhanced Output</span>
                        <audio controls src="{enhanced_filename}"></audio>
                    </div>
                    <div class="audio-row">
                        <span class="audio-label label-clean">🟢 Clean Reference (Ground Truth)</span>
                        <audio controls src="{clean_filename}"></audio>
                    </div>
                </div>
            </div>
        </div>
        """
        samples_html_list.append(sample_html)

    # Compute averages
    avg_delta_sdr = total_delta_sdr / actual_samples_processed
    avg_pesq = total_enhanced_pesq / actual_samples_processed
    avg_estoi = total_enhanced_estoi / actual_samples_processed
    avg_dnsmos = total_enhanced_dnsmos / actual_samples_processed

    # Prepare index.html content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NanoSE Speech Enhancement Evaluation</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.4);
            --card-border: rgba(255, 255, 255, 0.06);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-primary: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.15);
            --color-noisy: #ef4444;
            --color-clean: #10b981;
            --color-enhanced: #3b82f6;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            background-image: radial-gradient(circle at top, #1e1b4b 0%, #0b0f19 80%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem 1.5rem;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-background-fill-color: transparent;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        header p {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}

        .summary-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 3rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }}

        .summary-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
        }}

        .stat-box {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
        }}

        .stat-box:hover {{
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 0 15px var(--accent-glow);
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.25rem;
        }}

        .stat-label {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .samples-list {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }}

        .sample-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}

        .sample-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.25);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3), 0 0 20px var(--accent-glow);
        }}

        .sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }}

        .sample-id {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #f8fafc;
        }}

        .metric-badges {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}

        .badge {{
            font-size: 0.85rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        }}

        .badge-sdr {{ background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); color: #60a5fa; }}
        .badge-pesq {{ background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3); color: #c084fc; }}
        .badge-estoi {{ background: rgba(236, 72, 153, 0.1); border: 1px solid rgba(236, 72, 153, 0.3); color: #f472b6; }}
        .badge-dns {{ background: rgba(20, 184, 166, 0.1); border: 1px solid rgba(20, 184, 166, 0.3); color: #2dd4bf; }}

        .improvement {{
            color: #10b981;
            font-weight: 600;
        }}

        .content-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }}

        @media (min-width: 900px) {{
            .content-grid {{
                grid-template-columns: 1.2fr 1fr;
            }}
        }}

        .spec-container {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: #1e293b;
        }}

        .spec-image {{
            width: 100%;
            display: block;
            height: auto;
            transition: transform 0.3s ease;
        }}

        .spec-container:hover .spec-image {{
            transform: scale(1.02);
        }}

        .audio-container {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 1.25rem;
        }}

        .audio-row {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .audio-label {{
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .label-noisy {{ color: var(--color-noisy); }}
        .label-clean {{ color: var(--color-clean); }}
        .label-enhanced {{ color: var(--color-enhanced); }}

        audio {{
            width: 100%;
            height: 36px;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NanoSE Speech Enhancement Evaluation</h1>
            <p>Interactive listening and metrics platform powered by lightweight Convolutional Recurrent Network (CRN)</p>
            <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #64748b;">Checkpoint: <code>{os.path.basename(checkpoint_path)}</code></p>
        </header>

        <div class="summary-card">
            <div class="summary-title">📊 Global Test Subset Summary</div>
            <div class="summary-grid">
                <div class="stat-box">
                    <div class="stat-value" style="color: #10b981;">+{avg_delta_sdr:.2f} dB</div>
                    <div class="stat-label">Avg SI-SDR Improvement</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color: #c084fc;">{avg_pesq:.2f}</div>
                    <div class="stat-label">Avg Enhanced PESQ</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color: #f472b6;">{avg_estoi:.3f}</div>
                    <div class="stat-label">Avg Enhanced STOI</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color: #2dd4bf;">{avg_dnsmos:.2f}</div>
                    <div class="stat-label">Avg Enhanced DNSMOS</div>
                </div>
            </div>
        </div>

        <div class="samples-list">
            {"".join(samples_html_list)}
        </div>
    </div>
</body>
</html>
"""

    html_path = os.path.join(args.output_dir, "index.html")
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"\nSuccessfully processed {actual_samples_processed} samples.")
    print(f"Results saved to directory: '{args.output_dir}'")
    print(
        f"Open this file in your browser to listen and compare: '{os.path.abspath(html_path)}'"
    )


if __name__ == "__main__":
    main()
