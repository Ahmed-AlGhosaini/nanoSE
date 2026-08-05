# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import argparse
import os
import random
import time
import warnings

# Suppress noisy PyTorch MPS-specific resizing warnings
# pylint: disable-module-level-import
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*resized since it had shape.*"
)

# flake8: noqa: E402
import fast_bss_eval
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import VoiceBankDemandDataset
from metrics import SpeechMetrics
from model import enhance_waveform, get_model, waveform_to_spectrogram


# Autocast helper to safely handle MPS and CPU mixed precision
class AutocastManager:
    def __init__(self, device):
        self.device = device
        self.autocast = None

    def __enter__(self):
        if self.device.type in ["mps", "cuda"]:
            try:
                self.autocast = torch.amp.autocast(
                    device_type=self.device.type, dtype=torch.float16
                )
                self.autocast.__enter__()
            except Exception:
                pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.autocast is not None:
            self.autocast.__exit__(exc_type, exc_val, exc_tb)


def create_spec_plot(noisy, clean, enhanced):
    """
    Creates a matplotlib figure comparing noisy, clean, and enhanced spectrograms.
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Log scale magnitudes for visual contrast
    axes[0].imshow(np.log1p(noisy_spec), aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Noisy Spectrogram")
    axes[0].set_ylabel("Frequency Bins")
    axes[0].set_xlabel("Time Frames")

    axes[1].imshow(np.log1p(clean_spec), aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Clean Spectrogram")
    axes[1].set_xlabel("Time Frames")

    axes[2].imshow(
        np.log1p(enhanced_spec), aspect="auto", origin="lower", cmap="viridis"
    )
    axes[2].set_title("Enhanced Spectrogram")
    axes[2].set_xlabel("Time Frames")

    plt.tight_layout()
    return fig


def run_validation(
    model,
    test_loader,
    device,
    metrics,
    eval_audio_indices,
    writer,
    global_step,
    eval_samples_limit=20,
):
    model.eval()
    val_si_sdr = 0.0
    val_pesq = []
    val_estoi = []
    val_dnsmos = []

    eval_count = 0

    val_bar = tqdm(test_loader, desc=f"Validation (Step {global_step})", leave=False)
    with torch.no_grad():
        for idx, (noisy, clean) in enumerate(val_bar):
            noisy, clean = noisy.to(device), clean.to(device)
            enhanced = enhance_waveform(model, noisy)

            # Log audio and spectrogram plots to TensorBoard for uniformly selected samples
            if idx in eval_audio_indices:
                writer.add_audio(
                    f"audio/noisy/{idx}",
                    noisy[0].cpu(),
                    global_step,
                    sample_rate=16000,
                )
                writer.add_audio(
                    f"audio/clean/{idx}",
                    clean[0].cpu(),
                    global_step,
                    sample_rate=16000,
                )
                writer.add_audio(
                    f"audio/enhanced/{idx}",
                    enhanced[0].cpu(),
                    global_step,
                    sample_rate=16000,
                )

                fig = create_spec_plot(noisy[0], clean[0], enhanced[0])
                writer.add_figure(f"plots/spectrogram/{idx}", fig, global_step)
                plt.close(fig)

            # Compute fast test SI-SDR
            current_val_sdr = metrics.compute_si_sdr(clean, enhanced).mean().item()
            val_si_sdr += current_val_sdr

            # Compute heavy eval metrics on a subset
            if eval_count < eval_samples_limit:
                batch_rem = eval_samples_limit - eval_count
                batch_ref = clean[:batch_rem]
                batch_est = enhanced[:batch_rem]

                scores = metrics.compute_eval_metrics(batch_ref, batch_est)
                val_pesq.append(scores["pesq"])
                val_estoi.append(scores["estoi"])
                val_dnsmos.append(scores["dnsmos"])
                eval_count += batch_ref.size(0)

            val_bar.set_postfix(sdr=f"{current_val_sdr:.1f}dB")

    val_si_sdr /= len(test_loader)
    avg_pesq = np.nanmean(val_pesq) if val_pesq else float("nan")
    avg_estoi = np.nanmean(val_estoi) if val_estoi else float("nan")
    avg_dnsmos = np.nanmean(val_dnsmos) if val_dnsmos else float("nan")

    writer.add_scalar("eval/dnsmos", avg_dnsmos, global_step)
    writer.add_scalar("eval/estoi", avg_estoi, global_step)
    writer.add_scalar("eval/pesq", avg_pesq, global_step)
    writer.add_scalar("eval/si-sdr", val_si_sdr, global_step)

    return {
        "val_si_sdr": val_si_sdr,
        "pesq": avg_pesq,
        "estoi": avg_estoi,
        "dnsmos": avg_dnsmos,
    }


def save_metrics_to_jsonl(metrics_dict, epoch, log_dir):
    # Add the epoch number to the metrics dict before saving
    metrics_dict["epoch"] = epoch

    import json

    os.makedirs(log_dir, exist_ok=True)
    jsonl_path = os.path.join(log_dir, "validation_metrics.jsonl")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics_dict) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Speech Enhancement Training Pipeline")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--wd", type=float, default=None, help="Weight decay")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Linear LR warmup length in steps (default 500, capped at half the run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a quick dry-run with a tiny dataset fraction",
    )
    parser.add_argument(
        "--compile",
        action="store_const",
        const=True,
        default=None,
        help="Compile model using torch.compile()",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="The random number generator seed."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "crn",
            "crntiny",
            "fastenhancer",
            "fastenhancercompact",
            "glumaskd",
            "glumasked",
            "comfi_fastgrnn",
        ],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--remix",
        action="store_const",
        const=True,
        default=None,
        help="Apply dynamic random noise remixing in the batch during training.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Label for this training run. Defaults to model name if not specified.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.py",
        help="Path to python configuration file defining override variables.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name to identify the run, used in log directory formatting.",
    )
    args = parser.parse_args()

    # 1. Load the default config file first
    config_vars = {}
    default_config_path = "config/default_config.py"
    if os.path.exists(default_config_path):
        try:
            with open(default_config_path, "r") as f:
                exec(f.read(), {}, config_vars)
        except Exception as e:
            print(
                f"Warning: Failed to load default config from {default_config_path}: {e}"
            )

    # 2. Load the custom config file if specified and different from default
    if args.config and args.config != default_config_path:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        try:
            with open(args.config, "r") as f:
                exec(f.read(), {}, config_vars)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            raise RuntimeError(f"Error executing configuration file {args.config}: {e}")

    # 3. Apply CLI overrides (any argument that is explicitly passed/not None)
    for name_var, val in vars(args).items():
        if name_var == "config" or name_var == "dry_run":
            continue
        if val is not None:
            config_vars[name_var] = val
            print(f"Command line override: {name_var} = {val}")

    # 4. Apply all configuration variables back to args
    for name_var, val in list(config_vars.items()):
        if (
            name_var.startswith("__")
            or name_var == "dry_run"
            or isinstance(val, type(os))
            or isinstance(val, type)
        ):
            continue
        setattr(args, name_var, val)

    if args.dry_run:
        args.epochs = 1
        print("Dry-run mode active: forcing epochs = 1")

    # Set random seed for reproducibility / determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # NanoGPT-style floating point precision tweak
    torch.set_float32_matmul_precision("high")

    # Choose device (CUDA GPU preferred, then Metal GPU, then CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running training on device: {device.type.upper()}")

    # Setup directories and label
    from datetime import datetime

    label = args.run_label if args.run_label else args.model
    if not isinstance(label, str):
        label = label.__class__.__name__

    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")

    if args.name:
        run_dir = f"{timestamp}_{args.name}"
    else:
        run_dir = f"{label}_{timestamp}"

    tb_dir = f"runs/{run_dir}"
    checkpoint_dir = os.path.join(tb_dir, "checkpoints")

    # Copy configuration file to log folder if provided
    if args.config:
        import shutil

        os.makedirs(tb_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(tb_dir, "config.py"))
        print(f"Copied config file to log folder: {tb_dir}/config.py")

    # Save the command line that can be used to reproduce the experiment
    import sys

    reproduce_args = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--config" and i + 1 < len(sys.argv):
            reproduce_args.append(arg)
            copied_config_path = os.path.abspath(os.path.join(tb_dir, "config.py"))
            reproduce_args.append(copied_config_path)
            i += 2
        elif arg.startswith("--config="):
            copied_config_path = os.path.abspath(os.path.join(tb_dir, "config.py"))
            reproduce_args.append(f"--config={copied_config_path}")
            i += 1
        else:
            if i == 0:
                reproduce_args.append(f"python {os.path.basename(arg)}")
            else:
                reproduce_args.append(arg)
            i += 1

    reproduce_cmd = " ".join(reproduce_args)
    reproduce_file_path = os.path.join(tb_dir, "reproduce.sh")
    try:
        os.makedirs(tb_dir, exist_ok=True)
        with open(reproduce_file_path, "w") as f_reprod:
            f_reprod.write("#!/bin/bash\n")
            f_reprod.write(reproduce_cmd + "\n")
        os.chmod(reproduce_file_path, 0o755)
        print(f"Saved reproduction command to: {reproduce_file_path}")
    except Exception as e:
        print(f"Failed to save reproduction command: {e}")

    # Initialize metric helper
    metrics = SpeechMetrics(device=device)

    # Initialize the TB writer.
    writer = SummaryWriter(tb_dir)

    # Dataloader setups
    mock_size = 64 if args.dry_run else None
    # For CUDA and MPS devices, load the whole dataset onto the GPU device
    ds_device = device if device.type in ("cuda", "mps") else None
    train_dataset = VoiceBankDemandDataset(split="train", mock_size=mock_size, device=ds_device)
    test_dataset = VoiceBankDemandDataset(split="test", mock_size=mock_size, device=ds_device)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Select 4 evaluation audio samples uniformly to log to TensorBoard
    num_eval_samples = len(test_dataset)
    eval_audio_indices = np.linspace(0, num_eval_samples - 1, num=4, dtype=int).tolist()
    print(
        f"Uniformly selected eval audio indices for TensorBoard logging: {eval_audio_indices}"
    )

    # Initialize model
    if isinstance(args.model, torch.nn.Module):
        model = args.model.to(device)
        print(f"Using custom model object from config: {model.__class__.__name__}")
    else:
        model = get_model(args.model).to(device)

    # Try compiling model if requested
    if args.compile:
        try:
            model_name = (
                args.model
                if isinstance(args.model, str)
                else args.model.__class__.__name__
            )
            print(f"Compiling model {model_name} using torch.compile()...")
            model = torch.compile(model, mode="max-autotune")
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Failed to compile model (falling back to eager mode): {e}")

    # Try enabling fused optimizer (saves kernel launch overhead)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.wd, fused=True
        )
        print("Using fused AdamW optimizer.")
    except Exception:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.wd, fused=False
        )
        print("Using standard AdamW optimizer.")

    # Initialize Cosine Annealing learning rate scheduler with linear warmup (step-wise)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = getattr(args, "warmup_steps", None)
    if warmup_steps is None:
        warmup_steps = 500
    # A warmup longer than the run itself would mean the LR never reaches its peak
    warmup_steps = min(int(warmup_steps), max(1, total_steps // 2))
    print(f"LR schedule: {warmup_steps} warmup steps out of {total_steps} total steps")

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay from 1.0 to 0.0 over the remaining steps
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Running initial validation pass (Epoch 0)...")
    val_results = run_validation(
        model=model,
        test_loader=test_loader,
        device=device,
        metrics=metrics,
        eval_audio_indices=eval_audio_indices,
        writer=writer,
        global_step=0,
        eval_samples_limit=20,
    )
    save_metrics_to_jsonl(val_results, epoch=0, log_dir=tb_dir)
    print(
        f"Epoch 00/{args.epochs:02d} (Initial) | "
        f"Val SI-SDR: {val_results['val_si_sdr']:.2f}dB | "
        f"PESQ: {val_results['pesq']:.2f} | STOI: {val_results['estoi']:.2f} | DNSMOS: {val_results['dnsmos']:.2f}\n"
    )

    print("\n=== Start Training ===")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs")
    global_step = 0
    for epoch in epoch_bar:
        model.train()
        epoch_start = time.time()

        train_loss = 0.0
        train_si_sdr = 0.0
        train_metrics = {}

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs} (Train)", leave=False
        )
        for noisy, clean in train_bar:
            noisy, clean = noisy.to(device), clean.to(device)

            if args.remix:
                # noise = noisy - clean, mix with shuffled noise
                noise = noisy - clean
                perm = torch.randperm(noisy.size(0), device=device)
                noisy = clean + noise[perm]

            # Autocast context for mixed precision
            with AutocastManager(device):
                enhanced, loss, metrics_dict = model.compute_loss(noisy, clean)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            # Compute real-time online training SI-SDR (for tracking)
            with torch.no_grad():
                current_sdr = metrics.compute_si_sdr(clean, enhanced).mean().item()

            train_loss += loss.item()
            train_si_sdr += current_sdr
            for key, val in metrics_dict.items():
                train_metrics[key] = train_metrics.get(key, 0.0) + val
            global_step += 1

            # Log stepwise every 20 steps
            if global_step % 20 == 0:
                writer.add_scalar("train/loss/step", loss.detach().item(), global_step)
                writer.add_scalar("train/si-sdr/step", current_sdr, global_step)
                writer.add_scalar(
                    "train/lr/step", optimizer.param_groups[0]["lr"], global_step
                )
                for key, val in metrics_dict.items():
                    writer.add_scalar(f"train/{key}/step", val, global_step)

            train_bar.set_postfix(loss=f"{loss.item():.4f}", sdr=f"{current_sdr:.1f}dB")

        # Average training epoch values
        num_batches = len(train_loader)
        train_loss /= num_batches
        train_si_sdr /= num_batches
        for key in train_metrics:
            train_metrics[key] /= num_batches

        # Log epoch values
        writer.add_scalar("train/loss/epoch", train_loss, global_step)
        writer.add_scalar("train/si-sdr/epoch", train_si_sdr, global_step)
        for key, val in train_metrics.items():
            writer.add_scalar(f"train/{key}/epoch", val, global_step)

        # Validation / Evaluation Phase
        val_results = run_validation(
            model=model,
            test_loader=test_loader,
            device=device,
            metrics=metrics,
            eval_audio_indices=eval_audio_indices,
            writer=writer,
            global_step=global_step,
            eval_samples_limit=20,
        )
        save_metrics_to_jsonl(val_results, epoch=epoch, log_dir=tb_dir)

        epoch_time = time.time() - epoch_start
        losses_str = ", ".join(
            [f"{k.capitalize()}: {v:.4f}" for k, v in train_metrics.items()]
        )
        tqdm.write(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {train_loss:.4f} ({losses_str}) | "
            f"Train SI-SDR: {train_si_sdr:.2f}dB | "
            f"Val SI-SDR: {val_results['val_si_sdr']:.2f}dB | "
            f"PESQ: {val_results['pesq']:.2f} | STOI: {val_results['estoi']:.2f} | DNSMOS: {val_results['dnsmos']:.2f} | "
            f"Time: {epoch_time:.1f}s"
        )
        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            val_sdr=f"{val_results['val_si_sdr']:.1f}dB",
            pesq=f"{val_results['pesq']:.2f}",
        )

        # Save model checkpoint using args.model name
        torch.save(model.state_dict(), f"{checkpoint_dir}/epoch_{epoch:03d}.pt")

    print(f"\nTraining complete. Model checkpoints saved in '{checkpoint_dir}'.")


if __name__ == "__main__":
    main()
