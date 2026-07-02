# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import argparse
import os
import warnings

# Suppress noisy PyTorch MPS-specific resizing warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*resized since it had shape.*"
)

import torch
import torchaudio

from model import enhance_waveform, get_model


def main():
    parser = argparse.ArgumentParser(
        description="Enhance a noisy audio file using a trained checkpoint."
    )
    parser.add_argument("noisy_wav", type=str, help="Path to input noisy WAV file")
    parser.add_argument("output_wav", type=str, help="Path to output enhanced WAV file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="crn",
        choices=["crn", "crntiny", "fastenhancer", "fastenhancercompact", "glumaskd", "glumasked", "comfi_fastgrnn"],
        help="Model architecture of the checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to python configuration file defining the model."
    )
    args = parser.parse_args()

    if not os.path.exists(args.noisy_wav):
        print(f"Error: Noisy file '{args.noisy_wav}' not found.")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        return

    # Choose device (CUDA GPU preferred, then Metal GPU, then CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device.type.upper()}")

    # 1. Load trained model
    model_obj = None
    if args.config:
        print(f"Loading model configuration from {args.config}...")
        if not os.path.exists(args.config):
            print(f"Error: Config file '{args.config}' not found.")
            return

        from models.crn import CRN, CRNTiny
        from models.fastenhancer import FastEnhancerCompact
        from models.glumaskd import GLUMaskd
        from models.comfi_fastgrnn import ComfiFastGRNNModel

        config_vars = {}
        local_ns = {
            "CRN": CRN,
            "CRNTiny": CRNTiny,
            "FastEnhancerCompact": FastEnhancerCompact,
            "GLUMaskd": GLUMaskd,
            "ComfiFastGRNNModel": ComfiFastGRNNModel,
            "torch": torch
        }
        try:
            with open(args.config, "r") as f:
                exec(f.read(), local_ns, config_vars)
            model_obj = config_vars.get("model")
        except Exception as e:
            print(f"Error executing configuration file {args.config}: {e}")
            return

    if model_obj is not None:
        if isinstance(model_obj, torch.nn.Module):
            model = model_obj.to(device)
            model_name = model.__class__.__name__
        else:
            model = get_model(model_obj).to(device)
            model_name = str(model_obj)
        print(f"Loaded custom model object from config: {model_name}")
    else:
        print(f"Loading {args.model} model checkpoint...")
        model = get_model(args.model).to(device)

    # Load state dict (handle potential torch.compile wrapper)
    state_dict = torch.load(args.checkpoint, map_location=device)
    # Strip '_orig_mod.' prefix if checkpoint was saved from a compiled model
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    model.eval()

    # 2. Load audio
    print(f"Loading noisy audio from '{args.noisy_wav}'...")
    waveform, sr = torchaudio.load(args.noisy_wav)

    # Standardize channels: average multi-channel inputs to mono
    if waveform.size(0) > 1:
        print("Input has multiple channels; downmixing to mono.")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if necessary
    if sr != 16000:
        print(f"Resampling from {sr}Hz to 16000Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    # 3. Enhance audio
    print("Enhancing audio waveform...")
    # Add batch dimension: [B=1, num_samples]
    waveform = waveform.to(device)
    with torch.no_grad():
        # enhance_waveform handles STFT, cropping, masking, polar reconstruction, and iSTFT
        enhanced_wav = enhance_waveform(model, waveform)

    # Squeeze out batch dimension and move back to CPU
    enhanced_wav = enhanced_wav.cpu()

    # 4. Save enhanced audio
    os.makedirs(os.path.dirname(os.path.abspath(args.output_wav)), exist_ok=True)
    print(f"Saving enhanced audio to '{args.output_wav}'...")
    torchaudio.save(args.output_wav, enhanced_wav, sr)
    print("Enhancement complete!")


if __name__ == "__main__":
    main()
