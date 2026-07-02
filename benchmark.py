# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import argparse
import time
import torch
from model import get_model

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(model_name, batch_size=8, seq_len=64000, num_steps=50, warmup_steps=10, device="cpu"):
    device_obj = torch.device(device)
    try:
        model = get_model(model_name).to(device_obj)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

    # Use dummy inputs of shape [B, T] (representing 4 seconds of 16kHz audio)
    noisy = torch.randn(batch_size, seq_len, device=device_obj)
    clean = torch.randn(batch_size, seq_len, device=device_obj)

    # Warmup
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        _, loss, _ = model.compute_loss(noisy, clean)
        loss.backward()
        optimizer.step()
        
    if device_obj.type == "mps":
        torch.mps.synchronize()
    elif device_obj.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark loop
    start_time = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        _, loss, _ = model.compute_loss(noisy, clean)
        loss.backward()
        optimizer.step()
        
    if device_obj.type == "mps":
        torch.mps.synchronize()
    elif device_obj.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time
    avg_step_ms = (elapsed / num_steps) * 1000
    steps_per_sec = num_steps / elapsed
    num_params = get_num_params(model)

    return {
        "model_name": model_name,
        "num_params": num_params,
        "avg_step_ms": avg_step_ms,
        "steps_per_sec": steps_per_sec
    }

def main():
    parser = argparse.ArgumentParser(description="Nanose Model Training Benchmark")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for benchmarking")
    parser.add_argument("--duration", type=float, default=4.0, help="Audio duration in seconds (at 16kHz)")
    parser.add_argument("--steps", type=int, default=50, help="Number of benchmark steps")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (mps, cuda, cpu)")
    args = parser.parse_args()

    if args.device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    print(f"==================================================")
    print(f" Benchmarking Speech Enhancement Models")
    print(f" Device:      {device.upper()}")
    print(f" Batch Size:  {args.batch_size}")
    print(f" Audio Len:   {args.duration}s ({int(args.duration * 16000)} samples)")
    print(f" Warmup/Steps: {args.warmup} / {args.steps}")
    print(f"==================================================\n")

    models_to_test = ["crntiny", "crn", "fastenhancer", "glumaskd", "comfi_fastgrnn"]
    results = []

    seq_len = int(args.duration * 16000)

    for m in models_to_test:
        print(f"Benchmarking {m}...")
        res = benchmark_model(
            model_name=m,
            batch_size=args.batch_size,
            seq_len=seq_len,
            num_steps=args.steps,
            warmup_steps=args.warmup,
            device=device
        )
        if res:
            results.append(res)

    print("\nBenchmark Results:")
    print("| Model | Parameters | Avg Step Time (ms) | Throughput (steps/sec) |")
    print("|---|---|---|---|")
    for r in results:
        params_str = f"{r['num_params']:,}"
        print(f"| {r['model_name']} | {params_str} | {r['avg_step_ms']:.2f} ms | {r['steps_per_sec']:.2f} step/s |")
    print()

if __name__ == "__main__":
    main()
