#!/usr/bin/env python3
# Copyright (c) 2026 Robin Scheibler <fakufaku@gmail.com>
# License: MIT (see LICENSE file at the root of the repository)

import argparse
import os
import json
import torch
import numpy as np

# Import all models to resolve them when loading config.py
from models.crn import CRN, CRNTiny
from models.fastenhancer import FastEnhancerCompact
from models.glumaskd import GLUMaskd
from models.comfi_fastgrnn import ComfiFastGRNNModel

def load_config(log_dir):
    config_path = os.path.join(log_dir, "config.py")
    if not os.path.exists(config_path):
        return {}
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
        with open(config_path, "r") as f:
            exec(f.read(), local_ns, config_vars)
    except Exception as e:
        print(f"Warning: Failed to execute config {config_path}: {e}")
    return config_vars

def load_metrics(log_dir):
    metrics_path = os.path.join(log_dir, "validation_metrics.jsonl")
    if not os.path.exists(metrics_path):
        return []
    records = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records

def get_param_count(log_dir):
    # Locate checkpoint directory (new layout: runs/<run>/checkpoints)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        # Fallback to old layout: checkpoints/<run>
        run_name = os.path.basename(os.path.normpath(log_dir))
        checkpoint_dir = os.path.join("checkpoints", run_name)
    if os.path.exists(checkpoint_dir):
        pt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if pt_files:
            # Load first checkpoint found to estimate parameters
            first_pt = os.path.join(checkpoint_dir, pt_files[0])
            try:
                state_dict = torch.load(first_pt, map_location="cpu")
                return sum(p.numel() for p in state_dict.values())
            except Exception:
                pass
    return "N/A"

def format_params(count):
    if isinstance(count, (int, float)):
        if count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}K"
        else:
            return str(count)
    return str(count)

def main():
    parser = argparse.ArgumentParser(description="Create markdown comparison report from training logs.")
    parser.add_argument("log_dirs", nargs="+", help="Paths to log directories (e.g. runs/XYZ)")
    parser.add_argument("--best-epoch", action="store_true", help="Use best epoch metrics instead of last epoch")
    parser.add_argument("--output", type=str, default="report/report.md", help="Path to write the markdown report")
    args = parser.parse_args()

    # Read data from each log dir
    runs_data = []
    for log_dir in args.log_dirs:
        if not os.path.isdir(log_dir):
            print(f"Skipping {log_dir} as it is not a valid directory.")
            continue
            
        config = load_config(log_dir)
        metrics = load_metrics(log_dir)
        param_count = get_param_count(log_dir)
        
        runs_data.append({
            "log_dir": log_dir,
            "config": config,
            "metrics": metrics,
            "param_count": param_count
        })

    if not runs_data:
        print("No valid log directories found.")
        return

    # Generate Markdown Tables
    md_content = []
    md_content.append("# Model Training & Performance Report\n")
    md_content.append(f"This report compares {len(runs_data)} training runs. Metric selection mode: **{'Best Epoch (by Max SI-SDR)' if args.best_epoch else 'Last Epoch'}**.\n")

    # Table 1: Performance comparison
    md_content.append("## 1. Performance Comparison\n")
    md_content.append("| Log Folder | Model | Selected Epoch | Val SI-SDR (dB) | PESQ | STOI | DNSMOS |")
    md_content.append("|---|---|---|---|---|---|---|")
    
    for run in runs_data:
        metrics = run["metrics"]
        log_dir_name = os.path.basename(os.path.normpath(run["log_dir"]))
        
        # Get model name from config
        model_obj = run["config"].get("model")
        if model_obj:
            model_name = model_obj.__class__.__name__
        else:
            model_name = run["config"].get("model_name", "N/A")
            
        if not metrics:
            md_content.append(f"| {log_dir_name} | {model_name} | N/A | N/A | N/A | N/A | N/A |")
            continue
            
        if args.best_epoch:
            # Best epoch based on highest val_si_sdr
            # Note: filter out values that don't have val_si_sdr
            valid_metrics = [m for m in metrics if "val_si_sdr" in m]
            if valid_metrics:
                selected_metric = max(valid_metrics, key=lambda m: m.get("val_si_sdr", -999))
            else:
                selected_metric = metrics[-1]
        else:
            # Last epoch
            selected_metric = metrics[-1]
            
        epoch = selected_metric.get("epoch", "N/A")
        val_si_sdr = selected_metric.get("val_si_sdr", "N/A")
        pesq = selected_metric.get("pesq", "N/A")
        estoi = selected_metric.get("estoi", "N/A")
        dnsmos = selected_metric.get("dnsmos", "N/A")
        
        # format numbers
        val_si_sdr_str = f"{val_si_sdr:.2f}" if isinstance(val_si_sdr, (int, float)) else str(val_si_sdr)
        pesq_str = f"{pesq:.2f}" if isinstance(pesq, (int, float)) else str(pesq)
        estoi_str = f"{estoi:.2f}" if isinstance(estoi, (int, float)) else str(estoi)
        dnsmos_str = f"{dnsmos:.2f}" if isinstance(dnsmos, (int, float)) else str(dnsmos)
        
        md_content.append(f"| {log_dir_name} | {model_name} | {epoch} | {val_si_sdr_str} | {pesq_str} | {estoi_str} | {dnsmos_str} |")

    md_content.append("\n")

    # Table 2: Model Configuration & Parameters
    md_content.append("## 2. Model Configuration & Parameters\n")
    md_content.append("| Log Folder | Model | Parameters | Learning Rate | Weight Decay | Batch Size | Seed | Compiled | Remix |")
    md_content.append("|---|---|---|---|---|---|---|---|---|")
    
    for run in runs_data:
        log_dir_name = os.path.basename(os.path.normpath(run["log_dir"]))
        config = run["config"]
        param_count = run["param_count"]
        
        model_obj = config.get("model")
        if model_obj:
            model_name = model_obj.__class__.__name__
        else:
            model_name = config.get("model_name", "N/A")
            
        lr = config.get("lr", "N/A")
        wd = config.get("wd", "N/A")
        batch_size = config.get("batch_size", "N/A")
        seed = config.get("seed", "N/A")
        compiled = config.get("compile", "N/A")
        remix = config.get("remix", "N/A")
        
        param_str = format_params(param_count)
        
        md_content.append(f"| {log_dir_name} | {model_name} | {param_str} | {lr} | {wd} | {batch_size} | {seed} | {compiled} | {remix} |")

    # Write report
    report_text = "\n".join(md_content) + "\n"
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"Report successfully written to {args.output}")
    print("\n--- REPORT PREVIEW ---")
    print(report_text)

if __name__ == "__main__":
    main()
