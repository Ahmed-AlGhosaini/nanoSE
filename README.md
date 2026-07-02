# NanoSE Speech Enhancement

Lightweight speech enhancement model training platform.

## Installation

Ensure you have Python 3.8+ installed.
Here are two recommended distributions.

* [anaconda](https://www.anaconda.com/download)
* [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

Open a terminal and install the required environment.

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Training

Run training with customizable options:

```bash
# Train CRN model (default)
python train.py --model crn

# Train FastEnhancer model with dynamic noise remixing and custom run label
python train.py --model fastenhancer --remix --run-label fastenhancer_experiment
```

### Options:
* `--model`: `crn`, `crntiny`, `fastenhancer`, or `fastenhancercompact`.
* `--seed`: Reproducibility seed (default: `42`).
* `--remix`: Enable dynamic random noise remixing per batch.
* `--run-label`: Custom string prefix for directory naming.

### Monitoring:
Start TensorBoard to view logs, metrics, and audio/spectrogram outputs:
```bash
tensorboard --logdir runs
```

## Benchmarking

Benchmark the parameter count, average step time, and throughput (steps/second) of different models:

```bash
# Run benchmark on default device (e.g. MPS on macOS, CUDA on Linux, or CPU)
python benchmark.py --steps 50 --batch-size 8
```

## Evaluation

Calculate objective audio metrics (SI-SDR, PESQ, STOI, DNSMOS) on the test split:

```bash
python enhance_eval.py --model fastenhancer --checkpoint checkpoints/fastenhancer_experiment_<timestamp>/epoch_025.pt
```

## Web Page Visualization

Create a dark-themed HTML player page to listen to comparative audio samples and view spectrograms:

```bash
python enhance_eval.py --model fastenhancer --checkpoint checkpoints/fastenhancer_experiment_<timestamp>/epoch_025.pt -n 10 -o eval_results
```
Open `eval_results/index.html` in any browser to inspect the results.

## References

* **FastEnhancer**: Ahn, S. H. (2025). *FastEnhancer: Speed-Optimized Streaming Neural Speech Enhancement*. Accepted to ICASSP 2026. [arXiv:2509.21867](https://arxiv.org/abs/2509.21867).
* **CRN**: Zhao, H., et al. (2018). *A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement*. Interspeech 2018. [DOI:10.21437/Interspeech.2018-1322](https://doi.org/10.21437/Interspeech.2018-1322).


