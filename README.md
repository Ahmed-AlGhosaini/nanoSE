# NanoSE Speech Enhancement

Lightweight speech enhancement model training platform.

## Installation

Ensure you have Python 3.8+ installed.
Here are two recommended distributions.

* [anaconda](https://www.anaconda.com/download)
* [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

Open a terminal and install the required environment.

Verify that python and git work.
```bash
python --help
git --help
```

If you have git proceed. If you do not have git, you can get the [zip file of the code](https://github.com/fakufaku/nanoSE/archive/refs/heads/main.zip).

```bash
# Clone the repository
git clone https://github.com/fakufaku/nanoSE.git

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Training & Configuration

### 1. Training with Config Files
The platform supports Python-based configuration files where hyperparameters and neural networks are instantiated directly in Python:

```bash
# Train using the default config file (config/default_config.py is used if --config is omitted)
python train.py --config config/default_config.py

# Override hyperparameters from command-line (CLI overrides config)
python train.py --config config/default_config.py --epochs 10 --lr 5e-4
```

Example configuration file (`config/default_config.py`):
```python
from models.crn import CRNTiny

name = "crntiny_default"
epochs = 25
batch_size = 32
lr = 1e-3
wd = 0.02
compile = False
seed = 42
remix = False
model = CRNTiny()
```

### 2. Argument Precedence
Configurations follow a strict resolution chain:
1. **Default Config**: Initialized from `config/default_config.py`.
2. **Custom Config**: Loaded if `--config` is supplied.
3. **CLI Arguments**: Any arguments explicitly passed (like `--epochs 10`) override values defined in the configuration files.

### 3. Dry-Run Mode
Use `--dry-run` to run a quick validation pass of 1 epoch over a small fraction of the dataset (64 samples).
```bash
python train.py --dry-run
```

---

## Logging & Reproducibility

Every training run automatically generates output logs inside `runs/{timestamp}_{name}/` (or `runs/{label}_{timestamp}` if no name is defined) containing:
1. **Model Checkpoints**: Saved per-epoch state dicts (`.pt` files) in `checkpoints/{run_dir}/`.
2. **Archived Config**: A copy of the active configuration file saved as `config.py` in the log folder for accountability.
3. **Reproduction Script**: A `reproduce.sh` executable script containing the exact command line arguments used to trigger the run, dynamically updated to reference the archived `config.py`.
4. **Validation Metrics Log**: Validation metrics computed at Epoch 0 (pre-training baseline) and after each epoch are appended to `validation_metrics.jsonl`.
5. **TensorBoard Events**: Loss, learning rate, audio samples, and spectrogram plots for tracking.

Start TensorBoard to view progress:
```bash
tensorboard --logdir runs
```

---

## Comparing Experiments (Reports)

You can generate a comparative markdown report summarizing the validation performance and hyperparameter configurations of multiple runs:

```bash
# Compare runs using the last epoch metrics
python create_report.py runs/run_dir_1 runs/run_dir_2

# Compare runs using their best epoch (based on maximum Val SI-SDR)
python create_report.py --best-epoch runs/run_dir_1 runs/run_dir_2
```

The report is written to `report/report.md` by default (use `--output` to override) and contains two main tables:
1. **Performance Comparison**: Validation SI-SDR, PESQ, STOI, and DNSMOS metrics.
2. **Model Parameters & Config**: Model parameter count (extracted from checkpoints), LR, weight decay, batch size, compile, and remix flags.

---

## Benchmarking

Benchmark parameter count, step time, and throughput (steps/sec) of different models:

```bash
python benchmark.py --steps 50 --batch-size 8
```

---

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

---

## References

* **FastEnhancer**: Ahn, S. H. (2025). *FastEnhancer: Speed-Optimized Streaming Neural Speech Enhancement*. Accepted to ICASSP 2026. [arXiv:2509.21867](https://arxiv.org/abs/2509.21867).
* **CRN**: Zhao, H., et al. (2018). *A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement*. Interspeech 2018. [DOI:10.21437/Interspeech.2018-1322](https://doi.org/10.21437/Interspeech.2018-1322).
