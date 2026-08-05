# Homework notebooks

Solutions to the nanoSE homework assignment (*Neural Audio and Speech Processing —
Day 2*, R. Scheibler). Run them **in order**: `01` produces the baseline run that
every later notebook compares against.

| Notebook | Assignment task | Runtime on a T4 |
|---|---|---|
| [`00_setup_and_data.ipynb`](00_setup_and_data.ipynb) | environment, Voicebank-DEMAND, the `CRNTiny` model | ~10 min (no training) |
| [`01_baseline.ipynb`](01_baseline.ipynb) | the reference baseline run | ~1 h |
| [`02_activation_study.ipynb`](02_activation_study.ipynb) | **Task 1** — activation in `EncoderBlock` / `DecoderBlock` | ~2 h |
| [`03_lr_tuning.ipynb`](03_lr_tuning.ipynb) | **Task 2** — tune the step size `lr` | ~1.7 h |
| [`04_speed_and_single_epoch.ipynb`](04_speed_and_single_epoch.ipynb) | **Tasks 3 & 4** — SI-SDR per minute, best single epoch | ~1.5 h |
| [`05_final_experiment_and_report.ipynb`](05_final_experiment_and_report.ipynb) | **Task 5** — own idea + `report/nanose.md` | ~2 h |

Every notebook is standalone: it starts with a Colab bootstrap cell, then
`nb_utils.bootstrap()` which changes the working directory to the repository root.

## Running on Colab (recommended)

Training is only practical on a GPU — on 4 CPU threads a single 25-epoch run takes
over 15 hours. Upload a notebook to Colab, pick *Runtime > Change runtime type >
T4 GPU*, and run the first cell. It clones `REPO_URL` (set at the top of the cell)
and installs `requirements.txt`.

`REPO_URL` **must point at this fork**, not at `fakufaku/nanoSE`: the notebooks rely
on the homework modifications listed below.

The first data cell downloads Voicebank-DEMAND and builds `train_cache.bin`
(~2.2 GB) — a few minutes, once per session. To avoid paying that on every
reconnect, mount Drive and keep the caches there.

## Running locally

```bash
conda activate nanose
jupyter lab notebooks/
```

## How state is shared between notebooks

Finished runs are recorded in `notebooks/experiment_runs.json` by
`nb_utils.remember()`. Sweep loops call `nb_utils.recall()` first and skip anything
already trained, so a notebook re-run after a Colab disconnect resumes instead of
starting over — and notebooks `04`/`05` pick up the best activation and learning
rate found in `02`/`03` automatically.

Delete an entry from that file to force a re-run.

## Modifications to the framework

The notebooks drive the stock `train.py` / `create_report.py` workflow. Three
backward-compatible changes were needed to make the tasks configurable; all
defaults reproduce the original baseline exactly.

* **`models/crn.py`** — `EncoderBlock`, `DecoderBlock`, `CRN` and `CRNTiny` take an
  `activation` argument resolved by `get_activation()` (default `"leaky_relu"` =
  `nn.LeakyReLU(0.2)`, as before). Needed for Task 1.
* **`models/crn.py`** — `CRNTiny`/`CRN` take `compression`, `spec_weight` and
  `sdr_weight`; `compressed_mse()` applies power-law compression to the magnitude
  MSE. Defaults `1.0 / 1.0 / 0.01` are the original loss. Used by Task 5.
* **`train.py`** — the LR warmup length is read from the config (`warmup_steps`) or
  `--warmup-steps`, defaulting to 500 as before, and is capped at half the run so a
  short run cannot end while still warming up. Needed for Tasks 3 and 4.

`notebooks/nb_utils.py` is glue only: it writes config files, launches
`python train.py` as a subprocess while streaming its output into the notebook, and
reads the logs back for the plots. It additionally saves `train_log.txt`,
`progress.json` (per-epoch metrics **and** wall-clock time) and `timing.json` into
each run directory — `progress.json` is what makes the time-to-quality analysis in
notebook `04` possible.
