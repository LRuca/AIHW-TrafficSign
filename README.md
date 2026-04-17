# Traffic Sign Experiment Pipeline

This project is a PyTorch-based experiment pipeline for traffic sign classification on GTSRB-style data. It includes:

- config-driven experiment definitions
- automated training and evaluation
- validation and test metrics export
- a local dashboard for monitoring, comparison, archiving, and scheduling

## Project Structure

```text
configs/     Experiment configs and experiment plan
scripts/     Launch, training, batch, and dashboard scripts
src/         Training pipeline source code
web/         Dashboard frontend
data/        Dataset root (not tracked by git)
outputs/     Logs, checkpoints, plots, summaries (not tracked by git)
```

## Environment

The project is designed to run in the local conda environment:

```text
pytorch
```

Python is launched through:

```powershell
scripts\run_in_pytorch.ps1
```

## Dataset Layout

Place the dataset under:

```text
data/train_images/
data/val_images/
data/test_images/
```

Each split should contain 43 class folders.

Examples:

```text
data/train_images/00
data/train_images/01
...
data/test_images/42
```

## Start Dashboard

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_in_pytorch.ps1 scripts\launch_dashboard.py
```

Then open:

```text
http://127.0.0.1:8765/
```

## Run One Experiment

Example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_in_pytorch.ps1 scripts\run_experiment.py --config configs\experiments\backbone_mobilenetv2.yaml
```

## Run Experiments From Dashboard

In the dashboard:

1. Open `实验排程`
2. Click experiments from the catalog
3. Reorder or remove queue items
4. Click `启动队列`

The queue will run experiments sequentially.

## Important Defaults

- default training epochs: `30`
- model selection uses validation set
- final metrics are also evaluated on the test set

## Git Notes

The repository intentionally ignores:

- raw dataset files in `data/`
- all generated outputs in `outputs/`
- model checkpoints such as `.pt` and `.pth`

This keeps the repo lightweight and suitable for sharing with coworkers.
