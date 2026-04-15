# Refactored Standalone Project

This folder is a standalone refactor of the original project.

## Structure

- `src/`: all Python source code (datasets/gui/models/services/trainers/utils/workers)
- `configs/`: YAML configs
- `data/`: datasets
- `results/`: experiment outputs
- `run_gui.py`, `train.py`, `run_all.py`, `main.py`: entry scripts

## Conda Usage

```bash
cd refactored_project
conda activate <your_env>
python run_gui.py
```

Train:

```bash
python train.py --model_cfg configs/model/gcn_gru.yaml
```

Batch run:

```bash
python run_all.py
```

## Windows Quick Start

- Double click `rungui.bat`
- If your conda env is not auto-detected, set `CONDA_ENV_NAME` in `rungui.bat`

