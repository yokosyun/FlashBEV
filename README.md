# FlashBEV

## UV
install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
```
create environment
```
uv venv --python 3.8
source .venv/bin/activate
```

## Install

### Using pyproject.toml (recommended)
```bash
# For just using the library (core dependencies only)
uv sync

# For running benchmarks (includes matplotlib, tabulate, hydra-core)
uv sync --extra benchmark

# Install the package itself (required after uv sync)
cd flashbev && uv pip install -e .
```

**Note:** `uv sync` only installs packages listed in `pyproject.toml`. The `flashbev` package must be installed separately as it contains CUDA extensions built via `setup.py`.

# Benchmark

## Using default config
```bash
uv run python tools/benchmark.py
```

## Override config from command line
```bash
# With calibration file and height bins
uv run python tools/benchmark.py load_calib=inputs/calib_nuscenes.json num_height_bins=[8,10,16,20,40]
