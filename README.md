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
cd flashbev && uv pip install -e . && cd ..
```

**Note:** `uv sync` only installs packages listed in `pyproject.toml`. The `flashbev` package must be installed separately as it contains CUDA extensions built via `setup.py`.

# Benchmark

## command line
```bash
uv run python tools/profile_differences.py load_calib=inputs/calib_nuscenes.json

uv run python tools/profile_latency.py output_json=outputs/A6000.json load_calib=inputs/calib_nuscenes.json num_height_bins=[4,8,12,16]

uv run python tools/calculate_flops.py

uv run python tools/calculate_memory.py

uv run python tools/plot_latency.py
```


# Debuging
```
uv run python -m debugpy --listen 0.0.0.0:7777 --wait-for-client ${ARGS}
```