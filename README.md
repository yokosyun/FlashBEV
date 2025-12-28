# FlashBEV

## UV
```
uv venv --python 3.8
source .venv/bin/activate
```

## Install
```
uv pip install -r requirements.txt
cd flashbev && uv pip install . -e
```

# Benchmark

## Using default config
```bash
uv run python tools/benchmark.py
```

## Override config from command line
```bash
# With calibration file and height bins
uv run python tools/benchmark.py load_calib=inputs/calib_nuscenes.json num_height_bins=[8,10,16,20,40]
