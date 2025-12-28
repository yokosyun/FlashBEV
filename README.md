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
```
python3 tools/benchmark.py --num-height-bins 8,10,16,20,40
```