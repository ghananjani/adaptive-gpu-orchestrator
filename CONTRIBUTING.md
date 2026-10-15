# Contributing

Thanks for helping build `adaptive_gpu_orchestrator`.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

The core package intentionally has no mandatory third-party dependencies in v0.1 so that parsing and diagnosis logic can be tested on non-GPU CI runners.

## Pull requests

Please include:

- a clear problem statement
- tests for parser/diagnosis changes
- sample input when adding support for new NVIDIA/cluster output
- benchmark methodology for performance claims
- no proprietary cluster data or credentials

## Research changes

For experimental algorithms, add an experiment note under `docs/` or `research/` describing hypothesis, baseline, hardware/software environment and evaluation metrics.
