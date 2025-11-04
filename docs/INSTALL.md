# Dependency Installation Guide

## Quick Start

### One-Command Install (Recommended)

```bash
# Install everything (runtime + dev + test + extras)
./install_deps.sh

# Or for specific use cases:
./install_deps.sh --dev      # Development environment
./install_deps.sh --test     # Testing only
./install_deps.sh --minimal  # Runtime only
```

### Manual Install

If you prefer to install dependencies manually:

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install core runtime dependencies
pip install -r requirements.txt

# 3. Install development tools (optional)
pip install -r requirements-dev.txt

# 4. Install test framework (optional)
pip install -r requirements-test.txt

# 5. Install extra packages (optional)
pip install -r requirements-extra.txt
```

## Requirements Files Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `requirements.txt` | Core runtime dependencies (torch, librosa, music21) | Production/deployment |
| `requirements-dev.txt` | Development tools (pytest, jsonschema, tensorboard) | Local development |
| `requirements-test.txt` | Testing framework (pytest, httpx, fastapi) | CI/testing |
| `requirements-extra.txt` | Optional ML packages (optuna, streamlit) | Advanced features |
| `requirements-optional.txt` | Optional audio tools | Audio processing |
| `requirements-lamda.txt` | LAMDa analyzer dependencies | Stage2 evaluation |

## Virtual Environment Setup

### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv .venv311

# Activate (macOS/Linux)
source .venv311/bin/activate

# Activate (Windows)
.venv311\Scripts\activate

# Install dependencies
./install_deps.sh
```

### Using conda

```bash
# Create conda environment
conda create -n composer2-3 python=3.11

# Activate
conda activate composer2-3

# Install dependencies
./install_deps.sh
```

## Verification

After installation, verify critical packages:

```bash
# Core packages
python -c "import torch; import numpy; import librosa; import music21; print('✅ Core OK')"

# Test/CI packages
python -c "import pytest; import jsonschema; print('✅ Test OK')"

# Run test suite
pytest tests/test_music_guards_time_sigs.py -v
```

## Troubleshooting

### Issue: `torch==2.3.0+cpu` not found

**Solution**: Use the CPU-specific wheel URL:

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

### Issue: `jsonschema` missing in CI

**Solution**: Install dev requirements:

```bash
pip install jsonschema
# Or
pip install -r requirements-dev.txt
```

### Issue: Permission denied on install script

**Solution**: Make the script executable:

```bash
chmod +x install_deps.sh
```

## CI/CD Integration

For GitHub Actions, use the workflow file:

```yaml
- name: Install dependencies
  run: |
    pip install --upgrade pip
    pip install -r requirements-test.txt
    pip install jsonschema
```

## Platform-Specific Notes

### macOS
- Some packages may require Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

### Linux
- libsndfile may be required for soundfile:
  ```bash
  sudo apt-get install libsndfile1
  ```

### Windows
- Use PowerShell or Git Bash for script execution
- Some audio libraries may require Visual C++ redistributables

## Minimal Docker Setup

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-test.txt

COPY . .
CMD ["pytest", "tests/"]
```

## Next Steps

After successful installation:

1. **Run tests**: `pytest tests/ -v`
2. **Run evaluation**: `python scripts/quick_eval_stage2.py --help`
3. **Run CI gate**: `python scripts/ci_eval_gate.py --help`
4. **Check guards**: `pytest tests/test_music_guards_time_sigs.py`
