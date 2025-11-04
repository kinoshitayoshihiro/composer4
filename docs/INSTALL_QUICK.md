# Quick Reference: Installation Commands

## One-Liner Install (Choose One)

```bash
# Full installation (recommended for first-time setup)
./install_deps.sh

# Development environment
./install_deps.sh --dev

# Testing only
./install_deps.sh --test

# Minimal runtime
./install_deps.sh --minimal
```

## Manual Install (Step-by-Step)

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install all dependencies at once
pip install -r requirements.txt \
            -r requirements-dev.txt \
            -r requirements-test.txt \
            -r requirements-extra.txt
```

## Quick Verification

```bash
# Verify installation
python -c "import torch, numpy, librosa, music21, pytest, jsonschema; print('✅ All OK')"

# Verify code quality tools
python -c "import black, flake8; print('✅ Linting tools OK')"

# Verify audio tools
python -c "import dawdreamer; print('✅ Audio synthesis OK')"

# Run tests
pytest tests/test_music_guards_time_sigs.py -v
```

## Platform-Specific Shortcuts

### macOS/Linux (with virtual environment)
```bash
python -m venv .venv311 && \
source .venv311/bin/activate && \
./install_deps.sh
```

### Windows (PowerShell)
```powershell
python -m venv .venv311
.venv311\Scripts\Activate.ps1
.\install_deps.sh
```

## Troubleshooting One-Liners

```bash
# Fix torch CPU version
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Install missing jsonschema
pip install jsonschema

# Reinstall all with no cache
pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
```

See [INSTALL.md](INSTALL.md) for detailed instructions.
