# Code Quality and Formatting Guide

## Tools Installed

- **flake8**: Linting and style checking
- **black**: Automatic code formatting
- **dawdreamer**: Audio synthesis and processing

## Quick Commands

### Format Code with Black

```bash
# Format entire project
black .

# Format specific directory
black ml/ scripts/

# Check without modifying (dry-run)
black --check .

# Show diff without modifying
black --diff .
```

### Lint with Flake8

```bash
# Lint entire project
flake8 .

# Lint specific files
flake8 ml/ scripts/

# Show statistics
flake8 --statistics .

# Count errors per type
flake8 --count .
```

### Pre-commit Workflow

```bash
# Before committing, format and lint
black . && flake8 .

# Or use in CI
python -m black --check . && python -m flake8 .
```

## Configuration

### Black Settings (pyproject.toml)

- Line length: 100 characters
- Target: Python 3.11
- Excludes: build artifacts, virtual envs, outputs

### Flake8 Settings (.flake8)

- Max line length: 100 characters
- Ignored errors: E203, E266, E501, W503
- Excludes: `.venv`, `outputs`, `checkpoints`, etc.
- Per-file ignores: `__init__.py` allows F401

## Integration with CI

Add to `.github/workflows/`:

```yaml
- name: Check code formatting
  run: |
    pip install black flake8
    black --check .
    flake8 .
```

## VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "editor.formatOnSave": true
}
```

## DawDreamer Usage

DawDreamer enables real-time audio synthesis for MIDI evaluation:

```python
import dawdreamer as daw

# Create engine
engine = daw.RenderEngine(sample_rate=44100, block_size=512)

# Load MIDI and synthesize
# (See DawDreamer documentation for detailed examples)
```

Useful for:
- Real-time MIDI playback
- Audio rendering for evaluation
- VST plugin integration
- Advanced synthesis pipelines
