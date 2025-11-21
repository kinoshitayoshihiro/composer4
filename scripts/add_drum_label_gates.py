#!/usr/bin/env python3
"""Update quality_gates.yaml with drum label overrides.

Extends config/quality_gates.yaml to support label-based thresholds.
"""

from pathlib import Path
import yaml


LABEL_OVERRIDES = {
    "aggressive_halftime": {
        "grid_off_std_ms": {"max": 18},  # Tighter timing
        "kick_on_beat_rate": {"min": 0.70},  # Higher kick emphasis
        "snare_backbeat_rate": {"min": 0.50},  # Allow off-beat patterns
        "velocity_std_kick": {"range": [8, 16]},  # More punch variation
        "velocity_std_snare": {"range": [10, 20]},
        "hihat_density_per_bar": {"range": [6, 18]},  # Less dense hats
    },
    "sparkle_shuffle": {
        "grid_off_std_ms": {"max": 28},  # Loose swing timing
        "kick_on_beat_rate": {"min": 0.55},  # Less rigid
        "snare_backbeat_rate": {"min": 0.50},  # Flexible backbeat
        "velocity_std_kick": {"range": [4, 10]},  # Softer variation
        "velocity_std_snare": {"range": [6, 14]},
        "hihat_density_per_bar": {"range": [10, 28]},  # Dense swing hats
    },
    "intense_energy": {
        "grid_off_std_ms": {"max": 15},  # Very tight
        "kick_on_beat_rate": {"min": 0.75},  # Strong downbeats
        "snare_backbeat_rate": {"min": 0.65},
        "velocity_std_kick": {"range": [10, 18]},  # High dynamics
        "velocity_std_snare": {"range": [12, 22]},
        "hihat_density_per_bar": {"range": [12, 28]},  # Very dense
    },
    "melancholic_slow": {
        "grid_off_std_ms": {"max": 25},  # Loose, expressive
        "kick_on_beat_rate": {"min": 0.50},  # Minimal kick
        "snare_backbeat_rate": {"min": 0.40},  # Sparse snare
        "velocity_std_kick": {"range": [4, 10]},  # Soft dynamics
        "velocity_std_snare": {"range": [6, 12]},
        "hihat_density_per_bar": {"range": [4, 12]},  # Sparse hats
    },
    "calm_ambient": {
        "grid_off_std_ms": {"max": 30},  # Very loose
        "kick_on_beat_rate": {"min": 0.40},  # Minimal structure
        "snare_backbeat_rate": {"min": 0.30},
        "velocity_std_kick": {"range": [3, 8]},  # Very soft
        "velocity_std_snare": {"range": [4, 10]},
        "hihat_density_per_bar": {"range": [2, 8]},  # Very sparse
    },
}


def add_label_overrides(gates_path: Path, output_path: Path | None = None) -> None:
    """Add label_overrides section to quality_gates.yaml."""

    with open(gates_path, encoding="utf-8") as f:
        gates = yaml.safe_load(f)

    # Add label_overrides to drums section
    if "drums" not in gates:
        gates["drums"] = {}

    gates["drums"]["label_overrides"] = LABEL_OVERRIDES

    # Save
    output_path = output_path or gates_path
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(gates, f, allow_unicode=True, sort_keys=False, indent=2)

    print(f"✅ Updated {output_path}")
    print(f"   Added {len(LABEL_OVERRIDES)} drum label overrides")


if __name__ == "__main__":
    import sys

    gates_path = Path("config/quality_gates.yaml")
    if len(sys.argv) > 1:
        gates_path = Path(sys.argv[1])

    add_label_overrides(gates_path)
