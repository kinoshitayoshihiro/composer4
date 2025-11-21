#!/usr/bin/env python3
"""
Style Presets Generator

3種のスタイルプリセット（soft/standard/bright）を生成

Usage:
    python make_style_presets.py \
        --out-yaml analysis/style_presets.yaml
"""

import argparse
import yaml
from pathlib import Path


def generate_style_presets() -> dict:
    """スタイルプリセット定義"""

    presets = {
        "soft": {
            "description": "Soft, intimate arrangement with minimal instrumentation",
            "velocity": {
                "global_scale": 0.7,
                "piano": 0.6,
                "strings": 0.7,
                "guitar": 0.5,
                "drums": 0.5,
            },
            "density": {
                "global_scale": 0.6,
                "piano": 0.5,
                "strings": 0.6,
                "guitar": 0.4,
                "drums": 0.4,
            },
            "instrumentation": {
                "piano": True,
                "strings": True,
                "guitar": True,
                "drums": False,  # No drums in soft
                "synth_pad": True,
            },
            "effects": {
                "reverb": 0.8,
                "delay": 0.3,
                "chorus": 0.2,
            },
        },
        "standard": {
            "description": "Balanced arrangement with full instrumentation",
            "velocity": {
                "global_scale": 0.85,
                "piano": 0.8,
                "strings": 0.85,
                "guitar": 0.8,
                "drums": 0.8,
            },
            "density": {
                "global_scale": 0.75,
                "piano": 0.7,
                "strings": 0.75,
                "guitar": 0.7,
                "drums": 0.75,
            },
            "instrumentation": {
                "piano": True,
                "strings": True,
                "guitar": True,
                "drums": True,
                "synth_pad": True,
            },
            "effects": {
                "reverb": 0.5,
                "delay": 0.2,
                "chorus": 0.15,
            },
        },
        "bright": {
            "description": "Energetic, vibrant arrangement with emphasis on rhythm",
            "velocity": {
                "global_scale": 1.0,
                "piano": 0.95,
                "strings": 0.9,
                "guitar": 1.0,
                "drums": 1.0,
            },
            "density": {
                "global_scale": 0.9,
                "piano": 0.85,
                "strings": 0.8,
                "guitar": 0.9,
                "drums": 1.0,
            },
            "instrumentation": {
                "piano": True,
                "strings": True,
                "guitar": True,
                "drums": True,
                "synth_pad": False,  # No pad in bright
            },
            "effects": {
                "reverb": 0.3,
                "delay": 0.4,
                "chorus": 0.3,
            },
        },
    }

    return presets


def main():
    parser = argparse.ArgumentParser(description="Style presets generator")
    parser.add_argument("--out-yaml", type=Path, required=True)

    args = parser.parse_args()

    # スタイルプリセット生成
    presets = generate_style_presets()

    # YAML出力
    with open(args.out_yaml, "w") as f:
        yaml.dump(presets, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ Style presets generated: {len(presets)} variants")
    print(f"   Output: {args.out_yaml}")
    print(f"   Variants: {', '.join(presets.keys())}")


if __name__ == "__main__":
    main()
