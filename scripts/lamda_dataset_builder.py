#!/usr/bin/env python3
"""Unified CLI entry-point for LAMDa-style dataset builders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

import yaml

from lamda_tools import DrumLoopBuildConfig, MetricConfig, build_drumloops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LAMDa dataset builders defined by YAML configs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Override the input directory specified in the YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory specified in the YAML config.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="Override the metadata directory specified in the YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without deduplicating.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional path to write the JSON summary output.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the resulting summary dictionary as JSON to stdout.",
    )
    return parser.parse_args()


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML configuration must define a mapping at the root")
    return cast(Dict[str, Any], data)


def _build_drumloop_config(
    config: Dict[str, Any],
    *,
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
) -> DrumLoopBuildConfig:
    dataset = config.get("dataset", {})
    paths = config.get("paths", {})
    dedupe = config.get("dedupe", {})
    random_cfg = config.get("random", {})
    metrics_cfg = config.get("metrics", {})
    metadata_cfg = config.get("metadata", {})
    shard_cfg = metadata_cfg.get("shard", {})

    kwargs: Dict[str, Any] = {
        "dataset_name": dataset.get("name", "Drum Loops"),
        "input_dir": input_dir or Path(paths.get("input_dir", "data/loops")),
        "output_dir": (output_dir or Path(paths.get("output_dir", "output/drumloops_cleaned"))),
        "metadata_dir": metadata_dir
        or Path(paths.get("metadata_dir", "output/drumloops_metadata")),
        "tmidix_path": Path(paths.get("tmidix_path", "data/Los-Angeles-MIDI/CODE")),
        "min_notes": dedupe.get("min_notes", 16),
        "max_file_size": dedupe.get("max_file_size", 1_000_000),
        "save_interval": dedupe.get("save_interval", 5_000),
        "start_file_number": dedupe.get("start_file_number", 0),
        "require_polyphony": dedupe.get("require_polyphony", True),
    }

    if "allowed_extensions" in dedupe:
        kwargs["allowed_extensions"] = tuple(dedupe["allowed_extensions"])

    seed = random_cfg.get("seed", 42)
    kwargs["random_seed"] = seed

    metric_kwargs = {
        key: metrics_cfg[key]
        for key in (
            "ghost_velocity_threshold",
            "accent_velocity_threshold",
            "microtiming_tolerance_ratio",
            "swing_min_pairs",
            "min_base_step",
            "max_breakpoints",
            "round_digits",
        )
        if key in metrics_cfg
    }
    kwargs["metrics"] = MetricConfig(**metric_kwargs)

    if "max_loops" in shard_cfg:
        kwargs["metadata_shard_max_loops"] = shard_cfg.get("max_loops")
    if "prefix" in shard_cfg:
        kwargs["metadata_shard_prefix"] = shard_cfg.get("prefix")

    return DrumLoopBuildConfig(**kwargs)


def _run_builder(
    dataset_type: str,
    config: Dict[str, Any],
    overrides: Dict[str, Optional[Path]],
    dry_run: bool,
) -> Dict[str, Any]:
    if dataset_type == "drumloops":
        builder_config = _build_drumloop_config(
            config,
            input_dir=overrides.get("input_dir"),
            output_dir=overrides.get("output_dir"),
            metadata_dir=overrides.get("metadata_dir"),
        )
        return build_drumloops(builder_config, dry_run=dry_run)

    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def main() -> None:
    args = _parse_args()
    config_dict = _load_yaml_config(args.config)

    dataset_info = config_dict.get("dataset", {})
    dataset_type = dataset_info.get("type")
    if not dataset_type:
        raise ValueError("Config is missing 'dataset.type'")

    summary = _run_builder(
        dataset_type,
        config_dict,
        {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "metadata_dir": args.metadata_dir,
        },
        args.dry_run,
    )

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
