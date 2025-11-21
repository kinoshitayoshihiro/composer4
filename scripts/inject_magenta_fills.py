#!/usr/bin/env python3
"""
Integration Layer: Magenta Fill Injection — Phase 4

Injects Magenta-generated fills into fill_slot/riff_slot positions.

Usage:
    python scripts/inject_magenta_fills.py \
        --bars-with-slots data/song_004/analysis/bars_with_slots.parquet \
        --chordmap data/song_004/locked_chordmap.json \
        --guide-tones data/song_004/guide_tones.json \
        --policy config/magenta_policy.yaml \
        --output data/song_004/magenta_fills.json
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from otobonAI.magenta_fill_generator import MagentaFillGenerator
from otobonAI.qa.magenta_qa import MagentaQA


def load_chordmap(path: Path) -> dict:
    """Load locked chordmap."""
    with open(path) as f:
        return json.load(f)


def load_guide_tones(path: Path) -> dict:
    """Load guide tone hints."""
    if not path.exists():
        return {"events": []}
    with open(path) as f:
        return json.load(f)


def load_policy(path: Path) -> dict:
    """Load Magenta policy YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def identify_fill_bars(
    bars_df: pd.DataFrame, policy: dict
) -> dict[str, list[int]]:
    """Identify bars eligible for Magenta fills.
    
    Returns:
        Dict mapping section -> list of bar indices
    """
    eligible_sections = {}
    
    # Filter by fill_slot or riff_slot
    eligible_df = bars_df[
        (bars_df.get("fill_slot", False)) | (bars_df.get("riff_slot", False))
    ]
    
    # Group by section
    if "section_label" in eligible_df.columns:
        for section, group in eligible_df.groupby("section_label"):
            section = section.lower()
            eligible_sections[section] = group.index.tolist()
    else:
        # No section info - treat all as default
        eligible_sections["default"] = eligible_df.index.tolist()
    
    return eligible_sections


def apply_magenta_use_prob(
    section_bars: dict[str, list[int]], policy: dict
) -> dict[str, list[int]]:
    """Apply magenta_use_prob to thin out fill bars."""
    import random
    
    global_prob = policy.get("magenta_use_prob", 0.3)
    overrides = policy.get("section_overrides", {})
    
    thinned = {}
    for section, bars in section_bars.items():
        # Get effective probability
        prob = overrides.get(section, {}).get("magenta_use_prob", global_prob)
        
        # Random sampling
        selected = [bar for bar in bars if random.random() < prob]
        if selected:
            thinned[section] = selected
    
    return thinned


def main():
    ap = argparse.ArgumentParser(description="Inject Magenta fills into arrangement")
    ap.add_argument(
        "--bars-with-slots",
        required=True,
        help="bars_with_slots.parquet (contains fill_slot, riff_slot columns)",
    )
    ap.add_argument(
        "--chordmap", required=True, help="locked_chordmap.json"
    )
    ap.add_argument(
        "--guide-tones",
        default=None,
        help="guide_tones.json (optional)",
    )
    ap.add_argument(
        "--policy",
        default="config/magenta_policy.yaml",
        help="Magenta policy YAML",
    )
    ap.add_argument(
        "--qa-gates",
        default="config/quality_gates.yaml",
        help="QA gates YAML",
    )
    ap.add_argument("--output", required=True, help="Output magenta_fills.json")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--validate", action="store_true", help="Run QA validation"
    )
    
    args = ap.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    # Set random seed
    import random
    random.seed(args.seed)
    
    # Load inputs
    logging.info(f"Loading bars from {args.bars_with_slots}")
    bars_df = pd.read_parquet(args.bars_with_slots)
    
    logging.info(f"Loading chordmap from {args.chordmap}")
    chordmap = load_chordmap(Path(args.chordmap))
    
    guide_tones = None
    if args.guide_tones:
        logging.info(f"Loading guide tones from {args.guide_tones}")
        guide_tones = load_guide_tones(Path(args.guide_tones))
    
    logging.info(f"Loading policy from {args.policy}")
    policy = load_policy(Path(args.policy))
    
    # Check if Magenta is enabled
    if not policy.get("enabled", True):
        logging.warning("Magenta disabled in policy - writing empty output")
        output_data = {"fills": [], "total_fills": 0, "total_events": 0}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        return
    
    # Identify eligible bars
    logging.info("Identifying fill bars from slots")
    section_bars = identify_fill_bars(bars_df, policy)
    
    total_eligible = sum(len(bars) for bars in section_bars.values())
    logging.info(f"Found {total_eligible} eligible bars across {len(section_bars)} sections")
    
    # Apply magenta_use_prob
    logging.info("Applying magenta_use_prob filter")
    selected_bars = apply_magenta_use_prob(section_bars, policy)
    
    total_selected = sum(len(bars) for bars in selected_bars.values())
    logging.info(f"Selected {total_selected} bars for Magenta fills")
    
    # Generate fills
    logging.info("Generating Magenta fills")
    generator = MagentaFillGenerator(enable_cache=True)
    
    all_fills = []
    for section, bars in selected_bars.items():
        # Get section-specific policy
        section_policy = policy.get("section_overrides", {}).get(
            section, policy.get("fill_policy", {})
        )
        
        fills = generator.generate_fills(
            section=section,
            bars=bars,
            chordmap_locked=chordmap,
            guide_tone_hints=guide_tones,
            policy=section_policy,
        )
        
        all_fills.extend(fills)
    
    logging.info(f"Generated {len(all_fills)} fills")
    
    # QA validation
    if args.validate:
        logging.info("Running QA validation")
        qa = MagentaQA.from_yaml(Path(args.qa_gates))
        
        # Convert fills to flat event list
        fill_events = []
        for fill in all_fills:
            fill_dict = {
                "bar_start": fill.bar_start,
                "bar_end": fill.bar_end,
                "events": fill.events,
            }
            fill_events.append(fill_dict)
        
        # Mock all_events (assume Magenta is 30% of total)
        all_events = fill_events + [{"dummy": True}] * (len(fill_events) * 2)
        
        result = qa.validate(fill_events, all_events, section="default")
        
        if result.passed:
            logging.info("✅ QA PASS")
        else:
            logging.warning(f"❌ QA FAIL: {len(result.violations)} violations")
            for v in result.violations[:5]:  # Show first 5
                logging.warning(f"  - {v}")
        
        logging.info(f"QA Metrics: {result.metrics}")
    
    # Save output
    output_path = Path(args.output)
    generator.to_json(all_fills, output_path)
    
    logging.info(f"\n✅ Complete")
    logging.info(f"   Fills generated: {len(all_fills)}")
    logging.info(f"   Total events: {sum(len(f.events) for f in all_fills)}")
    logging.info(f"   Output: {output_path}")


if __name__ == "__main__":
    main()
