#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piano External Benchmark Trend Visualization (Phase 4.3)

Reads history JSONL and generates trend charts for Nightly CI.

Usage:
    python scripts/visualize_piano_trends.py \\
      --history output/reports/piano_external_bench_history.jsonl \\
      --out-dir output/reports/trends \\
      --png  # Optional: Generate PNG charts
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# Optional matplotlib support
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Threshold definitions (aligned with eval_piano_external.py)
THRESHOLDS = {
    "chord_tone_rate": {"min": 0.70},
    "hand_separation": {"min": 0.60},
    "velocity_std": {"min": 15.0, "max": 25.0},
    "bar_violation_rate": {"max": 0.02},
    "notes_per_bar": {"min": 8.0, "max": 16.0},
}


def load_history(history_file: Path) -> List[Dict[str, Any]]:
    """Load history JSONL file."""
    if not history_file.exists():
        return []
    
    entries = []
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return entries


def generate_markdown_report(entries: List[Dict], out_path: Path):
    """Generate markdown trend report."""
    if not entries:
        out_path.write_text("# Piano External Benchmark Trends\n\nNo data available yet.\n")
        return
    
    # Sort by timestamp
    entries = sorted(entries, key=lambda x: x.get('timestamp', ''))
    
    # Build markdown
    lines = [
        "# Piano External Benchmark Trends",
        "",
        f"**Total Evaluations:** {len(entries)}",
        f"**Date Range:** {entries[0].get('date', 'N/A')} to {entries[-1].get('date', 'N/A')}",
        "",
        "## Summary Statistics",
        "",
        "| Date | Chord Tone Rate | Hand Separation | Velocity Std | Bar Violation | Notes/Bar |",
        "|------|----------------|-----------------|--------------|---------------|-----------|",
    ]
    
    for entry in entries[-10:]:  # Last 10 entries
        date = entry.get('date', 'N/A')
        summary = entry.get('summary', {})
        
        chord_tone = summary.get('chord_tone_rate', {}).get('mean', 0.0)
        hand_sep = summary.get('hand_separation', {}).get('mean', 0.0)
        vel_std = summary.get('velocity_std', {}).get('mean', 0.0)
        bar_viol = summary.get('bar_violation_rate', {}).get('mean', 0.0)
        notes_bar = summary.get('notes_per_bar', {}).get('mean', 0.0)
        
        lines.append(
            f"| {date} | {chord_tone:.4f} | {hand_sep:.4f} | {vel_std:.2f} | {bar_viol:.4f} | {notes_bar:.2f} |"
        )
    
    lines.extend([
        "",
        "## Trend Analysis",
        "",
        "### Chord Tone Rate (Target: >0.70)",
        f"- Latest: **{entries[-1].get('summary', {}).get('chord_tone_rate', {}).get('mean', 0.0):.4f}**",
        f"- Baseline: {entries[0].get('summary', {}).get('chord_tone_rate', {}).get('mean', 0.0):.4f}",
        "",
        "### Hand Separation (Target: >0.60)",
        f"- Latest: **{entries[-1].get('summary', {}).get('hand_separation', {}).get('mean', 0.0):.4f}**",
        f"- Baseline: {entries[0].get('summary', {}).get('hand_separation', {}).get('mean', 0.0):.4f}",
        "",
        "### Bar Violation Rate (Target: <0.02)",
        f"- Latest: **{entries[-1].get('summary', {}).get('bar_violation_rate', {}).get('mean', 0.0):.4f}**",
        f"- Baseline: {entries[0].get('summary', {}).get('bar_violation_rate', {}).get('mean', 0.0):.4f}",
        "",
        "## Notes",
        "",
        "- Chord tone rate: Measure of harmonic consistency (pitch class diversity)",
        "- Hand separation: Proxy for left/right hand independence (pitch range spread)",
        "- Velocity std: Dynamic expression diversity",
        "- Bar violation rate: Notes spanning multiple bars (lower is better)",
        "- Notes per bar: Average density",
        "",
        "---",
        f"*Generated from {len(entries)} evaluations*",
    ])
    
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"[saved] Markdown report: {out_path}")


def generate_ascii_chart(entries: List[Dict], metric_key: str, metric_name: str) -> str:
    """Generate simple ASCII line chart."""
    if not entries:
        return f"No data for {metric_name}"
    
    # Extract values
    values = []
    for entry in entries:
        summary = entry.get('summary', {})
        val = summary.get(metric_key, {}).get('mean', 0.0)
        values.append(val)
    
    if not values:
        return f"No data for {metric_name}"
    
    # Normalize to 0-20 range for ASCII
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val > min_val else 1.0
    
    lines = [f"{metric_name} Trend (Last {len(values)} evaluations)"]
    lines.append(f"Range: {min_val:.4f} - {max_val:.4f}")
    lines.append("")
    
    for i, val in enumerate(values):
        norm = int(((val - min_val) / range_val) * 20)
        bar = "█" * norm + "▒" * (20 - norm)
        lines.append(f"{i+1:2d} | {bar} | {val:.4f}")
    
    return '\n'.join(lines)


def generate_png_charts(entries: List[Dict], out_dir: Path) -> List[Path]:
    """Generate PNG charts for all metrics (optional, requires matplotlib)."""
    if not HAS_MATPLOTLIB:
        print("[warn] matplotlib not available, skipping PNG generation")
        return []
    
    if not entries:
        print("[warn] No data for PNG charts")
        return []
    
    metrics = [
        ("chord_tone_rate", "Chord Tone Rate"),
        ("hand_separation", "Hand Separation"),
        ("velocity_std", "Velocity Std"),
        ("bar_violation_rate", "Bar Violation Rate"),
        ("notes_per_bar", "Notes Per Bar"),
    ]
    
    png_paths = []
    
    for metric_key, metric_name in metrics:
        # Extract values
        dates = [e.get('date', f"#{i+1}") for i, e in enumerate(entries)]
        values = [e.get('summary', {}).get(metric_key, {}).get('mean', 0.0) for e in entries]
        
        if not values or all(v == 0.0 for v in values):
            continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(values)), values, marker='o', linewidth=2, markersize=6, label=metric_name)
        
        # Draw threshold lines (if defined)
        th = THRESHOLDS.get(metric_key)
        if th:
            if "min" in th:
                ax.axhline(y=th["min"], color='g', linestyle='--', linewidth=1, alpha=0.6, label=f'Min: {th["min"]}')
            if "max" in th:
                ax.axhline(y=th["max"], color='r', linestyle='--', linewidth=1, alpha=0.6, label=f'Max: {th["max"]}')
        
        ax.set_xlabel('Evaluation #')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels (show every Nth date if too many)
        step = max(1, len(dates) // 10)
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        png_path = out_dir / f"{metric_key}_trend.png"
        plt.savefig(png_path, dpi=100)
        plt.close()
        
        png_paths.append(png_path)
        print(f"[saved] PNG chart: {png_path}")
    
    return png_paths


def main():
    ap = argparse.ArgumentParser(description="Visualize Piano External Benchmark Trends")
    ap.add_argument("--history", required=True, help="History JSONL file")
    ap.add_argument("--out-dir", required=True, help="Output directory for reports")
    ap.add_argument("--png", action="store_true", help="Generate PNG charts (requires matplotlib)")
    args = ap.parse_args()
    
    history_file = Path(args.history)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[info] Loading history: {history_file}")
    entries = load_history(history_file)
    
    if not entries:
        print("[warn] No history entries found")
        return
    
    print(f"[info] Found {len(entries)} entries")
    
    # Generate markdown report
    md_path = out_dir / "piano_external_trends.md"
    generate_markdown_report(entries, md_path)
    
    # Generate PNG charts (optional)
    png_paths = []
    if args.png:
        png_paths = generate_png_charts(entries, out_dir)
    
    # Generate ASCII charts (for terminal output)
    print("\n" + "=" * 60)
    print(generate_ascii_chart(entries, "chord_tone_rate", "Chord Tone Rate"))
    print("\n" + "=" * 60)
    print(generate_ascii_chart(entries, "hand_separation", "Hand Separation"))
    print("\n" + "=" * 60)
    print(generate_ascii_chart(entries, "bar_violation_rate", "Bar Violation Rate"))
    print("=" * 60 + "\n")
    
    print(f"[done] Trend visualization complete")
    print(f"       Markdown: {md_path}")
    if png_paths:
        print(f"       PNG charts: {len(png_paths)} files in {out_dir}")


if __name__ == "__main__":
    main()
