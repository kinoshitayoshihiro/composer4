#!/usr/bin/env python3
"""
GMD Groove Metrics Dashboard
==============================
Phase B-2: Groove指標ダッシュボード

入力:
  - data/GMD_processed/index.parquet

出力:
  - data/GMD_processed/dashboard.html（インタラクティブダッシュボード）
  - data/GMD_processed/metrics_summary.md（Markdown統計レポート）

Usage:
  python ops/gmd_dashboard.py \\
    --index data/GMD_processed/index.parquet \\
    --out data/GMD_processed/dashboard.html
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # バックエンド設定（GUIなし）
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not installed", file=sys.stderr)


def generate_summary_markdown(df: pd.DataFrame, out_path: Path):
    """Markdown統計レポート生成"""
    md = ["# GMD Groove Metrics Summary", ""]
    
    # Dataset overview
    md.append("## Dataset Overview")
    md.append(f"- Total files: {len(df)}")
    md.append(f"- Drummers: {df['drummer'].nunique()}")
    md.append(f"- Sessions: {df['session'].nunique()}")
    md.append(f"- Genres: {df['genre'].nunique()}")
    md.append("")
    
    # Groove metrics statistics
    md.append("## Groove Metrics Statistics")
    md.append("")
    md.append("| Metric | Mean | Std | Min | Max |")
    md.append("|--------|------|-----|-----|-----|")
    
    for col in ["velocity_std", "ioi_std", "note_density"]:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            md.append(f"| {col} | {mean:.2f} | {std:.2f} | {min_val:.2f} | {max_val:.2f} |")
    
    md.append("")
    
    # Genre distribution
    md.append("## Genre Distribution")
    md.append("")
    genre_counts = df["genre"].value_counts()
    for genre, count in genre_counts.head(10).items():
        md.append(f"- {genre}: {count} files")
    md.append("")
    
    # BPM distribution
    md.append("## BPM Distribution")
    md.append(f"- Mean: {df['bpm'].mean():.1f}")
    md.append(f"- Std: {df['bpm'].std():.1f}")
    md.append(f"- Range: {df['bpm'].min()}-{df['bpm'].max()}")
    md.append("")
    
    # Type distribution
    md.append("## Type Distribution")
    type_counts = df["type"].value_counts()
    for typ, count in type_counts.items():
        md.append(f"- {typ}: {count} files ({count/len(df)*100:.1f}%)")
    md.append("")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def generate_dashboard_html(df: pd.DataFrame, out_path: Path):
    """インタラクティブHTMLダッシュボード生成"""
    if not PLOT_AVAILABLE:
        print("⚠️  Plotting skipped (matplotlib/seaborn not available)")
        return
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Velocity std distribution
    ax = axes[0, 0]
    df["velocity_std"].hist(bins=30, ax=ax, color="skyblue", edgecolor="black")
    ax.set_title("Velocity Std Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Velocity Std")
    ax.set_ylabel("Frequency")
    ax.axvline(df["velocity_std"].mean(), color="red", linestyle="--", label=f"Mean: {df['velocity_std'].mean():.2f}")
    ax.legend()
    
    # 2. IOI std distribution
    ax = axes[0, 1]
    df["ioi_std"].hist(bins=30, ax=ax, color="lightgreen", edgecolor="black")
    ax.set_title("IOI Std Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("IOI Std (sec)")
    ax.set_ylabel("Frequency")
    ax.axvline(df["ioi_std"].mean(), color="red", linestyle="--", label=f"Mean: {df['ioi_std'].mean():.4f}")
    ax.legend()
    
    # 3. Note density distribution
    ax = axes[0, 2]
    df["note_density"].hist(bins=30, ax=ax, color="salmon", edgecolor="black")
    ax.set_title("Note Density Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Notes/sec")
    ax.set_ylabel("Frequency")
    ax.axvline(df["note_density"].mean(), color="red", linestyle="--", label=f"Mean: {df['note_density'].mean():.2f}")
    ax.legend()
    
    # 4. BPM distribution
    ax = axes[1, 0]
    df["bpm"].hist(bins=30, ax=ax, color="gold", edgecolor="black")
    ax.set_title("BPM Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("BPM")
    ax.set_ylabel("Frequency")
    ax.axvline(df["bpm"].mean(), color="red", linestyle="--", label=f"Mean: {df['bpm'].mean():.1f}")
    ax.legend()
    
    # 5. Genre distribution (top 10)
    ax = axes[1, 1]
    genre_counts = df["genre"].value_counts().head(10)
    genre_counts.plot(kind="barh", ax=ax, color="orchid", edgecolor="black")
    ax.set_title("Top 10 Genres", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("Genre")
    
    # 6. Velocity std vs IOI std scatter
    ax = axes[1, 2]
    ax.scatter(df["velocity_std"], df["ioi_std"], alpha=0.5, c="steelblue", edgecolors="black")
    ax.set_title("Velocity Std vs IOI Std", fontsize=14, fontweight="bold")
    ax.set_xlabel("Velocity Std")
    ax.set_ylabel("IOI Std (sec)")
    
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {out_path.with_suffix('.png')}")
    
    # Simple HTML wrapper
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GMD Groove Metrics Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ccc; margin: 20px 0; }}
        .stats {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>GMD Groove Metrics Dashboard</h1>
    <div class="stats">
        <h2>Dataset Overview</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Files</td><td>{len(df)}</td></tr>
            <tr><td>Drummers</td><td>{df['drummer'].nunique()}</td></tr>
            <tr><td>Genres</td><td>{df['genre'].nunique()}</td></tr>
        </table>
    </div>
    <img src="{out_path.with_suffix('.png').name}" alt="Groove Metrics Visualizations">
</body>
</html>
"""
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="GMD Groove Dashboard")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/GMD_processed/index.parquet"),
        help="GMD index.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/GMD_processed/dashboard.html"),
        help="Output dashboard.html"
    )
    args = parser.parse_args()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 GMD Groove Dashboard Generator")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Load index
    print(f"Loading: {args.index}")
    df = pd.read_parquet(args.index)
    print(f"   Loaded: {len(df)} files")
    print()
    
    # Generate markdown summary
    md_path = args.out.with_suffix(".md")
    generate_summary_markdown(df, md_path)
    print(f"✅ Saved: {md_path}")
    
    # Generate HTML dashboard
    generate_dashboard_html(df, args.out)
    print(f"✅ Saved: {args.out}")
    
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ Dashboard Generation Complete!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
