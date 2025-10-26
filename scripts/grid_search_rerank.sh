#!/bin/bash
set -euo pipefail
# Grid Search for Re-ranking Parameters
# Sweeps: threshold × Chorus weights

cd "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "=== Grid Search: threshold × Chorus weights ==="
echo "Target: Accent Delta >= +5%, ML Usage >= 70%, Family Match >= 80%"
echo ""

# Threshold sweep
for TH in 0.20 0.25 0.30 0.35; do
  echo "Testing threshold=$TH..."
  .venv311/bin/python scripts/ab_test_guitar_v3.py \
    --num-songs 50 \
    --v3-pickle data/patterns/stage2_guitar_v3_meta.pickle \
    --conf-thresh $TH \
    --w-proba 0.55 --w-accent 0.30 --w-density 0.10 --w-section 0.05 \
    --output "data/ab_v3_rerank_TH${TH}.csv" 2>&1 | tail -30
  echo ""
done

echo "=== Generating Summary ==="
.venv311/bin/python - << 'PY'
import glob, pandas as pd

rows = []
for f in sorted(glob.glob("data/ab_v3_rerank_TH*.csv")):
    th = f.split("TH")[1].split(".csv")[0]
    df = pd.read_csv(f)
    
    s = {
        "file": f,
        "threshold": float(th),
        "family_match%": 100 * df["family_match"].mean(),
        "accent_delta%": df["accent_delta"].mean() * 100,
        "density_abs": df["density_diff"].abs().median(),
        "ml_usage%": 100 * df["ml_used"].mean(),
        "top1_proba": df["top1_proba"].mean(),
        "samples": len(df),
    }
    
    # Section-wise Chorus
    chorus_df = df[df["section"] == "Chorus"]
    if len(chorus_df) > 0:
        s["chorus_accent%"] = chorus_df["accent_delta"].mean() * 100
        s["chorus_ml%"] = 100 * chorus_df["ml_used"].mean()
    
    rows.append(s)

summary_df = pd.DataFrame(rows)
summary_df.to_csv("data/ab_v3_grid_summary.csv", index=False)

print("\n=== Grid Search Summary ===")
print(summary_df.to_string(index=False))
print("\nSaved to: data/ab_v3_grid_summary.csv")

# Best parameters with KPI gate
print("\n=== Best Parameters (KPI-gated) ===")
passed = summary_df[
    (summary_df["ml_usage%"] >= 70.0) &
    (summary_df["family_match%"] >= 80.0) &
    (summary_df["density_abs"] <= 1.0)
]
if len(passed) == 0:
    print("⚠️  No config meets all KPIs yet. Pick the highest accent_delta% as reference.")
    best = summary_df.loc[summary_df["accent_delta%"].idxmax()]
else:
    print(f"✓ {len(passed)} configs meet KPIs. Selecting best accent_delta%.")
    best = passed.loc[passed["accent_delta%"].idxmax()]

print("\nBest configuration:")
print(best.to_string())

# Save best to YAML for reproducibility
import yaml, time
best_config = {
    "selected": {
        "threshold": float(best["threshold"]),
        "w_proba": 0.55, 
        "w_accent": 0.30, 
        "w_density": 0.10, 
        "w_section": 0.05
    },
    "metrics": {
        "accent_delta%": float(best["accent_delta%"]),
        "ml_usage%": float(best["ml_usage%"]),
        "family_match%": float(best["family_match%"]),
        "density_abs": float(best["density_abs"]),
        "top1_proba": float(best["top1_proba"])
    },
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
}
with open("data/ab_v3_best.yaml", "w") as f:
    yaml.safe_dump(best_config, f, default_flow_style=False)

print("\n✓ Saved best config to: data/ab_v3_best.yaml")
PY

echo ""
echo "=== Grid Search Complete ==="
