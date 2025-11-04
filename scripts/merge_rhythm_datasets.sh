#!/usr/bin/env bash
set -euo pipefail

#############################################
# Merge Rhythm AI Datasets
# drumclean (51,248) + groove (827) + E-GMD (45,537)
# Total: ~97,600 records
#############################################

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "$BASE_DIR" || exit 1

OUTPUT_DIR="output/rhythm_ai"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/rhythm_merge_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=========================================="
log "🔗 Merging Rhythm AI Datasets"
log "=========================================="
log "Base Dir: $BASE_DIR"
log "Output Dir: $OUTPUT_DIR"
log "Log File: $LOG_FILE"
log ""

## ========== 1. Verify Input Parquet Files ==========
log "📂 1. Verifying input Parquet files..."

DRUMCLEAN_PARQUET="$OUTPUT_DIR/drumclean_stage2/rhythm_features.parquet"
GROOVE_PARQUET="$OUTPUT_DIR/groove_stage2/rhythm_features.parquet"
EGMD_PARQUET="$OUTPUT_DIR/egmd_stage2/rhythm_features.parquet"

if [ ! -f "$DRUMCLEAN_PARQUET" ]; then
  log "❌ ERROR: drumclean parquet not found"
  exit 1
fi

log "✅ drumclean parquet found"

GROOVE_EXISTS=0
if [ -f "$GROOVE_PARQUET" ]; then
  log "✅ groove parquet found"
  GROOVE_EXISTS=1
else
  log "ℹ️  groove parquet not found (optional)"
fi

EGMD_EXISTS=0
if [ -f "$EGMD_PARQUET" ]; then
  log "✅ E-GMD parquet found"
  EGMD_EXISTS=1
else
  log "ℹ️  E-GMD parquet not found (optional)"
fi

## ========== 2. Merge Parquet Files ==========
log ""
log "🔗 2. Merging Parquet files..."

python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import pandas as pd
from pathlib import Path

output_dir = Path("output/rhythm_ai")
drumclean_path = output_dir / "drumclean_stage2" / "rhythm_features.parquet"
groove_path = output_dir / "groove_stage2" / "rhythm_features.parquet"
egmd_path = output_dir / "egmd_stage2" / "rhythm_features.parquet"

# Load drumclean (required)
print(f"📂 Loading drumclean: {drumclean_path}")
df_drumclean = pd.read_parquet(drumclean_path)
print(f"   Records: {len(df_drumclean)}")

# Load groove (optional)
try:
    print(f"📂 Loading groove: {groove_path}")
    df_groove = pd.read_parquet(groove_path)
    print(f"   Records: {len(df_groove)}")
except Exception as e:
    print(f"⚠️  groove not found: {e}")
    df_groove = pd.DataFrame(columns=df_drumclean.columns)

# Load E-GMD (optional)
try:
    print(f"📂 Loading E-GMD: {egmd_path}")
    df_egmd = pd.read_parquet(egmd_path)
    print(f"   Records: {len(df_egmd)}")
except Exception as e:
    print(f"⚠️  E-GMD not found: {e}")
    df_egmd = pd.DataFrame(columns=df_drumclean.columns)

# Merge all datasets
print("")
print("🔗 Merging datasets...")
df_merged = pd.concat([df_drumclean, df_groove, df_egmd], ignore_index=True)
print(f"   Total records (before dedup): {len(df_merged)}")

# Remove duplicates
df_merged = df_merged.drop_duplicates(subset=['loop_id'], keep='first')
print(f"   Total records (after dedup): {len(df_merged)}")

# Verify required columns
required_cols = [
    'loop_id', 'tempo_bpm', 'swing_pct', 'backbeat_strength',
    'kick_downbeat_rate', 'snare_backbeat_rate', 'hat_density', 'family_label'
]

missing_cols = [col for col in required_cols if col not in df_merged.columns]
if missing_cols:
    print(f"❌ ERROR: Missing columns: {missing_cols}")
    exit(1)

print(f"✅ All required columns present")

# Save merged parquet
output_path = output_dir / "rhythm_features_merged.parquet"
output_path.parent.mkdir(parents=True, exist_ok=True)
df_merged.to_parquet(output_path, compression='snappy', index=False)
print(f"✅ Merged parquet saved: {output_path}")

# Statistics
print("")
print("📊 Dataset Statistics:")
print(f"   drumclean: {len(df_drumclean)}")
print(f"   groove:    {len(df_groove)}")
print(f"   E-GMD:     {len(df_egmd)}")
print(f"   TOTAL:     {len(df_merged)}")
print("")
print(f"   Columns: {len(df_merged.columns)}")
print(f"   Family labels: {df_merged['family_label'].value_counts().to_dict()}")

PY

if [ $? -ne 0 ]; then
  log "❌ ERROR: Merge failed"
  exit 1
fi

log ""
log "✅ Dataset Merge COMPLETE"
log ""

## ========== 3. Train ML Model ==========
log "🚀 3. Training ML model on merged dataset..."

python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import pandas as pd
import pickle
from pathlib import Path

# Load merged dataset
df = pd.read_parquet('output/rhythm_ai/rhythm_features_merged.parquet')
print(f"📂 Loaded {len(df)} records")

# Prepare features
target_col = 'family_label' if 'family_label' in df.columns else 'label'
feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]

print(f"   Features: {len(feature_cols)}")
print(f"   Target: {target_col}")

X = df[feature_cols].fillna(0.0).values
y = df[target_col].astype(str).values

# Train model (XGBoost → LogisticRegression fallback)
print("")
print("🚀 Training model...")

try:
    from xgboost import XGBClassifier
    print("   Using XGBoost")
    model = XGBClassifier(
        objective="multi:softprob",
        max_depth=6,
        n_estimators=200,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X, y)
    meta = {"algo": "xgb"}
except Exception as e:
    print(f"   XGBoost failed: {e}")
    print("   Falling back to LogisticRegression")
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=4000, random_state=42)
    model.fit(X, y)
    meta = {"algo": "logreg"}

print(f"✅ Model trained: {meta['algo']}")

# Save unified pickle
output_dir = Path('data/patterns')
output_dir.mkdir(parents=True, exist_ok=True)

package = {
    "schema_version": "stage2_drums_v1",
    "model_meta": meta,
    "model": model,
    "class_labels": sorted(pd.unique(y).tolist()),
    "feature_names": feature_cols,
    "target_col": target_col
}

pickle_path = output_dir / 'stage2_drums_rhythm_ai.pickle'
with open(pickle_path, 'wb') as f:
    pickle.dump(package, f)

print(f"✅ Saved: {pickle_path}")
print("")
print(f"   Classes: {package['class_labels']}")
print(f"   Features: {len(package['feature_names'])}")

PY

if [ $? -ne 0 ]; then
  log "❌ ERROR: ML training failed"
  exit 1
fi

log ""
log "=========================================="
log "🎉 Rhythm AI Merge & Training Complete!"
log "=========================================="
log "Output files:"
log "  - Merged Parquet: output/rhythm_ai/rhythm_features_merged.parquet"
log "  - ML Model: data/patterns/stage2_drums_rhythm_ai.pickle"
log ""
log "Next steps:"
log "  1. Run benchmark validation"
log "  2. Verify Phase 27 optimization (p95 < 50ms)"
log ""
