#!/bin/bash
# scripts/run_rhythm_stage2.sh
# Rhythm AI Stage2実行

set -e

BASE_DIR="${BASE_DIR:-$(pwd)}"
cd "${BASE_DIR}"

source .venv311/bin/activate

LOG_FILE="${LOG_FILE:-logs/rhythm_stage2_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "========================================="
log "Rhythm AI Stage2 Processing"
log "========================================="
log "Log: ${LOG_FILE}"
log ""

# ========== Path Resolution ==========
EXTRACTOR_PATH="${EXTRACTOR_PATH:-scripts/rhythm_stage2_extractor.py}"
[ -f "$EXTRACTOR_PATH" ] || EXTRACTOR_PATH="rhythm_stage2_extractor.py"
CONFIG_PATH="${CONFIG_PATH:-configs/rhythm_stage2.yaml}"
[ -f "$CONFIG_PATH" ] || CONFIG_PATH="rhythm_stage2.yaml"

log "Extractor: $EXTRACTOR_PATH"
log "Config:    $CONFIG_PATH"
log ""

# ========== drumclean_midi Stage2 ==========
log "📂 1. Processing drumclean_midi..."

python "$EXTRACTOR_PATH" \
  --lamda-index output/rhythm_ai/drumclean_metadata/drums_index.pkl \
  --input-dir output/rhythm_ai/drumclean_midi \
  --output-dir output/rhythm_ai/drumclean_stage2 \
  --config "$CONFIG_PATH" \
  --verbose \
  2>&1 | tee -a "${LOG_FILE}"

if [ $? -eq 0 ]; then
    log "✅ drumclean_midi Stage2 completed"
else
    log "❌ drumclean_midi Stage2 failed"
    exit 1
fi

log ""

# ========== groove Stage2 (optional) ==========
if [ "${SKIP_GROOVE:-0}" = "1" ]; then
    log "⏩ 2. Skipping groove (SKIP_GROOVE=1)"
else
    if [ -f output/rhythm_ai/groove_metadata/drums_index.pkl ] && [ -d output/rhythm_ai/groove_cleaned ]; then
        log "📂 2. Processing groove..."
        
        python "$EXTRACTOR_PATH" \
          --lamda-index output/rhythm_ai/groove_metadata/drums_index.pkl \
          --input-dir output/rhythm_ai/groove_cleaned \
          --output-dir output/rhythm_ai/groove_stage2 \
          --config "$CONFIG_PATH" \
          --verbose \
          2>&1 | tee -a "${LOG_FILE}"
        
        if [ $? -eq 0 ]; then
            log "✅ groove Stage2 completed"
        else
            log "❌ groove Stage2 failed"
            exit 1
        fi
    else
        log "ℹ️  groove inputs not found; skipping"
        export SKIP_GROOVE=1
    fi
fi

log ""

# ========== 統合 ==========
log "========================================="
log "Merging Stage2 results..."
log "========================================="

python -c "
import pandas as pd
from pathlib import Path
import json

# 読み込み
df1 = pd.read_parquet('output/rhythm_ai/drumclean_stage2/rhythm_features.parquet')

try:
    df2 = pd.read_parquet('output/rhythm_ai/groove_stage2/rhythm_features.parquet')
except Exception:
    import pandas as _pd
    df2 = _pd.DataFrame(columns=df1.columns)
    print('⚠️  groove parquet not found, using drumclean only')

print(f'drumclean: {len(df1)} records')
print(f'groove:    {len(df2)} records')

# 統合
df_merged = pd.concat([df1, df2], ignore_index=True)

# 重複削除
df_merged = df_merged.drop_duplicates(subset=['loop_id'], keep='first')

print(f'Merged:    {len(df_merged)} records (after dedup)')

# 保存
output_path = Path('output/rhythm_ai/rhythm_features_merged.parquet')
output_path.parent.mkdir(parents=True, exist_ok=True)
df_merged.to_parquet(str(output_path), compression='snappy', index=False)

print(f'✅ Saved: {output_path}')

# 統計
stats = {
    'total_records': len(df_merged),
    'sources': {
        'drumclean': len(df1),
        'groove': len(df2)
    },
    'tempo': {
        'mean': float(df_merged['tempo_bpm'].mean()),
        'std': float(df_merged['tempo_bpm'].std()),
        'min': float(df_merged['tempo_bpm'].min()),
        'max': float(df_merged['tempo_bpm'].max())
    },
    'groove': {
        'swing_mean': float(df_merged['swing_pct'].mean()),
        'backbeat_mean': float(df_merged['backbeat_strength'].mean())
    },
    'kpis': {
        'kick_downbeat': float(df_merged['kick_downbeat_rate'].mean()),
        'snare_backbeat': float(df_merged['snare_backbeat_rate'].mean()),
        'hat_density': float(df_merged['hat_density'].mean())
    },
    'families': df_merged['family_label'].value_counts().to_dict()
}

with open('output/rhythm_ai/rhythm_stage2_merged_summary.json', 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f'✅ Stats saved: output/rhythm_ai/rhythm_stage2_merged_summary.json')
" 2>&1 | tee -a "${LOG_FILE}"

if [ $? -eq 0 ]; then
    log "✅ Merge completed"
else
    log "❌ Merge failed"
    exit 1
fi

log ""

# ========== サマリー表示 ==========
log "========================================="
log "Summary"
log "========================================="

python -c "
import pandas as pd
import json

df = pd.read_parquet('output/rhythm_ai/rhythm_features_merged.parquet')

print(f'')
print(f'📊 Total Records: {len(df)}')
print(f'')
print(f'🎵 Tempo:')
print(f'  Mean: {df[\"tempo_bpm\"].mean():.1f} BPM')
print(f'  Std:  {df[\"tempo_bpm\"].std():.1f} BPM')
print(f'  Range: {df[\"tempo_bpm\"].min():.0f} - {df[\"tempo_bpm\"].max():.0f} BPM')
print(f'')
print(f'🎸 Groove:')
print(f'  Swing:     {df[\"swing_pct\"].mean():.1f}%')
print(f'  Backbeat:  {df[\"backbeat_strength\"].mean():.3f}')
print(f'')
print(f'✅ KPIs:')
print(f'  Kick Downbeat:   {df[\"kick_downbeat_rate\"].mean():.3f}')
print(f'  Snare Backbeat:  {df[\"snare_backbeat_rate\"].mean():.3f}')
print(f'  Hat Density:     {df[\"hat_density\"].mean():.1f} notes/bar')
print(f'')
print(f'🏷️  Family Distribution:')
for family, count in df['family_label'].value_counts().items():
    pct = count / len(df) * 100
    print(f'  {family:15s}: {count:6d} ({pct:5.1f}%)')
print(f'')
" 2>&1 | tee -a "${LOG_FILE}"

log ""
log "========================================="
log "✅ All processing completed!"
log "========================================="
log ""

# ========== 自動学習 ==========
log "🚀 Training ML model (auto)..."

python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import pandas as pd
import pickle
import json
from pathlib import Path

# データロード
df = pd.read_parquet('output/rhythm_ai/rhythm_features_merged.parquet')
print(f'📂 Loaded: {len(df)} records')

# ターゲット列
target_col = 'family_label' if 'family_label' in df.columns else 'label'

# 特徴量抽出（数値列のみ）
ignore_cols = {target_col, 'loop_id'}
feats = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

X = df[feats].fillna(0.0).values
y = df[target_col].astype(str).values

print(f'📊 Features: {len(feats)}')
print(f'🎯 Target: {target_col} ({len(pd.unique(y))} classes)')

# 学習（XGBoost → LogisticRegression fallback）
try:
    from xgboost import XGBClassifier
    
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
    print(f'✅ Training completed: XGBoost')
    
except Exception as e:
    print(f'⚠️  XGBoost unavailable ({e}), fallback to LogisticRegression')
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(max_iter=4000, random_state=42)
    model.fit(X, y)
    meta = {"algo": "logreg"}
    print(f'✅ Training completed: LogisticRegression')

# Pickle保存
out_dir = Path('data/patterns')
out_dir.mkdir(parents=True, exist_ok=True)

pkg = {
    "schema_version": "stage2_drums_v1",
    "model_meta": meta,
    "model": model,
    "class_labels": sorted(pd.unique(y).tolist()),
    "feature_names": feats,
    "target_col": target_col
}

out_path = out_dir / 'stage2_drums_rhythm_ai.pickle'
with open(out_path, 'wb') as f:
    pickle.dump(pkg, f)

print(f'💾 Saved: {out_path}')
print(f'   Classes: {len(pkg["class_labels"])}')
print(f'   Features: {len(pkg["feature_names"])}')
PY

if [ $? -eq 0 ]; then
    log "✅ ML training completed"
else
    log "❌ ML training failed"
    exit 1
fi

log ""
log "========================================="
log "🎉 Rhythm AI Stage2 Pipeline Completed!"
log "========================================="
log ""
log "📊 Generated Files:"
log "  - output/rhythm_ai/rhythm_features_merged.parquet"
log "  - data/patterns/stage2_drums_rhythm_ai.pickle"
log ""
log "🚀 Next Steps (optional):"
log "  1. Run latency benchmark:"
log "     python scripts/benchmark_ml_latency.py \\"
log "       --instrument drums \\"
log "       --pickle data/patterns/stage2_drums_rhythm_ai.pickle \\"
log "       --batch-mode"
log ""
log "  2. Verify KPI metrics:"
log "     python scripts/test_drums_v3_integration.py \\"
log "       --midi_dir outputs/demo_10songs \\"
log "       --kpi_yaml gate_prod.yaml"
log ""
