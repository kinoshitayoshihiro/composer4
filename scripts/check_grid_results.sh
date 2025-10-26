#!/bin/bash
# Grid Search 結果確認用スクリプト

cd "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"

echo "=== Grid Search 進捗確認 ==="
echo ""

# 実行中のプロセス確認
if pgrep -f "ab_test_guitar_v3.py" > /dev/null; then
  echo "✓ Grid Search 実行中..."
  echo ""
  echo "最新ログ (末尾30行):"
  tail -30 grid_search_final.log 2>/dev/null
else
  echo "✓ Grid Search 完了"
  echo ""
  
  # サマリーCSVの確認
  if [ -f "data/ab_v3_grid_summary.csv" ]; then
    echo "=== Grid Search Summary ==="
    python - << 'PY'
import pandas as pd
df = pd.read_csv("data/ab_v3_grid_summary.csv")
print(df.sort_values("accent_delta%", ascending=False).to_string(index=False))
PY
    echo ""
    
    # KPIゲート通過確認
    echo "=== KPI Gate Analysis ==="
    python - << 'PY'
import pandas as pd
df = pd.read_csv("data/ab_v3_grid_summary.csv")
passed = df[
    (df["ml_usage%"] >= 70.0) &
    (df["family_match%"] >= 80.0) &
    (df["density_abs"] <= 1.0)
]
if len(passed) == 0:
    print("⚠️  No config meets all KPIs. Best by accent_delta%:")
    best = df.loc[df["accent_delta%"].idxmax()]
    print(best.to_string())
else:
    print(f"✓ {len(passed)} configs meet KPIs. Best by accent_delta%:")
    best = passed.loc[passed["accent_delta%"].idxmax()]
    print(best.to_string())
PY
    echo ""
    
    # ab_v3_best.yaml の確認
    if [ -f "data/ab_v3_best.yaml" ]; then
      echo "=== Best Config (ab_v3_best.yaml) ==="
      cat data/ab_v3_best.yaml
    else
      echo "⚠️  ab_v3_best.yaml not found"
    fi
  else
    echo "⚠️  ab_v3_grid_summary.csv not found. Grid search may still be running."
  fi
fi
