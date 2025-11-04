#!/usr/bin/env bash
################################################################################
# batch_validate_kpi.sh
# 
# 複数曲に対してKPI Gate検証を一括実行し、Pass率を集計
# 
# Usage:
#   ./scripts/batch_validate_kpi.sh [--song-packages-dir DIR] [--output REPORT]
#
# Options:
#   --song-packages-dir DIR  : song_packagesディレクトリパス（デフォルト: ./song_packages）
#   --output REPORT          : 集計レポート出力先（デフォルト: ./kpi_batch_summary.json）
#   --gate-config CONFIG     : gate config YAMLパス（デフォルト: ./configs/gate_prod.yaml）
#   --downbeats              : Downbeats準拠小節切りを有効化（デフォルト: true）
#   --parallel N             : 並列実行数（デフォルト: 1、シーケンシャル実行）
#
# Phase E: Production Scale-out
################################################################################

set -Eeuo pipefail

# デフォルト設定
SONG_PACKAGES_DIR="./song_packages"
OUTPUT_REPORT="./kpi_batch_summary.json"
GATE_CONFIG="./configs/gate_prod.yaml"
USE_DOWNBEATS="--downbeats"
PARALLEL=1

# 引数パース
while [[ $# -gt 0 ]]; do
    case "$1" in
        --song-packages-dir)
            SONG_PACKAGES_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_REPORT="$2"
            shift 2
            ;;
        --gate-config)
            GATE_CONFIG="$2"
            shift 2
            ;;
        --no-downbeats)
            USE_DOWNBEATS=""
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# 存在確認
if [[ ! -d "$SONG_PACKAGES_DIR" ]]; then
    echo "❌ Error: Song packages directory not found: $SONG_PACKAGES_DIR" >&2
    exit 1
fi

if [[ ! -f "$GATE_CONFIG" ]]; then
    echo "❌ Error: Gate config not found: $GATE_CONFIG" >&2
    exit 1
fi

# kpi_gate_enhanced.py 存在確認
KPI_GATE_SCRIPT="./scripts/kpi_gate_enhanced.py"
if [[ ! -f "$KPI_GATE_SCRIPT" ]]; then
    echo "❌ Error: kpi_gate_enhanced.py not found: $KPI_GATE_SCRIPT" >&2
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 KPI Gate Batch Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Song packages: $SONG_PACKAGES_DIR"
echo "Gate config: $GATE_CONFIG"
echo "Output report (JSON base): $OUTPUT_REPORT"
echo "Downbeats: ${USE_DOWNBEATS:-disabled}"
echo "Parallel: $PARALLEL"
echo ""

# song_packages配下のsong_*ディレクトリを検索
SONG_DIRS=()
while IFS= read -r -d '' song_dir; do
    SONG_DIRS+=("$song_dir")
done < <(find "$SONG_PACKAGES_DIR" -type d -name "song_*" -print0 | sort -z)

if [[ ${#SONG_DIRS[@]} -eq 0 ]]; then
    echo "⚠️  Warning: No song_* directories found in $SONG_PACKAGES_DIR" >&2
    exit 0
fi

echo "📊 Found ${#SONG_DIRS[@]} songs"
echo ""

# 依存コマンド確認（jq は任意）
HAVE_JQ=1
command -v jq >/dev/null 2>&1 || HAVE_JQ=0
if [[ $HAVE_JQ -eq 0 ]]; then
  echo "ℹ️  jq not found: JSON集計は Python フォールバックで実行します"
fi

# 並列実行関数（GNU parallel未使用、シンプルなバックグラウンド実行）
process_song() {
    local song_dir="$1"
    local song_name
    song_name=$(basename "$song_dir")
    
    # 必要ファイル存在確認
    local drums_midi="$song_dir/drums.mid"
    local bars_parquet="$song_dir/bars.parquet"
    local song_package_yaml="$song_dir/song_package.yaml"
    
    if [[ ! -f "$drums_midi" ]]; then
        echo "  ⚠️  $song_name: drums.mid not found, skipping" >&2
        return 1
    fi
    
    if [[ ! -f "$bars_parquet" ]]; then
        echo "  ⚠️  $song_name: bars.parquet not found, skipping" >&2
        return 1
    fi
    
    if [[ ! -f "$song_package_yaml" ]]; then
        echo "  ⚠️  $song_name: song_package.yaml not found, skipping" >&2
        return 1
    fi
    
    # BPM抽出（yamlから）
    local tempo_bpm
    tempo_bpm=$(python3 -c "
import yaml
with open('$song_package_yaml', 'r') as f:
    pkg = yaml.safe_load(f)
    print(pkg.get('meta', {}).get('bpm', pkg.get('meta', {}).get('tempo_bpm', 120.0)))
" 2>/dev/null || echo "120.0")
    
    # KPI Gate検証実行
    local output_report="$song_dir/kpi_gate_report_enhanced.json"
    
    echo "  ▶️  $song_name: Running KPI Gate (BPM: $tempo_bpm)..."
    
    if python3 "$KPI_GATE_SCRIPT" \
        --midi "$drums_midi" \
        --bars "$bars_parquet" \
        --gate-config "$GATE_CONFIG" \
        ${USE_DOWNBEATS} \
        --tempo-bpm "$tempo_bpm" \
        --output "$output_report" \
        > "$song_dir/kpi_gate_enhanced.log" 2>&1; then
        
        # Pass/Fail集計（jq が無い場合は Python で代替）
        local pass_count fail_count total_bars
        if [[ $HAVE_JQ -eq 1 ]]; then
          pass_count=$(jq -r '.summary.total_pass // 0' "$output_report" 2>/dev/null || echo "0")
          fail_count=$(jq -r '.summary.total_fail // 0' "$output_report" 2>/dev/null || echo "0")
        else
          read -r pass_count fail_count < <(python3 - "$output_report" <<'PY'
import json,sys
try:
  d=json.load(open(sys.argv[1]));print(d["summary"].get("total_pass",0),d["summary"].get("total_fail",0))
except Exception: print("0 0")
PY
)
        fi
        total_bars=$((pass_count + fail_count))
        
        if [[ $total_bars -gt 0 ]]; then
            local pass_rate
            pass_rate=$(python3 -c "print(f'{100.0 * $pass_count / $total_bars:.1f}')")
            echo "  ✅ $song_name: Pass $pass_rate% ($pass_count/$total_bars bars)"
        else
            echo "  ⚠️  $song_name: No bars validated"
        fi
    else
        echo "  ❌ $song_name: KPI Gate failed (see $song_dir/kpi_gate_enhanced.log)" >&2
        return 1
    fi
}

export -f process_song
export KPI_GATE_SCRIPT
export GATE_CONFIG
export USE_DOWNBEATS

# バッチ処理実行
SUCCESS_COUNT=0
FAIL_COUNT=0

if [[ $PARALLEL -eq 1 ]]; then
    # シーケンシャル実行
    for song_dir in "${SONG_DIRS[@]}"; do
        if process_song "$song_dir"; then
            ((SUCCESS_COUNT++)) || true
        else
            ((FAIL_COUNT++)) || true
        fi
    done
else
    # 並列実行（簡易版、GNU parallel不要）
    echo "  ℹ️  Parallel execution (N=$PARALLEL) - running in background..."
    
    pids=()
    idx=0
    
    for song_dir in "${SONG_DIRS[@]}"; do
        process_song "$song_dir" &
        pids+=($!)
        
        ((idx++)) || true
        
        # N個同時実行まで
        if [[ $((idx % PARALLEL)) -eq 0 ]]; then
            # 完了待ち
            for pid in "${pids[@]}"; do
                if wait "$pid"; then
                    ((SUCCESS_COUNT++)) || true
                else
                    ((FAIL_COUNT++)) || true
                fi
            done
            pids=()
        fi
    done
    
    # 残り待ち
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            ((SUCCESS_COUNT++)) || true
        else
            ((FAIL_COUNT++)) || true
        fi
    done
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Batch Validation Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Success: $SUCCESS_COUNT songs"
echo "Failed:  $FAIL_COUNT songs"
echo ""

# ① JSONサマリー出力（曲別 Pass 率も格納）
python3 - "$OUTPUT_REPORT" <<'PY'
import json,glob,sys,os
out=sys.argv[1]
root=os.path.dirname(out) or "."
songs=sorted(glob.glob("**/kpi_gate_report_enhanced.json", recursive=True))
items=[]
for rep in songs:
  try:
    d=json.load(open(rep,"r"))
    sm=d.get("summary",{})
    total=sm.get("total_pass",0)+sm.get("total_fail",0)
    pass_rate=(sm.get("total_pass",0)/total) if total else 0.0
    items.append({
      "song_dir": os.path.dirname(rep),
      "total_bars": total,
      "pass_bars": sm.get("total_pass",0),
      "fail_bars": sm.get("total_fail",0),
      "pass_rate": round(pass_rate,4)
    })
  except Exception:
    pass
summary={
  "generated_at": __import__("datetime").datetime.utcnow().isoformat()+"Z",
  "songs": items,
  "totals": {
    "songs": len(items),
    "total_bars": sum(i["total_bars"] for i in items),
    "total_pass": sum(i["pass_bars"] for i in items),
    "total_fail": sum(i["fail_bars"] for i in items),
    "avg_pass_rate": round(sum(i["pass_rate"] for i in items)/len(items),4) if items else 0.0
  }
}
json.dump(summary, open(out,"w"), indent=2, ensure_ascii=False)
print(f"✅ JSON summary saved: {out}")
PY

# ② 集計レポート生成（aggregate_kpi_reports.py呼び出し）
if [[ -f "./scripts/aggregate_kpi_reports.py" ]]; then
    echo "📈 Generating aggregate report..."
    python3 ./scripts/aggregate_kpi_reports.py \
        --root "$SONG_PACKAGES_DIR" \
        --out-csv "${OUTPUT_REPORT%.json}.csv" \
        --out-md "${OUTPUT_REPORT%.json}.md"
    
    echo "✅ Aggregate report saved: $OUTPUT_REPORT (.csv, .md)"
else
    echo "⚠️  Warning: aggregate_kpi_reports.py not found, skipping aggregate report" >&2
fi

echo ""
echo "🎉 Batch validation complete!"
