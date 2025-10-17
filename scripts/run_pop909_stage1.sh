#!/bin/bash
# POP909 Stage1: Stem分離版のみを処理
# v1 (melody) → piano, v2 (chords) → piano, v3 (bass) → bass

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Python実行環境
PYTHON="${PROJECT_ROOT}/.venv311/bin/python"
CLEAN_SCRIPT="${PROJECT_ROOT}/scripts/clean_midi.py"

# デフォルト設定
SHARD_SIZE="${SHARD_SIZE:-2000}"
JOBS="${JOBS:-1}"  # POP909は小規模なのでjobs=1推奨
EMIT_META_JSON="${EMIT_META_JSON:-off}"

echo "======================================================================"
echo "🎹 POP909 Stage1: Stem-Separated Processing"
echo "======================================================================"
echo "Strategy: Use v1+v2+v3 complete stems only (279 songs × 3 = 837 files)"
echo "  v1 (melody)  → piano"
echo "  v2 (chords)  → piano"
echo "  v3 (bass)    → bass"
echo ""
echo "Configuration:"
echo "  SHARD_SIZE:      $SHARD_SIZE"
echo "  JOBS:            $JOBS"
echo "  EMIT_META_JSON:  $EMIT_META_JSON"
echo "======================================================================"
echo ""

# ファイルリストの存在確認
if [[ ! -f "lists/pop909_complete_v1.txt" ]]; then
    echo "❌ Error: File lists not found. Running analyzer first..."
    $PYTHON scripts/analyze_pop909_stems.py
fi

# 処理関数
process_pop909_stem() {
    local STEM_VERSION="$1"
    local INSTRUMENT="$2"
    local LABEL="$3"
    local FILE_LIST="lists/pop909_complete_${STEM_VERSION}.txt"
    
    echo ""
    echo "======================================================================"
    echo "Processing: POP909 ${STEM_VERSION} (${LABEL})"
    echo "======================================================================"
    
    if [[ ! -f "$FILE_LIST" ]]; then
        echo "⚠️  Warning: $FILE_LIST not found, skipping..."
        return 1
    fi
    
    local FILE_COUNT=$(wc -l < "$FILE_LIST")
    echo "Files: $FILE_COUNT"
    echo "Instrument: $INSTRUMENT"
    echo "Output: output/pop909/clean/${LABEL}"
    echo ""
    
    # 一時ディレクトリを作成してファイルをコピー
    local TEMP_DIR="output/pop909/temp/${LABEL}"
    mkdir -p "$TEMP_DIR"
    
    # ファイルリストからコピー
    cat "$FILE_LIST" | while read -r filepath; do
        if [[ -f "$filepath" ]]; then
            cp "$filepath" "$TEMP_DIR/"
        fi
    done
    
    # クリーニング実行
    $PYTHON "$CLEAN_SCRIPT" \
        --in "$TEMP_DIR" \
        --out "output/pop909/clean/${LABEL}" \
        --quarantine "output/pop909/quarantine/${LABEL}" \
        --instrument "$INSTRUMENT" \
        --pickle-out "output/pop909/shards/${LABEL}" \
        --shard-size "$SHARD_SIZE" \
        --emit-meta-json "$EMIT_META_JSON" \
        --jobs "$JOBS" \
        2>&1 | tee "logs/pop909_${LABEL}_stage1_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "✅ Completed: ${LABEL}"
    echo "   Clean: $(find output/pop909/clean/${LABEL} -name "*.mid" | wc -l) files"
    echo "   Quarantine: $(find output/pop909/quarantine/${LABEL} -name "*.mid" 2>/dev/null | wc -l) files"
}

# ログディレクトリ作成
mkdir -p logs

# v1: Melody (piano)
process_pop909_stem "v1" "piano" "melody"

# v2: Chords/Accompaniment (piano)
process_pop909_stem "v2" "piano" "chords"

# v3: Bass (bass)
process_pop909_stem "v3" "bass" "bass"

echo ""
echo "======================================================================"
echo "✅ POP909 Stage1 Complete"
echo "======================================================================"
echo "Summary:"
echo "  Melody (v1): $(find output/pop909/clean/melody -name "*.mid" 2>/dev/null | wc -l) / 279"
echo "  Chords (v2): $(find output/pop909/clean/chords -name "*.mid" 2>/dev/null | wc -l) / 279"
echo "  Bass (v3):   $(find output/pop909/clean/bass -name "*.mid" 2>/dev/null | wc -l) / 279"
echo ""
echo "📦 Pickle shards:"
echo "  - output/pop909/shards/melody/"
echo "  - output/pop909/shards/chords/"
echo "  - output/pop909/shards/bass/"
echo "======================================================================"
