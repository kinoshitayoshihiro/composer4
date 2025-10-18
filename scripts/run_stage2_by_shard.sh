#!/bin/bash
# Stage2をシャード単位で実行(メモリ不足対策)

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "$BASE_DIR"

META_INDEX="output/drums_metadata/drums_index.pkl"
META_DIR="output/drums_metadata"
INPUT_DIR="output/drumloops_v3"
OUT_DIR="output/drumloops_v3_stage2"
CFG="configs/lamda/drums_stage2.yaml"
THRESHOLD=70.0

# 出力ディレクトリを作成
mkdir -p "$OUT_DIR"
mkdir -p "logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage2 Processing by Shard"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================="

# シャード数を取得
TOTAL_SHARDS=$(python3 -c "
import pickle
idx = pickle.load(open('$META_INDEX', 'rb'))
print(len(idx['shards']))
")

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total shards: $TOTAL_SHARDS"

# 各シャードを処理
for shard_idx in $(seq 0 $((TOTAL_SHARDS - 1))); do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing shard $shard_idx / $((TOTAL_SHARDS - 1))"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================="
    
    LOG_FILE="logs/stage2_shard_${shard_idx}_$(date +%Y%m%d_%H%M%S).log"
    
    # シャード単位で処理するPythonスクリプトを実行
    python3 scripts/run_stage2_single_shard.py \
        --metadata-index "$META_INDEX" \
        --metadata-dir "$META_DIR" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUT_DIR" \
        --config "$CFG" \
        --threshold "$THRESHOLD" \
        --shard-index "$shard_idx" \
        2>&1 | tee "$LOG_FILE"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Shard $shard_idx completed"
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All shards processed!"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========================================="

# 統計情報を生成
python3 scripts/merge_stage2_results.py \
    --input-dir "$OUT_DIR" \
    --output-summary "$OUT_DIR/stage2_summary.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Stage2 completed successfully!"
