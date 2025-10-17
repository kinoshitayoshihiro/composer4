#!/bin/bash
# Stage2バッチ処理スクリプト
# 全51,248ループを11バッチに分けて処理

set -e  # エラーで停止

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN=".venv311/bin/python"
SCRIPT="scripts/lamda_stage2_extractor.py"
METADATA_INDEX="output/drums_metadata/drums_index.pkl"
METADATA_DIR="output/drums_metadata"
INPUT_DIR="output/drumloops_v3"
CONFIG="configs/lamda/drums_stage2.yaml"
THRESHOLD_SOFT="${THRESHOLD_SOFT:-65.0}"
THRESHOLD_HARD="${THRESHOLD_HARD:-70.0}"
BATCH_SIZE=5000
STREAMING="${STREAMING:-true}"
RESUME="${RESUME:-true}"

echo "========================================="
echo "Stage2 バッチ処理開始"
echo "========================================="
echo "開始時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo "バッチサイズ: $BATCH_SIZE"
echo "しきい値 (soft): $THRESHOLD_SOFT"
echo "しきい値 (hard): $THRESHOLD_HARD"
echo "ストリーミング: $STREAMING"
echo "冪等再開: $RESUME"
echo "予想バッチ数: 11"
echo "========================================="
echo ""

# バッチ処理ループ
for i in {0..10}; do
  offset=$((i * BATCH_SIZE))
  output_dir="output/drumloops_v3_stage2_batch${i}"
  
  echo "========================================="
  echo "バッチ $((i+1))/11 開始"
  echo "========================================="
  echo "Offset: $offset"
  echo "Limit: $BATCH_SIZE"
  echo "出力先: $output_dir"
  echo "開始時刻: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
  
  # ストリーミングと冪等化のオプション構築
  EXTRA_OPTS=""
  if [ "$STREAMING" = "true" ]; then
    EXTRA_OPTS="$EXTRA_OPTS --streaming --parquet-row-group 8192"
  fi
  if [ "$RESUME" = "true" ]; then
    EXTRA_OPTS="$EXTRA_OPTS --resume --manifest-flush-n 200"
  fi
  
  # バッチ実行
  PYTHONPATH=. $PYTHON_BIN $SCRIPT \
    --metadata-index "$METADATA_INDEX" \
    --metadata-dir "$METADATA_DIR" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$output_dir" \
    --config "$CONFIG" \
    --threshold-soft $THRESHOLD_SOFT \
    --threshold-hard $THRESHOLD_HARD \
    --offset $offset \
    --limit $BATCH_SIZE \
    $EXTRA_OPTS \
    --print-summary
  
  echo ""
  echo "バッチ $((i+1))/11 完了"
  echo "完了時刻: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================="
  echo ""
  
  # 短い休憩
  sleep 2
done

echo "========================================="
echo "全バッチ処理完了!"
echo "========================================="
echo "完了時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "出力ディレクトリ:"
ls -d output/drumloops_v3_stage2_batch* 2>/dev/null || echo "なし"
echo ""
echo "次のステップ: バッチ結果をマージしてください"
echo "  python3 scripts/merge_stage2_batches.py"
echo "========================================="
