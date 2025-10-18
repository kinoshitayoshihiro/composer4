#!/bin/bash
# Drumloops本番実行スクリプト

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "$BASE_DIR"

# 仮想環境
source .venv311/bin/activate

# ログファイル名
LOG_FILE="logs/drumloops_production_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Drumloops 本番クリーニング開始"
echo "=================================="
echo ""
echo "📊 設定:"
echo "  入力:     data/loops (77,346ファイル)"
echo "  出力:     output/drumloops_v3"
echo "  隔離:     output/drumloops_v3_q"
echo "  Pickle:   output/drums_metadata"
echo "  Shard:    5,000件"
echo "  並列:     1 job (安定動作優先)"
echo "  Resume:   有効"
echo "  ログ:     $LOG_FILE"
echo ""
echo "⏱️  推定実行時間: 約2時間"
echo ""

# 実行
python -m scripts.clean_midi \
  --in data/loops \
  --out output/drumloops_v3 \
  --quarantine output/drumloops_v3_q \
  --instrument drums \
  --pickle-out output/drums_metadata \
  --shard-size 5000 \
  --resume \
  --emit-meta-json off \
  --jobs 1 \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ 処理完了！"
echo ""
echo "📊 結果確認:"
echo "  ./scripts/check_drumloops_progress.sh"
echo ""
echo "🔍 Stage2互換性チェック:"
echo "  python verify_stage2_compat.py output/drums_metadata"
