#!/bin/bash
# 高速テスト用スクリプト - 先頭100ファイルのみで動作確認

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "$BASE_DIR"

echo "🧪 Drumloops Quick Test - 先頭100ファイル"
echo "========================================"

# 仮想環境
source .venv311/bin/activate

# 出力ディレクトリ準備
TEST_IN="data/loops_test_100"
TEST_OUT="output/test_drums_100"
TEST_Q="output/test_drums_100_q"
TEST_PKL="output/test_drums_pkl"

# クリーンアップ
rm -rf "$TEST_IN" "$TEST_OUT" "$TEST_Q" "$TEST_PKL"
mkdir -p "$TEST_IN"

# 先頭100ファイルをコピー
echo "📋 先頭100ファイルをコピー中..."
find data/loops -name "*.mid" -type f | head -100 | while read f; do
    cp "$f" "$TEST_IN/"
done

COPIED=$(find "$TEST_IN" -name "*.mid" | wc -l | tr -d ' ')
echo "✅ コピー完了: ${COPIED}ファイル"

# クリーニング実行
echo ""
echo "🎵 クリーニング開始..."
python -m scripts.clean_midi \
  --in "$TEST_IN" \
  --out "$TEST_OUT" \
  --quarantine "$TEST_Q" \
  --instrument drums \
  --pickle-out "$TEST_PKL" \
  --shard-size 50 \
  --emit-meta-json off \
  --jobs 4

echo ""
echo "📊 結果:"
echo "  入力:           ${COPIED}ファイル"
echo "  クリーニング済: $(find "$TEST_OUT" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')ファイル"
echo "  隔離:           $(find "$TEST_Q" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')ファイル"
echo "  Pickle shards:  $(find "$TEST_PKL" -name "drums_*.pkl" 2>/dev/null | wc -l | tr -d ' ')個"

# 互換性チェック
echo ""
echo "🔍 Stage2互換性チェック..."
if [ -f "$TEST_PKL/drums_index.pkl" ]; then
    python verify_stage2_compat.py "$TEST_PKL"
else
    echo "❌ インデックスファイルが見つかりません"
fi

echo ""
echo "✅ テスト完了！"
