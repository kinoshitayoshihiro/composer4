#!/bin/bash
# clean_midi.py のpickle直書き運用検証スクリプト

set -e

echo "========================================"
echo "clean_midi.py Pickle直書き運用 検証"
echo "========================================"
echo ""

# 仮想環境アクティベート
source .venv311/bin/activate

# テスト用ディレクトリ
TEST_IN="data/lamda/raw/piano_test"
TEST_OUT="data/lamda/clean/piano_test"
TEST_QUARANTINE="data/lamda/quarantine/piano_test"
TEST_PICKLE="data/lamda/shards/piano_test"

echo "📋 ステップ1: テスト環境準備"
echo "----------------------------------------"

# 既存のテスト出力を削除
rm -rf "$TEST_OUT" "$TEST_QUARANTINE" "$TEST_PICKLE"

# テスト入力ディレクトリが存在するか確認
if [ ! -d "$TEST_IN" ]; then
    echo "⚠️  テスト入力ディレクトリが見つかりません: $TEST_IN"
    echo "   代わりに最初の10ファイルをコピーして使用します"
    
    mkdir -p "$TEST_IN"
    
    # 既存のMIDIファイルから最初の10個をコピー
    find data/lamda/raw/piano -name "*.mid" -o -name "*.midi" 2>/dev/null | head -10 | while read f; do
        cp "$f" "$TEST_IN/"
    done
fi

# テストファイル数確認
TEST_FILE_COUNT=$(find "$TEST_IN" -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ テストファイル数: $TEST_FILE_COUNT"
echo ""

if [ "$TEST_FILE_COUNT" -eq 0 ]; then
    echo "❌ テストファイルがありません。終了します。"
    exit 1
fi

echo "📋 ステップ2: 初回実行（pickle直書き）"
echo "----------------------------------------"

python -m scripts.clean_midi \
  --in "$TEST_IN" \
  --out "$TEST_OUT" \
  --quarantine "$TEST_QUARANTINE" \
  --instrument piano \
  --pickle-out "$TEST_PICKLE" \
  --shard-size 5000 \
  --resume \
  --emit-meta-json off \
  --jobs 4

echo ""
echo "📋 ステップ3: 結果検証"
echo "----------------------------------------"

# 1. pickle が作成されている
PICKLE_COUNT=$(find "$TEST_PICKLE" -name "*.pkl" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Pickleファイル数: $PICKLE_COUNT"

if [ "$PICKLE_COUNT" -eq 0 ]; then
    echo "❌ Pickleファイルが作成されていません！"
    exit 1
fi

# 2. .meta.json が成功ファイルに出力されていない
META_JSON_COUNT=$(find "$TEST_OUT" -name "*.meta.json" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ 成功ファイルの.meta.json数: $META_JSON_COUNT (0であるべき)"

if [ "$META_JSON_COUNT" -ne 0 ]; then
    echo "⚠️  .meta.jsonが出力されています（--emit-meta-json off のはず）"
fi

# 3. クリーニング済みファイル数
CLEANED_COUNT=$(find "$TEST_OUT" -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ クリーニング済みファイル数: $CLEANED_COUNT"

# 4. 隔離ファイル数
QUARANTINE_COUNT=$(find "$TEST_QUARANTINE" -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ 隔離ファイル数: $QUARANTINE_COUNT"

# 5. 合計
TOTAL=$((CLEANED_COUNT + QUARANTINE_COUNT))
echo "✅ 合計処理数: $TOTAL / $TEST_FILE_COUNT"

echo ""
echo "📋 ステップ4: 2回目実行（スキップ＆shard追加テスト）"
echo "----------------------------------------"

# 2回目実行（すべてスキップされるはず）
python -m scripts.clean_midi \
  --in "$TEST_IN" \
  --out "$TEST_OUT" \
  --quarantine "$TEST_QUARANTINE" \
  --instrument piano \
  --pickle-out "$TEST_PICKLE" \
  --shard-size 5000 \
  --resume \
  --emit-meta-json off \
  --jobs 4

echo ""
echo "📋 ステップ5: 再実行後の検証"
echo "----------------------------------------"

# pickleファイル数が変わっていないこと
PICKLE_COUNT_2=$(find "$TEST_PICKLE" -name "*.pkl" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Pickleファイル数（2回目）: $PICKLE_COUNT_2"

if [ "$PICKLE_COUNT" -ne "$PICKLE_COUNT_2" ]; then
    echo "⚠️  Pickleファイル数が変わっています（変わらないはず）"
fi

# クリーニング済みファイル数が変わっていないこと
CLEANED_COUNT_2=$(find "$TEST_OUT" -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l | tr -d ' ')
echo "✅ クリーニング済みファイル数（2回目）: $CLEANED_COUNT_2"

if [ "$CLEANED_COUNT" -ne "$CLEANED_COUNT_2" ]; then
    echo "❌ クリーニング済みファイル数が変わっています！"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ 検証完了"
echo "========================================"
echo ""
echo "📊 サマリー:"
echo "  - テストファイル数: $TEST_FILE_COUNT"
echo "  - クリーニング成功: $CLEANED_COUNT"
echo "  - 隔離: $QUARANTINE_COUNT"
echo "  - Pickle数: $PICKLE_COUNT"
echo "  - .meta.json数（成功）: $META_JSON_COUNT (0が正解)"
echo ""
echo "🎯 結論:"

if [ "$META_JSON_COUNT" -eq 0 ] && [ "$PICKLE_COUNT" -gt 0 ] && [ "$CLEANED_COUNT_2" -eq "$CLEANED_COUNT" ]; then
    echo "  ✅ Pickle直書き運用が正しく動作しています！"
    echo ""
    echo "次のステップ:"
    echo "  1. 本番ディレクトリで実行"
    echo "  2. レガシースクリプトを非推奨化"
    echo "  3. ドキュメント更新"
    exit 0
else
    echo "  ⚠️  一部の検証が失敗しました。詳細を確認してください。"
    exit 1
fi
