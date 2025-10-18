#!/bin/bash
# Drumloops本番実行の進捗確認スクリプト

echo "🥁 Drumloops クリーニング進捗状況"
echo "=================================="
echo ""

# プロセス確認
PID=$(ps aux | grep "python -m scripts.clean_midi" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✅ プロセス実行中 (PID: $PID)"
else
    echo "❌ プロセスが見つかりません"
fi
echo ""

# ファイル数確認
TOTAL=77346
CLEANED=$(find output/drumloops_v3 -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
QUARANTINED=$(find output/drumloops_v3_q -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
PROCESSED=$((CLEANED + QUARANTINED))

echo "📊 処理状況:"
echo "  ✅ クリーニング済み: ${CLEANED}"
echo "  🗑️  隔離:           ${QUARANTINED}"
echo "  📦 処理済み合計:    ${PROCESSED} / ${TOTAL}"
if [ "$PROCESSED" -gt 0 ]; then
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($PROCESSED/$TOTAL)*100}")
    echo "  📈 進捗率:          ${PERCENT}%"
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($CLEANED/$PROCESSED)*100}")
    echo "  ✨ 成功率:          ${SUCCESS_RATE}%"
fi
echo ""

# Pickle状況
echo "📦 Pickle生成状況:"
if [ -d "output/drums_metadata" ]; then
    SHARDS=$(find output/drums_metadata -name "drums_*.pkl" -not -name "*_index.pkl" 2>/dev/null | wc -l | tr -d ' ')
    if [ -f "output/drums_metadata/drums_index.pkl" ]; then
        echo "  ✅ Index: drums_index.pkl 存在"
    else
        echo "  ⏳ Index: まだ生成されていません"
    fi
    echo "  📦 Shards: ${SHARDS}個"
    if [ "$SHARDS" -gt 0 ]; then
        EXPECTED_SHARDS=$((CLEANED / 5000 + 1))
        echo "  📊 期待値: 約${EXPECTED_SHARDS}個 (cleaned=${CLEANED}, shard_size=5000)"
    fi
else
    echo "  ⏳ まだ生成されていません"
fi
echo ""

# 最新ログ
LATEST_LOG=$(ls -t logs/drumloops_cleaning_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "📄 最新ログ: $LATEST_LOG"
    echo "  最終行:"
    tail -3 "$LATEST_LOG" | sed 's/^/    /'
else
    echo "📄 ログファイルが見つかりません"
fi
echo ""

# 推定残り時間（処理速度から計算）
if [ -f "$LATEST_LOG" ]; then
    # ログから処理速度を抽出（例: "11.61it/s"）
    SPEED=$(tail -20 "$LATEST_LOG" | grep -o '[0-9.]*it/s' | tail -1 | sed 's/it\/s//')
    if [ -n "$SPEED" ]; then
        REMAINING=$((TOTAL - PROCESSED))
        REMAINING_SEC=$(awk "BEGIN {printf \"%.0f\", $REMAINING/$SPEED}")
        REMAINING_MIN=$((REMAINING_SEC / 60))
        REMAINING_HOUR=$((REMAINING_MIN / 60))
        REMAINING_MIN=$((REMAINING_MIN % 60))
        echo "⏱️  推定残り時間: ${REMAINING_HOUR}時間${REMAINING_MIN}分 (速度: ${SPEED}ファイル/秒)"
    fi
fi
echo ""

echo "コマンド:"
echo "  リアルタイム監視: watch -n 10 ./scripts/check_drumloops_progress.sh"
echo "  ログ確認:         tail -f $LATEST_LOG"
echo "  プロセス停止:     kill $PID"
