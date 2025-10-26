#!/bin/bash
# VSCode拡張機能の重複プロセスをクリーンアップ
# 各拡張機能タイプごとに最新の1-2プロセスのみ残して古いものを停止

echo "========================================"
echo "VSCode拡張機能プロセスクリーンアップ"
echo "========================================"

# プロセス数確認
TOTAL=$(ps aux | grep "python.*lsp" | grep -v grep | wc -l | tr -d ' ')
echo "現在のPython LSPプロセス数: $TOTAL"

if [ "$TOTAL" -lt 20 ]; then
    echo "✅ プロセス数は正常範囲内です（20未満）"
    exit 0
fi

echo ""
echo "⚠️  プロセス数が多すぎます。古いプロセスをクリーンアップします..."
echo ""

# 各拡張機能タイプごとに古いプロセスを停止（最新2つを残す）
EXTENSIONS=(
    "ms-python.isort.*lsp_server"
    "ms-python.isort.*lsp_runner"
    "ms-python.flake8.*lsp_server"
    "ms-python.flake8.*lsp_runner"
    "ms-python.black-formatter.*lsp_server"
    "ms-python.black-formatter.*lsp_runner"
    "ms-python.pylint.*lsp_server"
    "ms-python.pylint.*lsp_runner"
    "ms-python.autopep8.*lsp_server"
    "ms-python.mypy-type-checker.*lsp_server"
    "eeyore.yapf.*lsp_server"
)

for ext in "${EXTENSIONS[@]}"; do
    # 該当プロセスのPIDを取得して時刻でソート、古いものを停止（最新2つを残す）
    ALL_PIDS=$(ps aux | grep "python" | grep -E "$ext" | grep -v grep | awk '{print $2}')
    
    if [ -n "$ALL_PIDS" ]; then
        TOTAL_COUNT=$(echo "$ALL_PIDS" | wc -l | tr -d ' ')
        if [ "$TOTAL_COUNT" -gt 2 ]; then
            # 最新2つを除いて停止
            KEEP_COUNT=2
            KILL_COUNT=$(($TOTAL_COUNT - $KEEP_COUNT))
            PIDS_TO_KILL=$(echo "$ALL_PIDS" | head -n $KILL_COUNT)
            echo "🔄 $ext: $KILL_COUNT 個の古いプロセスを停止..."
            echo "$PIDS_TO_KILL" | xargs kill -9 2>/dev/null
        fi
    fi
done

# pytest収集プロセスも古いものをクリーンアップ（最新1つを残す）
ALL_PYTEST=$(ps aux | grep "pytest.*collect-only" | grep -v grep | awk '{print $2}')
if [ -n "$ALL_PYTEST" ]; then
    PYTEST_COUNT=$(echo "$ALL_PYTEST" | wc -l | tr -d ' ')
    if [ "$PYTEST_COUNT" -gt 1 ]; then
        KILL_PYTEST=$(($PYTEST_COUNT - 1))
        PYTEST_PIDS=$(echo "$ALL_PYTEST" | head -n $KILL_PYTEST)
        if [ -n "$PYTEST_PIDS" ]; then
            echo "🔄 pytest収集プロセス: $KILL_PYTEST 個の古いプロセスを停止..."
            echo "$PYTEST_PIDS" | xargs kill -9 2>/dev/null
        fi
    fi
fi

echo ""
echo "✅ クリーンアップ完了"
echo ""

# 最終確認
TOTAL_AFTER=$(ps aux | grep "python.*lsp" | grep -v grep | wc -l | tr -d ' ')
echo "クリーンアップ後のPython LSPプロセス数: $TOTAL_AFTER"
echo "削減数: $(($TOTAL - $TOTAL_AFTER))"
echo ""
echo "========================================"
