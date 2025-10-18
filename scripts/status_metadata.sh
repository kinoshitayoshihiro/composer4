#!/bin/bash
# メタデータ生成進捗確認

echo "🎵 Metadata Generation Progress"
echo "================================"
echo ""

# プロセス確認
if ps aux | grep "build_drumloops_metadata.py" | grep -v grep > /dev/null; then
    echo "✅ Status: RUNNING"
    PID=$(ps aux | grep "build_drumloops_metadata.py" | grep -v grep | awk '{print $2}')
    echo "   PID: ${PID}"
else
    echo "⏹️  Status: NOT RUNNING"
fi

echo ""

# ファイル確認
if [ -f output/drumloops_v3_metadata/drumloops_v3_metadata.pickle ]; then
    SIZE=$(ls -lh output/drumloops_v3_metadata/drumloops_v3_metadata.pickle | awk '{print $5}')
    echo "📦 Pickle file size: ${SIZE}"
fi

echo ""
echo "Expected: 51,248 files"
echo "Estimated time: ~17 minutes"
