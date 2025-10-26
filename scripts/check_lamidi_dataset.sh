#!/bin/bash
# ========================================
# Los-Angeles-MIDI データセット情報確認
# ========================================

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
MIDI_DIR="${BASE_DIR}/data/Los-Angeles-MIDI/MIDIs"

echo "📊 Los-Angeles-MIDI Dataset Information"
echo "================================================"
echo ""

if [ ! -d "${MIDI_DIR}" ]; then
    echo "❌ Error: ${MIDI_DIR} not found"
    exit 1
fi

echo "📁 Counting MIDI files (this may take a while)..."
MIDI_COUNT=$(find "${MIDI_DIR}" -type f \( -iname "*.mid" -o -iname "*.midi" \) 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "Results:"
echo "  Input Directory: ${MIDI_DIR}"
echo "  Total MIDI Files: ${MIDI_COUNT}"
echo ""

# 総数をファイルに保存（monitor_lamda.shが自動読込）
echo "${MIDI_COUNT}" > "${BASE_DIR}/data/lamda_expected_total.txt"
echo "  → Saved to data/lamda_expected_total.txt (for monitor)"
echo ""

if [ ${MIDI_COUNT} -eq 0 ]; then
    echo "⚠️  No MIDI files found!"
    echo ""
    echo "Directory structure:"
    ls -la "${MIDI_DIR}" 2>/dev/null | head -20
else
    echo "✅ Ready to clean!"
    echo ""
    echo "Recommended commands:"
    echo ""
    echo "  # ドライラン（コマンド確認のみ）"
    echo "  ./scripts/run_lamidi_full.sh --dry-run"
    echo ""
    echo "  # 実行"
    echo "  ./scripts/run_lamidi_full.sh"
    echo ""
    echo "  # 進捗モニター（別ターミナルで）"
    echo "  ./scripts/monitor_lamda.sh"
    echo ""
    echo "Note: Total file count (${MIDI_COUNT}) saved to data/lamda_expected_total.txt"
    echo "      monitor_lamda.sh will automatically use this value."
fi
