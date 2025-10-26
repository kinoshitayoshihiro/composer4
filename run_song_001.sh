#!/bin/bash
# ================================================================
# Song 001 "おれの女房" - Full Production Run
# ================================================================
# 新しいgenerator編成に対応:
#   - guitar_generator.py
#   - strings_generator.py
#   - bass_generator.py
#   - piano_generator.py
# ================================================================

set -e  # エラー時に即座に終了

SONG_DIR="data/suno_ai/suno_themesong/song_001"
ANALYSIS_DIR="${SONG_DIR}/analysis"
OUTPUT_DIR="output/song_001"

echo "🎵 OtoKotoba Engine - Song 001 Production Run"
echo "=============================================="
echo "Song: おれの女房 (Ore no Nyoubou)"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 必須ファイルチェック
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "📂 Step 1/4: Checking required files..."
MISSING=0

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(ls -lh "$1" | awk '{print $5}')
        echo "  ✅ $1 ($SIZE)"
        return 0
    else
        echo "  ❌ $1 - MISSING"
        MISSING=$((MISSING + 1))
        return 1
    fi
}

# 必須ファイル
check_file "config/main_cfg.yml"
check_file "${ANALYSIS_DIR}/chordmap.json"
check_file "${ANALYSIS_DIR}/sections.json"
check_file "${ANALYSIS_DIR}/tempo_map.json"
check_file "${ANALYSIS_DIR}/lyric_anchors.json"
check_file "data/rhythm_library.yml"

# Generator files
echo ""
echo "  Generator files:"
check_file "generator/guitar_generator.py"
check_file "generator/strings_generator.py"
check_file "generator/bass_generator.py"
check_file "generator/piano_generator.py"

echo ""
if [ $MISSING -gt 0 ]; then
    echo "❌ Missing $MISSING required files. Please prepare them first."
    exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: 出力ディレクトリ作成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "📁 Step 2/4: Preparing output directory..."
mkdir -p "${OUTPUT_DIR}"
echo "  ✅ Created ${OUTPUT_DIR}"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: 設定情報の表示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "⚙️  Step 3/4: Configuration Summary"
echo "─────────────────────────────────────────────"

# sections.jsonから情報抽出
TOTAL_BARS=$(jq -r '.sections_layout[-1].end_bar' "${ANALYSIS_DIR}/sections.json")
NUM_SECTIONS=$(jq '.sections | length' "${ANALYSIS_DIR}/sections.json")
echo "  Total bars: ${TOTAL_BARS}"
echo "  Sections: ${NUM_SECTIONS}"

# セクション一覧表示
echo ""
echo "  Section Layout:"
jq -r '.sections[] | "    Bar \(.bar): \(.label) (Key: \(.key_hint), Preset: \(.preset))"' \
    "${ANALYSIS_DIR}/sections.json"

echo ""
echo "  Generators enabled:"
echo "    • Piano (with ML velocity model)"
echo "    • Guitar (Stage2 with articulation)"
echo "    • Strings (Stage2 with Phase 31)"
echo "    • Bass (Stage2 with Phase 31)"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: modular_composer 実行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "🚀 Step 4/4: Running modular_composer..."
echo "─────────────────────────────────────────────"
echo ""

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_FILENAME="ore_no_nyoubou_${TIMESTAMP}.mid"

echo "Command:"
cat << 'CMD'
docker run --rm -v "$(pwd)":/app -w /app composer2 python modular_composer.py \
  --main-cfg config/main_cfg.yml \
  --chordmap data/suno_ai/suno_themesong/song_001/analysis/chordmap.json \
  --rhythm data/rhythm_library.yml \
  --tempo-curve data/suno_ai/suno_themesong/song_001/analysis/tempo_map.json \
  --output-dir output/song_001 \
  --output-filename [OUTPUT_FILENAME] \
  --verbose
CMD
echo ""
echo "Output: ${OUTPUT_DIR}/${OUTPUT_FILENAME}"
echo ""

# 実行確認
read -p "▶️  Execute production run? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏸️  Production run cancelled by user."
    echo ""
    echo "To run manually:"
    echo "  docker run --rm -v \"\$(pwd)\":/app -w /app composer2 python modular_composer.py \\"
    echo "    --main-cfg config/main_cfg.yml \\"
    echo "    --chordmap ${ANALYSIS_DIR}/chordmap.json \\"
    echo "    --rhythm data/rhythm_library.yml \\"
    echo "    --tempo-curve ${ANALYSIS_DIR}/tempo_map.json \\"
    echo "    --output-dir ${OUTPUT_DIR} \\"
    echo "    --output-filename ${OUTPUT_FILENAME} \\"
    echo "    --verbose"
    exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 実行開始
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "═══════════════════════════════════════════════"
echo "  Starting MIDI Generation..."
echo "═══════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)

docker run --rm -v "$(pwd)":/app -w /app composer2 python modular_composer.py \
  --main-cfg config/main_cfg.yml \
  --chordmap "${ANALYSIS_DIR}/chordmap.json" \
  --rhythm data/rhythm_library.yml \
  --tempo-curve "${ANALYSIS_DIR}/tempo_map.json" \
  --output-dir "${OUTPUT_DIR}" \
  --output-filename "${OUTPUT_FILENAME}" \
  --verbose

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "═══════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ MIDI Generation Complete!"
    echo "═══════════════════════════════════════════════"
    echo ""
    echo "📊 Summary:"
    echo "  • Duration: ${DURATION} seconds"
    echo "  • Output: ${OUTPUT_DIR}/${OUTPUT_FILENAME}"
    
    if [ -f "${OUTPUT_DIR}/${OUTPUT_FILENAME}" ]; then
        FILE_SIZE=$(ls -lh "${OUTPUT_DIR}/${OUTPUT_FILENAME}" | awk '{print $5}')
        echo "  • File size: ${FILE_SIZE}"
    fi
    
    echo ""
    echo "🎹 Next steps:"
    echo "  1. Open MIDI in DAW (Logic Pro, Ableton, etc.)"
    echo "  2. Load appropriate instrument patches"
    echo "  3. Adjust mix and apply effects"
    echo "  4. Export final audio"
    echo ""
    echo "📝 Generated with:"
    echo "  • Phase 31 scale constraint (Piano: 0.8, Strings: 0.85, Guitar: 0.75, Bass: 0.5)"
    echo "  • Preset system (10 genre-specific presets with chord-relative mode)"
    echo "  • Multi-stem chordmap analysis"
    echo "  • Lyric-aware anchoring (${NUM_SECTIONS} sections)"
    echo ""
else
    echo "  ❌ MIDI Generation Failed"
    echo "═══════════════════════════════════════════════"
    echo ""
    echo "Exit code: ${EXIT_CODE}"
    echo "Duration: ${DURATION} seconds"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "  1. Check log output above for error messages"
    echo "  2. Verify all JSON files are valid (use jq to validate)"
    echo "  3. Ensure Docker container has latest code"
    echo "  4. Check generator configuration in main_cfg.yml"
    echo ""
    exit $EXIT_CODE
fi
