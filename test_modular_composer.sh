#!/bin/bash
# modular_composer クイックテスト実行スクリプト

SONG_DIR="data/suno_ai/suno_themesong/song_001"

echo "🎵 Modular Composer - Quick Test"
echo "================================"
echo ""

# 必須ファイルチェック
echo "📂 Checking required files..."
MISSING=0

check_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $1"
    else
        echo "  ❌ $1 - MISSING"
        MISSING=$((MISSING + 1))
    fi
}

# 必須ファイル
check_file "config/main_cfg.yml"
check_file "$SONG_DIR/analysis/chordmap_auto.json"
check_file "$SONG_DIR/analysis/sections.json"
check_file "$SONG_DIR/analysis/tempo_map.json"
check_file "$SONG_DIR/analysis/lyric_anchors.json"

# オプショナルファイル
echo ""
echo "📂 Optional files..."
check_file "data/rhythm_library.yml" && RHYTHM_OK=1 || RHYTHM_OK=0
check_file "data/groove_profile.json" && GROOVE_OK=1 || GROOVE_OK=0

echo ""
if [ $MISSING -gt 0 ]; then
    echo "❌ Missing $MISSING required files. Please prepare them first."
    exit 1
fi

# rhythm_library.yml がない場合は最小限作成
if [ $RHYTHM_OK -eq 0 ]; then
    echo "⚠️  rhythm_library.yml not found. Creating minimal version..."
    cat > data/rhythm_library.yml << 'EOF'
# Minimal rhythm library for testing
drums:
  basic:
    pattern: [1, 0, 0, 0, 1, 0, 0, 0]
    velocity: [90, 0, 0, 0, 85, 0, 0, 0]
EOF
    echo "  ✅ Created data/rhythm_library.yml"
fi

echo ""
echo "🚀 Ready to run modular_composer!"
echo ""
echo "Test command:"
echo "─────────────────────────────────────────────────────"
cat << 'CMD'
docker run --rm -v "$(pwd)":/app -w /app composer2 python modular_composer.py \
  --main-cfg config/main_cfg.yml \
  --chordmap data/suno_ai/suno_themesong/song_001/analysis/chordmap_auto.json \
  --rhythm data/rhythm_library.yml \
  --tempo-curve data/suno_ai/suno_themesong/song_001/analysis/tempo_map.json \
  --output-dir output/song_001 \
  --output-filename test_output.mid \
  --verbose
CMD
echo "─────────────────────────────────────────────────────"
echo ""
echo "Or use simplified version (if paths in main_cfg.yml):"
echo "─────────────────────────────────────────────────────"
cat << 'CMD2'
docker run --rm -v "$(pwd)":/app -w /app composer2 python modular_composer.py \
  --main-cfg config/main_cfg.yml \
  -v
CMD2
echo "─────────────────────────────────────────────────────"
echo ""

# 実行確認
read -p "Run test now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🎵 Executing modular_composer..."
    echo ""
    
    docker run --rm -v "$(pwd)":/app -w /app composer2 python modular_composer.py \
      --main-cfg config/main_cfg.yml \
      --chordmap data/suno_ai/suno_themesong/song_001/analysis/chordmap_auto.json \
      --rhythm data/rhythm_library.yml \
      --tempo-curve data/suno_ai/suno_themesong/song_001/analysis/tempo_map.json \
      --output-dir output/song_001 \
      --output-filename test_output.mid \
      --verbose
    
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Success! Output: output/song_001/test_output.mid"
    else
        echo "❌ Failed with exit code $EXIT_CODE"
    fi
else
    echo ""
    echo "Skipped execution. Run the command manually when ready."
fi
