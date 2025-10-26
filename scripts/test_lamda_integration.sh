#!/bin/bash
# LAMDA統合 & 自己循環学習 クイックテスト

set -e

echo "🧪 LAMDA Integration Quick Test"
echo "================================"

# 1. LAMDAのCHORDSデータ変換テスト
echo ""
echo "📝 Step 1: Testing LAMDA CHORDS → chordmap conversion..."
python3 adapters/lamda_chords_to_chordmap.py \
  data/Los-Angeles-MIDI/CHORDS_DATA/LAMDa_CHORDS_DATA_5000.pickle \
  /tmp/test_chordmap.json

if [ -f "/tmp/test_chordmap.json" ]; then
  echo "✅ Chordmap conversion successful!"
  echo "   Preview:"
  head -20 /tmp/test_chordmap.json
else
  echo "❌ Chordmap conversion failed"
  exit 1
fi

# 2. 階層化スキーマテスト
echo ""
echo "📊 Step 2: Testing tiered data schema..."
python3 schemas/tiered_data_schema.py

if [ -f "data/tiered_corpus.jsonl" ]; then
  echo "✅ Tiered schema test successful!"
  echo "   Preview:"
  head -5 data/tiered_corpus.jsonl
else
  echo "❌ Tiered schema test failed"
  exit 1
fi

# 3. fluidsynthの存在確認
echo ""
echo "🎵 Step 3: Checking fluidsynth availability..."
FLUIDSYNTH_DIR="data/Los-Angeles-MIDI/CODE/fluidsynth-master"

if [ -d "$FLUIDSYNTH_DIR" ]; then
  echo "✅ fluidsynth-master found at: $FLUIDSYNTH_DIR"
  echo "   Contents:"
  ls -lh "$FLUIDSYNTH_DIR" | head -10
else
  echo "❌ fluidsynth-master not found"
  exit 1
fi

# 4. DawDreamerの存在確認
echo ""
echo "🎹 Step 4: Checking DawDreamer availability..."
if python3 -c "import dawdreamer; print(f'DawDreamer version: {dawdreamer.__version__}')" 2>/dev/null; then
  echo "✅ DawDreamer is available"
else
  echo "⚠️  DawDreamer not installed (optional for Phase A/B)"
  echo "   Install with: pip install dawdreamer>=0.7.0"
fi

# 5. カバレッジ格子の計算
echo ""
echo "📊 Step 5: Calculating coverage grid size..."
python3 -c "
grid = {
    'keys': 12,
    'tempo_bins': 6,
    'time_signatures': 3,
    'genres': 10,
    'emotions': 8,
}
total = 1
for k, v in grid.items():
    total *= v
    print(f'{k:20} : {v:5} variants')
print(f'{"="*30}')
print(f'Total cells         : {total:5}')
print(f'Target (1-2/cell)   : {total}-{total*2} songs')
print(f'Recommended GOLD    : 5,000-10,000 songs')
"

# 6. LAMDA統計
echo ""
echo "📚 Step 6: LAMDA CHORDS_DATA statistics..."
CHORDS_DIR="data/Los-Angeles-MIDI/CHORDS_DATA"
if [ -d "$CHORDS_DIR" ]; then
  NUM_FILES=$(ls "$CHORDS_DIR"/LAMDa_CHORDS_DATA_*.pickle 2>/dev/null | wc -l)
  echo "✅ Found $NUM_FILES CHORDS pickle files"
  echo "   Estimated total songs: $((NUM_FILES * 2500)) (~400k)"
else
  echo "❌ CHORDS_DATA directory not found"
fi

echo ""
echo "================================"
echo "✅ All core tests passed!"

echo ""
echo "🧪 Optional: A/B chord audit (外部 vs 内部)"
if [ -d "data/lamda_chordmaps" ] && [ -d "output/stage2/test/json" ]; then
  python3 scripts/ab_chord_audit.py \
    --ext-dir data/lamda_chordmaps \
    --int-dir output/stage2/test/json \
    --out-csv analysis/ab_chords_audit.csv || true
  echo "   → analysis/ab_chords_audit.csv を確認してください（任意ゲート）"
else
  echo "   (skip: ext/int ディレクトリ未検出)"
fi

echo ""
echo "📋 Next steps:"
echo "   1. Build fluidsynth: cd $FLUIDSYNTH_DIR/build && cmake .. && make"
echo "   2. Generate Suno GOLD: 5-10k songs with coverage grid"
echo "   3. Implement teacher_v1: python models/teacher_v1.py"
echo "   4. LAMDA → SILVER: bash scripts/lamda_to_silver.sh"
echo ""
echo "📖 See docs/LAMDA_INTEGRATION_PLAN.md for details"
