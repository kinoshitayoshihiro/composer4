#!/bin/bash
# ChordMap自動生成 - クイックコマンド集
# Usage: source chordmap_commands.sh

# =========================
# 基本コマンド（マルチステム版）
# =========================

# J-POP（推奨）
generate_chordmap_jpop() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python generate_chordmap_with_scale.py \
      --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
      --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
      --output data/suno_ai/suno_themesong/song_001/analysis/chordmap_auto.json \
      --genre j-pop \
      --alpha 0.25 \
      --hop-length 512
}

# Ballad/Acoustic
generate_chordmap_ballad() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python generate_chordmap_with_scale.py \
      --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
      --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
      --output data/suno_ai/suno_themesong/song_001/analysis/chordmap_ballad.json \
      --genre ballad \
      --alpha 0.28 \
      --hop-length 512
}

# J-Rock
generate_chordmap_jrock() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python generate_chordmap_with_scale.py \
      --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
      --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
      --output data/suno_ai/suno_themesong/song_001/analysis/chordmap_jrock.json \
      --genre j-rock \
      --alpha 0.25 \
      --hop-length 512
}

# City Pop
generate_chordmap_citypop() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python generate_chordmap_with_scale.py \
      --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
      --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
      --output data/suno_ai/suno_themesong/song_001/analysis/chordmap_citypop.json \
      --genre citypop \
      --alpha 0.25 \
      --hop-length 512
}

# 演歌（Enka）
generate_chordmap_enka() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python generate_chordmap_with_scale.py \
      --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
      --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
      --output data/suno_ai/suno_themesong/song_001/analysis/chordmap_enka.json \
      --genre enka \
      --alpha 0.30 \
      --hop-length 512
}

# =========================
# Alpha調整版（実験用）
# =========================

# 弱め（Chromagram優先）
generate_chordmap_weak() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python generate_chordmap_with_scale.py \
      --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
      --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
      --output data/suno_ai/suno_themesong/song_001/analysis/chordmap_weak.json \
      --genre j-pop \
      --alpha 0.20 \
      --hop-length 512
}

# 強め（Scale Prior優先）
generate_chordmap_strong() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python generate_chordmap_with_scale.py \
      --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
      --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
      --output data/suno_ai/suno_themesong/song_001/analysis/chordmap_strong.json \
      --genre j-pop \
      --alpha 0.35 \
      --hop-length 512
}

# =========================
# シングルオーディオ版（フォールバック）
# =========================
generate_chordmap_single() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python generate_chordmap_with_scale.py \
      --audio data/suno_ai/suno_themesong/song_001/full.wav \
      --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
      --output data/suno_ai/suno_themesong/song_001/analysis/chordmap_single.json \
      --genre j-pop \
      --alpha 0.25 \
      --hop-length 512
}

# =========================
# 結果確認
# =========================
view_chordmap() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python -c "
import json
with open('data/suno_ai/suno_themesong/song_001/analysis/chordmap_auto.json', 'r') as f:
    data = json.load(f)
print('📊 ChordMap Statistics:')
print(f'  Total bars: {len(data[\"chords\"])}')
print(f'  Genre: {data[\"meta\"][\"genre\"]}')
print(f'  Alpha: {data[\"meta\"][\"scale_prior_alpha\"]}')
print(f'  Method: {data[\"meta\"][\"method\"]}')
print('\\n🎼 First 20 bars:')
for c in data['chords'][:20]:
    print(f'  Bar {c[\"bar\"]:3d}: {c[\"chord\"]}')
print('\\n🎛️  Preset Assignments:')
for bar, preset in sorted(data['meta']['presets'].items())[:8]:
    print(f'  Bar {bar}: {preset}')
"
}

# プリセット一覧
list_presets() {
    docker run --rm -v "$(pwd)":/app -w /app composer2 python -c "
from ops.scale_modes import list_presets, describe_preset
print('🎛️  Available Presets:')
for p in list_presets():
    desc = describe_preset(p)
    print(f'  {p:25s} | Mode: {desc.get(\"mode\", \"N/A\"):12s} | Blues: {desc.get(\"blues\", 0):.2f}')
"
}

# =========================
# 使用例
# =========================
echo "✅ ChordMap generation commands loaded!"
echo ""
echo "Quick Start:"
echo "  generate_chordmap_jpop      # J-POP（推奨）"
echo "  generate_chordmap_ballad    # Ballad"
echo "  generate_chordmap_citypop   # City Pop"
echo ""
echo "View Results:"
echo "  view_chordmap               # 結果確認"
echo "  list_presets                # プリセット一覧"
echo ""
echo "Experimental:"
echo "  generate_chordmap_weak      # α=0.20（弱め）"
echo "  generate_chordmap_strong    # α=0.35（強め）"
