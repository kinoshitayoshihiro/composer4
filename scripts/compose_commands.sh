#!/bin/bash
# Composer2-3 作曲システム統合コマンド集（midi_writer.py統一版）
# Usage: source scripts/compose_commands.sh

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "${BASE_DIR}"

source .venv311/bin/activate

# ========================================
# 1. Full Pipeline（推奨メイン経路）
# ========================================

# Suno Song → 完全MIDI生成（Stage1 → Plan → midi_writer → 検証）
compose_full_pipeline() {
    local VOCAL_WAV="${1:-data/suno_ai/suno_themesong/song_001/stemswav_001/stem_wav_001_\(Vocals\).wav}"
    local ACCOMPANIMENT_WAV="${2:-data/suno_ai/suno_themesong/song_001/stemswav_001/stem_wav_001_\(Other\).wav}"
    local OUTPUT_DIR="${3:-output/full_pipeline}"
    local TEMPO="${4:-120}"
    
    echo "🚀 Full Pipeline実行（midi_writer.py統一版）"
    echo "  Vocal: ${VOCAL_WAV}"
    echo "  Accompaniment: ${ACCOMPANIMENT_WAV}"
    
    # Stage1: 分析（chordmap/sections/lyric_anchors生成）
    python scripts/generate_stage1_jsons.py \
        --song-dir "$(dirname "$(dirname "${VOCAL_WAV}")")" \
        --use-enhanced \
        --exclude Vocals \
        --force-key C
    
    # Stage2-4: MIDI生成（全楽器Plan生成 → midi_writer → 検証）
    python scripts/full_pipeline.py \
        --vocal "${VOCAL_WAV}" \
        --accompaniment "${ACCOMPANIMENT_WAV}" \
        --output "${OUTPUT_DIR}" \
        --tempo "${TEMPO}" \
        --max-drift-ms 30.0
}

# ========================================
# 2. Stage1分析（JSON生成）
# ========================================

# Suno Song → Stage1 JSON生成（chordmap/sections/lyric_anchors）
analyze_suno_song() {
    local SONG_DIR="${1:-data/suno_ai/suno_themesong/song_001}"
    local FORCE_KEY="${2:-C}"
    
    echo "📊 Stage1パイプライン実行"
    echo "  Song: ${SONG_DIR}"
    
    python scripts/generate_stage1_jsons.py \
        --song-dir "${SONG_DIR}" \
        --use-enhanced \
        --exclude Vocals \
        --force-key "${FORCE_KEY}" \
        --window-mode class \
        --sibilant-scale 1.6
}

# WAV → Chord Recognition（7th chords対応）
analyze_chords_from_wav() {
    local STEMS_DIR="${1:-data/suno_ai/suno_themesong/song_001/stemswav_001}"
    local OUTPUT="${2:-output/chordmap.json}"
    local USE_7TH="${3:-true}"
    local FORCE_KEY="${4:-C}"
    
    echo "🎼 WAV → Chord Recognition"
    
    if [ "${USE_7TH}" = "true" ]; then
        python ops/stem_harmony_7th.py \
            --stems "${STEMS_DIR}" \
            --out "${OUTPUT}" \
            --force-key "${FORCE_KEY}" \
            --exclude Vocals
    else
        python ops/stem_harmony.py \
            --stems "${STEMS_DIR}" \
            --out "${OUTPUT}" \
            --force-key "${FORCE_KEY}" \
            --exclude Vocals
    fi
}

# ========================================
# 3. YAML → MIDI生成（推奨）
# ========================================

# 構造YAML → MIDI生成（全楽器Plan → midi_writer）
compose_from_yaml() {
    local YAML_PATH="${1:-data/suno_structures/song1.yaml}"
    local OUTPUT_DIR="${2:-output/midi/from_yaml}"
    
    echo "📄 YAML → MIDI生成（midi_writer統一版）"
    echo "  YAML: ${YAML_PATH}"
    
    python scripts/arrange/arrange_from_yaml.py \
        --input "${YAML_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --enable-quality-gates
}

# ========================================
# 4. Tempo修復ツール
# ========================================

# 既存MIDI → Tempo Track注入（956s問題修復）
inject_tempo_track() {
    local INPUT_MIDI="${1:-output/full_pipeline/full_arrangement.mid}"
    local OUTPUT_MIDI="${2:-output/full_pipeline/full_arrangement_fixed.mid}"
    local TEMPO_SOURCE="${3:-bpm}"
    local BPM="${4:-120}"
    local TEMPO_MAP_JSON="${5}"
    
    echo "🔧 Tempo Track注入"
    echo "  Input: ${INPUT_MIDI}"
    
    if [ "${TEMPO_SOURCE}" = "bpm" ]; then
        python scripts/inject_tempo_track.py \
            --in "${INPUT_MIDI}" \
            --out "${OUTPUT_MIDI}" \
            --bpm "${BPM}" \
            --ts 4/4
    else
        python scripts/inject_tempo_track.py \
            --in "${INPUT_MIDI}" \
            --out "${OUTPUT_MIDI}" \
            --tempo-map "${TEMPO_MAP_JSON}" \
            --ts 4/4 \
            --beats-per-bar 4
    fi
}

# ========================================
# 5. MIDI → WAV レンダリング
# ========================================

# MIDI → WAV（FluidSynth）
render_midi_to_wav() {
    local MIDI_PATH="${1:-output/full_pipeline/full_arrangement.mid}"
    local OUTPUT_WAV="${2:-output/rendered_audio/output.wav}"
    local SOUNDFONT="${3:-/usr/share/soundfonts/GeneralUser.sf2}"
    
    echo "🔊 MIDI → WAV レンダリング"
    
    mkdir -p "$(dirname "${OUTPUT_WAV}")"
    
    fluidsynth -ni \
        "${SOUNDFONT}" \
        "${MIDI_PATH}" \
        -F "${OUTPUT_WAV}" \
        -r 44100
}

# ========================================
# 6. バッチ処理
# ========================================

# 複数Suno Song → 一括MIDI生成
compose_batch() {
    local BASE_DIR="${1:-data/suno_ai}"
    local OUTPUT_DIR="${2:-output/batch_composed}"
    local WORKERS="${3:-4}"
    
    echo "📦 バッチMIDI生成"
    
    python scripts/batch_chord_test_parallel.py \
        --base "${BASE_DIR}" \
        --output "${OUTPUT_DIR}" \
        --workers "${WORKERS}" \
        --use-7th
}

# ========================================
# 7. A/B比較テスト
# ========================================

# 2つのMIDI生成結果を比較
compare_midi_ab() {
    local RUN_A="${1:-output/run_a}"
    local RUN_B="${2:-output/run_b}"
    local OUTPUT_MD="${3:-output/ab_comparison.md}"
    
    echo "📊 A/B比較テスト"
    
    python scripts/ab_summarize.py \
        --a "${RUN_A}" \
        --b "${RUN_B}" \
        --threshold 50.0 \
        --out "${OUTPUT_MD}"
    
    cat "${OUTPUT_MD}"
}

# ========================================
# 8. Emotion別プリセット（簡易版）
# ========================================

compose_energetic() {
    compose_full_pipeline "$1" "$2" "${3:-output/energetic_midi}" 140
}

compose_melancholic() {
    compose_full_pipeline "$1" "$2" "${3:-output/melancholic_midi}" 90
}

compose_calm() {
    compose_full_pipeline "$1" "$2" "${3:-output/calm_midi}" 80
}

# ========================================
# 9. クイックスタート（デモ）
# ========================================

demo_compose() {
    echo "🎵 Composer2-3 デモ実行（最新版）"
    
    local SONG_DIR="data/suno_ai/suno_themesong/song_001"
    
    # 1. Stage1分析
    analyze_suno_song "${SONG_DIR}" "C"
    
    # 2. Full Pipeline実行
    compose_full_pipeline \
        "${SONG_DIR}/stemswav_001/stem_wav_001_(Vocals).wav" \
        "${SONG_DIR}/stemswav_001/stem_wav_001_(Other).wav" \
        "output/demo_midi" \
        120
    
    # 3. WAVレンダリング
    render_midi_to_wav \
        "output/demo_midi/full_arrangement.mid" \
        "output/demo_audio.wav" \
        "/usr/share/soundfonts/GeneralUser.sf2"
    
    echo "✅ デモ完了！"
    echo "   MIDI: output/demo_midi/full_arrangement.mid"
    echo "   WAV:  output/demo_audio.wav"
}

# ========================================
# ヘルプ表示
# ========================================

show_compose_help() {
    cat << 'EOF'
🎵 Composer2-3 作曲システム - コマンド一覧（midi_writer統一版）

【Full Pipeline（推奨）】
  compose_full_pipeline <vocal_wav> <accompaniment_wav> <output_dir> <tempo>
  
  例:
    compose_full_pipeline \
      data/suno_ai/suno_themesong/song_001/stemswav_001/stem_wav_001_\(Vocals\).wav \
      data/suno_ai/suno_themesong/song_001/stemswav_001/stem_wav_001_\(Other\).wav \
      output/full_pipeline \
      120

【Stage1分析】
  analyze_suno_song <song_dir> <force_key>
  analyze_chords_from_wav <stems_dir> <output> <use_7th> <force_key>

【YAML → MIDI】
  compose_from_yaml <yaml_path> <output_dir>

【Tempo修復】
  inject_tempo_track <input_midi> <output_midi> <source> <bpm> <tempo_map_json>
  
  例（固定BPM）:
    inject_tempo_track \
      output/full_pipeline/full_arrangement.mid \
      output/full_pipeline/full_arrangement_fixed.mid \
      bpm \
      120
  
  例（sections.json使用）:
    inject_tempo_track \
      output/full_pipeline/full_arrangement.mid \
      output/full_pipeline/full_arrangement_fixed.mid \
      map \
      "" \
      data/suno_ai/suno_themesong/song_001/analysis/sections.json

【レンダリング】
  render_midi_to_wav <midi_path> <output_wav> <soundfont>

【バッチ処理】
  compose_batch <base_dir> <output_dir> <workers>

【A/B比較】
  compare_midi_ab <run_a> <run_b> <output_md>

【Emotion別プリセット】
  compose_energetic <vocal_wav> <accompaniment_wav> <output_dir>
  compose_melancholic <vocal_wav> <accompaniment_wav> <output_dir>
  compose_calm <vocal_wav> <accompaniment_wav> <output_dir>

【デモ】
  demo_compose

【ヘルプ】
  show_compose_help

EOF
}

echo "✅ Composer2-3 コマンド読み込み完了（midi_writer統一版）"
echo "   ヘルプ: show_compose_help"