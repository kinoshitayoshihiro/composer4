#!/usr/bin/env bash
set -euo pipefail

# e2e_suno_arrangement.patched.sh
# - Phase A/B の主要ステップを存在確認しながら順次実行
# - 多様な設定ファイル名に耐える（見つかったものを優先採用）
# - drums_midi_to_plan_real.py の新オプション（style/accent/auto-regen）に対応

ROOT="${1:-.}"
SONG_ID="${SONG_ID:-song_004}"
ANALYSIS="${ANALYSIS_DIR:-$ROOT/analysis}"
FEATURES="${FEATURES_DIR:-$ROOT/features}"
PLANS="${PLANS_DIR:-$ROOT/plans}"
MIDI_DIR="${MIDI_DIR:-$ROOT/midi}"
ROLE_BARS_DIR="${ROLE_BARS_DIR:-$ROOT/analysis/role_bars}"
STYLE_YAML="${STYLE_YAML:-$ROOT/drums_style.yaml}"
ACCENT_PLAN="${ACCENT_PLAN:-$ANALYSIS/drum_accent_plan.json}"
STEMS_FEATS="${STEMS_FEATS:-$ROOT/stem_features.parquet}"  # 互換リンク想定

mkdir -p "$ANALYSIS" "$FEATURES" "$PLANS" "$MIDI_DIR"

py() { python3 "$@"; }

# ---------- Phase A (必要最小) ----------
# bars / sections がなければスキップせずに失敗させる（最低限必要）
test -f "$ANALYSIS/bars.parquet" || { echo "ERROR: $ANALYSIS/bars.parquet not found"; exit 2; }
test -f "$ANALYSIS/sections.json" || { echo "ERROR: $ANALYSIS/sections.json not found"; exit 2; }

# lyric anchors (任意)
[ -f "$ANALYSIS/lyric_anchors.json" ] || echo '{"anchors":[]}' > "$ANALYSIS/lyric_anchors.json"

# chordmap (auto) 任意
if [ ! -f "$ANALYSIS/chordmap.json" ] && [ -f "$ROOT/scripts/estimate_chordmap.py" ]; then
  py "$ROOT/scripts/estimate_chordmap.py" --bars "$ANALYSIS/bars.parquet" --out "$ANALYSIS/chordmap.json" || true
fi

# stems_features（互換名もケア）
if [ ! -f "$STEMS_FEATS" ] && [ -f "$ROOT/stems_features.parquet" ]; then
  STEMS_FEATS="$ROOT/stems_features.parquet"
fi

# role_bars（任意）
GUITAR_RB="$ROLE_BARS_DIR/guitar.parquet"
PIANO_RB="$ROLE_BARS_DIR/piano.parquet"
STRINGS_RB="$ROLE_BARS_DIR/strings.parquet"
BASS_RB="$ROLE_BARS_DIR/bass.parquet"

# ---------- Phase B ----------
# Step 17: chordmap_m21（存在すればスキップ）
if [ -f "$ROOT/scripts/chordmap_to_music21.py" ]; then
  [ -f "$ANALYSIS/chordmap_m21.json" ] || py "$ROOT/scripts/chordmap_to_music21.py" \
    --chordmap "$ANALYSIS/chordmap_locked.json" \
    --out "$ANALYSIS/chordmap_m21.json" || true
fi

# Step 18: chordmap views
if [ -f "$ROOT/scripts/make_instrument_chordmap_views.py" ]; then
  py "$ROOT/scripts/make_instrument_chordmap_views.py" \
    --chordmap "$ANALYSIS/chordmap_locked.json" \
    --out-dir "$ANALYSIS" || true
fi

# Step 19: plans（各ロール）
# Piano
if [ -f "$ROOT/scripts/instrument_midi_to_plan_real.py" ]; then
  py "$ROOT/scripts/instrument_midi_to_plan_real.py" \
    --role piano \
    --song-package "$ROOT/song_package_standard.yaml" \
    --bars "$ANALYSIS/bars.parquet" \
    --chordmap "$ANALYSIS/chordmap_locked.json" \
    --sections "$ANALYSIS/sections.json" \
    --stems-features "$STEMS_FEATS" \
    --role-bars "$PIANO_RB" \
    --view "$ANALYSIS/chordmap_view_piano.json" \
    --policy "$ROOT/chordmap_view_piano.yaml" \
    --roman-json "$ANALYSIS/roman_map.json" \
    --melody-hotspots "$ANALYSIS/melody_hotspots.json" \
    --tension-policy auto \
    --voice-leading \
    --out "$PLANS/piano_plan.json" || true
fi

# Guitar
if [ -f "$ROOT/scripts/instrument_midi_to_plan_real.py" ]; then
  py "$ROOT/scripts/instrument_midi_to_plan_real.py" \
    --role guitar \
    --song-package "$ROOT/song_package_standard.yaml" \
    --bars "$ANALYSIS/bars.parquet" \
    --chordmap "$ANALYSIS/chordmap_locked.json" \
    --sections "$ANALYSIS/sections.json" \
    --stems-features "$STEMS_FEATS" \
    --role-bars "$GUITAR_RB" \
    --view "$ANALYSIS/chordmap_view_guitar.json" \
    --policy "$ROOT/chordmap_view_guitar.yaml" \
    --roman-json "$ANALYSIS/roman_map.json" \
    --melody-hotspots "$ANALYSIS/melody_hotspots.json" \
    --tension-policy auto \
    --strum --voice-leading \
    --out "$PLANS/guitar_plan.json" || true
fi

# Strings
if [ -f "$ROOT/scripts/instrument_midi_to_plan_real.py" ]; then
  py "$ROOT/scripts/instrument_midi_to_plan_real.py" \
    --role strings \
    --song-package "$ROOT/song_package_standard.yaml" \
    --bars "$ANALYSIS/bars.parquet" \
    --chordmap "$ANALYSIS/chordmap_locked.json" \
    --sections "$ANALYSIS/sections.json" \
    --stems-features "$STEMS_FEATS" \
    --role-bars "$STRINGS_RB" \
    --view "$ANALYSIS/chordmap_view_strings.json" \
    --policy "$ROOT/chordmap_view_strings.yaml" \
    --roman-json "$ANALYSIS/roman_map.json" \
    --melody-hotspots "$ANALYSIS/melody_hotspots.json" \
    --tension-policy auto \
    --out "$PLANS/strings_plan.json" || true
fi

# Bass
if [ -f "$ROOT/scripts/instrument_midi_to_plan_real.py" ]; then
  py "$ROOT/scripts/instrument_midi_to_plan_real.py" \
    --role bass \
    --song-package "$ROOT/song_package_standard.yaml" \
    --bars "$ANALYSIS/bars.parquet" \
    --chordmap "$ANALYSIS/chordmap_locked.json" \
    --sections "$ANALYSIS/sections.json" \
    --stems-features "$STEMS_FEATS" \
    --role-bars "$BASS_RB" \
    --view "$ANALYSIS/chordmap_view_bass.json" \
    --policy "$ROOT/chordmap_view_bass.yaml" \
    --roman-json "$ANALYSIS/roman_map.json" \
    --melody-hotspots "$ANALYSIS/melody_hotspots.json" \
    --tension-policy auto \
    --walking-bass \
    --out "$PLANS/bass_plan.json" || true
fi

# Drums (新オプション対応・自動リジェネON)
if [ -f "$ROOT/scripts/drums_midi_to_plan_real.py" ]; then
  # style yaml: 無ければ example をコピー
  if [ ! -f "$STYLE_YAML" ] && [ -f "$ROOT/drums_style.yaml" ]; then
    STYLE_YAML="$ROOT/drums_style.yaml"
  elif [ ! -f "$STYLE_YAML" ] && [ -f "$ROOT/drums_style.example.yaml" ]; then
    STYLE_YAML="$ROOT/drums_style.example.yaml"
  fi
  py "$ROOT/scripts/drums_midi_to_plan_real.py" \
    --bars "$ANALYSIS/bars.parquet" \
    --stems-features "$STEMS_FEATS" \
    --style-yaml "$STYLE_YAML" \
    --accent-plan "$ACCENT_PLAN" \
    --auto-regen --min-events 8 --seed 777 \
    --out "$PLANS/drums_plan.json" || true
fi

# Step 20: 統合MIDI
if [ -f "$ROOT/scripts/merge_plans_to_midi.py" ]; then
  py "$ROOT/scripts/merge_plans_to_midi.py" \
    --bars "$ANALYSIS/bars.parquet" \
    --tempo-map "$ANALYSIS/tempo_map.json" \
    --plans "$PLANS" \
    --out "$MIDI_DIR/${SONG_ID}_integrated.mid" || true
fi

echo "[E2E] Done. MIDI: $MIDI_DIR/${SONG_ID}_integrated.mid"
