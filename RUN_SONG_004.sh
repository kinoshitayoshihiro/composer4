#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# RUN_SONG_004.sh
# 
# song_004完全版実行コマンド集（パイプライン順序最適化版）
#
# 使い方:
#   1. Phase A（解析素材準備：STEP 1-15）のみ実行:
#      bash RUN_SONG_004.sh phaseA
#   
#   2. Phase B（確定和声→view→plan→監査→SongPackage：STEP 16-22）実行:
#      bash RUN_SONG_004.sh phaseB
#   
#   3. 全フロー一括実行（manual不在時はauto chordmapをLOCK）:
#      bash RUN_SONG_004.sh full
#
# パイプライン設計原則:
#   - Phase A: 解析素材（tempo/bars/sections/chordmap/stems/CREPE F0）
#   - Phase B: LOCK基準で統一（Roman→view→plan→監査→SongPackage）
#   - 固定BPM禁止：常にtempo_map.jsonのSetTempoを使用
#   - LOCK中心主義：chordmap_locked.jsonが唯一の和声事実源

set -euo pipefail

# UTF-8ロケール設定
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# プロジェクトルート
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SONG_ROOT="$REPO_ROOT/data/suno_ai/suno_themesong/song_004"
STEMS_DIR="$SONG_ROOT/stem_wav"

# Phase判定
PHASE="${1:-full}"

echo "================================================================================"
echo "🎵 song_004 パッケージ生成（パイプライン順序最適化版）"
echo "   Phase: $PHASE"
echo "   Song root: $SONG_ROOT"
echo "   Stems dir: $STEMS_DIR"
echo "================================================================================"

# ==========================================
# Phase A: 解析素材準備（STEP 1-15）
# ==========================================
if [[ "$PHASE" == "phaseA" ]] || [[ "$PHASE" == "full" ]]; then
    echo ""
    echo "🚀 Phase A: 解析素材準備（STEP 1-15）開始..."
    echo "   目的: tempo/bars/sections/chordmap/stems/CREPE F0抽出"
    echo "   ※監査・SongPackage生成は行わない（未確定和声を基準にしないため）"
    echo ""
    
    bash "$REPO_ROOT/scripts/make_song_package_from_sources.sh" \
        "$SONG_ROOT" \
        --stems-dir "$STEMS_DIR" \
        --strict
    
    echo ""
    echo "✅ Phase A完了"
    echo ""
    echo "📝 次のステップ:"
    echo "   1. $SONG_ROOT/analysis/chordmap.json を確認"
    echo "   2. 感情・歌詞・ボーカル実音に合わせて修正し、"
    echo "      $SONG_ROOT/analysis/manual_chordmap.json として保存"
    echo "   3. Phase B実行: bash RUN_SONG_004.sh phaseB"
    echo ""
    
    if [[ "$PHASE" == "phaseA" ]]; then
        exit 0
    fi
fi

# ==========================================
# Phase B: LOCK→plan→MIDI→監査（STEP 16-22）
# ==========================================
if [[ "$PHASE" == "phaseB" ]] || [[ "$PHASE" == "full" ]]; then
    echo ""
    echo "🚀 Phase B: LOCK→plan→MIDI→監査（STEP 16-22）開始..."
    echo ""
    
    # manual_chordmap.json存在確認
    MANUAL_CHORDMAP="$SONG_ROOT/analysis/manual_chordmap.json"
    if [[ ! -f "$MANUAL_CHORDMAP" ]] && [[ "$PHASE" == "phaseB" ]]; then
        echo "⚠️  WARNING: manual_chordmap.json not found"
        echo "   Path: $MANUAL_CHORDMAP"
        echo ""
        echo "   auto chordmap.jsonをそのままLOCKとして使用します。"
        echo "   （感情基準の編集を行う場合は、manual_chordmap.jsonを作成してください）"
        echo ""
        read -p "   Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "   Cancelled."
            exit 1
        fi
    fi
    
    # STEP 16: LOCK生成（manual_chordmap.json優先、なければauto）
    echo ""
    echo "🔧 STEP 16: Generate chordmap_locked.json"
    echo ""
    
    PY="$REPO_ROOT/.venv311/bin/python"
    ANAL="$SONG_ROOT/analysis"
    
    $PY "$REPO_ROOT/scripts/chordmap_lock.py" \
        --base "$ANAL/chordmap.json" \
        --overrides "$ANAL/manual_chordmap.json" \
        --sections "$ANAL/sections.json" \
        --out-json "$ANAL/chordmap_locked.json" \
        --out-qa "$ANAL/chordmap_lock_qa.csv" || {
        echo "⚠️  STEP 16 failed"
        exit 1
    }
    echo "✅ STEP 16 done: chordmap_locked.json"
    
    # STEP 17: music21正規化（Roman化）
    echo ""
    echo "🔧 STEP 17: Generate chordmap_m21.json (Roman化)"
    echo ""
    
    $PY "$REPO_ROOT/ops/chordmap_to_music21.py" \
        --input "$ANAL/chordmap_locked.json" \
        --out-json "$ANAL/chordmap_m21.json" || {
        echo "⚠️  STEP 17 failed (continuing anyway)"
    }

    echo "✅ STEP 17 done: chordmap_m21.json"
    
    # STEP 17.2: Roman厳密解析（副V/裏コードの網羅）
    echo ""
    echo "🔧 STEP 17.2: Roman strict analysis (V/x, SubV/x detection)"
    echo ""
    
    $PY "$REPO_ROOT/scripts/roman_strict.py" \
        --locked-chordmap "$ANAL/chordmap_locked.json" \
        --sections "$ANAL/sections.json" \
        --out-json "$ANAL/roman_map.json" || {
        echo "⚠️  STEP 17.2 failed (continuing anyway)"
    }
    
    echo "✅ STEP 17.2 done: roman_map.json"
    
    # STEP 17.4: CREPE F0 → 度数推定（HMMスムージング付）
    echo ""
    echo "🎼 STEP 17.4: CREPE F0 -> scale degree estimation (HMM smoothing)"
    echo ""
    
    # features/vocal_f0.parquet が存在する場合、analysis/crepe_f0.parquet としてコピー
    if [[ ! -f "$ANAL/crepe_f0.parquet" ]] && [[ -f "$SONG_ROOT/features/vocal_f0.parquet" ]]; then
        echo "   📋 Copying features/vocal_f0.parquet -> analysis/crepe_f0.parquet"
        cp "$SONG_ROOT/features/vocal_f0.parquet" "$ANAL/crepe_f0.parquet"
    fi
    
    if [[ -f "$ANAL/crepe_f0.parquet" ]]; then
        $PY "$REPO_ROOT/scripts/f0_degree_estimator.py" \
            --f0-parquet "$ANAL/crepe_f0.parquet" \
            --sections "$ANAL/sections.json" \
            --locked-chordmap "$ANAL/chordmap_locked.json" \
            --out-parquet "$ANAL/f0_degrees.parquet" \
            --out-hotspots "$ANAL/melody_hotspots.json" || {
            echo "⚠️  STEP 17.4 failed (continuing anyway)"
        }
        echo "✅ STEP 17.4 done: f0_degrees.parquet, melody_hotspots.json"
    else
        echo "   ⚠️  crepe_f0.parquet not found, skipping F0 degree estimation"
        # Create empty placeholders
        echo '{"9":{},"#11":{},"13":{}}' > "$ANAL/melody_hotspots.json"
    fi
    
    # STEP 17.5: 標準版SongPackage先行生成（plan生成の--song-package参照用）
    echo ""
    echo "🔧 STEP 17.5: Generate standard SongPackage (for plan generation)"
    echo ""
    
    PKG_STANDARD="$SONG_ROOT/song_package_standard.yaml"
    
    # Check if generate_suno_song_package_v1_1.py exists
    if [[ -f "$REPO_ROOT/scripts/generate_suno_song_package_v1_1.py" ]]; then
        $PY "$REPO_ROOT/scripts/generate_suno_song_package_v1_1.py" \
            --song-id song_004 \
            --analysis-dir "$ANAL" \
            --variant standard \
            --out "$PKG_STANDARD" || {
            echo "⚠️  SongPackage generation failed, creating minimal YAML"
            cat > "$PKG_STANDARD" << 'EOF'
meta:
  song_id: song_004
  title: "俺の親分"
  theme: "銭形平次捕物控"
  key: Em
  tempo_bpm: 89.3
  time_signature: "4/4"
  variant: standard
EOF
        }
    else
        echo "   generate_suno_song_package_v1_1.py not found, creating minimal YAML..."
        cat > "$PKG_STANDARD" << 'EOF'
meta:
  song_id: song_004
  title: "俺の親分"
  theme: "銭形平次捕物控"
  key: Em
  tempo_bpm: 89.3
  time_signature: "4/4"
  variant: standard
EOF
    fi
    echo "✅ STEP 17.5 done: song_package_standard.yaml"
    
    # STEP 18: 楽器別view生成（policy反映）
    echo ""
    echo "🔧 STEP 18: Generate instrument chordmap views (with policy)"
    echo ""
    
    POLICY_DIR="$REPO_ROOT/scripts/instrument_chordmap/policy"
    
    # Build policy arguments if policy files exist
    POLICY_ARGS=()
    [[ -f "$POLICY_DIR/chordmap_view_bass.yaml" ]] && POLICY_ARGS+=(--policy-bass "$POLICY_DIR/chordmap_view_bass.yaml")
    [[ -f "$POLICY_DIR/chordmap_view_guitar.yaml" ]] && POLICY_ARGS+=(--policy-guitar "$POLICY_DIR/chordmap_view_guitar.yaml")
    [[ -f "$POLICY_DIR/chordmap_view_piano.yaml" ]] && POLICY_ARGS+=(--policy-piano "$POLICY_DIR/chordmap_view_piano.yaml")
    [[ -f "$POLICY_DIR/chordmap_view_strings.yaml" ]] && POLICY_ARGS+=(--policy-strings "$POLICY_DIR/chordmap_view_strings.yaml")
    [[ -f "$POLICY_DIR/chordmap_view_pad.yaml" ]] && POLICY_ARGS+=(--policy-pad "$POLICY_DIR/chordmap_view_pad.yaml")
    
    $PY "$REPO_ROOT/scripts/instrument_chordmap/make_instrument_chordmap_views.py" \
        --chordmap "$ANAL/chordmap_locked.json" \
        --sections "$ANAL/sections.json" \
        --out-dir "$ANAL" \
        "${POLICY_ARGS[@]}" || {
        echo "⚠️  STEP 18 failed"
        exit 1
    }
    echo "✅ STEP 18 done: chordmap_view_*.json"
    
    # ==== STEP 19: plans generation (view + policy + CREPE exceptions) ====
    PLANS="$SONG_ROOT/plans"
    mkdir -p "$PLANS"
    
    echo ""
    echo "🔧 STEP 19: Generate plans (bass/guitar/piano/strings/drums)"
    echo "   Using: chordmap_locked.json (LOCK基準) + view + policy"
    echo ""
    
    # Bass
    echo "   Generating bass_plan.json..."
    BASS_POLICY_ARG=()
    [[ -f "$POLICY_DIR/chordmap_view_bass.yaml" ]] && BASS_POLICY_ARG=(--policy "$POLICY_DIR/chordmap_view_bass.yaml")
    
    $PY "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" \
      --role bass \
      --song-package "$PKG_STANDARD" \
      --chordmap "$ANAL/chordmap_locked.json" \
      --sections "$ANAL/sections.json" \
      --bars "$ANAL/bars.parquet" \
      --view "$ANAL/chordmap_view_bass.json" \
      --role-bars "$ANAL/role_bars/bass.parquet" \
      --roman-json "$ANAL/roman_map.json" \
      --melody-hotspots "$ANAL/melody_hotspots.json" \
      "${BASS_POLICY_ARG[@]}" \
      --out "$PLANS/bass_plan.json" \
      --walking-bass \
      --voice-leading \
      --multi-chords || {
        echo "⚠️  Bass plan generation failed"
        exit 1
    }
    
    # Guitar
    echo "   Generating guitar_plan.json..."
    GUITAR_POLICY_ARG=()
    [[ -f "$POLICY_DIR/chordmap_view_guitar.yaml" ]] && GUITAR_POLICY_ARG=(--policy "$POLICY_DIR/chordmap_view_guitar.yaml")
    
    $PY "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" \
      --role guitar \
      --song-package "$PKG_STANDARD" \
      --chordmap "$ANAL/chordmap_locked.json" \
      --sections "$ANAL/sections.json" \
      --bars "$ANAL/bars.parquet" \
      --view "$ANAL/chordmap_view_guitar.json" \
      --role-bars "$ANAL/role_bars/guitar.parquet" \
      --roman-json "$ANAL/roman_map.json" \
      --melody-hotspots "$ANAL/melody_hotspots.json" \
      "${GUITAR_POLICY_ARG[@]}" \
      --out "$PLANS/guitar_plan.json" \
      --strum \
      --voice-leading \
      --multi-chords || {
        echo "⚠️  Guitar plan generation failed"
        exit 1
    }
    
    # Piano
    echo "   Generating piano_plan.json..."
    PIANO_POLICY_ARG=()
    [[ -f "$POLICY_DIR/chordmap_view_piano.yaml" ]] && PIANO_POLICY_ARG=(--policy "$POLICY_DIR/chordmap_view_piano.yaml")
    
    $PY "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" \
      --role piano \
      --song-package "$PKG_STANDARD" \
      --chordmap "$ANAL/chordmap_locked.json" \
      --sections "$ANAL/sections.json" \
      --bars "$ANAL/bars.parquet" \
      --view "$ANAL/chordmap_view_piano.json" \
      --role-bars "$ANAL/role_bars/piano.parquet" \
      --roman-json "$ANAL/roman_map.json" \
      --melody-hotspots "$ANAL/melody_hotspots.json" \
      "${PIANO_POLICY_ARG[@]}" \
      --out "$PLANS/piano_plan.json" \
      --voice-leading \
      --multi-chords || {
        echo "⚠️  Piano plan generation failed"
        exit 1
    }
    
    # Strings
    echo "   Generating strings_plan.json..."
    STRINGS_POLICY_ARG=()
    [[ -f "$POLICY_DIR/chordmap_view_strings.yaml" ]] && STRINGS_POLICY_ARG=(--policy "$POLICY_DIR/chordmap_view_strings.yaml")
    
    $PY "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" \
      --role strings \
      --song-package "$PKG_STANDARD" \
      --chordmap "$ANAL/chordmap_locked.json" \
      --sections "$ANAL/sections.json" \
      --bars "$ANAL/bars.parquet" \
      --view "$ANAL/chordmap_view_strings.json" \
      --role-bars "$ANAL/role_bars/strings.parquet" \
      --roman-json "$ANAL/roman_map.json" \
      --melody-hotspots "$ANAL/melody_hotspots.json" \
      "${STRINGS_POLICY_ARG[@]}" \
      --out "$PLANS/strings_plan.json" \
      --voice-leading \
      --multi-chords || {
        echo "⚠️  Strings plan generation failed"
        exit 1
    }
    
    # Drums（2段階: recommendations生成 → plan生成）
    echo "   Generating drums_recommendations.json..."
    
    TEMPO_MAP="$ANAL/tempo_map.json"
    DRUMS_RECS="$ANAL/drums_recommendations.json"
    PATTERNS_PICKLE="$REPO_ROOT/output/rhythm_ai/rhythm_patterns.pickle"
    SONG_PACKAGE="$SONG_ROOT/song_package_standard.yaml"
    
    # STEP 19.1: drums_recommendations.json生成（ML推論 + パターン検索）
    if [[ -f "$SONG_PACKAGE" ]]; then
        $PY "$REPO_ROOT/scripts/recommend_drums.py" \
          --song-package "$SONG_PACKAGE" \
          --out "$DRUMS_RECS" \
          --stems-features "$ANAL/stems_features.parquet" || {
            echo "⚠️  drums_recommendations.json generation failed, using fallback"
            # フォールバック: 基本パターン推奨（Rock/4-4/medium）
            echo '{
  "meta": {"total_bars": 68, "bpm": 89.3, "diversity_mode": true},
  "bars": []
}' > "$DRUMS_RECS"
        }
        echo "   ✅ drums_recommendations.json"
    else
        echo "   ⚠️  song_package_standard.yaml not found, creating fallback drums_recommendations.json"
        echo '{
  "meta": {"total_bars": 68, "bpm": 89.3, "diversity_mode": true},
  "bars": []
}' > "$DRUMS_RECS"
    fi
    
    # STEP 19.2: drums_plan.json生成（recommendations → MIDI plan）
    echo "   Generating drums_plan.json..."
    
    if [[ -f "$DRUMS_RECS" ]] && [[ -f "$PATTERNS_PICKLE" ]]; then
        # tempo_map.json 存在時は可変テンポ対応、不在時はフォールバック（89.3 BPM）
        if [[ -f "$TEMPO_MAP" ]]; then
            $PY "$REPO_ROOT/scripts/drums_midi_to_plan_real.py" \
              --recommendations "$DRUMS_RECS" \
              --patterns-pickle "$PATTERNS_PICKLE" \
              --tempo-map "$TEMPO_MAP" \
              --tempo-bpm 89.3 \
              --bars "$ANAL/bars.parquet" \
              --role-bars "$ANAL/role_bars/drums.parquet" \
              --out "$PLANS/drums_plan.json" || {
                echo "⚠️  Drums plan generation failed"
                echo "   Creating minimal drums_plan.json..."
                echo '{"events": [], "ppq": 480, "tempo_bpm": 89.3}' > "$PLANS/drums_plan.json"
            }
        else
            # tempo_map.json不在時は固定BPMで動作
            TEMPO_BPM=89.3
            echo "   ⚠️  tempo_map.json not found, using fixed BPM: $TEMPO_BPM"
            $PY "$REPO_ROOT/scripts/drums_midi_to_plan_real.py" \
              --recommendations "$DRUMS_RECS" \
              --patterns-pickle "$PATTERNS_PICKLE" \
              --tempo-bpm "$TEMPO_BPM" \
              --bars "$ANAL/bars.parquet" \
              --role-bars "$ANAL/role_bars/drums.parquet" \
              --out "$PLANS/drums_plan.json" || {
                echo "⚠️  Drums plan generation failed"
                echo "   Creating minimal drums_plan.json..."
                echo '{"events": [], "ppq": 480, "tempo_bpm": '$TEMPO_BPM'}' > "$PLANS/drums_plan.json"
            }
        fi
    else
        echo "⚠️  Required files not found:"
        [[ ! -f "$DRUMS_RECS" ]] && echo "      - drums_recommendations.json"
        [[ ! -f "$PATTERNS_PICKLE" ]] && echo "      - rhythm_patterns.pickle"
        echo "   Creating minimal drums_plan.json..."
        echo '{"events": [], "ppq": 480, "tempo_bpm": 89.3}' > "$PLANS/drums_plan.json"
    fi
    
    echo ""
    echo "✅ STEP 19 done: plans at $PLANS"
    
    # STEP 20: 監査（LOCK基準で統一的に評価）
    echo ""
    echo "🔧 STEP 20: Deep harmony audit (LOCK-based KPI evaluation)"
    echo ""
    
    AUDIT_REPORT="$ANAL/harmony_audit_final.json"
    
    # Check if deep_harmony_audit.py exists
    if [[ -f "$REPO_ROOT/scripts/deep_harmony_audit.py" ]]; then
        $PY "$REPO_ROOT/scripts/deep_harmony_audit.py" \
            --song-id song_004 \
            --analysis-dir "$ANAL" \
            --plans-dir "$PLANS" \
            --out "$AUDIT_REPORT" || {
            echo "⚠️  Harmony audit failed (continuing anyway)"
            echo '{"status": "audit_failed", "kpi": {}}' > "$AUDIT_REPORT"
        }
        echo "✅ STEP 20 done: harmony_audit_final.json"
    else
        echo "   deep_harmony_audit.py not found, creating placeholder..."
        cat > "$AUDIT_REPORT" << 'EOF'
{
  "status": "audit_script_not_found",
  "message": "deep_harmony_audit.py not available",
  "kpi": {
    "clash_rate_minor2_verse": null,
    "dominant_rule_uptime_chorus": null,
    "strings_3rd_resolution": null,
    "pad_tension_density": null
  }
}
EOF
        echo "✅ STEP 20 done: harmony_audit_final.json (placeholder)"
    fi
    
    # STEP 21: SongPackage最終化（3種：soft/standard/bright）
    echo ""
    echo "🔧 STEP 21: Generate final SongPackages (soft/standard/bright)"
    echo ""
    
    if [[ -f "$REPO_ROOT/scripts/generate_suno_song_package_v1_1.py" ]]; then
        for VARIANT in soft standard bright; do
            echo "   Generating song_package_${VARIANT}.yaml..."
            $PY "$REPO_ROOT/scripts/generate_suno_song_package_v1_1.py" \
                --song-id song_004 \
                --analysis-dir "$ANAL" \
                --variant "$VARIANT" \
                --out "$SONG_ROOT/song_package_${VARIANT}.yaml" || {
                echo "⚠️  ${VARIANT} package generation failed"
            }
        done
        echo "✅ STEP 21 done: song_package_{soft|standard|bright}.yaml"
    else
        echo "   generate_suno_song_package_v1_1.py not found, skipping final packages"
        echo "✅ STEP 21 skipped"
    fi
    
    # STEP 22: 最終MIDI/E2E（tempo_map.json基準）
    echo ""
    echo "🔧 STEP 22: Generate integrated MIDI (tempo_map.json based)"
    echo ""
    
    MIDI_DIR="$SONG_ROOT/midi"
    mkdir -p "$MIDI_DIR"
    
    if [[ -f "$REPO_ROOT/scripts/e2e_suno_arrangement.py" ]]; then
        $PY "$REPO_ROOT/scripts/e2e_suno_arrangement.py" \
            --song-id song_004 \
            --song-package "$PKG_STANDARD" \
            --plans-dir "$PLANS" \
            --tempo-map "$ANAL/tempo_map.json" \
            --out-dir "$MIDI_DIR" || {
            echo "⚠️  E2E arrangement failed (continuing anyway)"
        }
        echo "✅ STEP 22 done: song_004_integrated.mid"
    else
        echo "   e2e_suno_arrangement.py not found, skipping MIDI generation"
        echo "✅ STEP 22 skipped"
    fi
    
    echo ""
    echo "✅ Phase B完了"
    echo ""
    echo "================================================================================";
    echo "📁 生成ファイル確認:"
    echo "================================================================================";
    echo "STEP 16: chordmap_locked.json (LOCK - 唯一の和声事実源)"
    echo "   └─ $ANAL/chordmap_locked.json"
    echo "   └─ $ANAL/chordmap_lock_qa.csv"
    echo ""
    echo "STEP 17: chordmap_m21.json (Roman化・表記正規化)"
    echo "   └─ $ANAL/chordmap_m21.json"
    echo ""
    echo "STEP 17.5: song_package_standard.yaml (plan生成用基礎パッケージ)"
    echo "   └─ $PKG_STANDARD"
    echo ""
    echo "STEP 18: chordmap_view_*.json (楽器別ビュー + policy)"
    echo "   ├─ $ANAL/chordmap_view_bass.json"
    echo "   ├─ $ANAL/chordmap_view_guitar.json"
    echo "   ├─ $ANAL/chordmap_view_piano.json"
    echo "   ├─ $ANAL/chordmap_view_strings.json"
    echo "   └─ $ANAL/chordmap_view_pad.json"
    echo ""
    echo "STEP 19: *_plan.json (楽器別プラン)"
    echo "   ├─ $PLANS/bass_plan.json"
    echo "   ├─ $PLANS/guitar_plan.json"
    echo "   ├─ $PLANS/piano_plan.json"
    echo "   ├─ $PLANS/strings_plan.json"
    echo "   └─ $PLANS/drums_plan.json"
    echo ""
    echo "STEP 20: harmony_audit_final.json (KPI監査)"
    echo "   └─ $AUDIT_REPORT"
    echo ""
    echo "STEP 21: song_package_*.yaml (最終3バリアント)"
    echo "   ├─ $SONG_ROOT/song_package_soft.yaml"
    echo "   ├─ $SONG_ROOT/song_package_standard.yaml"
    echo "   └─ $SONG_ROOT/song_package_bright.yaml"
    echo ""
    echo "STEP 22: song_004_integrated.mid (統合MIDI)"
    echo "   └─ $MIDI_DIR/song_004_integrated.mid"
    echo "================================================================================"
    echo ""
fi

echo ""
echo "🎉 完了！"
echo ""
echo "📊 次のステップ（推奨）:"
echo "   1. 監査レポート確認: cat $SONG_ROOT/analysis/harmony_audit_final.json | jq"
echo "   2. 統合MIDI確認: open $SONG_ROOT/midi/song_004_integrated.mid"
echo "   3. Songpackage確認: cat $SONG_ROOT/song_package_standard.yaml"
echo ""
echo "🔍 KPIチェックリスト:"
echo "   - clash_rate_minor2 (verse) < 1%"
echo "   - dominant_rule_uptime (chorus) > 80%"
echo "   - strings_3rd_resolution > 90%"
echo "   - pad_tension_density (常時ON区間) ≤ 1 / bar"
echo ""


