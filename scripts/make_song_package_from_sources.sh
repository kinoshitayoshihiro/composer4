#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# scripts/make_song_package_from_sources.sh
# 
# Song Package Pipeline — Phase A/B/C（plans-only / slots / V2）
#
# Phase A: 解析（instrument.wavを強制使用）/ CREPEは原音参照、symlink撤廃
# Phase B: manual_chordmap.json → LOCK → V2レンダラ → Harmony監査
# Phase C: MIDI統合（可変テンポ、split-tracks）
#
# 使い方:
#   bash scripts/make_song_package_from_sources.sh \
#     data/suno_ai/suno_themesong/song_001 \
#     --stems-dir "data/suno_ai/suno_themesong/song_001/stem_wav"
#
# オプション:
#   --stems-dir DIR       Stems WAV格納ディレクトリ（必須）
#   --mix-wav PATH        Mix WAVパス（指定しない場合は自動検出: instrument.wav優先）
#   --vocal-wav PATH      Vocal WAVパス（指定しない場合は自動検出）
#   --dry-run             実行コマンド表示のみ
#   --strict              失敗時即終了

set -euo pipefail

# UTF-8ロケール設定（日本語パス対応）
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# === User-tunable flags ===
# STEM_FEATURES: 0=skip / 1=run（V2運用では通常0。QA可視化時のみ1）
: "${STEM_FEATURES:=0}"
# STRICT_VOCAL: 1=ボーカルstem未検出なら即停止
: "${STRICT_VOCAL:=1}"

# デフォルト設定
DRY_RUN=0
STRICT=0
STEMS_DIR=""
MIX_WAV=""
VOCAL_WAV=""
PHASE_B_ONLY=0  # Phase B（STEP 16-22）のみ実行

# スクリプトディレクトリ（絶対パス）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Python実行環境（仮想環境優先）
if [[ -f "$REPO_ROOT/.venv311/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv311/bin/python"
elif [[ -f "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "   Python: $PYTHON_BIN"

# 引数パース
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stems-dir)
      STEMS_DIR="$2"
      shift 2
      ;;
    --mix-wav)
      MIX_WAV="$2"
      shift 2
      ;;
    --vocal-wav)
      VOCAL_WAV="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

# 必須引数チェック
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <song_root_dir> --stems-dir <stems_dir> [options]"
  echo ""
  echo "Options:"
  echo "  --stems-dir DIR       Stems WAV格納ディレクトリ（必須）"
  echo "  --mix-wav PATH        Mix WAVパス（指定しない場合は自動検出）"
  echo "  --vocal-wav PATH      Vocal WAVパス（指定しない場合は自動検出）"
  echo "  --dry-run             実行コマンド表示のみ"
  echo "  --strict              失敗時即終了"
  exit 1
fi

SONG_ROOT="$1"
SONG_ROOT="$(cd "$SONG_ROOT" && pwd)"  # 絶対パス化

# SONG_DIR定義（シンボリックリンク作成で使用）
SONG_DIR="$SONG_ROOT"

# stems-dir必須チェック
if [[ -z "$STEMS_DIR" ]]; then
  echo "❌ Error: --stems-dir is required"
  exit 1
fi

# stems-dir絶対パス化
if [[ ! "$STEMS_DIR" = /* ]]; then
  # 相対パスの場合、カレントディレクトリ基準で解決
  if [[ -d "$STEMS_DIR" ]]; then
    STEMS_DIR="$(cd "$STEMS_DIR" && pwd)"
  elif [[ -d "$SONG_ROOT/$STEMS_DIR" ]]; then
    STEMS_DIR="$(cd "$SONG_ROOT/$STEMS_DIR" && pwd)"
  else
    echo "❌ Error: stems-dir not found: $STEMS_DIR"
    exit 1
  fi
fi

echo "🎵 Song Package Generation: $SONG_ROOT"
echo "   Stems dir: $STEMS_DIR"

# ==========================================
# Mix WAV自動検出（instrument.wav優先）
# ==========================================
if [[ -z "$MIX_WAV" ]]; then
  echo "🔍 Auto-detecting Mix WAV (instrument.wav優先)..."
  
  # パターン1: instrument.wav（Suno AI標準）
  MIX_WAV="$STEMS_DIR/instrument.wav"
  if [[ ! -f "$MIX_WAV" ]]; then
    MIX_WAV="$(find "$STEMS_DIR" -iname "*instrument*.wav" | head -n1)"
  fi
  
  # パターン2: other.wav / mix.wav
  if [[ -z "$MIX_WAV" ]] || [[ ! -f "$MIX_WAV" ]]; then
    MIX_WAV="$STEMS_DIR/other.wav"
    if [[ ! -f "$MIX_WAV" ]]; then
      MIX_WAV="$STEMS_DIR/mix.wav"
    fi
  fi
  
  # パターン3: Strings / Piano / Guitar（フォールバック）
  if [[ -z "$MIX_WAV" ]] || [[ ! -f "$MIX_WAV" ]]; then
    MIX_WAV="$(find "$STEMS_DIR" -iname "*Strings*.wav" -o -iname "*Piano*.wav" -o -iname "*Guitar*.wav" | head -n1)"
  fi
  
  # パターン4: _auto_Other.wav（最終フォールバック）
  if [[ -z "$MIX_WAV" ]] || [[ ! -f "$MIX_WAV" ]]; then
    MIX_WAV="$STEMS_DIR/_auto_Other.wav"
  fi
  
  if [[ -z "$MIX_WAV" ]] || [[ ! -f "$MIX_WAV" ]]; then
    echo "⚠️  Mix WAV not found, creating from stems..."
    MIX_WAV="$STEMS_DIR/_auto_Other.wav"
    
    # Mix WAV生成（Vocals除外）
    CMD_MIX=("$PYTHON_BIN" "$REPO_ROOT/ops/create_mix_wav.py" \
             --stems-dir "$STEMS_DIR" \
             --out "$MIX_WAV" \
             --exclude Vocals)
    
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[DRY-RUN] ${CMD_MIX[*]}"
    else
      echo "🎚️  Running: ${CMD_MIX[*]}"
      "${CMD_MIX[@]}" || {
        echo "❌ Failed to create Mix WAV"
        [[ $STRICT -eq 1 ]] && exit 1
      }
    fi
  fi
  
  echo "   Mix WAV: $MIX_WAV"
fi

# ==========================================
# Vocal WAV自動検出
# ==========================================
if [[ -z "$VOCAL_WAV" ]]; then
  echo "🔍 Auto-detecting Vocal WAV..."
  VOCAL_WAV="$(find "$STEMS_DIR" -iname "*Vocal*.wav" -o -iname "*Vocals*.wav" -o -iname "*VOX*.wav" | head -n1)"
  
  if [[ -z "$VOCAL_WAV" ]] || [[ ! -f "$VOCAL_WAV" ]]; then
    echo "⚠️  Vocal WAV not found (skipping lyric anchors)"
    VOCAL_WAV=""
  else
    echo "   Vocal WAV: $VOCAL_WAV"
  fi
fi

# ==========================================
# 出力先ディレクトリ準備
# ==========================================
ANALYSIS_DIR="$SONG_ROOT/analysis"
mkdir -p "$ANALYSIS_DIR"

# ==========================================
# STEP 1: tempo_map.json + bars.parquet
# ==========================================
echo ""
echo "🕐 STEP 1/4: tempo_map.json + bars.parquet"

STEP1_OUT_JSON="$ANALYSIS_DIR/tempo_map.json"
STEP1_OUT_BARS="$ANALYSIS_DIR/bars.parquet"

# 曲の長さを取得（soundfile経由）
DURATION_SEC=$("$PYTHON_BIN" -c "
import soundfile as sf
with sf.SoundFile('$MIX_WAV') as f:
    print(f.frames / f.samplerate)
")

# tempo_map_cli.py を新規生成モードまたはマージモードで実行
if [[ -f "$STEP1_OUT_BARS" ]]; then
  # マージモード: 既存 bars の編集列を保持
  echo "   📋 Merge mode: Updating existing bars.parquet..."
  CMD1=("$PYTHON_BIN" "$REPO_ROOT/ops/tempo_map_cli.py" \
        --audio "$MIX_WAV" \
        --bars-in "$STEP1_OUT_BARS" \
        --out-bars "${STEP1_OUT_BARS}.tmp" \
        --out-tempo "$STEP1_OUT_JSON" \
        --bpb 4 \
        --prefer-madmom)
else
  # 新規生成モード: ダミー不要
  echo "   🆕 New generation mode: Creating bars.parquet from scratch..."
  CMD1=("$PYTHON_BIN" "$REPO_ROOT/ops/tempo_map_cli.py" \
        --audio "$MIX_WAV" \
        --duration-sec "$DURATION_SEC" \
        --out-bars "${STEP1_OUT_BARS}.tmp" \
        --out-tempo "$STEP1_OUT_JSON" \
        --bpb 4 \
        --prefer-madmom)
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] ${CMD1[*]}"
else
  echo "   Running: ${CMD1[*]}"
  "${CMD1[@]}" || {
    echo "❌ STEP 1 failed"
    [[ $STRICT -eq 1 ]] && exit 1
  }
  # 安全に置き換え（atomic operation）
  mv -f "${STEP1_OUT_BARS}.tmp" "$STEP1_OUT_BARS"
fi

# ==========================================
# STEP 1.5: bars.parquet拡張（start_sec/end_sec/density_target/swing_target追加）
# ==========================================
echo ""
echo "🕐.5 STEP 1.5/5: bars.parquet拡張（完全版生成）"

# tempo_map.jsonから中央値BPM取得
MEDIAN_BPM=$("$PYTHON_BIN" -c "
import json
from pathlib import Path
tempo_map = json.loads(Path('$STEP1_OUT_JSON').read_text())
# tempo_map format: {'tempo_points': [[time_sec, bpm], ...]}
if isinstance(tempo_map, dict) and 'tempo_points' in tempo_map:
    tempo_points = tempo_map['tempo_points']
    # tempo_points is list of [time_sec, bpm]
    tempos = [p[1] for p in tempo_points if isinstance(p, list) and len(p) >= 2]
elif isinstance(tempo_map, list):
    # Legacy format: list of dicts
    tempos = [p.get('bpm', 120.0) for p in tempo_map if isinstance(p, dict)]
else:
    tempos = []
import statistics
print(statistics.median(tempos) if tempos else 120.0)
")

echo "   Median BPM: $MEDIAN_BPM"

# start_sec/end_sec/density_target/swing_target追加
"$PYTHON_BIN" -c "
import pandas as pd
import numpy as np
from pathlib import Path
import json

# bars.parquet読み込み
bars = pd.read_parquet('$STEP1_OUT_BARS')

# tempo_map.jsonからBPM配列を取得
tempo_map_path = Path('$STEP1_OUT_JSON')
if tempo_map_path.exists():
    tempo_map = json.loads(tempo_map_path.read_text())
    
    # tempo_points形式: [[time_sec, bpm], ...]
    if isinstance(tempo_map, dict) and 'tempo_points' in tempo_map:
        tempo_points = tempo_map['tempo_points']
        
        # 各小節にBPMを割り当て（start_secに最も近いtempo_pointを使用）
        if 'start_sec' in bars.columns:
            tempo_bpms = []
            for _, row in bars.iterrows():
                t = float(row['start_sec'])
                # t以前の最も近いtempo_pointを探す
                valid_points = [p for p in tempo_points if isinstance(p, list) and len(p) >= 2 and p[0] <= t]
                if valid_points:
                    bpm = valid_points[-1][1]  # 最後（最も近い）のBPM
                else:
                    # t以前のポイントが無い場合、最初のBPM
                    bpm = tempo_points[0][1] if tempo_points else float('$MEDIAN_BPM')
                tempo_bpms.append(float(bpm))
            
            bars['tempo_bpm'] = tempo_bpms
            print(f'   Added tempo_bpm from tempo_map.json (median={np.median(tempo_bpms):.2f})')
        else:
            # start_sec未生成時のフォールバック
            bars['tempo_bpm'] = float('$MEDIAN_BPM')
            print(f'   Added tempo_bpm (constant={float(\"$MEDIAN_BPM\"):.2f})')
    else:
        # Legacy形式またはtempo_points無し
        bars['tempo_bpm'] = float('$MEDIAN_BPM')
        print(f'   Added tempo_bpm (constant={float(\"$MEDIAN_BPM\"):.2f})')
else:
    # tempo_map.json無し
    bars['tempo_bpm'] = float('$MEDIAN_BPM')
    print(f'   Added tempo_bpm (constant={float(\"$MEDIAN_BPM\"):.2f}, no tempo_map.json)')

# time_signature追加（デフォルト4/4、後で更新可能）
if 'time_signature' not in bars.columns:
    bars['time_signature'] = '4/4'

# start_sec/end_secが無い場合は計算
if 'start_sec' not in bars.columns or 'end_sec' not in bars.columns:
    bpm = float('$MEDIAN_BPM')
    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * 4
    
    bars['start_sec'] = bars.index * bar_sec
    bars['end_sec'] = (bars.index + 1) * bar_sec
    print(f'   Added start_sec/end_sec (BPM={bpm:.1f})')

# start_beat/end_beatが無い場合は計算
if 'start_beat' not in bars.columns:
    bars['start_beat'] = bars.index * 4.0
if 'end_beat' not in bars.columns:
    bars['end_beat'] = (bars.index + 1) * 4.0

# density_target/swing_target追加（デフォルト値、後でsections.jsonから更新）
if 'density_target' not in bars.columns:
    bars['density_target'] = 0.7  # デフォルト: 標準密度
if 'swing_target' not in bars.columns:
    bars['swing_target'] = 0.0    # デフォルト: ストレート

# 保存
bars.to_parquet('$STEP1_OUT_BARS', index=False)
print(f'✅ Extended bars.parquet: {len(bars)} bars')
print(f'   Columns: {list(bars.columns)}')
" || {
    echo "❌ Failed to extend bars.parquet"
    [[ $STRICT -eq 1 ]] && exit 1
}

# ==========================================
# STEP 2: sections.json
# ==========================================
echo ""
echo "🕑 STEP 2/5: sections.json"

STEP2_OUT_JSON="$ANALYSIS_DIR/sections.json"

CMD2=("$PYTHON_BIN" "$REPO_ROOT/ops/sections_from_audio.py" \
      --stems "$STEMS_DIR" \
      --out "$STEP2_OUT_JSON")

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] ${CMD2[*]}"
else
  echo "   Running: ${CMD2[*]}"
  "${CMD2[@]}" || {
    echo "❌ STEP 2 failed"
    [[ $STRICT -eq 1 ]] && exit 1
  }
fi

# ==========================================
# STEP 2.5: density_target/swing_target/section_label更新（sections.jsonベース）
# ==========================================
echo ""
echo "🕑.5 STEP 2.5/5: density_target/swing_target/section_label更新"

"$PYTHON_BIN" -c "
import pandas as pd
import json
from pathlib import Path

# bars.parquet読み込み
bars = pd.read_parquet('$STEP1_OUT_BARS')

# sections.json読み込み
sections_data = json.loads(Path('$STEP2_OUT_JSON').read_text())
sections = sections_data.get('sections', [])

# セクション別デフォルト値
section_defaults = {
    'intro': {'density': 0.5, 'swing': 0.0},
    'verse': {'density': 0.6, 'swing': 0.0},
    'chorus': {'density': 0.9, 'swing': 0.0},
    'bridge': {'density': 0.7, 'swing': 0.1},
    'outro': {'density': 0.4, 'swing': 0.0},
    'pre_chorus': {'density': 0.75, 'swing': 0.0},
    'break': {'density': 0.3, 'swing': 0.0},
}

# section_label初期化（デフォルトは'verse'）
bars['section_label'] = 'verse'

# セクション情報でdensity_target/swing_target/section_label更新
for sec in sections:
    start_bar = sec.get('start_bar', 0)
    end_bar = sec.get('end_bar', len(bars))
    label = sec.get('label', 'verse').lower()
    
    # デフォルト値取得（未定義セクションはverse扱い）
    defaults = section_defaults.get(label, section_defaults['verse'])
    
    # 該当バーに適用
    mask = (bars.index >= start_bar) & (bars.index <= end_bar)
    bars.loc[mask, 'density_target'] = defaults['density']
    bars.loc[mask, 'swing_target'] = defaults['swing']
    bars.loc[mask, 'section_label'] = label

# 保存
bars.to_parquet('$STEP1_OUT_BARS', index=False)
print(f'✅ Updated density_target/swing_target/section_label for {len(sections)} sections')

# セクション別統計
for sec in sections:
    start_bar = sec.get('start_bar', 0)
    end_bar = sec.get('end_bar', len(bars))
    label = sec.get('label', 'verse')
    mask = (bars.index >= start_bar) & (bars.index <= end_bar)
    avg_density = bars.loc[mask, 'density_target'].mean()
    print(f'   {label:<12} bars {start_bar:>3}-{end_bar:<3}: density={avg_density:.2f}')
" || {
    echo "❌ Failed to update density_target/swing_target/section_label"
    [[ $STRICT -eq 1 ]] && exit 1
}

# ==========================================
# STEP 3: lyric_anchors.json
# ==========================================
echo ""
echo "🕒 STEP 3/5: lyric_anchors.json"

STEP3_OUT_JSON="$ANALYSIS_DIR/lyric_anchors.json"

if [[ -n "$VOCAL_WAV" ]] && [[ -f "$VOCAL_WAV" ]]; then
  CMD3=("$PYTHON_BIN" "$REPO_ROOT/ops/anchors_from_vocal.py" \
        --vocal "$VOCAL_WAV" \
        --bars "$STEP1_OUT_BARS" \
        --tempo-map "$STEP1_OUT_JSON" \
        --out "$STEP3_OUT_JSON")
  
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${CMD3[*]}"
  else
    echo "   Running: ${CMD3[*]}"
    "${CMD3[@]}" || {
      echo "❌ STEP 3 failed"
      [[ $STRICT -eq 1 ]] && exit 1
    }
  fi
else
  echo "   ⏭️  Skipped (Vocal WAV not found)"
  # ダミーlyric_anchors.json生成
  echo '[]' > "$STEP3_OUT_JSON"
fi

# ==========================================
# STEP 4: analysis/chordmap.json
# ==========================================
echo ""
echo "🕓 STEP 4/5: analysis/chordmap.json"

STEP4_OUT_JSON="$ANALYSIS_DIR/chordmap.json"

# --- Choose mix audio deterministically for chord recognition ---
MIX_AUDIO_CANDIDATES=(
  "$STEMS_DIR/instrument.wav"
  "$STEMS_DIR/mix.wav"
  "$STEMS_DIR/Mix.wav"
  "$STEMS_DIR/other.wav"
  "$STEMS_DIR/Other.wav"
)
CHOSEN_MIX_AUDIO=""
for f in "${MIX_AUDIO_CANDIDATES[@]}"; do
  if [[ -f "$f" ]]; then
    CHOSEN_MIX_AUDIO="$f"
    break
  fi
done
if [[ -z "$CHOSEN_MIX_AUDIO" ]]; then
  echo "❌ STEP 4: mix audio not found in $STEMS_DIR (instrument/mix/other)."
  [[ $STRICT -eq 1 ]] && exit 1
fi
echo "   🎧 Chord recognition audio: ${CHOSEN_MIX_AUDIO}"

CMD4=("$PYTHON_BIN" "$REPO_ROOT/ops/stem_harmony_bar_level.py" \
      --stems "$STEMS_DIR" \
      --audio "$CHOSEN_MIX_AUDIO" \
      --bars "$STEP1_OUT_BARS" \
      --out "$STEP4_OUT_JSON")

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] ${CMD4[*]}"
else
  echo "   Running: ${CMD4[*]}"
  "${CMD4[@]}" || {
    echo "❌ STEP 4 failed"
    [[ $STRICT -eq 1 ]] && exit 1
  }
fi

# ==========================================
# STEP 5: stems_features.parquet（オプション: QA/可視化用途のみ）
# ==========================================
echo ""
echo "🕔 STEP 5/5: stems_features.parquet (optional, STEM_FEATURES=$STEM_FEATURES)"

# ファイル名統一: stems_features.parquet を正式名称とする
STEP5_OUT_FEATURES="$ANALYSIS_DIR/stems_features.parquet"

if [[ "$STEM_FEATURES" == "1" ]]; then
  echo "   ℹ️  Generating stem features (STEM_FEATURES=1)"
  
  # lyric_anchors.jsonが存在する場合のみ--anchors指定
  ANCHORS_ARG=()
  if [[ -f "$STEP3_OUT_JSON" ]]; then
    ANCHORS_ARG=("--anchors" "$STEP3_OUT_JSON")
  fi

  CMD5=("$PYTHON_BIN" "$REPO_ROOT/ops/stems_features.py" \
        --stems "$STEMS_DIR" \
        --bars "$STEP1_OUT_BARS" \
        "${ANCHORS_ARG[@]}" \
        --output "$STEP5_OUT_FEATURES" \
        --tempo-bpm "$MEDIAN_BPM" \
        --inst-activity)

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${CMD5[*]}"
  else
    echo "   Running: ${CMD5[*]}"
    "${CMD5[@]}" || {
      echo "❌ STEP 5 failed"
      [[ $STRICT -eq 1 ]] && exit 1
    }
  fi

  # stems_features.parquetからすべての有用なカラムをbars.parquetにマージ（完全版生成）
  echo "   Merging stems_features to bars.parquet (完全版生成)..."
  "$PYTHON_BIN" -c "
import pandas as pd

# bars.parquet読み込み
bars = pd.read_parquet('$STEP1_OUT_BARS')

# stems_features.parquet読み込み
features = pd.read_parquet('$STEP5_OUT_FEATURES')

# マージ対象カラム（bar列以外のすべて）
merge_columns = [
    'drums_active',      # ドラムアクティブ判定（必須）
    'energy_curve',      # エネルギーカーブ（必須）
    'hat_density',       # ハイハット密度
    'kick_peak_db',      # キックピーク強度
    'snare_backbeat',    # スネアバックビート
    'fill_likelihood',   # Fill確率
    'loudness_db',       # ラウドネス
    'vocal_stress',      # Vocalストレス
    'guitar_activity',   # ギターアクティビティ
    'piano_activity',    # ピアノアクティビティ
    'strings_activity',  # ストリングスアクティビティ
]

# 存在するカラムのみマージ
merged_count = 0
for col in merge_columns:
    if col in features.columns:
        col_map = features.set_index('bar')[col].to_dict()
        default_value = 0.0 if 'activity' in col or col == 'drums_active' else 0.5
        bars[col] = bars.index.map(lambda x: col_map.get(x, default_value))
        merged_count += 1

print(f'✅ Merged {merged_count} columns from stem_features.parquet to bars.parquet')

# マージされたカラムの統計
if 'drums_active' in bars.columns:
    print(f'   drums_active: {int(bars[\"drums_active\"].sum())} active / {len(bars)} total bars')
if 'energy_curve' in bars.columns:
    print(f'   energy_curve: {bars[\"energy_curve\"].min():.3f} - {bars[\"energy_curve\"].max():.3f}')
if 'guitar_activity' in bars.columns:
    active_bars = int((bars['guitar_activity'] > 0.1).sum())
    print(f'   guitar_activity: {active_bars} active bars (mean={bars[\"guitar_activity\"].mean():.3f})')
if 'piano_activity' in bars.columns:
    active_bars = int((bars['piano_activity'] > 0.1).sum())
    print(f'   piano_activity: {active_bars} active bars (mean={bars[\"piano_activity\"].mean():.3f})')
if 'strings_activity' in bars.columns:
    active_bars = int((bars['strings_activity'] > 0.1).sum())
    print(f'   strings_activity: {active_bars} active bars (mean={bars[\"strings_activity\"].mean():.3f})')

# 保存
bars.to_parquet('$STEP1_OUT_BARS', index=False)
print(f'✅ bars.parquet完全版保存: {len(bars)} bars, {len(bars.columns)} columns')
" || {
    echo "❌ Failed to merge stem_features"
    [[ $STRICT -eq 1 ]] && exit 1
}

else
  echo "   ℹ️  Skipping stem features generation (STEM_FEATURES=0)"
  echo "   ℹ️  Set STEM_FEATURES=1 to enable for QA/visualization"
fi

# ==========================================
# analysisディレクトリへのコピー（E2E処理用）
# ==========================================
echo ""
echo "📋 Copying files to analysis/ directory..."

# cp -f で強制上書き（exit code 1回避）
cp -f "$STEP1_OUT_BARS" "$ANALYSIS_DIR/bars.parquet" 2>/dev/null || true
cp -f "$STEP2_OUT_JSON" "$ANALYSIS_DIR/sections.json" 2>/dev/null || true
cp -f "$STEP1_OUT_JSON" "$ANALYSIS_DIR/tempo_map.json" 2>/dev/null || true

if [[ -f "$STEP3_OUT_JSON" ]]; then
  cp -f "$STEP3_OUT_JSON" "$ANALYSIS_DIR/lyric_anchors.json" 2>/dev/null || true
fi

if [[ -f "$STEP5_OUT_FEATURES" ]]; then
  cp -f "$STEP5_OUT_FEATURES" "$ANALYSIS_DIR/stems_features.parquet" 2>/dev/null || true
  # 互換性のためstem_features.parquetへのシンボリックリンク作成
  (
    cd "$ANALYSIS_DIR" && ln -sf stems_features.parquet stem_features.parquet
  )
  echo "   ✅ stems_features.parquet (main) + stem_features.parquet (symlink)"
fi

echo "✅ Files copied to analysis/"

# ==========================================
# STEP 6: fill_slot / riff_slot 追加（フィル/リフ発火システム）
# ==========================================
echo ""
echo "🎯 Step 6: Adding fill/riff slots to bars.parquet"

BARS_WITH_SLOTS="$ANALYSIS_DIR/bars_with_slots.parquet"

if [[ -f "$REPO_ROOT/scripts/add_fill_riff_slots.py" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/add_fill_riff_slots.py" \
        --bars "$ANALYSIS_DIR/bars.parquet" \
        --sections "$ANALYSIS_DIR/sections.json" \
        --out "$BARS_WITH_SLOTS" \
        --energy-jump-thresh 0.06 \
        --fill-likelihood-thresh 0.15 \
        --boundary-fill always \
        --riff-sections pre_chorus chorus bridge \
        --min-riff-activity 0.2 || {
        echo "⚠️  Fill/riff slot addition failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    
    # bars.parquet を bars_with_slots.parquet に置き換え
    if [[ -f "$BARS_WITH_SLOTS" ]]; then
        cp "$ANALYSIS_DIR/bars.parquet" "$ANALYSIS_DIR/bars_original.parquet"
        cp "$BARS_WITH_SLOTS" "$ANALYSIS_DIR/bars.parquet"
        echo "   ✅ bars.parquet updated with fill_slot/riff_slot"
        echo "   📁 Original backed up to bars_original.parquet"
    fi
else
    echo "⚠️  scripts/add_fill_riff_slots.py not found, skipping"
fi

# ==========================================
# STEP 7: CREPE/OaF用シンボリックリンク作成（廃止: 原音stem直参照に変更）
# ==========================================
echo ""
echo "🔗 Step 7: (DEPRECATED) Creating symbolic links for CREPE/OaF"
echo "   ℹ️  Symlinks no longer required; ops scripts now reference original stems directly"

# 互換性のため、既存のシンボリックリンクがあれば削除しない
# 新規作成は行わない

# ==========================================
# STEP 8: Stem features → role_bars & stems_features（廃止）
# ==========================================
echo ""
echo "🎵 Step 8: (REMOVED) Extracting stem features..."
echo "   ℹ️  This step has been removed. Use STEM_FEATURES=1 in STEP 5 if needed for QA"

# ==========================================
# STEP 9: Vocal features → bars追記 & energy/valence保証
# ==========================================
echo ""
echo "🎤 Step 9: Extracting vocal features..."

# Vocal WAV自動検出（未指定の場合）
if [[ -z "$VOCAL_WAV" ]]; then
    VOCAL_WAV=$(find "$STEMS_DIR" -name "*Vocals*.wav" -o -name "*vocals*.wav" -o -name "*vocal*.wav" 2>/dev/null | head -1)
fi

if [[ -n "$VOCAL_WAV" ]] && [[ -f "$SCRIPT_DIR/extract_vocal_features.py" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/extract_vocal_features.py" \
        --audio "$VOCAL_WAV" \
        --bars "$ANALYSIS_DIR/bars.parquet" \
        --out "$ANALYSIS_DIR/vocal_features.parquet" \
        --merge-into-bars "$ANALYSIS_DIR/bars.parquet" || {
        echo "⚠️  extract_vocal_features.py failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "✅ Vocal features extracted & merged into bars.parquet"
elif [[ -f "$SCRIPT_DIR/post_analysis_merge.py" ]]; then
    # Vocal WAV無い場合：energy/valence保証のみ実行
    echo "⚠️  No vocal stem found, running post_analysis_merge.py for energy/valence guarantee"
    "$PYTHON_BIN" "$SCRIPT_DIR/post_analysis_merge.py" "$ANALYSIS_DIR/bars.parquet" || {
        echo "⚠️  post_analysis_merge.py failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "✅ energy/valence guaranteed in bars.parquet"
else
    echo "⚠️  Neither vocal stem nor post_analysis_merge.py found, skipping"
fi

# ==========================================
# STEP 9.5: energy/valence付与（bars.parquet 23列化）
# ==========================================
echo ""
echo "🎵 Step 9.5: Adding energy/valence columns (23-column bars.parquet)"

"$PYTHON_BIN" -c "
import pandas as pd
import json
from pathlib import Path
import numpy as np

# bars.parquet読み込み
bars_path = Path('$ANALYSIS_DIR/bars.parquet')
bars = pd.read_parquet(bars_path)

# energy列がまだない場合のみ追加
if 'energy' not in bars.columns:
    # sections.jsonからenergy_curve転写を試みる
    sections_path = Path('$ANALYSIS_DIR/sections.json')
    if sections_path.exists():
        sections_data = json.loads(sections_path.read_text())
        
        # sections.jsonのメタデータからenergy取得
        energy_meta = sections_data.get('meta', {}).get('energy', [])
        if energy_meta:
            # [[bar_index, energy], ...] 形式
            energy_map = {int(b): float(e) for b, e in energy_meta}
            bars['energy'] = bars['bar_index'].map(lambda x: energy_map.get(x, 0.5))
            print('   energy列追加（sections.jsonメタから転写）')
        else:
            # energy_curveがあればそれを使用
            if 'energy_curve' in bars.columns:
                bars['energy'] = bars['energy_curve'].fillna(0.5)
                print('   energy列追加（energy_curve列から転写）')
            # loudness_dbからmin-max正規化
            elif 'loudness_db' in bars.columns:
                ld = bars['loudness_db'].values.astype(float)
                e = (ld - np.nanmin(ld)) / max(1e-6, (np.nanmax(ld) - np.nanmin(ld)))
                bars['energy'] = np.clip(e, 0.0, 1.0)
                print('   energy列追加（loudness_dbから正規化）')
            else:
                bars['energy'] = 0.5
                print('   energy列追加（デフォルト0.5）')
    else:
        # sections.jsonなし：loudness_dbからmin-max正規化
        if 'loudness_db' in bars.columns:
            ld = bars['loudness_db'].values.astype(float)
            e = (ld - np.nanmin(ld)) / max(1e-6, (np.nanmax(ld) - np.nanmin(ld)))
            bars['energy'] = np.clip(e, 0.0, 1.0)
            print('   energy列追加（loudness_dbから正規化）')
        else:
            bars['energy'] = 0.5
            print('   energy列追加（デフォルト0.5）')
else:
    print('   energy列は既に存在します')

# valence列がまだない場合のみ追加
if 'valence' not in bars.columns:
    # TODO: 和声推定からvalence計算（暫定0.5）
    bars['valence'] = 0.5
    print('   valence列追加（暫定0.5、後で和声推定に差し替え）')
else:
    print('   valence列は既に存在します')

# 保存
bars.to_parquet(bars_path, index=False)

print(f'✅ bars.parquet 23列化完了: {len(bars)} bars, {len(bars.columns)} columns')
print(f'   energy範囲: {bars[\"energy\"].min():.3f} - {bars[\"energy\"].max():.3f}')
print(f'   valence: 0.5 (暫定)')
" || {
    echo "⚠️  energy/valence付与に失敗しました"
    [[ $STRICT -eq 1 ]] && exit 1
}

# ==========================================
# STEP 10: CREPE連続F0生成（呼吸するparquet）
# ==========================================
echo ""
echo "🎤 Step 10: CREPE F0 extraction (continuous pitch)"

VOCALS_WAV="$SONG_ROOT/audio/vocals.wav"
VOCAL_F0_PARQUET="$SONG_ROOT/features/vocal_f0.parquet"

if [[ ! -f "$VOCALS_WAV" ]] && [[ -n "$VOCAL_WAV" ]]; then
    # vocal.wavからコピー
    mkdir -p "$SONG_ROOT/audio"
    cp "$VOCAL_WAV" "$VOCALS_WAV"
    echo "   Copied vocal.wav -> audio/vocals.wav"
fi

if [[ -f "$VOCALS_WAV" ]]; then
    if [[ -f "$REPO_ROOT/scripts/CREPE/create_vocal_f0_parquet.py" ]]; then
        "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/create_vocal_f0_parquet.py" \
            --audio "$VOCALS_WAV" \
            --out "$VOCAL_F0_PARQUET" || {
            echo "⚠️  CREPE F0 extraction failed"
            [[ $STRICT -eq 1 ]] && exit 1
        }
        echo "✅ CREPE F0 extracted: $VOCAL_F0_PARQUET"
    else
        echo "⚠️  CREPE script not found, skipping F0 extraction"
    fi
else
    echo "⚠️  vocals.wav not found, skipping CREPE F0 extraction"
fi

# ==========================================
# STEP 11: CREPE系plan再計画（Strings/Guitar/Piano/Synth）
# ==========================================
echo ""
echo "🎵 Step 11: CREPE-based plan regeneration"

PLANS_DIR="$SONG_ROOT/plans"
FEATURES_DIR="$SONG_ROOT/features"
mkdir -p "$PLANS_DIR"

# 必要なファイル確認
SECTIONS_JSON="$ANALYSIS_DIR/sections.json"
CHORDMAP_JSON="$ANALYSIS_DIR/chordmap.json"
ANCHORS_JSON="$ANALYSIS_DIR/lyric_anchors.json"
VOCAL_EVENTS="$FEATURES_DIR/vocal_features.parquet"

# Strings counter-melody from CREPE
if [[ -f "$VOCAL_F0_PARQUET" ]] && [[ -f "$REPO_ROOT/scripts/CREPE/generate_strings_countermelody.py" ]]; then
    echo "=== Strings counter-melody from CREPE ==="
    "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/generate_strings_countermelody.py" \
        --vocal-f0 "$VOCAL_F0_PARQUET" \
        --sections "$SECTIONS_JSON" \
        --chordmap "$CHORDMAP_JSON" \
        --policy "$REPO_ROOT/scripts/CREPE/policy/strings_countermelody.yaml" \
        --out "$PLANS_DIR/strings_countermelody_plan.json" || {
        echo "⚠️  Strings counter-melody generation failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    
    # Strings VoiceLeading enhancement
    if [[ -f "$REPO_ROOT/scripts/CREPE/strings_voiceleading_enhancer.py" ]]; then
        "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/strings_voiceleading_enhancer.py" \
            --strings-plan "$PLANS_DIR/strings_countermelody_plan.json" \
            --chordmap "$CHORDMAP_JSON" \
            --out "$PLANS_DIR/strings_countermelody_plan_vl.json" \
            --kpi-csv "$ANALYSIS_DIR/strings_vl_kpi.csv" || {
            echo "⚠️  Strings VoiceLeading enhancement failed"
            [[ $STRICT -eq 1 ]] && exit 1
        }
        echo "✅ Strings VoiceLeading enhanced"
    fi
fi

# Guitar duration optimize & microtiming
if [[ -f "$PLANS_DIR/guitar_plan.json" ]] && [[ -f "$VOCAL_F0_PARQUET" ]]; then
    if [[ -f "$REPO_ROOT/scripts/CREPE/optimize_guitar_duration.py" ]]; then
        echo "=== Guitar duration optimize & microtiming ==="
        "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/optimize_guitar_duration.py" \
            --guitar-plan "$PLANS_DIR/guitar_plan.json" \
            --vocal-f0 "$VOCAL_F0_PARQUET" \
            --vocal-events "$VOCAL_EVENTS" \
            --sections "$SECTIONS_JSON" \
            --policy "$REPO_ROOT/scripts/CREPE/policy/guitar_duration_policy.yaml" \
            --out "$PLANS_DIR/guitar_plan_optimized.json" \
            --csv "$ANALYSIS_DIR/guitar_duration_changes.csv" || {
            echo "⚠️  Guitar duration optimization failed"
            [[ $STRICT -eq 1 ]] && exit 1
        }
        
        if [[ -f "$REPO_ROOT/scripts/CREPE/guitar_microtiming_apply.py" ]]; then
            "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/guitar_microtiming_apply.py" \
                --guitar-plan "$PLANS_DIR/guitar_plan_optimized.json" \
                --sections "$SECTIONS_JSON" \
                --policy "$REPO_ROOT/scripts/CREPE/policy/guitar_microtiming.yaml" \
                --out "$PLANS_DIR/guitar_plan_optimized_micro.json" \
                --csv "$ANALYSIS_DIR/guitar_microtiming.csv" || {
                echo "⚠️  Guitar microtiming failed"
                [[ $STRICT -eq 1 ]] && exit 1
            }
            echo "✅ Guitar plan optimized & microtiming applied"
        fi
    fi
fi

# Piano Hybrid generation
if [[ -f "$VOCAL_F0_PARQUET" ]] && [[ -f "$REPO_ROOT/scripts/CREPE/piano_hybrid_generator.py" ]]; then
    echo "=== Piano Hybrid (点×線) generation ==="
    "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/piano_hybrid_generator.py" \
        --vocal-f0 "$VOCAL_F0_PARQUET" \
        --vocal-events "$VOCAL_EVENTS" \
        --sections "$SECTIONS_JSON" \
        --policy "$REPO_ROOT/scripts/CREPE/policy/piano_hybrid_policy.yaml" \
        --out "$PLANS_DIR/piano_plan_hybrid.json" || {
        echo "⚠️  Piano Hybrid generation failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "✅ Piano Hybrid plan generated"
fi

# Synth Pad automation
if [[ -f "$VOCAL_F0_PARQUET" ]] && [[ -f "$REPO_ROOT/scripts/CREPE/synth_pad_automation.py" ]]; then
    echo "=== Synth Pad automation ==="
    "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/synth_pad_automation.py" \
        --vocal-f0 "$VOCAL_F0_PARQUET" \
        --sections "$SECTIONS_JSON" \
        --policy "$REPO_ROOT/scripts/CREPE/policy/synth_pad_policy.yaml" \
        --out "$ANALYSIS_DIR/pad_automation.json" || {
        echo "⚠️  Synth Pad automation failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "✅ Synth Pad automation generated"
fi

# ==========================================
# ==========================================
# STEP 12: 補助ファイル生成（無効化）
# ==========================================
# 🔕 補助ファイル（drum_accent_plan, bassline_plan, voicings_guide, style_presets）の
#    生成は無効化。これらは「ミュート化」「和声単純化」「セクション無視」を誘発するため、
#    生成経路から除外。QA/可視化用途のみに限定。
#
# 理由:
#   - 上流は「和声と拍位置に忠実な、常時オンのベースライン」に徹する
#   - DAW 側で最終的な音の組合せを決定（Magenta 等の生成を後段で重ねる）
#   - 将来の「演奏法AI」が成熟したら完全廃止予定
#
# 参照: ChatGPT guidance (2025-11-11)
#   「補助ファイルは生成経路では廃止（無効化）してOK。
#    役割をQA/可視化限定に縮退。」
# ==========================================
echo ""
echo "🔕 Step 12: Auxiliary file generation (DISABLED)"
echo "   補助ファイル生成はスキップ（QA/可視化用途に限定）"
echo "   - drum_accent_plan.json: スキップ"
echo "   - bassline_plan.csv: スキップ"
echo "   - voicings_guide.csv: スキップ"
echo "   - style_presets.yaml: スキップ"
echo "   理由: ミュート化・和声単純化を防ぐため生成経路から除外"

# ==========================================
# STEP 13: 可変テンポ対応MIDI統合
# ==========================================
echo ""
echo "🎹 Step 13: Variable tempo MIDI integration"

MIDI_DIR="$SONG_ROOT/midi"
mkdir -p "$MIDI_DIR"

TEMPO_MAP="$ANALYSIS_DIR/tempo_map.json"
SONG_ID=$(basename "$SONG_ROOT")

echo ""
echo "🎵 Step 13.5: CREPE MIDI generation"
echo "   ⚠️  PHASE-A-BLOCKED: CREPE→MIDI統合はPhase B（RUN_SONG_004.sh STEP 22）で実施"
echo "   （Phase Aでは CREPE F0抽出のみ、MIDI化は未確定和声LockベースのPhase Bで実施）"

# PHASE-A-BLOCKED: if [[ -f "$REPO_ROOT/scripts/CREPE/plan_to_midi.py" ]]; then
# PHASE-A-BLOCKED:     # 統合対象planを収集
# PHASE-A-BLOCKED:     TRACK_PLANS=()
# PHASE-A-BLOCKED:     [[ -f "$PLANS_DIR/piano_plan_hybrid.json" ]] && TRACK_PLANS+=("$PLANS_DIR/piano_plan_hybrid.json")
# PHASE-A-BLOCKED:     [[ -f "$PLANS_DIR/strings_countermelody_plan_vl.json" ]] && TRACK_PLANS+=("$PLANS_DIR/strings_countermelody_plan_vl.json")
# PHASE-A-BLOCKED:     [[ -f "$PLANS_DIR/guitar_plan_optimized_micro.json" ]] && TRACK_PLANS+=("$PLANS_DIR/guitar_plan_optimized_micro.json")
# PHASE-A-BLOCKED:     
# PHASE-A-BLOCKED:     if [[ ${#TRACK_PLANS[@]} -gt 0 ]]; then
# PHASE-A-BLOCKED:         # 各planを個別MIDI化
# PHASE-A-BLOCKED:         for plan in "${TRACK_PLANS[@]}"; do
# PHASE-A-BLOCKED:             plan_name=$(basename "$plan" .json)
# PHASE-A-BLOCKED:             midi_out="$MIDI_DIR/${plan_name}.mid"
# PHASE-A-BLOCKED:             
# PHASE-A-BLOCKED:             "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/plan_to_midi.py" \
# PHASE-A-BLOCKED:                 "$plan" \
# PHASE-A-BLOCKED:                 "$midi_out" \
# PHASE-A-BLOCKED:                 --bpm "$MEDIAN_BPM" || {
# PHASE-A-BLOCKED:                 echo "⚠️  MIDI conversion failed: $plan_name"
# PHASE-A-BLOCKED:                 [[ $STRICT -eq 1 ]] && exit 1
# PHASE-A-BLOCKED:             }
# PHASE-A-BLOCKED:             echo "   ✅ $plan_name.mid"
# PHASE-A-BLOCKED:         done
# PHASE-A-BLOCKED:         
# PHASE-A-BLOCKED:         # 統合MIDI生成
# PHASE-A-BLOCKED:         if [[ -f "$REPO_ROOT/scripts/CREPE/merge_crepe_midis.py" ]]; then
# PHASE-A-BLOCKED:             MERGE_ARGS=("--output" "$MIDI_DIR/${SONG_ID}_hybrid_crepe.mid" --bpm" "$MEDIAN_BPM")
# PHASE-A-BLOCKED:             [[ -f "$MIDI_DIR/piano_plan_hybrid.mid" ]] && MERGE_ARGS+=("--piano" "$MIDI_DIR/piano_plan_hybrid.mid")
# PHASE-A-BLOCKED:             [[ -f "$MIDI_DIR/strings_countermelody_plan_vl.mid" ]] && MERGE_ARGS+=("--strings" "$MIDI_DIR/strings_countermelody_plan_vl.mid")
# PHASE-A-BLOCKED:             [[ -f "$MIDI_DIR/guitar_plan_optimized_micro.mid" ]] && MERGE_ARGS+=("--guitar" "$MIDI_DIR/guitar_plan_optimized_micro.mid")
# PHASE-A-BLOCKED:             
# PHASE-A-BLOCKED:             "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/merge_crepe_midis.py" "${MERGE_ARGS[@]}" || {
# PHASE-A-BLOCKED:                 echo "⚠️  MIDI merge failed"
# PHASE-A-BLOCKED:                 [[ $STRICT -eq 1 ]] && exit 1
# PHASE-A-BLOCKED:             }
# PHASE-A-BLOCKED:             echo "✅ Integrated MIDI: ${SONG_ID}_hybrid_crepe.mid"
# PHASE-A-BLOCKED:         fi
# PHASE-A-BLOCKED:     else
# PHASE-A-BLOCKED:         echo "⚠️  No CREPE plans found for MIDI conversion"
# PHASE-A-BLOCKED:     fi
# PHASE-A-BLOCKED: fi

# ==========================================
# STEP 14: deep_harmony_audit（Phase B STEP 21へ移設）
# ==========================================
echo ""
echo "🔍 Step 14: (MOVED) Harmony audit"
echo "   ℹ️  Harmony audit has been moved to Phase B (STEP 21) after chordmap_locked"
echo "   ℹ️  This ensures audit against finalized, manually-confirmed harmony"

# ==========================================
# STEP 15: Songpackage生成（3 variant）
# ==========================================
echo ""
echo "📦 Step 15: Songpackage generation (3 variants)"
echo "   ⚠️  PHASE-A-BLOCKED: SongPackage生成はPhase B（STEP 17.5 + STEP 21）で実施"
echo "   （未確定和声を基準にしないため、Phase Aでは生成しません）"

# PHASE-A-BLOCKED: if [[ -f "$REPO_ROOT/scripts/generate_suno_song_package_v1_1.py" ]]; then
# PHASE-A-BLOCKED:     for variant in soft standard bright; do
# PHASE-A-BLOCKED:         echo "   Generating ${variant} variant..."
# PHASE-A-BLOCKED:         "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_suno_song_package_v1_1.py" \
# PHASE-A-BLOCKED:             --song-id "$SONG_ID" \
# PHASE-A-BLOCKED:             --analysis-dir "$ANALYSIS_DIR" \
# PHASE-A-BLOCKED:             --variant "$variant" \
# PHASE-A-BLOCKED:             --out "$SONG_ROOT/song_package_${variant}.yaml" || {
# PHASE-A-BLOCKED:             echo "⚠️  Songpackage generation failed: $variant"
# PHASE-A-BLOCKED:             [[ $STRICT -eq 1 ]] && exit 1
# PHASE-A-BLOCKED:         }
# PHASE-A-BLOCKED:         echo "   ✅ song_package_${variant}.yaml"
# PHASE-A-BLOCKED:     done
# PHASE-A-BLOCKED:     echo "✅ All 3 variants generated"
# PHASE-A-BLOCKED: else
# PHASE-A-BLOCKED:     echo "⚠️  generate_suno_song_package_v1_1.py not found"
# PHASE-A-BLOCKED: fi

# ==========================================
# STEP 16: manual_chordmap.json → LOCK（手動編集後の再開ポイント）
# ==========================================
echo ""
echo "📋 Step 16: Chordmap LOCK (manual → locked + QA)"

MANUAL_CHORDMAP="$ANALYSIS_DIR/manual_chordmap.json"
MANUAL_CHORDMAP_ENRICHED="$ANALYSIS_DIR/manual_chordmap_enriched.json"
LOCKED_CHORDMAP="$ANALYSIS_DIR/chordmap_locked.json"
CHORDMAP_QA="$ANALYSIS_DIR/chordmap_qa.csv"

if [[ -f "$MANUAL_CHORDMAP" ]]; then
    echo "   Found manual_chordmap.json, creating LOCK..."
    
    "$PYTHON_BIN" "$REPO_ROOT/scripts/chordmap_lock.py" \
        --base "$STEP4_OUT_JSON" \
        --overrides "$MANUAL_CHORDMAP" \
        --sections "$STEP2_OUT_JSON" \
        --out-json "$LOCKED_CHORDMAP" \
        --out-qa "$CHORDMAP_QA" || {
        echo "❌ Chordmap LOCK failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    
    echo "   ✅ chordmap_locked.json + QA report"
else
    echo "   ⚠️  manual_chordmap.json not found, using auto chordmap as-is"
    cp "$STEP4_OUT_JSON" "$LOCKED_CHORDMAP"
    echo "   (To enable manual editing: create $MANUAL_CHORDMAP and re-run)"
fi

# ==========================================
# STEP 17: music21正規化（LOCK直後）
# ==========================================
echo ""
echo "🎼 Step 17: music21 normalization (chordmap_to_music21)"

M21_CHORDMAP="$ANALYSIS_DIR/chordmap_m21.json"
KEY_HINT="$ANALYSIS_DIR/key_hint.json"

M21_ARGS=("--input" "$LOCKED_CHORDMAP" "--out-json" "$M21_CHORDMAP")
[[ -f "$KEY_HINT" ]] && M21_ARGS+=("--key-hint" "$KEY_HINT")

if [[ -f "$REPO_ROOT/ops/chordmap_to_music21.py" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/ops/chordmap_to_music21.py" "${M21_ARGS[@]}" || {
        echo "⚠️  music21 normalization failed (continuing)"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "   ✅ chordmap_m21.json"
else
    echo "   ⚠️  chordmap_to_music21.py not found, skipping"
fi

# ==========================================
# STEP 18: 楽器別chordmap view生成（Pad/Gt/Pf/Strings/Bass）
# ==========================================
echo ""
echo "🎸 Step 18: Instrument-specific chordmap views"

INSTR_VIEW_TOOL="$REPO_ROOT/scripts/instrument_chordmap/make_instrument_chordmap_views.py"
POLICY_DIR="$REPO_ROOT/scripts/instrument_chordmap/policy"
CHORDMAP_FOR_VIEWS="$LOCKED_CHORDMAP"

if [[ -f "$MANUAL_CHORDMAP_ENRICHED" ]]; then
    CHORDMAP_FOR_VIEWS="$MANUAL_CHORDMAP_ENRICHED"
elif [[ -f "$MANUAL_CHORDMAP" ]]; then
    CHORDMAP_FOR_VIEWS="$MANUAL_CHORDMAP"
fi

if [[ -f "$INSTR_VIEW_TOOL" ]]; then
    if [[ "$CHORDMAP_FOR_VIEWS" != "$LOCKED_CHORDMAP" ]]; then
        echo "   Using $(basename \"$CHORDMAP_FOR_VIEWS\") for instrument chordmap views"
    fi

    VIEW_ARGS=("--chordmap" "$CHORDMAP_FOR_VIEWS" "--sections" "$STEP2_OUT_JSON" "--out-dir" "$ANALYSIS_DIR")
    
    # 各policyファイル存在確認して引数追加
    [[ -f "$POLICY_DIR/chordmap_view_pad.yaml" ]]     && VIEW_ARGS+=("--policy-pad" "$POLICY_DIR/chordmap_view_pad.yaml")
    [[ -f "$POLICY_DIR/chordmap_view_guitar.yaml" ]]  && VIEW_ARGS+=("--policy-guitar" "$POLICY_DIR/chordmap_view_guitar.yaml")
    [[ -f "$POLICY_DIR/chordmap_view_piano.yaml" ]]   && VIEW_ARGS+=("--policy-piano" "$POLICY_DIR/chordmap_view_piano.yaml")
    [[ -f "$POLICY_DIR/chordmap_view_strings.yaml" ]] && VIEW_ARGS+=("--policy-strings" "$POLICY_DIR/chordmap_view_strings.yaml")
    [[ -f "$POLICY_DIR/chordmap_view_bass.yaml" ]]    && VIEW_ARGS+=("--policy-bass" "$POLICY_DIR/chordmap_view_bass.yaml")
    
    "$PYTHON_BIN" "$INSTR_VIEW_TOOL" "${VIEW_ARGS[@]}" || {
        echo "⚠️  Instrument views generation failed (continuing)"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    
    echo "   ✅ chordmap_view_{pad|guitar|piano|strings|bass}.json"
    echo "   ✅ voicings_guide_{instrument}.csv"
else
    echo "   ⚠️  make_instrument_chordmap_views.py not found, skipping"
fi

# ==========================================
# STEP 19: 各楽器plan生成（chordmap_locked.json参照）
# ==========================================
echo ""
echo "🎼 Step 19: Instrument plan generation (using locked chordmap)"

# Define V2 system variables (slot-based rendering)
# POLICY_YAML should live inside the song root (e.g. data/.../song_004/policy/song_004.yaml)
POLICY_YAML="$SONG_ROOT/policy/$(basename "$SONG_ROOT").yaml"
CHORDMAP_EXTENDED="$ANALYSIS_DIR/chordmap_locked_extended.json"

# Bass plan生成（V2優先 → view_bass → fallback）
if [[ -f "$REPO_ROOT/scripts/generate_bass_plan_v2.py" && -f "$BARS_WITH_SLOTS" && -f "$POLICY_YAML" ]]; then
    # V2 (PREFERRED): Slot-based bass renderer
    
    echo "   Generating Bass plan (V2: slot-based)..."
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_bass_plan_v2.py" \
        --bars "$BARS_WITH_SLOTS" \
        --sections "$SECTIONS_JSON" \
        --chordmap "$CHORDMAP_EXTENDED" \
        --policy "$POLICY_YAML" \
        --out "$PLANS_DIR/bass_plan.json" || {
        echo "⚠️  Bass plan V2 generation failed, trying legacy..."
        
        # Fallback to legacy
        if [[ -f "$REPO_ROOT/scripts/bass/generate_bass_plan.py" ]]; then
            BASS_VIEW="$ANALYSIS_DIR/chordmap_view_bass.json"
            BASS_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
            
            if [[ -f "$BASS_VIEW" ]]; then
                echo "   Generating Bass plan (legacy: using view_bass)..."
                "$PYTHON_BIN" "$REPO_ROOT/scripts/bass/generate_bass_plan.py" \
                    --chordmap "$BASS_CHORDMAP" \
                    --view "$BASS_VIEW" \
                    --sections "$SECTIONS_JSON" \
                    --tempo-map "$TEMPO_MAP" \
                    --out "$PLANS_DIR/bass_plan.json" || {
                    echo "⚠️  Bass plan legacy generation failed"
                    [[ $STRICT -eq 1 ]] && exit 1
                }
                echo "   ✅ bass_plan.json (legacy)"
            else
                echo "   ⚠️  chordmap_view_bass.json not found, skipping Bass plan"
            fi
        fi
    }
    
    if [[ -f "$PLANS_DIR/bass_plan.json" ]]; then
        echo "   ✅ bass_plan.json (V2)"
    fi
elif [[ -f "$REPO_ROOT/scripts/bass/generate_bass_plan.py" ]]; then
    BASS_VIEW="$ANALYSIS_DIR/chordmap_view_bass.json"
    BASS_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    
    if [[ -f "$BASS_VIEW" ]]; then
        echo "   Generating Bass plan (using view_bass)..."
        "$PYTHON_BIN" "$REPO_ROOT/scripts/bass/generate_bass_plan.py" \
            --chordmap "$BASS_CHORDMAP" \
            --view "$BASS_VIEW" \
            --sections "$SECTIONS_JSON" \
            --tempo-map "$TEMPO_MAP" \
            --out "$PLANS_DIR/bass_plan.json" || {
            echo "⚠️  Bass plan generation failed"
            [[ $STRICT -eq 1 ]] && exit 1
        }
        echo "   ✅ bass_plan.json"
    else
        echo "   ⚠️  chordmap_view_bass.json not found, skipping Bass plan"
    fi
elif [[ -f "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" ]]; then
    # Fallback: use instrument_midi_to_plan_real.py
    BASS_VIEW="$ANALYSIS_DIR/chordmap_view_bass.json"
    BASS_POLICY="$REPO_ROOT/scripts/instrument_chordmap/policy/chordmap_view_bass.yaml"
    BASS_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    BASS_ROLE_BARS="$ANALYSIS_DIR/role_bars/bass.parquet"
    
    echo "   Generating Bass plan (fallback: instrument_midi_to_plan_real.py)..."
    VIEW_ARG=()
    if [[ -f "$BASS_VIEW" ]]; then
        VIEW_ARG=("--view" "$BASS_VIEW")
    elif [[ -f "$BASS_POLICY" ]]; then
        VIEW_ARG=("--policy" "$BASS_POLICY")
    fi
    
    ROLE_BARS_ARG=()
    if [[ -f "$BASS_ROLE_BARS" ]]; then
        ROLE_BARS_ARG=("--role-bars" "$BASS_ROLE_BARS")
    fi
    
    # Build command with optional args
    BASS_CMD=(
        "$PYTHON_BIN" "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py"
        --role bass
        --song-package "$SONG_ROOT"
        --chordmap "$BASS_CHORDMAP"
        --sections "$SECTIONS_JSON"
        --bars "$STEP1_OUT_BARS"
    )
    [[ ${#VIEW_ARG[@]} -gt 0 ]] && BASS_CMD+=("${VIEW_ARG[@]}")
    [[ ${#ROLE_BARS_ARG[@]} -gt 0 ]] && BASS_CMD+=("${ROLE_BARS_ARG[@]}")
    BASS_CMD+=(--walking-bass --voice-leading --multi-chords)
    
    "${BASS_CMD[@]}" || {
        echo "⚠️  Bass plan generation (fallback) failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "   ✅ bass_plan.json (via fallback)"
else
    echo "   ⚠️  bass plan generator not found, skipping"
fi

# Guitar plan生成（V2優先 → view_guitar → fallback）
if [[ -f "$REPO_ROOT/scripts/generate_guitar_plan_v2.py" && -f "$BARS_WITH_SLOTS" && -f "$POLICY_YAML" ]]; then
    # V2 (PREFERRED): Slot-based guitar renderer
    
    echo "   Generating Guitar plan (V2: slot-based)..."
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_guitar_plan_v2.py" \
        --bars "$BARS_WITH_SLOTS" \
        --sections "$SECTIONS_JSON" \
        --chordmap "$CHORDMAP_EXTENDED" \
        --policy "$POLICY_YAML" \
        --out "$PLANS_DIR/guitar_plan.json" || {
        echo "⚠️  Guitar plan V2 generation failed, trying legacy..."
        
        # Fallback to legacy
        if [[ -f "$REPO_ROOT/scripts/guitar/generate_guitar_plan.py" ]]; then
            GUITAR_VIEW="$ANALYSIS_DIR/chordmap_view_guitar.json"
            GUITAR_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
            
            if [[ -f "$GUITAR_VIEW" ]]; then
                echo "   Generating Guitar plan (legacy: using view_guitar)..."
                "$PYTHON_BIN" "$REPO_ROOT/scripts/guitar/generate_guitar_plan.py" \
                    --chordmap "$GUITAR_CHORDMAP" \
                    --view "$GUITAR_VIEW" \
                    --sections "$SECTIONS_JSON" \
                    --tempo-map "$TEMPO_MAP" \
                    --out "$PLANS_DIR/guitar_plan.json" || {
                    echo "⚠️  Guitar plan legacy generation failed"
                    [[ $STRICT -eq 1 ]] && exit 1
                }
                echo "   ✅ guitar_plan.json (legacy)"
            else
                echo "   ⚠️  chordmap_view_guitar.json not found, skipping Guitar plan"
            fi
        fi
    }
    
    if [[ -f "$PLANS_DIR/guitar_plan.json" ]]; then
        echo "   ✅ guitar_plan.json (V2)"
    fi
elif [[ -f "$REPO_ROOT/scripts/guitar/generate_guitar_plan.py" ]]; then
    GUITAR_VIEW="$ANALYSIS_DIR/chordmap_view_guitar.json"
    GUITAR_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    
    if [[ -f "$GUITAR_VIEW" ]]; then
        echo "   Generating Guitar plan (using view_guitar)..."
        "$PYTHON_BIN" "$REPO_ROOT/scripts/guitar/generate_guitar_plan.py" \
            --chordmap "$GUITAR_CHORDMAP" \
            --view "$GUITAR_VIEW" \
            --sections "$SECTIONS_JSON" \
            --tempo-map "$TEMPO_MAP" \
            --out "$PLANS_DIR/guitar_plan.json" || {
            echo "⚠️  Guitar plan generation failed"
            [[ $STRICT -eq 1 ]] && exit 1
        }
        echo "   ✅ guitar_plan.json"
    else
        echo "   ⚠️  chordmap_view_guitar.json not found, skipping Guitar plan"
    fi
elif [[ -f "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" ]]; then
    # Fallback: use instrument_midi_to_plan_real.py
    GUITAR_VIEW="$ANALYSIS_DIR/chordmap_view_guitar.json"
    GUITAR_POLICY="$REPO_ROOT/scripts/instrument_chordmap/policy/chordmap_view_guitar.yaml"
    GUITAR_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    GUITAR_ROLE_BARS="$ANALYSIS_DIR/role_bars/guitar.parquet"
    
    echo "   Generating Guitar plan (fallback: instrument_midi_to_plan_real.py)..."
    VIEW_ARG=()
    if [[ -f "$GUITAR_VIEW" ]]; then
        VIEW_ARG=("--view" "$GUITAR_VIEW")
    elif [[ -f "$GUITAR_POLICY" ]]; then
        VIEW_ARG=("--policy" "$GUITAR_POLICY")
    fi
    
    ROLE_BARS_ARG=()
    [[ -f "$GUITAR_ROLE_BARS" ]] && ROLE_BARS_ARG=("--role-bars" "$GUITAR_ROLE_BARS")
    
    "$PYTHON_BIN" "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" \
        --role guitar \
        --song-package "$SONG_ROOT" \
        --chordmap "$GUITAR_CHORDMAP" \
        --sections "$SECTIONS_JSON" \
        --bars "$STEP1_OUT_BARS" \
        "${VIEW_ARG[@]}" \
        "${ROLE_BARS_ARG[@]}" \
        --strum \
        --open-voicing auto \
        --voice-leading \
        --multi-chords || {
        echo "⚠️  Guitar plan generation (fallback) failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "   ✅ guitar_plan.json (via fallback)"
else
    echo "   ⚠️  guitar plan generator not found, skipping"
fi

# Piano plan生成（V2優先 → view_piano → fallback）
if [[ -f "$REPO_ROOT/scripts/generate_piano_plan_v2.py" && -f "$BARS_WITH_SLOTS" && -f "$POLICY_YAML" ]]; then
    # V2 (PREFERRED): Slot-based piano renderer
    
    echo "   Generating Piano plan (V2: slot-based)..."
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_piano_plan_v2.py" \
        --bars "$BARS_WITH_SLOTS" \
        --sections "$SECTIONS_JSON" \
        --chordmap "$CHORDMAP_EXTENDED" \
        --policy "$POLICY_YAML" \
        --out "$PLANS_DIR/piano_plan.json" || {
        echo "⚠️  Piano plan V2 generation failed, trying legacy..."
        
        # Fallback to legacy
        if [[ -f "$REPO_ROOT/scripts/piano/generate_piano_plan.py" ]]; then
            PIANO_VIEW="$ANALYSIS_DIR/chordmap_view_piano.json"
            PIANO_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
            
            if [[ -f "$PIANO_VIEW" ]]; then
                echo "   Generating Piano plan (legacy: using view_piano)..."
                "$PYTHON_BIN" "$REPO_ROOT/scripts/piano/generate_piano_plan.py" \
                    --chordmap "$PIANO_CHORDMAP" \
                    --view "$PIANO_VIEW" \
                    --sections "$SECTIONS_JSON" \
                    --tempo-map "$TEMPO_MAP" \
                    --out "$PLANS_DIR/piano_plan.json" || {
                    echo "⚠️  Piano plan legacy generation failed"
                    [[ $STRICT -eq 1 ]] && exit 1
                }
                echo "   ✅ piano_plan.json (legacy)"
            else
                echo "   ⚠️  chordmap_view_piano.json not found, skipping Piano plan"
            fi
        fi
    }
    
    if [[ -f "$PLANS_DIR/piano_plan.json" ]]; then
        echo "   ✅ piano_plan.json (V2)"
    fi
elif [[ -f "$REPO_ROOT/scripts/piano/generate_piano_plan.py" ]]; then
    PIANO_VIEW="$ANALYSIS_DIR/chordmap_view_piano.json"
    PIANO_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    
    if [[ -f "$PIANO_VIEW" ]]; then
        echo "   Generating Piano plan (using view_piano)..."
        "$PYTHON_BIN" "$REPO_ROOT/scripts/piano/generate_piano_plan.py" \
            --chordmap "$PIANO_CHORDMAP" \
            --view "$PIANO_VIEW" \
            --sections "$SECTIONS_JSON" \
            --tempo-map "$TEMPO_MAP" \
            --out "$PLANS_DIR/piano_plan.json" || {
            echo "⚠️  Piano plan generation failed"
            [[ $STRICT -eq 1 ]] && exit 1
        }
        echo "   ✅ piano_plan.json"
    else
        echo "   ⚠️  chordmap_view_piano.json not found, skipping Piano plan"
    fi
elif [[ -f "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" ]]; then
    # Fallback: use instrument_midi_to_plan_real.py
    PIANO_VIEW="$ANALYSIS_DIR/chordmap_view_piano.json"
    PIANO_POLICY="$REPO_ROOT/scripts/instrument_chordmap/policy/chordmap_view_piano.yaml"
    PIANO_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    PIANO_ROLE_BARS="$ANALYSIS_DIR/role_bars/piano.parquet"
    
    echo "   Generating Piano plan (fallback: instrument_midi_to_plan_real.py)..."
    VIEW_ARG=()
    if [[ -f "$PIANO_VIEW" ]]; then
        VIEW_ARG=("--view" "$PIANO_VIEW")
    elif [[ -f "$PIANO_POLICY" ]]; then
        VIEW_ARG=("--policy" "$PIANO_POLICY")
    fi
    
    ROLE_BARS_ARG=()
    [[ -f "$PIANO_ROLE_BARS" ]] && ROLE_BARS_ARG=("--role-bars" "$PIANO_ROLE_BARS")
    
    "$PYTHON_BIN" "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" \
        --role piano \
        --song-package "$SONG_ROOT" \
        --chordmap "$PIANO_CHORDMAP" \
        --sections "$SECTIONS_JSON" \
        --bars "$STEP1_OUT_BARS" \
        "${VIEW_ARG[@]}" \
        "${ROLE_BARS_ARG[@]}" \
        --voice-leading \
        --multi-chords || {
        echo "⚠️  Piano plan generation (fallback) failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "   ✅ piano_plan.json (via fallback)"
else
    echo "   ⚠️  piano plan generator not found, skipping"
fi

# Strings plan生成（V2優先 → CREPE enhanced → view_strings → fallback）
if [[ -f "$REPO_ROOT/scripts/generate_strings_plan_v2.py" && -f "$BARS_WITH_SLOTS" && -f "$POLICY_YAML" ]]; then
    # V2 (PREFERRED): Slot-based strings renderer
    VOCAL_F0="$ANALYSIS_DIR/vocal_f0_crepe.parquet"
    
    echo "   Generating Strings plan (V2: slot-based)..."
    VOCAL_F0_ARG=()
    [[ -f "$VOCAL_F0" ]] && VOCAL_F0_ARG=("--vocal-f0" "$VOCAL_F0")
    
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_strings_plan_v2.py" \
        --bars "$BARS_WITH_SLOTS" \
        --sections "$SECTIONS_JSON" \
        --chordmap "$CHORDMAP_EXTENDED" \
        --policy "$POLICY_YAML" \
        "${VOCAL_F0_ARG[@]}" \
        --out "$PLANS_DIR/strings_plan.json" || {
        echo "⚠️  Strings plan V2 generation failed, trying legacy..."
        
        # Fallback to legacy (CREPE or view_strings)
        if [[ -f "$PLANS_DIR/strings_countermelody_plan_vl.json" ]]; then
            echo "   Using CREPE-enhanced Strings plan (already generated in STEP 11)"
            echo "   ✅ strings_countermelody_plan_vl.json (CREPE enhanced)"
        elif [[ -f "$REPO_ROOT/scripts/strings/generate_strings_plan.py" ]]; then
            STRINGS_VIEW="$ANALYSIS_DIR/chordmap_view_strings.json"
            STRINGS_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
            
            if [[ -f "$STRINGS_VIEW" ]]; then
                echo "   Generating Strings plan (legacy: using view_strings)..."
                "$PYTHON_BIN" "$REPO_ROOT/scripts/strings/generate_strings_plan.py" \
                    --chordmap "$STRINGS_CHORDMAP" \
                    --view "$STRINGS_VIEW" \
                    --sections "$SECTIONS_JSON" \
                    --tempo-map "$TEMPO_MAP" \
                    --out "$PLANS_DIR/strings_plan.json" || {
                    echo "⚠️  Strings plan legacy generation failed"
                    [[ $STRICT -eq 1 ]] && exit 1
                }
                echo "   ✅ strings_plan.json (legacy)"
            else
                echo "   ⚠️  chordmap_view_strings.json not found, skipping Strings plan"
            fi
        fi
    }
    
    if [[ -f "$PLANS_DIR/strings_plan.json" ]]; then
        echo "   ✅ strings_plan.json (V2)"
    fi
elif [[ -f "$PLANS_DIR/strings_countermelody_plan_vl.json" ]]; then
    echo "   Using CREPE-enhanced Strings plan (already generated in STEP 11)"
    echo "   ✅ strings_countermelody_plan_vl.json (CREPE enhanced)"
elif [[ -f "$REPO_ROOT/scripts/strings/generate_strings_plan.py" ]]; then
    STRINGS_VIEW="$ANALYSIS_DIR/chordmap_view_strings.json"
    STRINGS_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    
    if [[ -f "$STRINGS_VIEW" ]]; then
        echo "   Generating Strings plan (using view_strings)..."
        "$PYTHON_BIN" "$REPO_ROOT/scripts/strings/generate_strings_plan.py" \
            --chordmap "$STRINGS_CHORDMAP" \
            --view "$STRINGS_VIEW" \
            --sections "$SECTIONS_JSON" \
            --tempo-map "$TEMPO_MAP" \
            --out "$PLANS_DIR/strings_plan.json" || {
            echo "⚠️  Strings plan generation failed"
            [[ $STRICT -eq 1 ]] && exit 1
        }
        echo "   ✅ strings_plan.json"
    else
        echo "   ⚠️  chordmap_view_strings.json not found, skipping Strings plan"
    fi
elif [[ -f "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" ]]; then
    # Fallback: use instrument_midi_to_plan_real.py
    STRINGS_VIEW="$ANALYSIS_DIR/chordmap_view_strings.json"
    STRINGS_POLICY="$REPO_ROOT/scripts/instrument_chordmap/policy/chordmap_view_strings.yaml"
    STRINGS_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    STRINGS_ROLE_BARS="$ANALYSIS_DIR/role_bars/strings.parquet"
    
    echo "   Generating Strings plan (fallback: instrument_midi_to_plan_real.py)..."
    VIEW_ARG=()
    if [[ -f "$STRINGS_VIEW" ]]; then
        VIEW_ARG=("--view" "$STRINGS_VIEW")
    elif [[ -f "$STRINGS_POLICY" ]]; then
        VIEW_ARG=("--policy" "$STRINGS_POLICY")
    fi
    
    ROLE_BARS_ARG=()
    [[ -f "$STRINGS_ROLE_BARS" ]] && ROLE_BARS_ARG=("--role-bars" "$STRINGS_ROLE_BARS")
    
    "$PYTHON_BIN" "$REPO_ROOT/scripts/instrument_midi_to_plan_real.py" \
        --role strings \
        --song-package "$SONG_ROOT" \
        --chordmap "$STRINGS_CHORDMAP" \
        --sections "$SECTIONS_JSON" \
        --bars "$STEP1_OUT_BARS" \
        "${VIEW_ARG[@]}" \
        "${ROLE_BARS_ARG[@]}" \
        --voice-leading \
        --multi-chords || {
        echo "⚠️  Strings plan generation (fallback) failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "   ✅ strings_plan.json (via fallback)"
else
    echo "   ⚠️  strings plan generator not found, skipping"
fi

# Pad plan生成（view_pad参照）
if [[ -f "$REPO_ROOT/scripts/pad/generate_pad_plan.py" ]]; then
    PAD_VIEW="$ANALYSIS_DIR/chordmap_view_pad.json"
    PAD_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    
    if [[ -f "$PAD_VIEW" ]]; then
        echo "   Generating Pad plan (using view_pad)..."
        "$PYTHON_BIN" "$REPO_ROOT/scripts/pad/generate_pad_plan.py" \
            --chordmap "$PAD_CHORDMAP" \
            --view "$PAD_VIEW" \
            --sections "$SECTIONS_JSON" \
            --tempo-map "$TEMPO_MAP" \
            --out "$PLANS_DIR/pad_plan.json" || {
            echo "⚠️  Pad plan generation failed"
            [[ $STRICT -eq 1 ]] && exit 1
        }
        echo "   ✅ pad_plan.json"
    else
        echo "   ⚠️  chordmap_view_pad.json not found, skipping Pad plan"
    fi
else
    # NOTE: Padはinstrument_midi_to_plan_real.pyのchoicesに含まれていないため、
    # 専用生成器のみサポート。見つからなければスキップ。
    echo "   ⚠️  pad plan generator not found, skipping (requires dedicated generator)"
fi

# ==============================================================================
# STEP 19: Drums Plan Generation (V2: Collaborative Architecture)
# ==============================================================================
# Purpose: Generate drums plan with slot-based fill system and collaborative hooks.
#
# Architecture (三段ロケット):
#   1. recommend_drums (optional): Pattern suggestions (future)
#   2. generate_drums_plan_v2 (core): Slot-based rendering ← PRIORITY
#   3. adapt_drums_to_plan (optional): Kit/humanization (future)
#   4. postprocess_plans_ignore_mute: Mute removal (always active)
#
# Design: 「位置決め（スロット）は bars/sections。表現の造形は楽器別レンダラ。」
#
# Priority:
#   1. generate_drums_plan_v2.py (slot-based, policy-driven) ← PREFERRED
#   2. Fallback: drums_midi_to_plan_real.py (real MIDI source)
#   3. Fallback: drums/generate_drums_plan.py (legacy V1)
# ==============================================================================

DRUMS_PLAN_GENERATED=0

if [[ -f "$REPO_ROOT/scripts/generate_drums_plan_v2.py" ]]; then
    # V2: Slot-based fill system (PREFERRED)
    POLICY_YAML="$SONG_ROOT/policy/$(basename "$SONG_ROOT").yaml"
    
    echo "   🥁 Generating Drums plan (V2: slot-based, collaborative architecture)..."
    
    # Check policy YAML existence (create minimal if missing)
    if [[ ! -f "$POLICY_YAML" ]]; then
        echo "   ⚠️  Policy YAML not found at $POLICY_YAML"
        echo "   Creating minimal fallback policy (boundary_fill=always, min_fill_prob=0.15)"
        mkdir -p "$(dirname "$POLICY_YAML")"
        cat > "$POLICY_YAML" << 'EOF'
name: minimal_drums_policy
instruments:
  drums:
    boundary_fill: always
    min_fill_prob: 0.15
    energy_jump_thresh: 0.06
    typical_fill_len_beats: [2.0, 4.0]
    long_fill_len_beats: 8.0
    humanize_timing_ms: 12
    humanize_velocity: 8
    accent_patterns:
      short_fill:
        - type: "buildup"
          pattern: [0.5, 0.6, 0.7, 0.9]
      standard_fill:
        - type: "uplifting"
          pattern: [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 0.9, 0.8]
sections:
  intro: {drums: 0.3}
  verse: {drums: 0.5}
  pre_chorus: {drums: 0.7}
  chorus: {drums: 0.95}
  bridge: {drums: 0.7}
  outro: {drums: 0.4}
EOF
    fi
    
    # V2 generation (core)
    # NOTE: drums V2 expects the bars_with_slots parquet (slot-based system)
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_drums_plan_v2.py" \
        --bars "$BARS_WITH_SLOTS" \
        --sections "$SECTIONS_JSON" \
        --policy "$POLICY_YAML" \
        --out "$PLANS_DIR/drums_plan.json" && {
        DRUMS_PLAN_GENERATED=1
        echo "   ✅ drums_plan.json (V2: boundary_fill guaranteed)"
    } || {
        echo "   ⚠️  Drums plan generation (V2) failed, trying fallback..."
    }

elif [[ -f "$REPO_ROOT/scripts/drums_midi_to_plan_real.py" && $DRUMS_PLAN_GENERATED -eq 0 ]]; then
    # Fallback 1: drums_midi_to_plan_real.py (real MIDI source)
    DRUMS_RECOMMENDATIONS="$ANALYSIS_DIR/drums_recommendations.json"
    PATTERNS_PICKLE="$REPO_ROOT/output/rhythm_ai/rhythm_patterns.pickle"
    DRUMS_ROLE_BARS="$ANALYSIS_DIR/role_bars/drums.parquet"
    
    # Get BPM from tempo_map.json
    TEMPO_BPM=$(python3 -c "import json; tm=json.load(open('$TEMPO_MAP')); print(tm['events'][0]['tempo'] if tm.get('events') else 120)")
    
    echo "   Generating Drums plan (fallback 1: drums_midi_to_plan_real.py, BPM=$TEMPO_BPM)..."
    ROLE_BARS_ARG=()
    [[ -f "$DRUMS_ROLE_BARS" ]] && ROLE_BARS_ARG=("--role-bars" "$DRUMS_ROLE_BARS")
    
    if [[ -f "$DRUMS_RECOMMENDATIONS" && -f "$PATTERNS_PICKLE" ]]; then
        "$PYTHON_BIN" "$REPO_ROOT/scripts/drums_midi_to_plan_real.py" \
            --recommendations "$DRUMS_RECOMMENDATIONS" \
            --patterns-pickle "$PATTERNS_PICKLE" \
            --out "$PLANS_DIR/drums_plan.json" \
            --tempo-bpm "$TEMPO_BPM" \
            --bars "$BARS_WITH_SLOTS" \
            "${ROLE_BARS_ARG[@]}" && {
            DRUMS_PLAN_GENERATED=1
            echo "   ✅ drums_plan.json (via fallback 1: real MIDI)"
        } || {
            echo "   ⚠️  Drums plan generation (fallback 1) failed"
        }
    else
        echo "   ⚠️  drums_recommendations.json or rhythm_patterns.pickle not found"
    fi

elif [[ -f "$REPO_ROOT/scripts/drums/generate_drums_plan.py" && $DRUMS_PLAN_GENERATED -eq 0 ]]; then
    # Fallback 2: Legacy V1 (drum_accent_plan.json based)
    DRUM_ACCENT="$ANALYSIS_DIR/drum_accent_plan.json"
    
    echo "   Generating Drums plan (fallback 2: V1 legacy, drum_accent_plan)..."
    "$PYTHON_BIN" "$REPO_ROOT/scripts/drums/generate_drums_plan.py" \
        --sections "$SECTIONS_JSON" \
        --tempo-map "$TEMPO_MAP" \
        --accent-plan "$DRUM_ACCENT" \
        --out "$PLANS_DIR/drums_plan.json" && {
        DRUMS_PLAN_GENERATED=1
        echo "   ✅ drums_plan.json (via fallback 2: V1 legacy)"
    } || {
        echo "   ⚠️  Drums plan generation (fallback 2) failed"
    }
fi

# Final check + postprocess (mute removal)
if [[ $DRUMS_PLAN_GENERATED -eq 1 && -f "$PLANS_DIR/drums_plan.json" ]]; then
    # Postprocess: Remove mute/velocity_factor (always active policy)
    if [[ -f "$REPO_ROOT/scripts/postprocess_plans_ignore_mute.py" ]]; then
        echo "   🔧 Postprocessing drums_plan.json (mute removal, density restoration)..."
        "$PYTHON_BIN" "$REPO_ROOT/scripts/postprocess_plans_ignore_mute.py" \
            --plan "$PLANS_DIR/drums_plan.json" \
            --out "$PLANS_DIR/drums_plan.json" || {
            echo "   ⚠️  Postprocess failed (continuing with original plan)"
        }
    fi
else
    echo "   ⚠️  drums plan generator not found or all methods failed, skipping"
    [[ $STRICT -eq 1 ]] && exit 1
fi

# ==========================================
# STEP 19.5: Quality Gate — Fill/Riff Coverage & Density Validation
# ==========================================
echo ""
echo "🎯 Step 19.5: Quality Gate (Fill/Riff coverage, density validation)"

if [[ -f "$REPO_ROOT/scripts/quality_gate_fill_riff.py" ]]; then
    QG_ARGS=(
        --bars "$BARS_WITH_SLOTS"
        --plans "$PLANS_DIR"
    )
    
    # STRICT mode: Quality Gate failure = immediate exit
    if [[ $STRICT -eq 1 ]]; then
        QG_ARGS+=(--strict)
        echo "   ⚠️  STRICT mode: Quality Gate failure will abort pipeline"
    fi
    
    "$PYTHON_BIN" "$REPO_ROOT/scripts/quality_gate_fill_riff.py" "${QG_ARGS[@]}" && {
        echo "   ✅ Quality Gate: PASSED"
    } || {
        echo "   ❌ Quality Gate: FAILED"
        if [[ $STRICT -eq 1 ]]; then
            echo "   Aborting due to quality gate failure (--strict mode)"
            exit 1
        else
            echo "   Continuing despite quality gate failure (non-strict mode)"
        fi
    }
else
    echo "   ℹ️  quality_gate_fill_riff.py not found, skipping validation"
fi

# ==========================================
# STEP 20: 全plan統合MIDI生成（可変テンポ対応）
# ==========================================
echo ""
echo "🎹 Step 20: Integrated MIDI generation (variable tempo)"

INTEGRATED_MIDI="$MIDI_DIR/${SONG_ID}_integrated.mid"

if [[ -f "$REPO_ROOT/scripts/merge_plans_to_midi.py" ]]; then
    # 存在するplanファイルを収集
    ALL_PLANS=()
    [[ -f "$PLANS_DIR/bass_plan.json" ]] && ALL_PLANS+=("--bass" "$PLANS_DIR/bass_plan.json")
    [[ -f "$PLANS_DIR/guitar_plan.json" ]] && ALL_PLANS+=("--guitar" "$PLANS_DIR/guitar_plan.json")
    [[ -f "$PLANS_DIR/guitar_plan_optimized_micro.json" ]] && ALL_PLANS+=("--guitar" "$PLANS_DIR/guitar_plan_optimized_micro.json")
    [[ -f "$PLANS_DIR/piano_plan.json" ]] && ALL_PLANS+=("--piano" "$PLANS_DIR/piano_plan.json")
    [[ -f "$PLANS_DIR/piano_plan_hybrid.json" ]] && ALL_PLANS+=("--piano" "$PLANS_DIR/piano_plan_hybrid.json")
    [[ -f "$PLANS_DIR/strings_plan.json" ]] && ALL_PLANS+=("--strings" "$PLANS_DIR/strings_plan.json")
    [[ -f "$PLANS_DIR/strings_countermelody_plan_vl.json" ]] && ALL_PLANS+=("--strings" "$PLANS_DIR/strings_countermelody_plan_vl.json")
    [[ -f "$PLANS_DIR/pad_plan.json" ]] && ALL_PLANS+=("--pad" "$PLANS_DIR/pad_plan.json")
    [[ -f "$PLANS_DIR/drums_plan.json" ]] && ALL_PLANS+=("--drums" "$PLANS_DIR/drums_plan.json")
    
    if [[ ${#ALL_PLANS[@]} -gt 0 ]]; then
        echo "   Merging ${#ALL_PLANS[@]} plans into integrated MIDI..."
        "$PYTHON_BIN" "$REPO_ROOT/scripts/merge_plans_to_midi.py" \
            --tempo-map "$TEMPO_MAP" \
            "${ALL_PLANS[@]}" \
            --output "$INTEGRATED_MIDI" || {
            echo "⚠️  Integrated MIDI generation failed"
            [[ $STRICT -eq 1 ]] && exit 1
        }
        echo "   ✅ ${SONG_ID}_integrated.mid"
    else
        echo "   ⚠️  No plans found for MIDI integration"
    fi
else
    echo "   ⚠️  merge_plans_to_midi.py not found, skipping"
fi

# ==========================================
# STEP 21: deep_harmony_audit最終監査（LOCKED参照）
# ==========================================
echo ""
echo "🔍 Step 21: Final harmony audit (using locked chordmap)"

FINAL_AUDIT="$ANALYSIS_DIR/harmony_audit_final.json"

if [[ -f "$REPO_ROOT/ops/deep_harmony_audit.py" ]]; then
    AUDIT_CHORDMAP="${LOCKED_CHORDMAP:-$CHORDMAP_JSON}"
    
    "$PYTHON_BIN" "$REPO_ROOT/ops/deep_harmony_audit.py" \
        --song-root "$SONG_ROOT" \
        --tempo-map "$TEMPO_MAP" \
        --sections "$SECTIONS_JSON" \
        --chordmap "$AUDIT_CHORDMAP" \
        --m21 "$M21_CHORDMAP" \
        --report "$FINAL_AUDIT" || {
        echo "⚠️  Final harmony audit failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    echo "   ✅ harmony_audit_final.json"
else
    echo "   ⚠️  deep_harmony_audit.py not found, skipping"
fi

# ==========================================
# STEP 22: Songpackage最終生成（3 variant、LOCKED参照）
# ==========================================
echo ""
echo "📦 Step 22: Final songpackage generation (3 variants, using locked)"
echo "   ⚠️  PHASE-A-BLOCKED: SongPackage生成はPhase B（RUN_SONG_004.sh STEP 21）で実施"
echo "   （未確定和声を基準にしないため、Phase Aでは生成しません）"

# PHASE-A-BLOCKED: if [[ -f "$REPO_ROOT/scripts/generate_suno_song_package_v1_1.py" ]]; then
# PHASE-A-BLOCKED:     for variant in soft standard bright; do
# PHASE-A-BLOCKED:         echo "   Generating ${variant} variant (final)..."
# PHASE-A-BLOCKED:         
# PHASE-A-BLOCKED:         FINAL_PACKAGE="$SONG_ROOT/song_package_${variant}_final.yaml"
# PHASE-A-BLOCKED:         
# PHASE-A-BLOCKED:         "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_suno_song_package_v1_1.py" \
# PHASE-A-BLOCKED:             --song-id "$SONG_ID" \
# PHASE-A-BLOCKED:             --analysis-dir "$ANALYSIS_DIR" \
# PHASE-A-BLOCKED:             --plans-dir "$PLANS_DIR" \
# PHASE-A-BLOCKED:             --midi-dir "$MIDI_DIR" \
# PHASE-A-BLOCKED:             --variant "$variant" \
# PHASE-A-BLOCKED:             --chordmap "$LOCKED_CHORDMAP" \
# PHASE-A-BLOCKED:             --out "$FINAL_PACKAGE" || {
# PHASE-A-BLOCKED:             echo "⚠️  Final songpackage generation failed: $variant"
# PHASE-A-BLOCKED:             [[ $STRICT -eq 1 ]] && exit 1
# PHASE-A-BLOCKED:         }
# PHASE-A-BLOCKED:         echo "   ✅ song_package_${variant}_final.yaml"
# PHASE-A-BLOCKED:     done
# PHASE-A-BLOCKED:     echo "✅ All 3 final variants generated"
# PHASE-A-BLOCKED: else
# PHASE-A-BLOCKED:     echo "⚠️  generate_suno_song_package_v1_1.py not found"
# PHASE-A-BLOCKED: fi

# ==========================================
# 完了
# ==========================================
echo ""
echo "✅ Song package generation complete (Full pipeline: CREPE + Instrument Views + Plans + MIDI + Audit)!"
echo ""
echo "   🎯 Phase A (Auto): STEP 1-15"
echo "     - tempo_map.json + bars.parquet (完全版)"
echo "     - sections.json + lyric_anchors.json"
echo "     - chordmap.json (auto) + stems_features.parquet"
echo "     - CREPE F0 + plans (Strings/Guitar/Piano/Synth)"
echo "     - 補助4点（drum_accent/bassline/voicings/style）"
echo "     - CREPE統合MIDI + 監査 + Songpackage (3 variant初回)"
echo ""
echo "   🎼 Phase B (Manual→LOCK→Plan→MIDI): STEP 16-22"
echo "     - STEP 16: chordmap_locked.json + QA report"
echo "     - STEP 17: chordmap_m21.json (music21正規化)"
echo "     - STEP 18: 楽器別view（Pad/Guitar/Piano/Strings/Bass）"
echo "     - STEP 19: 各楽器plan生成（LOCKED参照）"
echo "     - STEP 20: 統合MIDI（可変テンポ）"
echo "     - STEP 21: 最終監査（LOCKED参照）"
echo "     - STEP 22: Songpackage最終版（3 variant）"
echo ""
echo "   📁 Generated files:"
if [[ -f "$LOCKED_CHORDMAP" ]]; then
    echo "     ✅ $LOCKED_CHORDMAP"
fi
if [[ -f "$M21_CHORDMAP" ]]; then
    echo "     ✅ $M21_CHORDMAP"
fi
if [[ -f "$INTEGRATED_MIDI" ]]; then
    echo "     ✅ $INTEGRATED_MIDI"
fi
if [[ -f "$FINAL_AUDIT" ]]; then
    echo "     ✅ $FINAL_AUDIT"
fi
echo ""
echo "   🎹 Instrument Views:"
for role in pad guitar piano strings bass; do
    view_json="$ANALYSIS_DIR/chordmap_view_${role}.json"
    if [[ -f "$view_json" ]]; then
        echo "     ✅ chordmap_view_${role}.json"
    fi
done
echo ""
echo "   🎵 Plans:"
for plan in bass guitar piano strings pad drums; do
    plan_json="$PLANS_DIR/${plan}_plan.json"
    if [[ -f "$plan_json" ]]; then
        echo "     ✅ ${plan}_plan.json"
    fi
done
# CREPE enhanced plans
[[ -f "$PLANS_DIR/guitar_plan_optimized_micro.json" ]] && echo "     ✅ guitar_plan_optimized_micro.json (CREPE)"
[[ -f "$PLANS_DIR/piano_plan_hybrid.json" ]] && echo "     ✅ piano_plan_hybrid.json (CREPE)"
[[ -f "$PLANS_DIR/strings_countermelody_plan_vl.json" ]] && echo "     ✅ strings_countermelody_plan_vl.json (CREPE)"
echo ""
echo "   bars.parquet (完全版) columns:"
"$PYTHON_BIN" -c "
import pandas as pd
bars = pd.read_parquet('$STEP1_OUT_BARS')
print(f'     Total bars: {len(bars)}')
print(f'     Total columns: {len(bars.columns)}')
print()
print('     必須カラム:')
required_cols = ['bar_index', 'tempo_bpm', 'time_signature', 'start_sec', 'end_sec', 
                 'start_beat', 'end_beat', 'density_target', 'swing_target', 'section_label']
for col in required_cols:
    status = '✅' if col in bars.columns else '❌'
    print(f'       {status} {col}')
print()
print('     stem_features由来カラム:')
feature_cols = ['drums_active', 'energy_curve', 'hat_density', 'kick_peak_db', 
                'snare_backbeat', 'fill_likelihood', 'loudness_db', 'vocal_stress',
                'guitar_activity', 'piano_activity', 'strings_activity']
for col in feature_cols:
    status = '✅' if col in bars.columns else '⚠️ '
    print(f'       {status} {col}')
print()
print('     全カラムリスト:')
for col in bars.columns:
    print(f'       - {col}')
"
echo ""
echo "   analysis/ directory files:"
"$PYTHON_BIN" -c "
from pathlib import Path
analysis_dir = Path('$ANALYSIS_DIR')
if analysis_dir.exists():
    for f in sorted(analysis_dir.glob('*')):
        if f.is_file():
            size = f.stat().st_size
            print(f'     ✅ {f.name:<30} {size:>8} bytes')
"

############################################
# Phase C: Plan統合 → MIDI書き出し（plans-only）
############################################
integrate_midi() {
    local SONG_DIR="$1"
    local PPQ_VAL="${PPQ:-480}"
    local REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    local E2E="$REPO_ROOT/scripts/e2e_integrate_midi.sh"
    
    if [[ ! -x "$E2E" ]]; then
        chmod +x "$E2E" 2>/dev/null || true
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎼 [Phase C] Plan統合→MIDI書き出し (PPQ=${PPQ_VAL})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [[ ! -d "$SONG_DIR/plans" ]]; then
        echo "❌ plans ディレクトリが見つかりません: $SONG_DIR/plans"
        return 1
    fi
    
    "$E2E" "$SONG_DIR" --ppq "$PPQ_VAL"
    local STATUS=$?
    
    if [[ $STATUS -eq 0 ]]; then
        echo ""
        echo "✅ Phase C 完了: MIDI統合成功"
    else
        echo ""
        echo "❌ Phase C 失敗: MIDI統合エラー"
        return 1
    fi
}

# Phase C を実行
if [[ -d "$PLANS_DIR" ]]; then
    integrate_midi "$SONG_DIR" || {
        echo "⚠️  Phase C failed but continuing..."
    }
fi

