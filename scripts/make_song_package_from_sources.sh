#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# scripts/make_song_package_from_sources.sh
# 
# Suno AI曲パッケージを生成（4ステップ統合）
#
# 使い方:
#   bash scripts/make_song_package_from_sources.sh \
#     data/suno_ai/suno_themesong/song_001 \
#     --stems-dir "data/suno_ai/suno_themesong/song_001/stemswav_001"
#
# オプション:
#   --stems-dir DIR       Stems WAV格納ディレクトリ（必須）
#   --mix-wav PATH        Mix WAVパス（指定しない場合は自動検出）
#   --vocal-wav PATH      Vocal WAVパス（指定しない場合は自動検出）
#   --dry-run             実行コマンド表示のみ
#   --strict              失敗時即終了

set -euo pipefail

# UTF-8ロケール設定（日本語パス対応）
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# デフォルト設定
DRY_RUN=0
STRICT=0
STEMS_DIR=""
MIX_WAV=""
VOCAL_WAV=""

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
# Mix WAV自動検出
# ==========================================
if [[ -z "$MIX_WAV" ]]; then
  echo "🔍 Auto-detecting Mix WAV..."
  
  # パターン1: _auto_Other.wav
  MIX_WAV="$STEMS_DIR/_auto_Other.wav"
  if [[ ! -f "$MIX_WAV" ]]; then
    # パターン2: *Mix*.wav
    MIX_WAV="$(find "$STEMS_DIR" -iname "*Mix*.wav" | head -n1)"
  fi
  
  # パターン3: Vocal/Drums以外の最初のWAV
  if [[ -z "$MIX_WAV" ]] || [[ ! -f "$MIX_WAV" ]]; then
    MIX_WAV="$(find "$STEMS_DIR" -name "*.wav" ! -iname "*Vocal*" ! -iname "*Drums*" | head -n1)"
  fi
  
  if [[ -z "$MIX_WAV" ]] || [[ ! -f "$MIX_WAV" ]]; then
    echo "⚠️  Mix WAV not found, creating from stems..."
    MIX_WAV="$SONG_ROOT/_auto_Other.wav"
    
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

STEP1_OUT_JSON="$SONG_ROOT/tempo_map.json"
STEP1_OUT_BARS="$SONG_ROOT/bars.parquet"

# ダミーbars.parquet生成（初回のみ）
if [[ ! -f "$STEP1_OUT_BARS" ]]; then
  echo "   Creating dummy bars.parquet..."
  "$PYTHON_BIN" -c "
import pandas as pd
from pathlib import Path
import soundfile as sf

# Mix WAVの長さ取得
y, sr = sf.read('$MIX_WAV')
duration = len(y) / sr

# 仮定: 4/4拍子、120BPM
bpm = 120.0
beat_sec = 60.0 / bpm
bar_sec = beat_sec * 4
n_bars = int(duration / bar_sec) + 1

bars_df = pd.DataFrame([
    {'bar_index': i, 'tempo_bpm': bpm, 'time_signature': '4/4'}
    for i in range(n_bars)
])
bars_df.to_parquet('$STEP1_OUT_BARS', index=False)
print(f'✅ Dummy bars.parquet created: {n_bars} bars')
" || {
    echo "❌ Failed to create dummy bars.parquet"
    [[ $STRICT -eq 1 ]] && exit 1
  }
fi

CMD1=("$PYTHON_BIN" "$REPO_ROOT/ops/tempo_map_cli.py" \
      --audio "$MIX_WAV" \
      --bars "$STEP1_OUT_BARS" \
      --out-bars "$STEP1_OUT_BARS" \
      --out-tempo "$STEP1_OUT_JSON" \
      --bpb 4 \
      --prefer-madmom)

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] ${CMD1[*]}"
else
  echo "   Running: ${CMD1[*]}"
  "${CMD1[@]}" || {
    echo "❌ STEP 1 failed"
    [[ $STRICT -eq 1 ]] && exit 1
  }
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
tempos = [p['bpm'] for p in tempo_map.get('tempo_points', [])]
import statistics
print(statistics.median(tempos) if tempos else 120.0)
")

echo "   Median BPM: $MEDIAN_BPM"

# start_sec/end_sec/density_target/swing_target追加
"$PYTHON_BIN" -c "
import pandas as pd
import numpy as np
from pathlib import Path

# bars.parquet読み込み
bars = pd.read_parquet('$STEP1_OUT_BARS')

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

STEP2_OUT_JSON="$SONG_ROOT/sections.json"

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

STEP3_OUT_JSON="$SONG_ROOT/lyric_anchors.json"

if [[ -n "$VOCAL_WAV" ]] && [[ -f "$VOCAL_WAV" ]]; then
  CMD3=("$PYTHON_BIN" "$REPO_ROOT/ops/anchors_from_vocal.py" \
        --vocal "$VOCAL_WAV" \
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

CMD4=("$PYTHON_BIN" "$REPO_ROOT/ops/stem_harmony_bar_level.py" \
      --stems "$STEMS_DIR" \
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
# STEP 5: stem_features.parquet + drums_active追加
# ==========================================
echo ""
echo "🕔 STEP 5/5: stem_features.parquet + drums_active"

STEP5_OUT_FEATURES="$SONG_ROOT/stem_features.parquet"

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

# stem_features.parquetからすべての有用なカラムをbars.parquetにマージ（完全版生成）
echo "   Merging stem_features to bars.parquet (完全版生成)..."
"$PYTHON_BIN" -c "
import pandas as pd

# bars.parquet読み込み
bars = pd.read_parquet('$STEP1_OUT_BARS')

# stem_features.parquet読み込み
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

# ==========================================
# analysisディレクトリへのコピー（E2E処理用）
# ==========================================
echo ""
echo "📋 Copying files to analysis/ directory..."

cp "$STEP1_OUT_BARS" "$ANALYSIS_DIR/bars.parquet"
cp "$STEP2_OUT_JSON" "$ANALYSIS_DIR/sections.json"
cp "$STEP1_OUT_JSON" "$ANALYSIS_DIR/tempo_map.json"

if [[ -f "$STEP3_OUT_JSON" ]]; then
  cp "$STEP3_OUT_JSON" "$ANALYSIS_DIR/lyric_anchors.json"
fi

if [[ -f "$STEP5_OUT_FEATURES" ]]; then
  cp "$STEP5_OUT_FEATURES" "$ANALYSIS_DIR/stem_features.parquet"
fi

echo "✅ Files copied to analysis/"

# ==========================================
# STEP 7: CREPE/OaF用シンボリックリンク作成
# ==========================================
echo ""
echo "🔗 Step 7: Creating symbolic links for CREPE/OaF"

# vocal.wavシンボリックリンク作成
VOCAL_STEM=$(find "$STEMS_DIR" -name "*Vocals*.wav" -o -name "*vocals*.wav" -o -name "*vocal*.wav" 2>/dev/null | head -1)
if [[ -n "$VOCAL_STEM" ]]; then
    # 相対パスに変換
    VOCAL_RELATIVE=$(python3 -c "import os; print(os.path.relpath('$VOCAL_STEM', '$SONG_DIR'))")
    ln -sf "$VOCAL_RELATIVE" "$SONG_DIR/vocal.wav"
    echo "✅ Created symbolic link: vocal.wav -> $VOCAL_RELATIVE"
else
    echo "⚠️  No vocal stem found in $STEMS_DIR"
fi

# piano.wavシンボリックリンク作成
PIANO_STEM=$(find "$STEMS_DIR" -name "*Keyboard*.wav" -o -name "*Piano*.wav" -o -name "*piano*.wav" 2>/dev/null | head -1)
if [[ -n "$PIANO_STEM" ]]; then
    # 相対パスに変換
    PIANO_RELATIVE=$(python3 -c "import os; print(os.path.relpath('$PIANO_STEM', '$SONG_DIR'))")
    ln -sf "$PIANO_RELATIVE" "$SONG_DIR/piano.wav"
    echo "✅ Created symbolic link: piano.wav -> $PIANO_RELATIVE"
else
    echo "⚠️  No piano/keyboard stem found in $STEMS_DIR"
fi

# ==========================================
# 完了
# ==========================================
echo ""
echo "✅ Song package generation complete!"
echo "   Generated files:"
echo "     - $STEP1_OUT_JSON"
echo "     - $STEP1_OUT_BARS"
echo "     - $STEP2_OUT_JSON"
echo "     - $STEP3_OUT_JSON"
echo "     - $STEP4_OUT_JSON"
echo "     - $STEP5_OUT_FEATURES"
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
