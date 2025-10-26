#!/usr/bin/env bash
# ========================================
# Los-Angeles-MIDI (LAMDA) クリーニング専用ラッパー
# 楽器別クリーニング: piano/strings/guitar/bass
# ========================================
#
# LAMDA は約40万ファイルのマルチトラック楽曲データセット
# → 楽器別に分離してクリーニングします
#
# Pickle-Direct v2運用:
#   - .meta.json を出さない (--emit-meta-json off)
#   - Pickle シャードに直接書き込み
#   - SSD停止からのレジューム対応 (--resume)
#
# 使用方法:
#   ./scripts/run_lamda_full.sh                    # 全楽器実行
#   ./scripts/run_lamda_full.sh --dry-run          # コマンド確認のみ
#   ./scripts/run_lamda_full.sh piano              # piano のみ実行
#   ./scripts/run_lamda_full.sh guitar bass        # guitar と bass のみ

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 楽器名を LAMDA_* データセット名にマッピング
map_inst() {
  case "$1" in
    piano)   echo "LAMDA_PIANO"   ;;
    strings) echo "LAMDA_STRINGS" ;;
    guitar)  echo "LAMDA_GUITAR"  ;;
    bass)    echo "LAMDA_BASS"    ;;
    drums)   echo "LAMDA_DRUMS"   ;;
    LAMDA_*) echo "$1" ;;  # 既にLAMDA_*形式なら素通し
    --dry-run) echo "$1" ;;  # フラグは素通し
    *)       echo "$1" ;;  # その他は素通し
  esac
}

# 引数チェック: --dry-run の有無と楽器名マッピング
DRY_RUN_FLAG=""
INSTRUMENTS=()

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
  else
    MAPPED=$(map_inst "$arg")
    if [[ "$MAPPED" =~ ^LAMDA_ ]]; then
      INSTRUMENTS+=("$MAPPED")
    fi
  fi
done

# 楽器指定がなければ全楽器（drumsを除く推奨4楽器）
if [ ${#INSTRUMENTS[@]} -eq 0 ]; then
  INSTRUMENTS=("LAMDA_PIANO" "LAMDA_STRINGS" "LAMDA_GUITAR" "LAMDA_BASS")
  echo "ℹ️  No instruments specified. Running: piano, strings, guitar, bass"
  echo "   (drums excluded by default due to size. Run explicitly if needed)"
  echo ""
fi

# 共通ランナーを呼び出し
exec "${SCRIPT_DIR}/run_dataset_full.sh" ${DRY_RUN_FLAG} "${INSTRUMENTS[@]}"
