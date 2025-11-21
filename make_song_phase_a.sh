#!/usr/bin/env bash
# make_song_phase_a.sh — Phase A refresher wrapper
# -------------------------------------------------
# Phase 0 では tempo/bars 情報を再抽出して最新解析を保証する必要がある。
# このスクリプトは song package ディレクトリを受け取り、既存の Phase A 産物
# (tempo_map.json / bars*.parquet / sections.json / lyric_anchors.json など) を安全に
# バックアップした上で `scripts/make_song_package_phase_a.sh` を実行する。
#
# Usage:
#   ./make_song_phase_a.sh <song_package_dir> [PhaseA options...]
# Options:
#   --no-clean         既存アーティファクトを保持し、再抽出前のクリーンアップをスキップ
#   --phase-script P   デフォルト以外の Phase A スクリプトを明示 (デフォルト: scripts/make_song_package_phase_a.sh)
#   -h|--help          このヘルプを表示
#
# それ以外の引数はすべて `make_song_package_phase_a.sh` にそのまま渡される。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
DEFAULT_PHASE_SCRIPT="$REPO_ROOT/scripts/make_song_package_phase_a.sh"
PHASE_SCRIPT="$DEFAULT_PHASE_SCRIPT"
CLEAN_ARTIFACTS=1
SONG_DIR=""
FORWARD_ARGS=()

print_help() {
  sed -n '1,40p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-clean)
      CLEAN_ARTIFACTS=0
      shift
      ;;
    --phase-script)
      PHASE_SCRIPT="${2:?Missing value for --phase-script}"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        FORWARD_ARGS+=("$1")
        shift
      done
      ;;
    *)
      if [[ -z "$SONG_DIR" ]]; then
        SONG_DIR="$1"
      else
        FORWARD_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$SONG_DIR" ]]; then
  echo "Usage: $0 <song_package_dir> [options...]" >&2
  exit 1
fi

if [[ ! -x "$PHASE_SCRIPT" ]]; then
  echo "❌ Phase A script not executable: $PHASE_SCRIPT" >&2
  exit 1
fi

SONG_DIR="$(cd "$SONG_DIR" && pwd)"
ANALYSIS_DIR="$SONG_DIR/analysis"
mkdir -p "$ANALYSIS_DIR"

backup_and_remove() {
  local target="$1"
  local pretty="$2"
  local backup_dir="$ANALYSIS_DIR/.phase_a_backup_$(date +%Y%m%d_%H%M%S)"
  local abs=""

  for path in "$ANALYSIS_DIR/$target" "$SONG_DIR/$target"; do
    if [[ -e "$path" ]]; then
      abs="$path"
      break
    fi
  done

  [[ -z "$abs" ]] && return 0

  mkdir -p "$backup_dir"
  mv "$abs" "$backup_dir/$(basename "$abs")" 2>/dev/null || cp -f "$abs" "$backup_dir/"
  rm -f "$abs"
  echo "   ⚙️  Removed stale $pretty → backup: $backup_dir"
}

if [[ $CLEAN_ARTIFACTS -eq 1 ]]; then
  echo "🧹 Phase A artifacts cleanup (tempo/bars 再抽出)"
  backup_and_remove tempo_map.json "tempo_map.json"
  backup_and_remove bars.parquet "bars.parquet"
  backup_and_remove bars_with_slots.parquet "bars_with_slots.parquet"
  backup_and_remove sections.json "sections.json"
  backup_and_remove lyric_anchors.json "lyric_anchors.json"
else
  echo "ℹ️  --no-clean specified. Existing Phase A artifacts will be reused."
fi

echo "🚀 Running Phase A script: $PHASE_SCRIPT"
if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
  exec "$PHASE_SCRIPT" "$SONG_DIR" "${FORWARD_ARGS[@]}"
else
  exec "$PHASE_SCRIPT" "$SONG_DIR"
fi
