#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <song-dir> [OPTIONS]"
  exit 1
fi
SONG_DIR="$1"; shift

# Ensure song_package.yaml exists (fallback to song_package_standard.yaml)
if [[ ! -f "$SONG_DIR/song_package.yaml" && -f "$SONG_DIR/song_package_standard.yaml" ]]; then
  echo "🔁 Creating symlink: song_package.yaml -> song_package_standard.yaml"
  (cd "$SONG_DIR" && ln -sf song_package_standard.yaml song_package.yaml)
fi

# Ensure bars.parquet exists at root (fallback to analysis/bars.parquet)
if [[ ! -f "$SONG_DIR/bars.parquet" && -f "$SONG_DIR/analysis/bars.parquet" ]]; then
  echo "🔁 Creating symlink: bars.parquet -> analysis/bars.parquet"
  (cd "$SONG_DIR" && ln -sf analysis/bars.parquet bars.parquet)
fi

# Forward to the original e2e script (same directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG="$SCRIPT_DIR/e2e_suno_arrangement.sh"
if [[ ! -x "$ORIG" ]]; then
  chmod +x "$ORIG" 2>/dev/null || true
fi

exec "$ORIG" "$SONG_DIR" "$@"
