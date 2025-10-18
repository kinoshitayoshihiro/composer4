#!/usr/bin/env bash
# ========================================
# POP909 クリーニング専用ラッパー
# 共通ランナー (run_dataset_full.sh) へのエイリアス
# ========================================
#
# Pickle-Direct v2運用に対応:
#   - .meta.json を出さない (--emit-meta-json off)
#   - Pickle シャードに直接書き込み
#   - SSD停止からのレジューム対応 (--resume)
#
# 使用方法:
#   ./scripts/run_pop909_full.sh           # 通常実行
#   ./scripts/run_pop909_full.sh --dry-run # コマンド確認のみ

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 共通ランナーを呼び出し（POP909データセットのみ実行）
exec "${SCRIPT_DIR}/run_dataset_full.sh" "$@" POP909
