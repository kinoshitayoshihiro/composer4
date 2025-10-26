#!/bin/bash
# check_cleanup_status.sh
# クリーンアップ状況の確認スクリプト

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
LOCAL_LAMDA="${BASE_DIR}/data/Los-Angeles-MIDI/LOCAL_LAMDA"

echo "=========================================="
echo "クリーンアップ状況チェック"
echo "=========================================="
echo ""

# ========== 1. 旧形式フラットJSON確認 ==========
echo "=== 1. 旧形式フラットJSON ==="

MOISESDB_DIR="${LOCAL_LAMDA}/Local_Lamda_wav/wav_guide/moisesdb"
if [ -d "${MOISESDB_DIR}" ]; then
    FLAT_JSON_COUNT=$(find "${MOISESDB_DIR}" -maxdepth 1 -type f -name "*.json" | wc -l | tr -d ' ')
    SONG_DIR_COUNT=$(find "${MOISESDB_DIR}" -maxdepth 1 -type d -not -name "moisesdb" | wc -l | tr -d ' ')
    
    echo "  moisesdb:"
    echo "    - フラットJSON数: ${FLAT_JSON_COUNT} $([ ${FLAT_JSON_COUNT} -gt 0 ] && echo '⚠️  削除推奨' || echo '✅')"
    echo "    - 曲ディレクトリ数: ${SONG_DIR_COUNT}"
else
    echo "  moisesdb: なし"
fi

MUSDB18_DIR="${LOCAL_LAMDA}/Local_Lamda_wav/wav_guide/musdb18"
if [ -d "${MUSDB18_DIR}" ]; then
    FLAT_JSON_COUNT=$(find "${MUSDB18_DIR}" -maxdepth 1 -type f -name "*.json" | wc -l | tr -d ' ')
    SONG_DIR_COUNT=$(find "${MUSDB18_DIR}" -maxdepth 1 -type d -not -name "musdb18" | wc -l | tr -d ' ')
    
    echo "  musdb18:"
    echo "    - フラットJSON数: ${FLAT_JSON_COUNT} $([ ${FLAT_JSON_COUNT} -gt 0 ] && echo '⚠️  削除推奨' || echo '✅')"
    echo "    - 曲ディレクトリ数: ${SONG_DIR_COUNT}"
else
    echo "  musdb18: なし（まだ未処理）"
fi

echo ""

# ========== 2. DB確認 ==========
echo "=== 2. DB状態 ==="

echo "  テストDB:"
for db in "${BASE_DIR}/data/moisesdb_wav_test.db" \
          "${BASE_DIR}/data/moisesdb_wav_unified_test.db" \
          "${BASE_DIR}/data/musdb18_wav_test.db"; do
    if [ -f "${db}" ]; then
        SIZE=$(ls -lh "${db}" | awk '{print $5}')
        echo "    - $(basename ${db}): ${SIZE} ⚠️  削除推奨"
    fi
done

echo "  本番DB:"
PROD_DBS=(
    "${BASE_DIR}/data/moisesdb_wav_unified.db"
    "${BASE_DIR}/data/musdb18_wav_unified.db"
    "${LOCAL_LAMDA}/local_lamda_registry.db"
)
for db in "${PROD_DBS[@]}"; do
    if [ -f "${db}" ]; then
        SIZE=$(ls -lh "${db}" | awk '{print $5}')
        echo "    - $(basename ${db}): ${SIZE} ℹ️  バックアップのみ推奨"
    fi
done

echo ""

# ========== 3. JSONL確認 ==========
echo "=== 3. JSONL ==="

JSONL_COUNT=$(ls "${BASE_DIR}/data"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
if [ ${JSONL_COUNT} -gt 0 ]; then
    echo "  - 発見: ${JSONL_COUNT}個 ⚠️  削除推奨"
    ls -lh "${BASE_DIR}/data"/*.jsonl 2>/dev/null | awk '{print "    - " $9 ": " $5}'
else
    echo "  - なし ✅"
fi

echo ""

# ========== 4. テストディレクトリ確認 ==========
echo "=== 4. テスト用ディレクトリ ==="

TEST_DIRS=(
    "${LOCAL_LAMDA}/Local_Lamda_wav/wav_guide/moisesdb_test"
    "${BASE_DIR}/data/local_lamda_wav_features/moisesdb_test"
)

for dir in "${TEST_DIRS[@]}"; do
    if [ -d "${dir}" ]; then
        SIZE=$(du -sh "${dir}" 2>/dev/null | cut -f1)
        echo "  - $(basename ${dir}): ${SIZE} ⚠️  削除推奨"
    fi
done

echo ""

# ========== 5. キャッシュ確認 ==========
echo "=== 5. キャッシュ ==="

CACHE_DIR="${LOCAL_LAMDA}/.cache/local_lamda"
if [ -d "${CACHE_DIR}" ]; then
    SIZE=$(du -sh "${CACHE_DIR}" 2>/dev/null | cut -f1)
    echo "  - サイズ: ${SIZE} ℹ️  削除は任意"
else
    echo "  - なし ✅"
fi

echo ""

# ========== 6. 推奨アクション ==========
echo "=========================================="
echo "推奨アクション"
echo "=========================================="
echo ""

NEEDS_CLEANUP=0

if [ ${FLAT_JSON_COUNT:-0} -gt 0 ]; then
    echo "⚠️  旧形式フラットJSONが残っています"
    NEEDS_CLEANUP=1
fi

if [ -f "${BASE_DIR}/data/moisesdb_wav_test.db" ] || \
   [ -f "${BASE_DIR}/data/moisesdb_wav_unified_test.db" ] || \
   [ -f "${BASE_DIR}/data/musdb18_wav_test.db" ]; then
    echo "⚠️  テストDBが残っています"
    NEEDS_CLEANUP=1
fi

if [ ${JSONL_COUNT:-0} -gt 0 ]; then
    echo "⚠️  旧JSONLが残っています"
    NEEDS_CLEANUP=1
fi

if [ ${NEEDS_CLEANUP} -eq 1 ]; then
    echo ""
    echo "📋 クリーンアップ実行コマンド:"
    echo "   ./scripts/cleanup_before_full_processing.sh"
    echo ""
else
    echo "✅ クリーンアップ不要です。全曲処理を開始できます。"
    echo ""
    echo "📋 全曲処理実行コマンド:"
    echo "   # MoisesDB"
    echo "   python scripts/local_lamda_moises_integration.py \\"
    echo "     --input-dir data/Los-Angeles-MIDI/LOCAL_LAMDA/Local_Lamda_wav/CLEANED_WAV/moisesdb_original \\"
    echo "     --output-db data/moisesdb_wav_unified.db \\"
    echo "     --source-name moisesdb \\"
    echo "     --policy-yaml config/stem_policy.yaml \\"
    echo "     --verbose 2>&1 | tee moisesdb_full_processing.log"
    echo ""
fi
