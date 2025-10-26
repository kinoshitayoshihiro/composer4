#!/bin/bash
# cleanup_before_full_processing.sh
# 全曲処理前のクリーンアップスクリプト（統合レイアウト準拠版）

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
LOCAL_LAMDA="${BASE_DIR}/data/Los-Angeles-MIDI/LOCAL_LAMDA"

echo "=========================================="
echo "全曲処理前クリーンアップ（統合レイアウト準拠版）"
echo "=========================================="
echo ""

# バックアップディレクトリ作成
BACKUP_DIR="${BASE_DIR}/data/backups/pre_full_processing_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"

echo "📦 バックアップ先: ${BACKUP_DIR}"
echo ""

# ========== 1. 旧形式WAV成果物の削除 ==========
echo "=== 1. 旧形式WAV成果物のクリーンアップ ==="

# 旧形式のフラットJSON（moisesdb）
OLD_MOISESDB_FLAT="${LOCAL_LAMDA}/Local_Lamda_wav/wav_guide/moisesdb"
if [ -d "${OLD_MOISESDB_FLAT}" ]; then
    # song_id 単独ディレクトリのみを残す（フラットJSONは削除）
    echo "  - ${OLD_MOISESDB_FLAT} 内の旧形式JSON削除..."
    
    # バックアップ（念のため直近10曲分のサンプル）
    mkdir -p "${BACKUP_DIR}/moisesdb_sample"
    find "${OLD_MOISESDB_FLAT}" -maxdepth 1 -name "*.json" | head -10 | while read f; do
        cp "$f" "${BACKUP_DIR}/moisesdb_sample/"
    done
    
    # 旧形式JSON削除（ディレクトリ以外）
    find "${OLD_MOISESDB_FLAT}" -maxdepth 1 -type f -name "*.json" -delete
    echo "  ✓ 旧形式JSON削除完了"
else
    echo "  - ${OLD_MOISESDB_FLAT} なし（スキップ）"
fi

# 旧形式のフラットJSON（musdb18）
OLD_MUSDB18_FLAT="${LOCAL_LAMDA}/Local_Lamda_wav/wav_guide/musdb18"
if [ -d "${OLD_MUSDB18_FLAT}" ]; then
    echo "  - ${OLD_MUSDB18_FLAT} 内の旧形式JSON削除..."
    find "${OLD_MUSDB18_FLAT}" -maxdepth 1 -type f -name "*.json" -delete 2>/dev/null || true
    echo "  ✓ 旧形式JSON削除完了"
fi

echo ""

# ========== 2. 旧DB削除 ==========
echo "=== 2. 旧DB削除 ==="

# テストDB
for db in "${BASE_DIR}/data/moisesdb_wav_test.db" \
          "${BASE_DIR}/data/moisesdb_wav_unified_test.db" \
          "${BASE_DIR}/data/musdb18_wav_test.db"; do
    if [ -f "${db}" ]; then
        echo "  - バックアップ: $(basename ${db})"
        cp "${db}" "${BACKUP_DIR}/"
        rm -f "${db}"
        rm -f "${db}-shm" "${db}-wal" 2>/dev/null || true
        echo "  ✓ 削除: ${db}"
    fi
done

# 本番DB（念のためバックアップのみ、削除はしない）
PROD_DBS=(
    "${BASE_DIR}/data/moisesdb_wav_unified.db"
    "${BASE_DIR}/data/musdb18_wav_unified.db"
    "${LOCAL_LAMDA}/local_lamda_registry.db"
)
for db in "${PROD_DBS[@]}"; do
    if [ -f "${db}" ]; then
        echo "  - 本番DB発見（バックアップのみ、削除なし）: $(basename ${db})"
        cp "${db}" "${BACKUP_DIR}/"
    fi
done

echo ""

# ========== 3. 旧JSONL削除 ==========
echo "=== 3. 旧JSONL削除 ==="

for jsonl in "${BASE_DIR}/data"/*.jsonl; do
    if [ -f "${jsonl}" ]; then
        echo "  - バックアップ: $(basename ${jsonl})"
        cp "${jsonl}" "${BACKUP_DIR}/"
        rm -f "${jsonl}"
        echo "  ✓ 削除: ${jsonl}"
    fi
done

echo ""

# ========== 4. テスト用ディレクトリ削除 ==========
echo "=== 4. テスト用ディレクトリ削除 ==="

TEST_DIRS=(
    "${LOCAL_LAMDA}/Local_Lamda_wav/wav_guide/moisesdb_test"
    "${BASE_DIR}/data/local_lamda_wav_features/moisesdb_test"
    "${BASE_DIR}/data/local_lamda_wav_features/musdb18_test"
)

for dir in "${TEST_DIRS[@]}"; do
    if [ -d "${dir}" ]; then
        echo "  - バックアップ: $(basename ${dir})"
        cp -r "${dir}" "${BACKUP_DIR}/"
        rm -rf "${dir}"
        echo "  ✓ 削除: ${dir}"
    fi
done

echo ""

# ========== 5. キャッシュ削除（任意） ==========
echo "=== 5. キャッシュ削除（任意） ==="

CACHE_DIR="${LOCAL_LAMDA}/.cache/local_lamda"
if [ -d "${CACHE_DIR}" ]; then
    CACHE_SIZE=$(du -sh "${CACHE_DIR}" | cut -f1)
    echo "  - キャッシュサイズ: ${CACHE_SIZE}"
    echo "  - 削除しますか？ [y/N]"
    # 自動化のためデフォルトはN（手動実行時のみ削除）
    # read -r answer
    # if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    #     rm -rf "${CACHE_DIR}"
    #     echo "  ✓ キャッシュ削除完了"
    # else
    echo "  - スキップ（手動で rm -rf ${CACHE_DIR} で削除可能）"
    # fi
fi

echo ""

# ========== 6. 確認 ==========
echo "=== 6. クリーンアップ後の状態確認 ==="

echo ""
echo "WAV成果物（moisesdb）:"
if [ -d "${OLD_MOISESDB_FLAT}" ]; then
    echo "  - ディレクトリ数: $(find ${OLD_MOISESDB_FLAT} -maxdepth 1 -type d | wc -l | tr -d ' ')"
    echo "  - 残存JSON数: $(find ${OLD_MOISESDB_FLAT} -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
else
    echo "  - なし"
fi

echo ""
echo "WAV成果物（musdb18）:"
if [ -d "${OLD_MUSDB18_FLAT}" ]; then
    echo "  - ディレクトリ数: $(find ${OLD_MUSDB18_FLAT} -maxdepth 1 -type d | wc -l | tr -d ' ')"
    echo "  - 残存JSON数: $(find ${OLD_MUSDB18_FLAT} -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
else
    echo "  - なし"
fi

echo ""
echo "DB:"
ls -lh "${BASE_DIR}/data"/*.db 2>/dev/null | tail -5 || echo "  - なし"

echo ""
echo "=========================================="
echo "✅ クリーンアップ完了"
echo "=========================================="
echo ""
echo "📦 バックアップ: ${BACKUP_DIR}"
echo ""
echo "次のステップ:"
echo "  1. 全曲処理実行（MoisesDB/MUSDB18）"
echo "  2. Song Package生成"
echo "  3. レンダー＆QA"
echo ""
