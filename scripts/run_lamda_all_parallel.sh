#!/bin/bash
# LAMDA全サブフォルダ処理（2楽器並列）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  LAMDA Batch Processing - 2 Parallel${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 楽器リスト（2楽器ずつ並列処理）
INSTRUMENTS=("piano" "guitar" "bass" "strings" "drums")
SUBFOLDERS=(0 1 2 3 4 5 6 7 8 9 a b c d e f)

# 開始時刻
START_TIME=$(date +%s)
echo -e "${GREEN}⏰ Start time: $(date)${NC}"
echo ""

# 2楽器ずつ処理
for ((i=0; i<${#INSTRUMENTS[@]}; i+=2)); do
    INSTRUMENT1="${INSTRUMENTS[$i]}"
    INSTRUMENT2="${INSTRUMENTS[$i+1]}"
    
    if [ -z "$INSTRUMENT2" ]; then
        # 最後の楽器が1つだけの場合
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}  Processing: ${INSTRUMENT1} (single)${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        INSTRUMENT_START=$(date +%s)
        
        for subfolder in "${SUBFOLDERS[@]}"; do
            echo -e "${BLUE}📂 ${INSTRUMENT1} - Subfolder ${subfolder}${NC}"
            ./scripts/run_lamda_subfolder.sh "$INSTRUMENT1" "$subfolder"
            
            # 進捗表示
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - INSTRUMENT_START))
            echo -e "${GREEN}  ✅ Completed in ${ELAPSED}s${NC}"
            echo ""
        done
        
    else
        # 2楽器並列処理
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}  Processing: ${INSTRUMENT1} + ${INSTRUMENT2} (parallel)${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        INSTRUMENT_START=$(date +%s)
        
        for subfolder in "${SUBFOLDERS[@]}"; do
            echo -e "${BLUE}📂 Subfolder ${subfolder}: ${INSTRUMENT1} + ${INSTRUMENT2}${NC}"
            
            # 2楽器を並列実行
            ./scripts/run_lamda_subfolder.sh "$INSTRUMENT1" "$subfolder" &
            PID1=$!
            
            ./scripts/run_lamda_subfolder.sh "$INSTRUMENT2" "$subfolder" &
            PID2=$!
            
            # 両方の完了を待機
            wait $PID1
            STATUS1=$?
            
            wait $PID2
            STATUS2=$?
            
            # 結果表示
            if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
                echo -e "${GREEN}  ✅ Both completed successfully${NC}"
            else
                echo -e "${RED}  ⚠️  Warning: Some processes failed (${INSTRUMENT1}:${STATUS1}, ${INSTRUMENT2}:${STATUS2})${NC}"
            fi
            
            # 進捗表示
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - INSTRUMENT_START))
            echo -e "${GREEN}  Time: ${ELAPSED}s${NC}"
            echo ""
        done
    fi
    
    # 楽器ペア完了
    INSTRUMENT_END=$(date +%s)
    INSTRUMENT_ELAPSED=$((INSTRUMENT_END - INSTRUMENT_START))
    INSTRUMENT_ELAPSED_MIN=$((INSTRUMENT_ELAPSED / 60))
    echo -e "${GREEN}✅ ${INSTRUMENT1} + ${INSTRUMENT2} completed in ${INSTRUMENT_ELAPSED_MIN} minutes${NC}"
    echo ""
done

# 終了時刻
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🎉 All processing completed!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}⏰ End time: $(date)${NC}"
echo -e "${GREEN}⏱️  Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m${NC}"
echo ""

# 結果サマリー
echo -e "${BLUE}📊 Output Summary:${NC}"
for instrument in "${INSTRUMENTS[@]}"; do
    COUNT=$(ls -1 data/lamda_${instrument}_metadata/*.pickle 2>/dev/null | wc -l)
    SIZE=$(du -sh data/lamda_${instrument}_metadata/ 2>/dev/null | cut -f1)
    echo -e "  ${instrument}: ${COUNT} pickle files (${SIZE})"
done
echo ""

echo -e "${YELLOW}📝 Logs: logs/lamda_*_*.log${NC}"
echo -e "${YELLOW}📦 Output: data/lamda_*_metadata/$(*)_shard_*.pickle${NC}"
