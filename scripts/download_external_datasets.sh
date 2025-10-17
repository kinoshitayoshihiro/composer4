#!/bin/bash
# Download External Datasets for Gap Filling
#
# 不足奏法補完用の外部データセットをダウンロード
#
# Usage:
#   bash scripts/download_external_datasets.sh [--dataset <name>]
#
# Options:
#   --dataset: ダウンロードするデータセット名（all/guitarset/urmp/maestro/smd）
#              指定しない場合はallを実行

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/external"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Dataset selection
DATASET="${1:-all}"

echo -e "${GREEN}[INFO]${NC} External Dataset Downloader"
echo -e "${GREEN}[INFO]${NC} Target: ${DATASET}"
echo -e "${GREEN}[INFO]${NC} Output: ${DATA_DIR}"
echo ""

# Create directories
mkdir -p "${DATA_DIR}"

# ========================================
# GuitarSet (Priority: 🔴 High)
# ========================================
download_guitarset() {
    echo -e "${YELLOW}[GuitarSet]${NC} Downloading..."
    
    GUITARSET_DIR="${DATA_DIR}/guitarset"
    
    if [[ -d "${GUITARSET_DIR}" ]]; then
        echo -e "${YELLOW}[GuitarSet]${NC} Already exists: ${GUITARSET_DIR}"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}[GuitarSet]${NC} Skipped"
            return
        fi
        rm -rf "${GUITARSET_DIR}"
    fi
    
    # Clone repository
    cd "${DATA_DIR}"
    git clone https://github.com/marl/guitarset.git
    
    # Download audio/MIDI files (if not included in repo)
    # Note: GuitarSet requires manual download of audio files
    cd "${GUITARSET_DIR}"
    
    if [[ ! -d "audio" ]] || [[ ! -d "annotation" ]]; then
        echo -e "${RED}[WARNING]${NC} GuitarSet audio/annotation files not found"
        echo -e "${RED}[WARNING]${NC} Please download manually from:"
        echo -e "${RED}[WARNING]${NC} https://zenodo.org/record/3371780"
        echo -e "${RED}[WARNING]${NC} Extract to: ${GUITARSET_DIR}/"
    fi
    
    echo -e "${GREEN}[GuitarSet]${NC} ✅ Repository cloned"
    echo -e "${GREEN}[GuitarSet]${NC} Expected files: ~360 MIDI (after manual download)"
}

# ========================================
# URMP (Priority: 🔴 High)
# ========================================
download_urmp() {
    echo -e "${YELLOW}[URMP]${NC} Downloading..."
    
    URMP_DIR="${DATA_DIR}/urmp"
    mkdir -p "${URMP_DIR}"
    
    if [[ -f "${URMP_DIR}/urmp_dataset.tar.gz" ]]; then
        echo -e "${YELLOW}[URMP]${NC} Archive already exists"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}[URMP]${NC} Skipped"
            return
        fi
    fi
    
    cd "${URMP_DIR}"
    
    # Download dataset (注意: 大容量 ~10GB)
    echo -e "${YELLOW}[URMP]${NC} Downloading dataset (~10GB, this may take a while)..."
    wget -c http://www2.ece.rochester.edu/projects/air/resource/urmp_dataset.tar.gz
    
    # Extract
    echo -e "${YELLOW}[URMP]${NC} Extracting..."
    tar -xzf urmp_dataset.tar.gz
    
    # Verify
    MIDI_COUNT=$(find . -name "*.mid" -o -name "*.midi" | wc -l)
    echo -e "${GREEN}[URMP]${NC} ✅ Downloaded and extracted"
    echo -e "${GREEN}[URMP]${NC} MIDI files found: ${MIDI_COUNT}"
}

# ========================================
# MAESTRO (Priority: 🟡 Medium)
# ========================================
download_maestro() {
    echo -e "${YELLOW}[MAESTRO]${NC} Downloading..."
    
    MAESTRO_DIR="${DATA_DIR}/maestro"
    mkdir -p "${MAESTRO_DIR}"
    
    if [[ -f "${MAESTRO_DIR}/maestro-v3.0.0-midi.zip" ]]; then
        echo -e "${YELLOW}[MAESTRO]${NC} Archive already exists"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}[MAESTRO]${NC} Skipped"
            return
        fi
    fi
    
    cd "${MAESTRO_DIR}"
    
    # Download MIDI-only version (~200MB)
    echo -e "${YELLOW}[MAESTRO]${NC} Downloading MIDI dataset (~200MB)..."
    wget -c https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
    
    # Extract
    echo -e "${YELLOW}[MAESTRO]${NC} Extracting..."
    unzip -q maestro-v3.0.0-midi.zip
    
    # Verify
    MIDI_COUNT=$(find maestro-v3.0.0 -name "*.mid" -o -name "*.midi" | wc -l)
    echo -e "${GREEN}[MAESTRO]${NC} ✅ Downloaded and extracted"
    echo -e "${GREEN}[MAESTRO]${NC} MIDI files found: ${MIDI_COUNT}"
}

# ========================================
# SMD (Synthetic MIDI Dataset) (Priority: 🟡 Medium)
# ========================================
download_smd() {
    echo -e "${YELLOW}[SMD]${NC} Downloading..."
    
    SMD_DIR="${DATA_DIR}/smd"
    
    if [[ -d "${SMD_DIR}" ]]; then
        echo -e "${YELLOW}[SMD]${NC} Already exists: ${SMD_DIR}"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}[SMD]${NC} Skipped"
            return
        fi
        rm -rf "${SMD_DIR}"
    fi
    
    # Clone repository
    cd "${DATA_DIR}"
    git clone https://github.com/bytedance/SMD.git smd
    
    cd "${SMD_DIR}"
    
    # SMDは大容量のため、Bass用のサブセットのみダウンロード推奨
    echo -e "${RED}[WARNING]${NC} SMD is very large (~150k files)"
    echo -e "${RED}[WARNING]${NC} Recommended: Download bass subset only"
    echo -e "${RED}[WARNING]${NC} Full dataset: https://github.com/bytedance/SMD"
    
    # Download metadata/subset information
    if [[ -f "download_subset.py" ]]; then
        echo -e "${YELLOW}[SMD]${NC} Running subset downloader..."
        python download_subset.py --instrument bass --max-files 500
    else
        echo -e "${YELLOW}[SMD]${NC} Manual download required"
        echo -e "${YELLOW}[SMD]${NC} See: https://github.com/bytedance/SMD#download"
    fi
    
    echo -e "${GREEN}[SMD]${NC} ✅ Repository cloned (data download may require manual steps)"
}

# ========================================
# Lakh MIDI Dataset (Priority: 🟢 Low)
# ========================================
download_lakh() {
    echo -e "${YELLOW}[Lakh]${NC} Lakh MIDI Dataset (176k files, ~30GB)"
    echo -e "${RED}[WARNING]${NC} This is a very large dataset"
    echo -e "${RED}[WARNING]${NC} Recommended: Download matched subset only"
    
    read -p "Continue with full download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}[Lakh]${NC} Skipped"
        return
    fi
    
    LAKH_DIR="${DATA_DIR}/lakh"
    mkdir -p "${LAKH_DIR}"
    cd "${LAKH_DIR}"
    
    # Download matched subset (推奨)
    echo -e "${YELLOW}[Lakh]${NC} Downloading matched subset..."
    wget -c http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
    
    # Extract
    echo -e "${YELLOW}[Lakh]${NC} Extracting..."
    tar -xzf lmd_matched.tar.gz
    
    MIDI_COUNT=$(find . -name "*.mid" -o -name "*.midi" | wc -l)
    echo -e "${GREEN}[Lakh]${NC} ✅ Downloaded matched subset"
    echo -e "${GREEN}[Lakh]${NC} MIDI files found: ${MIDI_COUNT}"
}

# ========================================
# Main execution
# ========================================

case "${DATASET}" in
    all)
        echo -e "${GREEN}[INFO]${NC} Downloading all priority datasets..."
        download_guitarset
        download_urmp
        download_maestro
        download_smd
        ;;
    guitarset)
        download_guitarset
        ;;
    urmp)
        download_urmp
        ;;
    maestro)
        download_maestro
        ;;
    smd)
        download_smd
        ;;
    lakh)
        download_lakh
        ;;
    *)
        echo -e "${RED}[ERROR]${NC} Unknown dataset: ${DATASET}"
        echo "Available: all, guitarset, urmp, maestro, smd, lakh"
        exit 1
        ;;
esac

# ========================================
# Summary
# ========================================

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}[SUMMARY]${NC} Download Status"
echo -e "${GREEN}============================================${NC}"

for ds in guitarset urmp maestro smd lakh; do
    DS_DIR="${DATA_DIR}/${ds}"
    if [[ -d "${DS_DIR}" ]]; then
        MIDI_COUNT=$(find "${DS_DIR}" -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l | tr -d ' ')
        echo -e "${GREEN}✅ ${ds}:${NC} ${MIDI_COUNT} MIDI files"
    else
        echo -e "${YELLOW}⏸️  ${ds}:${NC} Not downloaded"
    fi
done

echo ""
echo -e "${GREEN}[NEXT STEPS]${NC}"
echo "1. Verify downloaded datasets in: ${DATA_DIR}"
echo "2. Run import scripts:"
echo "   python scripts/import_guitarset.py"
echo "   python scripts/import_urmp.py"
echo "   python scripts/import_maestro.py"
echo "3. Integrate into Stage1 pipeline (MULTI_DATASET_RUNNER_GUIDE.md)"
echo ""
echo -e "${GREEN}[COMPLETE]${NC} Dataset download finished"
