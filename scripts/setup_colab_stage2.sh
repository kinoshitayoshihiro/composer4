#!/bin/bash
# Stage2 Colab Setup Script
# Google Drive共有フォルダからデータをダウンロードしてStage2を準備

set -e

echo "🚀 Stage2 Colab Setup - Starting..."

# ===== 1) リポジトリと依存関係のセットアップ =====
echo ""
echo "📦 Step 1: Repository and Dependencies"
if [ ! -d "/content/composer4" ]; then
    git clone https://github.com/kinoshitayoshihiro/composer4.git /content/composer4
    echo "✅ Repository cloned"
else
    cd /content/composer4
    git pull origin main
    echo "✅ Repository updated"
fi

cd /content/composer4
pip install -q torch transformers pytest gdown numpy scipy mido

# ===== 2) Google Drive共有フォルダのダウンロード =====
echo ""
echo "📥 Step 2: Downloading from Google Drive"
FOLDER_URL="https://drive.google.com/drive/folders/1zUg85irbGgcHZggCGXiCwOZWHR6uKX5T?usp=sharing"
DEST="/content/composer4/_drive_download"
mkdir -p "$DEST"

echo "Downloading shared folder to: $DEST"
gdown --folder --fuzzy "$FOLDER_URL" -O "$DEST" || {
    echo "⚠️ gdown failed. Trying alternative method..."
    gdown --folder "$FOLDER_URL" -O "$DEST" --remaining-ok
}

# ===== 3) ダウンロード内容の確認 =====
echo ""
echo "🔍 Step 3: Analyzing downloaded structure"
echo "Downloaded structure:"
find "$DEST" -maxdepth 3 -type d | head -n 20

# ===== 4) 'output' ディレクトリの探索 =====
echo ""
echo "🔎 Step 4: Searching for 'output' directory"

# 複数の可能性を探索
POSSIBLE_PATHS=(
    "$DEST/output"
    "$DEST/*/output"
    "$DEST/*/*/output"
    "$DEST/composer4/output"
    "$DEST/composer2-3/output"
)

OUTPUT_SRC=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        OUTPUT_SRC="$path"
        echo "✅ Found output directory: $OUTPUT_SRC"
        break
    fi
done

# 見つからない場合の詳細診断
if [ -z "$OUTPUT_SRC" ]; then
    echo ""
    echo "❌ ERROR: 'output' directory not found in downloaded content"
    echo ""
    echo "📂 Downloaded directory structure (detailed):"
    tree -L 3 "$DEST" 2>/dev/null || find "$DEST" -type d | head -n 30
    echo ""
    echo "🔍 All directories named 'output' (recursive search):"
    find "$DEST" -type d -name "output" 2>/dev/null || echo "  (none found)"
    echo ""
    echo "💡 Suggestions:"
    echo "  1. Check Google Drive folder structure"
    echo "  2. Ensure 'output' directory exists in shared folder"
    echo "  3. Verify sharing permissions"
    echo "  4. Alternative: Upload 'output' directory directly to Colab"
    echo ""
    exit 1
fi

# ===== 5) シンボリックリンクの作成 =====
echo ""
echo "🔗 Step 5: Creating symbolic link"
ln -sfn "$OUTPUT_SRC" /content/composer4/output
echo "✅ Linked: $OUTPUT_SRC  ->  /content/composer4/output"

# ===== 6) 内容の検証 =====
echo ""
echo "📊 Step 6: Verifying output directory contents"
if [ -d "/content/composer4/output" ]; then
    echo "Output directory structure:"
    ls -lah /content/composer4/output | head -n 30
    echo ""
    
    # Stage2に必要なサブディレクトリの確認
    REQUIRED_DIRS=("drum_metadata" "drum_cleaned")
    echo "Checking required subdirectories:"
    for dir in "${REQUIRED_DIRS[@]}"; do
        if [ -d "/content/composer4/output/$dir" ]; then
            file_count=$(find "/content/composer4/output/$dir" -type f | wc -l)
            echo "  ✅ $dir: $file_count files"
        else
            echo "  ⚠️ $dir: NOT FOUND (may need to create)"
        fi
    done
else
    echo "❌ ERROR: Symbolic link creation failed"
    exit 1
fi

# ===== 7) GPU確認 =====
echo ""
echo "🎮 Step 7: GPU Information"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️ No GPU detected (CPU mode)"
fi

# ===== 8) 準備完了 =====
echo ""
echo "✅ ========================================="
echo "✅ Stage2 Setup Complete!"
echo "✅ ========================================="
echo ""
echo "📂 Working directory: /content/composer4"
echo "📂 Output directory: /content/composer4/output"
echo ""
echo "🚀 Next steps:"
echo "  1. Verify data: cd /content/composer4 && ls -la output/"
echo "  2. Run Stage2: bash scripts/run_stage2_drum.sh"
echo ""
