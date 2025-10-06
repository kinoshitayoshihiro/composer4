#!/bin/bash
# Colab環境セットアップスクリプト
# 使い方: Colabで !bash setup_colab.sh を実行

set -e

echo "🚀 Composer4 Colab Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Gitクローン
echo "📥 Cloning repository..."
git clone --depth 1 https://github.com/kinoshitayoshihiro/composer4.git /content/composer4
cd /content/composer4

# 2. 依存関係インストール
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# 3. GCS認証（手動で実行が必要）
echo "🔐 Please run authentication manually:"
echo "    from google.colab import auth"
echo "    auth.authenticate_user()"
echo "    !gcloud config set project charged-camera-450413-k2"

# 4. データコピー準備
echo "📊 To copy data from GCS, run:"
echo "    !gsutil -m cp -r gs://otocotoba/data/phrase_csv data/"
echo "    !gsutil -m cp -r gs://otocotoba/data/lamd data/"

echo ""
echo "✅ Setup complete! Ready for training."
