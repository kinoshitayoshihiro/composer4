#!/usr/bin/env python3
"""
LAMDa Database Builder for Vertex AI Colab Enterprise
Vertex AI Colab Enterpriseで実行するLAMDaデータベース構築スクリプト
"""

import sys
from pathlib import Path
import time
import subprocess

print("=" * 70)
print("LAMDa Database Builder - Vertex AI Colab Enterprise")
print("=" * 70)

# ステップ1: 環境確認
print("\n[1/6] Checking environment...")
print(f"Python: {sys.version}")
print(f"Working directory: {Path.cwd()}")

# ステップ2: リポジトリクローン
print("\n[2/6] Cloning repository...")
repo_path = Path("/home/jupyter/composer4")
if repo_path.exists():
    print("Repository already exists, pulling latest...")
    subprocess.run(["git", "-C", str(repo_path), "pull"], check=True)
else:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/kinoshitayoshihiro/composer4.git",
            str(repo_path),
        ],
        check=True,
    )

# ステップ3: 依存関係インストール
print("\n[3/6] Installing dependencies...")
subprocess.run(["pip", "install", "-q", "music21", "numpy"], check=True)

# ステップ4: GCSからデータダウンロード
print("\n[4/6] Downloading CHORDS_DATA from GCS...")
data_dir = repo_path / "data" / "Los-Angeles-MIDI"
data_dir.mkdir(parents=True, exist_ok=True)

tar_file = data_dir / "CHORDS_DATA.tar.gz"
subprocess.run(
    ["gsutil", "cp", "gs://otocotoba/data/lamda/CHORDS_DATA.tar.gz", str(tar_file)], check=True
)

print(f"Downloaded: {tar_file}")
print(f"Size: {tar_file.stat().st_size / 1024 / 1024:.1f} MB")

# ステップ5: 解凍
print("\n[5/6] Extracting CHORDS_DATA...")
subprocess.run(["tar", "-xzf", str(tar_file), "-C", str(data_dir)], check=True)

chords_dir = data_dir / "CHORDS_DATA"
pickle_files = list(chords_dir.glob("*.pickle"))
print(f"Found {len(pickle_files)} pickle files")

# ステップ6: データベース構築
print("\n[6/6] Building database...")
sys.path.insert(0, str(repo_path))

from lamda_analyzer import LAMDaAnalyzer

start_time = time.time()
analyzer = LAMDaAnalyzer(chords_dir)

try:
    analyzer.analyze_all_files()
    elapsed = time.time() - start_time
    print(f"\n✅ Database built successfully in {elapsed:.1f} seconds")

    # データベースをGCSに保存
    db_path = data_dir.parent / "lamda_progressions.db"
    print(f"\nUploading database to GCS...")
    subprocess.run(
        ["gsutil", "cp", str(db_path), "gs://otocotoba/data/lamda/lamda_progressions.db"],
        check=True,
    )

    print("\n" + "=" * 70)
    print("SUCCESS! Database available at:")
    print("  gs://otocotoba/data/lamda/lamda_progressions.db")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
