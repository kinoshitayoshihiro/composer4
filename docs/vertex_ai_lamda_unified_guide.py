# LAMDa Unified Database Build - Vertex AI Colab Enterprise Guide
# =================================================================
# 統合データベース構築ガイド (CHORDS + KILO + SIGNATURES + TOTALS)
#
# 実行環境: Vertex AI Colab Enterprise
# インスタンス: shimogami88-Default (e2-standard-4: 4 vCPU, 16GB RAM)
# リージョン: us-central1
# 推定実行時間: 90-120分
# 推定コスト: ¥30-50
#
# データソース統合:
# • CHORDS_DATA (15GB) → コード進行抽出
# • KILO_CHORDS_DATA (602MB) → 整数シーケンス
# • SIGNATURES_DATA (290MB) → 楽曲特徴量
# • TOTALS_MATRIX (33MB) → 統計マトリックス

# ============================================================================
# Cell 1: 環境確認と認証
# ============================================================================
import sys
from pathlib import Path
import subprocess
import time

print("=" * 80)
print("LAMDa Unified Database Builder")
print("統合データベース構築 (CHORDS + KILO + SIGNATURES)")
print("=" * 80)
print(f"Python: {sys.version}")
print(f"Working directory: {Path.cwd()}")

# GCS認証確認
print("\n📋 Checking GCS access...")
result = subprocess.run(["gsutil", "ls", "gs://otobon/lamda/"], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ GCS access OK")
    print("   Files in gs://otobon/lamda/:")
    for line in result.stdout.strip().split("\n")[:10]:
        print(f"     {line}")
else:
    print("❌ GCS access failed. Please authenticate:")
    print("   !gcloud auth application-default login")


# ============================================================================
# Cell 2: リポジトリクローンと依存関係インストール
# ============================================================================
print("\n" + "=" * 80)
print("[Step 1/7] Cloning repository and installing dependencies...")
print("=" * 80)

# 作業ディレクトリ
WORK_DIR = Path("/home/jupyter/lamda_unified_work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

REPO_PATH = WORK_DIR / "composer2-3"

# リポジトリクローン
if REPO_PATH.exists():
    print(f"✅ Repository exists: {REPO_PATH}")
    subprocess.run(["git", "-C", str(REPO_PATH), "pull"], check=True)
else:
    print(f"📥 Cloning repository to {REPO_PATH}...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/kinoshitayoshihiro/composer4.git",
            str(REPO_PATH),
        ],
        check=True,
    )

# 依存関係インストール
print("\n📦 Installing dependencies...")
subprocess.run(["pip", "install", "-q", "music21", "numpy", "tqdm"], check=True)

print("✅ Repository and dependencies ready")

# Python pathに追加
sys.path.insert(0, str(REPO_PATH))


# ============================================================================
# Cell 3: データダウンロード (CHORDS_DATA)
# ============================================================================
print("\n" + "=" * 80)
print("[Step 2/7] Downloading CHORDS_DATA (575MB compressed → 15GB extracted)")
print("=" * 80)

import tarfile

LAMDA_GCS_PATH = "gs://otobon/lamda/"
DATA_DIR = WORK_DIR / "data" / "Los-Angeles-MIDI"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# CHORDS_DATA.tar.gzダウンロード
tar_file = DATA_DIR / "CHORDS_DATA.tar.gz"

if tar_file.exists():
    print(f"✅ Already downloaded: {tar_file}")
else:
    print(f"📥 Downloading from {LAMDA_GCS_PATH}CHORDS_DATA.tar.gz...")
    start = time.time()
    subprocess.run(
        ["gsutil", "cp", f"{LAMDA_GCS_PATH}CHORDS_DATA.tar.gz", str(tar_file)], check=True
    )
    elapsed = time.time() - start
    print(f"✅ Downloaded in {elapsed:.1f}s ({tar_file.stat().st_size / 1024 / 1024:.1f} MB)")

# 解凍
chords_dir = DATA_DIR / "CHORDS_DATA"
if chords_dir.exists() and list(chords_dir.glob("*.pickle")):
    print(f"✅ Already extracted: {chords_dir}")
    print(f"   Files: {len(list(chords_dir.glob('*.pickle')))} pickle files")
else:
    print("📦 Extracting CHORDS_DATA (this may take 5-10 minutes)...")
    start = time.time()
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    elapsed = time.time() - start
    print(f"✅ Extracted in {elapsed:.1f}s")
    print(f"   Files: {len(list(chords_dir.glob('*.pickle')))} pickle files")


# ============================================================================
# Cell 4: データダウンロード (KILO, SIGNATURES, TOTALS)
# ============================================================================
print("\n" + "=" * 80)
print("[Step 3/7] Downloading KILO_CHORDS (602MB) + SIGNATURES (290MB) + TOTALS (33MB)")
print("=" * 80)

# KILO_CHORDS_DATA
kilo_dir = DATA_DIR / "KILO_CHORDS_DATA"
if kilo_dir.exists():
    print(f"✅ KILO_CHORDS_DATA exists: {kilo_dir}")
else:
    print("📥 Downloading KILO_CHORDS_DATA (602MB)...")
    start = time.time()
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", f"{LAMDA_GCS_PATH}KILO_CHORDS_DATA", str(DATA_DIR)], check=True
    )
    elapsed = time.time() - start
    print(f"✅ Downloaded in {elapsed:.1f}s")

# SIGNATURES_DATA
sig_dir = DATA_DIR / "SIGNATURES_DATA"
if sig_dir.exists():
    print(f"✅ SIGNATURES_DATA exists: {sig_dir}")
else:
    print("📥 Downloading SIGNATURES_DATA (290MB)...")
    start = time.time()
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", f"{LAMDA_GCS_PATH}SIGNATURES_DATA", str(DATA_DIR)], check=True
    )
    elapsed = time.time() - start
    print(f"✅ Downloaded in {elapsed:.1f}s")

# TOTALS_MATRIX
totals_dir = DATA_DIR / "TOTALS_MATRIX"
if totals_dir.exists():
    print(f"✅ TOTALS_MATRIX exists: {totals_dir}")
else:
    print("📥 Downloading TOTALS_MATRIX (33MB)...")
    start = time.time()
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", f"{LAMDA_GCS_PATH}TOTALS_MATRIX", str(DATA_DIR)], check=True
    )
    elapsed = time.time() - start
    print(f"✅ Downloaded in {elapsed:.1f}s")

print("\n✅ All data sources ready:")
print(f"   • CHORDS_DATA: {len(list(chords_dir.glob('*.pickle')))} files")
print(f"   • KILO_CHORDS_DATA: {kilo_dir}")
print(f"   • SIGNATURES_DATA: {sig_dir}")
print(f"   • TOTALS_MATRIX: {totals_dir}")


# ============================================================================
# Cell 5: 統合データベース構築 (メイン処理 - 60-90分)
# ============================================================================
print("\n" + "=" * 80)
print("[Step 4/7] Building unified database (estimated 60-90 minutes)")
print("=" * 80)

from lamda_unified_analyzer import LAMDaUnifiedAnalyzer

DB_PATH = WORK_DIR / "lamda_unified.db"

print(f"📊 Initializing LAMDaUnifiedAnalyzer...")
print(f"   Data directory: {DATA_DIR}")
print(f"   Output database: {DB_PATH}")

analyzer = LAMDaUnifiedAnalyzer(DATA_DIR)

print("\n🔨 Starting database build...")
print("   This will process:")
print("   • CHORDS_DATA → Extract chord progressions (longest step)")
print("   • KILO_CHORDS_DATA → Store integer sequences")
print("   • SIGNATURES_DATA → Store feature signatures")
print("   Progress bars will show detailed status...")

start = time.time()
analyzer.build_unified_database(DB_PATH)
elapsed = time.time() - start

print(f"\n✅ Database built successfully in {elapsed / 60:.1f} minutes")
print(f"   Size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")


# ============================================================================
# Cell 6: データベースアップロード
# ============================================================================
print("\n" + "=" * 80)
print("[Step 5/7] Uploading database to GCS...")
print("=" * 80)

gcs_db_path = f"{LAMDA_GCS_PATH}lamda_unified.db"

print(f"📤 Uploading to {gcs_db_path}...")
start = time.time()
subprocess.run(["gsutil", "cp", str(DB_PATH), gcs_db_path], check=True)
elapsed = time.time() - start

print(f"✅ Uploaded in {elapsed:.1f}s")
print(f"   GCS Path: {gcs_db_path}")


# ============================================================================
# Cell 7: 完了サマリー
# ============================================================================
print("\n" + "=" * 80)
print("🎉 LAMDa Unified Database Build COMPLETE!")
print("=" * 80)

print("\n📊 Database Contents:")
print("   • Chord progressions from CHORDS_DATA")
print("   • Integer sequences from KILO_CHORDS_DATA")
print("   • Feature signatures from SIGNATURES_DATA")
print("")
print(f"💾 Database Location:")
print(f"   Local: {DB_PATH}")
print(f"   GCS: {gcs_db_path}")
print(f"   Size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")
print("")
print("🔍 Database Statistics:")

import sqlite3

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM progressions")
prog_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM kilo_sequences")
kilo_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM signatures")
sig_count = cursor.fetchone()[0]

print(f"   • Total progressions: {prog_count:,}")
print(f"   • Total kilo sequences: {kilo_count:,}")
print(f"   • Total signatures: {sig_count:,}")

conn.close()

print("\n" + "=" * 80)
print("✅ You can now download the database for local use:")
print(f"   gsutil cp {gcs_db_path} ./lamda_unified.db")
print("=" * 80)
