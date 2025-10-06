#!/usr/bin/env python3
"""
LAMDa Unified Database Builder for Vertex AI Colab Enterprise
==============================================================
全LAMDaデータソースを統合した完全なデータベース構築スクリプト

データソース:
- CHORDS_DATA (15GB): 詳細MIDIイベント → コード進行抽出
- KILO_CHORDS_DATA (602MB): 整数シーケンス → 高速検索
- SIGNATURES_DATA (290MB): 特徴量 → 類似度マッチング
- TOTALS_MATRIX (33MB): 統計マトリックス → 正規化

実行環境: Vertex AI Colab Enterprise
推奨インスタンス: e2-standard-4 (4 vCPU, 16GB RAM)
推定実行時間: 90-120分 (全データ処理)
推定コスト: ¥30-50
"""

import sys
from pathlib import Path
import subprocess
import tarfile
import time

print("=" * 80)
print("LAMDa Unified Database Builder - Vertex AI Colab Enterprise")
print("全データソースを統合処理 (CHORDS + KILO + SIGNATURES + TOTALS)")
print("=" * 80)

# GCS設定
BUCKET_NAME = "otobon"
LAMDA_GCS_PATH = "gs://otobon/lamda/"

# 作業ディレクトリ
WORK_DIR = Path("/home/jupyter/lamda_work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# リポジトリパス
REPO_PATH = WORK_DIR / "composer2-3"

# データディレクトリ
DATA_DIR = WORK_DIR / "data" / "Los-Angeles-MIDI"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 出力DBパス
DB_PATH = WORK_DIR / "lamda_unified.db"


def step1_clone_repo():
    """ステップ1: リポジトリクローン"""
    print("\n" + "=" * 80)
    print("[1/7] Cloning repository...")
    print("=" * 80)

    if REPO_PATH.exists():
        print(f"✅ Repository already exists: {REPO_PATH}")
        print("   Pulling latest changes...")
        subprocess.run(["git", "-C", str(REPO_PATH), "pull"], check=True)
    else:
        print(f"📥 Cloning to {REPO_PATH}...")
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

    print(f"✅ Repository ready: {REPO_PATH}")


def step2_install_dependencies():
    """ステップ2: 依存関係インストール"""
    print("\n" + "=" * 80)
    print("[2/7] Installing dependencies...")
    print("=" * 80)

    print("📦 Installing: music21, numpy, tqdm...")
    subprocess.run(["pip", "install", "-q", "music21", "numpy", "tqdm"], check=True)

    print("✅ Dependencies installed")


def step3_download_chords_data():
    """ステップ3: CHORDS_DATAダウンロード"""
    print("\n" + "=" * 80)
    print("[3/7] Downloading CHORDS_DATA (575MB compressed)...")
    print("=" * 80)

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
        print(f"✅ Downloaded in {elapsed:.1f}s")

    print(f"   Size: {tar_file.stat().st_size / 1024 / 1024:.1f} MB")

    # 解凍
    chords_dir = DATA_DIR / "CHORDS_DATA"
    if chords_dir.exists():
        print(f"✅ Already extracted: {chords_dir}")
    else:
        print("📦 Extracting CHORDS_DATA...")
        start = time.time()
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(DATA_DIR)
        elapsed = time.time() - start
        print(f"✅ Extracted in {elapsed:.1f}s")


def step4_download_kilo_signatures():
    """ステップ4: KILO_CHORDS_DATA + SIGNATURES_DATAダウンロード"""
    print("\n" + "=" * 80)
    print("[4/7] Downloading KILO_CHORDS_DATA (602MB) + SIGNATURES_DATA (290MB)...")
    print("=" * 80)

    # KILO_CHORDS_DATA
    kilo_dir = DATA_DIR / "KILO_CHORDS_DATA"
    if kilo_dir.exists():
        print(f"✅ KILO_CHORDS_DATA already exists: {kilo_dir}")
    else:
        print("📥 Downloading KILO_CHORDS_DATA...")
        start = time.time()
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{LAMDA_GCS_PATH}KILO_CHORDS_DATA", str(DATA_DIR)],
            check=True,
        )
        elapsed = time.time() - start
        print(f"✅ Downloaded in {elapsed:.1f}s")

    # SIGNATURES_DATA
    sig_dir = DATA_DIR / "SIGNATURES_DATA"
    if sig_dir.exists():
        print(f"✅ SIGNATURES_DATA already exists: {sig_dir}")
    else:
        print("📥 Downloading SIGNATURES_DATA...")
        start = time.time()
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{LAMDA_GCS_PATH}SIGNATURES_DATA", str(DATA_DIR)],
            check=True,
        )
        elapsed = time.time() - start
        print(f"✅ Downloaded in {elapsed:.1f}s")


def step5_download_totals():
    """ステップ5: TOTALS_MATRIXダウンロード"""
    print("\n" + "=" * 80)
    print("[5/7] Downloading TOTALS_MATRIX (33MB)...")
    print("=" * 80)

    totals_dir = DATA_DIR / "TOTALS_MATRIX"
    if totals_dir.exists():
        print(f"✅ TOTALS_MATRIX already exists: {totals_dir}")
    else:
        print("📥 Downloading TOTALS_MATRIX...")
        start = time.time()
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{LAMDA_GCS_PATH}TOTALS_MATRIX", str(DATA_DIR)],
            check=True,
        )
        elapsed = time.time() - start
        print(f"✅ Downloaded in {elapsed:.1f}s")


def step6_build_database():
    """ステップ6: 統合データベース構築"""
    print("\n" + "=" * 80)
    print("[6/7] Building unified database...")
    print("       Processing: CHORDS + KILO + SIGNATURES")
    print("=" * 80)

    # Python pathにリポジトリを追加
    sys.path.insert(0, str(REPO_PATH))

    from lamda_unified_analyzer import LAMDaUnifiedAnalyzer

    print(f"📊 Initializing LAMDaUnifiedAnalyzer...")
    print(f"   Data directory: {DATA_DIR}")
    print(f"   Output database: {DB_PATH}")

    analyzer = LAMDaUnifiedAnalyzer(DATA_DIR)

    start = time.time()
    analyzer.build_unified_database(DB_PATH)
    elapsed = time.time() - start

    print(f"\n✅ Database built in {elapsed / 60:.1f} minutes")
    print(f"   Size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")


def step7_upload_database():
    """ステップ7: データベースをGCSにアップロード"""
    print("\n" + "=" * 80)
    print("[7/7] Uploading database to GCS...")
    print("=" * 80)

    gcs_db_path = f"{LAMDA_GCS_PATH}lamda_unified.db"

    print(f"📤 Uploading to {gcs_db_path}...")
    start = time.time()
    subprocess.run(["gsutil", "cp", str(DB_PATH), gcs_db_path], check=True)
    elapsed = time.time() - start

    print(f"✅ Uploaded in {elapsed:.1f}s")
    print(f"   GCS Path: {gcs_db_path}")


def main():
    """メイン実行"""
    overall_start = time.time()

    try:
        step1_clone_repo()
        step2_install_dependencies()
        step3_download_chords_data()
        step4_download_kilo_signatures()
        step5_download_totals()
        step6_build_database()
        step7_upload_database()

        overall_elapsed = time.time() - overall_start

        print("\n" + "=" * 80)
        print("🎉 SUCCESS! LAMDa Unified Database Build Complete")
        print("=" * 80)
        print(f"⏱️  Total time: {overall_elapsed / 60:.1f} minutes")
        print(f"💾 Database: gs://otobon/lamda/lamda_unified.db")
        print(f"📊 Size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")
        print("\n統合データベースには以下が含まれています:")
        print("  • CHORDS_DATA からコード進行抽出")
        print("  • KILO_CHORDS_DATA の整数シーケンス")
        print("  • SIGNATURES_DATA の楽曲特徴量")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
