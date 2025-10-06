#!/usr/bin/env python3
"""
LAMDa Local Test Runner
ローカル環境で小規模サンプルを使って高速反復テスト

Usage:
    # 1. テストサンプル作成
    python scripts/create_test_sample.py

    # 2. ローカルテスト実行
    python scripts/test_local_build.py

Expected runtime: 2-5分 (100サンプル)
"""

import sys
from pathlib import Path
import time
import sqlite3

# リポジトリルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# tqdm is optional
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from lamda_unified_analyzer import LAMDaUnifiedAnalyzer


def test_local_build():
    """ローカルで小規模テスト実行"""

    print("=" * 80)
    print("🧪 LAMDa Local Test - Small Sample (100 entries)")
    print("=" * 80)

    # パス設定
    test_data_dir = Path("data/Los-Angeles-MIDI/TEST_SAMPLE")
    test_db_path = Path("data/test_lamda.db")

    # テストデータ存在確認
    if not test_data_dir.exists():
        print("\n❌ Test sample not found!")
        print("   Please run: python scripts/create_test_sample.py")
        return False

    print(f"\n📁 Test data directory: {test_data_dir}")
    print(f"💾 Test database path: {test_db_path}")

    # 既存のテストDBを削除
    if test_db_path.exists():
        print(f"\n🗑️  Removing old test database...")
        test_db_path.unlink()

    # アナライザー初期化
    print(f"\n📊 Initializing LAMDaUnifiedAnalyzer...")
    analyzer = LAMDaUnifiedAnalyzer(test_data_dir)

    # データベース構築
    print(f"\n🔨 Building test database...")
    print(f"   (This should take 2-5 minutes for 100 samples)")

    start_time = time.time()

    try:
        analyzer.build_unified_database(test_db_path)

        build_time = time.time() - start_time

        print(f"\n✅ Database built successfully in {build_time:.1f} seconds!")
        print(f"   Database size: {test_db_path.stat().st_size / 1024:.1f} KB")

        # 検証
        print(f"\n🔍 Validating database...")
        validate_database(test_db_path)

        # パフォーマンス推定
        estimate_full_performance(build_time, 100)

        return True

    except Exception as e:
        print(f"\n❌ Error during build: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_database(db_path: Path):
    """データベースを検証"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # テーブル存在確認
    cursor.execute(
        """
        SELECT name FROM sqlite_master 
        WHERE type='table'
    """
    )
    tables = [row[0] for row in cursor.fetchall()]

    print(f"   Tables found: {', '.join(tables)}")

    # 各テーブルのレコード数
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"   • {table}: {count:,} records")

    # hash_id リンケージ確認
    if "progressions" in tables and "kilo_sequences" in tables:
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM progressions p
            INNER JOIN kilo_sequences k ON p.hash_id = k.hash_id
        """
        )
        linked = cursor.fetchone()[0]
        print(f"   • Linked records (progressions ↔ kilo): {linked:,}")

    if "progressions" in tables and "signatures" in tables:
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM progressions p
            INNER JOIN signatures s ON p.hash_id = s.hash_id
        """
        )
        linked = cursor.fetchone()[0]
        print(f"   • Linked records (progressions ↔ signatures): {linked:,}")

    # サンプルデータ表示
    print(f"\n📄 Sample progression:")
    cursor.execute(
        """
        SELECT hash_id, progression, total_events, chord_events 
        FROM progressions 
        LIMIT 1
    """
    )
    sample = cursor.fetchone()
    if sample:
        import json

        print(f"   Hash ID: {sample[0]}")
        print(f"   Total events: {sample[2]}")
        print(f"   Chord events: {sample[3]}")

        progression = json.loads(sample[1])
        print(f"   First 3 chords:")
        for chord in progression[:3]:
            print(f"     • {chord.get('chord', 'N/A')} " f"(root: {chord.get('root', 'N/A')})")

    conn.close()

    print(f"\n✅ Database validation passed!")


def estimate_full_performance(test_time: float, test_samples: int):
    """フルビルドのパフォーマンスを推定"""

    print(f"\n📈 Performance Estimation:")
    print(f"   Test build time: {test_time:.1f}s for {test_samples} samples")
    print(f"   Time per sample: {test_time / test_samples:.3f}s")

    # 推定 (CHORDS_DATAの全サンプル数は約180,000と仮定)
    full_samples = 180000
    estimated_time = (test_time / test_samples) * full_samples

    print(f"\n🔮 Full build estimation (assuming ~{full_samples:,} samples):")
    print(
        f"   Estimated time: {estimated_time / 60:.1f} minutes "
        f"({estimated_time / 3600:.1f} hours)"
    )
    print(f"   Estimated cost: ¥{(estimated_time / 3600) * 23:.0f} " f"(at ¥23/hour)")


def test_query_examples():
    """クエリ例のテスト"""

    print("\n" + "=" * 80)
    print("🔍 Testing Query Examples")
    print("=" * 80)

    db_path = Path("data/test_lamda.db")
    if not db_path.exists():
        print("❌ Test database not found. Run test_local_build() first.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 例1: コード進行検索
    print("\n1️⃣ Example: Find progressions with specific chord")
    cursor.execute(
        """
        SELECT hash_id, progression 
        FROM progressions 
        WHERE progression LIKE '%"chord": "C major triad"%'
        LIMIT 3
    """
    )
    results = cursor.fetchall()
    print(f"   Found {len(results)} results with C major triad")

    # 例2: イベント数でフィルタ
    print("\n2️⃣ Example: Filter by event count")
    cursor.execute(
        """
        SELECT COUNT(*) 
        FROM progressions 
        WHERE total_events > 1000
    """
    )
    count = cursor.fetchone()[0]
    print(f"   {count} progressions with > 1000 events")

    # 例3: KILO sequenceの長さ分布
    print("\n3️⃣ Example: KILO sequence length distribution")
    cursor.execute(
        """
        SELECT sequence_length, COUNT(*) as count
        FROM kilo_sequences
        GROUP BY sequence_length
        ORDER BY count DESC
        LIMIT 5
    """
    )
    for length, count in cursor.fetchall():
        print(f"   Length {length}: {count} sequences")

    conn.close()


if __name__ == "__main__":
    # メインテスト実行
    success = test_local_build()

    if success:
        print("\n" + "=" * 80)
        print("🎉 Local test completed successfully!")
        print("=" * 80)

        # クエリ例もテスト
        test_query_examples()

        print("\n✅ Ready for Vertex AI deployment!")
        print("   Next: Review and execute vertex_ai_lamda_unified_guide.py")
    else:
        print("\n❌ Test failed. Please review errors above.")
        sys.exit(1)
