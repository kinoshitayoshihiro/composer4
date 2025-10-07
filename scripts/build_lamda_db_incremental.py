#!/usr/bin/env python3
"""
LAMDa Incremental Database Builder
段階的にLAMDaデータベースを構築（テスト用から完全版まで）
"""

import sqlite3
import pickle
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LAMDaEvent:
    """LAMDaイベント（MIDIノート情報）"""

    time_delta: int
    duration: int
    patch: int
    pitch: int
    velocity: int


class IncrementalDBBuilder:
    """段階的データベース構築クラス"""

    # データサイズオプション
    SIZE_OPTIONS = {
        "test": 4,  # 10,000曲（テスト用）
        "small": 20,  # 50,000曲（実用版小）
        "medium": 40,  # 100,000曲（実用版中）
        "large": 80,  # 200,000曲（大規模）
        "full": 161,  # 405,000曲（完全版）
    }

    def __init__(self, lamda_root: Path, output_path: Path, size: str = "test"):
        self.lamda_root = lamda_root
        self.output_path = output_path
        self.size = size
        self.max_files = self.SIZE_OPTIONS.get(size, 4)

        # データディレクトリ
        self.chords_dir = lamda_root / "CHORDS_DATA"
        self.kilo_path = lamda_root / "KILO_CHORDS_DATA" / "LAMDa_KILO_CHORDS_DATA.pickle"
        self.signatures_path = lamda_root / "SIGNATURES_DATA" / "LAMDa_SIGNATURES_DATA.pickle"

        logger.info(
            f"Database size: {size} ({self.max_files} files, ~{self.max_files * 2500} songs)"
        )

    def parse_lamda_event(self, event_data: List) -> List[LAMDaEvent]:
        """LAMDaイベント配列をパース

        LAMDaフォーマット: [[time_delta, dur, patch, pitch, vel, ...], ...]
        """
        events = []

        # event_dataはネストされた配列
        for event_group in event_data:
            if not isinstance(event_group, list) or len(event_group) < 5:
                continue

            # 各グループから5要素ずつ取得
            i = 0
            while i + 4 < len(event_group):
                event = LAMDaEvent(
                    time_delta=event_group[i],
                    duration=event_group[i + 1],
                    patch=event_group[i + 2],
                    pitch=event_group[i + 3],
                    velocity=event_group[i + 4],
                )
                events.append(event)
                i += 5

        return events

    def extract_chord_progression(self, events: List[LAMDaEvent]) -> List[Dict[str, Any]]:
        """イベントからコード進行を抽出"""
        progressions = []
        current_time = 0

        for event in events:
            current_time += event.time_delta

            # コード的なノート（複数音の和音）を抽出
            # 簡易版：pitch情報からコードを推定
            progression_entry = {
                "time": current_time,
                "pitch": event.pitch,
                "duration": event.duration,
                "patch": event.patch,
                "velocity": event.velocity,
            }
            progressions.append(progression_entry)

        return progressions

    def build_database(self):
        """データベースを段階的に構築"""
        logger.info(f"Starting {self.size} database build...")

        # 出力ディレクトリ作成
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # データベース接続
        conn = sqlite3.connect(self.output_path)
        cursor = conn.cursor()

        # テーブル作成
        self._create_tables(cursor)

        # CHORDS_DATAファイル一覧取得
        chords_files = sorted(self.chords_dir.glob("LAMDa_CHORDS_DATA_*.pickle"))
        chords_files = chords_files[: self.max_files]  # 指定数まで制限

        logger.info(f"Processing {len(chords_files)} CHORDS_DATA files...")

        # 進捗カウンタ
        total_songs = 0
        total_progressions = 0

        # CHORDS_DATA処理
        for pickle_file in tqdm(chords_files, desc="CHORDS files"):
            songs_data = self._load_pickle(pickle_file)

            for song_entry in songs_data:
                hash_id = song_entry[0]
                event_data = song_entry[1]

                # イベントパース
                events = self.parse_lamda_event(event_data)

                # コード進行抽出
                progressions = self.extract_chord_progression(events)

                # データベースに挿入
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO progressions 
                    (hash_id, progression, total_events, chord_events, source_file, patch_distribution)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        hash_id,
                        json.dumps(progressions),
                        len(event_data),
                        len(progressions),
                        pickle_file.name,
                        json.dumps(self._get_patch_distribution(events)),
                    ),
                )

                total_songs += 1
                total_progressions += len(progressions)

        # KILO_CHORDS_DATA処理
        if self.kilo_path.exists():
            logger.info("Processing KILO_CHORDS_DATA...")
            self._process_kilo_data(cursor)

        # SIGNATURES_DATA処理
        if self.signatures_path.exists():
            logger.info("Processing SIGNATURES_DATA...")
            self._process_signatures_data(cursor)

        # インデックス作成
        logger.info("Creating indexes...")
        self._create_indexes(cursor)

        # コミット
        conn.commit()
        conn.close()

        # 統計情報
        logger.info("=" * 60)
        logger.info(f"✅ Database build complete!")
        logger.info(f"   Output: {self.output_path}")
        logger.info(f"   Size: {self.output_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(f"   Songs: {total_songs:,}")
        logger.info(f"   Progressions: {total_progressions:,}")
        logger.info("=" * 60)

        return {
            "output_path": str(self.output_path),
            "total_songs": total_songs,
            "total_progressions": total_progressions,
            "size_mb": self.output_path.stat().st_size / 1024 / 1024,
        }

    def _create_tables(self, cursor):
        """テーブル作成"""
        # progressionsテーブル（拡張版）
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS progressions (
                hash_id TEXT PRIMARY KEY,
                progression TEXT NOT NULL,
                total_events INTEGER,
                chord_events INTEGER,
                source_file TEXT,
                patch_distribution TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # kilo_sequencesテーブル
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS kilo_sequences (
                hash_id TEXT PRIMARY KEY,
                sequence TEXT NOT NULL,
                sequence_length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # signaturesテーブル
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS signatures (
                hash_id TEXT PRIMARY KEY,
                pitch_distribution TEXT NOT NULL,
                top_pitches TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # メタデータテーブル
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    def _create_indexes(self, cursor):
        """インデックス作成（検索高速化）"""
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_progressions_source ON progressions(source_file)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_progressions_events ON progressions(chord_events)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_kilo_length ON kilo_sequences(sequence_length)"
        )

    def _process_kilo_data(self, cursor):
        """KILO_CHORDS_DATA処理"""
        kilo_data = self._load_pickle(self.kilo_path)

        if isinstance(kilo_data, dict):
            for hash_id, sequence in tqdm(kilo_data.items(), desc="KILO data"):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO kilo_sequences 
                    (hash_id, sequence, sequence_length)
                    VALUES (?, ?, ?)
                """,
                    (
                        hash_id,
                        json.dumps(sequence) if isinstance(sequence, list) else str(sequence),
                        len(sequence) if hasattr(sequence, "__len__") else 0,
                    ),
                )

    def _process_signatures_data(self, cursor):
        """SIGNATURES_DATA処理"""
        signatures_data = self._load_pickle(self.signatures_path)

        if isinstance(signatures_data, dict):
            for hash_id, signature in tqdm(signatures_data.items(), desc="SIGNATURES data"):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO signatures 
                    (hash_id, pitch_distribution, top_pitches)
                    VALUES (?, ?, ?)
                """,
                    (
                        hash_id,
                        json.dumps(signature) if isinstance(signature, dict) else str(signature),
                        (
                            json.dumps(list(signature.keys())[:10])
                            if isinstance(signature, dict)
                            else "[]"
                        ),
                    ),
                )

    def _load_pickle(self, path: Path) -> Any:
        """Pickleファイルロード"""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path.name}: {e}")
            return None

    def _get_patch_distribution(self, events: List[LAMDaEvent]) -> Dict[int, int]:
        """楽器パッチの分布を取得"""
        distribution = {}
        for event in events:
            distribution[event.patch] = distribution.get(event.patch, 0) + 1
        return distribution


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Build LAMDa database incrementally")
    parser.add_argument(
        "--size",
        choices=["test", "small", "medium", "large", "full"],
        default="test",
        help="Database size (test=10k, small=50k, medium=100k, large=200k, full=405k songs)",
    )
    parser.add_argument(
        "--lamda-root",
        type=Path,
        default=Path("data/Los-Angeles-MIDI"),
        help="LAMDa dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output database path (default: output/lamda_{size}.db)",
    )

    args = parser.parse_args()

    # 出力パス決定
    if args.output is None:
        args.output = Path(f"output/lamda_{args.size}.db")

    # ビルダー作成
    builder = IncrementalDBBuilder(
        lamda_root=args.lamda_root, output_path=args.output, size=args.size
    )

    # データベース構築
    result = builder.build_database()

    print("\n" + "=" * 60)
    print(f"✅ Success! Database created:")
    print(f"   Path: {result['output_path']}")
    print(f"   Size: {result['size_mb']:.2f} MB")
    print(f"   Songs: {result['total_songs']:,}")
    print(f"   Progressions: {result['total_progressions']:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
