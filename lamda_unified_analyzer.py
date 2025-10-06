"""
LAMDa Unified Analyzer
=======================
Los Angeles MIDI Dataset の全データソースを統合的に活用するアナライザー

データソース:
- CHORDS_DATA: 詳細なMIDIイベント (時間, dur, patch, pitch, vel)
- KILO_CHORDS_DATA: 整数エンコードされたコード進行シーケンス
- SIGNATURES_DATA: 各楽曲の特徴量（ピッチ/コードの出現頻度）
- TOTALS_MATRIX: データセット全体の統計マトリックス
- META_DATA: ファイルメタデータ

設計思想:
- 各データは独立した離れ小島ではなく、連邦のように助け合う
- CHORDS_DATAから詳細分析、KILO_CHORDSで高速検索、SIGNATURESで特徴マッチング
"""

import sqlite3
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
from tqdm import tqdm
import music21


class LAMDaEvent:
    """LAMDaイベント構造

    フォーマット: [time_delta, dur1, patch1, pitch1, vel1, dur2, patch2, pitch2, vel2, ...]
    - time_delta: 前のイベントからの時間差 (単位は16分音符)
    - その後4要素ずつがノート情報: [duration, patch, pitch, velocity]
    """

    def __init__(self, raw_event: list):
        self.time_delta = raw_event[0] if len(raw_event) > 0 else 0
        self.notes = []

        # 4要素ずつグループ化
        note_data = raw_event[1:]
        for i in range(0, len(note_data), 4):
            if i + 3 < len(note_data):
                self.notes.append(
                    {
                        "duration": note_data[i],
                        "patch": note_data[i + 1],
                        "pitch": note_data[i + 2],
                        "velocity": note_data[i + 3],
                    }
                )

    def get_pitches(self) -> List[int]:
        """イベント内の全ピッチを取得"""
        return [n["pitch"] for n in self.notes if n["pitch"] < 128]

    def is_chord(self) -> bool:
        """和音かどうか（複数の音が同時）"""
        return len(self.notes) > 1

    def get_chord_notes(self) -> List[int]:
        """和音を構成する音（低い順）"""
        pitches = self.get_pitches()
        return sorted(pitches) if pitches else []


class LAMDaUnifiedAnalyzer:
    """LAMDa統合アナライザー"""

    def __init__(self, lamd_data_dir: Path):
        """
        Args:
            lamd_data_dir: data/Los-Angeles-MIDI/ ディレクトリのパス
        """
        self.data_dir = lamd_data_dir
        self.chords_dir = lamd_data_dir / "CHORDS_DATA"
        self.kilo_dir = lamd_data_dir / "KILO_CHORDS_DATA"
        self.signatures_dir = lamd_data_dir / "SIGNATURES_DATA"
        self.totals_dir = lamd_data_dir / "TOTALS_MATRIX"
        self.meta_dir = lamd_data_dir / "META_DATA"

        # キャッシュ
        self._kilo_data = None
        self._signatures_data = None
        self._totals_matrix = None

    def load_kilo_chords(self) -> list:
        """KILO_CHORDS_DATA をロード（キャッシュ付き）"""
        if self._kilo_data is None:
            kilo_file = self.kilo_dir / "LAMDa_KILO_CHORDS_DATA.pickle"
            with open(kilo_file, "rb") as f:
                self._kilo_data = pickle.load(f)
        return self._kilo_data

    def load_signatures(self) -> list:
        """SIGNATURES_DATA をロード（キャッシュ付き）"""
        if self._signatures_data is None:
            sig_file = self.signatures_dir / "LAMDa_SIGNATURES_DATA.pickle"
            with open(sig_file, "rb") as f:
                self._signatures_data = pickle.load(f)
        return self._signatures_data

    def load_totals_matrix(self) -> list:
        """TOTALS_MATRIX をロード（キャッシュ付き）"""
        if self._totals_matrix is None:
            totals_file = self.totals_dir / "LAMDa_TOTALS.pickle"
            with open(totals_file, "rb") as f:
                self._totals_matrix = pickle.load(f)
        return self._totals_matrix

    def analyze_chords_file(self, pickle_path: Path) -> Dict:
        """CHORDS_DATAファイルを解析してコード進行を抽出"""
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        progressions = []

        for sample in tqdm(data, desc=f"Analyzing {pickle_path.name}"):
            hash_id = sample[0]
            events = sample[1]

            # イベントをLAMDaEvent形式に変換
            lamd_events = [LAMDaEvent(e) for e in events]

            # 和音イベントのみ抽出
            chord_events = [e for e in lamd_events if e.is_chord()]

            if not chord_events:
                continue

            # コード進行として解析
            progression = []
            for event in chord_events[:50]:  # 最初の50和音まで
                chord_notes = event.get_chord_notes()
                if len(chord_notes) >= 3:  # 3音以上
                    # music21でコード名を推定
                    try:
                        chord_obj = music21.chord.Chord(chord_notes)
                        chord_name = chord_obj.pitchedCommonName
                        root = chord_obj.root().name
                        progression.append(
                            {
                                "chord": chord_name,
                                "root": root,
                                "notes": chord_notes,
                                "time_delta": event.time_delta,
                            }
                        )
                    except:
                        pass

            if progression:
                progressions.append(
                    {
                        "hash_id": hash_id,
                        "progression": progression,
                        "total_events": len(events),
                        "chord_events": len(chord_events),
                    }
                )

        return progressions

    def build_unified_database(self, db_path: Path):
        """統合データベースを構築

        テーブル構成:
        - progressions: コード進行（CHORDS_DATAから）
        - kilo_sequences: 整数シーケンス（KILO_CHORDS_DATAから）
        - signatures: 楽曲特徴（SIGNATURES_DATAから）
        - metadata: 楽曲情報（hash_idで紐付け）
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # テーブル作成
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS progressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT NOT NULL,
                progression TEXT NOT NULL,
                total_events INTEGER,
                chord_events INTEGER,
                source_file TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS kilo_sequences (
                hash_id TEXT PRIMARY KEY,
                sequence TEXT NOT NULL,
                sequence_length INTEGER
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS signatures (
                hash_id TEXT PRIMARY KEY,
                pitch_distribution TEXT NOT NULL,
                top_pitches TEXT
            )
        """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash_id ON progressions(hash_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON progressions(source_file)")

        # CHORDS_DATA 処理
        print("\n📁 Processing CHORDS_DATA...")
        chords_files = sorted(self.chords_dir.glob("*.pickle"))

        for pickle_path in chords_files:
            print(f"\n  Analyzing {pickle_path.name}...")
            progressions = self.analyze_chords_file(pickle_path)

            for prog_data in progressions:
                import json

                cursor.execute(
                    """
                    INSERT INTO progressions (hash_id, progression, total_events, chord_events, source_file)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        prog_data["hash_id"],
                        json.dumps(prog_data["progression"]),
                        prog_data["total_events"],
                        prog_data["chord_events"],
                        pickle_path.name,
                    ),
                )

            conn.commit()

        # KILO_CHORDS_DATA 処理
        print("\n📁 Processing KILO_CHORDS_DATA...")
        kilo_data = self.load_kilo_chords()

        for sample in tqdm(kilo_data, desc="  Loading sequences"):
            hash_id = sample[0]
            sequence = sample[1]

            cursor.execute(
                """
                INSERT OR REPLACE INTO kilo_sequences (hash_id, sequence, sequence_length)
                VALUES (?, ?, ?)
            """,
                (hash_id, str(sequence), len(sequence)),
            )

        conn.commit()

        # SIGNATURES_DATA 処理
        print("\n📁 Processing SIGNATURES_DATA...")
        signatures = self.load_signatures()

        for sample in tqdm(signatures, desc="  Loading signatures"):
            hash_id = sample[0]
            pitch_counts = sample[1]  # [[pitch, count], ...]

            # トップ10ピッチ
            top_pitches = sorted(pitch_counts, key=lambda x: x[1], reverse=True)[:10]

            cursor.execute(
                """
                INSERT OR REPLACE INTO signatures (hash_id, pitch_distribution, top_pitches)
                VALUES (?, ?, ?)
            """,
                (hash_id, str(pitch_counts), str(top_pitches)),
            )

        conn.commit()
        conn.close()

        print(f"\n✅ Unified database created: {db_path}")
        self.print_database_stats(db_path)

    def print_database_stats(self, db_path: Path):
        """データベース統計を表示"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM progressions")
        prog_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM kilo_sequences")
        kilo_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM signatures")
        sig_count = cursor.fetchone()[0]

        print("\n📊 Database Statistics:")
        print(f"  Total progressions: {prog_count:,}")
        print(f"  Total kilo sequences: {kilo_count:,}")
        print(f"  Total signatures: {sig_count:,}")

        conn.close()


if __name__ == "__main__":
    # テスト実行
    data_dir = Path("data/Los-Angeles-MIDI")
    db_path = Path("data/lamda_unified.db")

    analyzer = LAMDaUnifiedAnalyzer(data_dir)
    analyzer.build_unified_database(db_path)
