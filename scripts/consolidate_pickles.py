#!/usr/bin/env python3
"""
Pickle統合スクリプト - output/ → LOCAL_LAMDA/pickles/

Features:
- output/ の一時Pickleを LOCAL_LAMDA/pickles/ に統合
- 楽器別・データセット別にディレクトリ分離
- 既存Pickleのバックアップ
- Index再生成

Usage:
    python scripts/consolidate_pickles.py \
        --output-dir output \
        --target-dir data/Los-Angeles-MIDI/LOCAL_LAMDA/pickles \
        --backup \
        --verbose
"""

import argparse
import json
import logging
import pickle
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ========== Config ==========


class ConsolidationConfig:
    """統合設定"""

    # 統合対象ディレクトリマッピング（既存データセット優先）
    source_mappings = {
        # WAV Datasets (10/30作成)
        "wav_metadata/moisesdb": "moisesdb_wav",
        "wav_metadata/musdb18": "musdb18_wav",
        # Rhythm AI MIDI (10/28作成)
        "rhythm_ai/drumclean_metadata": "drums_midi",
        "rhythm_ai/groove_metadata": "groove_midi",
        "rhythm_ai/egmd_metadata": "egmd_midi",
        # Rhythm AI WAV (10/31作成)
        "rhythm_wav/groove_metadata": "groove_wav",
    }

    # バックアップディレクトリ名
    backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")


# ========== Consolidator ==========


class PickleConsolidator:
    """Pickle統合プロセッサ"""

    def __init__(
        self,
        output_dir: Path,
        target_dir: Path,
        config: ConsolidationConfig,
        backup: bool = True,
        verbose: bool = True,
    ):
        self.output_dir = output_dir
        self.target_dir = target_dir
        self.config = config
        self.backup = backup
        self.verbose = verbose

        # 統計
        self.stats = defaultdict(
            lambda: {"pickles_moved": 0, "total_records": 0, "backup_created": False}
        )

    def consolidate(self):
        """全統合実行"""
        print(f"\n{'='*70}")
        print(f"Pickle Consolidation")
        print(f"{'='*70}")
        print(f"Source: {self.output_dir}")
        print(f"Target: {self.target_dir}")
        print(f"{'='*70}\n")

        # ターゲットディレクトリ作成
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # 各カテゴリを統合
        for source_subdir, target_subdir in self.config.source_mappings.items():
            source_path = self.output_dir / source_subdir
            target_path = self.target_dir / target_subdir

            if not source_path.exists():
                if self.verbose:
                    print(f"⚠️  Skipping (not found): {source_path}")
                continue

            print(f"\n📦 Consolidating: {source_subdir} → {target_subdir}")

            # バックアップ
            if self.backup and target_path.exists():
                self._backup_existing(target_path, target_subdir)

            # コピー実行
            self._consolidate_directory(source_path, target_path, target_subdir)

        # サマリー表示
        self._print_summary()

    def _backup_existing(self, target_path: Path, category: str):
        """既存Pickleをバックアップ"""
        backup_dir = self.target_dir.parent / f"pickles_backup_{self.config.backup_suffix}"
        backup_path = backup_dir / category

        backup_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(target_path, backup_path, dirs_exist_ok=True)

        self.stats[category]["backup_created"] = True

        print(f"   💾 Backup created: {backup_path}")

    def _consolidate_directory(self, source_path: Path, target_path: Path, category: str):
        """ディレクトリ統合"""
        target_path.mkdir(parents=True, exist_ok=True)

        # Pickleファイル収集
        pickle_files = list(source_path.glob("*.pkl")) + list(source_path.glob("*.pickle"))

        total_records = 0

        for pkl_file in pickle_files:
            # コピー
            target_file = target_path / pkl_file.name
            shutil.copy2(pkl_file, target_file)

            # レコード数カウント（Indexファイル以外）
            if "index" not in pkl_file.name:
                try:
                    with open(pkl_file, "rb") as f:
                        data = pickle.load(f)

                    if isinstance(data, list):
                        total_records += len(data)
                    elif isinstance(data, dict) and "loops" in data:
                        total_records += len(data["loops"])
                except:
                    pass

            self.stats[category]["pickles_moved"] += 1

        self.stats[category]["total_records"] = total_records

        print(f"   ✅ Moved {len(pickle_files)} pickle files")
        print(f"   📊 Total records: {total_records}")

    def _print_summary(self):
        """統計サマリー表示"""
        print(f"\n{'='*70}")
        print(f"Consolidation Summary")
        print(f"{'='*70}")

        total_pickles = 0
        total_records = 0

        for category, stats in self.stats.items():
            print(f"\n{category}:")
            print(f"  Pickles moved:  {stats['pickles_moved']}")
            print(f"  Total records:  {stats['total_records']}")

            if stats["backup_created"]:
                print(f"  Backup:         ✅")

            total_pickles += stats["pickles_moved"]
            total_records += stats["total_records"]

        print(f"\n{'='*70}")
        print(f"Total pickles:  {total_pickles}")
        print(f"Total records:  {total_records}")
        print(f"{'='*70}\n")

        # サマリーJSON保存
        summary_path = self.target_dir / "consolidation_summary.json"

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "source_dir": str(self.output_dir),
                    "target_dir": str(self.target_dir),
                    "stats": dict(self.stats),
                    "total_pickles": total_pickles,
                    "total_records": total_records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"📄 Summary saved: {summary_path}")


# ========== CLI ==========


def main():
    parser = argparse.ArgumentParser(
        description="Pickle統合スクリプト - output/ → LOCAL_LAMDA/pickles/"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Source directory (default: output)"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/Los-Angeles-MIDI/LOCAL_LAMDA/pickles"),
        help="Target directory (default: data/Los-Angeles-MIDI/LOCAL_LAMDA/pickles)",
    )
    parser.add_argument(
        "--backup", action="store_true", default=True, help="Backup existing pickles"
    )
    parser.add_argument("--no-backup", action="store_false", dest="backup", help="Skip backup")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    # ロギング設定
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 統合実行
    config = ConsolidationConfig()

    consolidator = PickleConsolidator(
        output_dir=args.output_dir,
        target_dir=args.target_dir,
        config=config,
        backup=args.backup,
        verbose=args.verbose,
    )

    consolidator.consolidate()


if __name__ == "__main__":
    main()
