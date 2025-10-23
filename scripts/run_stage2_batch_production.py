#!/usr/bin/env python3
"""
本番Stage2バッチ処理スクリプト

output/stage1配下の全MIDIファイルに対してstage2ラベリングを実行。
- Progress tracking（tqdm）
- エラーハンドリング（失敗時も継続）
- CSV集計（stage2_aggregate.csv）
- ログ出力（エラー詳細）
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# 新実装を使用
from scripts.lamda_v2.stage2_extractor import extract_stage2_metadata


def setup_logging(log_path: Path) -> logging.Logger:
    """ロガーのセットアップ"""
    logger = logging.getLogger("stage2_batch")
    logger.setLevel(logging.INFO)
    
    # ファイルハンドラ
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # コンソールハンドラ（WARNINGのみ）
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def collect_midi_files(input_dir: Path, pattern: str = "**/*.mid") -> List[Path]:
    """MIDI ファイルを再帰的に収集"""
    midi_files = []
    for ext in ["*.mid", "*.midi"]:
        midi_files.extend(sorted(input_dir.rglob(ext)))
    return sorted(set(midi_files))  # 重複排除


def process_batch(
    midi_files: List[Path],
    output_dir: Path,
    logger: logging.Logger,
    resume_from: Path | None = None,
) -> tuple[int, int, List[Dict[str, Any]]]:
    """
    バッチ処理メイン関数
    
    Returns:
        (success_count, failed_count, csv_rows)
    """
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = 0
    csv_rows: List[Dict[str, Any]] = []
    
    # Resume処理
    if resume_from and resume_from.exists():
        logger.info(f"Resuming from: {resume_from}")
        processed = set()
        with open(resume_from, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    processed.add(Path(line))
        logger.info(f"Skipping {len(processed)} already processed files")
        midi_files = [f for f in midi_files if f not in processed]
    
    # 進捗バー付きで処理
    with tqdm(total=len(midi_files), desc="Stage2 Processing", unit="file") as pbar:
        for midi_path in midi_files:
            try:
                # Stage2解析（新実装はmidi_pathのみ受け取る）
                start_time = time.time()
                meta = extract_stage2_metadata(midi_path)
                elapsed = time.time() - start_time
                
                # JSON出力
                json_out = json_dir / f"{midi_path.stem}.stage2.json"
                with json_out.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                
                # CSV行作成
                tempo_map = meta.get("tempo_map", [[0.0, 120.0]])
                timesig_map = meta.get("timesig_map", [[0, "4/4"]])
                bpm0 = tempo_map[0][1] if tempo_map else 120.0
                timesig0 = timesig_map[0][1] if timesig_map else "4/4"
                
                row = {
                    "file": midi_path.stem,
                    "dataset": midi_path.parent.name,  # lamda/loops/pop909/slakh
                    "bpm0": bpm0,
                    "timesig0": timesig0,
                    "n_downbeats": len(meta.get("downbeats_sec", [])),
                    "n_chords": len(meta.get("chordmap", {}).get("events", [])),
                    "n_sections": len(meta.get("sections_auto", {}).get("sections", [])),
                    "key_main": meta.get("key_modulations", {}).get("main_key", "C"),
                    "n_modulations": len(meta.get("key_modulations", {}).get("modulations", [])),
                    "swing_pct": meta.get("groove", {}).get("swing_pct", 0.0),
                    "backbeat_strength": meta.get("groove", {}).get("backbeat_strength", 0.5),
                    "controls_integrity": meta.get("controls", {}).get("integrity", 1.0),
                    "processing_time_sec": elapsed,
                }
                csv_rows.append(row)
                
                success += 1
                logger.info(f"✓ {midi_path.name} ({elapsed:.2f}s)")
                pbar.set_postfix({"success": success, "failed": failed})
            
            except Exception as e:
                failed += 1
                logger.error(f"✗ {midi_path.name}: {e}", exc_info=True)
                pbar.set_postfix({"success": success, "failed": failed})
            
            finally:
                pbar.update(1)
    
    return success, failed, csv_rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="本番Stage2バッチ処理（output/stage1 → output/stage2）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/stage1"),
        help="Input directory (default: output/stage1)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/stage2"),
        help="Output directory (default: output/stage2)",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/stage2_batch.log"),
        help="Log file path (default: logs/stage2_batch.log)",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume file (list of already processed files)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (count files only, no processing)",
    )
    args = ap.parse_args()
    
    # ディレクトリ準備
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # ロガー
    logger = setup_logging(args.log_file)
    logger.info(f"{'='*60}")
    logger.info(f"本番Stage2バッチ処理開始")
    logger.info(f"  Input:  {args.input_dir}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Log:    {args.log_file}")
    logger.info(f"{'='*60}")
    
    # MIDIファイル収集
    print(f"📂 Collecting MIDI files from {args.input_dir}...")
    midi_files = collect_midi_files(args.input_dir)
    print(f"✓ Found {len(midi_files)} MIDI files")
    
    if args.dry_run:
        print(f"\n[Dry Run] Would process {len(midi_files)} files")
        # データセット別の内訳
        from collections import Counter
        datasets = Counter(f.parent.name for f in midi_files)
        print(f"\nDataset breakdown:")
        for ds, count in sorted(datasets.items()):
            print(f"  {ds}: {count}")
        return 0
    
    # バッチ処理
    print(f"\n🚀 Starting batch processing...")
    start_total = time.time()
    success, failed, csv_rows = process_batch(
        midi_files,
        args.output_dir,
        logger,
        resume_from=args.resume_from,
    )
    elapsed_total = time.time() - start_total
    
    # CSV集計
    if csv_rows:
        csv_path = args.output_dir / "stage2_aggregate.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n✓ CSV aggregate saved: {csv_path}")
    
    # サマリ
    print(f"\n{'='*60}")
    print(f"📊 Batch Processing Summary")
    print(f"{'='*60}")
    print(f"  Total files:     {len(midi_files)}")
    print(f"  Success:         {success} ({success / len(midi_files) * 100:.1f}%)")
    print(f"  Failed:          {failed} ({failed / len(midi_files) * 100:.1f}%)")
    print(f"  Total time:      {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")
    print(f"  Avg time/file:   {elapsed_total / len(midi_files):.2f}s")
    print(f"  Log:             {args.log_file}")
    print(f"{'='*60}")
    
    logger.info(f"本番Stage2バッチ処理完了: {success} success, {failed} failed")
    
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
