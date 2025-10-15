#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合MIDIクリーニングツール
共通クリーニング → 楽器別クリーニング → 隔離/保存

Usage:
    python scripts/clean_midi.py \\
        --in data/lamda/raw/piano \\
        --out data/lamda/clean/piano \\
        --instrument piano \\
        --quarantine data/lamda/quarantine/piano \\
        --jobs 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import pretty_midi
from tqdm import tqdm

# クリーナーをインポート
sys.path.append(str(Path(__file__).parent))
from cleaners.common import (
    atomic_write_json,
    common_clean,
    compute_fileset_hash,
    make_provenance,
    seeded_rng,
    stable_list_midis,
)
from cleaners.piano import clean_piano
from cleaners.guitar import clean_guitar
from cleaners.bass import clean_bass
from cleaners.strings import clean_strings
from cleaners.drums import clean_drums

# 楽器別クリーナーのレジストリ
REGISTRY = {
    "piano": clean_piano,
    "guitar": clean_guitar,
    "bass": clean_bass,
    "strings": clean_strings,
    "drums": clean_drums,
}


def process_one_file(
    midi_path: Path,
    output_dir: Path,
    quarantine_dir: Path,
    instrument: str,
    force: bool,
) -> Tuple[bool, Dict]:
    """
    単一MIDIファイルを処理
    
    Returns:
        (success, metadata)
    """
    meta: Dict = {}
    
    # 再入可能性: 既存 .meta.json があり --force でなければスキップ
    relative_path = midi_path.relative_to(midi_path.parents[len(list(output_dir.parents))])
    meta_path = output_dir / relative_path.parent / (midi_path.stem + ".meta.json")
    
    if meta_path.exists() and not force:
        return (True, {"skipped": True, "reason": "already_processed"})
    
    try:
        # 1. 読み込み
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        
        # 2. 共通クリーニング
        pm, meta_common = common_clean(pm)
        meta.update(meta_common)
        
        # 3. 楽器別クリーニング
        instrument_cleaner = REGISTRY[instrument]
        pm, meta_inst, reason_codes = instrument_cleaner(pm)
        meta.update(meta_inst)
        meta["reason_codes"] = meta.get("reason_codes", []) + reason_codes
        
        # 4. 判定: 隔離 or 保存
        should_quarantine = _should_quarantine(meta["reason_codes"])
        
        # 5. 保存先決定 (階層構造を維持)
        if should_quarantine:
            output_path = quarantine_dir / relative_path
            meta_save_path = quarantine_dir / relative_path.parent / (midi_path.stem + ".meta.json")
        else:
            output_path = output_dir / relative_path
            meta_save_path = meta_path
        
        # ディレクトリ作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 6. MIDI保存
        pm.write(str(output_path))
        
        # 7. メタデータ保存 (原子的)
        atomic_write_json(meta, meta_save_path)
        
        return (not should_quarantine, meta)
    
    except Exception as e:
        # パースエラー
        meta["reason_codes"] = ["parse_error"]
        meta["error_message"] = str(e)
        
        # 元ファイルをコピー (階層維持)
        output_path = quarantine_dir / relative_path
        meta_save_path = quarantine_dir / relative_path.parent / (midi_path.stem + ".meta.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(midi_path.read_bytes())
        atomic_write_json(meta, meta_save_path)
        
        return (False, meta)


def _should_quarantine(reason_codes: list) -> bool:
    """隔離判定"""
    # ハードエラー
    if "hard_fail" in reason_codes:
        return True
    
    # 致命的な問題
    critical_codes = {
        "too_short",
        "too_few_notes",
        "parse_error",
        "tempo_change_excess",
    }
    if any(code in critical_codes for code in reason_codes):
        return True
    
    # 警告が多すぎる (3つ以上)
    if len(reason_codes) >= 3:
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="統合MIDIクリーニング (共通 + 楽器別)"
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="入力MIDIディレクトリ",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="出力MIDIディレクトリ",
    )
    parser.add_argument(
        "--instrument",
        required=True,
        choices=list(REGISTRY.keys()),
        help="楽器タイプ",
    )
    parser.add_argument(
        "--quarantine",
        required=True,
        help="隔離ディレクトリ (エラー/警告ファイル)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ドライラン: 件数のみ表示",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="並列処理数 (デフォルト: 1=直列)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存 .meta.json があっても再生成",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="cleaning-default",
        help="乱数シード",
    )
    
    args = parser.parse_args()
    
    # ディレクトリ準備
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    quarantine_dir = Path(args.quarantine)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    # 決定論的ファイル列挙
    midi_files = stable_list_midis(input_dir)
    
    if not midi_files:
        print(f"⚠️  No MIDI files found in {input_dir}")
        return 0
    
    # ドライラン
    if args.dry_run:
        print(f"[DRY RUN] {len(midi_files)} files under {input_dir}")
        print(f"   Instrument: {args.instrument}")
        print(f"   Output:     {output_dir}")
        print(f"   Quarantine: {quarantine_dir}")
        return 0
    
    # Fileset hash & Provenance
    fileset_hash = compute_fileset_hash(midi_files)
    provenance = make_provenance()
    
    print(f"🎵 Processing {len(midi_files)} MIDI files ({args.instrument})")
    print(f"   Input:        {input_dir}")
    print(f"   Output:       {output_dir}")
    print(f"   Quarantine:   {quarantine_dir}")
    print(f"   Jobs:         {args.jobs}")
    print(f"   Fileset Hash: {fileset_hash}")
    print()
    
    # 統計
    stats = {
        "total": len(midi_files),
        "success": 0,
        "quarantine": 0,
        "skipped": 0,
        "parse_error": 0,
        "reason_codes": {},
    }
    
    # メタインデックス準備
    index_path = output_dir / "meta_index.jsonl"
    if index_path.exists() and not args.force:
        # 既存インデックスを退避
        index_path.rename(output_dir / "meta_index.jsonl.bak")
    
    def _work(p: Path) -> Tuple[bool, Dict]:
        success, meta = process_one_file(
            p, output_dir, quarantine_dir, args.instrument, args.force
        )
        
        # スキップされた場合
        if meta.get("skipped"):
            return (True, meta)
        
        # Provenance追加
        meta.setdefault("schema_version", provenance["schema_version"])
        meta.setdefault("fileset_hash", fileset_hash)
        meta.setdefault("provenance", provenance)
        
        # メタインデックスに追記
        idx_line = {
            "path": str(p),
            "fileset_hash": fileset_hash,
            "reason_codes": meta.get("reason_codes", []),
            "tempo": meta.get("tempo"),
            "bars": meta.get("bars"),
            "notes": meta.get("notes"),
            "density": meta.get("density"),
        }
        
        with open(index_path, "a", encoding="utf-8") as w:
            w.write(json.dumps(idx_line, ensure_ascii=False) + "\n")
        
        return (success, meta)
    
    # 並列/直列の切り替え
    if args.jobs == 1:
        # 直列 (既存互換)
        for midi_path in tqdm(midi_files, desc="Cleaning"):
            success, meta = _work(midi_path)
            
            if meta.get("skipped"):
                stats["skipped"] += 1
            elif success:
                stats["success"] += 1
            else:
                stats["quarantine"] += 1
            
            # reason_codes 統計
            for code in meta.get("reason_codes", []):
                stats["reason_codes"][code] = stats["reason_codes"].get(code, 0) + 1
    else:
        # 並列
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_work, p): p for p in midi_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cleaning"):
                success, meta = future.result()
                
                if meta.get("skipped"):
                    stats["skipped"] += 1
                elif success:
                    stats["success"] += 1
                else:
                    stats["quarantine"] += 1
                
                for code in meta.get("reason_codes", []):
                    stats["reason_codes"][code] = stats["reason_codes"].get(code, 0) + 1
    
    # 結果レポート
    print()
    print("=" * 70)
    print("✅ Cleaning Complete")
    print("=" * 70)
    print(f"Total:      {stats['total']}")
    print(f"Success:    {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Skipped:    {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    print(f"Quarantine: {stats['quarantine']} ({stats['quarantine']/stats['total']*100:.1f}%)")
    print()
    
    if stats["reason_codes"]:
        print("Top Reason Codes:")
        sorted_codes = sorted(
            stats["reason_codes"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for code, count in sorted_codes[:10]:
            print(f"  - {code}: {count}")
    
    # JSONレポート保存
    report_path = output_dir.parent / f"{args.instrument}_clean_report.json"
    report = {
        **stats,
        "fileset_hash": fileset_hash,
        "provenance": provenance,
    }
    atomic_write_json(report, report_path)
    
    print()
    print(f"📊 Report saved: {report_path}")
    print(f"📊 Index saved:  {index_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
