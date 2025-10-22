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
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import pretty_midi
from tqdm import tqdm

# クリーナーをインポート
sys.path.append(str(Path(__file__).parent))
from cleaners.common import (
    ShardWriter,
    atomic_write_json,
    common_clean,
    compute_fileset_hash,
    extract_lamda_metadata,
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
    input_dir: Path | str,
    output_dir: Path | str,
    quarantine_dir: Path | str,
    instrument: str,
    force: bool,
    emit_meta_json: str = "off",
) -> Tuple[bool, Dict]:
    """
    単一MIDIファイルを処理
    
    Returns:
        (success, result_dict)
        result_dict = {
            "skipped": bool,
            "quarantined": bool,
            "lamda": dict (成功時のみ),
            "reason_codes": list,
            "meta": dict (デバッグ用),
        }
    """
    # Path型に変換
    input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    quarantine_dir = Path(quarantine_dir) if isinstance(quarantine_dir, str) else quarantine_dir
    
    result: Dict = {}
    
    # 再入可能性: 相対パスを解決
    try:
        relative_path = midi_path.relative_to(input_dir)
    except ValueError:
        # input_dir外のファイルの場合は名前のみ
        relative_path = Path(midi_path.name)
    
    meta_path = output_dir / relative_path.parent / (midi_path.stem + ".meta.json")
    quarantine_meta_path = quarantine_dir / relative_path.parent / (midi_path.stem + ".meta.json")
    cleaned_out = output_dir / relative_path
    
    # 再入可能性: emit_meta_json=off のときは .meta.json ではなく
    # 「クリーニング済み .mid の存在」でスキップ判定する
    if not force:
        already_processed = False
        already_quarantined = False
        
        if emit_meta_json == "off":
            # pickle直書き運用: .midの存在で判定
            already_processed = cleaned_out.exists()
            already_quarantined = (quarantine_dir / relative_path).exists()
        else:
            # 従来運用: .meta.jsonの存在で判定
            already_processed = meta_path.exists()
            already_quarantined = quarantine_meta_path.exists()
        
        if already_processed:
            # スキップ時も shard に LAMDA エントリを追加できるよう、
            # cleaned_out を再パースして lamda を作る（失敗したら None）
            lamda_entry = None
            try:
                pm2 = pretty_midi.PrettyMIDI(str(cleaned_out))
                lamda_entry = extract_lamda_metadata(
                    pm2,
                    input_path=str(midi_path),
                    output_path=str(cleaned_out),
                    base_dir=str(input_dir),
                    genre=instrument,  # 楽器名をgenreとして使用
                )
            except Exception:
                lamda_entry = None
            
            return (True, {
                "skipped": True,
                "reason": "already_processed",
                "input": str(midi_path),
                "cleaned_file": str(cleaned_out),
                "lamda": lamda_entry,  # ★ メイン側で shard に詰められる
            })
        
        if already_quarantined:
            return (False, {"skipped": True, "reason": "already_quarantined"})
    
    meta: Dict = {}
    
    try:
        # シンボリックリンクを実パスに解決
        resolved_path = midi_path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Symlink target not found: {resolved_path}")
        
        # 1. 読み込み
        pm = pretty_midi.PrettyMIDI(str(resolved_path))
        
        # 2. 共通クリーニング
        pm, meta_common = common_clean(pm)
        meta.update(meta_common)
        
        # 3. 楽器別クリーニング
        instrument_cleaner = REGISTRY[instrument]
        pm, meta_inst, reason_codes = instrument_cleaner(pm)
        meta.update(meta_inst)
        meta["reason_codes"] = meta.get("reason_codes", []) + reason_codes
        
        # 4. 判定: 隔離 or 保存
        should_quarantine = _should_quarantine(meta["reason_codes"], meta)
        
        # 5. 保存先決定 (階層構造を維持)
        if should_quarantine:
            output_path = quarantine_dir / relative_path
            meta_save_path = quarantine_dir / relative_path.parent / (midi_path.stem + ".meta.json")
            meta["quarantined"] = True
        else:
            output_path = output_dir / relative_path
            meta_save_path = meta_path
            meta["quarantined"] = False
            
            # 6. LAMDA互換メタデータを抽出 (成功ファイルのみ)
            lamda_meta = extract_lamda_metadata(
                pm,
                input_path=resolved_path,
                output_path=output_path,
                base_dir=output_dir,  # 出力ディレクトリを基準に相対パス計算
                genre=instrument,  # 楽器名をgenreとして使用
            )
            meta["lamda"] = lamda_meta
        
        # ディレクトリ作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 7. MIDI保存
        pm.write(str(output_path))
        
        # 8. LAMDAメタデータを結果に格納（常に）
        result["quarantined"] = should_quarantine
        result["reason_codes"] = meta.get("reason_codes", [])
        result["meta"] = meta
        
        if not should_quarantine:
            result["lamda"] = lamda_meta
        
        # 9. メタデータJSON保存（emit_meta_jsonモードに応じて）
        should_write_meta = (
            emit_meta_json == "on" or
            (emit_meta_json == "auto" and (should_quarantine or len(meta.get("reason_codes", [])) > 0))
        )
        
        if should_write_meta:
            atomic_write_json(meta, meta_save_path)
        
        return (not should_quarantine, result)
    
    except Exception as e:
        # パースエラー
        meta["reason_codes"] = ["parse_error"]
        meta["error_message"] = str(e)
        
        # シンボリックリンクを実パスに解決してコピー
        resolved_path = midi_path.resolve() if midi_path.is_symlink() else midi_path
        
        # 元ファイルをコピー (階層維持)
        output_path = quarantine_dir / relative_path
        meta_save_path = quarantine_dir / relative_path.parent / (midi_path.stem + ".meta.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if resolved_path.exists():
            output_path.write_bytes(resolved_path.read_bytes())
        
        # JSON出力（エラーは常に記録）
        if emit_meta_json != "off":
            atomic_write_json(meta, meta_save_path)
        
        result["quarantined"] = True
        result["reason_codes"] = meta.get("reason_codes", [])
        result["meta"] = meta
        
        return (False, result)


# グローバル変数 (並列処理用)
_global_input_dir = None
_global_output_dir = None
_global_quarantine_dir = None
_global_instrument = None
_global_force = None
_global_fileset_hash = None
_global_provenance = None
_global_index_path = None
_global_emit_meta_json = None  # 新規: .meta.json 出力モード


def _init_worker(input_dir, output_dir, quarantine_dir, instrument, force, fileset_hash, provenance, index_path, emit_meta_json):
    """ワーカープロセスの初期化関数"""
    global _global_input_dir, _global_output_dir, _global_quarantine_dir, _global_instrument, _global_force
    global _global_fileset_hash, _global_provenance, _global_index_path, _global_emit_meta_json
    
    _global_input_dir = input_dir
    _global_output_dir = output_dir
    _global_quarantine_dir = quarantine_dir
    _global_instrument = instrument
    _global_force = force
    _global_fileset_hash = fileset_hash
    _global_provenance = provenance
    _global_index_path = index_path
    _global_emit_meta_json = emit_meta_json


def _work_wrapper(p: Path) -> Tuple[bool, Dict]:
    """並列処理用のワーカー関数（トップレベルでpickle可能）"""
    # グローバル変数がNoneの場合のエラーハンドリング
    if _global_input_dir is None or _global_output_dir is None or _global_quarantine_dir is None:
        return (False, {
            "skipped": False,
            "quarantined": True,
            "reason_codes": ["GLOBAL_VAR_ERROR"],
            "meta": {"error": "Global variables not initialized in worker process"}
        })
    
    success, result = process_one_file(
        p, _global_input_dir, _global_output_dir, _global_quarantine_dir, 
        _global_instrument, _global_force, _global_emit_meta_json
    )
    
    # スキップされた場合
    if result.get("skipped"):
        return (success, result)
    
    # Provenance追加
    if "meta" in result:
        result["meta"].setdefault("schema_version", _global_provenance["schema_version"])
        result["meta"].setdefault("fileset_hash", _global_fileset_hash)
        result["meta"].setdefault("provenance", _global_provenance)
    
    # メタインデックスに追記
    meta = result.get("meta", {})
    idx_line = {
        "path": str(p),
        "fileset_hash": _global_fileset_hash,
        "reason_codes": result.get("reason_codes", []),
        "tempo": meta.get("tempo"),
        "bars": meta.get("bars"),
        "notes": meta.get("notes"),
        "density": meta.get("density"),
    }
    
    with open(_global_index_path, "a", encoding="utf-8") as w:
        w.write(json.dumps(idx_line, ensure_ascii=False) + "\n")
    
    return (success, result)


def _should_quarantine(reason_codes: list, meta: Dict) -> bool:
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
    """メイン処理"""
    # グローバル変数（シグナルハンドラーからアクセスするため）
    global _shard_writer_instance
    _shard_writer_instance = None
    
    def signal_handler(sig, frame):
        """Ctrl+C でも安全にpickleを保存"""
        print("\n\n⚠️  Interrupt received. Saving progress...", flush=True)
        if _shard_writer_instance:
            try:
                if hasattr(_shard_writer_instance, 'buffer') and _shard_writer_instance.buffer:
                    print(f"💾 Flushing {len(_shard_writer_instance.buffer)} entries to pickle...", flush=True)
                    _shard_writer_instance.flush()
                    print("✅ Progress saved successfully!", flush=True)
                else:
                    print("ℹ️  No pending data to save.", flush=True)
            except Exception as e:
                print(f"❌ Error saving progress: {e}", flush=True)
        print("Exiting...", flush=True)
        sys.exit(0)
    
    # シグナルハンドラー登録
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="統合MIDIクリーニングツール"
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
        "--pickle-out",
        required=True,
        help="Pickle出力ディレクトリ（シャード＋インデックス）",
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
        "--emit-meta-json",
        choices=["off", "auto", "on"],
        default="off",
        help=".meta.json 出力モード: off=出さない(推奨), auto=隔離/警告のみ, on=全ファイル",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Shardあたりのファイル数 (デフォルト: 5000、推奨)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="既存シャードから再開（途中停止対応）",
    )
    parser.add_argument(
        "--subfolder-id",
        type=str,
        default=None,
        help="サブフォルダID (0-9, a-f) - LAMDA専用モード",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="cleaning-default",
        help="乱数シード",
    )
    parser.add_argument(
        "--subfolder-mode",
        type=str,
        default="",
        help="サブフォルダモード: 単一pickleファイルを生成 (例: '0', 'a', 'f')",
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
    
    # グローバル変数設定（並列処理用）
    global _global_input_dir, _global_output_dir, _global_quarantine_dir, _global_instrument, _global_force
    global _global_fileset_hash, _global_provenance, _global_index_path, _global_emit_meta_json
    
    _global_input_dir = input_dir
    _global_output_dir = output_dir
    _global_quarantine_dir = quarantine_dir
    _global_instrument = args.instrument
    _global_force = args.force
    _global_fileset_hash = fileset_hash
    _global_provenance = provenance
    _global_index_path = index_path
    _global_emit_meta_json = args.emit_meta_json
    
    # ShardWriter初期化（pickle-outが指定されている場合のみ）
    shard_writer = None
    subfolder_mode = args.subfolder_id.strip() if args.subfolder_id else ""
    
    if args.pickle_out:
        pickle_out_dir = Path(args.pickle_out)
        pickle_out_dir.mkdir(parents=True, exist_ok=True)
        
        if subfolder_mode:
            # サブフォルダモード: 単一pickleファイルを生成
            # shard_size を大きくして1ファイルにまとめる
            shard_writer = ShardWriter(
                out_dir=pickle_out_dir,
                instrument=args.instrument,
                shard_size=len(midi_files) + 1000,  # 全ファイルが1シャードに収まるように
                resume=args.resume,
                subfolder_id=subfolder_mode,  # サブフォルダIDを渡す
            )
            print(f"📦 ShardWriter initialized (SUBFOLDER MODE: {subfolder_mode})")
            print(f"   Output: {pickle_out_dir}/{args.instrument}_shard_{subfolder_mode}.pickle")
            print(f"   Files to process: {len(midi_files)}")
        else:
            # 通常モード: 複数シャードに分割
            shard_writer = ShardWriter(
                out_dir=pickle_out_dir,
                instrument=args.instrument,
                shard_size=args.shard_size,
                resume=args.resume,
            )
            print(f"📦 ShardWriter initialized: {pickle_out_dir}")
            print(f"   Shard size: {args.shard_size}")
            print(f"   Resume mode: {args.resume}")
        
        # グローバル変数に保存（シグナルハンドラーから参照）
        _shard_writer_instance = shard_writer
        print()
    
    # 並列/直列の切り替え
    if args.jobs == 1:
        # 直列 (既存互換)
        for midi_path in tqdm(midi_files, desc="Cleaning"):
            success, meta = _work_wrapper(midi_path)
            
            if meta.get("skipped"):
                stats["skipped"] += 1
                # スキップされた場合でもLAMDAエントリがあればshardに追加
                if shard_writer and "lamda" in meta and meta["lamda"] is not None:
                    shard_writer.add(meta["lamda"])
            elif success:
                stats["success"] += 1
                # ShardWriterにLAMDAメタデータを追加
                if shard_writer and "lamda" in meta:
                    shard_writer.add(meta["lamda"])
            else:
                stats["quarantine"] += 1
            
            # reason_codes 統計
            for code in meta.get("reason_codes", []):
                stats["reason_codes"][code] = stats["reason_codes"].get(code, 0) + 1
    else:
        # 並列
        checkpoint_interval = 500  # 500ファイルごとに自動保存
        processed_count = 0
        
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_worker,
            initargs=(
                input_dir, output_dir, quarantine_dir, args.instrument, args.force,
                fileset_hash, provenance, index_path, args.emit_meta_json
            )
        ) as executor:
            futures = {executor.submit(_work_wrapper, p): p for p in midi_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cleaning"):
                success, meta = future.result()
                processed_count += 1
                
                if meta.get("skipped"):
                    stats["skipped"] += 1
                    # スキップされた場合でもLAMDAエントリがあればshardに追加
                    if shard_writer and "lamda" in meta and meta["lamda"] is not None:
                        shard_writer.add(meta["lamda"])
                elif success:
                    stats["success"] += 1
                    # ShardWriterにLAMDAメタデータを追加
                    if shard_writer and "lamda" in meta:
                        shard_writer.add(meta["lamda"])
                else:
                    stats["quarantine"] += 1
                
                for code in meta.get("reason_codes", []):
                    stats["reason_codes"][code] = stats["reason_codes"].get(code, 0) + 1
                
                # チェックポイント: 500ファイルごとに自動保存
                if shard_writer and processed_count % checkpoint_interval == 0:
                    if shard_writer.buffer:
                        print(f"\n💾 Checkpoint: Saving {len(shard_writer.buffer)} entries...", flush=True)
                        shard_writer.flush()
                        print(f"✅ Checkpoint saved (processed: {processed_count}/{len(futures)})", flush=True)
    
    # ShardWriter の最終処理（flush & index生成）
    index_pickle_path = None
    if shard_writer:
        print()
        print("=" * 70)
        print("🔨 Finalizing LAMDA sharded pickles...")
        print("=" * 70)
        
        # 残りのバッファをflush
        if shard_writer.buffer:
            shard_writer.flush()
        
        # サブフォルダモードではインデックス不要
        if not subfolder_mode:
            # インデックス生成（通常モードのみ）
            index_pickle_path = shard_writer.write_index()
            
            total_shards = shard_writer.shard_idx + (1 if shard_writer.buffer else 0)
            print()
            print(f"📚 Index pickle saved: {index_pickle_path}")
            print(f"   Format: LAMDA Stage2 compatible")
            print(f"   Total shards: {total_shards}")
            print(f"   Shard size:   {args.shard_size}")
            print()
            print("✅ Ready for Stage2 processing:")
            print(f"   python scripts/lamda_stage2_extractor.py \\")
            print(f"       --metadata-index {index_pickle_path} \\")
            print(f"       --input-dir {output_dir.parent}")
        else:
            # サブフォルダモード: 単一pickleファイルのパスを表示
            pickle_file = pickle_out_dir / f"{args.instrument}_shard_{subfolder_mode}.pickle"
            print()
            print(f"📦 Subfolder pickle saved: {pickle_file}")
            print(f"   Entries: {len(shard_writer.buffer) if hasattr(shard_writer, 'buffer') else 'flushed'}")
            print(f"   Format: LAMDA Stage2 compatible (single shard)")
        
        print("=" * 70)
    
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
    if index_pickle_path:
        print(f"📦 Pickle index: {index_pickle_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
