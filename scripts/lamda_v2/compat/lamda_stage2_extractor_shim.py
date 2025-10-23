#!/usr/bin/env python3
"""
互換レイヤー（shim）: 旧lamda_stage2_extractor.py のCLI引数を受け取り、
新実装 scripts.lamda_v2.stage2_extractor に透過的に流します。

旧スクリプトを触らずに新実装へ移行できます。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# 新実装に依存
from scripts.lamda_v2.stage2_extractor import extract_stage2_metadata


def _iter_midi_files(p: Path) -> Iterator[Path]:
    """
    旧実装と同じセマンティクス：単一ファイルまたはディレクトリから.mid/.midiを列挙。
    """
    if p.is_file() and p.suffix.lower() in (".mid", ".midi"):
        yield p
    elif p.is_dir():
        # ソート済みで再帰的に列挙
        for q in sorted(p.rglob("*.mid")):
            yield q
        for q in sorted(p.rglob("*.midi")):
            yield q


def main() -> int:
    """
    旧CLI互換のエントリーポイント。
    
    引数:
        --input-dir: MIDIファイルまたはディレクトリ（旧セマンティクス）
        --output-dir: 出力ディレクトリ（json/を作成）
        --lamda-chords-dir: 外部chordmaps（オプション、なければ内部解析）
        --whitelist-validate: music21検証フラグ（互換のため維持、NO-OP）
        --emit-csv: aggregate → stage2_aggregate.csv を出力
        --print-summary: 処理状況を標準出力に表示
    
    Returns:
        0: 成功
        1: 失敗
    """
    ap = argparse.ArgumentParser(
        description="Compat shim: legacy lamda_stage2_extractor → lamda_v2.stage2_extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 単一ファイル処理
  python scripts/lamda_v2/compat/lamda_stage2_extractor_shim.py \\
    --input-dir tests/fixtures/midi/smoke.mid \\
    --output-dir output/stage2/smoke \\
    --emit-csv aggregate --print-summary
  
  # ディレクトリ一括処理
  python scripts/lamda_v2/compat/lamda_stage2_extractor_shim.py \\
    --input-dir output/stage1/test/clean \\
    --output-dir output/stage2/test \\
    --lamda-chords-dir data/lamda_chordmaps \\
    --emit-csv aggregate
        """
    )
    ap.add_argument(
        "--input-dir", 
        required=True, 
        help="MIDI file or directory of MIDIs (legacy semantics)"
    )
    ap.add_argument(
        "--output-dir", 
        required=True, 
        help="Output directory for .stage2.json files"
    )
    ap.add_argument(
        "--lamda-chords-dir", 
        default=None, 
        help="External chordmaps directory (optional, falls back to internal analysis)"
    )
    ap.add_argument(
        "--whitelist-validate", 
        type=int, 
        default=0, 
        help="music21 validation flag (kept for compatibility, NO-OP in new impl)"
    )
    ap.add_argument(
        "--emit-csv", 
        default=None, 
        choices=[None, "aggregate"], 
        help="Emit stage2_aggregate.csv summary"
    )
    ap.add_argument(
        "--print-summary", 
        action="store_true",
        help="Print processing status to stdout"
    )
    args = ap.parse_args()

    # パス準備
    in_path = Path(args.input_dir)
    if not in_path.exists():
        print(f"Error: Input path does not exist: {in_path}", file=sys.stderr)
        return 1
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    chord_dir = Path(args.lamda_chords_dir) if args.lamda_chords_dir else None
    if chord_dir and not chord_dir.exists():
        print(f"Warning: Chordmap directory does not exist: {chord_dir}", file=sys.stderr)
        chord_dir = None

    csv_rows: List[Dict[str, Any]] = []
    count = 0
    failed = 0

    # MIDI列挙と処理
    for midi_path in _iter_midi_files(in_path):
        try:
            # 新実装の呼び出し
            # 外部chordmapがあれば、新実装側で {base}.chordmap.json を探します
            chordmap_json = None
            if chord_dir:
                candidate = chord_dir / f"{midi_path.stem}.chordmap.json"
                if candidate.exists():
                    chordmap_json = candidate
            
            meta = extract_stage2_metadata(
                midi_path,
                chordmap_json=chordmap_json,
                chordmap_dir=chord_dir,  # fallback用
            )
            
            # 出力（旧構造：output/stage2/test/json/ を想定）
            json_dir = out_dir / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            
            base = midi_path.stem
            json_out = json_dir / f"{base}.stage2.json"
            with json_out.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            # CSV集計用
            if args.emit_csv == "aggregate":
                tempo_map = meta.get("tempo_map", [[0.0, 120.0]])
                timesig_map = meta.get("timesig_map", [[0, "4/4"]])
                bpm0 = tempo_map[0][1] if tempo_map else 120.0
                timesig0 = timesig_map[0][1] if timesig_map else "4/4"
                
                row = {
                    "file": base,
                    "bpm0": bpm0,
                    "timesig0": timesig0,
                    "n_downbeats": len(meta.get("downbeats_ql", meta.get("downbeats_sec", []))),
                    "n_chords": len(meta.get("chordmap", {}).get("events", [])),
                    "n_sections": len(meta.get("sections_auto", {}).get("sections", [])),
                    "key_main": meta.get("key_modulations", {}).get("main_key", "C"),
                    "n_modulations": len(meta.get("key_modulations", {}).get("modulations", [])),
                    "swing_pct": meta.get("groove", {}).get("swing_pct", 0.0),
                    "backbeat_strength": meta.get("groove", {}).get("backbeat_strength", 0.5),
                    "controls_integrity": meta.get("controls", {}).get("integrity", 1.0),
                }
                csv_rows.append(row)
            
            if args.print_summary:
                print(f"[OK] {midi_path.name} → {json_out.name}")
            
            count += 1
        
        except Exception as e:
            failed += 1
            if args.print_summary:
                print(f"[FAIL] {midi_path.name}: {e}", file=sys.stderr)
    
    # CSV出力
    if args.emit_csv == "aggregate" and csv_rows:
        csv_path = out_dir / "stage2_aggregate.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        if args.print_summary:
            print(f"[CSV] {csv_path}")
    
    # サマリ
    if args.print_summary:
        print(f"\n{'='*60}")
        print(f"Total processed: {count}")
        print(f"Failed: {failed}")
        print(f"Success rate: {count / (count + failed) * 100:.1f}%" if (count + failed) > 0 else "N/A")
        print(f"{'='*60}")
    
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
