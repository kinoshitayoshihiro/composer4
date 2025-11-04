#!/usr/bin/env python3
"""
Rhythm AI Song Package Generator

MIDIガイド（drumclean_midi）からSong Package生成

Usage:
    python scripts/generate_rhythm_song_packages.py \
        --midi-root data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/drumclean_midi \
        --output-root output/rhythm_ai/song_packages \
        --index-out output/rhythm_ai/rhythm_song_packages_index.csv \
        --jobs 8
"""

import argparse
import csv
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

try:
    import mido  # for MIDI length / tempo map (best effort)
except Exception:
    mido = None


# Helper to parse meta from filename
def parse_meta_from_stem(stem: str) -> Tuple[Optional[int], Optional[str], str]:
    """
    Parse tempo (BPM), time signature (e.g., '4/4'), and genre from filename stem.

    強化点:
      - genre 抽出時に 'fill','beat','groove' を除外（stopwords）
      - tempo 抽出で `_110_beat_` / `_95_fill_` / `120bpm` / `[number]_[TS]` を認識

    Examples:
      20_latin-brazilian-baiao_110_beat_4-4 → tempo=110, ts=4/4, genre=latin-brazilian-baiao
      104_funk_95_fill_4-4 → tempo=95, ts=4/4, genre=funk
      ballad_72_3-4 → tempo=72, ts=3/4, genre=ballad
    """
    tempo: Optional[int] = None
    time_sig: Optional[str] = None
    genre: str = "unknown"

    tokens = stem.split("_")
    lower = [t.lower() for t in tokens]

    # 1) Time Signature（末尾側を優先）
    ts_idx: Optional[int] = None
    for i in range(len(lower) - 1, -1, -1):
        m = re.fullmatch(r"(\d+)[-/](\d+)", lower[i])
        if m:
            num, denom = int(m.group(1)), int(m.group(2))
            if num > 0 and denom > 0:
                time_sig = f"{num}/{denom}"
                ts_idx = i
                break

    # 2) Tempo（多段階フォールバック）
    tempo_idx: Optional[int] = None
    search_end = ts_idx if ts_idx is not None else len(lower)

    # 2-1) 明示的BPM表記（120bpm / 140BPM）
    for i, tok in enumerate(lower[:search_end]):
        m = re.fullmatch(r"(\d+)\s*bpm", tok, flags=re.IGNORECASE)
        if m:
            tempo = int(m.group(1))
            tempo_idx = i
            break

    # 2-2) 数字 + マーカー（_110_beat_ / _95_fill_）
    if tempo is None:
        markers = {"beat", "beats", "fill", "fills"}
        for i in range(search_end - 1):
            if re.fullmatch(r"\d{2,3}", lower[i]) and lower[i + 1] in markers:
                tempo = int(lower[i])
                tempo_idx = i
                break

    # 2-3) 数字直後にTS（_72_3-4）
    if tempo is None and ts_idx is not None and ts_idx - 1 >= 0:
        if re.fullmatch(r"\d{2,3}", lower[ts_idx - 1]):
            tempo = int(lower[ts_idx - 1])
            tempo_idx = ts_idx - 1

    # 2-4) フォールバック: TS周辺5トークン内の妥当値（40-260 BPM）
    if tempo is None:
        rng_end = search_end
        rng_start = max(0, rng_end - 5)
        for i in range(rng_end - 1, rng_start - 1, -1):
            if re.fullmatch(r"\d{2,3}", lower[i]):
                v = int(lower[i])
                if 40 <= v <= 260:
                    tempo = v
                    tempo_idx = i
                    break

    # 3) Genre（stopwords除外強化）
    stop = {"beat", "beats", "fill", "fills", "groove"}
    genre_search_end = min([x for x in [ts_idx, tempo_idx] if x is not None], default=len(lower))
    for i in range(genre_search_end - 1, -1, -1):
        tok = lower[i]
        # スキップ条件: 空/数字/TS/bpm/stopwords
        if not tok or re.fullmatch(r"\d+", tok):
            continue
        if re.fullmatch(r"(\d+)[-/](\d+)", tok):
            continue
        if tok in stop or tok == "bpm":
            continue
        # 英字を含むトークンを採択（複合語OK: hiphop-groove6, latin-brazilian-baiao）
        if re.search(r"[a-z]", tok):
            genre = tokens[i]  # 元の表記で返す
            break

    return tempo, time_sig, genre


def process_midi_file(
    midi_file: Path,
    midi_root: Path,
    output_root: Path,
    default_bars: int,
    skip_existing: bool = False,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """MIDIファイルからSong Package生成"""
    try:
        # ファイル名からメタデータ抽出
        # 例: drummer1_session1_001_funk_120bpm_4-4.mid
        stem = midi_file.stem
        # Parse filename metadata
        tempo_from_name, ts_from_name, genre = parse_meta_from_stem(stem)
        # song_id生成
        song_id = stem

        # 出力ディレクトリ
        song_dir = output_root / song_id
        song_dir.mkdir(parents=True, exist_ok=True)

        package_path = song_dir / "song_package.yaml"
        bars_path = song_dir / f"{song_id}.bars.parquet"
        if skip_existing and package_path.exists() and bars_path.exists():
            return (
                True,
                "skip_existing",
                {
                    "song_id": song_id,
                    "package_path": str(package_path),
                    "bars_path": str(bars_path),
                    "tempo": None,
                    "time_signature": None,
                },
            )

        # song_package.yaml生成
        package_data = {
            "song_id": song_id,
            "source_midi": str(midi_file),
            "dataset": "groove_midi",
            "paths": {"midi": str(midi_file.relative_to(midi_root)), "package_dir": str(song_dir)},
        }

        # Tempo/TimeSig from filename (fallback defaults)
        if tempo_from_name is not None:
            tempo = tempo_from_name
        else:
            tempo = 120
        time_sig = ts_from_name or "4/4"
        package_data["tempo"] = tempo
        package_data["time_signature"] = time_sig
        package_data["genre"] = genre

        # is_fill / is_beat フラグ（ファイル名パターンから判定）
        stem_lower = stem.lower()
        package_data["is_fill"] = "_fill_" in stem_lower or stem_lower.endswith("_fill")
        package_data["is_beat"] = "_beat_" in stem_lower or stem_lower.endswith("_beat")

        # YAML保存
        with open(package_path, "w", encoding="utf-8") as f:
            yaml.dump(package_data, f, default_flow_style=False, allow_unicode=True)

        # === bars.parquet生成（メーター対応・長さに基づく概算） ===
        # バー長 = (quarter-notes per bar) * (60 / tempo)
        try:
            num, denom = map(int, time_sig.split("/"))
        except Exception:
            num, denom = 4, 4
        quarter_per_bar = 4.0 * num / float(denom)
        bar_duration = (quarter_per_bar * 60.0) / float(tempo if tempo else 120)

        # MIDI全体の長さ（秒）を可能なら取得（midoが無ければ推定8小節）
        total_length_sec = None
        if mido is not None:
            try:
                mf = mido.MidiFile(str(midi_file))
                total_length_sec = float(mf.length)
            except Exception:
                total_length_sec = None

        if total_length_sec is None:
            est_bars = max(1, int(default_bars))
        else:
            est_bars = max(1, int(round(total_length_sec / bar_duration)))

        bars_data: List[Dict[str, Any]] = []
        t = 0.0
        for bar_id in range(est_bars):
            start_sec = t
            end_sec = t + bar_duration
            bars_data.append(
                {
                    "bar_id": bar_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "time_signature": f"{num}/{denom}",
                }
            )
            t = end_sec

        bars_df = pd.DataFrame(bars_data)
        bars_path = song_dir / f"{song_id}.bars.parquet"
        bars_df.to_parquet(bars_path, index=False)

        return (
            True,
            "ok",
            {
                "song_id": song_id,
                "package_path": str(package_path),
                "bars_path": str(bars_path),
                "tempo": tempo,
                "time_signature": f"{num}/{denom}",
            },
        )

    except Exception as e:
        return False, f"error: {e}", None


def main():
    parser = argparse.ArgumentParser(description="Rhythm AI Song Package Generator")
    parser.add_argument(
        "--midi-root", type=Path, required=True, help="MIDI root directory (drumclean_midi)"
    )
    parser.add_argument(
        "--output-root", type=Path, required=True, help="Output root directory for song packages"
    )
    parser.add_argument("--index-out", type=Path, required=True, help="Index CSV output path")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--default-bars",
        type=int,
        default=8,
        help="Fallback number of bars when MIDI length is unavailable",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation if package files already exist",
    )

    args = parser.parse_args()

    # ロギング設定
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # MIDIファイル収集
    midi_files = sorted(args.midi_root.rglob("*.mid"))

    print(f"\n{'='*70}")
    print(f"Rhythm AI Song Package Generation")
    print(f"{'='*70}")
    print(f"MIDI root: {args.midi_root}")
    print(f"Total MIDI files: {len(midi_files)}")
    print(f"{'='*70}\n")

    # 並列処理
    results = []

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_midi_file,
                midi_file,
                args.midi_root,
                args.output_root,
                args.default_bars,
                args.skip_existing,
            ): midi_file
            for midi_file in midi_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            midi_file = futures[future]
            success, reason, metadata = future.result()
            if success:
                results.append(metadata)

    # Index CSV保存
    args.index_out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.index_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["song_id", "package_path", "bars_path", "tempo", "time_signature"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"Total packages: {len(results)}")
    print(f"Index CSV: {args.index_out}")
    print(f"{'='*70}\n")

    # 拍子分布を即時表示（ログ便利機能）
    try:
        from collections import Counter

        sig_counts = Counter(
            [r.get("time_signature") for r in results if r and r.get("time_signature")]
        )
        if sig_counts:
            print("Time Signature Distribution:")
            for k, v in sorted(sig_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {k}: {v}")
            print()
    except Exception:
        pass


if __name__ == "__main__":
    main()
