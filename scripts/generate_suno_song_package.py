#!/usr/bin/env python3
"""
Generate SunoAI Song Package

SunoAI楽曲ディレクトリからsong_package.yamlを生成します。

Usage:
    # 単一楽曲
    python scripts/generate_suno_song_package.py \
        --song-dir data/suno_ai/suno_themesong/song_003 \
        --output song_package.yaml
    
    # 複数楽曲（一括生成）
    python scripts/generate_suno_song_package.py \
        --batch data/suno_ai/suno_themesong \
        --pattern "song_*"

Features:
    - 必須ファイル検出（bars.parquet, chordmap.json, sections.json）
    - Phase 113補助ファイル自動検出（style_presets, voicings_guide, bassline_plan, drum_accent_plan）
    - KPI統計自動抽出（deep_harmony_audit.json）
    - NO-OP保証（ファイル無し→paths未記載）
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
import yaml


def setup_logging(debug: bool = False):
    """ロギング設定"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def detect_files(song_dir: Path) -> dict:
    """
    Detect required, optional, and auxiliary files in the song directory.

    Args:
        song_dir: Path to song directory

    Returns:
        dict with keys: required, optional, auxiliary, missing
    """
    required_files = {
        "bars": song_dir / "analysis" / "bars.parquet",
        "chordmap": song_dir / "analysis" / "chordmap.json",
        "sections": song_dir / "analysis" / "sections.json",
    }

    optional_files = {
        "lyric_anchors": song_dir / "analysis" / "lyric_anchors.json",
        "audio": song_dir / "audio" / f"{song_dir.name}.wav",
        "tempo_map": song_dir / "analysis" / "tempo_map.json",
        "keys_timeline": song_dir / "analysis" / "keys_timeline.json",
    }

    auxiliary_files = {
        "style_presets": song_dir / "analysis" / "style_presets.yaml",
        "voicings_guide": song_dir / "analysis" / "voicings_guide.csv",
        "bassline_plan": song_dir / "analysis" / "bassline_plan.csv",
        "drum_accent_plan": song_dir / "analysis" / "drum_accent_plan.json",
    }

    # Check existence
    files = {}
    for name, path in {**required_files, **optional_files, **auxiliary_files}.items():
        files[name] = path if path.exists() else None

    return files


def extract_metadata(files: Dict[str, Optional[Path]], song_dir: Path) -> Dict[str, Any]:
    """
    検出ファイルからメタデータを抽出

    Returns:
        Dict with tempo_bpm, total_bars, duration_sec, key_center, etc.
    """
    metadata = {
        "tempo_bpm": 120.0,
        "time_signature": "4/4",
        "total_bars": 0,
        "duration_sec": 0.0,
        "key_center": None,
        "sections_count": 0,
        "chord_events_count": 0,
    }

    # bars.parquetから基本情報取得
    if files["bars"]:
        try:
            bars_df = pd.read_parquet(files["bars"])
            metadata["total_bars"] = len(bars_df)

            # duration推定
            if "end_beats" in bars_df.columns or "end_beat" in bars_df.columns:
                end_col = "end_beats" if "end_beats" in bars_df.columns else "end_beat"
                max_beats = bars_df[end_col].max()
                metadata["duration_sec"] = (max_beats / 4.0) * (60.0 / metadata["tempo_bpm"])

            # tempo_bpm（bars内にあれば）
            if "tempo_bpm" in bars_df.columns:
                tempo_vals = bars_df["tempo_bpm"].dropna()
                if len(tempo_vals) > 0:
                    metadata["tempo_bpm"] = float(tempo_vals.iloc[0])

            # time_signature（bars内にあれば）
            if "time_signature" in bars_df.columns:
                ts_vals = bars_df["time_signature"].dropna()
                if len(ts_vals) > 0:
                    metadata["time_signature"] = str(ts_vals.iloc[0])

        except Exception as e:
            logging.warning(f"Failed to read bars.parquet: {e}")

    # chordmap.jsonから情報取得
    if files["chordmap"]:
        try:
            with open(files["chordmap"], "r", encoding="utf-8") as f:
                chordmap = json.load(f)

            metadata["chord_events_count"] = len(chordmap.get("events", []))

            # mode/scaleからkey_center推定
            mode = chordmap.get("mode", "")
            scale = chordmap.get("scale", "")
            if mode or scale:
                metadata["key_center"] = f"{scale or ''} {mode or ''}".strip()

        except Exception as e:
            logging.warning(f"Failed to read chordmap.json: {e}")

    # sections.jsonからセクション数取得
    if files["sections"]:
        try:
            with open(files["sections"], "r", encoding="utf-8") as f:
                sections = json.load(f)

            metadata["sections_count"] = len(sections)

        except Exception as e:
            logging.warning(f"Failed to read sections.json: {e}")

    # tempo_map.jsonから正確なテンポ取得
    if files["tempo_map"]:
        try:
            with open(files["tempo_map"], "r", encoding="utf-8") as f:
                tempo_map = json.load(f)

            tempo_points = tempo_map.get("tempo_points", [])
            if tempo_points and len(tempo_points) > 0:
                # 最初のテンポポイントを使用
                metadata["tempo_bpm"] = float(tempo_points[0][2])

        except Exception as e:
            logging.warning(f"Failed to read tempo_map.json: {e}")

    # keys_timelineから調性情報取得
    if files["keys_timeline"]:
        try:
            with open(files["keys_timeline"], "r", encoding="utf-8") as f:
                keys_timeline = json.load(f)

            timeline = keys_timeline.get("timeline", [])
            if timeline and len(timeline) > 0:
                # 最初のキーセグメントを使用
                first_key = timeline[0]
                key_name = first_key.get("key", "")
                mode_name = first_key.get("mode", "")
                if key_name:
                    metadata["key_center"] = f"{key_name} {mode_name}".strip()

        except Exception as e:
            logging.warning(f"Failed to read keys_timeline.json: {e}")

    return metadata


def extract_harmony_stats(audit_path: Path) -> dict:
    """
    Extract harmony statistics from deep_harmony_audit.json.

    Args:
        audit_path: Path to deep_harmony_audit.json

    Returns:
        dict with tension_usage, cadence_score, etc.
    """
    audit = json.loads(audit_path.read_text())

    # Extract from summary section
    summary = audit.get("summary", {})

    # Extract cadence scores from cadences list
    cadences = audit.get("cadences", [])
    if cadences:
        cadence_scores = [c.get("cadence_score", 0.0) for c in cadences]
        cadence_avg = sum(cadence_scores) / len(cadence_scores)
    else:
        cadence_avg = 0.0

    # Extract KPI metrics
    stats = {
        "tension_usage": summary.get("tension_ratio_percent", 0.0) / 100.0,
        "cadence_score": cadence_avg,
        "anchor_near_change": summary.get("anchor_near_change_ratio_percent", 0.0) / 100.0,
        "key_confidence": summary.get("avg_key_confidence", 0.0),
        "enharmonic_consistency": summary.get("enharmonic_consistency", 1.0),
    }

    return stats


def generate_song_package(
    song_dir: Path,
    output_path: Optional[Path] = None,
    dataset: str = "suno_ai",
    source: str = "suno_themesong",
) -> Dict[str, Any]:
    """
    song_package.yaml生成

    Args:
        song_dir: 楽曲ディレクトリパス
        output_path: 出力先（Noneの場合はsong_dir/song_package.yaml）
        dataset: データセット名
        source: ソース名

    Returns:
        生成されたパッケージデータ
    """
    song_id = song_dir.name

    if output_path is None:
        output_path = song_dir / "song_package.yaml"

    logging.info(f"[{song_id}] Generating song package...")

    # ファイル検出
    files = detect_files(song_dir)

    # 必須ファイルチェック
    required_files = ["bars", "chordmap", "sections"]
    missing = [f for f in required_files if files[f] is None]

    if missing:
        logging.error(f"[{song_id}] Missing required files: {', '.join(missing)}")
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    # メタデータ抽出
    metadata = extract_metadata(files, song_dir)

    # KPI抽出
    audit_path = song_dir / "deep_harmony_audit.json"
    if audit_path.exists():
        harmony_stats = extract_harmony_stats(audit_path)
    else:
        harmony_stats = {}

    # paths構築（相対パス）
    paths = {}
    for key, path in files.items():
        if path is not None:
            try:
                rel_path = path.relative_to(song_dir)
                paths[key] = str(rel_path)
            except ValueError:
                # 絶対パスの場合はそのまま
                paths[key] = str(path)

    # Phase 113補助ファイル検出フラグ
    auxiliary_files = {
        "style_presets": files["style_presets"] is not None,
        "voicings_guide": files["voicings_guide"] is not None,
        "bassline_plan": files["bassline_plan"] is not None,
        "drum_accent_plan": files["drum_accent_plan"] is not None,
    }

    # パッケージデータ構築
    package = {
        "song_id": song_id,
        "dataset": dataset,
        "source": source,
        "tempo_bpm": metadata["tempo_bpm"],
        "time_signature": metadata["time_signature"],
        "total_bars": metadata["total_bars"],
        "duration_sec": metadata["duration_sec"],
        "paths": paths,
        "meta": {
            "bpm": metadata["tempo_bpm"],
            "key_center": metadata["key_center"],
            "sections_count": metadata["sections_count"],
            "chord_events_count": metadata["chord_events_count"],
        },
        "harmony": {k: v for k, v in harmony_stats.items() if v is not None},
        "auxiliary_files": auxiliary_files,
        "generated_at": datetime.now().isoformat(),
    }

    # YAML保存
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(package, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logging.info(f"[{song_id}] ✅ Package saved: {output_path}")

    return package


def batch_generate(
    batch_dir: Path,
    pattern: str = "song_*",
    dataset: str = "suno_ai",
    source: str = "suno_themesong",
) -> List[Dict[str, Any]]:
    """
    複数楽曲の一括生成

    Args:
        batch_dir: バッチ処理対象ディレクトリ
        pattern: 楽曲ディレクトリのパターン
        dataset: データセット名
        source: ソース名

    Returns:
        生成されたパッケージデータのリスト
    """
    song_dirs = sorted(batch_dir.glob(pattern))
    song_dirs = [d for d in song_dirs if d.is_dir()]

    if not song_dirs:
        logging.warning(f"No song directories found matching pattern: {pattern}")
        return []

    logging.info(f"Found {len(song_dirs)} song directories")

    packages = []
    success_count = 0

    for song_dir in song_dirs:
        try:
            package = generate_song_package(song_dir, dataset=dataset, source=source)
            packages.append(package)
            success_count += 1

        except Exception as e:
            logging.error(f"[{song_dir.name}] ❌ Failed: {e}")
            continue

    logging.info(f"✅ Success: {success_count}/{len(song_dirs)}")

    return packages


def main():
    parser = argparse.ArgumentParser(
        description="Generate SunoAI Song Package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 単一楽曲
    python scripts/generate_suno_song_package.py \\
        --song-dir data/suno_ai/suno_themesong/song_003
    
    # 複数楽曲（一括生成）
    python scripts/generate_suno_song_package.py \\
        --batch data/suno_ai/suno_themesong \\
        --pattern "song_*"
    
    # カスタム出力先
    python scripts/generate_suno_song_package.py \\
        --song-dir data/suno_ai/suno_themesong/song_003 \\
        --output custom_package.yaml
        """,
    )

    # モード選択
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--song-dir", type=str, help="Single song directory")
    mode_group.add_argument("--batch", type=str, help="Batch directory containing multiple songs")

    # オプション
    parser.add_argument("--output", type=str, help="Output path (single mode only)")
    parser.add_argument(
        "--pattern", type=str, default="song_*", help="Song directory pattern (batch mode only)"
    )
    parser.add_argument("--dataset", type=str, default="suno_ai", help="Dataset name")
    parser.add_argument("--source", type=str, default="suno_themesong", help="Source name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    setup_logging(debug=args.debug)

    try:
        if args.song_dir:
            # 単一楽曲モード
            song_dir = Path(args.song_dir)
            output_path = Path(args.output) if args.output else None

            package = generate_song_package(
                song_dir, output_path=output_path, dataset=args.dataset, source=args.source
            )

            print("\n" + "=" * 80)
            print("Song Package Generated")
            print("=" * 80)
            print(f"Song ID:        {package['song_id']}")
            print(f"Total Bars:     {package['total_bars']}")
            print(f"Duration:       {package['duration_sec']:.1f} sec")
            print(f"Tempo:          {package['tempo_bpm']} BPM")
            print(f"Key Center:     {package['meta'].get('key_center', 'N/A')}")
            print(f"Sections:       {package['meta']['sections_count']}")
            print(f"Chord Events:   {package['meta']['chord_events_count']}")

            if package["harmony"]:
                print("\nHarmony KPIs:")
                for k, v in package["harmony"].items():
                    if isinstance(v, float):
                        print(f"  {k:20s}: {v:.3f}")
                    else:
                        print(f"  {k:20s}: {v}")

            aux_enabled = [k for k, v in package["auxiliary_files"].items() if v]
            if aux_enabled:
                print(f"\nPhase 113 Files: {', '.join(aux_enabled)}")

            print("=" * 80)

        else:
            # バッチモード
            batch_dir = Path(args.batch)

            packages = batch_generate(
                batch_dir, pattern=args.pattern, dataset=args.dataset, source=args.source
            )

            print("\n" + "=" * 80)
            print("Batch Generation Summary")
            print("=" * 80)
            print(f"Total Songs:    {len(packages)}")
            print(f"Dataset:        {args.dataset}")
            print(f"Source:         {args.source}")

            if packages:
                total_bars = sum(p["total_bars"] for p in packages)
                total_duration = sum(p["duration_sec"] for p in packages)
                aux_counts = {
                    k: sum(1 for p in packages if p["auxiliary_files"][k])
                    for k in [
                        "style_presets",
                        "voicings_guide",
                        "bassline_plan",
                        "drum_accent_plan",
                    ]
                }

                print(f"Total Bars:     {total_bars}")
                print(f"Total Duration: {total_duration:.1f} sec ({total_duration/60:.1f} min)")
                print(f"\nPhase 113 Coverage:")
                for k, count in aux_counts.items():
                    pct = count / len(packages) * 100
                    print(f"  {k:20s}: {count}/{len(packages)} ({pct:.0f}%)")

            print("=" * 80)

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
