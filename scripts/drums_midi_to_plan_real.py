#!/usr/bin/env python3
"""
drums_midi_to_plan_real.py
--------------------------
実パターンPickleからDrums Planを生成（スケルトン卒業版）

rhythm_patterns.pickleから直接ノート情報を取得してPlan化

Usage:
    python3 scripts/drums_midi_to_plan_real.py \
      --recommendations song_packages/suno_project/song_001/drums_recommendations.json \
      --patterns-pickle output/rhythm_ai/rhythm_patterns.pickle \
      --out song_packages/suno_project/song_001/drums_plan.json \
      --tempo-bpm 74.677
"""
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

# GM Drum Mapping（拡張版）
GM_MAP = {
    35: "kick",  # Acoustic Bass Drum
    36: "kick",  # Bass Drum 1
    38: "snare",  # Acoustic Snare
    40: "snare",  # Electric Snare
    42: "hat",  # Closed Hi-Hat
    43: "hat",  # HH variant (データセット固有)
    44: "hat_pedal",  # Pedal Hi-Hat
    46: "hat_open",  # Open Hi-Hat
    49: "crash",  # Crash Cymbal 1
    51: "ride",  # Ride Cymbal 1
    52: "cymbal",  # Chinese Cymbal
    53: "ride_bell",  # Ride Bell
    57: "crash",  # Crash Cymbal 2
    59: "ride",  # Ride Cymbal 2
    82: "shaker",  # Shaker
}


# ========== Stems特徴重み付けヘルパー関数 (Phase A追加) ==========


def _load_stems_features(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    stems_features.parquetを読み込み、bar密度重み付けに必要な列を抽出

    期待カラム: bar / bar_index, hat_density, drums_active
    """
    if not path:
        return None
    df = pd.read_parquet(path)
    # bar_index正規化
    if "bar_index" not in df.columns and "bar" in df.columns:
        df = df.rename(columns={"bar": "bar_index"})
    keep = [c for c in ("bar_index", "hat_density", "drums_active") if c in df.columns]
    if "bar_index" not in keep:
        return None
    df = df[keep].copy()
    df["bar_index"] = df["bar_index"].astype(int)
    # stemsのhat_densityは相対値なので 0..12 程度にスケール（要調整）
    if "hat_density" in df.columns:
        df["hat_density_scaled"] = (df["hat_density"] * 4.0).clip(lower=0.0, upper=12.0)
    else:
        df["hat_density_scaled"] = np.nan
    return df


def _build_density_override(
    bars_df: pd.DataFrame,
    stems_df: Optional[pd.DataFrame],
    w_bars: float = 0.7,
    w_stems: float = 0.3,
) -> Dict[int, float]:
    """
    barごとの目標密度を上書きするテーブルを作る:
      target_density = 0.7 * bars.parquet.density_target + 0.3 * stems.hat_density_scaled
    stemsが無ければ空dictを返す
    """
    if stems_df is None or "hat_density_scaled" not in stems_df.columns:
        return {}
    if "density_target" not in bars_df.columns:
        return {}
    d = {}
    bd = (
        bars_df.set_index("bar_index")["density_target"]
        if "bar_index" in bars_df.columns
        else bars_df["density_target"]
    )
    for row in stems_df.itertuples(index=False):
        b = int(row.bar_index)
        if b in bd.index:
            d[b] = float(
                w_bars * float(bd.loc[b])
                + w_stems * float(getattr(row, "hat_density_scaled", np.nan))
            )
    return d


def _density_target_for(
    bar_idx: int, bars_df: pd.DataFrame, density_override: Dict[int, float]
) -> float:
    """
    barごとの目標密度を取得（stems重み付き優先、fallbackはbars.parquet）
    """
    if bar_idx in density_override:
        return density_override[bar_idx]
    # フォールバック: bars.parquet の density_target
    return (
        float(bars_df.loc[bar_idx, "density_target"])
        if "density_target" in bars_df.columns
        else 6.0
    )


# ========== 既存関数 ==========


def load_pattern_from_pickle(
    pattern_id: str, patterns_pickle: Path, tempo_bpm: float, verbose: bool = False
) -> Optional[List[Dict]]:
    """
    rhythm_patterns.pickleからpattern_id指定でパターン取得

    構造:
    - data['song_packages'][pattern_id]['source_midi']: MIDIファイルパス
    - data['song_packages'][pattern_id]['paths']['midi']: 相対パス

    注意:
    - recommendations.jsonのpattern_idには末尾にループインデックス（例: _13）が付加されている
    - Pickle内のキーには末尾インデックスが無い（例: 2_rock_105_beat_4-4）
    - → 末尾の_数字を削除してマッチング

    Returns:
        List[Dict]: イベントリスト [{"bar":0, "beat":1.0, "pitch":36, "dur_beats":0.25, "vel":90}, ...]
        None: パターン不在時
    """
    try:
        import pretty_midi
    except ImportError:
        print("⚠️  PrettyMIDI not installed, using fallback patterns")
        return None

    with open(patterns_pickle, "rb") as f:
        data = pickle.load(f)

    song_packages = data.get("song_packages", {})

    # pattern_id正規化: 末尾の_数字を削除
    # 例: "2_rock_105_beat_4-4_13" → "2_rock_105_beat_4-4"
    import re

    normalized_id = re.sub(r"_\d+$", "", pattern_id)

    if normalized_id not in song_packages:
        if verbose:
            print(f"⚠️  Pattern ID not found in pickle: {pattern_id} (normalized: {normalized_id})")
        return None

    pattern = song_packages[normalized_id]
    midi_path = pattern.get("source_midi", None)
    if midi_path is None:
        if verbose:
            print(f"⚠️  No source_midi for pattern: {normalized_id}")
        return None

    midi_path = Path(midi_path)
    if not midi_path.exists():
        if verbose:
            print(f"⚠️  MIDI file not found: {midi_path}")
        return None

    # PrettyMIDIで読み込み
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        if verbose:
            print(f"⚠️  Failed to load MIDI {midi_path}: {e}")
        return None

    # Drumトラック検索（channel 9 or instrument name "Drums"）
    drum_track = None
    for instrument in pm.instruments:
        if instrument.is_drum or instrument.program == 0:
            drum_track = instrument
            break

    if drum_track is None:
        if verbose:
            print(f"⚠️  No drum track found in {midi_path}")
        return None

    # ノート → イベントリスト変換
    # MIDIのtick → beat変換（PPQ=480前提、4/4拍子）
    ppq = 480
    ticks_per_beat = ppq
    sec_per_beat = 60.0 / tempo_bpm

    events = []
    for note in drum_track.notes:
        # tick → beat（1小節=4拍、beat 1.0～5.0）
        start_beat = (note.start / sec_per_beat) % 4.0 + 1.0
        dur_beats = (note.end - note.start) / sec_per_beat

        # GM Drum Mapフィルタ（36-59が主要ドラム音）
        if note.pitch not in GM_MAP:
            continue

        events.append(
            {
                "bar": 0,  # bar_idxは呼び出し側で調整
                "beat": start_beat,
                "pitch": note.pitch,
                "dur_beats": max(0.125, dur_beats),  # 最小duration
                "vel": note.velocity,
            }
        )

    if verbose and len(events) > 0:
        print(f"   Loaded pattern {normalized_id}: {len(events)} events")

    return events if len(events) > 0 else None


def generate_drums_plan_from_pickle(
    recommendations_path: Path,
    patterns_pickle: Path,
    output_path: Path,
    tempo_bpm: float,
    ppq: int = 480,
    verbose: bool = True,
):
    """
    drums_recommendations.json + rhythm_patterns.pickle → drums_plan.json生成

    Args:
        recommendations_path: drums_recommendations.json
        patterns_pickle: rhythm_patterns.pickle
        output_path: drums_plan.json出力先
        tempo_bpm: テンポ（BPM）
        ppq: PPQ
        verbose: 詳細出力
    """
    # recommendations読み込み
    recs = json.loads(recommendations_path.read_text(encoding="utf-8"))

    # bar_0, bar_1, ... 形式 → リスト変換
    bars = []
    for key in sorted(recs.keys()):
        if key.startswith("bar_"):
            bars.append(recs[key])

    if verbose:
        print(f"📖 Loaded {len(bars)} bars from recommendations")

    # パターンID → イベントキャッシュ
    pattern_cache = {}

    all_events = []
    fallback_count = 0

    for bar_data in bars:
        bar_idx = int(bar_data.get("bar_index", bar_data.get("bar", 0)))
        pattern_info = bar_data.get("pattern", {})
        pattern_id = pattern_info.get("pattern_id", None)

        if not pattern_id:
            # フォールバック: 基本パターン
            all_events.extend(_fallback_pattern(bar_idx))
            fallback_count += 1
            continue

        # パターンPickleから取得（キャッシュ利用）
        if pattern_id not in pattern_cache:
            pattern_events = load_pattern_from_pickle(
                pattern_id, patterns_pickle, tempo_bpm, verbose
            )
            pattern_cache[pattern_id] = pattern_events

            if verbose and len(pattern_cache) % 50 == 0:
                print(f"   Cached {len(pattern_cache)} patterns...")

        pattern_events = pattern_cache.get(pattern_id)

        if pattern_events:
            # パターンイベントを現在の小節にマッピング
            for e in pattern_events:
                all_events.append(
                    {
                        "bar": bar_idx,
                        "beat": e["beat"],
                        "pitch": e["pitch"],
                        "dur_beats": e["dur_beats"],
                        "vel": e["vel"],
                    }
                )
        else:
            # フォールバック（パターン未発見）
            all_events.extend(_fallback_pattern(bar_idx))
            fallback_count += 1

    # Plan生成
    plan = {
        "ppq": ppq,
        "tempo_bpm": tempo_bpm,
        "tracks": [
            {"name": "Drums", "role": "drums", "channel": 9, "program": 0, "events": all_events}
        ],
    }

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if verbose:
        unique_patterns = len([v for v in pattern_cache.values() if v is not None])
        print(f"\n✅ Generated drums_plan.json:")
        print(f"   Total events: {len(all_events)}")
        print(f"   Total bars: {len(bars)}")
        print(f"   Unique patterns loaded: {unique_patterns}/{len(pattern_cache)}")
        print(f"   Fallback bars: {fallback_count}")
        print(f"   Output: {output_path}")


def _fallback_pattern(bar_idx: int) -> List[Dict[str, Any]]:
    """フォールバック基本パターン（Kick 1,3拍、Snare 2,4拍、Hat 16分）"""
    return [
        {"bar": bar_idx, "beat": 1.0, "pitch": 36, "dur_beats": 0.25, "vel": 90},
        {"bar": bar_idx, "beat": 3.0, "pitch": 36, "dur_beats": 0.25, "vel": 90},
        {"bar": bar_idx, "beat": 2.0, "pitch": 38, "dur_beats": 0.25, "vel": 85},
        {"bar": bar_idx, "beat": 4.0, "pitch": 38, "dur_beats": 0.25, "vel": 85},
    ] + [
        {"bar": bar_idx, "beat": 1.0 + i * 0.25, "pitch": 42, "dur_beats": 0.125, "vel": 75}
        for i in range(16)
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate Drums Plan from rhythm_patterns.pickle")
    parser.add_argument(
        "--recommendations", type=Path, required=True, help="drums_recommendations.json"
    )
    parser.add_argument(
        "--patterns-pickle", type=Path, required=True, help="rhythm_patterns.pickle"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output drums_plan.json")
    parser.add_argument("--tempo-bpm", type=float, required=True, help="Tempo in BPM")
    parser.add_argument("--ppq", type=int, default=480, help="PPQ (default: 480)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--stems-features",
        type=Path,
        default=None,
        help="(optional) stems_features.parquet を指定すると bar密度の加重平均に反映します",
    )

    args = parser.parse_args()

    generate_drums_plan_from_pickle(
        recommendations_path=args.recommendations,
        patterns_pickle=args.patterns_pickle,
        output_path=args.out,
        tempo_bpm=args.tempo_bpm,
        ppq=args.ppq,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
