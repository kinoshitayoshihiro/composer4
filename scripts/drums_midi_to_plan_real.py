#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drums_midi_to_plan_real.py

rhythm_patterns.pickle からドラムパターンを読み込んで drums_plan.json を生成。
drums_recommendations.json の各小節 bar_index に対し、推奨パターンをロードして
MIDIイベント（pitch, vel, beat, dur_beats）を組み立てます。

Usage:
  python drums_midi_to_plan_real.py \\
      --recommendations data/.../drums_recommendations.json \\
      --patterns-pickle output/rhythm_ai/rhythm_patterns.pickle \\
      --tempo-bpm 120 \\
      --out data/.../drums_plan.json
"""
import argparse
import json
import pickle
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pretty_midi import PrettyMIDI

# Optional runtime helper for style-based minimal auto-regeneration
try:
    from scripts.drums_style_runtime import ensure_nonempty_drums
except Exception:
    # Fallback no-op if runtime helper not importable
    def ensure_nonempty_drums(
        events, min_events, bars, energy_s, style_yaml, accent_plan_path, seed
    ):
        return events, {"auto_regen": False}


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


def _load_tempo_map(path: str) -> List[tuple]:
    """
    tempo_map.json を読み込んで [(time_sec, bpm), ...] を返す。

    期待フォーマット:
      [{"time": 0.0, "bpm": 90.0}, {"time": 30.5, "bpm": 120.0}, ...]

    Returns:
        昇順ソートされた [(time_sec, bpm), ...] のリスト
    """
    with open(path, "r", encoding="utf-8") as f:
        tm = json.load(f)

    seq = []
    if isinstance(tm, list) and tm and "bpm" in tm[0]:
        if "time" in tm[0]:
            seq = [(float(x.get("time", 0.0)), float(x["bpm"])) for x in tm]
        else:
            raise ValueError("tempo_map.json must include 'time' for each change")

    return sorted(seq, key=lambda t: t[0])


class QL2Sec:
    """
    可変テンポ環境で ql（四分音符長）→ sec 変換を行うクラス。

    逐区間積分（0.25拍刻み）で高精度に変換します。

    Attributes:
        tempo_seq: [(time_sec, bpm), ...] 昇順ソート済み
        ql_per_beat: 1拍あたりのQL（デフォルト1.0）
        times: tempo_seqから抽出した時刻リスト
        bpms: tempo_seqから抽出したBPMリスト
    """

    def __init__(self, tempo_seq: List[tuple], ql_per_beat: float = 1.0):
        self.tempo_seq = tempo_seq
        self.ql_per_beat = ql_per_beat
        self.times = [t for t, _ in tempo_seq]
        self.bpms = [b for _, b in tempo_seq]

    def at(self, time_sec: float) -> float:
        """
        指定時刻のテンポ（BPM）を取得。

        bisect_rightで直前のテンポチェンジを検索します。

        Args:
            time_sec: 時刻（秒）

        Returns:
            BPM
        """
        i = bisect_right(self.times, time_sec) - 1
        if i < 0:
            i = 0
        return self.bpms[i]

    def ql_to_sec(self, start_time_sec: float, delta_ql: float) -> float:
        """
        可変テンポ環境で ql → sec 変換（逐区間積分）。

        0.25拍刻みで小区間に分割し、各区間のテンポで積分します。

        Args:
            start_time_sec: 開始時刻（秒）
            delta_ql: QL（四分音符長）

        Returns:
            終了時刻（秒）
        """
        step = 0.25  # 16分音符刻み
        t = start_time_sec
        remain = float(delta_ql)

        while remain > 1e-9:
            h = min(step, remain)
            bpm = self.at(t)
            sec = (60.0 / bpm) * (h / self.ql_per_beat)
            t += sec
            remain -= h

        return t


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
    bars_path: Optional[Path] = None,
    role_bars_path: Optional[Path] = None,
    tempo_map_path: Optional[str] = None,
    style_yaml: Optional[str] = None,
    accent_plan: Optional[str] = None,
    auto_regen: bool = True,
    min_events: int = 8,
    seed: int = 777,
):
    """
    drums_recommendations.json + rhythm_patterns.pickle → drums_plan.json生成

    Args:
        recommendations_path: drums_recommendations.json
        patterns_pickle: rhythm_patterns.pickle
        output_path: drums_plan.json出力先
        tempo_bpm: フォールバックテンポ（BPM）
        ppq: PPQ
        verbose: 詳細出力
        bars_path: bars.parquet（テンポ・小節情報）
        role_bars_path: analysis/role_bars/drums.parquet（drums_active統合用）
        tempo_map_path: tempo_map.json（可変テンポ対応）
    """
    # tempo_map読み込み（可変テンポ対応）
    tempo_seq = None
    if tempo_map_path:
        try:
            tempo_seq = _load_tempo_map(tempo_map_path)
            if verbose and tempo_seq:
                print(f"🎵 Loaded tempo_map.json: {len(tempo_seq)} tempo changes")
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to load tempo_map.json: {e}")
                print(f"   Falling back to fixed BPM: {tempo_bpm}")

    ql2sec = None if not tempo_seq else QL2Sec(tempo_seq, ql_per_beat=1.0)

    # role_barsマージ（activityゲート用）
    active_bars = None
    if role_bars_path and role_bars_path.exists():
        role_bars = pd.read_parquet(role_bars_path)
        # drums_active または drums_activity列でフィルタリング
        activity_col = None
        if "drums_active" in role_bars.columns:
            activity_col = "drums_active"
        elif "drums_activity" in role_bars.columns:
            activity_col = "drums_activity"

        if activity_col:
            active_bars = set(role_bars[role_bars[activity_col] > 0.5]["bar_index"])
            if verbose:
                print(
                    f"🎯 Activity gate: {len(active_bars)} active bars (from {len(role_bars)} total)"
                )
        else:
            if verbose:
                print(f"⚠️  role_bars has no drums_active column, skipping activity gate")

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

        # activityゲート（drums_active=0の小節はスキップ）
        if active_bars is not None and bar_idx not in active_bars:
            continue

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

    # bar + beat → time_ql 変換（bars.parquetから小節開始時刻を取得）
    bars_df = None
    if bars_path and bars_path.exists():
        bars_df = pd.read_parquet(bars_path)
        if "bar_index" not in bars_df.columns:
            bars_df = bars_df.reset_index()

    for ev in all_events:
        bar_idx = ev["bar"]
        beat = ev["beat"]  # 1.0 ~ 5.0（1小節=4拍、beat=1.0が小節開始）

        # bar開始時刻（QL）を取得
        bar_start_ql = 0.0
        if bars_df is not None and bar_idx in bars_df["bar_index"].values:
            bar_row = bars_df[bars_df["bar_index"] == bar_idx].iloc[0]
            if "start_time" in bar_row:
                bar_start_ql = float(bar_row["start_time"])
        else:
            # フォールバック: bar_idx * 4.0（4/4拍子前提）
            bar_start_ql = float(bar_idx) * 4.0

        # beat → ql（beat 1.0 = 小節開始、beat 2.0 = +1拍）
        beat_offset_ql = beat - 1.0
        time_ql = bar_start_ql + beat_offset_ql
        dur_ql = ev["dur_beats"]

        # time_ql → time（sec）変換
        if ql2sec:
            t = ql2sec.ql_to_sec(start_time_sec=0.0, delta_ql=time_ql)
            d = ql2sec.ql_to_sec(start_time_sec=t, delta_ql=dur_ql) - t
        else:
            # フォールバック（固定BPM）
            sec_per_beat = 60.0 / tempo_bpm
            t = time_ql * sec_per_beat
            d = dur_ql * sec_per_beat

        ev["time_ql"] = time_ql
        ev["time"] = t
        ev["duration"] = d

    # Auto-regen: if requested and events are too few, invoke style runtime helper
    meta: Dict[str, Any] = {"auto_regen": False}
    if auto_regen:
        # prepare energy series if available
        energy_s = None
        if bars_df is not None and "bar_index" in bars_df.columns:
            if "energy_curve" in bars_df.columns:
                energy_s = bars_df.set_index("bar_index")["energy_curve"]
            elif "energy" in bars_df.columns:
                energy_s = bars_df.set_index("bar_index")["energy"]

        try:
            maybe_events, meta = ensure_nonempty_drums(
                all_events,
                int(min_events),
                bars_df if bars_df is not None else pd.DataFrame(),
                energy_s,
                style_yaml,
                accent_plan,
                int(seed),
            )
            # Normalise shape: ensure events have 'beat'/'dur_beats'/'vel' keys
            if maybe_events is not None:
                normalized: List[Dict[str, Any]] = []
                for e in maybe_events:
                    if isinstance(e, dict) and "start_beats" in e:
                        beat = float(e.get("start_beats", e.get("beat", 1.0)))
                        end_b = float(e.get("end_beats", beat + 0.25))
                        dur = max(0.125, end_b - beat)
                        vel = int(e.get("velocity", e.get("vel", 80)))
                        normalized.append(
                            {
                                "bar": int(e.get("bar", 0)),
                                "beat": beat,
                                "pitch": int(e.get("pitch", 42)),
                                "dur_beats": dur,
                                "vel": vel,
                            }
                        )
                    else:
                        normalized.append(e)

                # replace all_events with normalized output
                all_events = normalized
        except Exception as ex:
            if verbose:
                print(f"⚠️  Auto-regen helper failed: {ex}")

    # Plan生成
    plan = {
        "ppq": ppq,
        "tempo_bpm": tempo_bpm,
        "tracks": [
            {"name": "Drums", "role": "drums", "channel": 9, "program": 0, "events": all_events}
        ],
        "meta": meta,
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
    parser.add_argument(
        "--tempo-bpm",
        type=float,
        default=120.0,
        help="Fallback tempo in BPM (used when --tempo-map is not provided)",
    )
    parser.add_argument(
        "--tempo-map",
        type=str,
        help="Path to tempo_map.json (optional, overrides --tempo-bpm for variable tempo)",
    )
    parser.add_argument("--ppq", type=int, default=480, help="PPQ (default: 480)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--style-yaml",
        type=str,
        default=None,
        help="(optional) path to drums style YAML to control auto-regen/density",
    )
    parser.add_argument(
        "--accent-plan",
        type=str,
        default=None,
        help="(optional) path to drum_accent_plan.json (used by auto-regen)",
    )
    parser.add_argument(
        "--auto-regen",
        action="store_true",
        help="Enable automatic minimal regen when patterns are missing or too few",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=8,
        help="Minimum events required to consider pattern output non-empty (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Random seed for auto-regeneration deterministic behavior",
    )
    parser.add_argument(
        "--stems-features",
        type=Path,
        default=None,
        help="(optional) stems_features.parquet を指定すると bar密度の加重平均に反映します",
    )
    parser.add_argument(
        "--bars",
        type=Path,
        default=None,
        help="bars.parquet (tempo/小節情報)",
    )
    parser.add_argument(
        "--role-bars",
        type=Path,
        default=None,
        help="analysis/role_bars/drums.parquet (drums_active統合用)",
    )

    args = parser.parse_args()

    generate_drums_plan_from_pickle(
        recommendations_path=args.recommendations,
        patterns_pickle=args.patterns_pickle,
        output_path=args.out,
        tempo_bpm=args.tempo_bpm,
        ppq=args.ppq,
        verbose=not args.quiet,
        bars_path=args.bars,
        role_bars_path=args.role_bars,
        tempo_map_path=args.tempo_map,
        style_yaml=args.style_yaml,
        accent_plan=args.accent_plan,
        auto_regen=args.auto_regen,
        min_events=args.min_events,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
