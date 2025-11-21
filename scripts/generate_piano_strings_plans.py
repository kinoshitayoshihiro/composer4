#!/usr/bin/env python3
"""
generate_piano_strings_plans.py
--------------------------------
SongPackage + chordmap → piano_plan.json & strings_plan.json生成

Piano: chord voicing展開、セクション別ボイシング、ペダル制御
Strings: 3パート（violin/viola/cello）レイヤー、long/short混合

Usage:
    python3 scripts/generate_piano_strings_plans.py \
      --song-dir song_packages/suno_project/song_001 \
      --config configs/arranger_weights.yaml \
      --emit-piano \
      --emit-strings
"""
import argparse
import json
import yaml
import pandas as pd
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable, Optional

from melody_hint_utils import (
    MelodyHint,
    apply_melody_hint_filter,
    build_melody_hint_manifest_payload,
    build_melody_hint_table,
    summarize_melody_hints,
)


def make_bar_locator(
    bars_df: pd.DataFrame, tempo_bpm: float
) -> Callable[[Dict], Tuple[int, float]]:
    """
    chordmap イベントから (bar_idx, beat_in_bar) を堅牢に推定するロケータ

    Args:
        bars_df: bars.parquet DataFrame
        tempo_bpm: テンポ（BPM）

    Returns:
        locate関数: dict → (bar_idx, beat_in_bar)
    """
    beats_per_bar = 4.0  # 4/4 前提（将来: time_signature から取得）
    sec_per_beat = 60.0 / float(tempo_bpm)
    sec_per_bar = sec_per_beat * beats_per_bar

    # bars.parquet に start_sec があれば二分探索で正確に割付
    starts_sec = None
    bar_indices = None
    if "start_sec" in bars_df.columns:
        sorted_df = bars_df.sort_values("bar_index")
        starts_sec = np.asarray(sorted_df["start_sec"].values, dtype=float)
        bar_indices = np.asarray(sorted_df["bar_index"].values, dtype=int)

    def _locate(ev: Dict) -> Tuple[int, float]:
        # 1) 明示バーフィールド優先
        for key in ("bar", "bar_index"):
            if key in ev:
                try:
                    b = int(ev[key])
                    beat = float(ev.get("beat", 1.0))
                    return max(0, b), max(1.0, beat)
                except Exception:
                    pass

        # 2) 絶対時刻（秒 or QL）から推定
        time_val = None
        if "time_sec" in ev:
            time_val = float(ev["time_sec"])
        elif "time" in ev:
            # chordmap.jsonの'time'はQL単位の可能性が高い
            time_ql = float(ev["time"])
            # QL → 秒変換
            time_val = time_ql / (tempo_bpm / 60.0)

        if time_val is not None:
            if starts_sec is not None and bar_indices is not None:
                # bars.parquet ベース二分探索
                i = int(np.searchsorted(starts_sec, time_val, side="right") - 1)
                i = max(0, min(i, len(bar_indices) - 1))
                bar_idx = int(bar_indices[i])
                bar_start = starts_sec[i]
                beat = ((time_val - bar_start) / sec_per_beat) + 1.0
                return bar_idx, max(1.0, beat)
            else:
                # フォールバック: 時刻からbar計算
                b = int(math.floor(time_val / sec_per_bar))
                beat = ((time_val - b * sec_per_bar) / sec_per_beat) + 1.0
                return max(0, b), max(1.0, beat)

        # 3) 絶対クォータノート位置から推定
        for key in ("qstamp", "quarter", "start_quarter"):
            if key in ev:
                q = float(ev[key])
                b = int(math.floor(q / beats_per_bar))
                beat = (q - b * beats_per_bar) + 1.0
                return max(0, b), max(1.0, beat)

        # 4) 小節内の beat しかない場合（measure/小節番号があれば使う）
        if "measure" in ev and "beat" in ev:
            return int(ev["measure"]), float(ev["beat"])

        # 5) フォールバック（デフォルト値）
        return 0, float(ev.get("beat", 1.0))

    return _locate


def load_config(config_path: Path) -> Dict:
    """arranger_weights.yaml読み込み"""
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_bars(bars_path: Path) -> pd.DataFrame:
    """bars.parquet読み込み"""
    return pd.read_parquet(bars_path)


def load_chordmap(chordmap_path: Path) -> List[Dict]:
    """chordmap.json読み込み"""
    data = json.loads(chordmap_path.read_text(encoding="utf-8"))
    return data.get("events", [])


def load_vocal_f0(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load vocal F0 parquet if provided."""

    if path is None:
        return None
    if not path.exists():
        print(f"⚠️  vocal_f0 file not found: {path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"⚠️  Failed to load vocal F0 ({path}): {exc}")
        return None


def _debug_dump_mapping(
    name: str,
    chordmap: List[Dict],
    locate: Callable[[Dict], Tuple[int, float]],
    bars_df: pd.DataFrame,
    tempo_bpm: float,
    out_dir: Path,
    debug: bool,
):
    if not debug:
        return
    beats_per_bar = 4.0
    expected_total_beats = int(len(bars_df) * beats_per_bar)

    rows = []
    max_bar_idx = (
        int(bars_df["bar_index"].max()) if "bar_index" in bars_df.columns else len(bars_df) - 1
    )
    overbars = []

    has_time_sec = any("time_sec" in ev for ev in chordmap)
    has_time = any("time" in ev for ev in chordmap)
    has_qstamp = any(k in ev for ev in chordmap for k in ("qstamp", "quarter", "start_quarter"))

    max_time_val = None
    if has_time:
        try:
            max_time_val = max(float(ev["time"]) for ev in chordmap if "time" in ev)
        except Exception:
            max_time_val = None

    print(
        f"[DEBUG:{name}] chordmap size={len(chordmap)}, expected_total_beats≈{expected_total_beats}"
    )
    print(
        f"[DEBUG:{name}] keys: time_sec={has_time_sec}, time(QL?)={has_time}, qstamp-like={has_qstamp}"
    )
    if max_time_val is not None:
        print(f"[DEBUG:{name}] max(ev['time'])={max_time_val:.2f} (units unknown; often QL=beats)")

    for i, ev in enumerate(chordmap):
        bar_idx, beat = locate(ev)
        src = (
            "time_sec"
            if "time_sec" in ev
            else (
                "time"
                if "time" in ev
                else (
                    "qstamp"
                    if any(k in ev for k in ("qstamp", "quarter", "start_quarter"))
                    else "bar/beat"
                )
            )
        )
        tval = ev.get("time_sec", ev.get("time", ev.get("qstamp", None)))
        rows.append(
            {"i": i, "src": src, "raw_time": tval, "map_bar": int(bar_idx), "map_beat": float(beat)}
        )
        if bar_idx > max_bar_idx or bar_idx < 0 or beat < 1.0 or beat > 4.0:
            overbars.append((i, ev, (bar_idx, beat)))

    if overbars:
        print(f"[DEBUG:{name}] ⚠️ mapped out-of-range events: {len(overbars)} / {len(chordmap)}")
        for j, (i, ev, mb) in enumerate(overbars[:10]):
            print(
                f"  - ev#{i}: raw={{ {', '.join(f'{k}:{ev[k]}' for k in ev.keys() & {'time','time_sec','qstamp','bar','beat','root','quality'})} }} -> mapped={mb}"
            )
    else:
        print(f"[DEBUG:{name}] mapped out-of-range events: 0")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(
            f"[DEBUG:{name}] mapped_bar min..max = {int(df['map_bar'].min())}..{int(df['map_bar'].max())}"
        )
        print(
            f"[DEBUG:{name}] mapped_beat min..max = {df['map_beat'].min():.2f}..{df['map_beat'].max():.2f}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{name}_plan_debug_map.csv"
        df.to_csv(csv_path, index=False)
        print(f"[DEBUG:{name}] mapping table -> {csv_path}")
    print()


def generate_piano_plan(
    bars_df: pd.DataFrame,
    chordmap: List[Dict],
    config: Dict,
    tempo_bpm: float,
    melody_hints: Optional[Dict[int, MelodyHint]] = None,
) -> Dict:
    """
    Piano Plan生成（chord+voicing展開）

    - セクション別ボイシング（verse: close, chorus: spread, bridge: drop2）
    - add9/sus4確率的挿入
    - ペダル制御（legato_ratio > threshold）
    """
    heuristics = config.get("heuristics", {}).get("piano", {})
    voicing_by_section = heuristics.get(
        "voicing_by_section", {"verse": "close", "chorus": "spread", "bridge": "drop2"}
    )
    add9_prob = heuristics.get("add9_prob", 0.20)
    sus4_prob = heuristics.get("sus4_prob", 0.10)
    pedal_threshold = heuristics.get("pedal_threshold", 0.55)

    # 堅牢なbar_locator作成
    locate = make_bar_locator(bars_df, tempo_bpm)

    # bars.parquet の最大bar_index取得
    max_bar_idx = (
        int(bars_df["bar_index"].max()) if "bar_index" in bars_df.columns else len(bars_df) - 1
    )

    # chordmapをbar単位でグループ化（堅牢化）
    bar_chords = {int(b): [] for b in range(max_bar_idx + 1)}
    for ev in chordmap:
        bar_idx, beat_in_bar = locate(ev)
        bar_idx = int(max(0, min(bar_idx, max_bar_idx)))

        # コードシンボル取得
        root = ev.get("root", "C")
        quality = ev.get("quality", "")
        chord_sym = f"{root}{quality}"

        bar_chords[bar_idx].append(
            {
                "symbol": chord_sym,
                "bar": bar_idx,
                "beat": beat_in_bar,
                "time": ev.get("time"),  # 元のtime保持（デバッグ用）
                "confidence": ev.get("confidence", 1.0),
            }
        )

    events = []

    for idx, row in bars_df.iterrows():
        bar_idx = int(idx)
        section = str(row.get("section_label", "verse")).lower()
        energy = float(row.get("energy_curve", 0.5))

        # セクション別ボイシング
        voicing_style = voicing_by_section.get(section, "close")

        # コード取得
        chords_in_bar = bar_chords.get(bar_idx, [])

        for chord_ev in chords_in_bar:
            chord_sym = chord_ev.get("symbol", "C")
            actual_bar = int(chord_ev.get("bar", bar_idx))
            actual_beat = float(chord_ev.get("beat", 1.0))

            # add9/sus4変換（確率的）
            import random

            if random.random() < add9_prob and "7" not in chord_sym:
                chord_sym += "add9"
            elif random.random() < sus4_prob:
                chord_sym = chord_sym.replace("maj", "sus4").replace("m", "sus4")

            # velocity（energy依存）
            vel = int(85 + 10 * energy)

            events.append(
                {
                    "bar": actual_bar,
                    "beat": actual_beat,
                    "chord": chord_sym,
                    "voicing": {"style": voicing_style, "octave": 4},
                    "dur_beats": 1.0,  # デフォルト4分音符
                    "vel": vel,
                    "arp_ms": 0,
                }
            )

    filter_stats = {"annotated": 0, "removed": 0}
    if melody_hints:
        events, filter_stats = apply_melody_hint_filter(
            events,
            melody_hints,
            instrument="piano",
            drop_tags=(),
            annotate=True,
        )

    return {
        "ppq": 480,
        "tempo_bpm": tempo_bpm,
        "meta": {
            "total_bars": len(bars_df),
            "melody_hint": {
                "annotated": filter_stats.get("annotated", 0),
                "removed_for_strings": 0,
                "bars_with_hints": len(melody_hints or {}),
            },
        },
        "tracks": [
            {
                "name": "Piano",
                "role": "piano",
                "channel": 3,
                "program": 0,  # Acoustic Grand Piano
                "events": events,
            }
        ],
    }


def generate_strings_plan(
    bars_df: pd.DataFrame,
    chordmap: List[Dict],
    config: Dict,
    tempo_bpm: float,
    melody_hints: Optional[Dict[int, MelodyHint]] = None,
) -> Dict:
    """
    Strings Plan生成（3パート: violin/viola/cello）

    - layer_roles: [violin, viola, cello]
    - unison_prob: 全パート同音確率
    - octave_doubling_prob: オクターブ重ね確率
    - long_short_mix: サスティン/ショート混合
    - dynamics_scale: セクション別ダイナミクス係数
    """
    heuristics = config.get("heuristics", {}).get("strings", {})
    layer_roles = heuristics.get("layer_roles", ["violin", "viola", "cello"])
    unison_prob = heuristics.get("unison_prob", 0.30)
    octave_doubling_prob = heuristics.get("octave_doubling_prob", 0.25)
    dynamics_scale = heuristics.get("dynamics_scale", {"intro": 0.8, "chorus": 1.2})

    # 堅牢なbar_locator作成
    locate = make_bar_locator(bars_df, tempo_bpm)

    # bars.parquet の最大bar_index取得
    max_bar_idx = (
        int(bars_df["bar_index"].max()) if "bar_index" in bars_df.columns else len(bars_df) - 1
    )

    # chordmapをbar単位でグループ化（堅牢化）
    bar_chords = {int(b): [] for b in range(max_bar_idx + 1)}
    for ev in chordmap:
        bar_idx, beat_in_bar = locate(ev)
        bar_idx = int(max(0, min(bar_idx, max_bar_idx)))

        # コードシンボル取得
        root = ev.get("root", "C")
        quality = ev.get("quality", "")
        chord_sym = f"{root}{quality}"

        bar_chords[bar_idx].append(
            {
                "symbol": chord_sym,
                "bar": bar_idx,
                "beat": beat_in_bar,
                "time": ev.get("time"),
                "confidence": ev.get("confidence", 1.0),
            }
        )

    events = []

    for idx, row in bars_df.iterrows():
        bar_idx = int(idx)
        section = str(row.get("section_label", "verse")).lower()
        energy = float(row.get("energy_curve", 0.5))

        # セクション別ダイナミクス
        dyn_scale = dynamics_scale.get(section, 1.0)

        # コード取得
        chords_in_bar = bar_chords.get(bar_idx, [])

        for chord_ev in chords_in_bar:
            chord_sym = chord_ev.get("symbol", "C")
            actual_bar = int(chord_ev.get("bar", bar_idx))
            actual_beat = float(chord_ev.get("beat", 1.0))

            # velocity（energy + dynamics_scale）
            base_vel = int(80 * dyn_scale + 10 * energy)

            # 3パート展開（簡易: close voicing、オクターブ分散）
            import random

            if random.random() < unison_prob:
                # ユニゾン: 全パート同じオクターブ
                octaves = [4, 4, 4]
            else:
                # 分散: violin=5, viola=4, cello=3
                octaves = [5, 4, 3]

            for role, octave in zip(layer_roles, octaves):
                events.append(
                    {
                        "bar": actual_bar,
                        "beat": actual_beat,
                        "chord": chord_sym,
                        "voicing": {"style": "close", "octave": octave},
                        "dur_beats": 2.0,  # 長めのサスティン
                        "vel": base_vel,
                        "arp_ms": 0,
                        "role": role,  # メタ情報（任意）
                    }
                )

    filter_stats = {"annotated": 0, "removed": 0}
    if melody_hints:
        events, filter_stats = apply_melody_hint_filter(
            events,
            melody_hints,
            instrument="strings",
            drop_tags=("melody_hint_long",),
            drop_threshold_beats=2.0,
            annotate=True,
        )

    return {
        "ppq": 480,
        "tempo_bpm": tempo_bpm,
        "meta": {
            "total_bars": len(bars_df),
            "melody_hint": {
                "annotated": filter_stats.get("annotated", 0),
                "removed_for_strings": filter_stats.get("removed", 0),
                "bars_with_hints": len(melody_hints or {}),
            },
        },
        "tracks": [
            {
                "name": "Strings",
                "role": "strings",
                "channel": 4,
                "program": 48,  # String Ensemble
                "events": events,
            }
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Piano/Strings Plan生成")
    ap.add_argument("--song-dir", type=Path, required=True, help="SongPackage directory")
    ap.add_argument(
        "--config", type=Path, default=Path("configs/arranger_weights.yaml"), help="Config YAML"
    )
    ap.add_argument("--emit-piano", action="store_true", help="Generate piano_plan.json")
    ap.add_argument("--emit-strings", action="store_true", help="Generate strings_plan.json")
    ap.add_argument("--vocal-f0", type=Path, help="Path to vocal_f0_crepe.parquet (optional)")
    ap.add_argument(
        "--emit-melody-manifest",
        action="store_true",
        help="Write melody_hint_manifest.json (requires --vocal-f0)",
    )
    ap.add_argument(
        "--melody-manifest-path",
        type=Path,
        help="Override melody hint manifest path (default: <song_dir>/melody_hint_manifest.json)",
    )
    ap.add_argument("--debug", action="store_true", help="Verbose mapping debug")
    ap.add_argument(
        "--stems-features",
        type=Path,
        default=None,
        help="Path to stem_features.parquet (Phase 1 integration)",
    )
    args = ap.parse_args()

    # データ読み込み
    song_pkg_path = args.song_dir / "song_package.yaml"
    if not song_pkg_path.exists():
        print(f"❌ song_package.yaml not found: {song_pkg_path}")
        return

    song_pkg = yaml.safe_load(song_pkg_path.read_text(encoding="utf-8"))
    tempo_bpm = float(song_pkg["meta"].get("bpm", song_pkg["meta"].get("tempo_bpm", 120.0)))

    bars_path = args.song_dir / song_pkg["artifacts"]["bars"]
    bars_df = load_bars(bars_path)

    chordmap_path = args.song_dir / song_pkg["artifacts"]["chordmap"]
    chordmap = load_chordmap(chordmap_path)

    config = load_config(args.config)

    print(f"📂 SongPackage: {args.song_dir.name}")
    print(f"📊 Bars: {len(bars_df)}, Chords: {len(chordmap)}, Tempo: {tempo_bpm} BPM")

    vocal_f0 = load_vocal_f0(args.vocal_f0) if args.vocal_f0 else None
    melody_hints = build_melody_hint_table(bars_df, vocal_f0) if vocal_f0 is not None else {}
    if melody_hints:
        print("📊 Melody hint summary (CREPE):")
        for section, stats in summarize_melody_hints(melody_hints).items():
            print(
                f"   - {section}: bars={stats['bars']} long={stats['long']} phrase={stats['phrase']} gliss={stats['gliss']} avg_len={stats['avg_duration_beats']}"
            )
    elif args.emit_melody_manifest:
        print(
            "Melody hint manifest requested but no vocal F0 data provided; skipping manifest export."
        )

    manifest_out = args.melody_manifest_path
    if manifest_out and not manifest_out.is_absolute():
        manifest_out = args.song_dir / manifest_out
    if manifest_out is None:
        manifest_out = args.song_dir / "melody_hint_manifest.json"

    if args.emit_melody_manifest and melody_hints:
        manifest_payload = build_melody_hint_manifest_payload(
            melody_hints,
            bars_total=len(bars_df),
            song_id=(song_pkg.get("meta", {}) or {}).get("song_id") or args.song_dir.name,
            bars_path=bars_path,
            vocal_f0_path=args.vocal_f0,
            out_path=manifest_out,
        )
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rel_manifest = (
            manifest_out.relative_to(args.song_dir)
            if manifest_out.is_relative_to(args.song_dir)
            else manifest_out
        )
        print(f"melody_hint_manifest: {rel_manifest} (hints={len(melody_hints)})")

    # Stem特徴統合（Phase 1）
    stem_df = None
    if args.stems_features and args.stems_features.exists():
        stem_df = pd.read_parquet(args.stems_features)
        print(f"🎵 Stem features: loaded ({len(stem_df)} bars)")
        print(f"   energy_curve: {stem_df['energy_curve'].mean():.2f} avg")

        # arranger_weights.yaml設定読み込み
        stems_cfg = config.get("stems", {})
        use_stems = stems_cfg.get("use_stems", False)
        piano_blend = stems_cfg.get("piano", {}).get("loudness_blend", 0.5)
        strings_blend = stems_cfg.get("strings", {}).get("loudness_blend", 0.6)

        if use_stems:
            # Energy Curveブレンド（Piano用）
            if "energy" in bars_df.columns:
                bars_df["energy_original"] = bars_df["energy"].copy()
                bars_df["energy"] = (1 - piano_blend) * bars_df["energy"] + piano_blend * stem_df[
                    "energy_curve"
                ]
                print(f"   Piano energy blend: {piano_blend:.1%}")

            # Strings用にstem energy_curveを別カラムで保存
            bars_df["stem_energy"] = stem_df["energy_curve"]
            bars_df["strings_blend"] = strings_blend
            print(f"   Strings energy blend: {strings_blend:.1%}")

    print()

    # 共通ロケータ（デバッグでも使う）
    _locate = make_bar_locator(bars_df, tempo_bpm)
    _debug_dir = args.song_dir / "_debug"

    # Piano Plan生成
    if args.emit_piano:
        piano_plan = generate_piano_plan(bars_df, chordmap, config, tempo_bpm, melody_hints)
        piano_path = args.song_dir / "piano_plan.json"
        piano_path.write_text(
            json.dumps(piano_plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"✅ piano_plan.json: {len(piano_plan['tracks'][0]['events'])} events")
        _debug_dump_mapping("piano", chordmap, _locate, bars_df, tempo_bpm, _debug_dir, args.debug)

    # Strings Plan生成
    if args.emit_strings:
        strings_plan = generate_strings_plan(bars_df, chordmap, config, tempo_bpm, melody_hints)
        strings_path = args.song_dir / "strings_plan.json"
        strings_path.write_text(
            json.dumps(strings_plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"✅ strings_plan.json: {len(strings_plan['tracks'][0]['events'])} events")
        _debug_dump_mapping(
            "strings", chordmap, _locate, bars_df, tempo_bpm, _debug_dir, args.debug
        )

    print(
        f"\n🎉 Generated: {', '.join([n for n, f in [('piano_plan.json', args.emit_piano), ('strings_plan.json', args.emit_strings)] if f])}"
    )


if __name__ == "__main__":
    main()
