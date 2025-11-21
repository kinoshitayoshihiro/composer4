#!/usr/bin/env python3
"""
OaF Dynamics Mapper - Phase 125 (P1-2-4)

OaF転写データ（piano_onsets_frames.json）+ bars.parquet（energy/valence列）+ emotion_profile.yaml を参照し、
Piano planのvelocity/articulation/sustainを再設計する。

【Phase 125拡張: Energy/Valence統合】:
  - bars.parquet energy列（0..1）を直接参照
  - bars.parquet valence列（-1..+1）を直接参照
  - emotion_profile.yaml instrument_map.piano写像定義使用

【3因子合成マッピング式】:
  v0 = clamp(64 + 32*(conf-0.5), 30, 112)        # OaF信頼度ベース
  Δdur = map_duration(duration_sec, short=+8, long=-6)  # 音価（短音=強め）
  Δreg = {pitch<60:-4, 60..72:0, >72:+4}         # レジスター（高域=明るい）
  Δenergy = energy * velocity_gain_from_energy   # energy係数（Phase 125）
  Δvalence = valence * velocity_bias_from_valence # valence係数（Phase 125）
  v = clamp(round(v0 + Δdur + Δreg + Δenergy + Δvalence), 1, 127)

使い方:
  python scripts/oaf_dynamics_mapper.py \
    --oaf-json piano_onsets_frames.json \
    --plan-in piano_plan_phase121_new.json \
    --bars bars_with_emotion.parquet \
    --emotion-profile configs/emotion_profiles/base.yaml \
    --emotion-style ballad \
    --out-plan piano_plan_phase125.json \
    --report oaf_dynamics_report_phase125.json

出力:
  - velocity/articulation再設計済みのPiano plan
  - meta.provenance.oaf_dynamics_phase125（適用ノート数、分布、energy/valence統合）
  - oaf_dynamics_report_phase125.json（メトリクス）
"""

import argparse
import json
import sys
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Phase 125: EmotionAI階層プリセット
sys.path.insert(0, str(Path(__file__).parent))
from emotion_profile_loader import load_emotion_profile, get_section_emotion


def clamp(x: float, a: float, b: float) -> float:
    """値をa..bの範囲にクランプ"""
    return max(a, min(b, x))


def map_duration(sec: float, short: int = 8, long: int = -6) -> int:
    """
    Duration（秒）からvelocity offsetを計算

    Args:
        sec: duration（秒）
        short: 短音時のoffset（例: +8）
        long: 長音時のoffset（例: -6）

    Returns:
        velocity offset（-6..+8）

    Example:
        0.05s → +8（スタッカート）
        0.25s → +2
        0.8s → -6（レガート）
    """
    x = clamp(sec, 0.05, 0.8)
    # 対数圧縮（短音→長音への滑らかな遷移）
    t = (math.log(x) - math.log(0.05)) / (math.log(0.8) - math.log(0.05))
    return round(short * (1 - t) + long * t)


def base_from_conf(conf: float) -> int:
    """
    OaF confidenceからベースvelocityを計算

    Args:
        conf: OaF confidence（0.0..1.0）

    Returns:
        ベースvelocity（30..112）

    Example:
        conf=0.5 → 64（中央）
        conf=1.0 → 96（高信頼）
        conf=0.0 → 32（低信頼）
    """
    return int(clamp(64 + 32 * (conf - 0.5), 30, 112))


def reg_offset(pitch: int) -> int:
    """
    Pitchレジスターからvelocity offsetを計算

    Args:
        pitch: MIDI pitch

    Returns:
        velocity offset（-4/0/+4）

    Example:
        pitch=48（C3以下）→ -4（低音域=暗い）
        pitch=66（F#4）→ 0（中域=ニュートラル）
        pitch=84（C6以上）→ +4（高音域=明るい）
    """
    if pitch < 60:
        return -4
    elif pitch > 72:
        return +4
    else:
        return 0


def energy_offset(energy: float, velocity_gain: int) -> int:
    """
    Energy値（bars.parquet energy列）からvelocity offsetを計算

    Args:
        energy: 0.0..1.0（bars.parquet energy列）
        velocity_gain: emotion_profile.yaml instrument_map.piano.velocity_gain_from_energy

    Returns:
        velocity offset（0..velocity_gain）

    Example:
        energy=0.5, velocity_gain=15 → +8
        energy=1.0, velocity_gain=15 → +15
    """
    return int(energy * velocity_gain)


def valence_offset(valence: float, velocity_bias: int) -> int:
    """
    Valence値（bars.parquet valence列）からvelocity offsetを計算

    Args:
        valence: -1.0..+1.0（bars.parquet valence列）
        velocity_bias: emotion_profile.yaml instrument_map.piano.velocity_bias_from_valence

    Returns:
        velocity offset（-velocity_bias..+velocity_bias）

    Example:
        valence=0.0, velocity_bias=5 → 0
        valence=+1.0, velocity_bias=5 → +5（明るい）
        valence=-1.0, velocity_bias=5 → -5（暗い）
    """
    return int(valence * velocity_bias)


def get_articulation_from_valence(valence: float, profile: Dict[str, Any]) -> str:
    """
    Valence値からarticulation指定を取得

    Args:
        valence: -1.0..+1.0
        profile: emotion_profile辞書

    Returns:
        articulation文字列（"normal" or "legato" or "marcato"）

    Example:
        valence=+0.5 → "normal"（ballad styleではmarcato避ける）
        valence=-0.5 → "legato"
    """
    inst_map = profile.get("instrument_map", {}).get("piano", {})
    artic_map = inst_map.get("articulation_from_valence", {})

    if valence > 0.3:
        return artic_map.get("positive", "normal")
    elif valence < -0.3:
        return artic_map.get("negative", "legato")
    else:
        return artic_map.get("neutral", "normal")


def beats_to_sec(beats: float, bars: pd.DataFrame) -> float:
    """
    Beats → Sec変換（bars.parquetのstart_beat/end_beat線形補間）

    Args:
        beats: beat位置
        bars: bars.parquet DataFrame

    Returns:
        sec位置

    Fallback:
        bars.parquetにstart_beat/end_beatがない場合は120bpmで推定（4/4拍子前提）
    """
    if "start_beat" not in bars.columns or "end_beat" not in bars.columns:
        # フォールバック: 120bpm, 4/4拍子
        return beats * 0.5  # 60.0/120.0

    # beats範囲検索
    for _, row in bars.iterrows():
        if row["start_beat"] <= beats <= row["end_beat"]:
            # 線形補間
            beat_range = row["end_beat"] - row["start_beat"]
            sec_range = row["end_sec"] - row["start_sec"]
            t = (beats - row["start_beat"]) / beat_range if beat_range > 0 else 0
            return row["start_sec"] + t * sec_range

    # 範囲外フォールバック
    return beats * 0.5


def main():
    parser = argparse.ArgumentParser(
        description="OaF Dynamics Mapper - Phase 125 (Energy/Valence統合)"
    )
    parser.add_argument(
        "--oaf-json",
        type=Path,
        required=False,
        default=None,
        help="OaF転写データJSON（piano_onsets_frames.json）。未指定時は energy-only モードにフォールバック",
    )
    parser.add_argument(
        "--plan-in",
        type=Path,
        required=True,
        help="入力 plan JSON (roleに応じたトラック/parts を想定)",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="piano",
        help="Instrument role to process (piano|bass|guitar|strings|drums). Default: piano",
    )
    parser.add_argument(
        "--energy-only",
        action="store_true",
        help="Use energy/valence mapping from bars instead of OaF (useful when no OaF JSON is available)",
    )
    parser.add_argument(
        "--bars", type=Path, required=True, help="bars_with_emotion.parquet（energy/valence列付き）"
    )
    parser.add_argument(
        "--emotion-profile",
        type=Path,
        help="EmotionAI階層プリセットYAMLパス（song-level、オプション）",
    )
    parser.add_argument(
        "--emotion-style", type=str, help="EmotionAI styleプリセット名（例: ballad）"
    )
    parser.add_argument("--out-plan", type=Path, required=True, help="出力Piano plan JSON")
    parser.add_argument("--report", type=Path, help="メトリクスレポートJSON出力先（オプション）")
    parser.add_argument(
        "--window-ms", type=float, default=40.0, help="OaFマッチング窓幅（ms、デフォルト40.0）"
    )

    args = parser.parse_args()

    # OaF転写データ読み込み（オプション）
    use_energy_only = bool(args.energy_only) or args.oaf_json is None
    oaf_notes = []
    if not use_energy_only and args.oaf_json is not None:
        with open(args.oaf_json, "r", encoding="utf-8") as f:
            oaf_data = json.load(f)
        oaf_notes = oaf_data.get("notes", [])
        print(f"[INFO] OaF notes loaded: {len(oaf_notes)}")
    else:
        print(f"[INFO] OaF not provided or energy-only forced: using energy/valence mapping")

    # Plan読み込み
    with open(args.plan_in, "r", encoding="utf-8") as f:
        plan = json.load(f)
    # role に応じた events 取得（parts.<role>.events or tracks[0].events）
    role = args.role
    events = []
    if "parts" in plan and role in plan.get("parts", {}):
        events = plan["parts"][role].get("events", [])
    elif "tracks" in plan and plan["tracks"]:
        # Fallback: use first track's events if parts not present
        events = plan["tracks"][0].get("events", [])
    else:
        events = plan.get("events", [])
    print(f"[INFO] Plan events loaded for role='{role}': {len(events)}")

    # bars.parquet読み込み（energy/valence列必須）
    bars = pd.read_parquet(args.bars)

    # energy列優先（energy → energy_curve → エラー）
    energy_col = None
    if "energy" in bars.columns:
        energy_col = "energy"
        print(f"[INFO] bars.parquet: energy列を使用")
    elif "energy_curve" in bars.columns:
        energy_col = "energy_curve"
        print(f"[WARN] bars.parquet: energy列なし、energy_curve列をフォールバック使用")
    else:
        print(f"[ERROR] bars.parquet に energy または energy_curve 列が存在しません")
        print(f"        derive_emotion_numeric.py を先に実行してください")
        sys.exit(1)

    # valence列チェック
    if "valence" not in bars.columns:
        print(f"[ERROR] bars.parquet に valence列が存在しません")
        print(f"        derive_emotion_numeric.py を先に実行してください")
        sys.exit(1)

    print(f"[INFO] bars.parquet loaded: {len(bars)} bars")
    print(
        f"        Energy ({energy_col}): min={bars[energy_col].min():.3f}, max={bars[energy_col].max():.3f}, mean={bars[energy_col].mean():.3f}"
    )
    print(
        f"        Valence: min={bars['valence'].min():.3f}, max={bars['valence'].max():.3f}, mean={bars['valence'].mean():.3f}"
    )

    # EmotionAI階層プリセット読み込み
    base_profile_path = Path("configs/emotion_profiles/base.yaml")
    song_profile_path = args.emotion_profile

    try:
        emotion_profile = load_emotion_profile(
            base_path=base_profile_path, style=args.emotion_style, song_path=song_profile_path
        )
        print(f"[INFO] EmotionAI階層プリセット読み込み成功")

        if args.emotion_style:
            print(f"        Style: {args.emotion_style}")
        if args.emotion_profile:
            print(f"        Song-level: {args.emotion_profile}")

    except Exception as e:
        print(f"[ERROR] EmotionAI階層プリセット読み込み失敗: {e}")
        sys.exit(1)

    # instrument_map.<role> 取得 (fallbacks exist)
    inst_map = emotion_profile.get("instrument_map", {}).get(role, {})
    velocity_gain = inst_map.get("velocity_gain_from_energy", 15)
    velocity_bias = inst_map.get("velocity_bias_from_valence", 5)

    print(f"[INFO] {role.capitalize()} 写像定義:")
    print(f"        velocity_gain_from_energy: {velocity_gain}")
    print(f"        velocity_bias_from_valence: {velocity_bias}")

    # Plan events → start_sec変換（beats→sec）
    for ev in events:
        if "start_sec" not in ev and "start_beats" in ev:
            ev["start_sec"] = beats_to_sec(ev["start_beats"], bars)

    # OaF notes → Piano plan マッチング + velocity再設計
    applied_count = 0
    velocities = []

    window_sec = args.window_ms / 1000.0  # ms → sec

    for ev in events:
        start_sec = ev.get("start_sec")
        pitch = ev.get("pitch")

        if start_sec is None or pitch is None:
            continue

        # ±window_sec内の候補ノート検索（pitchマッチング）
        candidates = [
            oaf_note
            for oaf_note in oaf_notes
            if abs(oaf_note["onset"] - start_sec) <= window_sec and oaf_note["pitch"] == pitch
        ]

        if not candidates:
            # フォールバック: 最近傍1点（±window_sec外でも最近傍を取得）
            matched = min(
                [n for n in oaf_notes if n["pitch"] == pitch],
                key=lambda n: abs(n["onset"] - start_sec),
                default=None,
            )
            if matched is None:
                continue
            candidates = [matched]

        # 多点重み付け平均（confidence × 時間距離重み）
        def weight(oaf_note):
            dt = abs(oaf_note["onset"] - start_sec)
            time_dist = max(1e-6, window_sec - dt)  # 近いほど大きい
            conf = max(1e-3, float(oaf_note.get("confidence", 1.0)))
            return conf * time_dist

        total_weight = sum(weight(n) for n in candidates)

        if total_weight < 1e-6:
            continue

        # 重み付き平均
        conf = sum(weight(n) * float(n.get("confidence", 1.0)) for n in candidates) / total_weight
        duration_sec = (
            sum(weight(n) * float(n.get("duration", 0.25)) for n in candidates) / total_weight
        )
        # pitchは重み付き平均して四捨五入（ほぼ同じpitchの微小ずれ対策）
        pitch_weighted = sum(weight(n) * float(n["pitch"]) for n in candidates) / total_weight
        pitch = round(pitch_weighted)

        # bars.parquet energy/valence取得（該当bar検索）
        bar_row = bars[(bars["start_sec"] <= start_sec) & (bars["end_sec"] > start_sec)]

        if bar_row.empty:
            # フォールバック（最近傍bar）
            bar_row = bars.iloc[(bars["start_sec"] - start_sec).abs().argsort()[:1]]

        energy = float(bar_row.iloc[0][energy_col])
        valence = float(bar_row.iloc[0]["valence"])
        section_label = bar_row.iloc[0].get("section_label", "")

        # 3因子合成 + Energy/Valence統合
        v0 = base_from_conf(conf)
        Δdur = map_duration(duration_sec)
        Δreg = reg_offset(pitch)
        Δenergy = energy_offset(energy, velocity_gain)
        Δvalence = valence_offset(valence, velocity_bias)

        velocity = int(clamp(v0 + Δdur + Δreg + Δenergy + Δvalence, 1, 127))

        # articulation取得
        articulation = get_articulation_from_valence(valence, emotion_profile)

        # planへ適用
        ev["velocity"] = velocity
        ev["articulation"] = articulation
        ev["energy"] = round(energy, 3)
        ev["valence"] = round(valence, 3)

        applied_count += 1
        velocities.append(velocity)

    print(f"[INFO] OaF Dynamics Mapping完了:")
    print(f"        適用イベント: {applied_count}/{len(events)}")

    if velocities:
        print(
            f"        Velocity: min={min(velocities)}, max={max(velocities)}, mean={sum(velocities)/len(velocities):.1f}"
        )

    # Provenance刻印
    if "meta" not in plan:
        plan["meta"] = {}
    if "provenance" not in plan["meta"]:
        plan["meta"]["provenance"] = {}

    plan["meta"]["provenance"]["oaf_dynamics_phase125"] = {
        "applied_events": applied_count,
        "total_events": len(events),
        "velocity_dist": {
            "min": min(velocities) if velocities else None,
            "max": max(velocities) if velocities else None,
            "mean": sum(velocities) / len(velocities) if velocities else None,
        },
        "emotion_profile": {
            "style": args.emotion_style,
            "song_profile": str(args.emotion_profile) if args.emotion_profile else None,
            "velocity_gain_from_energy": velocity_gain,
            "velocity_bias_from_valence": velocity_bias,
        },
        "mapper_version": "0.2.0_phase125",
    }

    # context_sources拡張
    if "context_sources" not in plan["meta"]:
        plan["meta"]["context_sources"] = {}

    plan["meta"]["context_sources"]["oaf_piano_dynamics_phase125"] = True
    plan["meta"]["context_sources"]["emotion_ai_energy_valence"] = True

    # 出力
    with open(args.out_plan, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Piano plan保存: {args.out_plan}")

    # メトリクスレポート出力
    if args.report:
        report = {
            "oaf_notes_total": len(oaf_notes),
            "plan_events_total": len(events),
            "applied_events": applied_count,
            "velocity_dist": {
                "min": min(velocities) if velocities else None,
                "max": max(velocities) if velocities else None,
                "mean": sum(velocities) / len(velocities) if velocities else None,
            },
            "emotion_profile": {
                "style": args.emotion_style,
                "song_profile": str(args.emotion_profile) if args.emotion_profile else None,
                "velocity_gain_from_energy": velocity_gain,
                "velocity_bias_from_valence": velocity_bias,
            },
            "energy_stats": {
                "column": energy_col,
                "min": float(bars[energy_col].min()),
                "max": float(bars[energy_col].max()),
                "mean": float(bars[energy_col].mean()),
            },
            "valence_stats": {
                "min": float(bars["valence"].min()),
                "max": float(bars["valence"].max()),
                "mean": float(bars["valence"].mean()),
            },
        }

        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[INFO] メトリクスレポート保存: {args.report}")


if __name__ == "__main__":
    main()
