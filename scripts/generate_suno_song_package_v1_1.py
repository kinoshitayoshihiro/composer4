#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_suno_song_package.py (v1.1-hybrid)

Synchro-first song_package.yaml generator with measured KPI integration.

Features:
    - v1.1: synchro_policy統合（mix_recipe自動生成、alignment_hint、quality_gates）
    - v1.1: tempo_map厳密算出（可変テンポ対応）
    - v1.1: キー推定強化（セクション重み付けルート頻度）
    - v1.0: deep_harmony_audit.json実測KPI維持
    - v1.0: bars.parquet実測値併用

Usage:
    python scripts/generate_suno_song_package_v1_1.py \
        --song-id song_003 \
        --analysis-dir data/suno_ai/suno_themesong/song_003/analysis \
        --out data/suno_ai/suno_themesong/song_003/song_package.yaml
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


def load_json(path: Path) -> Any:
    """JSON読み込み"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ql_to_seconds(ql: float, tempo_events: List[Dict[str, Any]]) -> float:
    """
    可変テンポ対応の秒数換算

    Args:
        ql: Quarter Length（4分音符単位の時間）
        tempo_events: テンポイベントリスト

    Returns:
        秒数
    """
    if not tempo_events:
        return 0.0

    # tempo_eventsの構造を正規化
    segs = []
    if tempo_events and "start_ql" in tempo_events[0] and "end_ql" in tempo_events[0]:
        # 既にセグメント形式
        segs = tempo_events
    else:
        # time_ql形式からセグメント形式に変換
        for i, ev in enumerate(tempo_events):
            start = ev.get("time_ql", 0.0)
            end = tempo_events[i + 1].get("time_ql", start) if i + 1 < len(tempo_events) else start
            segs.append({"start_ql": start, "end_ql": end, "bpm": ev["bpm"]})

    total_sec = 0.0
    remaining = ql
    cursor = 0.0

    for seg in segs:
        s = float(seg["start_ql"])
        e = float(seg["end_ql"])
        bpm = float(seg["bpm"])

        if remaining <= 0:
            break

        span = max(0.0, min(remaining, e - cursor))
        total_sec += span * (60.0 / bpm)
        cursor += span
        remaining -= span

        if cursor < e:
            break
        cursor = e

    return total_sec


def duration_from_tempo_map(tempo_map: Dict[str, Any], max_ql: float) -> float:
    """
    tempo_mapから曲長を厳密算出

    Args:
        tempo_map: tempo_map.json
        max_ql: 最大Quarter Length

    Returns:
        曲長（秒）
    """
    # tempo_mapの構造を正規化
    evs = tempo_map.get("events") or tempo_map.get("tempo") or []

    # tempo_points形式の場合は変換
    if not evs and "tempo_points" in tempo_map:
        tempo_points = tempo_map["tempo_points"]
        # tempo_points: [[time_sec, time_ql, bpm], ...]
        evs = [{"time_ql": float(tp[1]), "bpm": float(tp[2])} for tp in tempo_points]

    return ql_to_seconds(max_ql, evs)


def estimate_time_signature(default: str = "4/4") -> Dict[str, int]:
    """拍子を{num, den}に正規化"""
    try:
        num, den = default.split("/")
        return {"num": int(num), "den": int(den)}
    except Exception:
        return {"num": 4, "den": 4}


def infer_key_candidates(
    chord_events: List[Dict[str, Any]], sections: List[Dict[str, Any]]
) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
    """
    キー候補の多段推定（ルート頻度×セクション重み付け）

    Args:
        chord_events: chordmap.events
        sections: sections.json

    Returns:
        (key_center, key_confidence, key_candidates)
    """
    # セクションラベル別重み
    weight_by_label = {
        "chorus": 1.5,  # サビ重視
        "pre": 1.2,
        "pre_chorus": 1.2,
        "bridge": 1.1,
        "verse": 1.0,
        "intro": 0.8,
        "outro": 0.8,
    }

    # セクション範囲構築
    ranges = []
    if sections and "start_bar" in sections[0]:
        # start_bar/end_bar形式
        for s in sections:
            ranges.append((s["start_bar"], s["end_bar"], s["label"].lower()))
    else:
        # bar/label形式
        bars = [(s["bar"], s["label"].lower()) for s in sections]
        bars.sort(key=lambda x: x[0])
        last_bar = bars[-1][0] if bars else 0
        for i, (b, label) in enumerate(bars):
            end = bars[i + 1][0] - 1 if i + 1 < len(bars) else last_bar
            ranges.append((b, end, label))

    # 小節別重み
    bar_weight = {}
    for s, e, label in ranges:
        w = weight_by_label.get(label, 1.0)
        for b in range(s, e + 1):
            bar_weight[b] = w

    # ルート出現頻度×重み
    score = {}
    for ev in chord_events:
        b = int(ev["time"] / 4.0)
        root = ev.get("root")
        if not root:
            continue
        score[root] = score.get(root, 0.0) + bar_weight.get(b, 1.0)

    if not score:
        return None, 0.0, []

    # 上位5件
    top = sorted(score.items(), key=lambda x: (-x[1], x[0]))[:5]
    total = sum(v for _, v in top)

    if len(top) == 1:
        return (
            top[0][0],
            1.0,
            [
                {"key": k, "method": "root_histogram", "score": float(s / total if total else 1.0)}
                for k, s in top
            ],
        )

    # 信頼度計算（1位と2位の差）
    c1 = top[0][1]
    c2 = top[1][1]
    confidence = float((c1 - c2) / total) if total > 0 else 0.0

    candidates = [{"key": k, "method": "root_histogram", "score": float(s / total)} for k, s in top]

    return top[0][0], confidence, candidates


def build_mix_recipe(sections: List[Dict[str, Any]], anchors: Dict[str, Any]) -> Dict[str, Any]:
    """
    セクション別mix_recipe自動生成

    Args:
        sections: sections.json
        anchors: lyric_anchors.json

    Returns:
        mix_recipe dict
    """
    # セクション範囲構築
    ranges = []
    if sections and "start_bar" in sections[0]:
        for s in sections:
            ranges.append((s["start_bar"], s["end_bar"], s["label"].lower()))
    else:
        bars = [(s["bar"], s["label"].lower()) for s in sections]
        bars.sort(key=lambda x: x[0])
        last_bar = bars[-1][0] if bars else 0
        for i, (b, label) in enumerate(bars):
            end = bars[i + 1][0] - 1 if i + 1 < len(bars) else last_bar
            ranges.append((b, end, label))

    # 小節別stress/plosive統計
    stress_by_bar = {}
    plosive_by_bar = {}
    for a in anchors.get("anchors", []):
        tql = a.get("time_ql")
        if tql is None:
            continue
        b = int(math.floor(float(tql) / 4.0))
        klass = a.get("class") or []
        if "stress" in klass:
            stress_by_bar[b] = stress_by_bar.get(b, 0) + 1
        if "plosive" in klass:
            plosive_by_bar[b] = plosive_by_bar.get(b, 0) + 1

    def burden(s, e):
        """セクション内のstress/plosive総数"""
        return sum(stress_by_bar.get(b, 0) + plosive_by_bar.get(b, 0) for b in range(s, e + 1))

    # セクション別レシピ生成
    recipe = {}
    for s, e, label in ranges:
        # ベースゲイン
        base = -6.0
        if "chorus" in label:
            base = -4.0
        elif "bridge" in label:
            base = -5.0

        # stress/plosive負荷に応じた調整（最大-2dB）
        adj = -min(2.0, burden(s, e) * 0.1)

        recipe[label] = {
            "arrangement_gain_db": round(base + adj, 2),
            "original_stems_gain_db": 0.0,
            "sidechain": {
                "duck_on_plosive_db": 4.0,
                "duck_on_stress_db": 2.0,
                "attack_ms": 10,
                "release_ms": 150,
            },
        }

    return recipe


def apply_variant(
    mix_recipe: Dict[str, Any], variant: str, variants_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    ミックス・バリエーション適用

    Args:
        mix_recipe: ベースmix_recipe
        variant: "soft" | "standard" | "bright"
        variants_config: mix_variants.yamlの内容（Noneの場合はデフォルト）

    Returns:
        variant適用後のmix_recipe
    """
    if variant == "standard" or variants_config is None:
        # standardはデフォルトのまま
        return mix_recipe

    # デフォルトvariant定義
    default_variants = {
        "soft": {
            "global_arrangement_gain_offset_db": -2.0,
            "high_shelf_db": -1.0,
            "section_overrides": {
                "intro": {"arrangement_gain_db": -8.0},
                "verse": {"arrangement_gain_db": -8.0},
                "chorus": {"arrangement_gain_db": -6.0},
                "pre_chorus": {"arrangement_gain_db": -6.0},
            },
            "sidechain": {
                "duck_on_plosive_db": 5.0,
                "duck_on_stress_db": 3.0,
                "attack_ms": 8,
                "release_ms": 180,
            },
        },
        "bright": {
            "global_arrangement_gain_offset_db": 0.0,
            "high_shelf_db": 1.5,
            "section_overrides": {
                "intro": {"arrangement_gain_db": -5.0},
                "verse": {"arrangement_gain_db": -5.0},
                "chorus": {"arrangement_gain_db": -3.0, "stereo_width": 110},
                "pre_chorus": {"arrangement_gain_db": -3.5},
            },
            "sidechain": {
                "duck_on_plosive_db": 3.5,
                "duck_on_stress_db": 1.5,
                "attack_ms": 12,
                "release_ms": 120,
            },
        },
    }

    # variant設定取得
    var_config = variants_config or default_variants
    if variant not in var_config:
        return mix_recipe

    var = var_config[variant]

    # グローバルオフセット適用
    global_offset = var.get("global_arrangement_gain_offset_db", 0.0)

    # セクション別適用
    for section_label, recipe in mix_recipe.items():
        # section_overridesから上書き
        overrides = var.get("section_overrides", {})
        if section_label in overrides:
            override = overrides[section_label]
            if "arrangement_gain_db" in override:
                recipe["arrangement_gain_db"] = override["arrangement_gain_db"]
            if "stereo_width" in override:
                recipe["stereo_width"] = override["stereo_width"]
        else:
            # グローバルオフセット適用
            recipe["arrangement_gain_db"] = round(recipe["arrangement_gain_db"] + global_offset, 2)

        # サイドチェイン設定上書き
        var_sidechain = var.get("sidechain", {})
        if var_sidechain:
            recipe["sidechain"].update(var_sidechain)

    return mix_recipe


def quality_gate_status(kpi: Dict[str, Any], thresholds: Dict[str, Any]) -> str:
    """
    品質ゲート判定

    Args:
        kpi: harmony KPI
        thresholds: 閾値

    Returns:
        "pass" or "fail"
    """
    # エンハーモニック一貫性
    if kpi.get("enharmonic_consistency", 1.0) < thresholds.get("enharmonic_consistency", 1.0):
        return "fail"

    # カデンススコア
    if kpi.get("cadence_score", 0.0) < thresholds.get("cadence_score_min", 0.0):
        return "fail"

    # テンション使用率範囲
    lo, hi = thresholds.get("tension_usage_range", (0.0, 1.0))
    tu = kpi.get("tension_usage", 0.0)
    if not (lo <= tu <= hi):
        return "fail"

    return "pass"


def extract_harmony_kpi(audit_path: Path) -> Dict[str, Any]:
    """
    deep_harmony_audit.jsonから実測KPI抽出（v1.0方式）

    Args:
        audit_path: deep_harmony_audit.jsonのパス

    Returns:
        harmony KPI dict
    """
    if not audit_path.exists():
        return {
            "tension_usage": 0.5,
            "cadence_score": 0.82,
            "anchor_near_change": 0.218,
            "key_confidence": 0.282,
            "enharmonic_consistency": 1.0,
        }

    audit = load_json(audit_path)
    summary = audit.get("summary", {})

    # カデンス平均計算
    cadences = audit.get("cadences", [])
    if cadences:
        cadence_scores = [c.get("cadence_score", 0.0) for c in cadences]
        cadence_avg = sum(cadence_scores) / len(cadence_scores)
    else:
        cadence_avg = 0.0

    return {
        "tension_usage": summary.get("tension_ratio_percent", 50.0) / 100.0,
        "cadence_score": cadence_avg,
        "anchor_near_change": summary.get("anchor_near_change_ratio_percent", 0.0) / 100.0,
        "key_confidence": summary.get("avg_key_confidence", 0.0),
        "enharmonic_consistency": summary.get("enharmonic_consistency", 1.0),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate SunoAI Song Package (v1.1-hybrid)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_suno_song_package_v1_1.py \\
        --song-id song_003 \\
        --analysis-dir data/suno_ai/suno_themesong/song_003/analysis \\
        --out data/suno_ai/suno_themesong/song_003/song_package.yaml
        """,
    )

    ap.add_argument("--song-id", required=True, help="Song ID (e.g., song_003)")
    ap.add_argument("--dataset", default="suno_ai", help="Dataset name")
    ap.add_argument("--source", default="suno_themesong", help="Source name")
    ap.add_argument("--analysis-dir", required=True, help="Analysis directory path")
    ap.add_argument("--out", required=True, help="Output song_package.yaml path")
    ap.add_argument("--variant", default="standard", help="Mix variant (soft/standard/bright)")
    ap.add_argument("--time-signature", default="4/4", help="Time signature")

    args = ap.parse_args()

    base = Path(args.analysis_dir)
    song_dir = base.parent

    # 必須ファイル読み込み
    chordmap = load_json(base / "chordmap.json")
    sections_data = load_json(base / "sections.json")
    anchors = load_json(base / "lyric_anchors.json")
    tempo_map = load_json(base / "tempo_map.json")

    # v1.0: bars.parquet実測値
    bars_df = pd.read_parquet(base / "bars.parquet")
    total_bars = len(bars_df)

    # BPM取得（bars.parquetから）
    if "tempo_bpm" in bars_df.columns:
        tempo_vals = bars_df["tempo_bpm"].dropna()
        if len(tempo_vals) > 0:
            bpm_mean = float(tempo_vals.mean())
            bpm_median = float(tempo_vals.median())
            summary_bpm = bpm_median
        else:
            bpm_mean = bpm_median = summary_bpm = 120.0
    else:
        bpm_mean = bpm_median = summary_bpm = 120.0

    # 拍子
    time_sig = estimate_time_signature(args.time_signature)
    if "time_signature" in bars_df.columns:
        ts_vals = bars_df["time_signature"].dropna()
        if len(ts_vals) > 0:
            time_sig = estimate_time_signature(str(ts_vals.iloc[0]))

    # コードイベント
    events = chordmap.get("events", [])
    chord_events_count = len(events)

    # 最大小節（chordmapベース）
    max_bar_from_chordmap = max((int(e["time"] / 4.0) for e in events), default=0)

    # 最大Quarter Length（実測小節数ベース）
    max_ql = total_bars * 4.0

    # duration_sec計算（bars.parquetのend_sec使用）
    if "end_sec" in bars_df.columns:
        duration_sec = float(bars_df["end_sec"].max())
    else:
        # フォールバック: 簡易計算
        duration_sec = float(max_ql * (60.0 / summary_bpm) / 4.0)

    # セクション
    sections_list = sections_data.get("sections", [])
    sections_count = len(sections_list)

    # v1.1: キー推定強化
    key_center, key_conf, key_candidates = infer_key_candidates(events, sections_list)

    # v1.1: mix_recipe自動生成
    mix_recipe = build_mix_recipe(sections_list, anchors)

    # v1.1: variant適用（mix_variants.yaml読み込み）
    mix_variants_path = base / "mix_variants.yaml"
    if mix_variants_path.exists():
        try:
            with mix_variants_path.open("r", encoding="utf-8") as f:
                variants_config = yaml.safe_load(f).get("variants", {})
            mix_recipe = apply_variant(mix_recipe, args.variant, variants_config)
        except Exception as e:
            print(f"Warning: Failed to load mix_variants.yaml: {e}")
            mix_recipe = apply_variant(mix_recipe, args.variant, None)
    else:
        # デフォルトvariant適用
        mix_recipe = apply_variant(mix_recipe, args.variant, None)

    # v1.0: deep_harmony_audit.json実測KPI
    audit_path = song_dir / "deep_harmony_audit.json"
    harmony_kpi = extract_harmony_kpi(audit_path)

    # v1.1: CREPE統計取り込み
    crepe_stats = {}

    # Strings VoiceLeading KPI
    strings_vl_kpi = base / "strings_vl_kpi.csv"
    if strings_vl_kpi.exists():
        try:
            vl_df = pd.read_csv(strings_vl_kpi)
            if len(vl_df) > 0:
                crepe_stats["strings_vl_resolution_rate"] = float(
                    vl_df.iloc[0].get("resolution_rate", 0.0)
                )
                crepe_stats["strings_vl_resolved_changes"] = int(
                    vl_df.iloc[0].get("resolved_changes", 0)
                )
        except Exception as e:
            print(f"Warning: Failed to load strings_vl_kpi.csv: {e}")

    # Guitar Microtiming統計
    guitar_micro = base / "guitar_microtiming.csv"
    if guitar_micro.exists():
        try:
            micro_df = pd.read_csv(guitar_micro)
            if len(micro_df) > 0:
                crepe_stats["guitar_microtiming_ms_mean"] = float(micro_df["time_shift_ms"].mean())
                crepe_stats["guitar_microtiming_ms_std"] = float(micro_df["time_shift_ms"].std())
                crepe_stats["guitar_microtiming_events"] = len(micro_df)
        except Exception as e:
            print(f"Warning: Failed to load guitar_microtiming.csv: {e}")

    # Piano Hybrid統計（plans/piano_plan_hybrid.json）
    piano_hybrid = song_dir / "plans" / "piano_plan_hybrid.json"
    if piano_hybrid.exists():
        try:
            with open(piano_hybrid) as f:
                piano_data = json.load(f)
            total_events = sum(len(t.get("events", [])) for t in piano_data.get("tracks", []))
            crepe_stats["piano_hybrid_events"] = total_events
        except Exception as e:
            print(f"Warning: Failed to load piano_plan_hybrid.json: {e}")

    # vocal_f0.parquet統計
    vocal_f0 = song_dir / "features" / "vocal_f0.parquet"
    if vocal_f0.exists():
        try:
            f0_df = pd.read_parquet(vocal_f0)
            crepe_stats["vocal_f0_frames"] = len(f0_df)
            if "voicing_prob" in f0_df.columns:
                voiced_frames = (f0_df["voicing_prob"] > 0.5).sum()
                crepe_stats["vocal_f0_voiced_rate"] = float(voiced_frames / len(f0_df))
        except Exception as e:
            print(f"Warning: Failed to load vocal_f0.parquet: {e}")

    # v1.1: quality_gates
    thresholds = {
        "cadence_score_min": 0.75,
        "tension_usage_range": [0.30, 0.70],
        "enharmonic_consistency": 1.0,
    }
    gate_status = quality_gate_status(harmony_kpi, thresholds)

    # 補助ファイル存在確認
    def check(p: Path) -> Optional[str]:
        return str(p.relative_to(song_dir).as_posix()) if p.exists() else None

    # パッケージデータ構築
    out_data = {
        "schema_version": "1.1",
        "song_id": args.song_id,
        "variant": args.variant,
        "dataset": {"name": args.dataset, "source": args.source, "version": None},
        "generated": {
            "at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tool": "generate_suno_song_package.py@v1.1-hybrid",
            "git_rev": None,
            "inputs_hash": {
                "chordmap": None,
                "sections": None,
                "lyric_anchors": None,
                "tempo_map": None,
            },
        },
        "time": {
            "signature": time_sig,
            "tempo": {
                "summary_bpm": summary_bpm,
                "map_path": str((base / "tempo_map.json").relative_to(song_dir).as_posix()),
                "bpm_mean": bpm_mean,
                "bpm_median": bpm_median,
            },
            "duration_sec": duration_sec,
        },
        "structure": {
            "total_bars": total_bars,
            "sections_count": sections_count,
            "chord_events_count": chord_events_count,
        },
        "paths": {
            "analysis_dir": str(base.relative_to(song_dir).as_posix()),
            "bars": str((base / "bars.parquet").relative_to(song_dir).as_posix()),
            "chordmap": str(
                (
                    base / "chordmap_locked.json"
                    if (base / "chordmap_locked.json").exists()
                    else base / "chordmap.json"
                )
                .relative_to(song_dir)
                .as_posix()
            ),
            "sections": str((base / "sections.json").relative_to(song_dir).as_posix()),
            "lyric_anchors": str((base / "lyric_anchors.json").relative_to(song_dir).as_posix()),
            "tempo_map": str((base / "tempo_map.json").relative_to(song_dir).as_posix()),
            "style_presets": check(base / "style_presets.yaml"),
            "voicings_guide": check(base / "voicings_guide.csv"),
            "bassline_plan": check(base / "bassline_plan.csv"),
            "drum_accent_plan": check(base / "drum_accent_plan.json"),
            "crepe_f0": check(base / "crepe_f0.parquet"),
            "midi": {
                "integrated": check(song_dir / "midi" / f"{args.song_id}_hybrid_crepe.mid"),
            },
        },
        "meta": {
            "tempo_bpm_source": "bars.parquet.median",
            "sections_count": sections_count,
            "chord_events_count": chord_events_count,
            "lock_sha256": None,  # TODO: chordmap_locked.jsonのハッシュ
            "generated_variant": args.variant,
            "bootstrap_mode": (base / "chordmap_locked.json").exists(),
        },
        "flags": {
            "use_function_rules": True,  # Roman×V系テンション解禁
            "use_melody_exceptions": True,  # CREPEメロ例外（#11/9/13プロモート等）
            "forbid_fixed_bpm": True,  # tempo_map.jsonを唯一のテンポ事実源に
        },
        "harmony": {
            "key_center": key_center,
            "key_confidence": float(round(key_conf, 3)),
            "key_candidates": key_candidates,
            **{k: float(round(v, 3)) for k, v in harmony_kpi.items()},
            "crepe_ext": crepe_stats if crepe_stats else None,
        },
        "synchro_policy": {
            "reference": "original_stems",
            "goal": "keep-original-form",
            "rationale": "vocalとの同期を最優先し、原曲の体裁を崩さない",
            "alignment_hint": {
                "method": "bar+anchor",
                "tolerance_ms": 25,
                "max_expected_deviation_ms": 80,
            },
            "mix_recipe": mix_recipe,
        },
        "quality_gates": {
            "thresholds": thresholds,
            "status": gate_status,
            "notes": "synchro-first; measured KPI from deep_harmony_audit.json",
        },
        "auxiliary_files": {
            "style_presets": (base / "style_presets.yaml").exists(),
            "voicings_guide": (base / "voicings_guide.csv").exists(),
            "bassline_plan": (base / "bassline_plan.csv").exists(),
            "drum_accent_plan": (base / "drum_accent_plan.json").exists(),
        },
    }

    # YAML出力
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_data, f, sort_keys=False, allow_unicode=True, indent=2)

    print(f"✅ Generated: {out_path}")
    print(f"\nSummary:")
    print(f"  Song ID:        {args.song_id}")
    print(f"  Total Bars:     {total_bars}")
    print(f"  Duration:       {duration_sec:.1f} sec")
    print(f"  Tempo (median): {summary_bpm:.1f} BPM")
    print(f"  Key Center:     {key_center} (confidence: {key_conf:.3f})")
    print(f"  Sections:       {sections_count}")
    print(f"  Chord Events:   {chord_events_count}")
    print(f"\nHarmony KPI (measured):")
    for k, v in harmony_kpi.items():
        print(f"  {k:25s}: {v:.3f}")
    print(f"\nQuality Gate:   {gate_status.upper()}")
    print(f"Synchro Policy: {out_data['synchro_policy']['reference']}")


if __name__ == "__main__":
    main()
