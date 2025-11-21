#!/usr/bin/env python3
"""
品質ゲート: plans/*.json のフィル/リフ発火率を検証

目的:
  postprocess_plans_ignore_mute.py の前に実行し、
  「入るべき所に入っていない」を即検知する。

検証項目:
  1. セクション境界あたりのフィル本数 ≥ 1
  2. Chorus 内のギター/ストリングス リフ率 ≥ X%
  3. コード外音率 ≤ 2%（music21検査、将来実装）
  4. 過密チェック: 16分音符の連続過多を防ぐ

使用例:
  python scripts/quality_gate_fill_riff.py \
    --plans-dir data/suno_ai/suno_themesong/song_004/plans \
    --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
    --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
    --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
    --min-boundary-fill-rate 0.8 \
    --min-chorus-riff-rate 0.3

終了コード:
  0: 全チェックPASS
  1: 警告あり（継続可能）
  2: エラーあり（修正必要）

参照:
  ChatGPT guidance (2025-11-12)
  「品質ゲートを postprocess 前に回せば、入るべき所に入っていないを即検知できます。」
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable, Optional
import pandas as pd
import yaml


def load_sections(sections_path: Path) -> List[Dict[str, Any]]:
    """Load sections.json"""
    with open(sections_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sections", data)


def load_policy(policy_path: Path) -> Dict[str, Any]:
    """Load policy YAML"""
    with open(policy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_plan(plan_path: Path) -> Dict[str, Any]:
    """Load plan JSON"""
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_plan_events(plan: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield all events in a plan regardless of storage layout."""
    if not plan:
        return

    events = plan.get("events")
    if isinstance(events, list):
        for event in events:
            yield event

    for track in plan.get("tracks", []):
        for event in track.get("events", []):
            yield event


def check_boundary_fills(
    drums_plan: Dict[str, Any],
    sections: List[Dict[str, Any]],
    bars_df: pd.DataFrame,
    min_rate: float = 0.8,
) -> Tuple[bool, str]:
    """
    Check: セクション境界あたりのフィル本数 ≥ min_rate

    Returns:
        (passed, message)
    """
    if not drums_plan:
        return False, "⚠️  Drums plan not found"

    # Extract fill events from drums plan
    fill_events = [
        event
        for event in iter_plan_events(drums_plan)
        if event.get("is_fill", False) or "fill" in str(event.get("type", "")).lower()
    ]

    # Check boundary coverage
    boundary_bars = []
    for sec in sections:
        end_bar = sec.get("end_bar", sec.get("bar_end"))
        if end_bar is not None and end_bar > 0:
            boundary_bars.append(end_bar - 1)

    if not boundary_bars:
        return True, "✓ No section boundaries to check"

    # Count fills at boundaries
    fills_at_boundaries = 0
    for bar_idx in boundary_bars:
        # Check if any fill event is in this bar
        bar_row = bars_df[bars_df["bar_index"] == bar_idx]
        if bar_row.empty:
            continue

        # Support both start_beat (legacy) and start_ql (new format)
        bar_start = bar_row.iloc[0].get(
            "start_beat", bar_row.iloc[0].get("start_ql", bar_idx * 4.0)
        )
        bar_end = bar_row.iloc[0].get(
            "end_beat", bar_row.iloc[0].get("end_ql", (bar_idx + 1) * 4.0)
        )

        for fill in fill_events:
            fill_time = fill.get("time_ql", fill.get("start_beats", 0))
            if bar_start <= fill_time < bar_end:
                fills_at_boundaries += 1
                break

    rate = fills_at_boundaries / len(boundary_bars) if boundary_bars else 0
    passed = rate >= min_rate

    status = "✅" if passed else "❌"
    msg = f"{status} Boundary fills: {fills_at_boundaries}/{len(boundary_bars)} ({rate*100:.1f}%) [threshold: {min_rate*100:.0f}%]"

    return passed, msg


def check_chorus_riff_rate(
    guitar_plan: Dict[str, Any],
    strings_plan: Dict[str, Any],
    sections: List[Dict[str, Any]],
    bars_df: pd.DataFrame,
    min_rate: float = 0.3,
) -> Tuple[bool, str]:
    """
    Check: Chorus 内のギター/ストリングス リフ率 ≥ min_rate

    Returns:
        (passed, message)
    """
    # Find chorus bars
    chorus_bars = []
    for sec in sections:
        label = sec.get("label", "")
        if "chorus" in label.lower():
            start_bar = sec.get("start_bar", sec.get("bar_start", 0))
            end_bar = sec.get("end_bar", sec.get("bar_end", 0))
            chorus_bars.extend(range(start_bar, end_bar))

    if not chorus_bars:
        return True, "✓ No chorus sections to check"

    # Extract riff events
    riff_events = []
    for plan in [guitar_plan, strings_plan]:
        if not plan:
            continue
        for event in iter_plan_events(plan):
            if event.get("is_riff", False) or "riff" in str(event.get("type", "")).lower():
                riff_events.append(event)

    # Count riff coverage in chorus
    riff_bars = set()
    for riff in riff_events:
        riff_time = riff.get("time_ql", riff.get("start_beats", 0))
        for bar_idx in chorus_bars:
            bar_row = bars_df[bars_df["bar_index"] == bar_idx]
            if bar_row.empty:
                continue
            bar_start = bar_row.iloc[0].get(
                "start_beat", bar_row.iloc[0].get("start_ql", bar_idx * 4.0)
            )
            bar_end = bar_row.iloc[0].get(
                "end_beat", bar_row.iloc[0].get("end_ql", (bar_idx + 1) * 4.0)
            )
            if bar_start <= riff_time < bar_end:
                riff_bars.add(bar_idx)
                break

    rate = len(riff_bars) / len(chorus_bars) if chorus_bars else 0
    passed = rate >= min_rate

    status = "✅" if passed else "❌"
    msg = f"{status} Chorus riff rate: {len(riff_bars)}/{len(chorus_bars)} bars ({rate*100:.1f}%) [threshold: {min_rate*100:.0f}%]"

    return passed, msg


def check_density(
    plans: List[Dict[str, Any]],
    max_density: float = 0.75,
) -> Tuple[bool, str]:
    """
    Check: 過密チェック（16分音符の連続過多を防ぐ）

    Returns:
        (passed, message)
    """
    # TODO: Implement density check
    # For now, always pass
    return True, f"✓ Density check (max: {max_density}) - not implemented yet"


def extract_events_from_plan(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize plan object to a flat list of events."""
    if not plan:
        return []
    events = []
    if "events" in plan and isinstance(plan["events"], list):
        events.extend(plan["events"])
    elif "tracks" in plan and isinstance(plan["tracks"], list):
        for t in plan["tracks"]:
            events.extend(t.get("events", []))
    else:
        # Unknown shape: try to find lists under keys
        for v in plan.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "time_ql" in item:
                        events.append(item)
    return events


def infer_bar_index(event: Dict[str, Any]) -> Optional[int]:
    """Best-effort bar index resolver for QA metrics."""
    for key in ("bar_idx", "bar_index"):
        if key in event:
            try:
                return int(event[key])
            except (TypeError, ValueError):
                continue

    t = event.get("time_ql", event.get("start_beats"))
    if t is None:
        return None

    try:
        return int(float(t) // 4)
    except (TypeError, ValueError):
        return None


def calc_active_bar_ratio(plan: Dict[str, Any], total_bars: int) -> float:
    events = extract_events_from_plan(plan)
    if not events or total_bars <= 0:
        return 0.0
    bars_with_sound = set()
    for e in events:
        b = infer_bar_index(e)
        if b is None:
            continue
        bars_with_sound.add(b)
    return float(len(bars_with_sound)) / float(total_bars)


def exceeds_notes_per_bar(plan: Dict[str, Any], limit: int) -> bool:
    events = extract_events_from_plan(plan)
    if not events:
        return False
    counts = {}
    for e in events:
        b = infer_bar_index(e)
        if b is None:
            continue
        counts[b] = counts.get(b, 0) + 1
        if counts[b] > limit:
            return True
    return False


def has_mute_events(plan: Dict[str, Any]) -> bool:
    events = extract_events_from_plan(plan)
    for e in events:
        if e.get("mute", False) or e.get("is_muted", False):
            return True
    return False


def check_rhythmai_density(
    plan: Optional[Dict[str, Any]],
    bars_df: pd.DataFrame,
    wiggle: int = 2,
    max_bad: int = 4,
) -> Tuple[bool, str]:
    """Validate that RhythmAI event density respects vocal-driven bounds."""

    if plan is None:
        return True, "✓ RhythmAI plan missing; density check skipped"

    required_cols = {"bar_index", "vocal_density_floor", "vocal_density_ceiling"}
    if not required_cols.issubset(set(bars_df.columns)):
        return True, "✓ Vocal density bounds unavailable; skipping RhythmAI density check"

    event_counts: Dict[int, int] = {}
    for event in extract_events_from_plan(plan):
        bar_idx = infer_bar_index(event)
        if bar_idx is None:
            continue
        event_counts[bar_idx] = event_counts.get(bar_idx, 0) + 1

    violations = []
    for row in bars_df.itertuples():
        floor = getattr(row, "vocal_density_floor", None)
        ceil = getattr(row, "vocal_density_ceiling", None)
        if floor is None or ceil is None:
            continue
        lower = max(0, int(floor) - wiggle)
        upper = int(ceil) + wiggle
        count = event_counts.get(int(row.bar_index), 0)
        if count < lower or count > upper:
            violations.append(
                {
                    "bar": int(row.bar_index),
                    "profile": getattr(row, "vocal_profile", "neutral"),
                    "count": count,
                    "target": [int(floor), int(ceil)],
                }
            )

    if not violations:
        return True, "✅ RhythmAI density within vocal target bounds (0 violations)"

    passed = len(violations) <= max_bad
    status = "✅" if passed else "❌"
    preview = ", ".join(
        f"bar {v['bar']}->{v['count']} (target {v['target'][0]}-{v['target'][1]}, {v['profile']})"
        for v in violations[:5]
    )
    msg = (
        f"{status} RhythmAI density violations: {len(violations)} bars "
        f"(allow ≤ {max_bad}) | {preview}"
    )
    return passed, msg


def estimate_tension_adoption(plans: List[Dict[str, Any]]) -> float:
    """A crude estimation: count events with an 'extension' flag or types indicating extension.
    Returns fraction in [0,1]."""
    total = 0
    ext = 0
    for p in plans:
        evs = extract_events_from_plan(p)
        for e in evs:
            total += 1
            if e.get("is_extension", False):
                ext += 1
            elif isinstance(e.get("type"), str) and any(
                x in e.get("type") for x in ["add9", "9th", "11th", "13th", "sus", "ext"]
            ):
                ext += 1
    if total == 0:
        return 0.0
    return float(ext) / float(total)


def check_register_violations(
    plans: Dict[str, Dict[str, Any]], policy: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Phase 1 register check: 各楽器のeventsが定義された音域内にあるか検証

    Returns:
        (passed, violations_list)
    """
    if not policy.get("qagate", {}).get("register_check", True):
        return True, []

    violations = []
    instrument_registers = policy.get("instruments", {})

    for inst_name, plan in plans.items():
        if not plan:
            continue

        reg = instrument_registers.get(inst_name, {}).get("register")
        if not reg or "min" not in reg or "max" not in reg:
            continue

        lo, hi = int(reg["min"]), int(reg["max"])
        events = extract_events_from_plan(plan)

        bad_events = []
        for e in events:
            note = e.get("note")
            if note is not None and not (lo <= int(note) <= hi):
                bad_events.append(e)

        if bad_events:
            violations.append(
                f"❌ {inst_name}: {len(bad_events)} notes out of register [{lo},{hi}]"
            )

    return len(violations) == 0, violations


def main():
    ap = argparse.ArgumentParser(description="Quality gate for fill/riff firing rate")
    ap.add_argument("--plans-dir", required=True, help="Directory containing *_plan.json")
    ap.add_argument("--bars", required=True, help="bars_with_slots.parquet")
    ap.add_argument("--sections", required=True, help="sections.json")
    ap.add_argument("--policy", required=True, help="policy YAML")
    ap.add_argument(
        "--min-boundary-fill-rate",
        type=float,
        default=0.8,
        help="Minimum boundary fill rate (default: 0.8 = 80%%)",
    )
    ap.add_argument(
        "--min-chorus-riff-rate",
        type=float,
        default=0.3,
        help="Minimum chorus riff rate (default: 0.3 = 30%%)",
    )
    ap.add_argument(
        "--max-density",
        type=float,
        default=0.75,
        help="Maximum 16th note density (default: 0.75)",
    )
    ap.add_argument(
        "--rhythmai-plan-name",
        default="drums_plan_v2_rhythmai.json",
        help="Filename of the RhythmAI drums plan inside --plans-dir (default: drums_plan_v2_rhythmai.json)",
    )
    ap.add_argument(
        "--density-wiggle",
        type=int,
        default=2,
        help="Allowance (events) when comparing measured density to vocal bounds (default: 2)",
    )
    ap.add_argument(
        "--max-density-violations",
        type=int,
        default=4,
        help="Maximum RhythmAI density violations permitted before failing (default: 4)",
    )
    args = ap.parse_args()

    plans_dir = Path(args.plans_dir)
    bars_df = pd.read_parquet(args.bars)
    sections = load_sections(Path(args.sections))
    policy = load_policy(Path(args.policy))

    print("=" * 70)
    print("🎯 Quality Gate: Fill/Riff Firing Rate")
    print("=" * 70)

    # Load plans
    drums_plan_path = plans_dir / "drums_plan.json"
    guitar_plan_path = plans_dir / "guitar_plan.json"
    strings_plan_path = plans_dir / "strings_plan.json"

    drums_plan = load_plan(drums_plan_path) if drums_plan_path.exists() else None
    guitar_plan = load_plan(guitar_plan_path) if guitar_plan_path.exists() else None
    strings_plan = load_plan(strings_plan_path) if strings_plan_path.exists() else None
    rhythmai_plan_path = plans_dir / args.rhythmai_plan_name
    rhythmai_plan = load_plan(rhythmai_plan_path) if rhythmai_plan_path.exists() else None

    # Checks (extended, policy-driven)
    qcfg = policy.get("qagate", {})
    report = {}

    # 1. Boundary fills
    passed_b, msg_b = check_boundary_fills(
        drums_plan,
        sections,
        bars_df,
        min_rate=qcfg.get("boundary_fill_coverage", args.min_boundary_fill_rate),
    )
    report["boundary_fill"] = {"ok": passed_b, "msg": msg_b}
    print(f"\n1. {msg_b}")

    # 2. Chorus riff rate
    passed_r, msg_r = check_chorus_riff_rate(
        guitar_plan,
        strings_plan,
        sections,
        bars_df,
        min_rate=qcfg.get("chorus_riff_coverage", args.min_chorus_riff_rate),
    )
    report["chorus_riff"] = {"ok": passed_r, "msg": msg_r}
    print(f"2. {msg_r}")

    passed_rd, msg_rd = check_rhythmai_density(
        rhythmai_plan,
        bars_df,
        wiggle=args.density_wiggle,
        max_bad=args.max_density_violations,
    )
    report["rhythmai_density"] = {"ok": passed_rd, "msg": msg_rd}
    print(f"3. {msg_rd}")

    # 3. Instrument density floors / ceilings
    per_inst = {}
    total_bars = len(bars_df)
    inst_plans = {"guitar": guitar_plan, "strings": strings_plan, "drums": drums_plan}
    min_den = qcfg.get("min_density", {})
    max_np = qcfg.get("max_density_per_bar", {})
    inst_ok = True
    for inst, pl in inst_plans.items():
        if not pl:
            per_inst[inst] = {"active_ratio": 0.0}
            if inst in min_den and float(min_den[inst]) > 0:
                inst_ok = False
            continue
        ar = calc_active_bar_ratio(pl, total_bars)
        per_inst[inst] = {"active_ratio": ar}
        if inst in min_den and ar < float(min_den[inst]):
            inst_ok = False
            print(f"❌ {inst} active_ratio {ar:.2f} < min {min_den[inst]}")
        if inst in max_np and exceeds_notes_per_bar(pl, int(max_np[inst])):
            inst_ok = False
            print(f"❌ {inst} exceeds per-bar notes limit {max_np[inst]}")

    report["per_instrument"] = per_inst

    # 4. Register check (Phase 1)
    piano_plan_path = plans_dir / "piano_plan.json"
    bass_plan_path = plans_dir / "bass_plan.json"
    piano_plan = load_plan(piano_plan_path) if piano_plan_path.exists() else None
    bass_plan = load_plan(bass_plan_path) if bass_plan_path.exists() else None

    all_plans_raw = {
        "guitar": guitar_plan,
        "piano": piano_plan,
        "strings": strings_plan,
        "bass": bass_plan,
    }
    all_plans = {k: v for k, v in all_plans_raw.items() if v is not None}

    reg_ok, reg_violations = check_register_violations(all_plans, policy)
    report["register_violations"] = reg_violations
    if not reg_ok:
        inst_ok = False
        for viol in reg_violations:
            print(viol)

    # 5. No-mute rule
    forbid_mute = bool(qcfg.get("forbid_mute", True))
    mute_found = False
    for pl in [guitar_plan, strings_plan, drums_plan, piano_plan, bass_plan]:
        if pl and has_mute_events(pl):
            mute_found = True
            break
    if forbid_mute and mute_found:
        print("❌ Mute events found while forbid_mute=true")
        inst_ok = False
    report["mute_found"] = mute_found

    # 6. Tension adoption (Phase 2)
    target_tension = qcfg.get("tension_adoption_target", {})
    tension_ok = True
    tension_results = {}
    if isinstance(target_tension, dict):
        # Per-instrument targets: guitar=0.20, piano=0.15, etc.
        inst_plans_for_tension = {
            "guitar": guitar_plan,
            "piano": piano_plan,
            "strings": strings_plan,
        }
        for inst, target in target_tension.items():
            plan = inst_plans_for_tension.get(inst)
            if not plan:
                tension_results[inst] = {"actual": 0.0, "target": target, "ok": False}
                print(f"⚠️  {inst}: No plan found, can't measure tension adoption")
                continue

            # Count events with is_tension metadata
            total = 0
            tension_count = 0
            for event in plan.get("events", []):
                total += 1
                if event.get("is_tension", False):
                    tension_count += 1

            actual_rate = tension_count / total if total > 0 else 0.0
            passed = actual_rate >= float(target)
            tension_results[inst] = {
                "actual": actual_rate,
                "target": target,
                "ok": passed,
                "total_events": total,
                "tension_events": tension_count,
            }

            status = "✅" if passed else "❌"
            print(
                f"{status} {inst} tension adoption: {actual_rate*100:.1f}% "
                f"(target: {float(target)*100:.0f}%, {tension_count}/{total} events)"
            )
            if not passed:
                tension_ok = False

    report["tension_adoption"] = tension_results
    inst_ok = inst_ok and tension_ok

    # adopt = estimate_tension_adoption([p for p in [guitar_plan, strings_plan, piano_plan] if p])
    # report["tension_adoption"] = adopt

    # Final decision
    overall_ok = (
        passed_b
        and passed_r
        and passed_rd
        and inst_ok
        and reg_ok
        and (not forbid_mute or not mute_found)
    )

    # Print JSON report
    print("\n=== QAGate Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if overall_ok:
        print("\n✅ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("\n❌ QAGate FAILED")
        if args.policy:
            print("Review policy thresholds or inspect plans at --plans-dir")
        sys.exit(2)


if __name__ == "__main__":
    main()
