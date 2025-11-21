#!/usr/bin/env python3
# see previous cell for full docstring; minimized to avoid tool length errors
import json, csv, sys, math, re
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import yaml
except Exception:
    sys.exit("ERROR: pip install pyyaml")


NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def note_to_pc(note: str) -> Optional[int]:
    if not note:
        return None
    key = note.strip().upper().replace("♭", "B").replace("♯", "#")
    return NOTE_TO_PC.get(key)


def hz_to_midi(freq: float) -> Optional[float]:
    if not freq or freq <= 0:
        return None
    try:
        return 69.0 + 12.0 * math.log2(freq / 440.0)
    except (ValueError, ZeroDivisionError):
        return None


def tension_to_semitone(label: str) -> Optional[int]:
    if not label:
        return None
    accidental = 0
    core = label
    if core.startswith("bb"):
        accidental = -2
        core = core[2:]
    elif core.startswith("b"):
        accidental = -1
        core = core[1:]
    elif core.startswith("##"):
        accidental = 2
        core = core[2:]
    elif core.startswith("#"):
        accidental = 1
        core = core[1:]
    try:
        deg = int(core)
    except ValueError:
        return None

    base_map = {
        2: 2,
        4: 5,
        5: 7,
        6: 9,
        7: 10,
        9: 2,
        11: 5,
        13: 9,
    }
    if deg not in base_map:
        return None
    return (base_map[deg] + accidental) % 12


class MelodyAnalyzer:
    def __init__(self, df, quantize_cents: float = 35.0):
        self.quantize_semitones = max(5.0, float(quantize_cents)) / 100.0
        self.by_bar: Dict[int, List[float]] = {}
        grouped = df.groupby("bar_index")
        for bar_idx, group in grouped:
            pcs: List[float] = []
            for _, row in group.iterrows():
                if not bool(row.get("voiced", True)):
                    continue
                midi = hz_to_midi(float(row.get("f0_hz", 0.0)))
                if midi is None:
                    continue
                pcs.append(midi % 12.0)
            if pcs:
                self.by_bar[int(bar_idx)] = pcs

    def hit_ratio(self, bar_index: int, target_pc: float) -> float:
        pcs = self.by_bar.get(int(bar_index))
        if not pcs:
            return 0.0
        tol = self.quantize_semitones
        hits = 0
        for pc in pcs:
            diff = abs(pc - target_pc)
            diff = min(diff, 12.0 - diff)
            if diff <= tol:
                hits += 1
        return hits / float(len(pcs))


def resolve_song_path(song_root: Path, candidate: str) -> Path:
    path = Path(candidate)
    if not path.is_absolute():
        path = song_root / path
    return path.resolve()


def load_melody_analyzer(song_root: Path, cfg: Dict[str, Any]) -> Optional[MelodyAnalyzer]:
    if not cfg or not cfg.get("enable"):
        return None
    rel = cfg.get("f0_source")
    if not rel:
        return None
    f0_path = resolve_song_path(song_root, rel)
    if not f0_path.exists():
        print(f"[WARN] melody_exceptions source missing: {f0_path}")
        return None
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] pandas required for melody_exceptions but not available: {exc}")
        return None

    df = pd.read_parquet(f0_path)
    needed = {"bar_index", "f0_hz"}
    if not needed.issubset(df.columns):
        print(
            f"[WARN] melody_exceptions requires columns {needed}, got {sorted(df.columns)}; skipping"
        )
        return None
    if "voiced" not in df.columns:
        df["voiced"] = True
    quant = float(cfg.get("quantize_cents", 35.0))
    return MelodyAnalyzer(df, quantize_cents=quant)


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_sections_ranges(sec_obj: Dict[str, Any]) -> List[tuple]:
    res = []
    items = sec_obj.get("sections") or []
    if items and ("start_bar" in items[0] or "start" in items[0]):
        for x in items:
            sb = int(x.get("start_bar", x.get("start", 0)))
            eb = int(x.get("end_bar", x.get("end", sb)))
            lbl = str(x.get("label", "")).lower()
            res.append((sb, eb, lbl))
    else:
        items = sorted(
            [(int(x.get("bar", 0)), str(x.get("label", "")).lower()) for x in items],
            key=lambda t: t[0],
        )
        for i, (b, lbl) in enumerate(items):
            e = items[i + 1][0] - 1 if i + 1 < len(items) else b + 7
            res.append((b, e, lbl))
    return res


def bar_label(bar: int, ranges: List[tuple]) -> str:
    for s, e, l in ranges:
        if s <= bar <= e:
            return l
    return ""


def normalize_chord_events(cm_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    evs = cm_obj.get("events") or cm_obj.get("chords") or cm_obj
    if not isinstance(evs, list):
        raise ValueError("Unsupported chordmap format")
    out = []
    for e in evs:
        d = dict(e)
        if "time" in d:
            tql = float(d["time"])
        elif "time_ql" in d:
            tql = float(d["time_ql"])
        elif "bar" in d:
            tql = float(d["bar"]) * 4.0
        else:
            tql = 0.0
        d["time_ql"] = tql
        out.append(d)
    out.sort(key=lambda x: x["time_ql"])
    return out


def apply_policy_to_event(
    ev: Dict[str, Any],
    policy: Dict[str, Any],
    label: str,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def extend_unique(target: List[str], additions: List[str]) -> None:
        for item in additions or []:
            if item not in target:
                target.append(item)

    extras = extras or {}
    bar_index = int(ev.get("time_ql", 0.0) // 4.0)
    tensions_cfg = policy.get("tensions", {})
    tensions_allowed = list(tensions_cfg.get("base_allow", []))
    avoid_global = list(tensions_cfg.get("avoid_global", []))
    allow_by_label = tensions_cfg.get("allow_by_label", {})
    avoid_by_label = tensions_cfg.get("avoid_by_label", {})
    by_quality = tensions_cfg.get("by_quality", {})

    density_scale_by_label = policy.get("density", {}).get("scale_by_label", {})
    omit_third_labels = set(policy.get("voicing", {}).get("omit_third_in_labels", []))
    prefer_inv_by_quality = policy.get("voicing", {}).get("prefer_inversion_by_quality", {})
    register = policy.get("register", {"low": 40, "high": 84})

    function_rule_hits: List[str] = []
    voicing_flags: List[str] = []
    melody_promotions: List[str] = []
    melody_conflicts: List[str] = []
    melody_hit_summary: Dict[str, float] = {}

    if label in allow_by_label:
        extend_unique(tensions_allowed, allow_by_label[label])

    avoid_tensions = list(avoid_global)
    if label in avoid_by_label:
        extend_unique(avoid_tensions, avoid_by_label[label])

    quality_raw = (ev.get("quality") or "").strip().lower()
    quality_candidates = [quality_raw, quality_raw.strip("\"'")]
    if quality_raw.startswith("maj"):
        quality_candidates.append("maj")
    if quality_raw.startswith("m") and quality_raw not in ("maj", "maj7"):
        quality_candidates.append("m")
    if quality_raw.endswith("7"):
        quality_candidates.append("7")

    for q in quality_candidates:
        if not q:
            continue
        if q in by_quality:
            extend_unique(tensions_allowed, by_quality[q].get("allow", []))
            extend_unique(avoid_tensions, by_quality[q].get("avoid", []))
            break

    # Function-rule hooks (regex on Roman numerals or labels)
    function_rules = policy.get("function_rules", {}) or {}
    roman_lookup = extras.get("roman_by_bar", {}) if extras else {}
    roman_source = (
        roman_lookup.get(bar_index)
        or ev.get("roman")
        or ev.get("roman_numeral")
        or ev.get("function")
        or (ev.get("analysis", {}) or {}).get("roman")
        or ""
    )
    symbol_text = ev.get("symbol") or ev.get("quality") or ""
    for rule_name, cfg in function_rules.items():
        if not cfg or not cfg.get("enable"):
            continue
        rn_regex = cfg.get("rn_regex")
        matched = False
        for candidate in (roman_source, symbol_text):
            if candidate and rn_regex:
                if re.search(rn_regex, str(candidate), re.IGNORECASE):
                    matched = True
                    break
        if not matched and cfg.get("labels"):
            matched = label in cfg.get("labels", [])
        if not matched:
            continue
        function_rule_hits.append(str(rule_name))
        extend_unique(tensions_allowed, [str(x) for x in cfg.get("allow", [])])
        extend_unique(avoid_tensions, [str(x) for x in cfg.get("avoid", [])])
        for flag_key in ("dominant_open_13", "prefer_drop2", "suppress_root"):
            if cfg.get(flag_key) and flag_key not in voicing_flags:
                voicing_flags.append(flag_key)

    # Melody exceptions hook
    melody_ctx = extras.get("melody") if extras else None
    analyzer = melody_ctx.get("analyzer") if melody_ctx else None
    mel_cfg = melody_ctx.get("config") if melody_ctx else None
    root_pc = note_to_pc(ev.get("root"))
    if analyzer and mel_cfg and root_pc is not None:
        threshold = float(mel_cfg.get("promote_threshold", 0.25))
        threshold = max(0.05, min(1.0, threshold))
        if label == "verse" and mel_cfg.get("strict_on_verse"):
            threshold *= 1.2
        elif label == "chorus" and mel_cfg.get("lax_on_chorus"):
            threshold *= 0.8

        promote_targets = mel_cfg.get("promote_if_melody_contains", []) or []
        for tension in promote_targets:
            tension_str = str(tension)
            semi = tension_to_semitone(tension_str)
            if semi is None:
                continue
            target_pc = (root_pc + semi) % 12
            ratio = analyzer.hit_ratio(bar_index, target_pc)
            melody_hit_summary[tension_str] = round(ratio, 4)
            if ratio >= threshold:
                extend_unique(melody_promotions, [tension_str])
                extend_unique(tensions_allowed, [tension_str])

        avoid_targets = mel_cfg.get("avoid_if_melody_semiminor_second_to", []) or []
        conflict_threshold = threshold * 0.8
        for tension in avoid_targets:
            tension_str = str(tension)
            semi = tension_to_semitone(tension_str)
            if semi is None:
                continue
            conflict_pc = (root_pc + semi - 1) % 12
            ratio = analyzer.hit_ratio(bar_index, conflict_pc)
            melody_hit_summary[f"{tension_str}_conflict"] = round(ratio, 4)
            if ratio >= conflict_threshold:
                extend_unique(melody_conflicts, [tension_str])
                extend_unique(avoid_tensions, [tension_str])

    quality_key = quality_raw or (ev.get("symbol") or "").lower()
    return {
        "tensions_allowed": tensions_allowed,
        "avoid_tensions": avoid_tensions,
        "omit_third": (label in omit_third_labels),
        "prefer_inversion": prefer_inv_by_quality.get(quality_key, "root"),
        "register_low": int(register.get("low", 40)),
        "register_high": int(register.get("high", 84)),
        "density_scale": float(density_scale_by_label.get(label, 1.0)),
        "label": label,
        "function_rule_hits": function_rule_hits,
        "voicing_flags": voicing_flags,
        "melody_promotions": melody_promotions,
        "melody_conflicts": melody_conflicts,
        "melody_hit_summary": melody_hit_summary,
        "bar_index": bar_index,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--chordmap", required=True)
    ap.add_argument("--sections", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--policy-pad", required=False)
    ap.add_argument("--policy-guitar", required=False)
    ap.add_argument("--policy-piano", required=False)
    ap.add_argument("--policy-strings", required=False)
    ap.add_argument("--policy-bass", required=False)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chordmap_path = Path(a.chordmap)
    cm = normalize_chord_events(load_json(chordmap_path))
    sec_ranges = load_sections_ranges(load_json(Path(a.sections)))
    song_root = chordmap_path.parent
    if song_root.name == "analysis":
        song_root = song_root.parent

    targets = []
    if a.policy_pad:
        targets.append(("pad", Path(a.policy_pad)))
    if a.policy_guitar:
        targets.append(("guitar", Path(a.policy_guitar)))
    if a.policy_piano:
        targets.append(("piano", Path(a.policy_piano)))
    if a.policy_strings:
        targets.append(("strings", Path(a.policy_strings)))
    if a.policy_bass:
        targets.append(("bass", Path(a.policy_bass)))

    for role, pol_path in targets:
        policy = yaml.safe_load(pol_path.read_text(encoding="utf-8"))
        extras: Dict[str, Any] = {}
        mel_cfg = policy.get("melody_exceptions")
        melody_ctx = None
        if mel_cfg and mel_cfg.get("enable"):
            analyzer = load_melody_analyzer(song_root, mel_cfg)
            if analyzer:
                melody_ctx = {"config": mel_cfg, "analyzer": analyzer}
        if melody_ctx:
            extras["melody"] = melody_ctx
        view_events = []
        guide_rows = []
        for ev in cm:
            bar = int(ev["time_ql"] // 4.0)
            lbl = bar_label(bar, sec_ranges)
            directives = apply_policy_to_event(ev, policy, lbl, extras)
            merged = dict(ev)
            merged.update(directives)
            view_events.append(merged)
            guide_rows.append(
                {
                    "time_ql": merged["time_ql"],
                    "bar": bar,
                    "label": merged["label"],
                    "root": ev.get("root", ""),
                    "quality": ev.get("quality", ""),
                    "tensions_allowed": " ".join(merged["tensions_allowed"]),
                    "avoid_tensions": " ".join(merged["avoid_tensions"]),
                    "omit_third": int(merged["omit_third"]),
                    "prefer_inversion": merged["prefer_inversion"],
                    "register_low": merged["register_low"],
                    "register_high": merged["register_high"],
                    "density_scale": merged["density_scale"],
                    "function_rule_hits": " ".join(merged.get("function_rule_hits", [])),
                    "voicing_flags": " ".join(merged.get("voicing_flags", [])),
                    "melody_promotions": " ".join(merged.get("melody_promotions", [])),
                    "melody_conflicts": " ".join(merged.get("melody_conflicts", [])),
                    "melody_hit_summary": json.dumps(
                        merged.get("melody_hit_summary", {}), ensure_ascii=False, sort_keys=True
                    ),
                }
            )
        out_json = out_dir / f"chordmap_view_{role}.json"
        out_json.write_text(
            json.dumps(
                {"meta": {"role": role, "ql_per_bar": 4}, "events": view_events},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        out_csv = out_dir / f"voicings_guide_{role}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(guide_rows[0].keys()))
            w.writeheader()
            w.writerows(guide_rows)
        print("Wrote:", out_json, "and", out_csv)


if __name__ == "__main__":
    main()
