#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep harmony/structure audit for chordmap + sections + tempo + anchors + bars
- music21 があれば厳密解析に自動切替（roman/Chord）
- 無ければ軽量パーサ＋機能和声（T/S/D）で代替
出力:
  /song_dir/deep_harmony_audit.{json,md}
  /song_dir/chord_events_enriched.csv
  /song_dir/cadence_by_section.png
  /song_dir/anchor_distance_hist.png
閾値（採用基準の例）:
  - 強勢歌詞アンカーのコードチェンジ整合: ≥ 20%
  - セクション境界の終止（cadence_score≥0.5）率: ≥ 70%
  - 係留テンション(9/11/13/add/alt)使用率: 10–60% 推奨
  - 局所キー信頼度（best-second）の平均: ≥ 0.15 推奨
"""
import json, math, re, pathlib, sys
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- CLI ----------
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("song_dir", help="song_packages/.../song_XXX")
ap.add_argument("--bars-file", default="bars.parquet")
ap.add_argument("--chordmap-file", default="analysis/chordmap.json")
ap.add_argument("--sections-file", default="sections.json")
ap.add_argument("--tempo-file", default="tempo_map.json")
ap.add_argument("--anchors-file", default="lyric_anchors.json")
ap.add_argument("--win-bars", type=int, default=8, help="局所キー決定の窓幅(小節)")
ap.add_argument("--bpb", type=float, default=4.0, help="beats per bar")
ap.add_argument(
    "--require-music21", action="store_true", help="music21必須にする（無ければエラー）"
)
args = ap.parse_args()

ROOT = pathlib.Path(args.song_dir)
OUT_JSON = ROOT / "deep_harmony_audit.json"
OUT_MD = ROOT / "deep_harmony_audit.md"
OUT_CSV = ROOT / "chord_events_enriched.csv"
OUT_P1 = ROOT / "cadence_by_section.png"
OUT_P2 = ROOT / "anchor_distance_hist.png"


def jload(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- tempo normalization ----------
def normalize_tempo(data):
    """tempo_map.jsonを正規化してpoints配列を返す

    対応形式:
    1. points配列（dict要素）: [{"bar": 0, "beat": 0.0, "bpm": 120}, ...]
    2. tempo_points配列（リスト要素）: [[time_sec, bpm], ...]
    3. グローバルBPM: {"bpm": 89.3, ...}
    """
    points = data.get("points") or data.get("events") or data.get("map")  # tempo_map.jsonの場合

    # tempo_points形式（[time_sec, bpm]のリスト）チェック
    tempo_points_list = data.get("tempo_points")
    if tempo_points_list and isinstance(tempo_points_list, list):
        if tempo_points_list and isinstance(tempo_points_list[0], list):
            # [[time_sec, bpm], ...] 形式
            # 時刻情報を無視し、BPM変化のみカウント
            unique_bpms = []
            for tp in tempo_points_list:
                if len(tp) >= 2:
                    bpm = float(tp[1])
                    if not unique_bpms or abs(bpm - unique_bpms[-1]) > 0.1:
                        unique_bpms.append(bpm)
            # 複数の異なるBPMがある場合、ポイント数を返す
            return [{"bar": 0, "beat": 0.0, "bpm": unique_bpms[0]}] if unique_bpms else []

    if not points:
        # グローバルBPM→1点にフォールバック
        bpm = data.get("bpm") or data.get("tempo_bpm") or data.get("qpm")
        if bpm:
            points = [{"bar": 0, "beat": 0.0, "bpm": float(bpm)}]
        else:
            return []

    # 正規化（キー名ゆらぎ吸収）
    global_bpm = data.get("bpm") or data.get("tempo_bpm") or data.get("qpm") or 120.0
    norm = []
    for p in points:
        if isinstance(p, dict):
            norm.append(
                {
                    "bar": int(p.get("bar", 0)),
                    "beat": float(p.get("beat", p.get("start_beat", 0.0))),
                    "bpm": float(p.get("bpm", p.get("tempo_bpm", p.get("qpm", global_bpm)))),
                }
            )
    return norm


# ---------- load ----------
bars_df = pd.read_parquet(ROOT / args.bars_file)
chordmap = jload(ROOT / args.chordmap_file)
sections = jload(ROOT / args.sections_file)
tempo_raw = jload(ROOT / args.tempo_file)
tempo_points = normalize_tempo(tempo_raw)
anchors = jload(ROOT / args.anchors_file)

# ---------- music21 (optional) ----------
m21 = None
if not args.require_music21:
    try:
        import music21 as m21  # type: ignore
    except Exception:
        m21 = None
else:
    import music21 as m21  # fail if not present


# ---------- chord ingest ----------
def coerce_chord_events(obj, bpb=4.0):
    if isinstance(obj, dict) and "events" in obj:
        ev = obj["events"]
    elif isinstance(obj, dict) and "chords" in obj:
        ev = obj["chords"]
    elif isinstance(obj, list):
        ev = obj
    else:
        ev = []
    out = []
    for e in ev:
        if isinstance(e, dict):
            bar = e.get("bar", e.get("bar_index"))
            beat = e.get("beat", e.get("start_beat", e.get("start_beats", 0.0)))
            # timeフィールドからbar/beatを計算
            if bar is None and "time" in e:
                time_ql = float(e["time"])
                bar = int(time_ql // bpb)
                beat = time_ql % bpb
            sym = e.get("symbol", e.get("chord", e.get("name")))
            if bar is None:
                continue
            out.append({"bar": int(bar), "beat": float(beat or 0.0), "symbol": sym})
        elif isinstance(e, list) and len(e) >= 2:
            bar, beat = int(e[0]), float(e[1])
            sym = e[2] if len(e) > 2 else None
            out.append({"bar": bar, "beat": beat, "symbol": sym})
    out.sort(key=lambda x: (x["bar"], x["beat"]))
    return out


events = coerce_chord_events(chordmap, bpb=args.bpb)
bars_total = int(bars_df["bar_index"].max()) + 1
bpb = args.bpb


def ensure_duration(ev):
    seq = []
    for i, e in enumerate(ev):
        sb = e["bar"] * bpb + e["beat"]
        if i + 1 < len(ev):
            en = ev[i + 1]
            eb = en["bar"] * bpb + en["beat"]
        else:
            eb = (int(sb // bpb)) * bpb + bpb
        seq.append({**e, "start_beats": sb, "end_beats": eb, "dur_beats": max(0.0, eb - sb)})
    return seq


events = ensure_duration(events)

# ---------- lightweight parser (fallback) ----------
NOTE_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}
PC_TO_NAME = {v: k for k, v in NOTE_TO_PC.items() if len(k) == 1}
QUAL_PAT = (
    r"(maj7|M7|Δ7|maj9|M9|maj|min7|min9|m7|m9|m|dim7|dim|aug|sus2|sus4|7|9|11|13|add9|add11|add13)"
)
ALT_PAT = r"(?:(?:[#b](?:5|9|11|13))+)?"
CHORD_RE = re.compile(
    r"^(?P<root>[A-G](?:#|b)?)"
    r"(?P<qual>(?:{q})*)"
    r"(?P<alts>{a})"
    r"(?:/(?P<bass>[A-G](?:#|b)?))?$".format(q=QUAL_PAT, a=ALT_PAT)
)


def parse_chord(sym):
    if not isinstance(sym, str):
        return {"nc": True}
    s = sym.strip()
    if s in ("N.C.", "NC", "X", ""):
        return {"nc": True}
    m = CHORD_RE.match(s)
    if not m:
        return {"raw": s, "root_pc": None, "pcs": set()}
    d = m.groupdict()
    root = d["root"]
    qual = d.get("qual") or ""
    alts = d.get("alts") or ""
    root_pc = NOTE_TO_PC.get(root, None)
    deg = {0, 4, 7}
    if "m" in qual and "maj" not in qual:
        deg = {0, 3, 7}
    if "dim" in qual:
        deg = {0, 3, 6}
    if "aug" in qual:
        deg = {0, 4, 8}
    if "sus4" in qual:
        deg = {0, 5, 7}
    if "sus2" in qual:
        deg = {0, 2, 7}
    if "maj7" in qual or "M7" in qual or "Δ7" in qual:
        deg |= {11}
    elif "7" in qual:
        deg |= {10}
    if "9" in qual or "add9" in qual:
        deg |= {2}
    if "11" in qual or "add11" in qual:
        deg |= {5}
    if "13" in qual or "add13" in qual:
        deg |= {9}
    for token in re.findall(r"([#b])(5|9|11|13)", alts):
        sign, d = token
        d = int(d)
        base = {5: 7, 9: 2, 11: 5, 13: 9}[d]
        val = (base + 1) % 12 if sign == "#" else (base - 1) % 12
        deg.add(val)
    pcs = {(root_pc + x) % 12 for x in deg} if root_pc is not None else set()
    return {"raw": s, "root_pc": root_pc, "pcs": pcs}


for e in events:
    e["_parsed"] = parse_chord(e["symbol"])

MAJOR = {0, 2, 4, 5, 7, 9, 11}
MINOR = {0, 2, 3, 5, 7, 8, 10}
ALL_KEYS = [(pc, mode) for pc in range(12) for mode in ("maj", "min")]


def kset(kpc, mode):
    return {(kpc + x) % 12 for x in (MAJOR if mode == "maj" else MINOR)}


def score_key(pcset, kpc, mode):
    return len(pcset & kset(kpc, mode)) / max(1, len(pcset)) if pcset else 0.0


# 局所キー
bybar = [[] for _ in range(bars_total)]
for e in events:
    if 0 <= e["bar"] < bars_total:
        bybar[e["bar"]].append(e["_parsed"]["pcs"])

keys_tl = []
for b in range(bars_total):
    lo = max(0, b - args.win_bars // 2)
    hi = min(bars_total, b + args.win_bars // 2 + 1)
    agg = set().union(*[set().union(*bybar[i]) if bybar[i] else set() for i in range(lo, hi)])
    sc = []
    for kpc, mode in ALL_KEYS:
        sc.append((score_key(agg, kpc, mode), kpc, mode))
    sc.sort(reverse=True)
    best_score, best_kpc, best_mode = sc[0]

    # Improved confidence: count competing keys within 90% of best score
    if best_score > 0:
        threshold = best_score * 0.9
        competing_roots = set()
        for score, kpc, mode in sc:
            if score >= threshold:
                competing_roots.add(kpc)

        # Confidence inversely proportional to number of competing interpretations
        num_competing = len(competing_roots)
        if num_competing == 1:
            conf = best_score
        elif num_competing == 2:
            conf = best_score * 0.7
        elif num_competing <= 4:
            conf = best_score * 0.4
        else:
            conf = best_score * 0.2
    else:
        conf = 0.0

    keys_tl.append(
        {
            "bar": b,
            "key_pc": int(best_kpc),
            "mode": best_mode,
            "key": f"{PC_TO_NAME.get(best_kpc,str(best_kpc))}{''if best_mode=='maj' else 'm'}",
            "confidence": round(conf, 3),
        }
    )
    keys_tl.append(
        {
            "bar": b,
            "key_pc": int(best_kpc),
            "mode": best_mode,
            "key": f"{PC_TO_NAME.get(best_kpc,str(best_kpc))}{'' if best_mode=='maj' else 'm'}",
            "confidence": round(conf, 3),
        }
    )
key_by_bar = {k["bar"]: k for k in keys_tl}


# 機能和声（T/S/D）分類
def func_cls(root_pc, key_pc, mode):
    if root_pc is None:
        return "N"
    deg = (root_pc - key_pc) % 12
    if deg in {0, 9, 4}:
        return "T"  # I, vi, iii
    if deg in {2, 5}:
        return "S"  # ii, IV
    if deg in {7, 11, 10}:
        return "D"  # V, vii°, bVII
    return "O"


for e in events:
    k = key_by_bar.get(e["bar"])
    e["_func"] = func_cls(e["_parsed"]["root_pc"], k["key_pc"], k["mode"]) if k else "N"


# セクション正規化（両対応：pointer型 / span型）
def norm_secs(obj, bars_total=100):
    L = (
        obj["sections"]
        if isinstance(obj, dict) and "sections" in obj
        else (obj if isinstance(obj, list) else [])
    )
    if not L:
        return []

    # span型（start_bar/end_bar付き）の場合
    if "start_bar" in L[0] and "end_bar" in L[0]:
        out = []
        for s in L:
            if not isinstance(s, dict):
                continue
            label = s.get("label", s.get("name", "section"))
            sb = s.get("start_bar")
            eb = s.get("end_bar")
            if sb is None or eb is None:
                continue
            out.append((label, int(sb), int(eb)))
        return sorted(out, key=lambda x: x[1])

    # pointer型（barのみ）の場合 → span型に変換
    if "bar" in L[0] and "start_bar" not in L[0]:
        ptrs = sorted(L, key=lambda x: int(x.get("bar", 0)))
        spans = []
        for i, it in enumerate(ptrs):
            start = int(it["bar"])
            label = it.get("label", it.get("name", "section"))
            # 次の境界-1をendとする（最後はbars_total-1）
            end = (int(ptrs[i + 1]["bar"]) - 1) if i + 1 < len(ptrs) else (bars_total - 1)
            end = max(end, start)
            spans.append((label, start, end))
        return spans

    # 旧形式（start_bar別名対応）
    out = []
    for i, s in enumerate(L):
        if not isinstance(s, dict):
            continue
        label = s.get("label", s.get("name", "section"))
        # start_bar優先、なければbarフィールド使用
        sb = s.get(
            "start_bar", s.get("bar_start", s.get("startBar", s.get("barIndexStart", s.get("bar"))))
        )
        # end_bar優先、なければ次のセクションの開始-1
        eb = s.get("end_bar", s.get("bar_end", s.get("endBar", s.get("barIndexEnd"))))
        if sb is None:
            continue
        # end_barがない場合、次のセクションの開始-1を使用
        if eb is None:
            if i + 1 < len(L):
                next_start = L[i + 1].get(
                    "start_bar", L[i + 1].get("bar_start", L[i + 1].get("bar"))
                )
                eb = next_start - 1 if next_start is not None else sb
            else:
                eb = bars_total - 1
        out.append((label, int(sb), int(eb)))
    return sorted(out, key=lambda x: x[1])


secs = norm_secs(sections, bars_total=bars_total)


def cadence_score(sb, eb):
    """Enhanced cadence scoring with tension chord support"""
    # Get chords at section end (last 3 bars) and start
    tail = [e for e in events if e["bar"] in {eb - 2, eb - 1, eb}]
    head = [e for e in events if e["bar"] == sb]

    if not tail:
        return 0.0

    # Check for dominant/tension indicators:
    # 1. D function (V, vii°)
    # 2. Dominant 7th (X7, X9 - not maj7)
    # 3. Sus4 chords
    # 4. add9 chords (moderate tension)
    has_strong_cadence = False
    has_weak_tension = False

    for e in tail:
        func = e.get("_func", "N")
        sym = e.get("symbol", "")

        # Strong cadence indicators
        if func == "D":
            has_strong_cadence = True
            break

        if "7" in sym and "maj7" not in sym.lower() and "M7" not in sym:
            has_strong_cadence = True
            break

        if "sus4" in sym.lower():
            has_strong_cadence = True
            break

        # Weak tension indicators
        if "add9" in sym.lower() or "add11" in sym.lower() or "add13" in sym.lower():
            has_weak_tension = True

    # Check for tonic/stable start (T function or 6th chords)
    has_stable_start = False
    for e in head:
        func = e.get("_func", "N")
        sym = e.get("symbol", "")
        if func == "T" or "6" in sym or "maj9" in sym.lower():
            has_stable_start = True
            break

    # Scoring:
    # 1.0: Strong cadence + stable start
    # 0.7: Strong cadence alone
    # 0.5: Stable start only
    # 0.3: Weak tension + stable start
    # 0.2: Weak tension alone
    # 0.0: No cadence features
    if has_strong_cadence and has_stable_start:
        return 1.0
    elif has_strong_cadence:
        return 0.7
    elif has_stable_start:
        return 0.5
    elif has_weak_tension and has_stable_start:
        return 0.3
    elif has_weak_tension:
        return 0.2
    else:
        return 0.0


cadences = [
    {"section": lab, "start_bar": sb, "end_bar": eb, "cadence_score": cadence_score(sb, eb)}
    for (lab, sb, eb) in secs
]

# 転調（短区間は圧縮）
mods = []
prev = None
run = 0
for k in keys_tl:
    tu = (k["key_pc"], k["mode"])
    if tu != prev:
        if prev is not None and run >= 4:
            mods[-1]["end_bar"] = k["bar"]
        mods.append(
            {
                "start_bar": k["bar"],
                "key": f"{PC_TO_NAME.get(tu[0],tu[0])}{'' if tu[1]=='maj' else 'm'}",
                "end_bar": k["bar"],
            }
        )
        prev = tu
        run = 1
    else:
        run += 1
        mods[-1]["end_bar"] = k["bar"]
mods_comp = []
for m in mods:
    if not mods_comp:
        mods_comp.append(m)
        continue
    if m["key"] == mods_comp[-1]["key"] or (m["end_bar"] - m["start_bar"] < 4):
        mods_comp[-1]["end_bar"] = m["end_bar"]
    else:
        mods_comp.append(m)

# アンカー整合
anc = (
    anchors["anchors"]
    if isinstance(anchors, dict) and "anchors" in anchors
    else (anchors if isinstance(anchors, list) else [])
)
strong = [
    a
    for a in anc
    if isinstance(a, dict)
    and "bar" in a
    and "beat" in a
    and (
        a.get("kind", "").lower() in {"stress", "accent"}
        or (
            isinstance(a.get("classes"), list)
            and any(c.lower() in {"stress", "accent"} for c in a.get("classes", []))
        )
    )
]
change_pts = [
    (events[i]["bar"], events[i]["beat"])
    for i in range(1, len(events))
    if events[i]["symbol"] != events[i - 1]["symbol"]
]


def nearest_dist(bar, beat):
    """Find minimum distance to any chord change, including adjacent bars"""
    min_dist = None
    anchor_time = bar * bpb + beat
    for cb, cbeat in change_pts:
        change_time = cb * bpb + cbeat
        dist = abs(anchor_time - change_time)
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return min_dist


dists = [d for a in strong if (d := nearest_dist(int(a["bar"]), float(a["beat"]))) is not None]
anchor_ratio = (sum(1 for d in dists if d <= 0.25) / len(dists) * 100) if dists else 0.0


# テンション比率
def has_tension(sym: str):
    return any(
        t in sym
        for t in ("9", "11", "13", "add9", "add11", "add13", "#5", "b5", "#9", "b9", "#11", "b13")
    )


tension_ratio = (
    sum(1 for e in events if isinstance(e["symbol"], str) and has_tension(e["symbol"]))
    / max(1, len(events))
    * 100.0
)

# 出力テーブル
sec_by_bar = {}
for lab, sb, eb in secs:
    for b in range(sb, eb + 1):
        sec_by_bar[b] = lab

enriched = []
for e in events:
    kb = key_by_bar.get(e["bar"], {"key": "", "confidence": 0.0})
    enriched.append(
        {
            "bar": e["bar"],
            "beat": e["beat"],
            "symbol": e["symbol"],
            "dur_beats": e["dur_beats"],
            "function": e["_func"],
            "section": sec_by_bar.get(e["bar"], ""),
            "local_key": kb["key"],
            "key_conf": kb["confidence"],
            "has_tension": 1 if (isinstance(e["symbol"], str) and has_tension(e["symbol"])) else 0,
        }
    )
pd.DataFrame(enriched).to_csv(OUT_CSV, index=False)

# 可視化（規約: matplotlib/単図/色指定なし）
fig1 = plt.figure()
plt.scatter(
    [(c["start_bar"] + c["end_bar"]) / 2 for c in cadences], [c["cadence_score"] for c in cadences]
)
plt.xlabel("Section midpoint (bar)")
plt.ylabel("Cadence score")
plt.title("Cadence by section")
plt.tight_layout()
plt.savefig(OUT_P1)
plt.close(fig1)

fig2 = plt.figure()
if dists:
    plt.hist(dists, bins=20)
    plt.xlabel("Distance to nearest chord change (beats)")
    plt.ylabel("Count")
    plt.title("Anchor ↔ Chord-change")
else:
    plt.text(0.5, 0.5, "No distances", ha="center", va="center")
plt.tight_layout()
plt.savefig(OUT_P2)
plt.close(fig2)

# サマリ & 受入判定
avg_key_conf = float(np.mean([k["confidence"] for k in keys_tl])) if keys_tl else 0.0
cad_ok = sum(1 for c in cadences if c["cadence_score"] >= 0.5)
cad_total = len(cadences)

# エンハーモニック整合性計算
sharp_roots = {"C#", "D#", "E#", "F#", "G#", "A#", "B#"}
flat_roots = {"Db", "Eb", "Fb", "Gb", "Ab", "Bb", "Cb"}
sharp_count = sum(1 for e in events if e.get("root", "") in sharp_roots)
flat_count = sum(1 for e in events if e.get("root", "") in flat_roots)
total_accidental = sharp_count + flat_count
enharmonic_consistency = (
    1.0 - (min(sharp_count, flat_count) / total_accidental) if total_accidental > 0 else 1.0
)

report = {
    "summary": {
        "bars_total": int(bars_total),
        "chord_events": len(events),
        "sections": int(len(secs)),
        "tempo_points": len(tempo_points),
        "tension_ratio_percent": round(tension_ratio, 1),
        "anchor_near_change_ratio_percent": round(anchor_ratio, 1),
        "avg_key_confidence": round(avg_key_conf, 3),
        "enharmonic_consistency": round(enharmonic_consistency, 3),
    },
    "cadences": cadences,
    "modulations": mods_comp,
    "keys_timeline_sample": keys_tl[:12],
    "acceptance_flags": {
        "anchors_ok": anchor_ratio >= 20.0,
        "cadence_ok_rate": f"{cad_ok}/{cad_total}",
        "cadence_ok": (cad_total == 0) or (cad_ok / cad_total >= 0.7),
        "tension_ok": 10.0 <= tension_ratio <= 60.0,
        "key_conf_ok": avg_key_conf >= 0.15,
    },
}
OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

md = []
md.append("# Deep Harmony Audit")
s = report["summary"]
md.append(
    f"- Bars: **{s['bars_total']}**, Chord events: **{s['chord_events']}**, Sections: **{s['sections']}**"
)
md.append(f"- Tempo points: **{s['tempo_points']}**")
md.append(f"- Tension usage: **{s['tension_ratio_percent']}%** (recommended 10–60%)")
md.append(
    f"- Anchor near-change (±0.25 beat): **{s['anchor_near_change_ratio_percent']}%** (≥20% recommended)"
)
md.append(f"- Avg key confidence: **{s['avg_key_confidence']}** (≥0.15 recommended)")
md.append(f"- Enharmonic consistency: **{s['enharmonic_consistency']}** (1.0=perfect)")
md.append("\n## Cadence by Section")
for c in cadences:
    md.append(f"- {c['section']} bars {c['start_bar']}–{c['end_bar']} → score={c['cadence_score']}")
md.append("\n## Modulation (compressed)")
for m in mods_comp:
    md.append(f"- {m['key']}: bars {m['start_bar']}–{m['end_bar']}")
md.append("\n## Files")
md.append("- `deep_harmony_audit.json`, `deep_harmony_audit.md`")
md.append("- `chord_events_enriched.csv`")
md.append("- `cadence_by_section.png`, `anchor_distance_hist.png`")
OUT_MD.write_text("\n".join(md), encoding="utf-8")

print("OK")
print(f"[Download JSON] (sandbox:{OUT_JSON})")
print(f"[Download MD]   (sandbox:{OUT_MD})")
print(f"[Download CSV]  (sandbox:{OUT_CSV})")
print(f"[Plot1]         (sandbox:{OUT_P1})")
print(f"[Plot2]         (sandbox:{OUT_P2})")
