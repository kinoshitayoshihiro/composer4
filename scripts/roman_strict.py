#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roman strict analyzer:
- 副V (V/x) の網羅
- 裏コード (SubV/x = tritone sub) の検出
- music21 が無い環境でも簡易 fallback
入力:  analysis/chordmap_locked.json, sections.json
出力:  analysis/roman_map.json  (barごとに {key, roman, function, target})
"""
from __future__ import annotations
import json, math, argparse
from pathlib import Path

PC = dict(zip("C C# D D# E F F# G G# A A# B".split(), range(12)))
ENH = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

DIATONIC_DEGREES = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
DEGREE_ROOTS_PC = {"I": 0, "ii": 2, "iii": 4, "IV": 5, "V": 7, "vi": 9, "vii°": 11}


def pc_of(name: str) -> int:
    n = name.replace("♭", "b").replace("＃", "#")
    n = ENH.get(n, n)
    return PC.get(n, None)


def parse_chord_symbol(sym: str):
    # 例: "G7", "Cmaj7", "Db7", "A7b9", "Bm7", "E7(#11)"
    import re

    m = re.match(r"^([A-G](?:#|b)?)(.*)$", sym)
    if not m:
        return None, ""
    root, qual = m.group(1), m.group(2)
    return root, qual


def is_dominant_quality(qual: str) -> bool:
    q = qual.lower()
    return ("7" in q and "maj" not in q) or ("9" in q and "maj" not in q)


def best_secondary_target(chord_pc: int, tonic_pc: int) -> tuple:
    """副V/裏コードの推定: 各diatonic degree=target d に対し
    V(d)= root_pc = tonic_pc + DEGREE_ROOTS_PC[d] の完全五度上(= +7)
    - ぴったりV(d)なら 'V/d'
    - tritone差なら 'SubV/d'
    """
    best = None
    for d in DIATONIC_DEGREES[:-1]:  # vii°は除外
        deg_pc = (tonic_pc + DEGREE_ROOTS_PC[d]) % 12
        v_pc = (deg_pc + 7) % 12
        sub_pc = (v_pc + 6) % 12  # tritone
        if chord_pc == v_pc:
            best = ("V/" + d, d)
            break
        if chord_pc == sub_pc:
            best = ("SubV/" + d, d)
    return best if best else ("", "")


def analyze_locked(chordmap_locked, sections):
    # tonic 推定: sections に key_hint があれば優先、無ければ最初のを流用
    tonic_pc = None
    # sections.jsonの構造に対応（{"sections": [...]} または [...] 形式）
    sections_list = sections.get("sections", sections) if isinstance(sections, dict) else sections
    for s in sections_list:
        if s.get("key_hint_root"):
            tonic_pc = pc_of(s["key_hint_root"])
            break
    if tonic_pc is None:
        tonic_pc = 0  # C を仮定

    out = []
    events = (
        chordmap_locked.get("events", chordmap_locked)
        if isinstance(chordmap_locked, dict)
        else chordmap_locked
    )
    for ev in events:
        bar = ev["bar"]
        # symbolフィールドがない場合はroot+qualityから構築
        sym = ev.get("symbol")
        if not sym:
            sym = ev.get("root", "C") + ev.get("quality", "")
        root_name, qual = parse_chord_symbol(sym)
        if root_name is None:
            out.append(
                {"bar": bar, "roman": "?", "function": "?", "target": "", "key_pc": tonic_pc}
            )
            continue
        chord_pc = pc_of(root_name)
        # 簡易: まずはダイアトニック一致
        roman = "?"
        function = "T/S/D"  # 形だけ
        target = ""
        # dominantか?
        if is_dominant_quality(qual):
            label, tgt = best_secondary_target(chord_pc, tonic_pc)
            if label:
                roman = label
                function = "D"
                target = tgt
            else:
                # ダイアトニックV扱い (例外系)
                roman = "V"
                function = "D"
        else:
            # ダイアトニック単純判定
            rel = (chord_pc - tonic_pc) % 12
            inv = {v: k for k, v in DEGREE_ROOTS_PC.items()}
            roman = inv.get(rel, "?")
            function = "T" if roman in {"I", "vi"} else ("S" if roman in {"ii", "IV"} else "D")
        out.append(
            {"bar": bar, "roman": roman, "function": function, "target": target, "key_pc": tonic_pc}
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-chordmap", required=True)
    ap.add_argument("--sections", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    locked = json.loads(Path(args.locked_chordmap).read_text())
    sections = json.loads(Path(args.sections).read_text())
    roman = analyze_locked(locked, sections)
    Path(args.out_json).write_text(json.dumps(roman, ensure_ascii=False, indent=2))
    print(f"✅ Generated: {args.out_json}")
    print(f"   Total bars: {len(roman)}")
    v_count = sum(1 for r in roman if r["roman"].startswith("V/") or r["roman"].startswith("SubV/"))
    print(f"   Secondary dominants (V/x, SubV/x): {v_count}")


if __name__ == "__main__":
    main()
