#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chordmap_lock.py
----------------
manual_chordmap.json を自動chordmap.jsonにマージし、chordmap_locked.jsonを生成。
QA機能付き（時間重複・未知quality・グリッド逸脱検出）。

Usage:
    python chordmap_lock.py \
        --base analysis/chordmap.json \
        --overrides analysis/manual_chordmap.json \
        --sections analysis/sections.json \
        --out-json analysis/chordmap_locked.json \
        --out-qa analysis/chordmap_qa.csv
"""
from __future__ import annotations
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Set


def load_json(path: Path) -> Dict[str, Any]:
    """JSON読み込み"""
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_quality(quality: str) -> str:
    """
    quality を正規化（minor の二重 m 禁止、統一表記）

    Examples:
        "m" -> "m"
        "minor" -> "m"
        "m7" -> "m7"
        "maj7" -> "maj7"
        "major" -> "" (major は省略)
    """
    q = (quality or "").strip().lower()

    # Empty/None
    if not q or q in ("major", "maj"):
        return ""

    # Minor variants
    if q in ("minor", "min"):
        return "m"

    # その他はそのまま（m7, maj7, 7, sus4, add9 など）
    return q


def build_symbol(root: str, quality: str, tensions: List[str] = None, bass: str = None) -> str:
    """
    root + quality + tensions + bass から正しい symbol を生成

    CRITICAL: root に "m" が含まれていても quality は独立して処理

    Examples:
        ("E", "m") -> "Em"
        ("Em", "m") -> "Em" (NOT "Emm")
        ("A", "m7") -> "Am7"
        ("Am", "m7") -> "Am7" (NOT "Amm7")
        ("G", "") -> "G"
        ("D", "7") -> "D7"
        ("G", "", ["add9"]) -> "Gadd9"
        ("C", "7", ["b9", "#5"]) -> "C7(b9,#5)"
        ("D", "m7", [], "F") -> "Dm7/F"
    """
    r = (root or "").strip()
    q = normalize_quality(quality)

    # root が既に minor 記号を含んでいる場合は除去
    # 例: "Em" + "m" -> "Em" (not "Emm")
    if r.endswith("m") and q.startswith("m"):
        # root の末尾 "m" を除去
        r = r[:-1]

    base = r if not q else f"{r}{q}"

    # tensions の追加 (add9, #5, b9 など)
    if tensions:
        tlist = [str(t).strip() for t in tensions if str(t).strip()]
        if tlist:
            # add9 を後ろに、その他はアルファベット順
            tlist = sorted(set(tlist), key=lambda x: ("add" not in x, x))
            base += "(" + ",".join(tlist) + ")"

    # bass (slash chord: /F, /G など)
    if bass:
        b = str(bass).strip()
        # bass が root と異なる場合のみ追加
        if b and b != r:
            base += f"/{b}"

    return base


def normalize_chord_events(cm_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    chordmap正規化:
    1. time_ql統一
    2. bar計算
    3. symbol生成（root + quality）
    4. quality正規化（minor二重m禁止）
    """
    evs = cm_obj.get("events") or cm_obj.get("chords") or cm_obj
    if not isinstance(evs, list):
        raise ValueError("Unsupported chordmap format")

    out = []
    for e in evs:
        d = dict(e)

        # time_ql統一
        if "time" in d:
            tql = float(d["time"])
        elif "time_ql" in d:
            tql = float(d["time_ql"])
        elif "bar" in d:
            tql = float(d["bar"]) * 4.0
        else:
            tql = 0.0

        d["time_ql"] = tql

        # bar計算（floor division）
        d["bar"] = int(tql // 4.0)

        # quality 正規化
        if "quality" in d:
            d["quality"] = normalize_quality(d["quality"])

        # tensions と bass を取得
        tensions = d.get("tensions") or d.get("extensions") or []
        bass = d.get("bass") or d.get("inversion")

        # symbol 生成（root + quality + tensions + bass から正しく構築）
        if "root" in d:
            d["symbol"] = build_symbol(
                d["root"], d.get("quality", ""), tensions=tensions if tensions else None, bass=bass
            )

        out.append(d)

    out.sort(key=lambda x: x["time_ql"])
    return out


def merge_chordmaps(base: List[Dict], overrides: List[Dict]) -> List[Dict]:
    """
    overridesを優先してbaseにマージ。
    同一time_qlの場合、overridesで上書き。
    """
    # overridesの時刻セット
    override_times = {e["time_ql"] for e in overrides}

    # baseから非重複分を抽出
    merged = [e for e in base if e["time_ql"] not in override_times]

    # overridesを追加
    merged.extend(overrides)

    # 時刻順ソート
    merged.sort(key=lambda x: x["time_ql"])

    return merged


def qa_check(events: List[Dict], sections: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    QAチェック:
    1. 時間重複（同一time_qlが複数）
    2. 未知quality（空文字・None・未定義記法）
    3. グリッド逸脱（time_qlが16分音符グリッドから外れる）
    """
    issues = []

    # 既知quality定義（拡張可能）
    known_qualities = {
        "",
        "major",
        "minor",
        "maj7",
        "m7",
        "7",
        "dim",
        "dim7",
        "aug",
        "sus2",
        "sus4",
        "6",
        "m6",
        "9",
        "m9",
        "maj9",
        "add9",
        "7sus4",
        "m7b5",
        "7b9",
        "7#9",
        "7b13",
        "7alt",
        "maj7#5",
        "m(maj7)",
        "6/9",
        "13",
        "m11",
        "11",
    }

    # 1) 時間重複チェック
    time_counts = {}
    for e in events:
        t = e["time_ql"]
        time_counts[t] = time_counts.get(t, 0) + 1

    for t, count in time_counts.items():
        if count > 1:
            issues.append(
                {
                    "type": "time_duplicate",
                    "time_ql": t,
                    "bar": int(t // 4.0),
                    "count": count,
                    "severity": "ERROR",
                    "message": f"Duplicate events at time_ql={t:.2f} (count={count})",
                }
            )

    # 2) 未知quality・3) グリッド逸脱
    for e in events:
        quality = (e.get("quality") or "").strip().lower()

        # 未知quality
        if quality not in known_qualities:
            issues.append(
                {
                    "type": "unknown_quality",
                    "time_ql": e["time_ql"],
                    "bar": e["bar"],
                    "root": e.get("root", ""),
                    "quality": quality,
                    "severity": "WARNING",
                    "message": f"Unknown quality '{quality}' at bar {e['bar']}",
                }
            )

        # グリッド逸脱（16分音符 = 0.25 qL）
        grid = 0.25
        deviation = abs(e["time_ql"] % grid)
        if deviation > 1e-6 and deviation < (grid - 1e-6):
            issues.append(
                {
                    "type": "grid_deviation",
                    "time_ql": e["time_ql"],
                    "bar": e["bar"],
                    "deviation": deviation,
                    "severity": "WARNING",
                    "message": f"Off-grid event at time_ql={e['time_ql']:.4f} (deviation={deviation:.4f})",
                }
            )

    return issues


def write_qa_csv(issues: List[Dict], out_path: Path):
    """QA結果をCSV出力"""
    if not issues:
        # 問題なし
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["status", "message"])
            w.writerow(["OK", "No issues found"])
        return

    # フィールド名統一
    fieldnames = ["type", "time_ql", "bar", "severity", "message"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(issues)


def main():
    parser = argparse.ArgumentParser(description="Chordmap LOCK generator with QA")
    parser.add_argument("--base", type=Path, required=True, help="Base chordmap.json (auto)")
    parser.add_argument(
        "--overrides", type=Path, required=True, help="Manual chordmap.json (human-edited)"
    )
    parser.add_argument("--sections", type=Path, required=True, help="sections.json (for metadata)")
    parser.add_argument("--out-json", type=Path, required=True, help="Output chordmap_locked.json")
    parser.add_argument("--out-qa", type=Path, required=True, help="Output QA report CSV")

    args = parser.parse_args()

    # 入力読み込み
    base_events = normalize_chord_events(load_json(args.base))
    override_events = normalize_chord_events(load_json(args.overrides))
    sections = load_json(args.sections)

    # マージ（overrides優先）
    locked_events = merge_chordmaps(base_events, override_events)

    print(f"✅ Merged chordmap:")
    print(f"   Base: {len(base_events)} events")
    print(f"   Overrides: {len(override_events)} events")
    print(f"   Locked: {len(locked_events)} events")

    # QAチェック
    issues = qa_check(locked_events, sections)

    # QA結果サマリ
    errors = [i for i in issues if i["severity"] == "ERROR"]
    warnings = [i for i in issues if i["severity"] == "WARNING"]

    print(f"\n📊 QA Report:")
    print(f"   Errors: {len(errors)}")
    print(f"   Warnings: {len(warnings)}")

    if errors:
        print("\n❌ CRITICAL ERRORS:")
        for e in errors[:5]:  # 最初5件表示
            print(f"   {e['type']}: {e['message']}")

    # 出力
    locked_obj = {
        "meta": {
            "source": "chordmap_lock.py",
            "base_events": len(base_events),
            "override_events": len(override_events),
            "total_events": len(locked_events),
            "qa_errors": len(errors),
            "qa_warnings": len(warnings),
        },
        "events": locked_events,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(locked_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    write_qa_csv(issues, args.out_qa)

    print(f"\n✅ Output:")
    print(f"   {args.out_json}")
    print(f"   {args.out_qa}")

    # エラーがある場合は警告
    if errors:
        print("\n⚠️  WARNING: Critical errors detected. Review QA report before proceeding.")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
