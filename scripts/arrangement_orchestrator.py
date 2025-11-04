#!/usr/bin/env python3
"""
arrangement_orchestrator.py
---------------------------
複数の*_plan.jsonを結合して arrangement_plan.json を生成。

Usage:
    python3 scripts/arrangement_orchestrator.py \
      --bass song_packages/suno_project/song_001/bass_plan.json \
      --guitar song_packages/suno_project/song_001/guitar_plan.json \
      --piano song_packages/suno_project/song_001/piano_plan.json \
      --tempo-bpm 75 \
      --out song_packages/suno_project/song_001/arrangement_plan.json
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def merge_plans(input_plans: Dict[str, Path], tempo_bpm: float, out_path: Path, ppq: int = 480):
    """
    複数のPlan JSONを統合

    Args:
        input_plans: {'bass': Path, 'guitar': Path, ...}
        tempo_bpm: テンポ（arrangement_plan全体に設定）
        out_path: 出力パス（arrangement_plan.json）
        ppq: PPQ（デフォルト480）
    """
    merged = {"ppq": ppq, "tempo_bpm": tempo_bpm, "meta": {}, "tracks": []}

    # ロール別のデフォルトchannel/program割り当て
    ROLE_DEFAULTS = {
        "bass": {"channel": 1, "program": 33},  # Acoustic Bass
        "guitar": {"channel": 2, "program": 25},  # Acoustic Guitar (steel)
        "piano": {"channel": 3, "program": 0},  # Acoustic Grand Piano
        "strings": {"channel": 4, "program": 48},  # String Ensemble 1
        "drums": {"channel": 9, "program": 0},  # Drums (channel 9固定)
    }

    # 最初のPlanから total_bars 取得
    total_bars = None

    for role, p in input_plans.items():
        if not p.exists():
            print(f"⚠️  Plan not found: {p} (skipping)")
            continue

        data = json.loads(p.read_text(encoding="utf-8"))

        # total_bars取得（最初の有効値を使用）
        if total_bars is None and "meta" in data and "total_bars" in data["meta"]:
            total_bars = data["meta"]["total_bars"]

        # metadata除外、track配列のみ抽出
        if "tracks" in data:
            for tr in data["tracks"]:
                # roleフィールド補完（未設定時はmeta.roleまたは引数roleを使用）
                if "role" not in tr or tr["role"] is None:
                    tr["role"] = data.get("meta", {}).get(
                        "role", data.get("metadata", {}).get("role", role)
                    )

                # channel/program自動割り当て（未設定の場合のみ）
                if "channel" not in tr or tr["channel"] is None:
                    tr_role = tr.get("role", role).lower()
                    defaults = ROLE_DEFAULTS.get(tr_role, {"channel": 0, "program": 0})
                    tr["channel"] = defaults["channel"]
                if "program" not in tr or tr["program"] is None:
                    tr_role = tr.get("role", role).lower()
                    defaults = ROLE_DEFAULTS.get(tr_role, {"channel": 0, "program": 0})
                    tr["program"] = defaults["program"]
                merged["tracks"].append(tr)
                print(
                    f"✅ Merged {role}: {tr.get('name', role)} (ch={tr['channel']}, prog={tr['program']}, {len(tr.get('events', []))} events)"
                )
        elif "plan" in data:
            # suno_arranger.py形式（旧互換）
            merged["tracks"].append(
                {
                    "name": data.get("metadata", {}).get("role", role),
                    "role": data.get("metadata", {}).get("role", role),
                    "channel": 0,  # 要設定
                    "program": 0,  # 要設定
                    "events": data["plan"],
                }
            )

    # total_bars を meta に設定
    if total_bars is not None:
        merged["meta"]["total_bars"] = total_bars

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"✅ Saved: {out_path} ({len(merged['tracks'])} tracks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="複数Plan結合")
    ap.add_argument("--bass", type=Path, default=None)
    ap.add_argument("--guitar", type=Path, default=None)
    ap.add_argument("--piano", type=Path, default=None)
    ap.add_argument("--strings", type=Path, default=None)
    ap.add_argument("--drums", type=Path, default=None)
    ap.add_argument("--tempo-bpm", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ppq", type=int, default=480)
    args = ap.parse_args()

    plans = {}
    if args.bass:
        plans["bass"] = args.bass
    if args.guitar:
        plans["guitar"] = args.guitar
    if args.piano:
        plans["piano"] = args.piano
    if args.strings:
        plans["strings"] = args.strings
    if args.drums:
        plans["drums"] = args.drums

    merge_plans(plans, args.tempo_bpm, args.out, args.ppq)
