#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arrangement_orchestrator.py
Plans-only Orchestrator CLI
 - 可変テンポ (tempo_map.json) / 固定テンポ (--tempo-bpm) に対応
 - 入力は *_plan.json(複数)。スキーマは:
     { "tracks": [ { "name": "...", "instrument": "...", "events": [...] }, ... ] }
   もしくは
     { "plan": { "tracks": [...] } }
 - CLIは2系統をサポート(後方互換):
     A) 楽器別: --bass --guitar --piano --strings --drums
     B) 汎用   : --plan <path>(複数回指定)

出力: arrangement_plan.json
    {
      "meta": { "ppq": 480, "tempo_map_path": "...", "tempo_bpm": 120.0 },
      "tracks": [ ...merged tracks... ]
    }
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_metadata(meta: Any) -> Dict[str, Any]:
    if isinstance(meta, dict):
        return {k: v for k, v in meta.items()}
    return {}


def _extract_tracks(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept:
      { "tracks": [...] }                                      # Legacy format
      { "plan": { "tracks": [...] } }                         # Legacy nested format
      { "instrument": "...", "events": [...] }                # V2 format (instrument at top)
      { "metadata": {"instrument": "..."}, "events": [...] }  # V2 format (instrument in metadata)
    """
    if isinstance(obj, dict):
        # V2 format: single instrument plan
        if "instrument" in obj and "events" in obj:
            track = dict(obj)
            track.setdefault("metadata", {})
            if isinstance(track["metadata"], dict):
                track["metadata"].setdefault("instrument", track.get("instrument"))
            return [track]
        # V2 format with metadata wrapper
        if "metadata" in obj and "events" in obj:
            metadata = _copy_metadata(obj.get("metadata"))
            instrument = obj.get("instrument") or metadata.get("instrument")
            name = metadata.get("role") or metadata.get("name") or obj.get("name")
            track = {
                "instrument": instrument or name or "track",
                "name": name or instrument or "track",
                "events": obj["events"],
                "metadata": metadata,
            }
            return [track]
        # Legacy formats
        if "tracks" in obj and isinstance(obj["tracks"], list):
            return obj["tracks"]
        if "plan" in obj and isinstance(obj["plan"], dict):
            plan = obj["plan"]
            if "tracks" in plan and isinstance(plan["tracks"], list):
                return plan["tracks"]
    return []


def _infer_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith("_plan.json"):
        base = base[: -len("_plan.json")]
    return base


def _ensure_track_fields(track: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
    # name
    if not track.get("name"):
        track["name"] = track.get("instrument") or fallback_name
    # instrument
    if not track.get("instrument"):
        track["instrument"] = track.get("name", fallback_name)
    # events
    track.setdefault("events", [])
    metadata = track.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if "instrument" not in metadata and track.get("instrument"):
        metadata["instrument"] = track["instrument"]
    if "role" not in metadata and track.get("name"):
        metadata["role"] = track["name"]
    track["metadata"] = metadata
    return track


def load_plan_file(path: str) -> List[Dict[str, Any]]:
    obj = _read_json(path)
    tracks = _extract_tracks(obj)
    fallback = _infer_name_from_path(path)
    out: List[Dict[str, Any]] = []
    for t in tracks:
        if not isinstance(t, dict):
            continue
        t = _ensure_track_fields(dict(t), fallback)
        out.append(t)
    return out


def merge_plans(
    plan_paths: List[str],
    ppq: int,
    tempo_map_path: Optional[str] = None,
    tempo_bpm: Optional[float] = None,
) -> Dict[str, Any]:
    merged_tracks: List[Dict[str, Any]] = []
    for p in plan_paths:
        try:
            tracks = load_plan_file(p)
        except Exception as e:
            raise RuntimeError(f"Failed to load plan: {p}: {e}") from e

        # 空トラックは落とす(event=0)
        for t in tracks:
            ev_cnt = len(t.get("events", []))
            if ev_cnt > 0:
                merged_tracks.append(t)

    if not merged_tracks:
        raise RuntimeError("No events in any provided plans.")

    # 楽器の並びを軽く整える(存在したものだけ)
    order = ["drums", "bass", "guitar", "piano", "strings", "pad", "synth", "vocals"]

    def key_fn(tr: Dict[str, Any]) -> int:
        inst = str(tr.get("instrument", "")).lower()
        for i, k in enumerate(order):
            if k in inst:
                return i
        return len(order) + 1

    merged_tracks.sort(key=key_fn)

    meta: Dict[str, Any] = {"ppq": int(ppq)}
    if tempo_map_path:
        meta["tempo_map_path"] = tempo_map_path
    if tempo_bpm is not None:
        meta["tempo_bpm"] = float(tempo_bpm)

    reference_layers = _collect_reference_layers(merged_tracks)
    if reference_layers:
        meta["reference_layers"] = reference_layers

    return {"meta": meta, "tracks": merged_tracks}


def _collect_reference_layers(tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_instrument: Dict[str, Dict[str, Any]] = {}
    for track in tracks:
        metadata = track.get("metadata")
        if not isinstance(metadata, dict):
            continue
        ref = metadata.get("reference_layers")
        if not isinstance(ref, dict) or not ref:
            continue
        names = {
            str(
                metadata.get("instrument")
                or track.get("instrument")
                or track.get("name")
                or "track"
            ).lower()
        }
        if metadata.get("role"):
            names.add(str(metadata["role"]).lower())
        if track.get("name"):
            names.add(str(track["name"]).lower())
        for key in filter(None, names):
            by_instrument.setdefault(key, ref)

    if not by_instrument:
        return {}

    global_summary: Dict[str, Dict[str, Any]] = {}
    for summary in by_instrument.values():
        for layer_name, payload in summary.items():
            if not isinstance(payload, dict):
                continue
            entry = global_summary.setdefault(layer_name, {"frames": 0, "notes": 0, "paths": []})
            frames = payload.get("frames")
            notes = payload.get("notes")
            try:
                if frames is not None:
                    entry["frames"] += int(frames)
            except (TypeError, ValueError):
                pass
            try:
                if notes is not None:
                    entry["notes"] += int(notes)
            except (TypeError, ValueError):
                pass
            path = payload.get("path")
            if path and path not in entry["paths"]:
                entry["paths"].append(path)

    return {"by_instrument": by_instrument, "global": global_summary}


def _format_reference_layers(meta: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    global_summary = meta.get("global", {}) if isinstance(meta, dict) else {}
    for layer_name, payload in global_summary.items():
        if not isinstance(payload, dict):
            continue
        details: List[str] = []
        frames = payload.get("frames")
        notes = payload.get("notes")
        if isinstance(frames, int) and frames > 0:
            details.append(f"frames={frames}")
        if isinstance(notes, int) and notes > 0:
            details.append(f"notes={notes}")
        paths = payload.get("paths")
        if isinstance(paths, list) and paths:
            details.append(f"paths={len(paths)}")
        detail_str = ", ".join(details) if details else "no metrics"
        lines.append(f"      · {layer_name}: {detail_str}")
    return lines


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge *_plan.json files into an arrangement_plan.json")
    # 出力
    p.add_argument("--out", required=True, help="Path to output arrangement_plan.json")
    # テンポ系
    p.add_argument(
        "--tempo-map",
        dest="tempo_map",
        default=None,
        help="Path to tempo_map.json (variable tempo)",
    )
    p.add_argument(
        "--tempo-bpm", dest="tempo_bpm", type=float, default=None, help="Fallback fixed tempo BPM"
    )
    p.add_argument("--ppq", type=int, default=480, help="PPQ resolution (default: 480)")

    # 2系統の指定方法に対応
    # A) 楽器別
    p.add_argument("--bass", default=None, help="bass_plan.json")
    p.add_argument("--guitar", default=None, help="guitar_plan.json")
    p.add_argument("--piano", default=None, help="piano_plan.json")
    p.add_argument("--strings", default=None, help="strings_plan.json")
    p.add_argument("--drums", default=None, help="drums_plan.json")

    # B) 汎用
    p.add_argument("--plan", action="append", default=[], help="*_plan.json (repeatable)")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    # 収集: --plan 複数 or 楽器別フラグ
    plan_paths: List[str] = list(args.plan or [])

    for k in ["bass", "guitar", "piano", "strings", "drums"]:
        v = getattr(args, k, None)
        if v:
            plan_paths.append(v)

    # 重複排除 & 存在チェック
    uniq: List[str] = []
    seen = set()
    for p in plan_paths:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    plan_paths = uniq

    if not plan_paths:
        print(
            "ERROR: No plan files provided. Use --plan or instrument-specific flags.",
            file=sys.stderr,
        )
        return 2

    for p in plan_paths:
        if not os.path.isfile(p):
            print(f"ERROR: Plan file not found: {p}", file=sys.stderr)
            return 2

    # 統合
    try:
        arrangement = merge_plans(
            plan_paths=plan_paths,
            ppq=args.ppq,
            tempo_map_path=args.tempo_map,
            tempo_bpm=args.tempo_bpm,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # メタの可視化ログ
    meta = arrangement.get("meta", {})
    print("📋 Merge summary")
    print(f"   PPQ          : {meta.get('ppq')}")
    if meta.get("tempo_map_path"):
        print(f"   Tempo Map    : {meta['tempo_map_path']}")
    if meta.get("tempo_bpm") is not None:
        print(f"   Fixed BPM    : {meta['tempo_bpm']}")
    if meta.get("reference_layers"):
        print("   Reference    : detected")
        for line in _format_reference_layers(meta["reference_layers"]):
            print(line)
    print(f"   Tracks       : {len(arrangement.get('tracks', []))}")

    # 書き出し
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(arrangement, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote arrangement plan → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
