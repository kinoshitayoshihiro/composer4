"""Onsets-and-Frames (basic_pitch) 互換アダプタ

このモジュールは basic-pitch のバージョン差を吸収して、常に同じ
JSON 出力（notes: [{onset, duration, pitch, velocity, confidence}, ...]）
を生成します。

CLI 例:
  python ops/oaf_adapter.py transcribe --audio /path/to/piano.wav --out piano_onsets_frames.json --model-size tiny
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class OAFNote:
    onset: float
    duration: float
    pitch: int
    velocity: int
    confidence: float


def _extract_notes(model_out: dict[str, Any]) -> list[OAFNote]:
    """モデル出力を正規化して OAFNote リストを返す。

    model_out は dict でも list でも受け取り可能。内部キー名の差分を吸収する。
    """
    notes: list[OAFNote] = []
    # いくつかの実装で使われるキーを順に参照
    candidate_arrays = []
    if isinstance(model_out, dict):
        # よくあるキー
        for k in ("notes", "note_events", "predictions", "preds"):
            v = model_out.get(k)
            if v:
                candidate_arrays.append(v)
        # ある実装では toppenriched arrays が nested にあることもある
        if not candidate_arrays:
            # try to find a list value
            for v in model_out.values():
                if isinstance(v, list):
                    candidate_arrays.append(v)
    elif isinstance(model_out, list):
        candidate_arrays.append(model_out)

    arr = candidate_arrays[0] if candidate_arrays else []

    for n in arr:
        # n may be dict-like or tuple/list
        if isinstance(n, dict):
            onset = float(n.get("start_time", n.get("onset", n.get("onset_time", 0.0))))
            # duration might be provided directly or as end_time
            if "duration" in n:
                dur = float(n.get("duration", 0.0))
            else:
                end_t = n.get("end_time", n.get("offset_end", None))
                if end_t is not None:
                    dur = float(end_t) - onset
                else:
                    dur = float(n.get("length", 0.0))
            midi = int(n.get("pitch", n.get("midi", n.get("note", 60))))
            vel = int(n.get("velocity", n.get("vel", n.get("velocity_estimate", 80))))
            conf = float(n.get("confidence", n.get("conf", n.get("probability", 0.0))))
        elif isinstance(n, (list, tuple)):
            # guess ordering: [onset, dur, midi, vel, conf]
            vals = list(n) + [0, 0, 60, 80, 0.0]
            onset = float(vals[0])
            dur = float(vals[1])
            midi = int(vals[2])
            vel = int(vals[3])
            conf = float(vals[4])
        else:
            # unknown format -> skip
            continue
        notes.append(OAFNote(onset=onset, duration=dur, pitch=midi, velocity=vel, confidence=conf))

    return notes


def transcribe_piano(audio_path: str, model_size: str = "tiny") -> list[OAFNote]:
    """basic-pitch の API 差を吸収してノート配列を返す。

    戻り値は OAFNote のリスト。
    """
    p = pathlib.Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # basic-pitch 0.4.0 の predict API を使用
    try:
        from basic_pitch.inference import predict

        # predict returns: (model_output_dict, midi_data, note_events)
        # note_events: List[Tuple[start_time, end_time, pitch, velocity, pitch_bends]]
        _, _, note_events = predict(str(p))
        
        notes = []
        for event in note_events:
            # event = (start_time, end_time, pitch, velocity, pitch_bends)
            start_time = float(event[0])
            end_time = float(event[1])
            pitch = int(event[2])
            velocity = int(event[3])
            confidence = 1.0  # basic-pitch doesn't provide confidence, use 1.0
            
            notes.append(OAFNote(
                onset=start_time,
                duration=end_time - start_time,
                pitch=pitch,
                velocity=velocity,
                confidence=confidence,
            ))
        
        return notes
    except Exception as exc:
        raise RuntimeError(f"basic_pitch transcribe failed: {exc}")


def save_oaf_outputs(notes: list[OAFNote], out_json: str) -> None:
    """新フォーマットでOaF結果を保存。
    
    出力形式:
      {"notes": [{"onset": 1.2, "duration": 0.5, "pitch": 60, "velocity": 80, "confidence": 0.9}, ...], "count": N}
    """
    data = {"notes": [n.__dict__ for n in notes], "count": len(notes)}
    p = pathlib.Path(out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _cli() -> None:
    ap = argparse.ArgumentParser(
        prog="oaf_adapter", description="OaF (basic-pitch) compatibility adapter"
    )
    sub = ap.add_subparsers(dest="cmd")

    t = sub.add_parser("transcribe", help="Transcribe audio to unified OaF JSON")
    t.add_argument("--audio", required=True, help="input audio (wav) path")
    t.add_argument("--out", required=True, help="output json path")
    t.add_argument(
        "--model-size", default="tiny", help="basic-pitch model size (tiny, small, ...) "
    )

    ns = ap.parse_args()
    if ns.cmd == "transcribe":
        notes = transcribe_piano(ns.audio, model_size=ns.model_size)
        save_oaf_outputs(notes, ns.out)
        print(f"Saved {ns.out} (notes: {len(notes)})")
    else:
        ap.print_help()


if __name__ == "__main__":
    try:
        _cli()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
