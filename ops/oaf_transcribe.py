#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/oaf_transcribe.py
------------------------------------------------------------
Piano転写: Onsets-and-Frames / piano_transcription_inference があれば使用
無ければ librosa オンセット + pyin で簡易ノート推定（最小限の代替）
- 入力: --audio, --bars, --tempo-bpm --ppq
- 出力: --out piano_oaf.json （notes + sustain segments + bar/beat 付与）
------------------------------------------------------------
"""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")


def seconds_to_ticks(sec, tempo_bpm, ppq):
    # 1拍=60/tempo, 1小節(4/4)=4拍
    beats = sec * (tempo_bpm / 60.0)
    return int(round(beats * ppq))


def attach_bar_beat(events, bars_df, tempo_bpm, ppq):
    # 各イベントに bar, beat を付与（start_beatsから）
    for e in events:
        sb = e.get("start_sec", 0.0)
        beats = sb * (tempo_bpm / 60.0)
        bar = int(beats // 4)
        beat_in_bar = beats - bar * 4.0
        e["bar"] = bar
        e["beat"] = round(beat_in_bar, 6)
        e["start_beats"] = round(beats, 6)
        e["end_beats"] = round(e.get("end_sec", sb) * (tempo_bpm / 60.0), 6)


def merge_adjacent_notes(notes, gap_threshold_sec=0.02):
    """同じMIDI番号で近接したノートを結合（gap < gap_threshold_sec）"""
    if not notes:
        return notes

    # MIDI番号でソート
    notes_sorted = sorted(notes, key=lambda n: (n["midi"], n["start_sec"]))
    merged = []
    current = None

    for note in notes_sorted:
        if current is None:
            current = note.copy()
        elif (
            note["midi"] == current["midi"]
            and note["start_sec"] - current["end_sec"] < gap_threshold_sec
        ):
            # 結合（終端を延長）
            current["end_sec"] = note["end_sec"]
            current["velocity"] = max(current["velocity"], note["velocity"])
            current["confidence"] = max(current["confidence"], note["confidence"])
        else:
            merged.append(current)
            current = note.copy()

    if current is not None:
        merged.append(current)

    # 開始時刻でソート
    merged.sort(key=lambda n: n["start_sec"])
    return merged


def transcribe_librosa(y, sr):
    """librosa fallback: onset detection + pyin pitch estimation"""
    import librosa

    print("⚠️  Using librosa fallback for piano transcription")
    # オンセット検出
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=o_env, sr=sr, backtrack=True, units="frames"
    )
    times = librosa.frames_to_time(onset_frames, sr=sr)

    # ピッチ推定（pyin）
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("A0"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    notes = []
    for i, t0 in enumerate(times):
        t1 = times[i + 1] if i + 1 < len(times) else (t0 + 0.4)  # 仮の長さ
        sel = (frame_times >= t0) & (frame_times < t1) & np.isfinite(f0)
        if not np.any(sel):
            midi = 60
        else:
            midi = int(round(69 + 12 * np.log2(np.median(f0[sel]) / 440.0)))
        vel = 90
        notes.append(
            dict(
                start_sec=float(t0),
                end_sec=float(t1),
                midi=int(np.clip(midi, 21, 108)),
                velocity=int(vel),
                confidence=0.5,
            )
        )
    pedals = []  # 簡易: ここでは空（必要ならRMSから推定可）
    return notes, pedals


def transcribe_oaf(y, sr, min_conf=0.2, min_note_ms=50, merge_gap_ms=20):
    """
    piano_transcription_inference (Onsets-and-Frames) がある場合に使用
    pip install piano-transcription-inference

    Args:
        min_conf: 信頼度閾値（これ未満のノートを除外）
        min_note_ms: 最小ノート長（ms、これ未満のノートを除外）
        merge_gap_ms: ノート結合窓（ms、同音で間隔がこれ未満なら結合）
    """
    try:
        from piano_transcription_inference import PianoTranscription
        import tempfile

        print(
            f"🎹 Using Onsets-and-Frames (min_conf={min_conf}, min_note_ms={min_note_ms}, merge_gap_ms={merge_gap_ms})"
        )
        # ライブラリは独自ローダを推奨するため一旦WAVに出す必要がある場合あり
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            transcriptor = PianoTranscription(device="cpu")
            transcribed_dict = transcriptor.transcribe(tmp.name, None)
            # 戻り値はバージョンにより異なる可能性があるため柔軟に対応
            if isinstance(transcribed_dict, dict):
                est_notes = transcribed_dict.get("est_note_events", [])
                est_pedals = transcribed_dict.get("est_pedal_events", [])
            else:
                # タプル形式の場合
                (est_piano_roll, est_onsets, est_frames), est_notes, est_pedals = transcribed_dict

        notes = []
        for n in est_notes:
            conf = float(n.get("confidence", 1.0))
            if conf < min_conf:
                continue  # 信頼度が低いノートを除外

            start = float(n["onset_time"])
            end = float(n["offset_time"])
            duration_ms = (end - start) * 1000.0

            if duration_ms < min_note_ms:
                continue  # 短すぎるノートを除外

            notes.append(
                dict(
                    start_sec=start,
                    end_sec=end,
                    midi=int(n["pitch"]),
                    velocity=int(n.get("velocity", 90)),
                    confidence=conf,
                )
            )

        # ノート結合処理（同じMIDI番号で近接したノートを結合）
        if merge_gap_ms > 0:
            notes = merge_adjacent_notes(notes, merge_gap_ms / 1000.0)

        pedals = [
            {"start_sec": float(p["onset_time"]), "end_sec": float(p["offset_time"])}
            for p in est_pedals
        ]
        return notes, pedals
    except Exception as e:
        print(f"⚠️  OaF not available ({e}), falling back to librosa")
        return transcribe_librosa(y, sr)


def main():
    ap = argparse.ArgumentParser(description="Piano transcription (OaF or librosa fallback)")
    ap.add_argument("--audio", required=True, help="Input audio file (WAV)")
    ap.add_argument("--bars", required=True, help="bars.parquet path")
    ap.add_argument("--out", required=True, help="Output JSON file")
    ap.add_argument("--tempo-bpm", type=float, default=120.0, help="Tempo in BPM")
    ap.add_argument("--ppq", type=int, default=480, help="Pulses per quarter note")
    ap.add_argument("--min-conf", type=float, default=0.2, help="Minimum confidence threshold")
    ap.add_argument("--min-note-ms", type=int, default=50, help="Minimum note duration (ms)")
    ap.add_argument("--merge-gap-ms", type=int, default=20, help="Merge gap threshold (ms)")
    args = ap.parse_args()

    print(f"🎹 Piano Transcription: {args.audio}")
    y, sr = sf.read(args.audio, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    bars = pd.read_parquet(args.bars)
    notes, pedals = transcribe_oaf(y, sr, args.min_conf, args.min_note_ms, args.merge_gap_ms)

    # 付帯情報を付与
    attach_bar_beat(notes, bars, args.tempo_bpm, args.ppq)

    obj = {
        "meta": {
            "backend": (
                "oaf_fallback"
                if len(pedals) == 0 and all(n.get("confidence", 0.5) <= 0.6 for n in notes)
                else "oaf"
            ),
            "min_conf": args.min_conf,
            "min_note_ms": args.min_note_ms,
            "merge_gap_ms": args.merge_gap_ms,
        },
        "notes": notes,
        "pedal": pedals,
        "ppq": args.ppq,
        "tempo_bpm": args.tempo_bpm,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved OaF-like transcript → {args.out}  (notes={len(notes)})")


if __name__ == "__main__":
    main()
