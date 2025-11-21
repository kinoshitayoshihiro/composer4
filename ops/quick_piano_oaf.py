#!/usr/bin/env python3
"""
Quick Piano OaF JSON Generator
piano.wavからbasic-pitchで簡易的にJSONを生成
"""
import json
import sys
from pathlib import Path

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    import pretty_midi

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def main():
    if len(sys.argv) < 3:
        print("Usage: python ops/quick_piano_oaf.py <piano.wav> <output.json>")
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    out_json = Path(sys.argv[2])

    if not AVAILABLE:
        print("⚠️  basic-pitch not available, creating dummy JSON")
        dummy = {"status": "not_available", "notes": []}
        with open(out_json, "w") as f:
            json.dump(dummy, f, indent=2)
        return

    if not wav_path.exists():
        print(f"⚠️  WAV not found: {wav_path}")
        dummy = {"status": "file_not_found", "notes": []}
        with open(out_json, "w") as f:
            json.dump(dummy, f, indent=2)
        return

    print(f"🎹 Transcribing: {wav_path}")

    # basic-pitch実行
    model_output, midi_data, note_events = predict(
        str(wav_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=58.0,  # ms
        minimum_frequency=27.5,
        maximum_frequency=4186.0,
        melodia_trick=False,
        debug_file=None,
    )

    # ノート情報抽出
    notes = []
    if midi_data and len(midi_data.instruments) > 0:
        for note in midi_data.instruments[0].notes:
            notes.append(
                {
                    "start": float(note.start),
                    "end": float(note.end),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                }
            )

    result = {"status": "success", "notes": notes, "total_notes": len(notes)}

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Saved {len(notes)} notes to {out_json}")


if __name__ == "__main__":
    main()
