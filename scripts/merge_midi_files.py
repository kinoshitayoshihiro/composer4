#!/usr/bin/env python3
"""
merge_midi_files.py - MIDI統合ツール

制御MIDI + ノートMIDI → 1つのMIDIファイルにマージ

Usage:
    python3 scripts/merge_midi_files.py \
      --note-midi melody.mid \
      --control-midi violin_controls.mid \
      --output merged.mid
"""

import argparse
from pathlib import Path

try:
    import mido
except ImportError:
    print("❌ Error: mido not installed.")
    print("   Install: pip install mido")
    raise


def merge_midi_files(note_midi_path: Path, control_midi_path: Path, output_path: Path):
    """制御MIDI + ノートMIDI → マージ"""

    # MIDI読み込み
    note_mid = mido.MidiFile(str(note_midi_path))
    control_mid = mido.MidiFile(str(control_midi_path))

    print(f"📖 Note MIDI: {note_midi_path}")
    print(f"   Tracks: {len(note_mid.tracks)}")
    print(f"   Ticks per beat: {note_mid.ticks_per_beat}")

    print(f"🎹 Control MIDI: {control_midi_path}")
    print(f"   Tracks: {len(control_mid.tracks)}")
    print(f"   Ticks per beat: {control_mid.ticks_per_beat}")

    # Ticks per beat統一（note_midiに合わせる）
    if note_mid.ticks_per_beat != control_mid.ticks_per_beat:
        print(
            f"⚠️  Warning: Ticks per beat mismatch ({note_mid.ticks_per_beat} vs {control_mid.ticks_per_beat})"
        )
        print(f"   Using note_midi's ticks_per_beat: {note_mid.ticks_per_beat}")

    # 新しいMIDIファイル作成
    merged_mid = mido.MidiFile(ticks_per_beat=note_mid.ticks_per_beat)

    # 制御MIDI Track追加（Track 0）
    control_track = mido.MidiTrack()
    for msg in control_mid.tracks[0]:
        control_track.append(msg)
    merged_mid.tracks.append(control_track)

    # ノートMIDI Tracks追加（Track 1以降）
    for track in note_mid.tracks:
        note_track = mido.MidiTrack()
        for msg in track:
            note_track.append(msg)
        merged_mid.tracks.append(note_track)

    # 出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_mid.save(str(output_path))

    print(f"✅ Merged MIDI saved: {output_path}")
    print(f"   Total tracks: {len(merged_mid.tracks)}")
    print(f"   Ticks per beat: {merged_mid.ticks_per_beat}")


def main():
    parser = argparse.ArgumentParser(description="MIDI統合ツール")
    parser.add_argument("--note-midi", type=Path, required=True, help="Note MIDI file path")
    parser.add_argument("--control-midi", type=Path, required=True, help="Control MIDI file path")
    parser.add_argument("--output", type=Path, required=True, help="Output merged MIDI file path")

    args = parser.parse_args()

    merge_midi_files(args.note_midi, args.control_midi, args.output)


if __name__ == "__main__":
    main()
