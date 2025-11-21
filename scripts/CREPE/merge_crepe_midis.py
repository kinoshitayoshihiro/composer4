#!/usr/bin/env python3
"""
CREPE系MIDI統合ツール

複数のCREPE生成MIDIを1つに統合し、トラック名・program変更
"""

import argparse
from pathlib import Path
import pretty_midi


def merge_crepe_midis(
    midi_files: list[tuple[Path, str, int]], output_path: Path, bpm: float = 120.0
) -> None:
    """
    複数MIDI → 統合MIDI

    Args:
        midi_files: [(midi_path, track_name, program), ...]
        output_path: 出力MIDIパス
        bpm: テンポ
    """

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    total_notes = 0

    for midi_path, track_name, program in midi_files:
        if not midi_path.exists():
            print(f"⚠️  {midi_path.name} が見つかりません（スキップ）")
            continue

        src_pm = pretty_midi.PrettyMIDI(str(midi_path))

        for src_inst in src_pm.instruments:
            # 新しいInstrument作成（program/name上書き）
            inst = pretty_midi.Instrument(program=program, name=track_name)
            inst.notes = src_inst.notes.copy()

            # CC/pitch_bendもコピー（あれば）
            inst.control_changes = getattr(src_inst, "control_changes", []).copy()
            inst.pitch_bends = getattr(src_inst, "pitch_bends", []).copy()

            pm.instruments.append(inst)
            total_notes += len(inst.notes)
            print(f"✅ {track_name}: {len(inst.notes)} notes（program {program}）")

    pm.write(str(output_path))
    print()
    print(f"📦 {output_path.name}: {len(pm.instruments)} tracks, {total_notes} notes")


def main():
    parser = argparse.ArgumentParser(description="CREPE系MIDI統合")
    parser.add_argument("--output", type=Path, required=True, help="統合出力MIDIパス")
    parser.add_argument("--bpm", type=float, default=120.0, help="テンポ")
    parser.add_argument("--piano", type=Path, help="Piano Hybrid MIDI")
    parser.add_argument("--strings", type=Path, help="Strings VoiceLeading MIDI")
    parser.add_argument("--guitar", type=Path, help="Guitar Microtiming MIDI")
    parser.add_argument("--melody", type=Path, help="Melody from F0 MIDI")

    args = parser.parse_args()

    # MIDIファイルリスト構築
    midi_files = []

    if args.piano:
        midi_files.append((args.piano, "Piano Hybrid", 0))  # Acoustic Grand Piano
    if args.strings:
        midi_files.append((args.strings, "Strings VoiceLeading", 48))  # String Ensemble 1
    if args.guitar:
        midi_files.append((args.guitar, "Guitar Microtiming", 24))  # Acoustic Guitar (nylon)
    if args.melody:
        midi_files.append((args.melody, "Melody from F0", 80))  # Square Lead

    if not midi_files:
        print("❌ 入力MIDIが指定されていません")
        return

    merge_crepe_midis(midi_files, args.output, args.bpm)


if __name__ == "__main__":
    main()
