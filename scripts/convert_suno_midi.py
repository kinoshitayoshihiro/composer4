#!/usr/bin/env python3
"""
Suno MIDI Converter - music21経由で再生成

Suno MIDIをmusic21で読み込み、クリーンなMIDIとして再出力する。
key signatureエラーを無視してノート情報のみを抽出。

Usage:
    python scripts/convert_suno_midi.py <input.mid> <output.mid>
"""

import argparse
import sys
from pathlib import Path


def convert_suno_midi(input_path: Path, output_path: Path, verbose: bool = True) -> bool:
    """
    Suno MIDIをmusic21経由で変換

    Args:
        input_path: 入力MIDIファイル
        output_path: 出力MIDIファイル
        verbose: 詳細出力

    Returns:
        成功した場合True
    """
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return False

    if verbose:
        print(f"🎵 Converting MIDI: {input_path.name}")

    try:
        # music21でMIDI読み込み (key signatureエラー無視)
        import music21

        if verbose:
            print("   Loading with music21...")

        # 警告を一時的に無効化
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = music21.converter.parse(str(input_path))

        if verbose:
            parts_count = len(score.parts)
            notes_count = len(score.flat.notes)
            print(f"   Loaded: {parts_count} parts, {notes_count} notes")

        # クリーンなMIDIとして再出力
        output_path.parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=str(output_path))

        if verbose:
            print(f"   ✅ Converted MIDI saved: {output_path.name}")

        return True

    except Exception as e:
        print(f"❌ Failed to convert MIDI: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert Suno MIDI via music21")
    parser.add_argument("input", type=Path, help="Input MIDI file")
    parser.add_argument("output", type=Path, help="Output MIDI file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    success = convert_suno_midi(args.input, args.output, verbose=not args.quiet)

    if success:
        print("=" * 70)
        print("🎉 MIDI conversion complete!")
        print("=" * 70)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
