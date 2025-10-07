#!/usr/bin/env python3
"""
Suno MIDI Fixer - 破損したkey signatureを除去

Sunoが生成するMIDIファイルは不正なkey signature (16 sharps)を含むことがある。
このツールはkey signatureメッセージをバイナリレベルで除去する。

Usage:
    python scripts/fix_suno_midi.py <input.mid> <output.mid>
"""

import argparse
import sys
from pathlib import Path


def fix_suno_midi(input_path: Path, output_path: Path, verbose: bool = True) -> bool:
    """
    Suno MIDIのkey signatureを除去

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
        print(f"🔧 Fixing MIDI: {input_path.name}")

    try:
        # MIDIファイルをバイナリで読み込み
        with open(input_path, "rb") as f:
            data = bytearray(f.read())

        # Key signatureメタイベント検索
        # Format: FF 59 02 [sf] [mi]
        # sf = number of sharps/flats (-7 to 7)
        # mi = mode (0=major, 1=minor)

        removed_count = 0
        i = 0
        while i < len(data) - 4:
            # Key signatureイベント検出 (FF 59 02)
            if data[i] == 0xFF and data[i + 1] == 0x59 and data[i + 2] == 0x02:
                sf = data[i + 3]  # sharps/flats
                mi = data[i + 4]  # mode

                # 不正なkey signature (16 sharps = 0x10)
                if sf > 7 or sf < -7 or mi > 1:
                    if verbose:
                        print(f"   Found invalid key signature at offset {i}: sf={sf}, mi={mi}")

                    # イベント全体を削除 (delta time含む)
                    # delta timeは可変長なので、遡って探す
                    start_pos = i
                    # delta timeの開始位置を探す (前のイベント終了直後)
                    # 簡易実装: FF 59の直前のバイトから5バイト削除
                    if i > 0:
                        # Delta timeは通常0x00 (イベント連続)または小さい値
                        delta_start = max(0, i - 1)
                        # FF 59 02 sf mi の5バイト + delta time 1バイト = 6バイト削除
                        del data[delta_start : i + 5]
                        removed_count += 1
                        i = delta_start  # 削除した分戻る
                        continue

            i += 1

        if removed_count > 0:
            # 修正後のファイルを保存
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(data)

            if verbose:
                print(f"   ✅ Removed {removed_count} invalid key signature(s)")
                print(f"   📄 Fixed MIDI saved: {output_path.name}")
            return True
        else:
            # 修正不要、コピー
            import shutil

            shutil.copy2(input_path, output_path)
            if verbose:
                print("   ℹ️  No invalid key signatures found, file copied")
            return True

    except Exception as e:
        print(f"❌ Failed to fix MIDI: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix Suno MIDI files with invalid key signatures")
    parser.add_argument("input", type=Path, help="Input MIDI file")
    parser.add_argument("output", type=Path, help="Output MIDI file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    success = fix_suno_midi(args.input, args.output, verbose=not args.quiet)

    if success:
        print("=" * 70)
        print("🎉 MIDI fix complete!")
        print("=" * 70)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
