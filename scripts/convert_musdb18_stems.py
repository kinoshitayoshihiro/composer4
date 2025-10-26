#!/usr/bin/env python3
"""
MUSDB18 stems (.mp4) → WAV 一括変換

MUSDB18 は "Native Instruments stems" 形式の .mp4（1ファイルに5つのステレオ音源）。
stempeg でデコードして WAV に分離。

Usage:
    python scripts/convert_musdb18_stems.py \\
        --musdb data/musdb18 \\
        --out data/musdb18_wavs \\
        --mono

Output:
    data/musdb18_wavs/Young Griffo - Pennies/
        ├── mix.wav
        ├── drums.wav
        ├── bass.wav
        ├── other.wav
        └── vocals.wav

Dependencies:
    pip install stempeg soundfile tqdm
"""
import argparse
import pathlib
import numpy as np

try:
    import stempeg
    import soundfile as sf
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install stempeg soundfile")
    exit(1)

# MUSDB18 規約の順序（0:mix, 1:drums, 2:bass, 3:other, 4:vocals）
NAMES = ["mix", "drums", "bass", "other", "vocals"]


def convert_one(src: pathlib.Path, dst_root: pathlib.Path, mono=False):
    """
    1つの .stem.mp4 を5つのWAVに分離

    Args:
        src: .stem.mp4 ファイルパス
        dst_root: 出力ルートディレクトリ
        mono: Trueならモノラル化（L/R平均）
    """
    try:
        # stempeg で読み込み（shape: (5, n_samples, channels)）
        stems, rate = stempeg.read_stems(str(src))
    except Exception as e:
        print(f"[skip] {src.name}: {e}")
        return

    # 出力ディレクトリ（曲名から.stem拡張子を除去）
    out_dir = dst_root / src.stem.replace(".stem", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(NAMES):
        y = stems[i]  # (n_samples, channels)

        # モノラル化（オプション）
        if mono and y.ndim == 2:
            y = np.mean(y, axis=1, keepdims=True)

        # WAV書き込み
        out_path = out_dir / f"{name}.wav"
        sf.write(out_path, y, rate)


def main():
    ap = argparse.ArgumentParser(
        description="MUSDB18 stems (.mp4) to WAV converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--musdb",
        required=True,
        help="musdb18 root directory (contains train/ and/or test/)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="output root for decoded WAVs",
    )
    ap.add_argument(
        "--mono",
        action="store_true",
        help="Convert to mono (L/R average)",
    )
    args = ap.parse_args()

    src_root = pathlib.Path(args.musdb)
    dst_root = pathlib.Path(args.out)

    if not src_root.exists():
        print(f"❌ Source directory not found: {src_root}")
        exit(1)

    # .stem.mp4 ファイルを収集
    paths = list(src_root.glob("**/*.stem.mp4"))
    if not paths:
        print(f"⚠️  No .stem.mp4 files found in {src_root}")
        exit(0)

    print(f"📂 Found {len(paths)} .stem.mp4 files")
    print(f"🎵 Output: {dst_root}")
    print(f"🔊 Mono: {args.mono}")
    print()

    # 一括変換
    for i, p in enumerate(paths, 1):
        if i % 5 == 0 or i == 1:
            print(f"[{i}/{len(paths)}] {p.name}")
        convert_one(p, dst_root, args.mono)

    print(f"\n✅ Conversion complete: {len(paths)} tracks")
    print(f"📁 Output directory: {dst_root}")


if __name__ == "__main__":
    main()
