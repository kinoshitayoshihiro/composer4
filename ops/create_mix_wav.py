#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/create_mix_wav.py
複数のStem WAVを合成してMix WAVを生成

使い方:
python ops/create_mix_wav.py \
  --stems-dir data/suno_ai/suno_themesong/song_002/oreno_vol1/WAV \
  --out data/suno_ai/suno_themesong/song_002/_auto_Other.wav \
  --exclude Vocals
"""
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List


def load_and_normalize(wav_path: Path, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """WAV読み込み＆正規化"""
    try:
        y, sr = sf.read(str(wav_path), dtype="float32")
        # モノラル化
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        # リサンプリング（必要なら）
        if sr != target_sr:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return y, sr
    except Exception as e:
        print(f"⚠️  Failed to load {wav_path}: {e}")
        return np.array([]), target_sr


def create_mix(
    stems_dir: Path, exclude: List[str], target_sr: int = 44100
) -> tuple[np.ndarray, int]:
    """Stem WAVを合成"""
    wav_files = sorted(stems_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {stems_dir}")

    # 除外パターン
    exclude_lower = [e.lower() for e in exclude]

    ys = []
    max_len = 0

    for wav_path in wav_files:
        # 除外チェック
        name_lower = wav_path.stem.lower()
        if any(exc in name_lower for exc in exclude_lower):
            print(f"  ⏭️  Skipped: {wav_path.name}")
            continue

        print(f"  ✓ Loading: {wav_path.name}")
        y, sr = load_and_normalize(wav_path, target_sr)
        if len(y) == 0:
            continue

        ys.append(y)
        max_len = max(max_len, len(y))

    if not ys:
        raise ValueError(f"No valid WAV files found (after excluding {exclude})")

    # ゼロパディング＆合成
    mix = np.zeros(max_len, dtype=np.float32)
    for y in ys:
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)), mode="constant")
        mix += y

    # 正規化（クリッピング防止）
    max_val = np.max(np.abs(mix))
    if max_val > 0:
        mix = mix / max_val * 0.95  # -0.5dB程度のヘッドルーム

    return mix, target_sr


def main():
    ap = argparse.ArgumentParser(description="Create mix WAV from stems")
    ap.add_argument("--stems-dir", type=Path, required=True, help="Directory with stem WAVs")
    ap.add_argument("--out", type=Path, required=True, help="Output mix WAV path")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude stems with these keywords (e.g., Vocals)",
    )
    ap.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100)")
    args = ap.parse_args()

    # デフォルト除外（Vocals）
    if not args.exclude:
        args.exclude = ["Vocals"]

    print(f"🎚️  Creating mix WAV from stems...")
    print(f"   Stems dir: {args.stems_dir}")
    print(f"   Exclude: {args.exclude}")

    mix, sr = create_mix(args.stems_dir, args.exclude, args.sr)

    # 保存
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), mix, sr, subtype="PCM_16")

    print(f"✅ Mix WAV created: {args.out}")
    print(f"   Duration: {len(mix) / sr:.2f} sec")
    print(f"   Sample rate: {sr} Hz")


if __name__ == "__main__":
    main()
