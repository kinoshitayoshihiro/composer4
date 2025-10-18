#!/usr/bin/env python3
"""
AB比較WAV自動生成ツール

奏法の違い（例: strum vs fingerpicking）を聴覚的に素早く比較するため、
2つのWAVファイルを短く切り出して交互に並べたAB比較音源を生成します。

使用方法:
    python scripts/generate_ab_comparison.py \\
      --wav-a out/wav/guitar_strum.wav \\
      --wav-b out/wav/guitar_fingerpicking.wav \\
      --output out/ab_comparison/strum_vs_fingerpicking.wav \\
      --duration 5.0 \\
      --offset 10.0
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional
import wave


def load_wav(filepath: Path, duration: Optional[float] = None, offset: float = 0.0):
    """
    WAVファイルを読み込み
    
    Args:
        filepath: WAVファイルパス
        duration: 読み込む長さ（秒）、Noneで全体
        offset: 開始位置（秒）
    
    Returns:
        (audio_data, sample_rate, channels)
    """
    with wave.open(str(filepath), 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        
        # オフセット計算
        start_frame = int(offset * sample_rate)
        wf.setpos(start_frame)
        
        # 読み込みフレーム数
        if duration is None:
            n_frames = wf.getnframes() - start_frame
        else:
            n_frames = int(duration * sample_rate)
        
        # データ読み込み
        audio_bytes = wf.readframes(n_frames)
        
        # numpy配列に変換
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        else:
            dtype = np.int32
        
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)
        
        # ステレオの場合はチャンネル分離
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2)
        
        return audio_data, sample_rate, channels, sample_width


def save_wav(filepath: Path, audio_data: np.ndarray, sample_rate: int, 
             channels: int, sample_width: int):
    """WAVファイルを保存"""
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def create_ab_comparison(
    wav_a_path: Path,
    wav_b_path: Path,
    output_path: Path,
    duration: float = 5.0,
    offset: float = 0.0,
    crossfade_ms: float = 50.0,
    repetitions: int = 3
) -> Path:
    """
    AB比較WAVを生成
    
    Args:
        wav_a_path: 技法Aの音源
        wav_b_path: 技法Bの音源
        output_path: 出力先
        duration: 各クリップの長さ（秒）
        offset: 開始位置（秒）
        crossfade_ms: クロスフェード長（ミリ秒）
        repetitions: A-B繰り返し回数
    
    Returns:
        出力ファイルパス
    """
    print(f"🔊 Creating AB comparison...")
    print(f"   A: {wav_a_path.name}")
    print(f"   B: {wav_b_path.name}")
    
    # 読み込み
    audio_a, sr_a, ch_a, sw_a = load_wav(wav_a_path, duration, offset)
    audio_b, sr_b, ch_b, sw_b = load_wav(wav_b_path, duration, offset)
    
    # パラメータ検証
    if sr_a != sr_b:
        raise ValueError(f"Sample rate mismatch: {sr_a} vs {sr_b}")
    if ch_a != ch_b:
        raise ValueError(f"Channel mismatch: {ch_a} vs {ch_b}")
    if sw_a != sw_b:
        raise ValueError(f"Sample width mismatch: {sw_a} vs {sw_b}")
    
    sample_rate = sr_a
    channels = ch_a
    sample_width = sw_a
    
    # クロスフェード窓
    crossfade_samples = int(crossfade_ms / 1000.0 * sample_rate)
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)
    
    if channels == 2:
        fade_out = fade_out[:, np.newaxis]
        fade_in = fade_in[:, np.newaxis]
    
    # A-B繰り返し
    segments = []
    for i in range(repetitions):
        # A
        seg_a = audio_a.copy().astype(np.float32)
        
        # クロスフェード適用
        if i > 0:  # 最初のAの前にはフェードイン不要
            seg_a[:crossfade_samples] *= fade_in
        if i < repetitions - 1:  # 最後のBの後にはフェードアウト不要
            seg_a[-crossfade_samples:] *= fade_out
        
        segments.append(seg_a)
        
        # B
        seg_b = audio_b.copy().astype(np.float32)
        seg_b[:crossfade_samples] *= fade_in
        
        if i < repetitions - 1:
            seg_b[-crossfade_samples:] *= fade_out
        
        segments.append(seg_b)
    
    # 結合
    output_audio = np.concatenate(segments, axis=0)
    
    # 整数型に戻す
    if sample_width == 1:
        output_audio = np.clip(output_audio, 0, 255).astype(np.uint8)
    elif sample_width == 2:
        output_audio = np.clip(output_audio, -32768, 32767).astype(np.int16)
    else:
        output_audio = np.clip(output_audio, -2147483648, 2147483647).astype(np.int32)
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(output_path, output_audio, sample_rate, channels, sample_width)
    
    total_duration = len(output_audio) / sample_rate
    print(f"✅ AB comparison saved: {output_path}")
    print(f"   Total duration: {total_duration:.1f}s")
    print(f"   Pattern: A-B-A-B-A-B (×{repetitions})")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate AB comparison WAV for technique evaluation'
    )
    parser.add_argument('--wav-a', type=Path, required=True,
                       help='First WAV file (technique A)')
    parser.add_argument('--wav-b', type=Path, required=True,
                       help='Second WAV file (technique B)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output AB comparison WAV')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duration of each clip in seconds (default: 5.0)')
    parser.add_argument('--offset', type=float, default=0.0,
                       help='Start offset in seconds (default: 0.0)')
    parser.add_argument('--crossfade-ms', type=float, default=50.0,
                       help='Crossfade duration in milliseconds (default: 50.0)')
    parser.add_argument('--repetitions', type=int, default=3,
                       help='Number of A-B repetitions (default: 3)')
    
    args = parser.parse_args()
    
    # 検証
    if not args.wav_a.exists():
        print(f"❌ File not found: {args.wav_a}")
        return 1
    
    if not args.wav_b.exists():
        print(f"❌ File not found: {args.wav_b}")
        return 1
    
    # 生成
    try:
        create_ab_comparison(
            wav_a_path=args.wav_a,
            wav_b_path=args.wav_b,
            output_path=args.output,
            duration=args.duration,
            offset=args.offset,
            crossfade_ms=args.crossfade_ms,
            repetitions=args.repetitions
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
