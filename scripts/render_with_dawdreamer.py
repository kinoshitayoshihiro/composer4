#!/usr/bin/env python3
"""
render_with_dawdreamer.py - DAWDreamer統合レンダラー

制御MIDI + ノートMIDI → DAWDreamer → WAV出力

Usage:
    python3 scripts/render_with_dawdreamer.py \
      --note-midi song_packages/<project>/<song>/melody.mid \
      --control-midi song_packages/<project>/<song>/violin_controls.mid \
      --vst-path /path/to/vst.vst3 \
      --output song_packages/<project>/<song>/rendered.wav \
      --duration 64.0
"""

import argparse
from pathlib import Path

try:
    import dawdreamer as daw
    from scipy.io import wavfile
    import numpy as np
except ImportError:
    print("❌ Error: dawdreamer not installed.")
    print("   Install: pip install dawdreamer")
    raise


def render_with_vst(
    note_midi_path: Path,
    control_midi_path: Path,
    vst_path: Path,
    output_path: Path,
    duration: float,
    sample_rate: int = 44100,
    buffer_size: int = 128,
):
    """VST + 制御MIDI/ノートMIDI → WAV出力"""

    # DAWDreamer engine作成
    engine = daw.RenderEngine(sample_rate, buffer_size)

    # VST読み込み
    print(f"📖 Loading VST: {vst_path}")
    vst = engine.make_plugin_processor("vst_instrument", str(vst_path))
    
    # VST状態ファイルがあればロード（プリセット復元）
    # SampleTank 4の場合: configs/sampletank4_piano_state.vststate
    vst_name = vst_path.stem.lower().replace(' ', '').replace('_', '')  # スペースとアンダースコアを削除
    state_file = Path("configs") / f"{vst_name}_piano_state.vststate"
    if state_file.exists():
        print(f"📂 Loading VST state: {state_file}")
        vst.load_state(str(state_file))
    else:
        print(f"⚠️  No state file found: {state_file} (using VST defaults)")

    # MergedまたはControl MIDI読み込み
    # control_midi_pathが"merged"を含む場合は、それを使用（Note+Control統合済み）
    # それ以外の場合は、control_midiのみを使用
    midi_to_load = control_midi_path
    if "merged" in str(control_midi_path):
        print(f"🎹 Loading merged MIDI (notes + controls): {control_midi_path}")
    else:
        print(f"� Loading control MIDI: {control_midi_path}")
        print(f"🎵 Note MIDI: {note_midi_path} (info only, using merged)")
    
    vst.load_midi(str(midi_to_load))

    # グラフ設定（load_graphメソッドを使用）
    engine.load_graph([(vst, [])])  # VST → 出力

    # レンダリング
    print(f"🎧 Rendering ({duration}s)...")
    engine.render(duration)

    # WAV出力
    audio = engine.get_audio()
    
    # デバッグ情報
    print(f"📊 Audio shape: {audio.shape}")
    print(f"📊 Audio dtype: {audio.dtype}")
    print(f"📊 Audio range: [{audio.min():.6f}, {audio.max():.6f}]")
    print(f"📊 Audio mean abs: {np.abs(audio).mean():.6f}")
    print(f"📊 Non-zero samples: {np.count_nonzero(audio):,} / {audio.size:,}")

    # Stereo変換（mono → stereo）
    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    # 正規化
    audio = audio.T  # (channels, samples) → (samples, channels)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767.0).astype(np.int16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_path), sample_rate, audio)

    print(f"✅ WAV saved: {output_path}")
    print(f"   Duration: {duration}s")
    print(f"   Sample rate: {sample_rate} Hz")


def main():
    parser = argparse.ArgumentParser(description="DAWDreamer統合レンダラー")
    parser.add_argument("--note-midi", type=Path, required=True, help="Note MIDI file path")
    parser.add_argument(
        "--control-midi",
        type=Path,
        required=True,
        help="Control MIDI file path (from vioptt_render_stub.py)",
    )
    parser.add_argument(
        "--vst-path", type=Path, required=True, help="VST plugin path (.vst3 or .dll)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output WAV file path")
    parser.add_argument(
        "--duration", type=float, default=64.0, help="Render duration in seconds (default: 64)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=44100, help="Sample rate (default: 44100)"
    )
    parser.add_argument("--buffer-size", type=int, default=128, help="Buffer size (default: 128)")

    args = parser.parse_args()

    # VST存在確認
    if not args.vst_path.exists():
        print(f"❌ Error: VST not found: {args.vst_path}")
        print(f"   Please install the VST or specify a valid path.")
        return 1

    # レンダリング実行
    render_with_vst(
        args.note_midi,
        args.control_midi,
        args.vst_path,
        args.output,
        args.duration,
        args.sample_rate,
        args.buffer_size,
    )

    return 0


if __name__ == "__main__":
    exit(main())
