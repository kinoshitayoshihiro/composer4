#!/usr/bin/env python3
"""
save_vst_state.py - VST状態の保存

SampleTank 4などのVSTの現在の状態（プリセット、パラメータ）を
バイナリファイルとして保存します。

Usage:
    # GUIでプリセットを選択した後、この状態を保存
    python3 scripts/save_vst_state.py \
      --vst-path "/Library/Audio/Plug-Ins/VST3/SampleTank 4.vst3" \
      --output configs/sampletank4_piano_state.vststate
"""

import argparse
import dawdreamer as daw
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Save VST state to file")
    parser.add_argument("--vst-path", required=True, help="Path to VST3 plugin")
    parser.add_argument("--output", required=True, help="Output .vststate file")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--buffer-size", type=int, default=512)
    
    args = parser.parse_args()
    
    print(f"🎹 Loading VST: {args.vst_path}")
    
    # エンジン初期化
    engine = daw.RenderEngine(args.sample_rate, args.buffer_size)
    
    # VST読み込み
    vst = engine.make_plugin_processor('vst_plugin', args.vst_path)
    
    print(f"✅ VST loaded: {vst.get_name()}")
    
    # 出力ディレクトリ作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 状態を保存（ファイルパスを直接渡す）
    vst.save_state(str(output_path))
    
    print(f"✅ State saved: {output_path}")
    print(f"📊 State size: {output_path.stat().st_size:,} bytes")
    
    print("\n📝 Note: このファイルをDAWDreamerで load_state() して使用します")
    print(f"    vst.load_state(open('{output_path}', 'rb').read())")


if __name__ == "__main__":
    main()
