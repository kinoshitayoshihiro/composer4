#!/usr/bin/env python3
"""
Batch Articulation Renderer
複数MIDIファイルに対して全奏法バリエーションを自動生成

VioPTT/MOSA-VPT準拠の合成データ生成パイプライン

Usage:
    # 単一ディレクトリ処理
    python scripts/daw/batch_articulation_renderer.py \
        --input-dir output/rhythm_ai/drumclean_midi \
        --output-dir output/articulations \
        --techniques detache spiccato pizzicato flageolet
    
    # 複数技法、並列処理
    python scripts/daw/batch_articulation_renderer.py \
        --input-dir data/suno_stems/violin \
        --output-dir output/mosa_vpt_synthetic \
        --techniques detache spiccato pizzicato flageolet legato staccato \
        --instrument violin \
        --jobs 8 \
        --render-wav \
        --vst-path "/Library/Audio/Plug-Ins/VST3/Synchron Solo Violin I.vst3"

Dependencies:
    - violin_articulation_automation.py (同一ディレクトリ)
    - technique_map.yaml
    - dawdreamer (WAVレンダリング時)
"""

import argparse
import multiprocessing as mp
import subprocess
from pathlib import Path
from typing import List


def process_single_file(args_tuple):
    """
    単一MIDIファイルを処理（マルチプロセス用）
    
    Args:
        args_tuple: (midi_path, output_dir, techniques, instrument, tech_map, render_wav, vst_path)
    """
    midi_path, output_dir, techniques, instrument, tech_map, render_wav, vst_path = args_tuple
    
    cmd = [
        "python3",
        "scripts/daw/violin_articulation_automation.py",
        "--input", str(midi_path),
        "--output-dir", str(output_dir),
        "--techniques", *techniques,
        "--instrument", instrument,
        "--tech-map", tech_map
    ]
    
    if render_wav:
        cmd.append("--render-wav")
        if vst_path:
            cmd.extend(["--vst-path", vst_path])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return (midi_path.name, "success", result.stdout)
    except subprocess.CalledProcessError as e:
        return (midi_path.name, "error", e.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Batch Articulation Renderer (VioPTT/MOSA-VPT準拠)"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="入力MIDIディレクトリ"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=["detache", "spiccato", "pizzicato", "flageolet"],
        help="適用する奏法リスト（VioPTT 4技法デフォルト）"
    )
    parser.add_argument(
        "--instrument",
        default="violin",
        help="楽器名（violin, guitar, drums等）"
    )
    parser.add_argument(
        "--tech-map",
        default="configs/labels/technique_map.yaml",
        help="technique_map.yamlパス"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="並列処理数"
    )
    parser.add_argument(
        "--render-wav",
        action="store_true",
        help="WAVファイルもレンダリング（DAWDreamer必須）"
    )
    parser.add_argument(
        "--vst-path",
        help="VST/VST3プラグインパス（WAVレンダリング時）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="処理ファイル数上限（テスト用）"
    )
    
    args = parser.parse_args()
    
    # 入力ディレクトリ確認
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # MIDIファイル収集
    midi_files = sorted(input_dir.glob("**/*.mid")) + sorted(input_dir.glob("**/*.midi"))
    
    if args.limit:
        midi_files = midi_files[:args.limit]
    
    if not midi_files:
        print(f"⚠️  No MIDI files found in {input_dir}")
        return
    
    print(f"📂 Found {len(midi_files)} MIDI files")
    print(f"🎵 Techniques: {', '.join(args.techniques)}")
    print(f"⚙️  Parallel jobs: {args.jobs}")
    print()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # マルチプロセス処理用引数準備
    process_args = [
        (
            midi_path,
            output_dir,
            args.techniques,
            args.instrument,
            args.tech_map,
            args.render_wav,
            args.vst_path
        )
        for midi_path in midi_files
    ]
    
    # 並列処理実行
    print(f"🚀 Starting batch processing ({len(midi_files)} files)...")
    
    with mp.Pool(processes=args.jobs) as pool:
        results = pool.map(process_single_file, process_args)
    
    # 結果集計
    success_count = sum(1 for _, status, _ in results if status == "success")
    error_count = len(results) - success_count
    
    print()
    print("=" * 50)
    print(f"✅ Success: {success_count}/{len(results)}")
    print(f"❌ Errors:  {error_count}/{len(results)}")
    
    if error_count > 0:
        print("\nErrors:")
        for filename, status, message in results:
            if status == "error":
                print(f"  - {filename}: {message[:100]}")
    
    # 統計情報
    total_generated = success_count * len(args.techniques)
    print()
    print(f"📊 Total articulation variants generated: {total_generated}")
    print(f"   ({success_count} files × {len(args.techniques)} techniques)")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
