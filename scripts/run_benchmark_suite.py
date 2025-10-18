#!/usr/bin/env python3
"""
run_benchmark_suite.py - ベンチマーク曲集フル実行スクリプト

全ベンチマーク曲を実行し、品質メトリクスを検証します。

Usage:
    python scripts/run_benchmark_suite.py [--config multi_song_benchmark.json]
    python scripts/run_benchmark_suite.py --single configs/benchmarks/pop_upbeat_simple.yaml
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_benchmark_config(config_path: Path) -> Dict[str, Any]:
    """ベンチマーク設定JSON読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml_config(yaml_path: Path) -> Dict[str, Any]:
    """YAML設定読み込み"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_modular_composer(yaml_path: Path, output_dir: Path) -> Optional[Path]:
    """modular_composer.pyでMIDI生成"""
    
    output_midi = output_dir / f"{yaml_path.stem}.mid"
    
    cmd = [
        sys.executable,
        'modular_composer.py',
        '--config', str(yaml_path),
        '--output', str(output_midi),
    ]
    
    print(f"   🎵 Generating MIDI: {yaml_path.stem}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2分タイムアウト
        )
        
        if result.returncode == 0 and output_midi.exists():
            print(f"      ✅ MIDI generated: {output_midi.name}")
            return output_midi
        else:
            print(f"      ❌ MIDI generation failed")
            if result.stderr:
                print(f"         Error: {result.stderr[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"      ❌ Timeout (>120s)")
        return None
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None


def validate_quality_thresholds(yaml_path: Path, midi_path: Path) -> Dict[str, Any]:
    """品質閾値検証 (簡易版)"""
    
    yaml_data = load_yaml_config(yaml_path)
    quality_thresholds = yaml_data.get('quality_thresholds', {})
    expected_metrics = yaml_data.get('meta', {}).get('expected_metrics', {})
    
    # 簡易検証: MIDIファイルが存在し、サイズが妥当か
    validation_result = {
        'file_exists': midi_path.exists(),
        'file_size_bytes': midi_path.stat().st_size if midi_path.exists() else 0,
        'has_thresholds': len(quality_thresholds) > 0,
        'expected_metrics': expected_metrics,
        'status': 'PASS' if midi_path.exists() and midi_path.stat().st_size > 100 else 'FAIL',
    }
    
    return validation_result


def run_single_benchmark(yaml_path: Path, output_dir: Path) -> Dict[str, Any]:
    """単一ベンチマーク実行"""
    
    start_time = time.time()
    
    # MIDI生成
    midi_path = run_modular_composer(yaml_path, output_dir)
    
    if not midi_path:
        return {
            'yaml': str(yaml_path.name),
            'status': 'FAILED',
            'error': 'MIDI generation failed',
            'duration_sec': time.time() - start_time,
        }
    
    # 品質検証
    validation = validate_quality_thresholds(yaml_path, midi_path)
    
    result = {
        'yaml': str(yaml_path.name),
        'midi': str(midi_path.name),
        'status': validation['status'],
        'validation': validation,
        'duration_sec': time.time() - start_time,
    }
    
    print(f"      Status: {result['status']} ({result['duration_sec']:.1f}s)")
    
    return result


def run_benchmark_suite(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    """ベンチマーク全曲実行"""
    
    print(f"\n🚀 Running Benchmark Suite")
    print(f"   Config: {config_path.name}")
    print(f"   Output: {output_dir}\n")
    
    # 設定読み込み
    config = load_benchmark_config(config_path)
    songs = config.get('songs', [])
    
    print(f"📊 Total benchmarks: {len(songs)}\n")
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各曲を実行
    results = []
    project_root = config_path.parent
    
    for i, song in enumerate(songs, 1):
        yaml_file = song['file']
        yaml_path = project_root / yaml_file
        
        print(f"[{i}/{len(songs)}] {song['metadata']['title']}")
        
        if not yaml_path.exists():
            print(f"   ❌ YAML not found: {yaml_path}")
            results.append({
                'yaml': yaml_file,
                'status': 'FAILED',
                'error': 'YAML file not found',
            })
            continue
        
        result = run_single_benchmark(yaml_path, output_dir)
        results.append(result)
    
    # 統計集計
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = len(results) - passed
    total_duration = sum(r.get('duration_sec', 0) for r in results)
    
    summary = {
        'generated': datetime.now().isoformat(),
        'total_benchmarks': len(results),
        'passed': passed,
        'failed': failed,
        'pass_rate': (passed / len(results) * 100) if results else 0,
        'total_duration_sec': total_duration,
        'results': results,
    }
    
    # 結果保存
    summary_path = output_dir / 'benchmark_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 結果表示
    print(f"\n{'='*60}")
    print(f"✅ Benchmark Suite Complete")
    print(f"   Passed: {passed}/{len(results)}")
    print(f"   Failed: {failed}/{len(results)}")
    print(f"   Pass Rate: {summary['pass_rate']:.1f}%")
    print(f"   Total Duration: {total_duration:.1f}s")
    print(f"   Summary: {summary_path}")
    print(f"{'='*60}\n")
    
    # 失敗があれば終了コード1
    sys.exit(0 if failed == 0 else 1)


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmark suite and validate quality metrics'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='multi_song_benchmark.json',
        help='Benchmark suite JSON config (default: multi_song_benchmark.json)'
    )
    parser.add_argument(
        '--single',
        type=str,
        help='Run single YAML file instead of full suite'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_outputs',
        help='Output directory for MIDI files (default: benchmark_outputs)'
    )
    
    args = parser.parse_args()
    
    # パス解決
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    
    # 単一ファイル実行モード
    if args.single:
        yaml_path = Path(args.single)
        
        if not yaml_path.exists():
            print(f"❌ YAML file not found: {yaml_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"\n🎯 Running Single Benchmark")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_single_benchmark(yaml_path, output_dir)
        
        print(f"\n{'='*60}")
        print(f"Result: {result['status']}")
        print(f"{'='*60}\n")
        
        sys.exit(0 if result['status'] == 'PASS' else 1)
    
    # フルスイート実行モード
    config_path = project_root / args.config
    
    if not config_path.exists():
        print(f"❌ Benchmark config not found: {config_path}", file=sys.stderr)
        print(f"   Run: python scripts/generate_benchmark_json.py", file=sys.stderr)
        sys.exit(1)
    
    run_benchmark_suite(config_path, output_dir)


if __name__ == '__main__':
    main()
