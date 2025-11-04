#!/usr/bin/env python3
"""
check_tempo_consistency.py
---------------------------
テンポ一貫性チェック

song_package.yaml の bpm と各Plan JSON の tempo_bpm が一致しているか検証。
ズレがあると humanize の絶対時間が変わるため、早期検出が重要。

Usage:
    python3 scripts/check_tempo_consistency.py \
        --song-dir song_packages/suno_project/song_001
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def load_song_package_tempo(song_pkg_path: Path) -> float:
    """song_package.yaml からテンポ取得"""
    with open(song_pkg_path, 'r', encoding='utf-8') as f:
        pkg = yaml.safe_load(f)
        meta = pkg.get('meta', {})
        return meta.get('bpm', meta.get('tempo_bpm', 120.0))


def load_plan_tempo(plan_path: Path) -> float:
    """Plan JSON からテンポ取得"""
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
        return plan.get('tempo_bpm', 0.0)


def check_tempo_consistency(
    song_dir: Path,
    tolerance: float = 0.1,
    verbose: bool = True
) -> Tuple[bool, Dict[str, float]]:
    """
    テンポ一貫性チェック
    
    Args:
        song_dir: Song directory
        tolerance: 許容誤差（BPM）
        verbose: 詳細出力
    
    Returns:
        (all_consistent, tempo_map)
    """
    if verbose:
        print(f"🎵 Checking tempo consistency: {song_dir}")
    
    # song_package.yaml テンポ取得
    song_pkg_path = song_dir / 'song_package.yaml'
    if not song_pkg_path.exists():
        print(f"❌ song_package.yaml not found: {song_pkg_path}")
        return False, {}
    
    reference_tempo = load_song_package_tempo(song_pkg_path)
    if verbose:
        print(f"   Reference tempo (song_package.yaml): {reference_tempo:.2f} BPM")
    
    tempo_map = {'song_package.yaml': reference_tempo}
    all_consistent = True
    
    # Plan JSONファイル検証
    plan_files = [
        'bass_plan.json',
        'guitar_plan.json',
        'piano_plan.json',
        'strings_plan.json',
        'drums_plan.json',
        'drums_plan_real.json',
        'full_arrangement.json'
    ]
    
    for plan_name in plan_files:
        plan_path = song_dir / plan_name
        if not plan_path.exists():
            continue
        
        plan_tempo = load_plan_tempo(plan_path)
        tempo_map[plan_name] = plan_tempo
        
        # 一貫性チェック
        diff = abs(plan_tempo - reference_tempo)
        if diff > tolerance:
            print(f"   ❌ {plan_name}: {plan_tempo:.2f} BPM (diff: {diff:.2f})")
            all_consistent = False
        else:
            if verbose:
                print(f"   ✅ {plan_name}: {plan_tempo:.2f} BPM")
    
    if verbose:
        print()
        if all_consistent:
            print(f"✅ All tempos consistent (tolerance: ±{tolerance} BPM)")
        else:
            print(f"❌ Tempo inconsistency detected!")
            print(f"   Recommendation: Regenerate plans with correct tempo")
    
    return all_consistent, tempo_map


def main():
    parser = argparse.ArgumentParser(description='Check tempo consistency across plans')
    parser.add_argument(
        '--song-dir',
        type=Path,
        required=True,
        help='Song directory (e.g., song_packages/suno_project/song_001)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.1,
        help='Tolerance for tempo difference (BPM, default: 0.1)'
    )
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument(
        '--json-output',
        type=Path,
        default=None,
        help='Optional: Save tempo map to JSON'
    )
    
    args = parser.parse_args()
    
    consistent, tempo_map = check_tempo_consistency(
        args.song_dir,
        tolerance=args.tolerance,
        verbose=not args.quiet
    )
    
    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump({
                'consistent': consistent,
                'tolerance': args.tolerance,
                'tempo_map': tempo_map
            }, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\n📄 Saved tempo map: {args.json_output}")
    
    exit(0 if consistent else 1)


if __name__ == '__main__':
    main()
