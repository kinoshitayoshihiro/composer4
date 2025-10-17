#!/usr/bin/env python3
"""
Generate integrated distribution counts from Stage2 results

統合分布カウント生成: Stage2結果から楽器×奏法×テンポ帯の実カウントを集約

Usage:
    python scripts/generate_distribution_counts.py \
      --results-dir output/test_results \
      --output reports/integrated_distribution_counts.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


TEMPO_BANDS = {
    "slow": (60, 95),
    "mid": (96, 130),
    "fast": (131, 180),
}


def classify_tempo_band(tempo: int) -> str:
    """テンポをslow/mid/fastに分類"""
    if tempo < 96:
        return "slow"
    elif tempo < 131:
        return "mid"
    else:
        return "fast"


def analyze_guitar(results: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Guitar奏法分布（テンポ帯別）"""
    distribution = {band: defaultdict(int) for band in ["slow", "mid", "fast"]}
    
    for entry in results:
        scores = entry.get('scores', {})
        
        # 奏法推定（スコアパターンから）
        arpeggio = scores.get('arpeggio_quality', 0)
        chord = scores.get('chord_coherence', 0)
        strum = scores.get('strumming_pattern', 0)
        
        # テンポ推定（仮：中速とする。実データがあればそれを使用）
        tempo_band = "mid"  # TODO: メタデータから取得
        
        # 簡易分類
        if arpeggio > 0.6 and chord < 0.4:
            distribution[tempo_band]['arpeggio'] += 1
        elif strum > 0.5:
            distribution[tempo_band]['strum'] += 1
        elif chord > 0.6:
            distribution[tempo_band]['chord_block'] += 1
        else:
            distribution[tempo_band]['mixed'] += 1
    
    return {k: dict(v) for k, v in distribution.items()}


def analyze_bass(results: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Bass奏法分布（テンポ帯別）"""
    distribution = {band: defaultdict(int) for band in ["slow", "mid", "fast"]}
    
    for entry in results:
        scores = entry.get('scores', {})
        groove = scores.get('groove_quality', 0)
        
        tempo_band = "mid"  # TODO: メタデータから取得
        
        # グルーヴ品質からパターン推定
        if groove < 0.5:
            distribution[tempo_band]['on_grid'] += 1
        elif groove < 0.7:
            distribution[tempo_band]['slight_swing'] += 1
        else:
            distribution[tempo_band]['walking'] += 1
    
    return {k: dict(v) for k, v in distribution.items()}


def analyze_strings(results: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Strings奏法分布（テンポ帯別）"""
    distribution = {band: defaultdict(int) for band in ["slow", "mid", "fast"]}
    
    for entry in results:
        scores = entry.get('scores', {})
        legato = scores.get('legato_quality', 0)
        
        tempo_band = "slow"  # Stringsは通常低速
        
        # レガート品質からアーティキュレーション推定
        if legato > 0.6:
            distribution[tempo_band]['legato'] += 1
        elif legato < 0.3:
            distribution[tempo_band]['staccato'] += 1
        else:
            distribution[tempo_band]['mixed'] += 1
    
    return {k: dict(v) for k, v in distribution.items()}


def analyze_piano(results: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Piano表現分布（テンポ帯別）"""
    distribution = {band: defaultdict(int) for band in ["slow", "mid", "fast"]}
    
    for entry in results:
        scores = entry.get('scores', {})
        melody_expr = scores.get('melody_expression', 0)
        
        tempo_band = "mid"  # POP909は主に中速
        
        # 表現レベルで分類
        if melody_expr > 0.7:
            distribution[tempo_band]['expressive'] += 1
        else:
            distribution[tempo_band]['standard'] += 1
    
    return {k: dict(v) for k, v in distribution.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="Stage2結果ディレクトリ")
    ap.add_argument("--output", required=True, help="出力JSON")
    args = ap.parse_args()
    
    results_dir = Path(args.results_dir)
    integrated = {}
    
    # Guitar
    guitar_file = results_dir / "guitar_full.json"
    if guitar_file.exists():
        with open(guitar_file) as f:
            data = json.load(f)
        integrated['guitar'] = analyze_guitar(data['results'])
        print(f"✅ Guitar: {sum(sum(v.values()) for v in integrated['guitar'].values())} files")
    
    # Bass
    bass_file = results_dir / "bass_full.json"
    if bass_file.exists():
        with open(bass_file) as f:
            data = json.load(f)
        integrated['bass'] = analyze_bass(data['results'])
        print(f"✅ Bass: {sum(sum(v.values()) for v in integrated['bass'].values())} files")
    
    # Strings
    strings_file = results_dir / "strings_full.json"
    if strings_file.exists():
        with open(strings_file) as f:
            data = json.load(f)
        integrated['strings'] = analyze_strings(data['results'])
        print(f"✅ Strings: {sum(sum(v.values()) for v in integrated['strings'].values())} files")
    
    # Piano (Melody)
    piano_file = results_dir / "piano_melody_full.json"
    if piano_file.exists():
        with open(piano_file) as f:
            data = json.load(f)
        integrated['piano'] = analyze_piano(data['results'])
        print(f"✅ Piano: {sum(sum(v.values()) for v in integrated['piano'].values())} files")
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(integrated, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 Integrated distribution saved to: {args.output}")
    print(json.dumps(integrated, ensure_ascii=False, indent=2))
    
    return 0


if __name__ == "__main__":
    exit(main())
