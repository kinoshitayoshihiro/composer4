#!/usr/bin/env python3
"""
Technique Distribution Analyzer

Stage2本番実行結果から奏法・アーティキュレーション分布を可視化

Usage:
    python scripts/analyze_technique_distribution.py \
      --instrument guitar \
      --input output/test_results/guitar_full.json \
      --output reports/guitar_technique_dist.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import numpy as np


def analyze_guitar_techniques(results: List[Dict]) -> Dict:
    """Guitar奏法分布の分析"""
    techniques = defaultdict(int)
    tempo_buckets = defaultdict(list)
    density_buckets = defaultdict(list)
    
    for result in results:
        filename = result.get('file', '')
        scores = result.get('scores', {})
        
        # 奏法推定（スコアパターンから）
        arpeggio = scores.get('arpeggio_quality', 0)
        chord = scores.get('chord_coherence', 0)
        strum = scores.get('strumming_pattern', 0)
        
        # 簡易分類
        if arpeggio > 0.6 and chord < 0.4:
            techniques['arpeggio'] += 1
        elif strum > 0.5:
            techniques['strum'] += 1
        elif chord > 0.6:
            techniques['chord_block'] += 1
        else:
            techniques['mixed'] += 1
        
        # テンポ・密度推定（TODO: メタデータから取得）
        total_score = result.get('total_score', 0)
        tempo_buckets['unknown'].append(total_score)
        density_buckets['unknown'].append(total_score)
    
    total = sum(techniques.values())
    distribution = {k: v/total for k, v in techniques.items()}
    
    return {
        'technique_distribution': distribution,
        'technique_counts': dict(techniques),
        'total_files': total,
        'recommendations': generate_recommendations(distribution, 'guitar')
    }


def analyze_bass_groove(results: List[Dict]) -> Dict:
    """Bass groove分布の分析"""
    groove_patterns = defaultdict(int)
    grid_adherence = []
    
    for result in results:
        scores = result.get('scores', {})
        
        groove = scores.get('groove_quality', 0)
        root_acc = scores.get('root_accuracy', 0)
        
        # グリッド推定（groove低 = グリッド強）
        if groove < 0.5:
            groove_patterns['on_grid'] += 1
        elif groove < 0.7:
            groove_patterns['slight_swing'] += 1
        else:
            groove_patterns['syncopated'] += 1
        
        grid_adherence.append(1.0 - groove)
    
    total = sum(groove_patterns.values())
    distribution = {k: v/total for k, v in groove_patterns.items()}
    
    return {
        'groove_distribution': distribution,
        'groove_counts': dict(groove_patterns),
        'grid_adherence_mean': np.mean(grid_adherence),
        'grid_adherence_std': np.std(grid_adherence),
        'total_files': total,
        'recommendations': generate_recommendations(distribution, 'bass')
    }


def analyze_strings_articulation(results: List[Dict]) -> Dict:
    """Strings アーティキュレーション分布の分析"""
    articulations = defaultdict(int)
    legato_scores = []
    
    for result in results:
        scores = result.get('scores', {})
        
        legato = scores.get('legato_quality', 0)
        bowing = scores.get('bowing_expression', 0)
        
        legato_scores.append(legato)
        
        # アーティキュレーション推定
        if legato > 0.6:
            articulations['legato'] += 1
        elif legato < 0.3:
            articulations['staccato'] += 1
        else:
            articulations['mixed'] += 1
    
    total = sum(articulations.values())
    distribution = {k: v/total for k, v in articulations.items()}
    
    return {
        'articulation_distribution': distribution,
        'articulation_counts': dict(articulations),
        'legato_mean': np.mean(legato_scores),
        'legato_std': np.std(legato_scores),
        'total_files': total,
        'recommendations': generate_recommendations(distribution, 'strings')
    }


def analyze_piano_expression(results: List[Dict]) -> Dict:
    """Piano 表現分布の分析"""
    dynamics_ranges = []
    rhythm_diversity = []
    
    for result in results:
        scores = result.get('scores', {})
        
        dynamics_ranges.append(scores.get('dynamics_range', 0))
        rhythm_diversity.append(scores.get('rhythm_diversity', 0))
    
    return {
        'dynamics_range_mean': np.mean(dynamics_ranges),
        'dynamics_range_std': np.std(dynamics_ranges),
        'rhythm_diversity_mean': np.mean(rhythm_diversity),
        'rhythm_diversity_std': np.std(rhythm_diversity),
        'total_files': len(results),
        'recommendations': generate_recommendations({}, 'piano')
    }


def generate_recommendations(distribution: Dict, instrument: str) -> List[str]:
    """分布に基づく改善提案を生成"""
    recommendations = []
    
    if instrument == 'guitar':
        arpeggio_ratio = distribution.get('arpeggio', 0)
        strum_ratio = distribution.get('strum', 0)
        
        if arpeggio_ratio > 0.5:
            recommendations.append(
                f"⚠️ Arpeggio過多 ({arpeggio_ratio:.1%}) - ストラムパターン補完推奨"
            )
            recommendations.append(
                f"💡 合成データで strum:arpeggio = 6:4 を目指す"
            )
        
        if strum_ratio < 0.3:
            recommendations.append(
                f"⚠️ ストラム不足 ({strum_ratio:.1%}) - 合成で25-30%補完"
            )
    
    elif instrument == 'bass':
        on_grid_ratio = distribution.get('on_grid', 0)
        
        if on_grid_ratio > 0.7:
            recommendations.append(
                f"⚠️ グリッド貼り付き過多 ({on_grid_ratio:.1%})"
            )
            recommendations.append(
                f"💡 ±10-20ms ランダムIOI揺らぎを付与"
            )
            recommendations.append(
                f"💡 合成は最小限（跳躍/経過音/クロマチック接続のみ）"
            )
    
    elif instrument == 'strings':
        legato_ratio = distribution.get('legato', 0)
        staccato_ratio = distribution.get('staccato', 0)
        
        if staccato_ratio > 0.4:
            recommendations.append(
                f"⚠️ Staccato過多 ({staccato_ratio:.1%})"
            )
            recommendations.append(
                f"💡 Legato比率を引き上げ（目標: legato:staccato:spiccato = 6:3:1）"
            )
        
        if legato_ratio < 0.5:
            recommendations.append(
                f"⚠️ Legato不足 ({legato_ratio:.1%}) - 合成で補完"
            )
    
    elif instrument == 'piano':
        recommendations.append(
            "✅ POP909データは高品質 - 合成は最小限（10-15%）"
        )
        recommendations.append(
            "💡 ペダリング品質改善を優先"
        )
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Analyze technique distribution from Stage2 results')
    parser.add_argument('--instrument', required=True, choices=['guitar', 'bass', 'strings', 'piano'])
    parser.add_argument('--input', required=True, help='Input JSON file (test results)')
    parser.add_argument('--output', required=True, help='Output JSON file (distribution report)')
    args = parser.parse_args()
    
    # 結果ファイル読み込み
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1
    
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    # 楽器別分析
    if args.instrument == 'guitar':
        analysis = analyze_guitar_techniques(results)
    elif args.instrument == 'bass':
        analysis = analyze_bass_groove(results)
    elif args.instrument == 'strings':
        analysis = analyze_strings_articulation(results)
    elif args.instrument == 'piano':
        analysis = analyze_piano_expression(results)
    
    # 結果出力
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # コンソール表示
    print(f"\n{'='*60}")
    print(f"📊 {args.instrument.upper()} Technique Distribution Analysis")
    print(f"{'='*60}\n")
    
    print(f"Total files analyzed: {analysis['total_files']}\n")
    
    # 分布表示
    if 'technique_distribution' in analysis:
        print("Technique Distribution:")
        for tech, ratio in analysis['technique_distribution'].items():
            print(f"  {tech:20s}: {ratio:6.1%}")
    elif 'groove_distribution' in analysis:
        print("Groove Distribution:")
        for pattern, ratio in analysis['groove_distribution'].items():
            print(f"  {pattern:20s}: {ratio:6.1%}")
        print(f"\nGrid Adherence: {analysis['grid_adherence_mean']:.3f} ± {analysis['grid_adherence_std']:.3f}")
    elif 'articulation_distribution' in analysis:
        print("Articulation Distribution:")
        for artic, ratio in analysis['articulation_distribution'].items():
            print(f"  {artic:20s}: {ratio:6.1%}")
        print(f"\nLegato Score: {analysis['legato_mean']:.3f} ± {analysis['legato_std']:.3f}")
    else:
        print(f"Dynamics Range: {analysis['dynamics_range_mean']:.3f} ± {analysis['dynamics_range_std']:.3f}")
        print(f"Rhythm Diversity: {analysis['rhythm_diversity_mean']:.3f} ± {analysis['rhythm_diversity_std']:.3f}")
    
    # 推奨事項
    print(f"\n{'='*60}")
    print("📋 Recommendations:")
    print(f"{'='*60}\n")
    for rec in analysis['recommendations']:
        print(f"  {rec}")
    
    print(f"\n✅ Report saved to: {output_path}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
