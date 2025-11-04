#!/usr/bin/env python3
"""
Safe-Kit Fallback Module
========================

KPI Gate失敗小節を安全なパターン（Safe-Kit）に自動置換する機能。

機能:
- KPI Gate失敗小節の検出
- Safe-Kit候補の検索（低backbeat_strength、中密度）
- 自動置換とrecommendations更新

使用例:
    python3 scripts/apply_safe_kit_fallback.py \
        --recommendations drums_recommendations.json \
        --kpi-report kpi_gate_report.json \
        --output drums_recommendations_fixed.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_safe_kit_patterns(rhythm_features_path, safe_kit_criteria):
    """
    Safe-Kit候補パターンを読み込む
    
    Args:
        rhythm_features_path: rhythm_features_merged.parquet
        safe_kit_criteria: Safe-Kit条件（dict）
    
    Returns:
        pd.DataFrame: Safe-Kit候補パターン
    """
    logger.info(f"📖 Loading rhythm features: {rhythm_features_path}")
    df = pd.read_parquet(rhythm_features_path)
    
    # Safe-Kit条件でフィルタリング
    safe_patterns = df[
        (df['backbeat_strength'] >= safe_kit_criteria['backbeat_strength']['min']) &
        (df['backbeat_strength'] <= safe_kit_criteria['backbeat_strength']['max']) &
        (df['hat_density'] >= safe_kit_criteria['density']['min']) &
        (df['hat_density'] <= safe_kit_criteria['density']['max']) &
        (df['swing_pct'] / 100 >= safe_kit_criteria['swing']['min']) &
        (df['swing_pct'] / 100 <= safe_kit_criteria['swing']['max'])
    ].copy()
    
    logger.info(f"✅ Safe-Kit candidates: {len(safe_patterns):,} patterns")
    logger.info(f"   Backbeat: {safe_patterns['backbeat_strength'].min():.2f} .. {safe_patterns['backbeat_strength'].max():.2f}")
    logger.info(f"   Density: {safe_patterns['hat_density'].min():.2f} .. {safe_patterns['hat_density'].max():.2f}")
    
    return safe_patterns


def try_micro_fix(failed_bar, fail_reasons):
    """
    KPI Fail理由に応じて微修正を試みる（置換前の軽微な調整）
    
    Args:
        failed_bar: 失敗小節データ（dict）
        fail_reasons: 失敗理由リスト（list of str）
    
    Returns:
        dict: 修正後の小節データ（修正不可ならNone）
    """
    fixed_bar = failed_bar.copy()
    applied_fixes = []
    
    for reason in fail_reasons:
        if 'backbeat_strength' in reason.lower() and 'high' in reason.lower():
            # Backbeat強度過多 → Snare Velocity減衰
            logger.info(f"   🔧 Micro-fix: Reduce backbeat velocity (0.90x)")
            fixed_bar['backbeat_strength'] = failed_bar.get('backbeat_strength', 1.0) * 0.90
            applied_fixes.append('backbeat_velocity_reduction')
            
        elif 'density' in reason.lower() and 'high' in reason.lower():
            # 密度過多 → Hat間引き（シミュレーション）
            logger.info(f"   🔧 Micro-fix: Reduce hat density (0.85x)")
            fixed_bar['density'] = failed_bar.get('density', 8.0) * 0.85
            applied_fixes.append('hat_density_reduction')
    
    if applied_fixes:
        fixed_bar['micro_fixes_applied'] = applied_fixes
        return fixed_bar
    
    return None


def find_safe_replacement(
    failed_bar, 
    safe_patterns, 
    family_preference=None,
    style_preference=None,
    song_tempo_bpm=None,
    used_patterns=None
):
    """
    失敗小節に対する最適Safe-Kitパターンを検索
    
    改善点:
    - テンポ±10%制約
    - 密度目標±1.0制約
    - Style優先（rock/jazz/electronic等）
    
    Args:
        failed_bar: 失敗小節データ（dict）
        safe_patterns: Safe-Kit候補（DataFrame）
        family_preference: 優先family（str, optional）
        style_preference: 優先スタイル（str, optional）
        song_tempo_bpm: 曲のテンポ（float, optional）
        used_patterns: 使用済みパターン（set, optional）
    
    Returns:
        dict: 最適Safe-Kitパターン
    """
    candidates = safe_patterns.copy()
    
    # テンポ制約（±10%）
    if song_tempo_bpm and 'tempo_bpm' in candidates.columns:
        tempo_tolerance = 0.10
        tempo_ok = (
            (candidates['tempo_bpm'] - song_tempo_bpm).abs() / song_tempo_bpm <= tempo_tolerance
        )
        tempo_candidates = candidates[tempo_ok]
        if len(tempo_candidates) > 0:
            candidates = tempo_candidates
            logger.info(f"   ✓ Tempo filter: {len(candidates)} candidates within ±10%")
    
    # 密度制約（±1.0）
    target_density = failed_bar.get('density_target', 6.0)
    density_tolerance = 1.0
    density_ok = (candidates['hat_density'] - target_density).abs() <= density_tolerance
    density_candidates = candidates[density_ok]
    if len(density_candidates) > 0:
        candidates = density_candidates
        logger.info(f"   ✓ Density filter: {len(candidates)} candidates within ±{density_tolerance}")
    
    # Style優先フィルタリング
    if style_preference and 'style' in candidates.columns:
        style_candidates = candidates[
            (candidates['style'] == style_preference) | 
            (candidates['style'].isna())
        ]
        if len(style_candidates) > 0:
            candidates = style_candidates
            logger.info(f"   ✓ Style filter: {len(candidates)} candidates match '{style_preference}'")
    
    # Family優先フィルタリング
    if family_preference and 'family_label' in candidates.columns:
        family_candidates = candidates[candidates['family_label'] == family_preference]
        if len(family_candidates) > 0:
            candidates = family_candidates
            logger.info(f"   ✓ Family filter: {len(candidates)} candidates match '{family_preference}'")
    
    # 目標値との距離計算
    target_swing = failed_bar.get('swing_target', 0.0)
    
    candidates['density_score'] = 1.0 / (1.0 + np.abs(candidates['hat_density'] - target_density))
    candidates['swing_score'] = 1.0 / (1.0 + np.abs(candidates['swing_pct'] / 100 - target_swing))
    candidates['total_score'] = candidates['density_score'] * 0.7 + candidates['swing_score'] * 0.3
    
    # 多様性ペナルティ
    if used_patterns:
        candidates['diversity_penalty'] = candidates['loop_id'].apply(
            lambda x: 0.3 if x in used_patterns else 0.0
        )
        candidates['total_score'] -= candidates['diversity_penalty']
    
    # 最適パターン選択
    best_idx = candidates['total_score'].idxmax()
    best_pattern = candidates.loc[best_idx]
    
    return {
        'loop_id': best_pattern['loop_id'],
        'family': best_pattern.get('family_label', 'STRAIGHT_8'),
        'confidence': 1.0,  # Safe-Kit信頼度は常に1.0
        'density': best_pattern['hat_density'],
        'swing': best_pattern['swing_pct'] / 100,
        'backbeat_strength': best_pattern['backbeat_strength'],
        'safe_kit_applied': True
    }


def apply_safe_kit_fallback(
    recommendations, 
    kpi_report, 
    safe_patterns,
    song_metadata=None,
    preserve_diversity=True
):
    """
    KPI Gate失敗小節にSafe-Kit Fallbackを適用
    
    改善点:
    - 微修正→置換の二段階処理
    - テンポ/スタイル情報の活用
    
    Args:
        recommendations: drums_recommendations.json（dict）
        kpi_report: kpi_gate_report.json（dict）
        safe_patterns: Safe-Kit候補（DataFrame）
        song_metadata: 曲メタデータ（dict, optional）
        preserve_diversity: 多様性保持（bool）
    
    Returns:
        dict: 修正後recommendations
    """
    results = kpi_report['results']
    
    # recommendationsは bar_0, bar_1... 形式のdict
    bars_dict = {k: v for k, v in recommendations.items() if k.startswith('bar_')}
    
    # 曲メタデータ取得
    song_tempo_bpm = song_metadata.get('tempo_bpm', 120.0) if song_metadata else 120.0
    song_style = song_metadata.get('style') if song_metadata else None
    
    # 失敗小節の検出
    failed_bars = []
    for bar_key, result in results.items():
        if not result['kpi_pass'] and result['safe_kit_fallback_recommended']:
            bar_index = result['bar_index']
            fail_reasons = result.get('fail_reasons', [])
            failed_bars.append((bar_index, bar_key, fail_reasons))
    
    logger.info(f"🔍 Found {len(failed_bars)} failed bars requiring Safe-Kit processing")
    
    if len(failed_bars) == 0:
        logger.info("✅ No failed bars, Safe-Kit fallback not needed")
        return recommendations
    
    # 使用済みパターン（多様性保持）
    used_patterns = set()
    if preserve_diversity:
        for bar_key, bar_data in bars_dict.items():
            bar_index = bar_data['bar_index']
            if bar_index not in [idx for idx, _, _ in failed_bars]:
                pattern_id = bar_data['pattern']['pattern_id']
                used_patterns.add(pattern_id)
    
    # Safe-Kit処理（微修正→置換の二段階）
    micro_fixed_count = 0
    replaced_count = 0
    
    for bar_index, bar_key, fail_reasons in failed_bars:
        bar_data = bars_dict.get(bar_key)
        if not bar_data:
            logger.warning(f"⚠️  Bar {bar_index} not found in recommendations")
            continue
        
        original_pattern_id = bar_data['pattern']['pattern_id']
        
        # ステップ1: 微修正を試みる
        logger.info(f"  📍 bar_{bar_index}: Attempting micro-fix first...")
        fixed_bar = try_micro_fix(bar_data, fail_reasons)
        
        if fixed_bar:
            # 微修正成功（実際のKPI再検証は後続で実施）
            bar_data.update(fixed_bar)
            bar_data['micro_fix_applied'] = True
            micro_fixed_count += 1
            logger.info(f"  ✓ bar_{bar_index}: Micro-fix applied (fixes: {fixed_bar.get('micro_fixes_applied', [])})")
            continue
        
        # ステップ2: 微修正失敗 → Safe-Kit置換
        logger.info(f"  📍 bar_{bar_index}: Micro-fix unavailable, applying Safe-Kit replacement...")
        
        # 最適Safe-Kitパターン検索
        safe_replacement = find_safe_replacement(
            failed_bar={
                'density_target': bar_data['density_target'],
                'swing_target': bar_data['swing_target']
            },
            safe_patterns=safe_patterns,
            family_preference=bar_data.get('predicted_family', None),
            style_preference=song_style,
            song_tempo_bpm=song_tempo_bpm,
            used_patterns=used_patterns if preserve_diversity else None
        )
        
        # 置換
        bar_data['pattern']['pattern_id'] = safe_replacement['loop_id']
        bar_data['pattern']['family'] = safe_replacement['family']
        bar_data['pattern']['density'] = safe_replacement['density']
        bar_data['pattern']['swing'] = safe_replacement['swing']
        bar_data['pattern']['backbeat_strength'] = safe_replacement['backbeat_strength']
        bar_data['safe_kit_applied'] = True
        
        used_patterns.add(safe_replacement['loop_id'])
        replaced_count += 1
        
        logger.info(f"  ✓ bar_{bar_index}: {original_pattern_id} → {safe_replacement['loop_id']} (Safe-Kit)")
    
    logger.info(f"✅ Safe-Kit processing completed:")
    logger.info(f"   Micro-fixes: {micro_fixed_count}")
    logger.info(f"   Replacements: {replaced_count}")
    
    # メタデータ更新
    if 'metadata' not in recommendations:
        recommendations['metadata'] = {}
    recommendations['metadata']['safe_kit_fallback_applied'] = True
    recommendations['metadata']['safe_kit_micro_fixed_count'] = micro_fixed_count
    recommendations['metadata']['safe_kit_replaced_count'] = replaced_count
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description='Apply Safe-Kit Fallback to failed bars'
    )
    parser.add_argument(
        '--recommendations',
        required=True,
        help='Path to drums_recommendations.json'
    )
    parser.add_argument(
        '--kpi-report',
        required=True,
        help='Path to kpi_gate_report.json'
    )
    parser.add_argument(
        '--rhythm-features',
        default='data/patterns/rhythm_features_merged.parquet',
        help='Path to rhythm_features_merged.parquet'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for fixed recommendations'
    )
    parser.add_argument(
        '--preserve-diversity',
        action='store_true',
        default=True,
        help='Preserve pattern diversity (avoid duplicates)'
    )
    
    args = parser.parse_args()
    
    # Safe-Kit条件
    safe_kit_criteria = {
        'backbeat_strength': {'min': 0.3, 'max': 0.75},  # 0.9より低い
        'density': {'min': 3.0, 'max': 9.0},
        'swing': {'min': 0.0, 'max': 0.5}
    }
    
    logger.info("=" * 60)
    logger.info("Safe-Kit Fallback Application")
    logger.info("=" * 60)
    logger.info(f"Recommendations: {args.recommendations}")
    logger.info(f"KPI Report: {args.kpi_report}")
    logger.info(f"Rhythm Features: {args.rhythm_features}")
    logger.info("")
    
    # データ読み込み
    with open(args.recommendations, 'r') as f:
        recommendations = json.load(f)
    
    with open(args.kpi_report, 'r') as f:
        kpi_report = json.load(f)
    
    safe_patterns = load_safe_kit_patterns(
        rhythm_features_path=args.rhythm_features,
        safe_kit_criteria=safe_kit_criteria
    )
    
    logger.info("")
    
    # 曲メタデータ抽出（recommendations.jsonから）
    song_metadata = {}
    if 'metadata' in recommendations:
        meta = recommendations['metadata']
        song_metadata['tempo_bpm'] = meta.get('tempo_bpm', 120.0)
        song_metadata['style'] = meta.get('style')
        logger.info(f"📋 Song metadata extracted:")
        logger.info(f"   Tempo: {song_metadata['tempo_bpm']} BPM")
        logger.info(f"   Style: {song_metadata.get('style', 'N/A')}")
    else:
        logger.warning("⚠️  No metadata in recommendations, using defaults")
        song_metadata['tempo_bpm'] = 120.0
        song_metadata['style'] = None
    
    logger.info("")
    
    # Safe-Kit Fallback適用
    fixed_recommendations = apply_safe_kit_fallback(
        recommendations=recommendations,
        kpi_report=kpi_report,
        safe_patterns=safe_patterns,
        song_metadata=song_metadata,  # 追加
        preserve_diversity=args.preserve_diversity
    )
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(fixed_recommendations, f, indent=2)
    
    logger.info("")
    logger.info(f"✅ Saved fixed recommendations: {output_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Re-run KPI Gate: python3 scripts/kpi_gate.py --recommendations {args.output} ...")
    logger.info(f"  2. Generate MIDI: python3 scripts/generate_drums_midi.py --recommendations {args.output} ...")
    logger.info("")


if __name__ == '__main__':
    main()
