#!/usr/bin/env python3
"""
EmotionAI Numeric Derivation - Phase 125
bars.parquet → energy/valence列追加

Energy推定（3段階フォールバック）:
  1. bars.parquet.energy_curve（セクション中央値 → 0..1正規化）
  2. stem_features.parquet（loudness_db 0.4 + drums_active 0.2 + hat_density 0.1 + fill_likelihood 0.1 + chord_tension 0.2加重平均）
  3. sectionデフォルト（emotion_profile.yaml）

Valence推定（和声ベース）:
  - ハーモニーポーラリティ: harmony_ai_report.json（maj/maj7→+0.35, dom7→+0.15, min/m7→-0.25, dim/m7b5→-0.45, sus→0.00）
  - Cadence安定度: 完全終止+0.20, 偽終止+0.05, 不安定-0.10
  - Spectral Brightness: stem_features.loudness_db → 0..1正規化 → ±0.10
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def normalize_to_range(
    values: pd.Series,
    min_val: float,
    max_val: float,
    clip: bool = True
) -> pd.Series:
    """
    値を指定範囲に正規化
    
    Args:
        values: 正規化対象のSeries
        min_val: 正規化後の最小値
        max_val: 正規化後の最大値
        clip: 範囲外をクリップするか
    
    Returns:
        正規化後のSeries
    """
    v_min = values.min()
    v_max = values.max()
    
    if v_max == v_min:
        return pd.Series([0.5] * len(values), index=values.index)
    
    normalized = min_val + (values - v_min) / (v_max - v_min) * (max_val - min_val)
    
    if clip:
        normalized = normalized.clip(min_val, max_val)
    
    return normalized


def estimate_energy_from_bars(
    bars: pd.DataFrame,
    stem_features: Optional[pd.DataFrame],
    emotion_profile: Dict[str, Any]
) -> pd.Series:
    """
    Energy推定（3段階フォールバック）
    
    Args:
        bars: bars.parquet DataFrame
        stem_features: stem_features.parquet DataFrame（None可）
        emotion_profile: emotion_profile.yaml辞書
    
    Returns:
        energy Series（0.0..1.0）
    
    Priority:
        1. bars.energy_curve（存在すれば0..1正規化）
        2. stem_features加重平均（loudness_db 0.4 + drums_active 0.2 + hat_density 0.1 + fill_likelihood 0.1）
        3. sectionデフォルト（emotion_profile.yaml）
    """
    # 1. bars.energy_curve優先
    if "energy_curve" in bars.columns:
        energy = normalize_to_range(bars["energy_curve"], 0.0, 1.0)
        return energy
    
    # 2. stem_features加重平均
    if stem_features is not None:
        weights = emotion_profile.get("weights", {}).get("energy", {})
        
        loudness_w = weights.get("loudness_db", 0.4)
        drums_w = weights.get("drums_active", 0.2)
        hat_w = weights.get("hat_density", 0.1)
        fill_w = weights.get("fill_likelihood", 0.1)
        
        energy = pd.Series(0.0, index=bars.index)
        
        if "loudness_db" in stem_features.columns:
            loudness_norm = normalize_to_range(stem_features["loudness_db"], 0.0, 1.0)
            energy += loudness_norm * loudness_w
        
        if "drums_active" in stem_features.columns:
            energy += stem_features["drums_active"] * drums_w
        
        if "hat_density" in stem_features.columns:
            hat_norm = normalize_to_range(stem_features["hat_density"], 0.0, 1.0)
            energy += hat_norm * hat_w
        
        if "fill_likelihood" in stem_features.columns:
            fill_norm = normalize_to_range(stem_features["fill_likelihood"], 0.0, 1.0)
            energy += fill_norm * fill_w
        
        # 正規化（加重和 → 0..1範囲）
        energy = energy.clip(0.0, 1.0)
        return energy
    
    # 3. sectionデフォルト
    sections = emotion_profile.get("sections", {})
    defaults = emotion_profile.get("defaults", {})
    default_energy = defaults.get("energy", 0.5)
    
    energy = []
    for _, row in bars.iterrows():
        sec_label = row.get("section_label", "")
        sec_data = sections.get(sec_label, {})
        energy.append(sec_data.get("energy", default_energy))
    
    return pd.Series(energy, index=bars.index)


def estimate_valence_from_harmony(
    bars: pd.DataFrame,
    harmony_ai_path: Optional[Path],
    stem_features: Optional[pd.DataFrame]
) -> pd.Series:
    """
    Valence推定（和声ベース）
    
    Args:
        bars: bars.parquet DataFrame
        harmony_ai_path: harmony_ai_report.json パス
        stem_features: stem_features.parquet DataFrame（None可）
    
    Returns:
        valence Series（-1.0..+1.0）
    
    Components:
        - ハーモニーポーラリティ: maj/maj7→+0.35, dom7→+0.15, min/m7→-0.25, dim/m7b5→-0.45, sus→0.00
        - Cadence安定度: 完全終止+0.20, 偽終止+0.05, 不安定-0.10
        - Spectral Brightness: stem_features.loudness_db → ±0.10
    """
    # 初期値（ニュートラル）
    valence = pd.Series(0.0, index=bars.index)
    
    # 1. ハーモニーポーラリティ（harmony_ai_report.json）
    if harmony_ai_path and harmony_ai_path.exists():
        try:
            with open(harmony_ai_path, "r", encoding="utf-8") as f:
                harmony = json.load(f)
            
            # セクション別コード分析
            polarity_map = {
                "maj": 0.35, "maj7": 0.35, "M": 0.35, "M7": 0.35,
                "dom7": 0.15, "7": 0.15,
                "min": -0.25, "m": -0.25, "m7": -0.25, "min7": -0.25,
                "dim": -0.45, "m7b5": -0.45, "dim7": -0.45,
                "sus": 0.0, "sus4": 0.0, "sus2": 0.0
            }
            
            for section in harmony.get("sections", []):
                sec_label = section.get("section_label", "")
                chords = section.get("chords", [])
                
                # セクション内コードのポーラリティ平均
                polarities = []
                for chord in chords:
                    chord_type = chord.get("chord_type", "")
                    polarity = polarity_map.get(chord_type, 0.0)
                    polarities.append(polarity)
                
                if polarities:
                    avg_polarity = np.mean(polarities)
                    
                    # 該当barインデックス
                    bar_indices = bars[bars["section_label"] == sec_label].index
                    valence.loc[bar_indices] += avg_polarity
        
        except Exception as e:
            print(f"⚠️  harmony_ai_report.json読み込みエラー: {e}")
    
    # 2. Spectral Brightness（stem_features.loudness_db → ±0.10）
    if stem_features is not None and "loudness_db" in stem_features.columns:
        brightness = normalize_to_range(stem_features["loudness_db"], -0.10, 0.10)
        valence += brightness
    
    # 3. Cadence安定度（将来拡張: harmony_ai_report.json cadence分析）
    # 現状は省略（harmony_ai_report.json にcadence情報がない場合）
    
    # Clip to -1.0..+1.0
    valence = valence.clip(-1.0, 1.0)
    
    return valence


def derive_emotion_numeric(
    bars_path: Path,
    stem_features_path: Optional[Path],
    harmony_ai_path: Optional[Path],
    emotion_profile_path: Path,
    output_bars_path: Path,
    output_profile_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    bars.parquet → energy/valence列追加
    
    Args:
        bars_path: bars.parquet パス
        stem_features_path: stem_features.parquet パス（None可）
        harmony_ai_path: harmony_ai_report.json パス（None可）
        emotion_profile_path: emotion_profile.yaml パス
        output_bars_path: 出力bars.parquet パス
        output_profile_path: 出力emotion_profile_auto.yaml パス（None可）
    
    Returns:
        メトリクス辞書
    """
    # bars.parquet読み込み
    bars = pd.read_parquet(bars_path)
    
    # stem_features読み込み（存在すれば）
    stem_features = None
    if stem_features_path and stem_features_path.exists():
        stem_features = pd.read_parquet(stem_features_path)
    
    # emotion_profile読み込み
    with open(emotion_profile_path, "r", encoding="utf-8") as f:
        emotion_profile = yaml.safe_load(f)
    
    # Energy推定
    energy = estimate_energy_from_bars(bars, stem_features, emotion_profile)
    
    # Valence推定
    valence = estimate_valence_from_harmony(bars, harmony_ai_path, stem_features)
    
    # bars.parquetにenergy/valence列追加
    bars["energy"] = energy
    bars["valence"] = valence
    
    # 出力
    bars.to_parquet(output_bars_path, index=False)
    
    # セクション別統計
    section_stats = {}
    for sec_label in bars["section_label"].unique():
        sec_bars = bars[bars["section_label"] == sec_label]
        section_stats[sec_label] = {
            "energy_median": float(sec_bars["energy"].median()),
            "energy_range": [float(sec_bars["energy"].min()), float(sec_bars["energy"].max())],
            "valence_median": float(sec_bars["valence"].median()),
            "valence_range": [float(sec_bars["valence"].min()), float(sec_bars["valence"].max())]
        }
    
    # emotion_profile_auto.yaml生成（オプション）
    if output_profile_path:
        auto_profile = {
            "version": 1,
            "metadata": {
                "name": "Auto-generated Emotion Profile",
                "generated_from": str(bars_path),
                "phase": 125
            },
            "sections": {}
        }
        
        for sec_label, stats in section_stats.items():
            auto_profile["sections"][sec_label] = {
                "energy": stats["energy_median"],
                "valence": stats["valence_median"],
                "description": f"Auto-derived from {sec_label}"
            }
        
        with open(output_profile_path, "w", encoding="utf-8") as f:
            yaml.dump(auto_profile, f, allow_unicode=True, sort_keys=False)
    
    # メトリクス
    metrics = {
        "bars_total": len(bars),
        "energy": {
            "min": float(bars["energy"].min()),
            "max": float(bars["energy"].max()),
            "mean": float(bars["energy"].mean()),
            "median": float(bars["energy"].median())
        },
        "valence": {
            "min": float(bars["valence"].min()),
            "max": float(bars["valence"].max()),
            "mean": float(bars["valence"].mean()),
            "median": float(bars["valence"].median())
        },
        "section_stats": section_stats
    }
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EmotionAI Numeric Derivation")
    parser.add_argument("--bars", type=Path, required=True,
                        help="bars.parquet path")
    parser.add_argument("--stem-features", type=Path,
                        help="stem_features.parquet path (optional)")
    parser.add_argument("--harmony-ai", type=Path,
                        help="harmony_ai_report.json path (optional)")
    parser.add_argument("--emotion-profile", type=Path, required=True,
                        help="emotion_profile.yaml path")
    parser.add_argument("--out-bars", type=Path, required=True,
                        help="Output bars.parquet path")
    parser.add_argument("--out-profile", type=Path,
                        help="Output emotion_profile_auto.yaml path (optional)")
    parser.add_argument("--report", type=Path,
                        help="Output metrics JSON path (optional)")
    
    args = parser.parse_args()
    
    try:
        metrics = derive_emotion_numeric(
            bars_path=args.bars,
            stem_features_path=args.stem_features,
            harmony_ai_path=args.harmony_ai,
            emotion_profile_path=args.emotion_profile,
            output_bars_path=args.out_bars,
            output_profile_path=args.out_profile
        )
        
        print(f"✅ EmotionAI Numeric Derivation Complete:")
        print(f"   Bars total: {metrics['bars_total']}")
        print(f"   Energy: min={metrics['energy']['min']:.3f}, max={metrics['energy']['max']:.3f}, mean={metrics['energy']['mean']:.3f}")
        print(f"   Valence: min={metrics['valence']['min']:.3f}, max={metrics['valence']['max']:.3f}, mean={metrics['valence']['mean']:.3f}")
        print(f"   Output bars: {args.out_bars}")
        
        if args.out_profile:
            print(f"   Output profile: {args.out_profile}")
        
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"   Metrics report: {args.report}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
