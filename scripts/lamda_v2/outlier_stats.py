#!/usr/bin/env python3
"""
外れ値統計計算（χ² 距離ベース）

**目的**:
- ローカル分布（1曲のpitch/dur/vel）vs グローバル分布（TOTALS）の差異を定量化
- 品質ゲート（SILVER/BRONZE判定）の素材を提供
- 過学習・異常値の自動検出

**活用箇所**:
- Stage2: outliers.{pitch, dur, vel} スコアをJSONに埋め込み
- CI/CD: χ² > 0.5 → BRONZE、> 0.3 → SILVER、≦0.3 → GOLD
- Sunoアレンジ: TOTALSベースの"普通に良い"分布へ誘導
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Sequence, Optional

def chi2_distance(
    local_hist: Sequence[float],
    global_hist: Sequence[float],
    eps: float = 1e-9
) -> float:
    """χ² 距離の計算（正規化済みヒストグラム間）
    
    Args:
        local_hist: ローカル分布（1曲のpitch/dur/vel）
        global_hist: グローバル分布（TOTALS）
        eps: ゼロ除算回避の微小値
    
    Returns:
        χ² 距離 (0.0〜∞)
        - 0.0 = 完全一致
        - < 0.1 = ほぼ同分布（GOLD）
        - 0.1〜0.3 = 軽微な差異（SILVER）
        - 0.3〜0.5 = 中程度の差異（BRONZE）
        - > 0.5 = 大きな差異（要レビュー）
    
    Formula:
        χ² = Σ (p_i - q_i)² / q_i
        where p_i, q_i are normalized probabilities
    """
    a = np.asarray(local_hist, dtype=float)
    b = np.asarray(global_hist, dtype=float) + eps
    
    # 正規化（確率分布化）
    a = a / (a.sum() + eps)
    b = b / b.sum()
    
    # χ² 距離
    chi2 = ((a - b) ** 2 / b).sum()
    
    return float(chi2)


def ks_distance(
    local_hist: Sequence[float],
    global_hist: Sequence[float]
) -> float:
    """Kolmogorov-Smirnov 距離（累積分布関数の最大差）
    
    Args:
        local_hist: ローカル分布（1曲のpitch/dur/vel）
        global_hist: グローバル分布（TOTALS）
    
    Returns:
        KS距離 (0.0〜1.0)
        - < 0.05 = ほぼ同分布
        - 0.05〜0.15 = 軽微な差異
        - > 0.15 = 大きな差異
    """
    a = np.asarray(local_hist, dtype=float)
    b = np.asarray(global_hist, dtype=float)
    
    # 累積分布関数（CDF）
    a_cdf = np.cumsum(a) / (a.sum() + 1e-9)
    b_cdf = np.cumsum(b) / (b.sum() + 1e-9)
    
    # 最大差
    ks = np.abs(a_cdf - b_cdf).max()
    
    return float(ks)


def hellinger_distance(
    local_hist: Sequence[float],
    global_hist: Sequence[float],
    eps: float = 1e-9
) -> float:
    """Hellinger 距離（確率分布間の類似度）
    
    Args:
        local_hist: ローカル分布（1曲のpitch/dur/vel）
        global_hist: グローバル分布（TOTALS）
        eps: ゼロ除算回避の微小値
    
    Returns:
        Hellinger距離 (0.0〜1.0)
        - 0.0 = 完全一致
        - < 0.1 = ほぼ同分布
        - > 0.3 = 大きな差異
    
    Formula:
        H = sqrt(1 - Σ sqrt(p_i * q_i))
    """
    a = np.asarray(local_hist, dtype=float)
    b = np.asarray(global_hist, dtype=float)
    
    # 正規化
    a = a / (a.sum() + eps)
    b = b / (b.sum() + eps)
    
    # Hellinger 距離
    bc = np.sqrt(a * b).sum()  # Bhattacharyya coefficient
    h = np.sqrt(1.0 - bc)
    
    return float(h)


def summarize_outliers(
    local: Dict[str, Sequence[float]],
    priors: Dict[str, Sequence[float]],
    method: str = "chi2"
) -> Dict[str, float]:
    """ローカル vs グローバル分布の外れ値スコアを計算
    
    Args:
        local: ローカル分布 {"pitch": [256], "dur": [256], "vel": [256]}
        priors: グローバル分布（TOTALS）
        method: 距離計算手法 ("chi2", "ks", "hellinger")
    
    Returns:
        {"pitch": 0.12, "dur": 0.08, "vel": 0.15}
        
    Example:
        >>> local_hist = {"pitch": [0]*60 + [100]*8 + [0]*188}  # C4周辺に集中
        >>> global_hist = {"pitch": [1]*256}  # 均等分布
        >>> scores = summarize_outliers(local_hist, global_hist)
        >>> scores["pitch"]  # → 0.45（中程度の差異）
    """
    out = {}
    
    dist_func = {
        "chi2": chi2_distance,
        "ks": ks_distance,
        "hellinger": hellinger_distance
    }.get(method, chi2_distance)
    
    for key in ("pitch", "dur", "vel"):
        if key in local and key in priors:
            out[key] = dist_func(local[key], priors[key])
    
    return out


def quality_gate(
    outlier_scores: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None
) -> str:
    """外れ値スコアから品質ゲートを判定
    
    Args:
        outlier_scores: {"pitch": 0.12, "dur": 0.08, "vel": 0.15}
        thresholds: カスタム閾値 {"gold": 0.1, "silver": 0.3, "bronze": 0.5}
    
    Returns:
        "GOLD", "SILVER", "BRONZE", or "REJECT"
    """
    if thresholds is None:
        thresholds = {
            "gold": 0.1,
            "silver": 0.3,
            "bronze": 0.5
        }
    
    # 最大スコア（最も外れている次元）
    max_score = max(outlier_scores.values()) if outlier_scores else 0.0
    
    if max_score <= thresholds["gold"]:
        return "GOLD"
    elif max_score <= thresholds["silver"]:
        return "SILVER"
    elif max_score <= thresholds["bronze"]:
        return "BRONZE"
    else:
        return "REJECT"


# ========================================
# CLI テスト用
# ========================================
if __name__ == "__main__":
    import argparse
    import json
    
    ap = argparse.ArgumentParser(description="Outlier Stats Test")
    ap.add_argument("--local-json", help="Local histogram JSON")
    ap.add_argument("--global-json", help="Global histogram JSON (TOTALS)")
    ap.add_argument("--method", default="chi2", choices=["chi2", "ks", "hellinger"])
    args = ap.parse_args()
    
    # テストデータ生成
    if not args.local_json or not args.global_json:
        print("🧪 Running synthetic test...")
        
        # 均等分布（TOTALS想定）
        global_hist = {"pitch": [1.0] * 256, "dur": [1.0] * 256, "vel": [1.0] * 256}
        
        # C4周辺に集中（異常値想定）
        local_hist = {
            "pitch": [0.0] * 60 + [100.0] * 8 + [0.0] * 188,
            "dur": [50.0] * 10 + [1.0] * 246,
            "vel": [0.0] * 64 + [80.0] * 20 + [0.0] * 172
        }
        
    else:
        with open(args.local_json) as f:
            local_hist = json.load(f)
        with open(args.global_json) as f:
            global_hist = json.load(f)
    
    # スコア計算
    scores = summarize_outliers(local_hist, global_hist, method=args.method)
    gate = quality_gate(scores)
    
    print(f"📊 Outlier Scores ({args.method}):")
    print(json.dumps(scores, indent=2))
    print(f"\n🎯 Quality Gate: {gate}")
    
    # 解釈ガイド
    print("\n📖 Interpretation:")
    for key, score in scores.items():
        if score < 0.1:
            level = "GOLD (excellent)"
        elif score < 0.3:
            level = "SILVER (good)"
        elif score < 0.5:
            level = "BRONZE (acceptable)"
        else:
            level = "REJECT (anomaly)"
        print(f"  {key:8s}: {score:6.3f} → {level}")
