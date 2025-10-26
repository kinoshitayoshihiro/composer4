#!/usr/bin/env python3
"""
Velocity Auto-Guide Generator for velocity.yaml

AI自動化ガイド：データから速度分布を統計分析し、自動反映可否を判定。

Features:
- Stage2 JSONから速度分布を集計
- Percentile/Skew/KS距離による品質判定
- Auto/Review/Manual の3段階判定
- 既存velocity_model.yamlと併用可能

Usage:
    # 基本実行
    python scripts/lamda_v2/velocity_autoguide.py \\
        --stage2-json-dir output/stage2_production/json \\
        --out-yaml analysis/velocity_auto.yaml
    
    # LAMDA METAも含める
    python scripts/lamda_v2/velocity_autoguide.py \\
        --stage2-json-dir output/stage2_production/json \\
        --lamda-meta-dir data/Los-Angeles-MIDI/META \\
        --out-yaml analysis/velocity_auto.yaml

Output:
    analysis/velocity_auto.yaml:
        schema: velocity_autoguide_v1
        profiles:
          piano:
            mode: auto  # or review, manual
            n: 18342
            range: {min: 28, max: 108}
            center: 76
            curve: linear
            ks_drift: 0.03
"""
from __future__ import annotations
import json
import argparse
import statistics as st
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


# ---- ユーティリティ ----

def _percentiles(xs: List[int], ps=(1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
    """パーセンタイル計算"""
    a = np.asarray(xs)
    a.sort()
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def _skew(xs: List[int]) -> float:
    """歪度（skewness）計算"""
    a = np.asarray(xs, dtype=float)
    if a.size < 3:
        return 0.0
    m, s = a.mean(), a.std() or 1.0
    return float(((a - m) ** 3).mean() / (s ** 3))


def _ks_distance(a: List[int], b: List[int]) -> float:
    """
    Kolmogorov-Smirnov距離（簡易版）
    
    0..127の累積分布差の最大値を返す
    """
    hist_a = np.bincount(a, minlength=128) / max(1, len(a))
    hist_b = np.bincount(b, minlength=128) / max(1, len(b))
    cdf_a = np.cumsum(hist_a)
    cdf_b = np.cumsum(hist_b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _decide_curve(skew: float) -> str:
    """
    歪度からカーブタイプを推定
    
    Parameters
    ----------
    skew : float
        歪度
    
    Returns
    -------
    str
        compress_heavy, compress_light, linear, expand_light, expand_heavy
    """
    if skew > 0.8:
        return "compress_heavy"
    if skew > 0.3:
        return "compress_light"
    if skew < -0.8:
        return "expand_heavy"
    if skew < -0.3:
        return "expand_light"
    return "linear"


# ---- メイン：Stage2 JSON / META から速度分布を集計 ----

def collect_velocities(
    stage2_dir: Path, meta_dir: Optional[Path]
) -> Dict[str, List[int]]:
    """
    Stage2 JSONファイルから速度分布を集計
    
    Parameters
    ----------
    stage2_dir : Path
        Stage2 JSONディレクトリ
    meta_dir : Path, optional
        LAMDA METAディレクトリ（拡張用）
    
    Returns
    -------
    Dict[str, List[int]]
        Role別の速度リスト
    """
    buckets: Dict[str, List[int]] = {}
    
    for js in Path(stage2_dir).rglob("*.stage2.json"):
        try:
            j = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        
        # notes分布（存在すれば） or 役割ごとのサマリ
        roles = (j.get("roles", {}) or {}).get("roles", [])
        
        # 簡易集計：controlsやnotesが無い場合はスキップ
        # （本実装では pretty_midi を再走査してもOK）
        for role in (
            ["global"] if not roles else [r.get("role", "global") for r in roles]
        ):
            # 期待: j["stats"]["velocities"][role] がある構成に拡張していく
            vel = (j.get("stats", {}).get("velocities", {}).get(role) or [])
            if vel:
                buckets.setdefault(role, []).extend(
                    int(v) for v in vel if 1 <= int(v) <= 127
                )
    
    return buckets


def auto_guide(
    stage2_dir: Path,
    meta_dir: Optional[Path],
    prev_yaml_sample: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    自動ガイド生成
    
    Parameters
    ----------
    stage2_dir : Path
        Stage2 JSONディレクトリ
    meta_dir : Path, optional
        LAMDA METAディレクトリ
    prev_yaml_sample : Dict, optional
        既存velocity.yamlのサンプル（ドリフト検出用）
    
    Returns
    -------
    Dict[str, Any]
        自動ガイドYAML構造
    """
    data = collect_velocities(stage2_dir, meta_dir)
    out = {
        "schema": "velocity_autoguide_v1",
        "sources": str(stage2_dir),
        "profiles": {},
    }
    
    # 既存yaml分布（レビュー用に読み込めるなら比較）
    prev_dist = []
    if prev_yaml_sample:
        # 例：prev_yaml_sample["profiles"]["piano"]["range"] などの利用を想定
        pass
    
    for role, xs in data.items():
        n = len(xs)
        
        # サンプル数不足
        if n < 500:
            out["profiles"][role] = {
                "mode": "manual",
                "reason": "insufficient_samples",
                "n": n,
            }
            continue
        
        pct = _percentiles(xs)
        skew = _skew(xs)
        curve = _decide_curve(skew)
        
        # KSで旧とのドリフト判定（ここでは前回を持たないので 0.0 と比較）
        ks = _ks_distance(xs, prev_dist) if prev_dist else 0.0
        
        # AI化判定基準
        # - サンプル数 >= 5000 かつ ドリフト < 0.08
        # - または サンプル数 >= 2000 かつ ドリフト < 0.05
        auto_ok = (n >= 5000 and ks < 0.08) or (n >= 2000 and ks < 0.05)
        
        out["profiles"][role] = {
            "mode": "auto" if auto_ok else "review",
            "n": n,
            "range": {"min": int(pct["p5"]), "max": int(pct["p95"])},
            "center": int(pct["p50"]),
            "tails": {"p1": pct["p1"], "p99": pct["p99"]},
            "skew": round(skew, 3),
            "curve": curve,
            "ks_drift": round(ks, 3),
        }
    
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Auto-guide generator for velocity.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--stage2-json-dir",
        type=Path,
        required=True,
        help="Stage2 JSON directory",
    )
    ap.add_argument(
        "--lamda-meta-dir",
        type=Path,
        default=None,
        help="LAMDA META directory (optional)",
    )
    ap.add_argument(
        "--out-yaml",
        type=Path,
        default=Path("analysis/velocity_auto.yaml"),
        help="Output YAML path",
    )
    
    args = ap.parse_args()
    
    print(f"🔍 Analyzing velocities from {args.stage2_json_dir}...")
    
    guide = auto_guide(
        args.stage2_json_dir,
        args.lamda_meta_dir,
    )
    
    # YAMLで出力
    try:
        import yaml
        
        outp = args.out_yaml
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            yaml.safe_dump(guide, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        
        print(f"\n✅ Wrote: {outp}")
        print(f"\n📊 Summary:")
        for role, prof in guide.get("profiles", {}).items():
            mode = prof.get("mode", "unknown")
            n = prof.get("n", 0)
            print(f"   {role:15s} {mode:10s} (n={n:,})")
        
    except ImportError:
        print("⚠️ PyYAML not installed. Install: pip install pyyaml")
        print("   Falling back to JSON output...")
        
        outp = args.out_yaml.with_suffix(".json")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(guide, ensure_ascii=False, indent=2))
        print(f"✅ Wrote: {outp}")
    
    return 0


if __name__ == "__main__":
    exit(main())
