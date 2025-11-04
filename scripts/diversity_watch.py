#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diversity_watch.py
--------------------------------------------------
KPI過適合防止のため、音楽的多様性指標（jSymbolic参照）を監視し、
前版との差分をレポートする。

Phase E拡張: ジャンル/テンポ帯別のz-score正規化による誤検出削減

監視指標:
  - Pitch Range (P系): 音域の広さ
  - IOI Variance (R系): リズムの多様性
  - Velocity Variance (D系): ダイナミクスの多様性
  - Polyphony Max (H系): 和声の複雑さ

Usage:
  python3 scripts/diversity_watch.py \
      --current song_packages/suno_project/song_001/full_arrangement.mid \
      --baseline song_packages/suno_project/song_001/full_arrangement_baseline.mid \
      --output song_packages/suno_project/song_001/diversity_report.json \
      --genre rock \
      --tempo-bpm 74.677

Output:
  - diversity_report.json: 現在/ベースライン比較レポート
  - 差分が大きい指標（z-score > 1.5σ）を警告
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import pretty_midi
except ImportError:
    print("❌ pretty_midi が見つかりません。`pip install pretty_midi` を実行してください。")
    sys.exit(1)

import numpy as np


# ジャンル/テンポ帯別の基準値（平均と標準偏差）
# 実測データから推定した参考値（Phase E初期値）
GENRE_TEMPO_NORMS = {
    # ジャンル: {テンポ帯: {"mean": {...}, "std": {...}}}
    "rock": {
        "slow": {  # < 100 BPM
            "mean": {
                "pitch_range": 55.0,
                "ioi_variance": 0.10,
                "velocity_variance": 220.0,
                "polyphony_max": 7.0,
            },
            "std": {
                "pitch_range": 8.0,
                "ioi_variance": 0.03,
                "velocity_variance": 40.0,
                "polyphony_max": 1.5,
            },
        },
        "mid": {  # 100-140 BPM
            "mean": {
                "pitch_range": 60.0,
                "ioi_variance": 0.12,
                "velocity_variance": 250.0,
                "polyphony_max": 8.0,
            },
            "std": {
                "pitch_range": 10.0,
                "ioi_variance": 0.04,
                "velocity_variance": 50.0,
                "polyphony_max": 2.0,
            },
        },
        "fast": {  # > 140 BPM
            "mean": {
                "pitch_range": 58.0,
                "ioi_variance": 0.15,
                "velocity_variance": 240.0,
                "polyphony_max": 7.5,
            },
            "std": {
                "pitch_range": 9.0,
                "ioi_variance": 0.05,
                "velocity_variance": 45.0,
                "polyphony_max": 1.8,
            },
        },
    },
    "pop": {
        "slow": {
            "mean": {
                "pitch_range": 50.0,
                "ioi_variance": 0.08,
                "velocity_variance": 200.0,
                "polyphony_max": 6.5,
            },
            "std": {
                "pitch_range": 7.0,
                "ioi_variance": 0.02,
                "velocity_variance": 35.0,
                "polyphony_max": 1.2,
            },
        },
        "mid": {
            "mean": {
                "pitch_range": 55.0,
                "ioi_variance": 0.10,
                "velocity_variance": 230.0,
                "polyphony_max": 7.5,
            },
            "std": {
                "pitch_range": 8.0,
                "ioi_variance": 0.03,
                "velocity_variance": 45.0,
                "polyphony_max": 1.5,
            },
        },
        "fast": {
            "mean": {
                "pitch_range": 53.0,
                "ioi_variance": 0.12,
                "velocity_variance": 220.0,
                "polyphony_max": 7.0,
            },
            "std": {
                "pitch_range": 7.5,
                "ioi_variance": 0.04,
                "velocity_variance": 40.0,
                "polyphony_max": 1.4,
            },
        },
    },
    "default": {  # ジャンル不明時のフォールバック
        "slow": {
            "mean": {
                "pitch_range": 55.0,
                "ioi_variance": 0.10,
                "velocity_variance": 220.0,
                "polyphony_max": 7.0,
            },
            "std": {
                "pitch_range": 10.0,
                "ioi_variance": 0.04,
                "velocity_variance": 50.0,
                "polyphony_max": 2.0,
            },
        },
        "mid": {
            "mean": {
                "pitch_range": 58.0,
                "ioi_variance": 0.12,
                "velocity_variance": 240.0,
                "polyphony_max": 7.5,
            },
            "std": {
                "pitch_range": 10.0,
                "ioi_variance": 0.04,
                "velocity_variance": 50.0,
                "polyphony_max": 2.0,
            },
        },
        "fast": {
            "mean": {
                "pitch_range": 56.0,
                "ioi_variance": 0.14,
                "velocity_variance": 230.0,
                "polyphony_max": 7.2,
            },
            "std": {
                "pitch_range": 10.0,
                "ioi_variance": 0.05,
                "velocity_variance": 50.0,
                "polyphony_max": 2.0,
            },
        },
    },
}


def get_tempo_bucket(tempo_bpm: float) -> str:
    """テンポをslow/mid/fastに分類"""
    if tempo_bpm < 100:
        return "slow"
    elif tempo_bpm <= 140:
        return "mid"
    else:
        return "fast"


def get_genre_tempo_norm(genre: str, tempo_bpm: float) -> Dict[str, Dict[str, float]]:
    """ジャンル/テンポ帯の基準値を取得"""
    genre_key = genre.lower() if genre and genre.lower() in GENRE_TEMPO_NORMS else "default"
    tempo_bucket = get_tempo_bucket(tempo_bpm)
    return GENRE_TEMPO_NORMS[genre_key][tempo_bucket]


def extract_diversity_features(midi_path: Path) -> Dict[str, float]:
    """
    jSymbolic参照の多様性指標を抽出（P/R/D/H系の代表4指標）
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # 全ノート集約
    all_notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)

    if not all_notes:
        return {
            "pitch_range": 0.0,
            "ioi_variance": 0.0,
            "velocity_variance": 0.0,
            "polyphony_max": 0.0,
        }

    # P系: Pitch Range（音域）
    pitches = [n.pitch for n in all_notes]
    pitch_range = float(max(pitches) - min(pitches))

    # R系: IOI Variance（音符間隔の分散）
    onsets = sorted([n.start for n in all_notes])
    iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    ioi_variance = float(np.var(iois)) if len(iois) > 1 else 0.0

    # D系: Velocity Variance（ベロシティの分散）
    velocities = [n.velocity for n in all_notes]
    velocity_variance = float(np.var(velocities)) if len(velocities) > 1 else 0.0

    # H系: Polyphony Max（最大同時発音数）
    # 簡易実装：全時間をスキャンせず、ノート密度の高い区間をサンプル
    time_bins = np.linspace(0, pm.get_end_time(), 100)
    polyphony_samples = []
    for t in time_bins:
        active = sum(1 for n in all_notes if n.start <= t < n.end)
        polyphony_samples.append(active)
    polyphony_max = float(max(polyphony_samples)) if polyphony_samples else 0.0

    return {
        "pitch_range": pitch_range,
        "ioi_variance": ioi_variance,
        "velocity_variance": velocity_variance,
        "polyphony_max": polyphony_max,
    }


def compare_features(
    current: Dict[str, float],
    baseline: Dict[str, float],
    genre: Optional[str] = None,
    tempo_bpm: Optional[float] = None,
    z_score_threshold: float = 1.5,
) -> Dict[str, Any]:
    """
    現在とベースラインの差分を計算し、z-score閾値超過を警告

    Phase E拡張: ジャンル/テンポ帯別の基準値でz-score正規化
    - z-score = (current - baseline) / std
    - |z-score| > 1.5σ で警告（誤検出削減）
    """
    diffs = {}
    warnings = []

    # ジャンル/テンポ帯の基準値取得
    if genre and tempo_bpm:
        norms = get_genre_tempo_norm(genre, tempo_bpm)
        use_zscore = True
    else:
        # フォールバック: 従来の±20%判定
        norms = None
        use_zscore = False

    for key in current.keys():
        curr_val = current[key]
        base_val = baseline.get(key, 0.0)
        absolute_change = curr_val - base_val

        # ゼロ除算回避（パーセント変化）
        if base_val == 0.0:
            if curr_val == 0.0:
                pct_change = 0.0
            else:
                pct_change = float("inf")
        else:
            pct_change = absolute_change / base_val

        diffs[key] = {
            "current": curr_val,
            "baseline": base_val,
            "absolute_change": absolute_change,
            "percent_change": pct_change,
        }

        # z-score判定（Phase E拡張）
        if use_zscore and norms:
            std = norms["std"].get(key, 1.0)  # デフォルト1.0（安全策）
            z_score = absolute_change / std if std > 0 else 0.0
            diffs[key]["z_score"] = z_score

            if abs(z_score) > z_score_threshold:
                warnings.append(
                    {
                        "metric": key,
                        "current": curr_val,
                        "baseline": base_val,
                        "z_score": z_score,
                        "threshold": z_score_threshold,
                        "message": f"{key} のz-score {z_score:.2f}σ が閾値 ±{z_score_threshold}σ を超えました（ジャンル: {genre}, テンポ: {tempo_bpm:.1f} BPM）",
                    }
                )
        else:
            # フォールバック: 従来の±20%判定
            threshold = 0.20
            if abs(pct_change) > threshold and pct_change != float("inf"):
                warnings.append(
                    {
                        "metric": key,
                        "current": curr_val,
                        "baseline": base_val,
                        "percent_change": pct_change,
                        "message": f"{key} が {pct_change*100:.1f}% 変化しました（閾値: ±{threshold*100:.0f}%）",
                    }
                )

    return {
        "diffs": diffs,
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser(description="音楽的多様性監視（KPI過適合防止）")
    ap.add_argument("--current", type=Path, required=True, help="現在のMIDIファイル")
    ap.add_argument(
        "--baseline", type=Path, default=None, help="ベースライン比較用MIDIファイル（任意）"
    )
    ap.add_argument(
        "--output", type=Path, default=Path("diversity_report.json"), help="レポート出力先"
    )
    ap.add_argument(
        "--genre", type=str, default=None, help="ジャンル（rock/pop等）。z-score正規化に使用"
    )
    ap.add_argument(
        "--tempo-bpm", type=float, default=None, help="テンポ（BPM）。z-score正規化に使用"
    )
    ap.add_argument(
        "--z-score-threshold", type=float, default=1.5, help="z-score警告閾値（既定1.5σ）"
    )
    args = ap.parse_args()

    # 現在の指標抽出
    print(f"📊 Extracting diversity features from: {args.current}")
    current_features = extract_diversity_features(args.current)

    # ベースライン比較（任意）
    comparison = None
    if args.baseline and args.baseline.exists():
        print(f"📊 Comparing with baseline: {args.baseline}")
        baseline_features = extract_diversity_features(args.baseline)

        # Phase E: ジャンル/テンポ帯別z-score正規化
        if args.genre and args.tempo_bpm:
            print(
                f"🎯 Using z-score normalization (genre={args.genre}, tempo={args.tempo_bpm:.1f} BPM)"
            )

        comparison = compare_features(
            current_features,
            baseline_features,
            genre=args.genre,
            tempo_bpm=args.tempo_bpm,
            z_score_threshold=args.z_score_threshold,
        )

    # レポート構築
    report = {
        "current_midi": str(args.current),
        "baseline_midi": str(args.baseline) if args.baseline else None,
        "genre": args.genre,
        "tempo_bpm": args.tempo_bpm,
        "z_score_threshold": args.z_score_threshold,
        "features": current_features,
        "comparison": comparison,
    }

    # JSON出力
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Diversity report saved: {args.output}")

    # 警告表示
    if comparison and comparison["warnings"]:
        print("\n⚠️  Diversity Warnings:")
        for w in comparison["warnings"]:
            print(f"   - {w['message']}")
        print()

    # サマリー表示
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Diversity Features:")
    for key, val in current_features.items():
        print(f"  {key:20s}: {val:8.2f}")

    if comparison:
        print("\nChange from Baseline:")
        for key, diff in comparison["diffs"].items():
            pct = diff["percent_change"]
            sign = "+" if pct > 0 else ""
            if pct == float("inf"):
                print(f"  {key:20s}: ∞ (baseline was 0)")
            else:
                print(f"  {key:20s}: {sign}{pct*100:6.1f}%")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
