#!/usr/bin/env python3
"""
extract_emotion_profile.py - bars.parquetからemotion特徴抽出

bars.parquetのenergy_curve、accent_score_target、density_targetから
valence（感情価）とarousal（覚醒度）を推定し、emotion_profile.jsonを生成します。

Usage:
    python3 scripts/extract_emotion_profile.py \
      --bars song_packages/<project>/<song>/bars.parquet \
      --output song_packages/<project>/<song>/emotion_profile.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def estimate_valence_arousal(
    energy_mean: float,
    energy_std: float,
    accent_mean: float,
    density_mean: float,
    swing_mean: float,
) -> Dict[str, float]:
    """
    energy_curve等の統計値からvalence/arousalを推定

    Valence（感情価）:
      - energy高い + swing高い → 陽性（0.7+）
      - energy低い + swing低い → 陰性（0.3-）

    Arousal（覚醒度）:
      - energy高い + density高い → 高覚醒（0.8+）
      - energy低い + density低い → 低覚醒（0.2-）
    """
    # Valence推定（0.0-1.0）
    valence = 0.5  # neutral baseline

    # energy高い → 陽性寄り
    if energy_mean > 0.6:
        valence += 0.2
    elif energy_mean < 0.4:
        valence -= 0.2

    # swing高い → 陽性寄り（グルーヴ感）
    if swing_mean > 0.3:
        valence += 0.15
    elif swing_mean < 0.15:
        valence -= 0.1

    # accent高い → 陽性寄り（力強さ）
    if accent_mean > 0.6:
        valence += 0.1

    # energy変動小さい → 安定 → やや陽性
    if energy_std < 0.1:
        valence += 0.05

    valence = np.clip(valence, 0.0, 1.0)

    # Arousal推定（0.0-1.0）
    arousal = 0.5  # neutral baseline

    # energy高い → 高覚醒
    if energy_mean > 0.6:
        arousal += 0.3
    elif energy_mean < 0.4:
        arousal -= 0.2

    # density高い → 高覚醒
    if density_mean > 7.0:
        arousal += 0.2
    elif density_mean < 4.0:
        arousal -= 0.2

    # accent高い → 高覚醒
    if accent_mean > 0.6:
        arousal += 0.1

    # energy変動大きい → 高覚醒
    if energy_std > 0.15:
        arousal += 0.1

    arousal = np.clip(arousal, 0.0, 1.0)

    return {"valence": float(valence), "arousal": float(arousal)}


def classify_emotion(valence: float, arousal: float) -> str:
    """
    Valence/Arousalからemotion labelを分類

    Russell's Circumplex Model:
      - High Arousal + High Valence → Excited/Happy
      - High Arousal + Low Valence → Angry/Tense
      - Low Arousal + High Valence → Calm/Relaxed
      - Low Arousal + Low Valence → Sad/Depressed
    """
    if arousal > 0.6:
        if valence > 0.6:
            return "excited"  # 高揚感
        elif valence < 0.4:
            return "tense"  # 緊張感
        else:
            return "energetic"  # エネルギッシュ
    elif arousal < 0.4:
        if valence > 0.6:
            return "calm"  # 穏やか
        elif valence < 0.4:
            return "sad"  # 悲しい
        else:
            return "neutral"  # ニュートラル
    else:  # medium arousal
        if valence > 0.6:
            return "happy"  # 幸福
        elif valence < 0.4:
            return "melancholic"  # メランコリック
        else:
            return "neutral"


def extract_emotion_profile(bars_path: Path) -> Dict:
    """bars.parquetからemotion_profile抽出"""
    df = pd.read_parquet(bars_path)

    # 必須カラム確認
    required = ["energy_curve", "accent_score_target", "density_target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in bars.parquet: {missing}")

    # 統計値計算
    energy_mean = df["energy_curve"].mean()
    energy_std = df["energy_curve"].std()
    accent_mean = df["accent_score_target"].mean()
    density_mean = df["density_target"].mean()
    swing_mean = df["swing_target"].mean() if "swing_target" in df.columns else 0.2

    # Valence/Arousal推定
    va = estimate_valence_arousal(
        energy_mean=energy_mean,
        energy_std=energy_std,
        accent_mean=accent_mean,
        density_mean=density_mean,
        swing_mean=swing_mean,
    )

    # Emotion分類
    emotion_label = classify_emotion(va["valence"], va["arousal"])

    # セクション別プロファイル（section_labelベース）
    section_profiles = []
    if "section_label" in df.columns:
        for section_label, section_df in df.groupby("section_label"):
            sec_energy_mean = section_df["energy_curve"].mean()
            sec_accent_mean = section_df["accent_score_target"].mean()
            sec_density_mean = section_df["density_target"].mean()
            sec_swing_mean = (
                section_df["swing_target"].mean() if "swing_target" in section_df.columns else 0.2
            )

            sec_va = estimate_valence_arousal(
                energy_mean=sec_energy_mean,
                energy_std=section_df["energy_curve"].std(),
                accent_mean=sec_accent_mean,
                density_mean=sec_density_mean,
                swing_mean=sec_swing_mean,
            )

            sec_emotion = classify_emotion(sec_va["valence"], sec_va["arousal"])

            section_profiles.append(
                {
                    "section_label": str(section_label),
                    "bar_start": int(section_df["bar_index"].min()),
                    "bar_end": int(section_df["bar_index"].max()),
                    "valence": sec_va["valence"],
                    "arousal": sec_va["arousal"],
                    "emotion": sec_emotion,
                    "energy_mean": float(sec_energy_mean),
                    "accent_mean": float(sec_accent_mean),
                    "density_mean": float(sec_density_mean),
                }
            )

    # 時系列データ（小節ごと）
    bar_timeline = []
    for _, row in df.iterrows():
        bar_energy = row["energy_curve"]
        bar_accent = row["accent_score_target"]

        # 簡易valence/arousal（小節単位）
        bar_valence = np.clip(0.5 + (bar_energy - 0.5) * 0.3 + (bar_accent - 0.5) * 0.2, 0.0, 1.0)
        bar_arousal = np.clip(0.5 + (bar_energy - 0.5) * 0.4, 0.0, 1.0)

        bar_timeline.append(
            {
                "bar_index": int(row["bar_index"]),
                "energy_curve": float(bar_energy),
                "accent_score": float(bar_accent),
                "valence": float(bar_valence),
                "arousal": float(bar_arousal),
            }
        )

    return {
        "global_emotion": {
            "valence": va["valence"],
            "arousal": va["arousal"],
            "emotion_label": emotion_label,
            "energy_mean": float(energy_mean),
            "energy_std": float(energy_std),
            "accent_mean": float(accent_mean),
            "density_mean": float(density_mean),
            "swing_mean": float(swing_mean),
        },
        "section_profiles": section_profiles,
        "bar_timeline": bar_timeline,
        "total_bars": len(df),
        "source_file": str(bars_path.name),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract emotion profile from bars.parquet")
    parser.add_argument("--bars", type=Path, required=True, help="Path to bars.parquet")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output emotion_profile.json path"
    )
    args = parser.parse_args()

    if not args.bars.exists():
        raise FileNotFoundError(f"bars.parquet not found: {args.bars}")

    print(f"📊 Extracting emotion profile from: {args.bars}")

    emotion_profile = extract_emotion_profile(args.bars)

    # 結果保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(emotion_profile, f, indent=2)

    print(f"✅ Emotion profile saved: {args.output}")
    print(f"\n📈 Global Emotion:")
    print(f"   Valence: {emotion_profile['global_emotion']['valence']:.2f}")
    print(f"   Arousal: {emotion_profile['global_emotion']['arousal']:.2f}")
    print(f"   Label: {emotion_profile['global_emotion']['emotion_label']}")
    print(f"   Energy mean: {emotion_profile['global_emotion']['energy_mean']:.2f}")
    print(f"   Accent mean: {emotion_profile['global_emotion']['accent_mean']:.2f}")

    if emotion_profile["section_profiles"]:
        print(f"\n📑 Section Profiles: {len(emotion_profile['section_profiles'])} sections")
        for sec in emotion_profile["section_profiles"]:
            print(
                f"   {sec['section_label']}: {sec['emotion']} (V={sec['valence']:.2f}, A={sec['arousal']:.2f})"
            )


if __name__ == "__main__":
    main()
