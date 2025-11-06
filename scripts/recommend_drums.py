#!/usr/bin/env python3
"""
Drums Recommender - ML推論+パターン検索

SongPackageのbars.parquetを読み込み、各小節で最適なドラムパターンを推奨:
1. bars.parquet読み込み（小節単位の目標値）
2. ML推論（stage2_drums_rhythm_ai.pickle）でfamily推定
3. rhythm_features_merged.parquetから最適パターン検索
4. KPI Gate検証準備
5. drums_recommendations.json出力

使用例:
    python3 scripts/recommend_drums.py \
        --song-package song_packages/sample_project/sample_song/song_package.yaml \
        --output song_packages/sample_project/sample_song/drums_recommendations.json
"""

import argparse
import json
import pickle
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_song_package(yaml_path: Path) -> dict:
    """SongPackage YAML読み込み"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ml_model(pickle_path: Path) -> dict:
    """学習済みMLモデル読み込み"""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def _family_col(df: pd.DataFrame) -> str:
    """family列名の頑健な検出"""
    for c in ["family_label", "family", "Family"]:
        if c in df.columns:
            return c
    raise KeyError("family label column not found in rhythm_features")


def load_rhythm_features(parquet_path: Path, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """rhythm_features_merged.parquet読み込み（高速化：必要列のみ）"""
    if cols:
        return pd.read_parquet(parquet_path, columns=cols)
    return pd.read_parquet(parquet_path)


def load_bars(parquet_path: Path) -> pd.DataFrame:
    """bars.parquet読み込み"""
    return pd.read_parquet(parquet_path)


def predict_family(
    bar_row: pd.Series, ml_model: dict, bpm: float, time_sig: str = "4/4"
) -> Tuple[str, float]:
    """ML推論でfamily推定

    Args:
        bar_row: bars.parquetの1行（1小節分）
        ml_model: stage2_drums_rhythm_ai.pickle
        bpm: テンポ
        time_sig: 拍子

    Returns:
        (family, confidence): 推定family文字列と確信度
    """
    model = ml_model["model"]
    feature_names = ml_model["feature_names"]
    class_labels = ml_model["class_labels"]

    # 特徴量構築（bars.parquetのカラムから推定）
    time_sig_parts = time_sig.split("/")
    time_sig_num = int(time_sig_parts[0])
    time_sig_denom = int(time_sig_parts[1])

    # 仮想特徴量（実際のMIDI特徴は不明なので、目標値から推定）
    features = {
        "tempo_bpm": bpm,
        "swing_pct": bar_row["swing_target"] * 100,  # 0..1 → 0..100
        "backbeat_strength": 0.7,  # 仮定（中程度）
        "kick_downbeat_rate": 0.8,  # 仮定（高め）
        "snare_backbeat_rate": 0.7,  # 仮定
        "hat_density": bar_row["density_target"],
        "time_sig_num": time_sig_num,
        "time_sig_denom": time_sig_denom,
        "slots": 32,  # 仮定（1/32音符グリッド）
        "num_notes": int(bar_row["density_target"] * 4),  # 密度から推定
        "kick_onset_count": 4,  # 仮定
        "snare_onset_count": 2,  # 仮定
        "hat_onset_count": int(bar_row["density_target"]),
        "onset_deviation_mean": 0.05 if bar_row["swing_target"] > 0.3 else 0.02,
        "onset_deviation_std": 0.03 if bar_row["swing_target"] > 0.3 else 0.01,
        "density_mean": bar_row["density_target"],
        "density_std": 1.0,
        "density_min": max(2.0, bar_row["density_target"] - 2.0),
        "density_max": min(12.0, bar_row["density_target"] + 2.0),
    }

    # feature_names順に並べる
    X = np.array([[features.get(fn, 0.0) for fn in feature_names]])

    # 推論
    if hasattr(model, "predict_proba"):
        # LogisticRegression等の確率出力
        proba = model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        family = class_labels[pred_idx]
        confidence = proba[pred_idx]
    else:
        # XGBoost等
        family = model.predict(X)[0]
        confidence = 0.5  # 仮定

    return family, confidence


def topk_candidates(candidates: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Top-K候補抽出"""
    return candidates.sort_values("total_score", ascending=False).head(k)


def search_best_pattern(
    family: str,
    density_target: float,
    swing_target: float,
    rhythm_features: pd.DataFrame,
    family_col_name: str,
    used_patterns: set,
    last_pattern_id: Optional[str] = None,
    last_family: Optional[str] = None,
    diversity_mode: bool = True,
    topk: int = 5,
    drums_active: bool = True,  # Phase A追加: drums_active==0でbreak優先
) -> Optional[Dict]:
    """最適パターン検索（Top-K + セクション多様性）

    Args:
        family: 推定family（STRAIGHT_8, SWING_8等）
        density_target: 目標密度
        swing_target: 目標スウィング
        rhythm_features: rhythm_features_merged.parquet
        family_col_name: family列名（_family_col()で取得）
        used_patterns: 既使用パターンID（多様性確保）
        last_pattern_id: 直前小節のpattern_id（セクション多様性）
        last_family: 直前小節のfamily（セクション多様性）
        diversity_mode: 多様性モード（同じpattern_idの連続使用を避ける）
        topk: Top-K候補数
        drums_active: Drumsアクティブ判定（False時はbreak系優先）

    Returns:
        最適パターン辞書 or None
    """
    # family絞り込み
    candidates = rhythm_features[rhythm_features[family_col_name] == family].copy()

    if len(candidates) == 0:
        return None

    # 密度スコア（目標に近いほど高スコア）
    candidates["density_score"] = 1.0 / (1.0 + np.abs(candidates["hat_density"] - density_target))

    # スウィングスコア（目標に近いほど高スコア）
    candidates["swing_score"] = 1.0 / (1.0 + np.abs(candidates["swing_pct"] / 100.0 - swing_target))

    # 総合スコア
    candidates["total_score"] = candidates["density_score"] * 0.6 + candidates["swing_score"] * 0.4

    # drums_active==0の場合、低密度パターン（break系）を優先
    if not drums_active:
        # hat_density < 3.0 のパターンにボーナス（0.5→0.3で抑制、不要なブレイク回避）
        break_bonus = (candidates["hat_density"] < 3.0).astype(float) * 0.3
        candidates["total_score"] += break_bonus

    # セクション多様性ペナルティ（直前小節との連続回避）
    if diversity_mode and last_pattern_id:
        candidates.loc[candidates["loop_id"] == last_pattern_id, "total_score"] -= 0.3
    if diversity_mode and last_family:
        candidates.loc[candidates[family_col_name] == last_family, "total_score"] -= 0.1

    # グローバル多様性ペナルティ
    if diversity_mode:
        candidates["diversity_penalty"] = candidates["loop_id"].apply(
            lambda x: 0.2 if x in used_patterns else 0.0
        )
        candidates["total_score"] -= candidates["diversity_penalty"]

    # Top-K抽出 + ランダムサンプリング
    topk_cands = topk_candidates(candidates, k=topk)

    if len(topk_cands) == 0:
        return None

    # Top-1選択（または確率的サンプリング拡張可能）
    best_row = topk_cands.iloc[0]

    return {
        "pattern_id": best_row["loop_id"],
        "family": family,
        "density": float(best_row["hat_density"]),
        "swing": float(best_row["swing_pct"] / 100.0),
        "tempo_bpm": float(best_row["tempo_bpm"]),
        "backbeat_strength": float(best_row.get("backbeat_strength", 0.7)),
        "score": float(best_row["total_score"]),
    }


def recommend_drums(
    song_package_path: Path,
    output_path: Path,
    diversity_mode: bool = True,
    use_ml: bool = True,
    topk: int = 5,
    verbose: bool = True,
    stems_features_path: Optional[Path] = None,
):
    """ドラムパターン推奨メイン処理

    Args:
        song_package_path: SongPackage YAMLパス
        output_path: drums_recommendations.json出力パス
        diversity_mode: 多様性モード
        use_ml: MLモデル使用（False時はルールベース）
        topk: Top-K候補数
        verbose: 詳細出力
        stems_features_path: Stem特徴Parquetパス（Phase 1統合）
    """
    # SongPackage読み込み
    if verbose:
        print(f"📖 Loading SongPackage: {song_package_path}")

    song_package = load_song_package(song_package_path)

    # パス解決（相対パス → 絶対パス）
    base_dir = song_package_path.parent

    # bars.parquet (schema v1.1対応: artifacts → paths)
    bars_key = "paths" if "paths" in song_package else "artifacts"
    bars_path = base_dir / song_package[bars_key]["bars"]
    bars_df = load_bars(bars_path)

    if verbose:
        print(f"   Total bars: {len(bars_df)}")

    # Stem特徴読み込み（Phase 1統合）
    stem_df = None
    if stems_features_path and stems_features_path.exists():
        stem_df = pd.read_parquet(stems_features_path)
        if verbose:
            print(f"   Stem features: loaded ({len(stem_df)} bars)")
            print(f"      hat_density: {stem_df['hat_density'].mean():.2f} avg")
            print(f"      fill_likelihood: {stem_df['fill_likelihood'].mean():.2f} avg")

    # arranger_weights.yaml読み込み（Stem統合パラメータ）
    config_path = Path(__file__).parent.parent / "configs" / "arranger_weights.yaml"
    arranger_cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            arranger_cfg = yaml.safe_load(f)

    stems_cfg = arranger_cfg.get("stems", {})
    use_stems = stems_cfg.get("use_stems", False) and stem_df is not None
    density_boost = stems_cfg.get("drums", {}).get("density_boost", 0.6)
    fill_boost = stems_cfg.get("drums", {}).get("fill_boost", 0.3)

    if verbose and use_stems:
        print(
            f"   Stem integration: ENABLED (density_boost={density_boost}, fill_boost={fill_boost})"
        )

    # Stem特徴統合（bars_df更新）
    if use_stems:
        # 密度ブースト: target = max(bars.target, stem.hat_density * boost)
        stem_density_boosted = stem_df["hat_density"] * density_boost
        bars_df["density_target_original"] = bars_df["density_target"].copy()
        bars_df["density_target"] = bars_df["density_target"].combine(stem_density_boosted, max)

        # Fill優先度（後続でスコア加点）
        bars_df["fill_priority"] = (stem_df["fill_likelihood"] > 0.6).astype(float) * fill_boost
        bars_df["vocal_stress"] = stem_df["vocal_stress"]

        if verbose:
            boosted_count = (bars_df["density_target"] > bars_df["density_target_original"]).sum()
            print(f"      Density boosted: {boosted_count}/{len(bars_df)} bars")
            fill_priority_count = (bars_df["fill_priority"] > 0).sum()
            print(f"      Fill priority: {fill_priority_count}/{len(bars_df)} bars")

    # ML model（オプショナル）
    ml_model = None
    if use_ml and "models" in song_package and "drums" in song_package["models"]:
        model_rel_path = song_package["models"]["drums"]
        # 相対パスの場合、プロジェクトルート（composer2-3）から解決
        if not model_rel_path.startswith("/"):
            current = song_package_path.parent
            while current.name != "composer2-3" and current.parent != current:
                current = current.parent
            project_root = current
            model_path = project_root / model_rel_path
        else:
            model_path = Path(model_rel_path)

        ml_model = try_load_ml_model(model_path)

    if verbose:
        if ml_model:
            print(f"   ML model: loaded ({ml_model.get('class_labels', [])})")
        else:
            print(f"   ML model: NOT FOUND (fallback to heuristic)")

    # デフォルトfamily定義（ML未使用時）
    default_families = ["STRAIGHT_8", "SWING_8", "STRAIGHT_16", "SWING_16"]
    class_labels = ml_model["class_labels"] if ml_model else default_families

    # rhythm_features_merged.parquet（必要列のみ高速読み込み）
    rhythm_features_rel_path = "output/rhythm_ai/rhythm_features_merged.parquet"
    if not rhythm_features_rel_path.startswith("/"):
        current = song_package_path.parent
        while current.name != "composer2-3" and current.parent != current:
            current = current.parent
        project_root = current
        rhythm_features_path = project_root / rhythm_features_rel_path
    else:
        rhythm_features_path = Path(rhythm_features_rel_path)

    # 必要列のみ読み込み（高速化）
    required_cols = ["loop_id", "tempo_bpm", "hat_density", "swing_pct", "backbeat_strength"]
    # family列を検出するため、まず全列読み込み（初回のみ）
    rhythm_features_full = load_rhythm_features(rhythm_features_path)
    fam_col = _family_col(rhythm_features_full)
    required_cols.append(fam_col)

    # 必要列のみ再読み込み
    rhythm_features = load_rhythm_features(rhythm_features_path, cols=required_cols)

    if verbose:
        print(f"   Rhythm features: {len(rhythm_features)} patterns")

    # meta情報（schema v1.1対応）
    if "meta" in song_package:
        # schema v1.0
        bpm = song_package["meta"].get("bpm") or song_package["meta"].get("tempo_bpm", 120.0)
        time_sig = song_package["meta"].get("time_signature", "4/4")
    else:
        # schema v1.1
        time_info = song_package.get("time", {})
        tempo_info = time_info.get("tempo", {})
        if isinstance(tempo_info, dict):
            bpm = tempo_info.get("summary_bpm") or tempo_info.get("bpm_median", 120.0)
        else:
            bpm = 120.0
        time_sig_info = time_info.get("signature", {})
        if isinstance(time_sig_info, dict):
            time_sig = f"{time_sig_info.get('num', 4)}/{time_sig_info.get('den', 4)}"
        else:
            time_sig = "4/4"

    # 推奨処理
    recommendations = {}
    used_patterns = set()

    family_counts = {label: 0 for label in class_labels}

    last_pattern_id = None
    last_family = None

    for idx, bar_row in bars_df.iterrows():
        bar_idx = bar_row["bar_index"]

        # ML推論 or ルールベース
        if ml_model:
            family, confidence = predict_family(bar_row, ml_model, bpm, time_sig)
        else:
            family, confidence = rule_based_family(bar_row, bpm, time_sig)
        family_counts[family] += 1

        # drums_active取得（stem_df利用時）
        drums_active = True
        if use_stems and "drums_active" in stem_df.columns:
            try:
                drums_active = bool(stem_df.loc[stem_df.index[bar_idx], "drums_active"])
            except (IndexError, KeyError):
                drums_active = True  # fallback

        # パターン検索（Top-K + セクション多様性 + drums_active）
        pattern = search_best_pattern(
            family,
            bar_row["density_target"],
            bar_row["swing_target"],
            rhythm_features,
            fam_col,
            used_patterns,
            last_pattern_id,
            last_family,
            diversity_mode,
            topk,
            drums_active,  # Phase A追加
        )

        # Fill優先度調整（Stem統合）
        if use_stems and pattern is not None:
            fill_priority = bar_row.get("fill_priority", 0.0)
            if fill_priority > 0:
                pattern["score"] += fill_priority
                if verbose and bar_idx % 16 == 0:  # 16小節ごとにログ
                    print(f"      Bar {bar_idx}: Fill priority +{fill_priority:.2f}")

        if pattern is None:
            # fallback: 全familyから検索
            for fallback_family in class_labels:
                if fallback_family != family:
                    pattern = search_best_pattern(
                        fallback_family,
                        bar_row["density_target"],
                        bar_row["swing_target"],
                        rhythm_features,
                        fam_col,
                        used_patterns,
                        last_pattern_id,
                        last_family,
                        diversity_mode,
                        topk,
                    )
                    if pattern is not None:
                        if verbose:
                            print(
                                f"   ⚠️  Bar {bar_idx}: {family} not found, fallback to {fallback_family}"
                            )
                        break

        if pattern is not None:
            used_patterns.add(pattern["pattern_id"])
            last_pattern_id = pattern["pattern_id"]
            last_family = pattern["family"]

            recommendations[f"bar_{bar_idx}"] = {
                "bar_index": int(bar_idx),
                "section_label": bar_row["section_label"],
                "energy_curve": float(bar_row["energy_curve"]),
                "density_target": float(bar_row["density_target"]),
                "swing_target": float(bar_row["swing_target"]),
                "predicted_family": family,
                "confidence": float(confidence),
                "pattern": pattern,
                "kpi_pass": True,  # KPI Gate実装後に更新
            }
        else:
            if verbose:
                print(f"   ❌ Bar {bar_idx}: No pattern found for {family}")
            last_pattern_id = None
            last_family = None

    # 統計
    if verbose:
        print(f"\n📊 Recommendation Statistics:")
        print(f"   Total bars: {len(bars_df)}")
        print(f"   Recommended: {len(recommendations)}")
        print(f"   Unique patterns: {len(used_patterns)}")
        print(f"\n   Family distribution:")
        for family, count in sorted(family_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(bars_df) * 100 if len(bars_df) > 0 else 0
            print(f"     {family}: {count} ({pct:.1f}%)")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n✅ Saved recommendations: {output_path}")


def rule_based_family(bar_row: pd.Series, bpm: float, time_sig: str = "4/4") -> Tuple[str, float]:
    """ルールベースfamily推定（MLモデル未使用時のフォールバック）"""
    swing = float(bar_row.get("swing_target", 0.0))
    density = float(bar_row.get("density_target", 4.0))

    if swing >= 0.25:
        return "SWING_8", 0.55
    if density >= 8.0:
        return "STRAIGHT_16", 0.55
    return "STRAIGHT_8", 0.55


def try_load_ml_model(pickle_path: Path) -> Optional[dict]:
    """ML モデル読み込み（失敗時はNone）"""
    try:
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Recommend drums patterns based on bars.parquet and ML model"
    )
    parser.add_argument("--song-package", type=Path, required=True, help="Path to SongPackage YAML")
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output drums_recommendations.json"
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Disable diversity mode (allow same pattern repeat)",
    )
    parser.add_argument(
        "--no-ml", action="store_true", help="Use heuristic family estimation (no ML model)"
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Top-K candidates for selection (default: 5)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--stems-features",
        type=Path,
        default=None,
        help="Path to stem_features.parquet (Phase 1 integration)",
    )

    args = parser.parse_args()

    recommend_drums(
        args.song_package,
        args.output,
        diversity_mode=not args.no_diversity,
        use_ml=not args.no_ml,
        topk=args.topk,
        verbose=not args.quiet,
        stems_features_path=args.stems_features,
    )


if __name__ == "__main__":
    main()
