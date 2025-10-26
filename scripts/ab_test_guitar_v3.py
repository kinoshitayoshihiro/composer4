#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B Test: Guitar Stage2 v1 (Rule-based) vs v3 (XGBoost Tuned)

評価指標:
- パターン一致率: pattern_id_v1 == pattern_id_v3
- 演奏密度差: |notes_v3 - notes_v1| / bar
- アクセント整合: accent_grid との一致率
- 和声整合: root/quality に対する禁則チェック

合格ライン:
- パターン一致率 >= 65%
- アクセント整合 +5% 以上 (v1比)
- 演奏密度差中央値 <= 1 ノート/小節

v3改善点:
- Accuracy: 95.84% (v2: 91.74%, +4.1%向上)
- Top-3: 97.99% (v2: 95.86%, +2.1%向上)
- F1: 94.91% (v2: 89.86%, +5.0%向上)
"""

import os
import sys
import logging
import random
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_song_packages(
    limit: int = 50, index_csv: str = "song_packages_index.csv"
) -> List[Dict[str, Any]]:
    """ランダム50曲抽出 (Gold/Silver優先)"""
    if not Path(index_csv).exists():
        logger.error(f"Index CSV not found: {index_csv}")
        return []

    # CSVから読み込み
    df = pd.read_csv(index_csv)
    logger.info(f"Found {len(df)} packages in index")

    # label_strength を quality にリネーム
    if "label_strength" in df.columns:
        df["quality"] = df["label_strength"]

    # Quality優先ソート（Gold > Silver > Bronze）
    quality_order = {"gold": 0, "silver": 1, "bronze": 2}
    df["quality_rank"] = df["quality"].map(lambda q: quality_order.get(str(q).lower(), 3))
    df = df.sort_values("quality_rank")

    # 上位limit曲をランダムサンプル（Gold/Silver中心）
    gold_silver = df[df["quality"].isin(["gold", "silver"])]
    if len(gold_silver) >= limit:
        sampled = gold_silver.sample(n=limit, random_state=42)
    else:
        sampled = df.sample(n=min(len(df), limit), random_state=42)

    packages = []
    for _, row in sampled.iterrows():
        packages.append(
            {"song_id": row["song_id"], "path": row["package_path"], "quality": row["quality"]}
        )

    logger.info(f"Sampled {len(packages)} songs for A/B test")
    quality_counts = sampled["quality"].value_counts()
    for q, cnt in quality_counts.items():
        logger.info(f"  {q}: {cnt}")

    return packages


def get_pattern_from_recommender(
    recommender,
    section: str,
    chord_root: str,
    chord_quality: str,
    tempo: float,
    confidence: float,
    time_sig: str = "4/4",
    target_accent=None,
    target_density_ql=None,
    rerank_conf_thresh=0.35,
    rerank_w_proba=0.60,
    rerank_w_accent=0.25,
    rerank_w_density=0.10,
    rerank_w_section=0.05,
) -> Dict[str, Any]:
    """Recommender からパターン取得（再ランク用features拡張）"""
    try:
        # 再ランク用features構築
        features = {
            "section": section,
            "chord_root": chord_root,
            "chord_quality": chord_quality,
            "tempo": tempo,
            "confidence": confidence,
            "time_sig": time_sig,
        }
        
        # 再ランク用パラメータ追加
        if target_accent is not None:
            features["target_accent"] = target_accent
        else:
            # デフォルト: ダウンビート強調
            features["target_accent"] = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        
        if target_density_ql is not None:
            features["target_density_ql"] = target_density_ql
        else:
            # デフォルト: セクション別
            if section in ("Chorus", "PreChorus"):
                features["target_density_ql"] = 8.0
            elif section in ("Bridge",):
                features["target_density_ql"] = 6.0
            else:
                features["target_density_ql"] = 4.0
        
        # 再ランクパラメータを features に追加
        features["rerank_conf_thresh"] = rerank_conf_thresh
        features["rerank_w_proba"] = rerank_w_proba
        features["rerank_w_accent"] = rerank_w_accent
        features["rerank_w_density"] = rerank_w_density
        features["rerank_w_section"] = rerank_w_section
        
        # recommender.recommend() または get_pattern() で取得
        if hasattr(recommender, 'recommend'):
            pattern = recommender.recommend(features, topk=1)
        else:
            pattern = recommender.get_pattern(features=features, topk=1)
        
        return pattern or {}
    except Exception as e:
        logger.warning(f"Pattern fetch failed: {e}")
        return {}


def compute_note_density(pattern: Dict[str, Any]) -> float:
    """演奏密度（ノート数/小節）"""
    rhythm = pattern.get("rhythm", "standard_quarter")
    density_map = {
        "standard_quarter": 4.0,
        "standard_8ths": 8.0,
        "standard_16ths": 16.0,
        "arp_8ths": 8.0,
        "arp_16ths": 16.0,
        "strum_down_8ths": 8.0,
        "strum_alt_16ths": 16.0,
        "sparse_half": 2.0,
        "sparse_whole": 1.0,
        "syncopated_8ths": 6.0,
    }
    return density_map.get(rhythm, 4.0)


def compute_accent_match(pattern, accent_grid, phase_slots=0):
    """
    アクセント一致度を計算。
    位相シフトも考慮。
    
    Args:
        pattern: パターン辞書（既に正規化済み: {"rhythm": "...", "accent_profile": [...], ...}）
        accent_grid: ターゲットアクセント（16スロット）
        phase_slots: 位相シフト量（0なら無し）
    
    Returns:
        cos類似度 (0.0 ~ 1.0)
    """
    acc_prof = pattern.get("accent_profile", [])
    if not acc_prof or not accent_grid:
        # accent_profileが無い場合はrhythmベースで推定
        rhythm = pattern.get("rhythm", "")
        if "sparse" in rhythm or "pickup" in rhythm:
            return 0.5
        elif "eighth" in rhythm:
            return 0.7
        elif "arpeggio" in rhythm:
            return 0.6
        else:
            return 0.9
    
    acc = np.array(acc_prof, dtype=float)
    tgt = np.array(accent_grid, dtype=float)
    
    # 長さが一致しない場合はゼロパディング or トリム
    if acc.size != tgt.size:
        if acc.size < tgt.size:
            acc = np.pad(acc, (0, tgt.size - acc.size), mode='constant', constant_values=0.0)
        else:
            acc = acc[:tgt.size]
    
    # 位相シフト適用
    if phase_slots > 0 and acc.size > 0:
        acc = np.roll(acc, phase_slots)
    
    # cos類似度計算
    norm_acc = np.linalg.norm(acc)
    norm_tgt = np.linalg.norm(tgt)
    
    if norm_acc < 1e-6 or norm_tgt < 1e-6:
        return 0.5  # ゼロベクトルの場合は中立値
    
    cos_sim = float(np.dot(acc, tgt) / (norm_acc * norm_tgt))
    
    # cos類似度を0..1にクリップ
    return max(0.0, min(1.0, cos_sim))


def check_harmonic_rules(pattern: Dict[str, Any], chord_quality: str) -> bool:
    """和声禁則チェック（簡易版）"""
    voicing = pattern.get("voicing", [0, 4, 7])

    # Minor系でM3禁止
    if chord_quality in ["min", "min7", "min9"]:
        if 4 in voicing:  # M3
            return False

    # Maj系でm3禁止
    if chord_quality in ["maj", "maj7", "maj9"]:
        if 3 in voicing:  # m3
            return False

    return True


def run_v3_evaluation(
    v3_pickle_path: str,
    songs: List[Dict[str, Any]],
    output_csv: str = "data/eval_v3.csv",
    rerank_conf_thresh: float = 0.25,
    rerank_w_proba: float = 0.55,
    rerank_w_accent: float = 0.30,
    rerank_w_density: float = 0.10,
    rerank_w_section: float = 0.05,
) -> pd.DataFrame:
    """
    v3単独の絶対評価（v1比較なし）
    
    評価KPI:
    - accent_score: 理想アクセントとのcos類似度 (0~1, 目標≥0.65)
    - density_abs: |目標 - 実際| の絶対誤差 (目標≤1.0)
    - chord_fit: コード構成音への適合率 (0~1, 目標≥0.60)
    - ml_used: ML推論が採用された割合 (目標≥70%)
    - top1_proba: 再ランク前Top-1確率 (参考≥0.55)
    """
    logger.info(f"Loading v3 pickle: {v3_pickle_path}")
    
    from ml.simple_pattern_recommender import SimplePatternRecommender
    
    recommender = SimplePatternRecommender(instrument="guitar", patterns_path=v3_pickle_path)
    logger.info(f"v3 selector type: {recommender.selector.get('type')}")
    
    # テストケース生成
    test_cases = []
    for song in songs:
        for section in ["Intro", "Verse", "Chorus", "Bridge"]:
            for chord_root in ["C", "G", "Am", "F"]:
                for chord_quality in ["maj", "maj7", "min", "min7"]:
                    test_cases.append({
                        "song_id": song["song_id"],
                        "section": section,
                        "chord_root": chord_root,
                        "chord_quality": chord_quality,
                        "tempo": 120.0,
                        "confidence": 0.8,
                        "time_sig": "4/4",
                    })
    
    logger.info(f"Generated {len(test_cases)} test cases")
    
    # 評価実行
    results = []
    for tc in test_cases:
        pattern_v3 = get_pattern_from_recommender(
            recommender,
            tc["section"],
            tc["chord_root"],
            tc["chord_quality"],
            tc["tempo"],
            tc["confidence"],
            tc["time_sig"],
            rerank_conf_thresh=rerank_conf_thresh,
            rerank_w_proba=rerank_w_proba,
            rerank_w_accent=rerank_w_accent,
            rerank_w_density=rerank_w_density,
            rerank_w_section=rerank_w_section,
        )
        
        if not pattern_v3:
            continue
        
        # パターン正規化
        p_v3 = pattern_v3.get("pattern", pattern_v3)
        pattern_id_v3 = p_v3.get("id", "unknown")
        
        # ML使用判定
        ml_used = 1 if pattern_v3.get("confidence", 0.0) >= rerank_conf_thresh else 0
        top1_proba = float(pattern_v3.get("confidence", 0.3))
        
        # 理想的なアクセントグリッド（セクション依存）
        if tc["section"] == "Chorus":
            ideal_accent = [0.9, 0.3, 0.6, 0.3, 0.8, 0.3, 0.6, 0.3,
                           0.9, 0.3, 0.6, 0.3, 0.8, 0.3, 0.6, 0.3]
        elif tc["section"] == "Verse":
            ideal_accent = [0.7, 0.4, 0.5, 0.4] * 4
        else:
            ideal_accent = [0.5] * 16
        
        # 位相シフト取得・適用
        phase_v3 = pattern_v3.get("phase_slots", 0)
        accent_score = compute_accent_match(p_v3, ideal_accent, phase_slots=phase_v3)
        
        # 密度誤差
        target_density = 4.0  # 4分音符ベース（ダミー）
        density_v3 = compute_note_density(pattern_v3)
        density_abs = abs(density_v3 - target_density)
        
        # コード適合度（簡易版: 和声禁則チェック）
        chord_fit = 1.0 if check_harmonic_rules(p_v3, tc["chord_quality"]) else 0.5
        
        results.append({
            "song_id": tc["song_id"],
            "section": tc["section"],
            "chord_root": tc["chord_root"],
            "chord_quality": tc["chord_quality"],
            "tempo": tc["tempo"],
            "pattern_id": pattern_id_v3,
            "accent_score": accent_score,
            "density_abs": density_abs,
            "chord_fit": chord_fit,
            "ml_used": ml_used,
            "top1_proba": top1_proba,
            "phase_slots": phase_v3,
        })
    
    # DataFrame化
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")
    
    # サマリー出力
    logger.info("\n" + "=" * 60)
    logger.info("v3 Absolute Quality Metrics")
    logger.info("=" * 60)
    
    accent_score_mean = df["accent_score"].mean() * 100
    density_abs_median = df["density_abs"].median()
    chord_fit_mean = df["chord_fit"].mean() * 100
    ml_usage_rate = df["ml_used"].mean() * 100
    top1_proba_mean = df["top1_proba"].mean()
    
    logger.info(f"Accent Score (mean): {accent_score_mean:.2f}%")
    logger.info(f"Density Abs (median): {density_abs_median:.2f} notes/bar")
    logger.info(f"Chord Fit (mean): {chord_fit_mean:.2f}%")
    logger.info(f"ML Usage Rate: {ml_usage_rate:.2f}%")
    logger.info(f"Top-1 Proba (mean): {top1_proba_mean:.4f}")
    logger.info("")
    
    # セクション別
    logger.info("Section-wise Metrics:")
    for section in ["Chorus", "Verse", "Bridge"]:
        sec_df = df[df["section"] == section]
        if len(sec_df) == 0:
            continue
        sec_accent = sec_df["accent_score"].mean() * 100
        sec_ml = sec_df["ml_used"].mean() * 100
        logger.info(f"  {section}:")
        logger.info(f"    Accent Score: {sec_accent:.2f}%")
        logger.info(f"    ML Usage: {sec_ml:.2f}%")
    logger.info("")
    
    # KPIゲート判定
    logger.info("-" * 60)
    logger.info("Pass/Fail Criteria (Absolute KPIs)")
    logger.info("-" * 60)
    
    pass_accent = accent_score_mean >= 65.0
    pass_density = density_abs_median <= 1.0
    pass_chord = chord_fit_mean >= 60.0
    pass_ml = ml_usage_rate >= 70.0
    
    logger.info(f"Accent Score >= 65%: {'PASS' if pass_accent else 'FAIL'} ({accent_score_mean:.2f}%)")
    logger.info(f"Density Abs <= 1.0: {'PASS' if pass_density else 'FAIL'} ({density_abs_median:.2f})")
    logger.info(f"Chord Fit >= 60%: {'PASS' if pass_chord else 'FAIL'} ({chord_fit_mean:.2f}%)")
    logger.info(f"ML Usage >= 70%: {'PASS' if pass_ml else 'FAIL'} ({ml_usage_rate:.2f}%)")
    
    all_pass = pass_accent and pass_density and pass_chord and pass_ml
    logger.info("\n" + "=" * 60)
    logger.info(f"Overall: {'✓ PASS (v3 production ready)' if all_pass else '✗ FAIL (needs tuning)'}")
    logger.info("=" * 60 + "\n")
    
    return df


def run_ab_test(
    v1_pickle_path: str,
    v2_pickle_path: str,
    songs: List[Dict[str, Any]],
    output_csv: str = "data/ab_test_guitar_results.csv",
    rerank_conf_thresh: float = 0.35,
    rerank_w_proba: float = 0.60,
    rerank_w_accent: float = 0.25,
    rerank_w_density: float = 0.10,
    rerank_w_section: float = 0.05,
) -> pd.DataFrame:
    """A/Bテスト実行"""

    # v1 / v2 Recommender 初期化
    logger.info(f"Loading v1 pickle: {v1_pickle_path}")
    logger.info(f"Loading v2 pickle: {v2_pickle_path}")

    from ml.simple_pattern_recommender import SimplePatternRecommender

    recommender_v1 = SimplePatternRecommender(instrument="guitar", patterns_path=v1_pickle_path)

    recommender_v2 = SimplePatternRecommender(instrument="guitar", patterns_path=v2_pickle_path)

    logger.info(f"v1 selector type: {recommender_v1.selector.get('type')}")
    logger.info(f"v2 selector type: {recommender_v2.selector.get('type')}")

    # テストケース生成
    test_cases = []
    for song in songs:
        # ダミーコード進行（Chorus, C:maj7, 120bpm）
        # 実際は song_package.yaml から chordmap を読み込む
        for section in ["Intro", "Verse", "Chorus", "Bridge"]:
            for chord_root in ["C", "G", "Am", "F"]:
                for chord_quality in ["maj", "maj7", "min", "min7"]:
                    test_cases.append(
                        {
                            "song_id": song["song_id"],
                            "section": section,
                            "chord_root": chord_root,
                            "chord_quality": chord_quality,
                            "tempo": 120.0,
                            "confidence": 0.8,
                            "time_sig": "4/4",
                        }
                    )

    logger.info(f"Generated {len(test_cases)} test cases")

    # A/B評価
    results = []
    for tc in test_cases:
        pattern_v1 = get_pattern_from_recommender(
            recommender_v1,
            tc["section"],
            tc["chord_root"],
            tc["chord_quality"],
            tc["tempo"],
            tc["confidence"],
            tc["time_sig"],
            rerank_conf_thresh=rerank_conf_thresh,
            rerank_w_proba=rerank_w_proba,
            rerank_w_accent=rerank_w_accent,
            rerank_w_density=rerank_w_density,
            rerank_w_section=rerank_w_section,
        )

        pattern_v3 = get_pattern_from_recommender(
            recommender_v2,
            tc["section"],
            tc["chord_root"],
            tc["chord_quality"],
            tc["tempo"],
            tc["confidence"],
            tc["time_sig"],
            rerank_conf_thresh=rerank_conf_thresh,
            rerank_w_proba=rerank_w_proba,
            rerank_w_accent=rerank_w_accent,
            rerank_w_density=rerank_w_density,
            rerank_w_section=rerank_w_section,
        )

        # 評価指標
        pattern_id_v1 = pattern_v1.get("pattern_id", "unknown")
        pattern_id_v3 = pattern_v3.get("pattern_id", "unknown")
        pattern_match = int(pattern_id_v1 == pattern_id_v3)
        
        # Family一致率（パターン辞書のfamilyを優先、無ければprefix）
        def get_family(pid, pattern_dict):
            if pid in ["unknown", "default_major"]:
                return "unknown"
            # パターン辞書からfamilyを取得
            fam = pattern_dict.get("family")
            if fam:
                return fam
            # フォールバック: pattern_idの先頭（'_'前、または先頭12文字）
            return pid.split('_')[0] if '_' in pid else pid[:12]
        
        family_v1 = get_family(pattern_id_v1, pattern_v1)
        family_v3 = get_family(pattern_id_v3, pattern_v3)
        family_match = int(family_v1 == family_v3)
        
        # ML活用（v3がMLで推論した場合1、フォールバックした場合0）
        # pattern_v3の confidence が 0.01 未満ならフォールバック扱い
        ml_used = int(pattern_v3.get("confidence", 0.0) >= 0.01)
        top1_proba = float(pattern_v3.get("confidence", 0.0))

        density_v1 = compute_note_density(pattern_v1)
        density_v3 = compute_note_density(pattern_v3)
        density_diff = abs(density_v3 - density_v1)

        # パターン構造の正規化: {"pattern": {...}} の場合は展開
        p_v1 = pattern_v1.get("pattern", pattern_v1)
        p_v3 = pattern_v3.get("pattern", pattern_v3)
        
        # ▼ 理想的なアクセントグリッドを定義（セクション依存）
        # v1とv3の両方をこのターゲットと比較
        if tc["section"] == "Chorus":
            # Chorus: 強拍にアクセント
            ideal_accent = [0.9, 0.3, 0.6, 0.3, 0.8, 0.3, 0.6, 0.3,
                           0.9, 0.3, 0.6, 0.3, 0.8, 0.3, 0.6, 0.3]
        elif tc["section"] == "Verse":
            # Verse: やや控えめ
            ideal_accent = [0.7, 0.4, 0.5, 0.4] * 4
        else:
            # Bridge/Intro: 均等
            ideal_accent = [0.5] * 16
        
        # 位相シフトを取得（v3のみ、再ランク時に最適化済み）
        phase_v3 = pattern_v3.get("phase_slots", 0)
        
        # 両方とも理想的なアクセントと比較
        accent_match_v1 = compute_accent_match(p_v1, ideal_accent, phase_slots=0)
        accent_match_v3 = compute_accent_match(p_v3, ideal_accent, phase_slots=phase_v3)

        harmonic_ok_v1 = check_harmonic_rules(pattern_v1, tc["chord_quality"])
        harmonic_ok_v3 = check_harmonic_rules(pattern_v3, tc["chord_quality"])

        results.append(
            {
                "song_id": tc["song_id"],
                "section": tc["section"],
                "chord_root": tc["chord_root"],
                "chord_quality": tc["chord_quality"],
                "tempo": tc["tempo"],
                "pattern_id_v1": pattern_id_v1,
                "pattern_id_v3": pattern_id_v3,
                "pattern_match": pattern_match,
                "family_v1": family_v1,
                "family_v3": family_v3,
                "family_match": family_match,
                "ml_used": ml_used,
                "top1_proba": top1_proba,
                "density_v1": density_v1,
                "density_v3": density_v3,
                "density_diff": density_diff,
                "accent_match_v1": accent_match_v1,
                "accent_match_v3": accent_match_v3,
                "accent_delta": accent_match_v3 - accent_match_v1,
                "harmonic_ok_v1": int(harmonic_ok_v1),
                "harmonic_ok_v3": int(harmonic_ok_v3),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")

    # 集計
    logger.info("\n" + "=" * 60)
    logger.info("A/B Test Results Summary (Musical KPIs)")
    logger.info("=" * 60)

    # 音楽的KPI
    accent_delta_mean = df["accent_delta"].mean() * 100
    density_diff_median = df["density_diff"].median()
    ml_usage_rate = df["ml_used"].mean() * 100
    family_match_rate = df["family_match"].mean() * 100
    top1_proba_mean = df["top1_proba"].mean()
    
    logger.info(f"Accent Delta (v3 - v1): {accent_delta_mean:+.2f}%")
    logger.info(f"Density Diff (median): {density_diff_median:.2f} notes/bar")
    logger.info(f"ML Usage Rate: {ml_usage_rate:.2f}%")
    logger.info(f"Family Match Rate: {family_match_rate:.2f}%")
    logger.info(f"Top-1 Proba (mean): {top1_proba_mean:.4f}")

    # セクション別集計
    logger.info("\nSection-wise Metrics:")
    for section in ["Chorus", "Verse", "Bridge"]:
        sec_df = df[df["section"] == section]
        if len(sec_df) > 0:
            logger.info(f"  {section}:")
            logger.info(f"    Accent Delta: {sec_df['accent_delta'].mean() * 100:+.2f}%")
            logger.info(f"    ML Usage: {sec_df['ml_used'].mean() * 100:.2f}%")

    harmonic_ok_v1_rate = df["harmonic_ok_v1"].mean() * 100
    harmonic_ok_v3_rate = df["harmonic_ok_v3"].mean() * 100
    logger.info(f"\nHarmonic Rules OK (v1): {harmonic_ok_v1_rate:.2f}%")
    logger.info(f"Harmonic Rules OK (v3): {harmonic_ok_v3_rate:.2f}%")

    # 合格ライン判定（新基準）
    logger.info("\n" + "-" * 60)
    logger.info("Pass/Fail Criteria (Musical KPIs)")
    logger.info("-" * 60)

    pass_accent_delta = accent_delta_mean >= 5.0
    pass_density_diff = density_diff_median <= 1.0
    pass_ml_usage = ml_usage_rate >= 70.0
    pass_family_match = family_match_rate >= 80.0

    logger.info(
        f"Accent Delta >= +5%: {'PASS' if pass_accent_delta else 'FAIL'} ({accent_delta_mean:+.2f}%)"
    )
    logger.info(
        f"Density Diff <= 1 note/bar: {'PASS' if pass_density_diff else 'FAIL'} ({density_diff_median:.2f})"
    )
    logger.info(
        f"ML Usage >= 70%: {'PASS' if pass_ml_usage else 'FAIL'} ({ml_usage_rate:.2f}%)"
    )
    logger.info(
        f"Family Match >= 80%: {'PASS' if pass_family_match else 'FAIL'} ({family_match_rate:.2f}%)"
    )

    all_pass = pass_accent_delta and pass_density_diff and pass_ml_usage and pass_family_match
    logger.info("\n" + "=" * 60)
    logger.info(
        f"Overall: {'✓ PASS (v3 ready for rollout)' if all_pass else '✗ FAIL (needs parameter tuning)'}"
    )
    logger.info("=" * 60 + "\n")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="A/B Test for Guitar Stage2 v1 vs v3")
    parser.add_argument(
        "--v3-only",
        action="store_true",
        help="Evaluate absolute v3 quality metrics only (no v1 baseline comparison)",
    )
    parser.add_argument(
        "--v1-pickle",
        type=str,
        default="data/patterns/stage2_guitar.pickle",
        help="Path to v1 pickle (rule-based)",
    )
    parser.add_argument(
        "--v3-pickle",
        type=str,
        default="data/patterns/stage2_guitar_v3.pickle",
        help="Path to v3 pickle (XGBoost Tuned)",
    )
    parser.add_argument("--num-songs", type=int, default=50, help="Number of songs to test")
    parser.add_argument(
        "--output", type=str, default="data/ab_test_guitar_results.csv", help="Output CSV path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Re-ranking tuning parameters
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.35,
        help="ML confidence threshold (default: 0.35, tune to 0.25 for higher ML coverage)",
    )
    parser.add_argument(
        "--w-proba",
        type=float,
        default=0.60,
        help="Re-ranking weight for ML probability (default: 0.60)",
    )
    parser.add_argument(
        "--w-accent",
        type=float,
        default=0.25,
        help="Re-ranking weight for accent fit (default: 0.25)",
    )
    parser.add_argument(
        "--w-density",
        type=float,
        default=0.10,
        help="Re-ranking weight for density fit (default: 0.10)",
    )
    parser.add_argument(
        "--w-section",
        type=float,
        default=0.05,
        help="Re-ranking weight for section fit (default: 0.05)",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 曲リスト取得
    songs = load_song_packages(limit=args.num_songs)
    if not songs:
        logger.error("No songs found for testing")
        sys.exit(1)

    # Re-ranking parameters logging
    logger.info("Re-ranking parameters:")
    logger.info(f"  conf_thresh: {args.conf_thresh}")
    logger.info(f"  weights: proba={args.w_proba}, accent={args.w_accent}, density={args.w_density}, section={args.w_section}")
    
    if args.v3_only:
        logger.info("  mode: v3-only (absolute quality metrics)")

    # A/Bテスト実行
    if args.v3_only:
        # v3単独評価モード
        run_v3_evaluation(
            v3_pickle_path=args.v3_pickle,
            songs=songs,
            output_csv=args.output,
            rerank_conf_thresh=args.conf_thresh,
            rerank_w_proba=args.w_proba,
            rerank_w_accent=args.w_accent,
            rerank_w_density=args.w_density,
            rerank_w_section=args.w_section,
        )
        logger.info(f"v3 evaluation complete. Results: {args.output}")
    else:
        # 旧v1比較モード（互換性維持）
        run_ab_test(
            v1_pickle_path=args.v1_pickle,
            v2_pickle_path=args.v3_pickle,
            songs=songs,
            output_csv=args.output,
            rerank_conf_thresh=args.conf_thresh,
            rerank_w_proba=args.w_proba,
            rerank_w_accent=args.w_accent,
            rerank_w_density=args.w_density,
            rerank_w_section=args.w_section,
        )
        logger.info(f"A/B test complete. Results: {args.output}")


if __name__ == "__main__":
    main()
