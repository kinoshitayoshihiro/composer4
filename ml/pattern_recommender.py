#!/usr/bin/env python3
"""
Pattern Recommender - Stage2パターンからML-based推薦

抽出されたStage2パターンから、与えられた条件に最適なパターンを推薦。

Features:
- 類似度計算（Tempo/Technique/Chord progression/Duration）
- Stage2品質スコア統合（類似度70% + 品質30%）
- Top-K推薦
- キャッシュ機能（高速化）

Usage:
    from ml.pattern_recommender import PatternRecommender, PatternQuery
    
    # Load recommender
    recommender = PatternRecommender("bass", "data/patterns/stage2_bass.pickle")
    
    # Query
    query = PatternQuery(
        tempo=120.0,
        technique="walking",
        duration=16.0,
    )
    
    # Recommend
    results = recommender.recommend(query, top_k=5, min_score=0.6)
"""

import pickle
from utilities.pickle_compat import load as pickle_load_compat
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import sys
import time
from functools import lru_cache
import csv
from datetime import datetime

import numpy as np

# Import ExtractedPattern from extract_stage2_patterns
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
try:
    from extract_stage2_patterns import ExtractedPattern, PatternMetadata, NoteEvent
except ImportError:
    logger_import = logging.getLogger(__name__)
    logger_import.warning("Could not import from extract_stage2_patterns. Using stub classes.")
    
    # Stub classes for type hints
    @dataclass
    class NoteEvent:
        pass
    
    @dataclass
    class PatternMetadata:
        pass
    
    @dataclass
    class ExtractedPattern:
        metadata: Any = None
        notes: List[Any] = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Query Data Class
# =============================================================================

@dataclass
class PatternQuery:
    """パターン検索クエリ"""
    tempo: float
    technique: Optional[str] = None
    duration: Optional[float] = None  # seconds
    chord_progression: Optional[List[str]] = None
    emotion: Optional[str] = None
    
    # Tolerance設定
    tempo_tolerance: float = 20.0  # ±20 BPM
    duration_tolerance: float = 4.0  # ±4 seconds


# =============================================================================
# Pattern Recommender
# =============================================================================

class PatternRecommender:
    """Stage2パターン推薦システム"""
    
    def __init__(self, instrument: str, patterns_path: str | Path):
        """
        Initialize recommender
        
        Args:
            instrument: 楽器名（bass/guitar/strings/melody/chords）
            patterns_path: パターンファイルパス（pickle）
        """
        self.instrument = instrument
        self.patterns_path = Path(patterns_path)
        
        # Load patterns
        self.patterns = self._load_patterns()
        
        # Statistics
        self._compute_statistics()
        
        # パターンインデックス構築（遅延最適化）
        self.pattern_index = self._build_pattern_index()
        
        logger.info(f"Initialized PatternRecommender for {instrument}")
        logger.info(f"  Total patterns: {len(self.patterns)}")
        logger.info(f"  Tempo range: {self.tempo_min:.1f} - {self.tempo_max:.1f} BPM")
        logger.info(f"  Score range: {self.score_min:.3f} - {self.score_max:.3f}")
        logger.info(f"  Index buckets: {len(self.pattern_index)}")
    
    def _load_patterns(self) -> List[Any]:
        """パターンをロード"""
        if not self.patterns_path.exists():
            raise FileNotFoundError(f"Patterns file not found: {self.patterns_path}")
        # 互換性のため、まずrename-aware loaderを試みる。その後通常のpickle.loadにフォールバック。
        try:
            with open(self.patterns_path, "rb") as f:
                data = pickle_load_compat(f)
        except Exception as e:
            logger.debug(f"pickle_compat failed ({e}), falling back to pickle.load")
            with open(self.patterns_path, "rb") as f:
                data = pickle.load(f)
        
        # データ構造確認（dict['patterns']またはlist）
        if isinstance(data, dict) and 'patterns' in data:
            patterns = data['patterns']
            logger.debug(f"Loaded patterns from dict['patterns'] (count: {len(patterns)})")
        elif isinstance(data, list):
            patterns = data
            logger.debug(f"Loaded patterns as list (count: {len(patterns)})")
        else:
            patterns = data
            logger.warning(f"Unknown pattern data structure: {type(data)}")
        
        return patterns
    
    def _compute_statistics(self):
        """パターン統計を計算（dict/ExtractedPattern両対応）"""
        if not self.patterns:
            self.tempo_min = self.tempo_max = 120.0
            self.score_min = self.score_max = 0.0
            self.techniques = set()
            return
        
        # dict or ExtractedPattern判定
        first_pattern = self.patterns[0] if isinstance(self.patterns, list) else list(self.patterns.values())[0]
        
        if isinstance(first_pattern, dict):
            # dict構造（v3_fixed.pickle形式）
            if isinstance(self.patterns, dict):
                # patterns = dict[pattern_id] = {metadata dict}
                # metadataがネストされている場合の対応
                tempos = []
                scores = []
                techniques = set()
                
                for p in self.patterns.values():
                    # metadata内のtempo/section取得
                    meta = p.get('metadata', {})
                    tempo_bin = meta.get('tempo_bin', 'medium')
                    # tempo_binからBPM推定
                    tempo_map = {'slow': 90.0, 'medium': 120.0, 'fast': 150.0}
                    tempo = tempo_map.get(tempo_bin, 120.0)
                    tempos.append(tempo)
                    
                    # sectionをtechniqueとして扱う
                    section = meta.get('section', 'unknown')
                    techniques.add(section.lower())
                    
                    # scoreはavg_confidenceを使用
                    score = meta.get('avg_confidence', 0.0)
                    scores.append(score)
                
                self.techniques = techniques
            else:
                # patterns = list[{metadata dict}]
                tempos = [p.get('tempo', 120.0) for p in self.patterns]
                scores = [p.get('score', 0.0) for p in self.patterns]
                self.techniques = set(p.get('technique', 'unknown') for p in self.patterns)
        else:
            # ExtractedPattern構造（通常形式）
            patterns_list = list(self.patterns.values()) if isinstance(self.patterns, dict) else self.patterns
            tempos = [p.metadata.tempo for p in patterns_list]
            scores = [p.metadata.score for p in patterns_list]
            self.techniques = set(p.metadata.technique for p in patterns_list)
        
        self.tempo_min = min(tempos) if tempos else 120.0
        self.tempo_max = max(tempos) if tempos else 120.0
        self.score_min = min(scores) if scores else 0.0
        self.score_max = max(scores) if scores else 1.0
        
        logger.info(f"  Techniques: {', '.join(sorted(self.techniques))}")
    
    def _build_pattern_index(self) -> dict:
        """
        パターンインデックス構築（遅延最適化）
        
        Tempo範囲（20 BPM刻み）とTechniqueでインデックス化
        検索時にO(N)全探索 → O(1)インデックスアクセスに高速化
        
        Returns:
            {
                (tempo_bucket, technique): [pattern1, pattern2, ...],
                ...
            }
        """
        from collections import defaultdict
        
        index = defaultdict(list)
        
        # dict or list判定
        patterns_list = list(self.patterns.values()) if isinstance(self.patterns, dict) else self.patterns
        
        for pattern in patterns_list:
            # dict or ExtractedPattern判定
            if isinstance(pattern, dict):
                # metadata構造の場合
                meta = pattern.get('metadata', {})
                tempo_bin = meta.get('tempo_bin', 'medium')
                tempo_map = {'slow': 90.0, 'medium': 120.0, 'fast': 150.0}
                tempo = tempo_map.get(tempo_bin, 120.0)
                technique = meta.get('section', 'unknown').lower()
            else:
                tempo = pattern.metadata.tempo
                technique = pattern.metadata.technique
            
            # Tempo bucket（20 BPM刻み: 80-100, 100-120, 120-140, ...）
            tempo_bucket = int(tempo // 20) * 20
            
            # インデックスキー
            key = (tempo_bucket, technique)
            index[key].append(pattern)
        
        logger.debug(f"Built index with {len(index)} buckets")
        
        return dict(index)
    
    def _filter_v3_patterns(self, patterns: list) -> list:
        """
        top1_proba=1.0のパターンのみ抽出（Phase 24横展開）
        
        Args:
            patterns: 候補パターンリスト
        
        Returns:
            top1_proba=1.0のパターンのみのリスト
        """
        v3_patterns = []
        
        for pattern in patterns:
            if isinstance(pattern, dict):
                meta = pattern.get('metadata', {})
                top1_proba = meta.get('top1_proba', 0.0)
            else:
                top1_proba = getattr(pattern.metadata, 'top1_proba', 0.0)
            
            # 1.0 with floating point tolerance
            if top1_proba >= 0.999:
                v3_patterns.append(pattern)
        
        return v3_patterns
    
    def _get_candidate_patterns(self, query: PatternQuery) -> list:
        """
        クエリに基づいて候補パターンを高速取得（インデックス活用）
        
        Args:
            query: 検索クエリ
        
        Returns:
            候補パターンリスト
        """
        if not self.pattern_index:
            # インデックスなし → 全パターン返却
            return self.patterns
        
        candidates = []
        
        # Tempo範囲計算（±tolerance）
        tempo_min = query.tempo - query.tempo_tolerance
        tempo_max = query.tempo + query.tempo_tolerance
        
        # Tempoバケット範囲
        bucket_min = int(tempo_min // 20) * 20
        bucket_max = int(tempo_max // 20) * 20
        
        # 検索対象バケット
        target_buckets = range(bucket_min, bucket_max + 20, 20)
        
        # Technique指定時は該当techniqueのみ、未指定時は全technique
        if query.technique:
            techniques = [query.technique]
        else:
            techniques = list(self.techniques)
        
        # インデックス検索
        for tempo_bucket in target_buckets:
            for technique in techniques:
                key = (tempo_bucket, technique)
                if key in self.pattern_index:
                    candidates.extend(self.pattern_index[key])
        
        logger.debug(f"Index search: {len(candidates)} candidates (from {len(self.patterns)} total)")
        
        return candidates
    
    def recommend(
        self,
        query: PatternQuery,
        top_k: int = 5,
        min_score: float = 0.5,
        similarity_weight: float = 0.7,
        quality_weight: float = 0.3,
        log_latency: bool = False,
        return_margin: bool = False,
        filter_v3_only: bool = False,
        min_proba: float = 0.15,
        min_margin: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """
        パターン推薦
        
        Args:
            query: 検索クエリ
            top_k: 推薦数
            min_score: 最小総合スコア（0.0-1.0）
            similarity_weight: 類似度の重み（デフォルト70%）
            quality_weight: 品質スコアの重み（デフォルト30%）
            log_latency: 遅延をCSVに記録するか（デフォルトFalse）
            return_margin: top-2確率マージンを返すか（Safety閾値用、デフォルトFalse）
            filter_v3_only: top1_proba=1.0のパターンのみ推薦（Phase 24横展開、デフォルトFalse）
            min_proba: 最小確率閾値（絶対KPI評価、filter_v3_only=True時有効）
            min_margin: 最小マージン閾値（絶対KPI評価、filter_v3_only=True時有効）
        
        Returns:
            推薦パターンリスト（スコア降順）
            [
                {
                    "pattern": ExtractedPattern,
                    "similarity": float,
                    "quality": float,
                    "total_score": float,
                    "file": str,
                    "technique": str,
                    "tempo": float,
                    # return_margin=True時のみ以下を追加
                    "top1_score": float,  # 1位のスコア
                    "top2_score": float,  # 2位のスコア
                    "margin": float,      # top1 - top2
                    # filter_v3_only=True時のみ以下を追加
                    "top1_proba": float,  # ML確率（top1）
                    "top2_proba": float,  # ML確率（top2）
                    "proba_margin": float,  # top1_proba - top2_proba
                    "kpi_passed": bool,   # KPI合格フラグ
                },
                ...
            ]
        """
        # 遅延計測開始
        start_time = time.time()
        
        if not self.patterns:
            logger.warning("No patterns available for recommendation")
            return []
        
        # インデックス活用で候補パターン取得（高速化）
        candidate_patterns = self._get_candidate_patterns(query)
        
        # V3フィルタ（top1_proba=1.0のみ）
        if filter_v3_only:
            candidate_patterns = self._filter_v3_patterns(candidate_patterns)
            logger.debug(f"V3 filter (proba=1.0): {len(candidate_patterns)} patterns")
        
        # Calculate scores for candidate patterns (インデックスで絞り込み済み)
        scored_patterns = []
        
        for pattern in candidate_patterns:
            # dict or ExtractedPattern判定
            if isinstance(pattern, dict):
                # dict構造の場合
                meta = pattern.get('metadata', {})
                tempo_bin = meta.get('tempo_bin', 'medium')
                tempo_map = {'slow': 90.0, 'medium': 120.0, 'fast': 150.0}
                pattern_tempo = tempo_map.get(tempo_bin, 120.0)
                pattern_technique = meta.get('section', 'unknown').lower()
                pattern_score = meta.get('avg_confidence', 0.5)
                pattern_file = pattern.get('key', 'unknown')  # pattern IDをfileとして使用
                pattern_duration = 16.0  # デフォルト4バー
            else:
                # ExtractedPattern構造
                pattern_tempo = pattern.metadata.tempo
                pattern_technique = pattern.metadata.technique
                pattern_score = pattern.metadata.score
                pattern_file = pattern.metadata.file
                pattern_duration = pattern.metadata.duration
            
            # Similarity score
            similarity = self._calculate_similarity_dict(
                query, 
                pattern_tempo,
                pattern_technique,
                pattern_duration
            )
            
            # Quality score (Stage2)
            quality = pattern_score
            
            # Total score (weighted)
            total_score = (
                similarity * similarity_weight +
                quality * quality_weight
            )
            
            # Filter by min_score
            if total_score >= min_score:
                scored_patterns.append({
                    "pattern": pattern,
                    "similarity": similarity,
                    "quality": quality,
                    "total_score": total_score,
                    "file": pattern_file,
                    "technique": pattern_technique,
                    "tempo": pattern_tempo,
                    "duration": pattern_duration,
                })
        
        # Sort by total_score (降順)
        scored_patterns.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Top-K
        results = scored_patterns[:top_k]
        
        # filter_v3_only=Trueの場合、KPI評価情報を追加
        if filter_v3_only and results:
            for result in results:
                pattern = result['pattern']
                
                # Extract top1_proba, top2_proba
                if isinstance(pattern, dict):
                    meta = pattern.get('metadata', {})
                    top1_proba = meta.get('top1_proba', 0.0)
                    top2_proba = meta.get('top2_proba', 0.0)
                else:
                    top1_proba = getattr(pattern.metadata, 'top1_proba', 0.0)
                    top2_proba = getattr(pattern.metadata, 'top2_proba', 0.0)
                
                proba_margin = top1_proba - top2_proba
                kpi_passed = (top1_proba >= min_proba) and (proba_margin >= min_margin)
                
                # Add KPI info
                result['top1_proba'] = top1_proba
                result['top2_proba'] = top2_proba
                result['proba_margin'] = proba_margin
                result['kpi_passed'] = kpi_passed
                
                logger.debug(f"KPI eval: proba={top1_proba:.3f}, margin={proba_margin:.3f}, passed={kpi_passed}")
        
        # return_margin=Trueの場合、top-2スコアとマージンを追加
        if return_margin and results:
            # Top-1スコア
            top1_score = results[0]['total_score'] if results else 0.0
            
            # Top-2スコア（2つ以上の結果がある場合）
            top2_score = results[1]['total_score'] if len(results) > 1 else 0.0
            
            # マージン計算
            margin = top1_score - top2_score
            
            # Top-1結果にマージン情報を追加
            if results:
                results[0]['top1_score'] = top1_score
                results[0]['top2_score'] = top2_score
                results[0]['margin'] = margin
                
                logger.debug(f"Top-2 scores: top1={top1_score:.3f}, top2={top2_score:.3f}, margin={margin:.3f}")
        
        logger.debug(f"Recommended {len(results)} patterns (from {len(scored_patterns)} candidates, {len(candidate_patterns)} indexed)")
        
        # 遅延計測終了
        latency_ms = (time.time() - start_time) * 1000
        
        # 遅延ログ記録（オプション）
        if log_latency:
            self._log_latency(latency_ms, query, len(results))
        
        return results
    
    def _log_latency(self, latency_ms: float, query: PatternQuery, num_results: int):
        """遅延をCSVに記録"""
        latency_csv = Path("data/pattern_recommender_latency.csv")
        
        # CSVヘッダー作成（初回のみ）
        if not latency_csv.exists():
            latency_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(latency_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'instrument', 'tempo', 'technique', 
                    'num_patterns', 'num_results', 'latency_ms'
                ])
        
        # 遅延データ追記
        with open(latency_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                self.instrument,
                query.tempo,
                query.technique or 'any',
                len(self.patterns),
                num_results,
                f"{latency_ms:.2f}"
            ])
        
        logger.debug(f"Logged latency: {latency_ms:.2f}ms to {latency_csv}")
    
    def _calculate_similarity_dict(
        self,
        query: PatternQuery,
        pattern_tempo: float,
        pattern_technique: str,
        pattern_duration: float
    ) -> float:
        """
        類似度スコア計算（dict構造対応版）
        
        Args:
            query: 検索クエリ
            pattern_tempo: パターンのテンポ
            pattern_technique: パターンのテクニック
            pattern_duration: パターンの長さ
        
        Returns:
            類似度スコア（0.0-1.0）
        """
        scores = []
        weights = []
        
        # 1. Tempo similarity (必須、高ウェイト) - キャッシュ版使用
        tempo_score = self._calculate_tempo_similarity_cached(
            query.tempo, 
            pattern_tempo, 
            query.tempo_tolerance
        )
        scores.append(tempo_score)
        weights.append(2.0)  # 2x weight
        
        # 2. Technique match (指定時)
        if query.technique:
            technique_score = 1.0 if query.technique == pattern_technique else 0.2
            scores.append(technique_score)
            weights.append(1.5)  # 1.5x weight
        
        # 3. Duration similarity (指定時)
        if query.duration:
            duration_score = self._duration_similarity(
                query.duration,
                pattern_duration,
                query.duration_tolerance
            )
            scores.append(duration_score)
            weights.append(1.0)
        
        # Weighted average
        if not scores:
            return 0.0
        
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum
    
    @lru_cache(maxsize=10000)
    def _calculate_tempo_similarity_cached(
        self, 
        query_tempo: float, 
        pattern_tempo: float, 
        tolerance: float
    ) -> float:
        """Tempo類似度計算（キャッシュ版）"""
        diff = abs(query_tempo - pattern_tempo)
        
        if diff <= tolerance:
            # Linear decay within tolerance
            return 1.0 - (diff / tolerance) * 0.5  # 0.5-1.0
        else:
            # Exponential decay outside tolerance
            excess = diff - tolerance
            return max(0.0, 0.5 * np.exp(-excess / tolerance))
    
    def _calculate_similarity(
        self,
        query: PatternQuery,
        pattern: Any,
    ) -> float:
        """
        類似度スコア計算（0.0-1.0）
        
        Score components:
        1. Tempo similarity (必須)
        2. Technique match (optional)
        3. Duration similarity (optional)
        4. Chord progression similarity (optional, Piano only)
        """
        scores = []
        weights = []
        
        # 1. Tempo similarity (必須、高ウェイト) - キャッシュ版使用
        tempo_score = self._calculate_tempo_similarity_cached(
            query.tempo, 
            pattern.metadata.tempo, 
            query.tempo_tolerance
        )
        scores.append(tempo_score)
        weights.append(2.0)  # 2x weight
        
        # 2. Technique match (指定時)
        if query.technique:
            technique_score = 1.0 if query.technique == pattern.metadata.technique else 0.2
            scores.append(technique_score)
            weights.append(1.5)  # 1.5x weight
        
        # 3. Duration similarity (指定時)
        if query.duration:
            duration_score = self._duration_similarity(
                query.duration,
                pattern.metadata.duration,
                query.duration_tolerance
            )
            scores.append(duration_score)
            weights.append(1.0)
        
        # 4. Chord progression similarity (Piano only)
        if query.chord_progression and pattern.metadata.chord_progression:
            chord_score = self._chord_similarity(
                query.chord_progression,
                pattern.metadata.chord_progression
            )
            scores.append(chord_score)
            weights.append(1.0)
        
        # Weighted average
        if not scores:
            return 0.0
        
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum
    
    def _tempo_similarity(self, query_tempo: float, pattern_tempo: float, tolerance: float) -> float:
        """
        Tempo類似度
        
        tolerance範囲内なら高スコア、範囲外は急速に減少
        """
        diff = abs(query_tempo - pattern_tempo)
        
        if diff <= tolerance:
            # Linear decay within tolerance
            return 1.0 - (diff / tolerance) * 0.5  # 0.5-1.0
        else:
            # Exponential decay outside tolerance
            excess = diff - tolerance
            return max(0.0, 0.5 * np.exp(-excess / tolerance))
    
    def _duration_similarity(self, query_duration: float, pattern_duration: float, tolerance: float) -> float:
        """Duration類似度"""
        diff = abs(query_duration - pattern_duration)
        
        if diff <= tolerance:
            return 1.0 - (diff / tolerance) * 0.5
        else:
            excess = diff - tolerance
            return max(0.0, 0.5 * np.exp(-excess / tolerance))
    
    def _chord_similarity(
        self,
        query_chords: List[str],
        pattern_chords: List[str],
    ) -> float:
        """
        Chord progression類似度（Jaccard index）
        
        例: ["C", "G", "Am", "F"] vs ["C", "Am", "F", "G"]
        """
        if not query_chords or not pattern_chords:
            return 0.5  # Neutral
        
        set1 = set(query_chords)
        set2 = set(pattern_chords)
        
        intersection = set1 & set2
        union = set1 | set2
        
        if not union:
            return 0.5
        
        jaccard = len(intersection) / len(union)
        
        # Also consider sequence similarity (simple overlap ratio)
        min_len = min(len(query_chords), len(pattern_chords))
        matches = sum(1 for i in range(min_len) if query_chords[i] == pattern_chords[i])
        sequence_sim = matches / min_len if min_len > 0 else 0.0
        
        # Combined score (70% Jaccard, 30% sequence)
        return jaccard * 0.7 + sequence_sim * 0.3
    
    def get_techniques(self) -> List[str]:
        """利用可能なTechniqueリストを取得"""
        return sorted(self.techniques)
    
    def filter_by_technique(self, technique: str) -> List[Any]:
        """Techniqueでフィルタ"""
        return [p for p in self.patterns if p.metadata.technique == technique]
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        if not self.patterns:
            return {}
        
        technique_counts = {}
        for p in self.patterns:
            tech = p.metadata.technique
            technique_counts[tech] = technique_counts.get(tech, 0) + 1
        
        scores = [p.metadata.score for p in self.patterns]
        tempos = [p.metadata.tempo for p in self.patterns]
        durations = [p.metadata.duration for p in self.patterns]
        
        return {
            "total_patterns": len(self.patterns),
            "techniques": technique_counts,
            "tempo": {
                "min": float(np.min(tempos)),
                "max": float(np.max(tempos)),
                "mean": float(np.mean(tempos)),
                "std": float(np.std(tempos)),
            },
            "duration": {
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
                "mean": float(np.mean(durations)),
                "std": float(np.std(durations)),
            },
            "score": {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            },
        }


# =============================================================================
# Recommender Factory（シングルトン管理）
# =============================================================================

class RecommenderFactory:
    """楽器別Recommenderファクトリー（シングルトン）"""
    
    _instances: Dict[str, PatternRecommender] = {}
    _base_dir = Path(__file__).parent.parent / "data" / "patterns"
    
    @classmethod
    def get_recommender(cls, instrument: str) -> PatternRecommender:
        """
        Recommenderインスタンス取得（シングルトン）
        
        Args:
            instrument: melody/chords/bass/guitar/strings
        
        Returns:
            PatternRecommender instance
        """
        if instrument not in cls._instances:
            patterns_path = cls._base_dir / f"stage2_{instrument}.pickle"
            
            if not patterns_path.exists():
                raise FileNotFoundError(
                    f"Pattern file not found: {patterns_path}\n"
                    f"Run: python scripts/extract_stage2_patterns.py --instrument {instrument}"
                )
            
            cls._instances[instrument] = PatternRecommender(instrument, patterns_path)
        
        return cls._instances[instrument]
    
    @classmethod
    def clear_cache(cls):
        """キャッシュをクリア"""
        cls._instances.clear()


# =============================================================================
# CLI / Testing
# =============================================================================

def main():
    """CLI testing interface"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Test Pattern Recommender")
    parser.add_argument("--instrument", required=True, help="Instrument name")
    parser.add_argument("--tempo", type=float, default=120.0, help="Query tempo")
    parser.add_argument("--technique", help="Query technique")
    parser.add_argument("--duration", type=float, help="Query duration (seconds)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum total score")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    
    args = parser.parse_args()
    
    # Load recommender
    try:
        recommender = RecommenderFactory.get_recommender(args.instrument)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Show statistics
    if args.stats:
        stats = recommender.get_statistics()
        print("\n📊 Pattern Statistics:")
        print(json.dumps(stats, indent=2))
        return
    
    # Create query
    query = PatternQuery(
        tempo=args.tempo,
        technique=args.technique,
        duration=args.duration,
    )
    
    # Recommend
    print(f"\n🔍 Query:")
    print(f"  Tempo: {query.tempo} BPM")
    if query.technique:
        print(f"  Technique: {query.technique}")
    if query.duration:
        print(f"  Duration: {query.duration:.1f}s")
    
    results = recommender.recommend(query, top_k=args.top_k, min_score=args.min_score)
    
    print(f"\n✅ Recommendations ({len(results)}):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['file']}")
        print(f"   Technique: {result['technique']}")
        print(f"   Tempo: {result['tempo']:.1f} BPM")
        print(f"   Duration: {result['duration']:.1f}s")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Quality: {result['quality']:.3f}")
        print(f"   Total Score: {result['total_score']:.3f}")


if __name__ == "__main__":
    main()
