#!/usr/bin/env python3
"""
TrafficSplitter: v3/v1 Shadow Testing Traffic Manager

v3（ML-based）とv1（Rule-based）のトラフィック分割と並行実行を管理。
リアルタイムKPI比較、自動フォールバック機能を提供。

Usage:
    splitter = TrafficSplitter(
        v3_pickle='data/patterns/stage2_guitar_v3_fixed.pickle',
        v1_pickle='data/patterns/stage2_guitar.pickle',
        v3_ratio=0.9  # 90% v3, 10% v1
    )
    
    pattern, comparison = splitter.route_and_compare(
        chord_root='C', tempo=120, section='Chorus'
    )
"""

import logging
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Sequence, Set
import csv

import numpy as np

from ml.pattern_recommender import PatternRecommender, PatternQuery


# =============================================================================
# Pattern Normalization Layer (dict/ExtractedPattern両対応)
# =============================================================================

@dataclass
class NormalizedPattern:
    """正規化されたパターン表現（v3 dict / v1 ExtractedPattern 両対応）"""
    pattern_id: str
    rhythm: Optional[str]
    voicing: Sequence[int]
    pc_set: Optional[Set[int]]  # 事前計算されたPC集合（0-11）
    tempo_bin: Optional[float]
    section: Optional[str]
    technique: Optional[str]
    accent_profile: Optional[Sequence[float]]  # 拍アクセントベクトル（16スロット想定）
    density_ql: Optional[float]  # Quarter-length密度


def _safe_get(d: Dict, *keys, default=None):
    """安全な多段階dict取得"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _normalize_pattern_obj(p: Any) -> NormalizedPattern:
    """
    dict形式とExtractedPattern形式の両方を正規化して統一表現に変換
    
    v3 dict例:
      {
        "pattern_id": "STRUM8_OPEN_B_...0123",
        "rhythm": "strum8_open",
        "voicing": [0,4,7],
        "metadata": {"tempo_bin": "medium", "section": "Chorus", "technique": "strum"},
        "accent_profile": [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]
      }
    
    v1 ExtractedPattern例:
      obj.pattern_id, obj.rhythm, obj.voicing, obj.tempo_bin, obj.section, obj.technique
    """
    if isinstance(p, dict):
        # v3 dict形式
        pattern_id = p.get("pattern_id") or p.get("key") or p.get("id") or ""
        rhythm     = p.get("rhythm") or _safe_get(p, "pattern", "rhythm")
        voicing    = p.get("voicing") or _safe_get(p, "pattern", "voicing") or []
        
        # metadataから取得
        meta       = p.get("metadata", {}) or {}
        tempo_bin_raw = meta.get("tempo_bin") or p.get("tempo_bin")
        
        # tempo_bin文字列→数値変換
        tempo_map = {'slow': 90.0, 'medium': 120.0, 'fast': 150.0}
        if isinstance(tempo_bin_raw, str):
            tempo_bin = tempo_map.get(tempo_bin_raw, 120.0)
        else:
            tempo_bin = float(tempo_bin_raw) if tempo_bin_raw else 120.0
        
        section    = meta.get("section") or p.get("section")
        technique  = meta.get("technique") or p.get("technique")
        
        # accent_profile取得
        acc = p.get("accent_profile") or _safe_get(p, "pattern", "accent_profile")
        if acc is not None:
            try:
                acc = list(acc)
            except Exception:
                acc = None
        
        # density取得
        density = meta.get("density_ql_per_bar") or p.get("density_ql_per_bar")
        if density is not None:
            density = float(density)
        
        # PC集合取得（事前計算されていればそれを使用）
        pc_set_raw = p.get("pc_set")
        if pc_set_raw:
            pc_set = set(pc_set_raw) if isinstance(pc_set_raw, (list, tuple)) else pc_set_raw
        else:
            pc_set = None
        
        return NormalizedPattern(
            pattern_id=pattern_id,
            rhythm=rhythm if rhythm else "unknown",
            voicing=list(voicing) if voicing else [],
            pc_set=pc_set,
            tempo_bin=tempo_bin,
            section=section,
            technique=technique,
            accent_profile=acc,
            density_ql=density
        )
    else:
        # ExtractedPattern形式
        pattern_id = getattr(p, "pattern_id", "")
        rhythm     = getattr(p, "rhythm", None)
        voicing    = getattr(p, "voicing", []) or []
        tempo_bin  = getattr(p, "tempo_bin", None)
        section    = getattr(p, "section", None)
        technique  = getattr(p, "technique", None)
        acc        = getattr(p, "accent_profile", None)
        density    = getattr(p, "density_ql_per_bar", None)
        pc_set_raw = getattr(p, "pc_set", None)
        
        if pc_set_raw:
            pc_set = set(pc_set_raw) if isinstance(pc_set_raw, (list, tuple)) else pc_set_raw
        else:
            pc_set = None
        
        return NormalizedPattern(
            pattern_id=pattern_id,
            rhythm=rhythm if rhythm else "unknown",
            voicing=list(voicing) if voicing else [],
            pc_set=pc_set,
            tempo_bin=float(tempo_bin) if tempo_bin else 120.0,
            section=section,
            technique=technique,
            accent_profile=acc,
            density_ql=float(density) if density else None
        )


def _fallback_accent_profile(length: int = 16) -> Sequence[float]:
    """デフォルトのアクセントプロファイル（4/4拍子のダウンビート強調）"""
    v = np.zeros(length, dtype=float)
    if length >= 16:
        v[0] = 1.0; v[4] = 0.6; v[8] = 0.9; v[12] = 0.6
    elif length >= 8:
        v[0] = 1.0; v[4] = 0.8
    else:
        v[0] = 1.0
    return v.tolist()


def _roll_to_max_cosine(target: Sequence[float], pattern: Sequence[float]) -> Tuple[float, int]:
    """
    位相を0..len-1で巡回シフトして最大コサイン類似度を探索
    
    Returns:
        (max_similarity, best_shift)
    """
    t = np.asarray(target, dtype=float)
    a = np.asarray(pattern, dtype=float)
    
    # 長さ調整（ゼロパディング）
    if t.size != a.size:
        n = max(t.size, a.size)
        t = np.pad(t, (0, n - t.size))
        a = np.pad(a, (0, n - a.size))
    
    best_sim, best_shift = -1.0, 0
    denom_t = np.linalg.norm(t) + 1e-9
    
    for shift in range(a.size):
        rolled = np.roll(a, shift)
        sim = float(t @ rolled) / ((np.linalg.norm(rolled) + 1e-9) * denom_t)
        if sim > best_sim:
            best_sim, best_shift = sim, shift
    
    return best_sim, best_shift


@dataclass
class ComparisonResult:
    """v3/v1比較結果"""
    timestamp: str
    primary_version: str  # 'v3' or 'v1'
    
    # 再現性メタデータ（トラブル時の切り戻し対応）
    run_id: str  # UUID for this execution session
    git_sha: str  # Git commit SHA (short)
    v3_model_sha256: str  # v3 pickle SHA256 (first 16 chars)
    v1_model_sha256: str  # v1 pickle SHA256 (first 16 chars)
    song_id: str  # Unique song identifier for tracking
    
    # 入力パラメータ
    chord_root: str
    tempo: float
    section: str
    key: str
    chord_type: str
    time_signature: str  # "3/4", "4/4", "6/8" など
    
    # v3結果
    v3_pattern_id: str
    v3_accent_score: float
    v3_accent_score_norm16: float  # 16スロット正規化後のaccent score
    v3_accent_phase: int  # 位相最適化シフト量（0..N-1）
    v3_chord_fit: float
    v3_density: float
    v3_ml_used: int
    v3_top1_proba: float
    v3_top2_proba: float  # Safety閾値用（2位スコア）
    v3_margin: float  # Safety閾値用（1位-2位）
    v3_safety_triggered: int  # 1 if safety threshold triggered, 0 otherwise
    v3_safety_reason: str  # 'low_p1' or 'low_margin' or ''
    v3_latency_ms: float
    v3_error: str
    
    # v1結果
    v1_pattern_id: str
    v1_accent_score: float
    v1_accent_score_norm16: float  # 16スロット正規化後のaccent score
    v1_accent_phase: int  # 位相最適化シフト量（0..N-1）
    v1_chord_fit: float
    v1_density: float
    v1_latency_ms: float
    v1_error: str
    
    # 比較メトリクス
    accent_delta: float  # v3 - v1
    chord_delta: float
    density_delta: float
    latency_delta_ms: float
    v3_wins: int  # 1 if v3 better, 0 otherwise
    pattern_agreement: int  # 1 if same pattern selected


class TrafficSplitter:
    """
    v3/v1トラフィック分割と並行実行マネージャー
    
    Features:
    - Configurable traffic split ratio (default 90% v3, 10% v1)
    - Always execute both versions for KPI comparison
    - Return primary version result to user
    - Log comparison results for monitoring
    - Export Prometheus metrics
    - Safe-Kit fallback support (emergency safety patterns)
    """
    
    def __init__(
        self,
        v3_pickle_path: str,
        v1_pickle_path: str,
        v3_ratio: float = 0.9,
        log_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        gate_config_path: Optional[str] = None,
        safe_kit_path: Optional[str] = None,
        enable_auto_recovery: bool = False,
        auto_recovery_window: int = 32,
        auto_recovery_threshold: int = 6,
        auto_recovery_cooldown: int = 16
    ):
        """
        Args:
            v3_pickle_path: v3 pattern pickle path
            v1_pickle_path: v1 pattern pickle path
            v3_ratio: Primary traffic ratio for v3 (0.0-1.0)
            log_path: CSV log file path for comparison results
            logger: Optional logger instance
            gate_config_path: KPI gate configuration YAML path (optional)
            safe_kit_path: Safe-Kit pattern YAML path (optional, default: data/patterns/safe_kit_guitar.yaml)
            enable_auto_recovery: Enable auto-recovery (v3↔v1 bidirectional switching)
            auto_recovery_window: Window size for breach counting (bars)
            auto_recovery_threshold: Max breaches allowed in window
            auto_recovery_cooldown: Cooldown period after version switch (bars)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.v3_ratio = v3_ratio
        
        # Generate reproducibility metadata
        import uuid
        import hashlib
        import subprocess
        
        self.run_id = str(uuid.uuid4())[:8]  # Short UUID for this session
        
        # Git SHA (short)
        try:
            git_sha = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except Exception:
            git_sha = 'unknown'
        self.git_sha = git_sha
        
        # Model SHA256 (first 16 chars)
        self.v3_model_sha256 = self._compute_file_sha256(v3_pickle_path)[:16]
        self.v1_model_sha256 = self._compute_file_sha256(v1_pickle_path)[:16]
        
        self.logger.info(f"Session: run_id={self.run_id}, git={self.git_sha}, v3_sha={self.v3_model_sha256}")
        
        # Load KPI gate configuration
        self.gate_config = self._load_gate_config(gate_config_path)
        
        # Load Safe-Kit patterns
        self.safe_kit_patterns = self._load_safe_kit(safe_kit_path)
        if self.safe_kit_patterns:
            self.logger.info(f"Safe-Kit loaded: {len(self.safe_kit_patterns.get('patterns', {}))} patterns")
        
        # Load pattern recommenders
        self.logger.info(f"Loading v3: {v3_pickle_path}")
        self.v3_recommender = PatternRecommender('guitar', v3_pickle_path)
        
        self.logger.info(f"Loading v1: {v1_pickle_path}")
        self.v1_recommender = PatternRecommender('guitar', v1_pickle_path)
        
        # Logging setup
        self.log_path = Path(log_path) if log_path else None
        self.results: List[ComparisonResult] = []
        
        # Initialize CSV log file
        if self.log_path:
            self._initialize_log_file()
        
        # Statistics counters
        self.stats = {
            'total_requests': 0,
            'v3_primary_count': 0,
            'v1_primary_count': 0,
            'v3_wins': 0,
            'v1_wins': 0,
            'ties': 0,
            'v3_errors': 0,
            'v1_errors': 0
        }
        
        # Auto-Recovery setup
        self.enable_auto_recovery = enable_auto_recovery
        if enable_auto_recovery:
            from ml.auto_recovery import AutoRecoveryManager
            self.auto_recovery = AutoRecoveryManager(
                window_size=auto_recovery_window,
                threshold=auto_recovery_threshold,
                cooldown=auto_recovery_cooldown,
                initial_version='v3',
                logger=self.logger
            )
            self.logger.info(
                f"Auto-Recovery enabled: window={auto_recovery_window}, "
                f"threshold={auto_recovery_threshold}, cooldown={auto_recovery_cooldown}"
            )
        else:
            self.auto_recovery = None
        
        self.logger.info(
            f"TrafficSplitter initialized: v3_ratio={v3_ratio:.1%}, "
            f"log_path={log_path}"
        )
    
    def route_and_compare(
        self,
        chord_root: str,
        tempo: float,
        section: str,
        key: str = "C",
        chord_type: str = "maj",
        time_signature: str = "4/4",
        ideal_accent: Optional[np.ndarray] = None
    ) -> Tuple[Dict, ComparisonResult]:
        """
        Route request to primary version and compare with shadow.
        
        Returns:
            (primary_pattern, comparison_result)
        """
        # Traffic routing decision (may be overridden by auto-recovery)
        primary_version = self._route_traffic()
        
        # Execute both versions in parallel (conceptually)
        v3_result = self._execute_v3(
            chord_root, tempo, section, key, chord_type, time_signature, ideal_accent
        )
        v1_result = self._execute_v1(
            chord_root, tempo, section, key, chord_type, time_signature, ideal_accent
        )
        
        # Compute comparison metrics
        comparison = self._compute_comparison(
            primary_version, chord_root, tempo, section, key, chord_type,
            time_signature, v3_result, v1_result
        )
        
        # Auto-Recovery判定（KPIゲート違反チェック & バージョン切替）
        if self.auto_recovery:
            from ml.auto_recovery import check_kpi_breach
            
            # v3のKPI違反をチェック
            is_breach = check_kpi_breach(v3_result, self.gate_config, section)
            self.auto_recovery.add_result(is_breach)
            
            # バージョン切替判定
            new_version = self.auto_recovery.should_switch_version()
            if new_version:
                self.auto_recovery.switch_version(new_version)
                # v3_ratioを動的に変更
                if new_version == 'v1':
                    self.v3_ratio = 0.0  # v1に完全切替
                    self.logger.warning(f"Auto-Recovery: Fallback to v1 (v3_ratio=0%)")
                elif new_version == 'v3':
                    self.v3_ratio = 0.9  # v3に復帰（90%）
                    self.logger.warning(f"Auto-Recovery: Recovery to v3 (v3_ratio=90%)")
            
            # クールダウンカウンター減算
            self.auto_recovery.tick_cooldown()
        
        # Update statistics
        self._update_stats(primary_version, comparison)
        
        # Log to CSV
        if self.log_path:
            self._log_comparison(comparison)
        
        # Store result
        self.results.append(comparison)
        
        # Return primary version pattern
        primary_pattern = v3_result['pattern'] if primary_version == 'v3' else v1_result['pattern']
        
        return primary_pattern, comparison
    
    def _route_traffic(self) -> str:
        """Determine primary version based on traffic split ratio"""
        return 'v3' if random.random() < self.v3_ratio else 'v1'
    
    def _execute_v3(
        self,
        chord_root: str,
        tempo: float,
        section: str,
        key: str,
        chord_type: str,
        time_signature: str,
        ideal_accent: Optional[np.ndarray]
    ) -> Dict:
        """Execute v3 recommendation"""
        start_time = time.time()
        
        try:
            # Create PatternQuery
            query = PatternQuery(
                tempo=tempo,
                technique=section.lower(),  # Use section as technique
                duration=16.0,  # Default 4 bars
                tempo_tolerance=20.0
            )
            
            # Recommend patterns (top-2 with margin for safety check)
            results = self.v3_recommender.recommend(
                query=query,
                top_k=2,  # Top-2 for margin calculation
                min_score=0.0,
                log_latency=False,
                return_margin=True  # Safety閾値チェック用
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if not results:
                raise ValueError("No patterns returned from v3")
            
            # Extract pattern from recommendation result
            # results[0] = {"pattern": <actual_pattern>, "similarity": ..., "quality": ..., "margin": ..., ...}
            recommendation = results[0]
            pattern = recommendation.get("pattern", recommendation)  # Fallback to full object if no "pattern" key
            
            # Safety閾値チェック（min_proba, min_margin）
            safety_triggered = False
            safety_reason = None
            
            if 'top1_score' in recommendation and 'margin' in recommendation:
                top1_score = recommendation['top1_score']
                margin = recommendation['margin']
                
                # gate_prod.yamlから閾値取得
                min_proba = self.gate_config.get('safety', {}).get('min_proba', 0.15)
                min_margin = self.gate_config.get('safety', {}).get('min_margin', 0.08)
                
                # Safety条件: (p1 < min_proba) OR (margin < min_margin)
                if top1_score < min_proba:
                    safety_triggered = True
                    safety_reason = 'low_p1'
                    self.logger.warning(f"Safety triggered: low_p1 (p1={top1_score:.3f} < {min_proba})")
                elif margin < min_margin:
                    safety_triggered = True
                    safety_reason = 'low_margin'
                    self.logger.warning(f"Safety triggered: low_margin (margin={margin:.3f} < {min_margin})")
                
                # TODO: Safety kitへのフォールバック実装
                # if safety_triggered:
                #     pattern = self._get_safe_kit_pattern(chord_root, section)
                
                # Safe-Kit Fallback実装（Phase 23.5）
                if safety_triggered and self.safe_kit_patterns:
                    safe_pattern = self._get_safe_kit_pattern(chord_root, section)
                    if safe_pattern:
                        self.logger.info(
                            f"Safe-Kit fallback triggered: reason={safety_reason}, "
                            f"section={section}, pattern={safe_pattern.get('pattern_id', 'unknown')}"
                        )
                        pattern = safe_pattern
                        norm_pattern = _normalize_pattern_obj(pattern)
                        
                        # Safe-Kitパターンで再計算
                        accent_score, accent_phase = self._compute_accent_score_v2(
                            norm_pattern.accent_profile,
                            ideal_accent
                        )
                        voicing_or_pc = norm_pattern.pc_set if norm_pattern.pc_set else norm_pattern.voicing
                        chord_fit = self._compute_chord_fit_v3(
                            voicing_or_pc,
                            norm_pattern.rhythm,
                            chord_root,
                            chord_type
                        )
                        density = norm_pattern.density_ql if norm_pattern.density_ql else 0.0
            
            # Normalize pattern (dict/ExtractedPattern両対応)
            norm_pattern = _normalize_pattern_obj(pattern)
            
            # Generate ideal accent if not provided
            if ideal_accent is None:
                ideal_accent = self._generate_ideal_accent(section, tempo)
            
            # Compute KPIs using v3 methods (Chord Fit v3 with beat-position awareness)
            accent_score, accent_phase = self._compute_accent_score_v2(
                norm_pattern.accent_profile,
                ideal_accent
            )
            # Use pre-computed PC set if available, otherwise use voicing
            voicing_or_pc = norm_pattern.pc_set if norm_pattern.pc_set else norm_pattern.voicing
            chord_fit = self._compute_chord_fit_v3(
                voicing_or_pc,
                norm_pattern.rhythm,  # Pass rhythm for beat-position analysis
                chord_root,
                chord_type
            )
            density = norm_pattern.density_ql if norm_pattern.density_ql else 0.0
            
            # 16スロット基準に正規化されたaccent score
            accent_score_norm16 = self._normalize_accent_score_to_16(
                accent_score,
                norm_pattern.accent_profile,
                ideal_accent,
                time_signature
            )
            
            return {
                'pattern': pattern,
                'pattern_id': norm_pattern.pattern_id,
                'accent_score': accent_score,
                'accent_score_norm16': accent_score_norm16,
                'accent_phase': accent_phase,
                'chord_fit': chord_fit,
                'density': density,
                'ml_used': 1,  # v3 is always ML
                'top1_proba': recommendation.get('top1_score', 0.0),
                'top2_proba': recommendation.get('top2_score', 0.0),
                'margin': recommendation.get('margin', 0.0),
                'safety_triggered': 1 if safety_triggered else 0,
                'safety_reason': safety_reason or '',
                'latency_ms': latency_ms,
                'error': ''
            }
        
        except Exception as e:
            self.logger.error(f"v3 execution error: {e}")
            return {
                'pattern': {},
                'pattern_id': 'error',
                'accent_score': 0.0,
                'accent_phase': 0,
                'chord_fit': 0.0,
                'density': 0.0,
                'ml_used': 0,
                'top1_proba': 0.0,
                'latency_ms': 0.0,
                'error': str(e)
            }
    
    def _execute_v1(
        self,
        chord_root: str,
        tempo: float,
        section: str,
        key: str,
        chord_type: str,
        time_signature: str,
        ideal_accent: Optional[np.ndarray]
    ) -> Dict:
        """Execute v1 recommendation"""
        start_time = time.time()
        
        try:
            # Create PatternQuery
            query = PatternQuery(
                tempo=tempo,
                technique=section.lower(),
                duration=16.0,
                tempo_tolerance=20.0
            )
            
            # Recommend patterns
            results = self.v1_recommender.recommend(
                query=query,
                top_k=1,
                min_score=0.0,
                log_latency=False
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if not results:
                raise ValueError("No patterns returned from v1")
            
            # Extract pattern from recommendation result
            recommendation = results[0]
            pattern = recommendation.get("pattern", recommendation)  # Fallback to full object if no "pattern" key
            
            # Normalize pattern (dict/ExtractedPattern両対応)
            norm_pattern = _normalize_pattern_obj(pattern)
            
            # Generate ideal accent if not provided
            if ideal_accent is None:
                ideal_accent = self._generate_ideal_accent(section, tempo)
            
            # Compute KPIs using v3 methods (Chord Fit v3 with beat-position awareness)
            accent_score, accent_phase = self._compute_accent_score_v2(
                norm_pattern.accent_profile,
                ideal_accent
            )
            # Use pre-computed PC set if available, otherwise use voicing
            voicing_or_pc = norm_pattern.pc_set if norm_pattern.pc_set else norm_pattern.voicing
            chord_fit = self._compute_chord_fit_v3(
                voicing_or_pc,
                norm_pattern.rhythm,  # Pass rhythm for beat-position analysis
                chord_root,
                chord_type
            )
            density = norm_pattern.density_ql if norm_pattern.density_ql else 0.0
            
            # 16スロット基準に正規化されたaccent score
            accent_score_norm16 = self._normalize_accent_score_to_16(
                accent_score,
                norm_pattern.accent_profile,
                ideal_accent,
                time_signature
            )
            
            return {
                'pattern': pattern,
                'pattern_id': norm_pattern.pattern_id,
                'accent_score': accent_score,
                'accent_score_norm16': accent_score_norm16,
                'accent_phase': accent_phase,
                'chord_fit': chord_fit,
                'density': density,
                'latency_ms': latency_ms,
                'error': ''
            }
        
        except Exception as e:
            self.logger.error(f"v1 execution error: {e}")
            return {
                'pattern': {},
                'pattern_id': 'error',
                'accent_score': 0.0,
                'accent_phase': 0,
                'chord_fit': 0.0,
                'density': 0.0,
                'latency_ms': 0.0,
                'error': str(e)
            }
    
    def _compute_comparison(
        self,
        primary_version: str,
        chord_root: str,
        tempo: float,
        section: str,
        key: str,
        chord_type: str,
        time_signature: str,
        v3_result: Dict,
        v1_result: Dict
    ) -> ComparisonResult:
        """Compute comparison metrics"""
        
        accent_delta = v3_result['accent_score'] - v1_result['accent_score']
        chord_delta = v3_result['chord_fit'] - v1_result['chord_fit']
        density_delta = v3_result['density'] - v1_result['density']
        latency_delta = v3_result['latency_ms'] - v1_result['latency_ms']
        
        # Determine winner (based on accent score)
        if v3_result['accent_score'] > v1_result['accent_score']:
            v3_wins = 1
        else:
            v3_wins = 0
        
        # Pattern agreement
        pattern_agreement = int(v3_result['pattern_id'] == v1_result['pattern_id'])
        
        return ComparisonResult(
            timestamp=datetime.now().isoformat(),
            primary_version=primary_version,
            # 再現性メタデータ
            run_id=self.run_id,
            git_sha=self.git_sha,
            v3_model_sha256=self.v3_model_sha256,
            v1_model_sha256=self.v1_model_sha256,
            song_id=f"{section}_{key}_{tempo:.0f}",  # Simple song identifier
            # 入力パラメータ
            chord_root=chord_root,
            tempo=tempo,
            section=section,
            key=key,
            chord_type=chord_type,
            time_signature=time_signature,
            # v3結果
            v3_pattern_id=v3_result['pattern_id'],
            v3_accent_score=v3_result['accent_score'],
            v3_accent_score_norm16=v3_result.get('accent_score_norm16', v3_result['accent_score']),
            v3_accent_phase=v3_result.get('accent_phase', 0),
            v3_chord_fit=v3_result['chord_fit'],
            v3_density=v3_result['density'],
            v3_ml_used=v3_result['ml_used'],
            v3_top1_proba=v3_result['top1_proba'],
            v3_top2_proba=v3_result.get('top2_proba', 0.0),
            v3_margin=v3_result.get('margin', 0.0),
            v3_safety_triggered=v3_result.get('safety_triggered', 0),
            v3_safety_reason=v3_result.get('safety_reason', ''),
            v3_latency_ms=v3_result['latency_ms'],
            v3_error=v3_result['error'],
            # v1結果
            v1_pattern_id=v1_result['pattern_id'],
            v1_accent_score=v1_result['accent_score'],
            v1_accent_score_norm16=v1_result.get('accent_score_norm16', v1_result['accent_score']),
            v1_accent_phase=v1_result.get('accent_phase', 0),
            v1_chord_fit=v1_result['chord_fit'],
            v1_density=v1_result['density'],
            v1_latency_ms=v1_result['latency_ms'],
            v1_error=v1_result['error'],
            # 比較メトリクス
            accent_delta=accent_delta,
            chord_delta=chord_delta,
            density_delta=density_delta,
            latency_delta_ms=latency_delta,
            v3_wins=v3_wins,
            pattern_agreement=pattern_agreement
        )
    
    def _update_stats(self, primary_version: str, comparison: ComparisonResult):
        """Update internal statistics"""
        self.stats['total_requests'] += 1
        
        if primary_version == 'v3':
            self.stats['v3_primary_count'] += 1
        else:
            self.stats['v1_primary_count'] += 1
        
        if comparison.v3_wins:
            self.stats['v3_wins'] += 1
        elif comparison.v3_accent_score < comparison.v1_accent_score:
            self.stats['v1_wins'] += 1
        else:
            self.stats['ties'] += 1
        
        if comparison.v3_error:
            self.stats['v3_errors'] += 1
        if comparison.v1_error:
            self.stats['v1_errors'] += 1
    
    def _initialize_log_file(self):
        """Initialize CSV log file with headers"""
        if not self.log_path.exists():
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    field.name for field in ComparisonResult.__dataclass_fields__.values()
                ])
                writer.writeheader()
            self.logger.info(f"Initialized log file: {self.log_path}")
    
    def _log_comparison(self, comparison: ComparisonResult):
        """Append comparison result to CSV log"""
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    field.name for field in ComparisonResult.__dataclass_fields__.values()
                ])
                writer.writerow(asdict(comparison))
        except Exception as e:
            self.logger.error(f"Failed to log comparison: {e}")
    
    def _compute_accent_score(
        self,
        rhythm: List[int],
        ideal_accent: np.ndarray,
        phase_shift: int = 0
    ) -> float:
        """Compute accent match score (cosine similarity)"""
        if len(rhythm) != len(ideal_accent):
            return 0.0
        
        rhythm_arr = np.array(rhythm, dtype=float)
        if phase_shift > 0:
            rhythm_arr = np.roll(rhythm_arr, phase_shift)
        
        norm_rhythm = np.linalg.norm(rhythm_arr)
        norm_ideal = np.linalg.norm(ideal_accent)
        
        if norm_rhythm == 0 or norm_ideal == 0:
            return 0.0
        
        return float(np.dot(rhythm_arr, ideal_accent) / (norm_rhythm * norm_ideal))
    
    def _compute_chord_fit(
        self,
        pitches: List[int],
        chord_root: str,
        chord_type: str
    ) -> float:
        """Compute chord fit score (chord tone ratio)"""
        root_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        root_pitch = root_map.get(chord_root, 0)
        
        # Define chord tones
        if 'maj' in chord_type:
            chord_tones = {root_pitch, (root_pitch + 4) % 12, (root_pitch + 7) % 12}
        else:  # min
            chord_tones = {root_pitch, (root_pitch + 3) % 12, (root_pitch + 7) % 12}
        
        # Calculate chord tone ratio
        valid_pitches = [p for p in pitches if p > 0]
        if not valid_pitches:
            return 0.5
        
        chord_tone_count = sum(1 for p in valid_pitches if (p % 12) in chord_tones)
        return chord_tone_count / len(valid_pitches)
    
    def _generate_ideal_accent(self, section: str, tempo: float) -> np.ndarray:
        """Generate ideal accent pattern based on section"""
        if section == "Chorus":
            # Emphasize strong beats
            return np.array([1.0, 0.3, 0.6, 0.3, 1.0, 0.3, 0.6, 0.3,
                           1.0, 0.3, 0.6, 0.3, 1.0, 0.3, 0.6, 0.3])
        else:
            # More subtle accent pattern
            return np.array([1.0, 0.2, 0.4, 0.2, 0.8, 0.2, 0.4, 0.2,
                           0.9, 0.2, 0.4, 0.2, 0.7, 0.2, 0.4, 0.2])
    
    def _compute_accent_score_v2(
        self,
        pattern_accent: Optional[Sequence[float]],
        ideal_accent: np.ndarray
    ) -> Tuple[float, int]:
        """
        位相最適化付きAccent Score計算（v2実装）
        
        Returns:
            (accent_score, best_phase)
        """
        # Fallbackアクセントプロファイル
        if pattern_accent is None:
            pattern_accent = _fallback_accent_profile(len(ideal_accent))
        
        # 位相最適化
        score, phase = _roll_to_max_cosine(ideal_accent.tolist(), pattern_accent)
        
        return float(score), int(phase)
    
    def _compute_chord_fit_v2(
        self,
        voicing_or_pc_set,  # Sequence[int]（voicing）またはSet[int]（pc_set）
        chord_root: str,
        chord_type: str
    ) -> float:
        """
        Voicing pitch class命中率によるChord Fit計算（v2実装）
        
        Args:
            voicing_or_pc_set: ボイシング（MIDIノート番号のリスト）またはPC集合（0-11のset/list）
            chord_root: コードルート（C, D, E, F, G, A, B）
            chord_type: コードタイプ（maj, min, 7, etc）
        
        Returns:
            0.0-1.0のChord Fit スコア
        """
        # ルートピッチマップ
        root_map = {
            'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
            'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10,
            'C#': 1, 'D#': 3, 'F#': 6, 'G#': 8, 'A#': 10
        }
        
        # '#'と'm'の処理
        clean_root = chord_root.replace('m', '')  # Am → A
        root_pc = root_map.get(clean_root, 0)
        
        # コードトーンとテンション定義
        chord_tones = set()
        allowed_tensions = set()
        
        if 'maj' in chord_type or chord_type == 'M' or (chord_type == '' and 'm' not in chord_root):
            # メジャー: R, 3, 5
            chord_tones = {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}
            allowed_tensions = {(root_pc + 9) % 12, (root_pc + 11) % 12}  # 6, M7
        elif 'min' in chord_type or 'm' in chord_root or chord_type == 'm':
            # マイナー: R, m3, 5
            chord_tones = {root_pc, (root_pc + 3) % 12, (root_pc + 7) % 12}
            allowed_tensions = {(root_pc + 10) % 12, (root_pc + 9) % 12}  # m7, 6
        elif '7' in chord_type:
            # ドミナント7: R, 3, 5, m7
            chord_tones = {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12, (root_pc + 10) % 12}
            allowed_tensions = {(root_pc + 9) % 12, (root_pc + 2) % 12}  # 6, 9
        else:
            # デフォルトはメジャー
            chord_tones = {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}
            allowed_tensions = {(root_pc + 9) % 12}
        
        # PC集合の取得（事前計算されていればそれを使用、なければvoicingから計算）
        if isinstance(voicing_or_pc_set, (set, list)) and voicing_or_pc_set and \
           all(isinstance(x, int) and 0 <= x <= 11 for x in voicing_or_pc_set if isinstance(x, int)):
            # すでにPC集合（0-11の範囲）
            voicing_pcs = set(voicing_or_pc_set)
        else:
            # Voicingなので変換（MIDIノート→PC）
            if not voicing_or_pc_set:
                return 0.5  # 空voicing時のフォールバック
            voicing_pcs = {(int(pitch) % 12) for pitch in voicing_or_pc_set if pitch > 0}
        
        if not voicing_pcs:
            return 0.5
        
        # 命中率計算
        hits = sum(1 for pc in voicing_pcs if pc in chord_tones or pc in allowed_tensions)
        return hits / len(voicing_pcs)
    
    def _is_strong_beat_position(self, rhythm: Optional[str], time_signature: str = "4/4") -> bool:
        """
        リズムパターンから強拍かどうかを判定
        
        Args:
            rhythm: リズムパターン文字列（例: "strum8_down", "arpeggio16"）
            time_signature: 拍子記号
        
        Returns:
            強拍位置ならTrue（1拍目、3拍目など）
        """
        if not rhythm:
            return True  # 不明な場合は強拍と見なす（安全側）
        
        rhythm_lower = rhythm.lower()
        
        # ダウンストローク、アルペジオの最初などは通常強拍
        if any(x in rhythm_lower for x in ['down', 'root', 'bass', 'kick', 'snare']):
            return True
        
        # アップストローク、裏拍は弱拍
        if any(x in rhythm_lower for x in ['up', 'offbeat', 'synco']):
            return False
        
        # 16分音符の2,4番目などは弱拍
        if '16' in rhythm_lower and any(x in rhythm_lower for x in ['_2', '_4']):
            return False
        
        # デフォルトは強拍
        return True
    
    def _compute_chord_fit_v3(
        self,
        voicing_or_pc_set,
        chord_root: str,
        chord_type: str,
        rhythm: Optional[str] = None,
        time_signature: str = "4/4",
        duration_ratio: float = 1.0
    ) -> float:
        """
        Chord Fit v3 - 音楽理論強化版（連続値化）
        
        v3.1の改善点（弁別力向上）:
        1. 音価重み導入: 強拍×長音=重要、弱拍×短音=軽視（duration_ratio活用）
        2. ペナルティ段階化: 3rd+11th衝突を -0.15（弱拍）/-0.30（強拍）
        3. ベースボーナス連続値化: 0.05-0.15で持続時間に比例
        4. 分布が[0.0-1.0]全域に広がり、p10/p90監視が有効化
        
        Args:
            voicing_or_pc_set: ボイシング（MIDIノート配列）またはPC集合
            chord_root: コードルート（C, Am等）
            chord_type: コード種類（maj, min, 7等）
            rhythm: リズムパターン（拍位置判定用）
            time_signature: 拍子記号
            duration_ratio: 音価比率（0.0-1.0、デフォルト1.0=全音符相当）
        
        Returns:
            Chord Fit スコア (0.0-1.0、連続値)
        """
        # ルートPC計算
        root_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        clean_root = chord_root.replace('m', '')
        root_pc = root_map.get(clean_root, 0)
        
        # コードトーンとテンション定義
        chord_tones = set()
        allowed_tensions = set()
        avoid_notes = set()  # アボイドノート
        
        if 'maj' in chord_type or chord_type == 'M' or (chord_type == '' and 'm' not in chord_root):
            # メジャー: R, 3, 5
            chord_tones = {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}
            allowed_tensions = {(root_pc + 9) % 12, (root_pc + 11) % 12}  # 6, M7
            avoid_notes = {(root_pc + 5) % 12}  # 11th (4th)はメジャーでアボイド
        elif 'min' in chord_type or 'm' in chord_root or chord_type == 'm':
            # マイナー: R, m3, 5
            chord_tones = {root_pc, (root_pc + 3) % 12, (root_pc + 7) % 12}
            allowed_tensions = {(root_pc + 10) % 12, (root_pc + 9) % 12, (root_pc + 5) % 12}  # m7, 6, 11
            avoid_notes = {(root_pc + 2) % 12}  # 9thは状況次第だが基本OK
        elif '7' in chord_type:
            # ドミナント7: R, 3, 5, m7
            chord_tones = {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12, (root_pc + 10) % 12}
            allowed_tensions = {(root_pc + 9) % 12, (root_pc + 2) % 12, (root_pc + 8) % 12}  # 6, 9, #11
            avoid_notes = {(root_pc + 5) % 12}  # nat 11th
        else:
            chord_tones = {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}
            allowed_tensions = {(root_pc + 9) % 12}
            avoid_notes = {(root_pc + 5) % 12}
        
        # PC集合の取得
        if isinstance(voicing_or_pc_set, set):
            voicing_pcs = voicing_or_pc_set
            voicing_notes = []  # ベース音判定にはMIDIノートが必要だが、PC集合からは不明
        elif isinstance(voicing_or_pc_set, (list, tuple)):
            if voicing_or_pc_set and all(isinstance(x, int) and 0 <= x <= 11 for x in voicing_or_pc_set if isinstance(x, int)):
                # PC集合
                voicing_pcs = set(voicing_or_pc_set)
                voicing_notes = []
            else:
                # MIDIノート
                voicing_notes = [p for p in voicing_or_pc_set if p > 0]
                voicing_pcs = {(int(pitch) % 12) for pitch in voicing_notes}
        else:
            return 0.5
        
        if not voicing_pcs:
            return 0.5
        
        # 音価重み: 強拍×長音=1.0、弱拍×短音=0.5
        is_strong_beat = self._is_strong_beat_position(rhythm, time_signature)
        note_weight = 1.0 if is_strong_beat else 0.5
        note_weight *= duration_ratio  # 音価比率を乗算（長い音ほど重要）
        
        # 基本スコア: コードトーン/テンションヒット率（重み付き）
        weighted_hits = 0.0
        weighted_total = 0.0
        
        for pc in voicing_pcs:
            weighted_total += note_weight
            if pc in chord_tones:
                weighted_hits += note_weight * 1.0  # コードトーン: 全重み
            elif pc in allowed_tensions:
                weighted_hits += note_weight * 0.8  # テンション: 80%
        
        base_score = weighted_hits / weighted_total if weighted_total > 0 else 0.5
        
        # ペナルティ計算（段階化）
        penalty = 0.0
        
        # 1. 3rd+11th衝突チェック（メジャーコードのみ）- 強拍/弱拍で差別化
        if 'maj' in chord_type or (chord_type == '' and 'm' not in chord_root):
            major_3rd = (root_pc + 4) % 12
            nat_11th = (root_pc + 5) % 12
            if major_3rd in voicing_pcs and nat_11th in voicing_pcs:
                if is_strong_beat:
                    penalty += 0.30  # 強拍: 強い減点
                else:
                    penalty += 0.15  # 弱拍: 軽度の減点
        
        # 2. アボイドノートチェック（拍位置×音価で段階化）
        avoid_present = voicing_pcs & avoid_notes
        
        if avoid_present:
            for _ in avoid_present:
                if is_strong_beat:
                    penalty += 0.20 * duration_ratio  # 強拍×長音: 減点大
                else:
                    penalty += 0.05 * duration_ratio  # 弱拍×短音: 経過音として許容
        
        # 3. ベース音整合度チェック（連続値化: 0.05-0.15）
        bass_bonus = 0.0
        if voicing_notes:
            bass_note = min(voicing_notes)
            bass_pc = bass_note % 12
            if bass_pc == root_pc:
                # ベースがルート: duration_ratioに比例（長く持続するほど加点大）
                bass_bonus = 0.05 + (0.10 * duration_ratio)  # 0.05-0.15の範囲
        
        # 最終スコア（連続値）
        final_score = base_score - penalty + bass_bonus
        return max(0.0, min(1.0, final_score))  # 0-1にクリップ
    
    def _load_gate_config(self, config_path: Optional[str]) -> Dict:
        """
        KPIゲート設定をYAMLから読み込み
        
        Args:
            config_path: YAML設定ファイルパス（省略時はデフォルト値）
        
        Returns:
            ゲート設定dict
        """
        if not config_path:
            # デフォルト設定
            return {
                'kpi_gate': {
                    'accent_score_min': 0.65,
                    'chord_fit_min': 0.60,
                    'density_abs_max': 1.0,
                    'ml_used_min': 0.70,
                    'per_section': {}
                }
            }
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded gate config from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load gate config: {e}, using defaults")
            return {
                'kpi_gate': {
                    'accent_score_min': 0.65,
                    'chord_fit_min': 0.60,
                    'density_abs_max': 1.0,
                    'ml_used_min': 0.70,
                    'per_section': {}
                }
            }
    
    def _get_gate_threshold(self, metric_name: str, section: str) -> float:
        """
        セクション別ゲート閾値を取得（オーバーライド対応）
        
        Args:
            metric_name: 'accent_score_min', 'chord_fit_min', 'density_abs_max' 等
            section: セクション名（'chorus', 'verse', 'bridge' 等）
        
        Returns:
            閾値（float）
        """
        gate_cfg = self.gate_config.get('kpi_gate', {})
        per_section = gate_cfg.get('per_section', {})
        
        # セクション小文字化（YAMLとの整合）
        section_lower = section.lower() if section else 'unknown'
        
        # セクション別オーバーライドがあればそれを使用
        if section_lower in per_section:
            section_cfg = per_section[section_lower]
            if metric_name in section_cfg:
                return float(section_cfg[metric_name])
        
        # なければデフォルト値
        return float(gate_cfg.get(metric_name, 0.0))
    
    def _get_slots_from_time_signature(self, time_signature: str) -> int:
        """
        拍子記号からスロット数を決定
        
        Args:
            time_signature: "3/4", "4/4", "6/8" などの拍子記号
        
        Returns:
            スロット数（12, 16, 24のいずれか、デフォルト16）
        """
        # 拍子記号をパース
        if '/' not in time_signature:
            return 16  # デフォルト
        
        try:
            numerator, denominator = time_signature.split('/')
            num = int(numerator)
            denom = int(denominator)
        except (ValueError, AttributeError):
            return 16
        
        # 拍子記号に応じたスロット数マッピング
        if num == 3 and denom == 4:
            return 12  # 3/4拍子 = 3拍 × 4分割 = 12スロット
        elif num in [4, 2] and denom == 4:
            return 16  # 4/4, 2/4拍子 = 4拍 × 4分割 = 16スロット
        elif num == 6 and denom == 8:
            return 24  # 6/8拍子 = 6拍 × 4分割 = 24スロット
        elif num == 12 and denom == 8:
            return 24  # 12/8拍子も24スロット
        else:
            # その他の拍子は16をデフォルト
            return 16
    
    def _normalize_accent_score_to_16(
        self,
        accent_score: float,
        pattern_accent: Optional[Sequence[float]],
        ideal_accent: np.ndarray,
        time_signature: str
    ) -> float:
        """
        Accent scoreを16スロット基準に正規化
        
        3/4拍子（12スロット）や6/8拍子（24スロット）のaccent scoreを
        4/4拍子（16スロット）基準に正規化して比較可能にする。
        
        Args:
            accent_score: 元のaccent score
            pattern_accent: パターンのアクセントプロファイル
            ideal_accent: 理想アクセント（16スロット）
            time_signature: 拍子記号
        
        Returns:
            16スロット基準に正規化されたaccent score
        """
        slots = self._get_slots_from_time_signature(time_signature)
        
        # 既に16スロットなら正規化不要
        if slots == 16:
            return accent_score
        
        # パターンアクセントがない場合は元のスコアを返す
        if not pattern_accent:
            return accent_score
        
        # 16スロットに線形補間で正規化
        pattern_accent_arr = np.array(pattern_accent, dtype=np.float32)
        
        # 元のスロット数から16スロットへ補間
        x_old = np.linspace(0, 1, len(pattern_accent_arr))
        x_new = np.linspace(0, 1, 16)
        pattern_accent_16 = np.interp(x_new, x_old, pattern_accent_arr)
        
        # 16スロット基準で再計算
        normalized_score, _ = self._compute_accent_score_v2(
            pattern_accent_16,
            ideal_accent[:16] if len(ideal_accent) >= 16 else ideal_accent
        )
        
        return normalized_score
    
    def _compute_file_sha256(self, file_path: str) -> str:
        """
        ファイルのSHA256ハッシュを計算
        
        Args:
            file_path: ファイルパス
        
        Returns:
            SHA256ハッシュ（hex文字列）
        """
        import hashlib
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to compute SHA256 for {file_path}: {e}")
            return "unknown"
    
    def _load_safe_kit(self, safe_kit_path: Optional[str] = None) -> Dict:
        """
        Safe-Kitパターンをファイルから読み込み
        
        Args:
            safe_kit_path: Safe-Kit YAML path (デフォルト: data/patterns/safe_kit_guitar.yaml)
        
        Returns:
            Safe-Kit設定dict（読み込み失敗時は空dict）
        """
        if not safe_kit_path:
            safe_kit_path = "data/patterns/safe_kit_guitar.yaml"
        
        try:
            import yaml
            from pathlib import Path
            
            safe_kit_file = Path(safe_kit_path)
            if not safe_kit_file.exists():
                self.logger.info(f"Safe-Kit file not found: {safe_kit_path}, fallback disabled")
                return {}
            
            with open(safe_kit_file, 'r', encoding='utf-8') as f:
                safe_kit_config = yaml.safe_load(f)
            
            self.logger.info(f"Safe-Kit loaded from {safe_kit_path}")
            return safe_kit_config
        
        except Exception as e:
            self.logger.warning(f"Failed to load Safe-Kit: {e}, fallback disabled")
            return {}
    
    def _get_safe_kit_pattern(
        self,
        chord_root: str,
        section: str,
        chord_type: str = "major"
    ) -> Optional[Dict]:
        """
        Safe-KitからSection/Chord Qualityに応じたパターンを取得
        
        Args:
            chord_root: コードルート（C, D, E等）
            section: セクション名（Chorus, Verse, Bridge等）
            chord_type: コードタイプ（major, minor, 7等）
        
        Returns:
            Safe-Kitパターンdict（ルート置換済み）、取得失敗時はNone
        """
        if not self.safe_kit_patterns:
            return None
        
        # セクション別デフォルト取得
        section_defaults = self.safe_kit_patterns.get('section_defaults', {})
        section_cap = section.capitalize()  # Chorus, Verse, etc
        
        pattern_name = section_defaults.get(section_cap)
        if not pattern_name:
            # フォールバック: Unknown→BASIC_SAFE
            pattern_name = section_defaults.get('Unknown', 'BASIC_SAFE')
        
        # パターン定義取得
        patterns_db = self.safe_kit_patterns.get('patterns', {})
        if pattern_name not in patterns_db:
            self.logger.warning(f"Safe-Kit pattern not found: {pattern_name}")
            return None
        
        safe_pattern_template = patterns_db[pattern_name]
        
        # ルート置換（voicing_templateをchord_rootに変換）
        voicing_template = safe_pattern_template.get('voicing_template', ['R', '5', '8'])
        actual_voicing = self._apply_chord_root_to_voicing(chord_root, voicing_template, chord_type)
        
        # Accent profile取得
        accent_profile = safe_pattern_template.get('accent_profile', [])
        
        # パターンdict構築
        pattern_dict = {
            'pattern_id': f"SAFE_KIT_{pattern_name}_{chord_root}",
            'rhythm': safe_pattern_template.get('family', 'SAFE'),
            'voicing': actual_voicing,
            'metadata': {
                'tempo_bin': 'medium',
                'section': section,
                'technique': safe_pattern_template.get('family', 'safe'),
                'density_ql_per_bar': safe_pattern_template.get('density_ql_per_bar', 4)
            },
            'accent_profile': accent_profile,
            'density_ql_per_bar': safe_pattern_template.get('density_ql_per_bar', 4),
            'safe_kit_source': pattern_name  # トレーサビリティ用
        }
        
        return pattern_dict
    
    def _apply_chord_root_to_voicing(
        self,
        chord_root: str,
        voicing_template: List[str],
        chord_type: str = "major"
    ) -> List[int]:
        """
        Voicing templateをchord_rootに適用してMIDIノート番号に変換
        
        Args:
            chord_root: コードルート（C, D, E等）
            voicing_template: ['R', '3', '5', '8']等のテンプレート
            chord_type: コードタイプ（major, minor, 7等）
        
        Returns:
            MIDIノート番号のリスト（例: [60, 64, 67]）
        """
        # ルートMIDIノート番号マップ（C4=60基準）
        root_midi_map = {
            'C': 60, 'C#': 61, 'Db': 61,
            'D': 62, 'D#': 63, 'Eb': 63,
            'E': 64,
            'F': 65, 'F#': 66, 'Gb': 66,
            'G': 67, 'G#': 68, 'Ab': 68,
            'A': 69, 'A#': 70, 'Bb': 70,
            'B': 71
        }
        
        # #とmの処理
        clean_root = chord_root.replace('m', '')  # Am → A
        root_midi = root_midi_map.get(clean_root, 60)  # デフォルトC
        
        # インターバルマップ
        interval_map = {
            'R': 0,      # Root
            '2': 2,      # Major 2nd
            'b3': 3,     # Minor 3rd
            '3': 4,      # Major 3rd (default)
            'M3': 4,
            'm3': 3,
            '4': 5,      # Perfect 4th
            '5': 7,      # Perfect 5th
            'b6': 8,     # Minor 6th
            '6': 9,      # Major 6th
            'b7': 10,    # Minor 7th
            '7': 11,     # Major 7th
            'M7': 11,
            'm7': 10,
            '8': 12,     # Octave
            '9': 14,     # 9th
            '10': 16,    # 10th
            '11': 17,    # 11th
            '#11': 18,
            '13': 21     # 13th
        }
        
        # Chord type別のデフォルト3rd処理
        if 'min' in chord_type or 'm' in chord_root:
            default_3rd = 3  # Minor 3rd
        else:
            default_3rd = 4  # Major 3rd
        
        voicing_midi = []
        for degree in voicing_template:
            if degree == '3':
                # chord_typeに応じて3rdを決定
                interval = default_3rd
            else:
                interval = interval_map.get(degree, 0)
            
            voicing_midi.append(root_midi + interval)
        
        return voicing_midi
    
    def _compute_file_sha256(self, file_path: str) -> str:
        """
        ファイルのSHA256ハッシュを計算
        
        Args:
            file_path: ファイルパス
        
        Returns:
            SHA256ハッシュ（hex文字列）
        """
        import hashlib
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to compute SHA256 for {file_path}: {e}")
            return "unknown"
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        if self.stats['total_requests'] == 0:
            return self.stats
        
        total = self.stats['total_requests']
        
        return {
            **self.stats,
            'v3_primary_ratio': self.stats['v3_primary_count'] / total,
            'v1_primary_ratio': self.stats['v1_primary_count'] / total,
            'v3_win_rate': self.stats['v3_wins'] / total,
            'v1_win_rate': self.stats['v1_wins'] / total,
            'tie_rate': self.stats['ties'] / total,
            'v3_error_rate': self.stats['v3_errors'] / total,
            'v1_error_rate': self.stats['v1_errors'] / total
        }
    
    def export_prometheus_metrics(self, output_path: str):
        """Export Prometheus metrics"""
        stats = self.get_statistics()
        
        # Compute aggregated metrics from results
        if self.results:
            v3_accents = [r.v3_accent_score for r in self.results if not r.v3_error]
            v1_accents = [r.v1_accent_score for r in self.results if not r.v1_error]
            v3_chords = [r.v3_chord_fit for r in self.results if not r.v3_error]
            v1_chords = [r.v1_chord_fit for r in self.results if not r.v1_error]
            accent_deltas = [r.accent_delta for r in self.results]
            v3_latencies = [r.v3_latency_ms for r in self.results]
            v1_latencies = [r.v1_latency_ms for r in self.results]
            pattern_agreements = [r.pattern_agreement for r in self.results]
            
            # 平均値計算
            v3_accent_mean = np.mean(v3_accents) if v3_accents else 0
            v1_accent_mean = np.mean(v1_accents) if v1_accents else 0
            v3_chord_mean = np.mean(v3_chords) if v3_chords else 0
            v1_chord_mean = np.mean(v1_chords) if v1_chords else 0
            accent_delta_mean = np.mean(accent_deltas) if accent_deltas else 0
            pattern_agreement_rate = np.mean(pattern_agreements) if pattern_agreements else 0
            
            # パーセンタイル計算（p10/p50/p90）
            # Accent Score
            v3_accent_p10 = np.percentile(v3_accents, 10) if v3_accents else 0
            v3_accent_p50 = np.percentile(v3_accents, 50) if v3_accents else 0
            v3_accent_p90 = np.percentile(v3_accents, 90) if v3_accents else 0
            v1_accent_p10 = np.percentile(v1_accents, 10) if v1_accents else 0
            v1_accent_p50 = np.percentile(v1_accents, 50) if v1_accents else 0
            v1_accent_p90 = np.percentile(v1_accents, 90) if v1_accents else 0
            
            # Chord Fit
            v3_chord_p10 = np.percentile(v3_chords, 10) if v3_chords else 0
            v3_chord_p50 = np.percentile(v3_chords, 50) if v3_chords else 0
            v3_chord_p90 = np.percentile(v3_chords, 90) if v3_chords else 0
            v1_chord_p10 = np.percentile(v1_chords, 10) if v1_chords else 0
            v1_chord_p50 = np.percentile(v1_chords, 50) if v1_chords else 0
            v1_chord_p90 = np.percentile(v1_chords, 90) if v1_chords else 0
            
            # Latency（既存のp95に加えてp50も追加）
            v3_latency_p50 = np.percentile(v3_latencies, 50) if v3_latencies else 0
            v3_latency_p95 = np.percentile(v3_latencies, 95) if v3_latencies else 0
            v1_latency_p50 = np.percentile(v1_latencies, 50) if v1_latencies else 0
            v1_latency_p95 = np.percentile(v1_latencies, 95) if v1_latencies else 0
        else:
            v3_accent_mean = 0
            v1_accent_mean = 0
            v3_chord_mean = 0
            v1_chord_mean = 0
            accent_delta_mean = 0
            pattern_agreement_rate = 0
            
            # パーセンタイル初期値
            v3_accent_p10 = v3_accent_p50 = v3_accent_p90 = 0
            v1_accent_p10 = v1_accent_p50 = v1_accent_p90 = 0
            v3_chord_p10 = v3_chord_p50 = v3_chord_p90 = 0
            v1_chord_p10 = v1_chord_p50 = v1_chord_p90 = 0
            v3_latency_p50 = v3_latency_p95 = 0
            v1_latency_p50 = v1_latency_p95 = 0
        
        lines = [
            "# HELP guitar_shadow_total_requests Total shadow testing requests",
            "# TYPE guitar_shadow_total_requests counter",
            f"guitar_shadow_total_requests {stats['total_requests']}",
            "",
            "# HELP guitar_shadow_v3_primary_count v3 primary routing count",
            "# TYPE guitar_shadow_v3_primary_count counter",
            f"guitar_shadow_v3_primary_count {stats['v3_primary_count']}",
            "",
            "# HELP guitar_shadow_v1_primary_count v1 primary routing count",
            "# TYPE guitar_shadow_v1_primary_count counter",
            f"guitar_shadow_v1_primary_count {stats['v1_primary_count']}",
            "",
            "# HELP guitar_shadow_v3_win_rate v3 win rate vs v1",
            "# TYPE guitar_shadow_v3_win_rate gauge",
            f"guitar_shadow_v3_win_rate {stats.get('v3_win_rate', 0):.4f}",
            "",
            "# HELP guitar_shadow_v1_win_rate v1 win rate vs v3",
            "# TYPE guitar_shadow_v1_win_rate gauge",
            f"guitar_shadow_v1_win_rate {stats.get('v1_win_rate', 0):.4f}",
            "",
            "# HELP guitar_v3_accent_score_mean v3 mean accent score",
            "# TYPE guitar_v3_accent_score_mean gauge",
            f"guitar_v3_accent_score_mean {v3_accent_mean:.4f}",
            "",
            "# HELP guitar_v1_accent_score_mean v1 mean accent score",
            "# TYPE guitar_v1_accent_score_mean gauge",
            f"guitar_v1_accent_score_mean {v1_accent_mean:.4f}",
            "",
            "# HELP guitar_v3_accent_score_p10 v3 p10 accent score",
            "# TYPE guitar_v3_accent_score_p10 gauge",
            f"guitar_v3_accent_score_p10 {v3_accent_p10:.4f}",
            "",
            "# HELP guitar_v3_accent_score_p50 v3 p50 accent score (median)",
            "# TYPE guitar_v3_accent_score_p50 gauge",
            f"guitar_v3_accent_score_p50 {v3_accent_p50:.4f}",
            "",
            "# HELP guitar_v3_accent_score_p90 v3 p90 accent score",
            "# TYPE guitar_v3_accent_score_p90 gauge",
            f"guitar_v3_accent_score_p90 {v3_accent_p90:.4f}",
            "",
            "# HELP guitar_v1_accent_score_p10 v1 p10 accent score",
            "# TYPE guitar_v1_accent_score_p10 gauge",
            f"guitar_v1_accent_score_p10 {v1_accent_p10:.4f}",
            "",
            "# HELP guitar_v1_accent_score_p50 v1 p50 accent score (median)",
            "# TYPE guitar_v1_accent_score_p50 gauge",
            f"guitar_v1_accent_score_p50 {v1_accent_p50:.4f}",
            "",
            "# HELP guitar_v1_accent_score_p90 v1 p90 accent score",
            "# TYPE guitar_v1_accent_score_p90 gauge",
            f"guitar_v1_accent_score_p90 {v1_accent_p90:.4f}",
            "",
            "# HELP guitar_shadow_accent_delta Accent score delta (v3 - v1)",
            "# TYPE guitar_shadow_accent_delta gauge",
            f"guitar_shadow_accent_delta {accent_delta_mean:.4f}",
            "",
            "# HELP guitar_v3_chord_fit_mean v3 mean chord fit",
            "# TYPE guitar_v3_chord_fit_mean gauge",
            f"guitar_v3_chord_fit_mean {v3_chord_mean:.4f}",
            "",
            "# HELP guitar_v1_chord_fit_mean v1 mean chord fit",
            "# TYPE guitar_v1_chord_fit_mean gauge",
            f"guitar_v1_chord_fit_mean {v1_chord_mean:.4f}",
            "",
            "# HELP guitar_v3_chord_fit_p10 v3 p10 chord fit",
            "# TYPE guitar_v3_chord_fit_p10 gauge",
            f"guitar_v3_chord_fit_p10 {v3_chord_p10:.4f}",
            "",
            "# HELP guitar_v3_chord_fit_p50 v3 p50 chord fit (median)",
            "# TYPE guitar_v3_chord_fit_p50 gauge",
            f"guitar_v3_chord_fit_p50 {v3_chord_p50:.4f}",
            "",
            "# HELP guitar_v3_chord_fit_p90 v3 p90 chord fit",
            "# TYPE guitar_v3_chord_fit_p90 gauge",
            f"guitar_v3_chord_fit_p90 {v3_chord_p90:.4f}",
            "",
            "# HELP guitar_v1_chord_fit_p10 v1 p10 chord fit",
            "# TYPE guitar_v1_chord_fit_p10 gauge",
            f"guitar_v1_chord_fit_p10 {v1_chord_p10:.4f}",
            "",
            "# HELP guitar_v1_chord_fit_p50 v1 p50 chord fit (median)",
            "# TYPE guitar_v1_chord_fit_p50 gauge",
            f"guitar_v1_chord_fit_p50 {v1_chord_p50:.4f}",
            "",
            "# HELP guitar_v1_chord_fit_p90 v1 p90 chord fit",
            "# TYPE guitar_v1_chord_fit_p90 gauge",
            f"guitar_v1_chord_fit_p90 {v1_chord_p90:.4f}",
            "",
            "# HELP guitar_v3_latency_p50_ms v3 p50 latency (ms, median)",
            "# TYPE guitar_v3_latency_p50_ms gauge",
            f"guitar_v3_latency_p50_ms {v3_latency_p50:.2f}",
            "",
            "# HELP guitar_v3_latency_p95_ms v3 p95 latency (ms)",
            "# TYPE guitar_v3_latency_p95_ms gauge",
            f"guitar_v3_latency_p95_ms {v3_latency_p95:.2f}",
            "",
            "# HELP guitar_v1_latency_p50_ms v1 p50 latency (ms, median)",
            "# TYPE guitar_v1_latency_p50_ms gauge",
            f"guitar_v1_latency_p50_ms {v1_latency_p50:.2f}",
            "",
            "# HELP guitar_v1_latency_p95_ms v1 p95 latency (ms)",
            "# TYPE guitar_v1_latency_p95_ms gauge",
            f"guitar_v1_latency_p95_ms {v1_latency_p95:.2f}",
            "",
            "# HELP guitar_v3_error_rate v3 error rate",
            "# TYPE guitar_v3_error_rate gauge",
            f"guitar_v3_error_rate {stats.get('v3_error_rate', 0):.4f}",
            "",
            "# HELP guitar_v1_error_rate v1 error rate",
            "# TYPE guitar_v1_error_rate gauge",
            f"guitar_v1_error_rate {stats.get('v1_error_rate', 0):.4f}",
            "",
            "# HELP guitar_shadow_pattern_agreement_rate Pattern agreement rate",
            "# TYPE guitar_shadow_pattern_agreement_rate gauge",
            f"guitar_shadow_pattern_agreement_rate {pattern_agreement_rate:.4f}",
            "",
            "# HELP guitar_shadow_v3_wins_total Total v3 wins",
            "# TYPE guitar_shadow_v3_wins_total counter",
            f"guitar_shadow_v3_wins_total {stats['v3_wins']}",
            "",
            "# HELP guitar_shadow_v1_wins_total Total v1 wins",
            "# TYPE guitar_shadow_v1_wins_total counter",
            f"guitar_shadow_v1_wins_total {stats['v1_wins']}",
            "",
            "# HELP guitar_shadow_ties_total Total ties",
            "# TYPE guitar_shadow_ties_total counter",
            f"guitar_shadow_ties_total {stats['ties']}",
            ""
        ]
        
        # Auto-Recovery メトリクス
        if self.auto_recovery:
            recovery_metrics = self.auto_recovery.get_metrics()
            
            lines.extend([
                "# HELP auto_recovery_switches_v3_to_v1_total v3→v1切替回数（累積）",
                "# TYPE auto_recovery_switches_v3_to_v1_total counter",
                f"auto_recovery_switches_v3_to_v1_total {recovery_metrics.switches_v3_to_v1}",
                "",
                "# HELP auto_recovery_switches_v1_to_v3_total v1→v3切替回数（累積）",
                "# TYPE auto_recovery_switches_v1_to_v3_total counter",
                f"auto_recovery_switches_v1_to_v3_total {recovery_metrics.switches_v1_to_v3}",
                "",
                "# HELP auto_recovery_cooldown_active クールダウン中フラグ（1=active, 0=inactive）",
                "# TYPE auto_recovery_cooldown_active gauge",
                f"auto_recovery_cooldown_active {1 if recovery_metrics.cooldown_active else 0}",
                "",
                "# HELP auto_recovery_cooldown_remaining クールダウン残りバー数",
                "# TYPE auto_recovery_cooldown_remaining gauge",
                f"auto_recovery_cooldown_remaining {recovery_metrics.cooldown_remaining}",
                "",
                "# HELP auto_recovery_breach_count ウィンドウ内の違反回数",
                "# TYPE auto_recovery_breach_count gauge",
                f"auto_recovery_breach_count {recovery_metrics.breach_count}",
                "",
                "# HELP auto_recovery_window_size ウィンドウサイズ（バー数）",
                "# TYPE auto_recovery_window_size gauge",
                f"auto_recovery_window_size {recovery_metrics.window_size}",
                "",
                "# HELP auto_recovery_threshold 違反閾値（バー数）",
                "# TYPE auto_recovery_threshold gauge",
                f"auto_recovery_threshold {recovery_metrics.threshold}",
                "",
                "# HELP auto_recovery_current_version_v3 現在v3がアクティブ（1=v3, 0=v1）",
                "# TYPE auto_recovery_current_version_v3 gauge",
                f"auto_recovery_current_version_v3 {1 if recovery_metrics.current_version == 'v3' else 0}",
                "",
                "# HELP auto_recovery_current_version_v1 現在v1がアクティブ（1=v1, 0=v3）",
                "# TYPE auto_recovery_current_version_v1 gauge",
                f"auto_recovery_current_version_v1 {1 if recovery_metrics.current_version == 'v1' else 0}",
                ""
            ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"Prometheus metrics exported: {output_path}")
    
    def get_section_statistics(self) -> Dict:
        """
        セクション別の統計を計算
        
        Returns:
            セクション名をキーとし、各セクションのKPI統計を値とする辞書
        """
        section_stats = {}
        
        if not self.results:
            return section_stats
        
        # セクション別にグループ化
        from collections import defaultdict
        sections = defaultdict(list)
        
        for result in self.results:
            section = result.section.lower()
            sections[section].append(result)
        
        # 各セクションの統計を計算
        for section, results in sections.items():
            v3_accents = [r.v3_accent_score for r in results if not r.v3_error]
            v1_accents = [r.v1_accent_score for r in results if not r.v1_error]
            v3_chords = [r.v3_chord_fit for r in results if not r.v3_error]
            v1_chords = [r.v1_chord_fit for r in results if not r.v1_error]
            
            section_stats[section] = {
                'count': len(results),
                'v3_accent_mean': np.mean(v3_accents) if v3_accents else 0,
                'v3_accent_p50': np.percentile(v3_accents, 50) if v3_accents else 0,
                'v1_accent_mean': np.mean(v1_accents) if v1_accents else 0,
                'v1_accent_p50': np.percentile(v1_accents, 50) if v1_accents else 0,
                'v3_chord_mean': np.mean(v3_chords) if v3_chords else 0,
                'v3_chord_p50': np.percentile(v3_chords, 50) if v3_chords else 0,
                'v1_chord_mean': np.mean(v1_chords) if v1_chords else 0,
                'v1_chord_p50': np.percentile(v1_chords, 50) if v1_chords else 0,
            }
        
        return section_stats
    
    def print_summary(self):
        """Print current statistics summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Traffic Splitter Summary")
        print("="*60)
        print(f"\nTotal Requests: {stats['total_requests']}")
        print(f"v3 Primary: {stats['v3_primary_count']} ({stats.get('v3_primary_ratio', 0)*100:.1f}%)")
        print(f"v1 Primary: {stats['v1_primary_count']} ({stats.get('v1_primary_ratio', 0)*100:.1f}%)")
        
        print("\n--- Win Rates ---")
        print(f"v3 Wins: {stats['v3_wins']} ({stats.get('v3_win_rate', 0)*100:.1f}%)")
        print(f"v1 Wins: {stats['v1_wins']} ({stats.get('v1_win_rate', 0)*100:.1f}%)")
        print(f"Ties: {stats['ties']} ({stats.get('tie_rate', 0)*100:.1f}%)")
        
        print("\n--- Error Rates ---")
        print(f"v3 Errors: {stats['v3_errors']} ({stats.get('v3_error_rate', 0)*100:.2f}%)")
        print(f"v1 Errors: {stats['v1_errors']} ({stats.get('v1_error_rate', 0)*100:.2f}%)")
        
        # セクション別統計
        section_stats = self.get_section_statistics()
        if section_stats:
            print("\n--- Section Statistics ---")
            for section, section_data in sorted(section_stats.items()):
                print(f"\n{section.capitalize()} (n={section_data['count']}):")
                print(f"  v3 Accent: mean={section_data['v3_accent_mean']:.3f}, median={section_data['v3_accent_p50']:.3f}")
                print(f"  v1 Accent: mean={section_data['v1_accent_mean']:.3f}, median={section_data['v1_accent_p50']:.3f}")
                print(f"  v3 Chord:  mean={section_data['v3_chord_mean']:.3f}, median={section_data['v3_chord_p50']:.3f}")
                print(f"  v1 Chord:  mean={section_data['v1_chord_mean']:.3f}, median={section_data['v1_chord_p50']:.3f}")
        
        print("="*60 + "\n")


# Example usage
if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    splitter = TrafficSplitter(
        v3_pickle_path='data/patterns/stage2_guitar_v3_fixed.pickle',
        v1_pickle_path='data/patterns/stage2_guitar.pickle',
        v3_ratio=0.9,
        log_path='data/shadow_traffic_log.csv'
    )
    
    # Test single request
    pattern, comparison = splitter.route_and_compare(
        chord_root='C',
        tempo=120.0,
        section='Chorus',
        key='C',
        chord_type='maj'
    )
    
    print(f"\nPrimary Version: {comparison.primary_version}")
    print(f"v3 Accent: {comparison.v3_accent_score:.2%}")
    print(f"v1 Accent: {comparison.v1_accent_score:.2%}")
    print(f"Delta: {comparison.accent_delta:+.2%}")
    print(f"v3 Wins: {bool(comparison.v3_wins)}")
    
    splitter.print_summary()
