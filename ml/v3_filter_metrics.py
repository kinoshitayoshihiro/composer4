#!/usr/bin/env python3
"""
V3 Filter Metrics for Prometheus

Phase 24.2: Guitar V3フィルタとKPI評価のモニタリングメトリクス

Metrics:
- guitar_v3_filter_total: V3フィルタ呼び出し総数（Counter）
- guitar_v3_kpi_passed_total: KPI合格数（Counter）
- guitar_v3_kpi_failed_total: KPI不合格数（Counter）
- guitar_v3_no_candidates_total: 候補なし数（Counter）
- guitar_v3_top1_proba: top1_proba分布（Histogram）
- guitar_v3_margin: margin分布（Histogram）
- guitar_safekit_fallback_total: Safe-Kit fallback数（Counter, reason label付き）

Usage:
    from ml.v3_filter_metrics import V3FilterMetrics
    
    metrics = V3FilterMetrics()
    
    # Record V3 filter result
    metrics.record_v3_filter_result(
        instrument='guitar',
        kpi_passed=True,
        top1_proba=0.95,
        margin=0.25
    )
    
    # Record Safe-Kit fallback
    metrics.record_safekit_fallback(
        instrument='guitar',
        reason='kpi_failed'
    )
"""

import logging
from typing import Optional

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available, metrics disabled")

logger = logging.getLogger(__name__)


class V3FilterMetrics:
    """V3フィルタとKPI評価のPrometheusメトリクス"""
    
    def __init__(self):
        """Initialize metrics"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics will be no-ops")
            self.enabled = False
            return
        
        self.enabled = True
        
        # V3 Filter総数（Counter）
        self.v3_filter_total = Counter(
            'guitar_v3_filter_total',
            'Total V3 filter calls',
            ['instrument']
        )
        
        # KPI合格数（Counter）
        self.v3_kpi_passed_total = Counter(
            'guitar_v3_kpi_passed_total',
            'Total V3 patterns passing KPI',
            ['instrument']
        )
        
        # KPI不合格数（Counter）
        self.v3_kpi_failed_total = Counter(
            'guitar_v3_kpi_failed_total',
            'Total V3 patterns failing KPI',
            ['instrument']
        )
        
        # 候補なし数（Counter）
        self.v3_no_candidates_total = Counter(
            'guitar_v3_no_candidates_total',
            'Total V3 filter with no candidates',
            ['instrument']
        )
        
        # top1_proba分布（Histogram）
        self.v3_top1_proba = Histogram(
            'guitar_v3_top1_proba',
            'Distribution of top1_proba values',
            ['instrument'],
            buckets=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        )
        
        # margin分布（Histogram）
        self.v3_margin = Histogram(
            'guitar_v3_margin',
            'Distribution of margin (top1_proba - top2_proba)',
            ['instrument'],
            buckets=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        
        # Safe-Kit fallback数（Counter, reason label付き）
        self.safekit_fallback_total = Counter(
            'guitar_safekit_fallback_total',
            'Total Safe-Kit fallbacks',
            ['instrument', 'reason']
        )
        
        # V3フィルタ合格率（Gauge, 計算用）
        self.v3_kpi_pass_rate = Gauge(
            'guitar_v3_kpi_pass_rate',
            'V3 filter KPI pass rate (passed / total)',
            ['instrument']
        )
        
        logger.info("V3FilterMetrics initialized with Prometheus")
    
    def record_v3_filter_result(
        self,
        instrument: str,
        kpi_passed: bool,
        top1_proba: Optional[float] = None,
        margin: Optional[float] = None,
        no_candidates: bool = False
    ):
        """
        V3フィルタ結果を記録
        
        Args:
            instrument: 楽器名（guitar, bass, piano, strings）
            kpi_passed: KPI合格フラグ
            top1_proba: top1確率（0.0-1.0）
            margin: top1-top2マージン（0.0-1.0）
            no_candidates: 候補なしフラグ
        """
        if not self.enabled:
            return
        
        # V3フィルタ総数
        self.v3_filter_total.labels(instrument=instrument).inc()
        
        if no_candidates:
            # 候補なし
            self.v3_no_candidates_total.labels(instrument=instrument).inc()
        elif kpi_passed:
            # KPI合格
            self.v3_kpi_passed_total.labels(instrument=instrument).inc()
            
            # proba/margin記録
            if top1_proba is not None:
                self.v3_top1_proba.labels(instrument=instrument).observe(top1_proba)
            if margin is not None:
                self.v3_margin.labels(instrument=instrument).observe(margin)
        else:
            # KPI不合格
            self.v3_kpi_failed_total.labels(instrument=instrument).inc()
        
        # 合格率更新（簡易計算、本来は_countから計算）
        # Note: Prometheusクエリで rate(guitar_v3_kpi_passed_total) / rate(guitar_v3_filter_total) を使用
    
    def record_safekit_fallback(
        self,
        instrument: str,
        reason: str
    ):
        """
        Safe-Kit fallback を記録
        
        Args:
            instrument: 楽器名
            reason: fallback理由（kpi_failed, no_candidates, error）
        """
        if not self.enabled:
            return
        
        self.safekit_fallback_total.labels(
            instrument=instrument,
            reason=reason
        ).inc()
        
        logger.debug(f"Safe-Kit fallback: {instrument} - {reason}")
    
    def get_kpi_pass_rate(self, instrument: str) -> Optional[float]:
        """
        KPI合格率を取得（ローカル計算用、Prometheusクエリではrate()使用）
        
        Args:
            instrument: 楽器名
        
        Returns:
            合格率（0.0-1.0）or None
        """
        if not self.enabled:
            return None
        
        try:
            # Note: prometheus_clientのCounterは_valueを直接取得できない
            # Prometheusクエリ側で rate() を使用することを推奨
            return None
        except Exception as e:
            logger.warning(f"Could not calculate KPI pass rate: {e}")
            return None


# Global singleton instance
_v3_filter_metrics = None

def get_v3_filter_metrics() -> V3FilterMetrics:
    """Get global V3FilterMetrics singleton"""
    global _v3_filter_metrics
    if _v3_filter_metrics is None:
        _v3_filter_metrics = V3FilterMetrics()
    return _v3_filter_metrics


# Demo
def demo_v3_filter_metrics():
    """Demo: V3FilterMetrics usage"""
    
    metrics = get_v3_filter_metrics()
    
    print("=== V3 Filter Metrics Demo ===\n")
    
    # Simulate 100 requests
    import random
    
    for i in range(100):
        kpi_passed = random.random() > 0.2  # 80% pass rate
        top1_proba = random.uniform(0.85, 1.0) if kpi_passed else random.uniform(0.1, 0.15)
        margin = random.uniform(0.1, 0.3) if kpi_passed else random.uniform(0.05, 0.1)
        no_candidates = random.random() < 0.05  # 5% no candidates
        
        metrics.record_v3_filter_result(
            instrument='guitar',
            kpi_passed=kpi_passed and not no_candidates,
            top1_proba=top1_proba if not no_candidates else None,
            margin=margin if not no_candidates else None,
            no_candidates=no_candidates
        )
        
        if not kpi_passed or no_candidates:
            reason = 'no_candidates' if no_candidates else 'kpi_failed'
            metrics.record_safekit_fallback(
                instrument='guitar',
                reason=reason
            )
    
    print("✅ Recorded 100 V3 filter results")
    print("\nPrometheus metrics available at /metrics endpoint:")
    print("  - guitar_v3_filter_total")
    print("  - guitar_v3_kpi_passed_total")
    print("  - guitar_v3_kpi_failed_total")
    print("  - guitar_v3_top1_proba (histogram)")
    print("  - guitar_v3_margin (histogram)")
    print("  - guitar_safekit_fallback_total")
    
    print("\nQuery examples:")
    print("  # KPI合格率（5分間）")
    print("  rate(guitar_v3_kpi_passed_total[5m]) / rate(guitar_v3_filter_total[5m])")
    print()
    print("  # Safe-Kit fallback率（5分間）")
    print("  rate(guitar_safekit_fallback_total[5m]) / rate(guitar_v3_filter_total[5m])")
    print()
    print("  # top1_probaのp10（過去24時間）")
    print("  histogram_quantile(0.10, rate(guitar_v3_top1_proba_bucket[24h]))")


if __name__ == '__main__':
    demo_v3_filter_metrics()
