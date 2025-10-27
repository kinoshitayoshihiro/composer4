#!/usr/bin/env python3
"""
Adaptive Safety Threshold Manager - Phase 24

7日間p10分布を基準に動的に安全閾値を調整。
固定閾値（min_proba=0.15）より柔軟で、データドリフトに追従。

Usage:
    from ml.safety_adaptive import AdaptiveThresholdManager
    
    manager = AdaptiveThresholdManager(
        prometheus_url='http://localhost:9090',
        base_ratio=0.90  # p10の90%を閾値に
    )
    
    adaptive_threshold = manager.get_adaptive_threshold('min_proba')
    # Returns: 0.18 if p10_7d=0.20
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

import requests
import numpy as np


class AdaptiveThresholdManager:
    """
    学習型安全閾値マネージャー
    
    Prometheusメトリクスから7日間p10を取得し、動的に閾値を調整。
    
    Features:
    - min_proba: p10_7d * 0.90（90%を下限に）
    - min_margin: margin_p10_7d * 0.90
    - section別カスタマイズ対応
    - 履歴トラッキング（JSON保存）
    """
    
    def __init__(
        self,
        prometheus_url: str = 'http://localhost:9090',
        base_ratio: float = 0.90,
        margin_ratio: float = 0.90,
        history_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            prometheus_url: Prometheus API URL
            base_ratio: p10に対する閾値比率（デフォルト90%）
            margin_ratio: margin p10に対する閾値比率（デフォルト90%）
            history_path: 履歴保存先JSONパス（デフォルト: data/adaptive_threshold_history.json）
            logger: ロガーインスタンス
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.base_ratio = base_ratio
        self.margin_ratio = margin_ratio
        self.logger = logger or logging.getLogger(__name__)
        
        self.history_path = Path(history_path) if history_path else Path('data/adaptive_threshold_history.json')
        
        # 最小値保証（極端に低い閾値を防ぐ）
        self.min_proba_floor = 0.10  # p10が0でも閾値は0.10以上
        self.min_margin_floor = 0.05  # margin p10が0でも閾値は0.05以上
        
        # 履歴読み込み
        self.history = self._load_history()
        
        self.logger.info(
            f"AdaptiveThresholdManager initialized: "
            f"base_ratio={base_ratio}, margin_ratio={margin_ratio}"
        )
    
    def get_adaptive_threshold(
        self,
        metric_name: str = 'min_proba',
        section: Optional[str] = None
    ) -> float:
        """
        適応的閾値を取得
        
        Args:
            metric_name: 'min_proba' or 'min_margin'
            section: セクション名（Chorus, Verse等）、Noneで全体
        
        Returns:
            適応的閾値（float）
        """
        try:
            if metric_name == 'min_proba':
                return self._get_adaptive_min_proba(section)
            elif metric_name == 'min_margin':
                return self._get_adaptive_min_margin(section)
            else:
                self.logger.warning(f"Unknown metric: {metric_name}, using default")
                return 0.15
        
        except Exception as e:
            self.logger.error(f"Failed to get adaptive threshold: {e}")
            # フォールバック: 固定閾値
            return 0.15 if metric_name == 'min_proba' else 0.08
    
    def _get_adaptive_min_proba(self, section: Optional[str] = None) -> float:
        """
        min_proba適応的閾値計算
        
        ロジック: p10_7d * base_ratio（デフォルト90%）
        """
        # Prometheusクエリ
        if section:
            # セクション別p10
            query = f'guitar_v3_top1_proba_p10{{section="{section}"}}'
        else:
            # 全体p10
            query = 'guitar_v3_top1_proba_p10'
        
        p10_7d = self._query_prometheus(query)
        
        if p10_7d is None:
            self.logger.warning("Failed to get p10_7d from Prometheus, using default")
            return 0.15  # デフォルト
        
        # 適応的閾値計算
        adaptive_threshold = float(p10_7d) * self.base_ratio
        
        # 最小値保証
        adaptive_threshold = max(adaptive_threshold, self.min_proba_floor)
        
        # 履歴保存
        self._save_threshold_to_history('min_proba', section, adaptive_threshold, p10_7d)
        
        self.logger.info(
            f"Adaptive min_proba: {adaptive_threshold:.3f} "
            f"(p10_7d={p10_7d:.3f}, ratio={self.base_ratio}, section={section})"
        )
        
        return adaptive_threshold
    
    def _get_adaptive_min_margin(self, section: Optional[str] = None) -> float:
        """
        min_margin適応的閾値計算
        
        ロジック: margin_p10_7d * margin_ratio（デフォルト90%）
        """
        # Prometheusクエリ
        if section:
            query = f'guitar_v3_margin_p10{{section="{section}"}}'
        else:
            query = 'guitar_v3_margin_p10'
        
        margin_p10_7d = self._query_prometheus(query)
        
        if margin_p10_7d is None:
            self.logger.warning("Failed to get margin_p10_7d from Prometheus, using default")
            return 0.08  # デフォルト
        
        # 適応的閾値計算
        adaptive_threshold = float(margin_p10_7d) * self.margin_ratio
        
        # 最小値保証
        adaptive_threshold = max(adaptive_threshold, self.min_margin_floor)
        
        # 履歴保存
        self._save_threshold_to_history('min_margin', section, adaptive_threshold, margin_p10_7d)
        
        self.logger.info(
            f"Adaptive min_margin: {adaptive_threshold:.3f} "
            f"(margin_p10_7d={margin_p10_7d:.3f}, ratio={self.margin_ratio}, section={section})"
        )
        
        return adaptive_threshold
    
    def _query_prometheus(self, query: str) -> Optional[float]:
        """
        Prometheusクエリ実行
        
        Args:
            query: PromQL query
        
        Returns:
            メトリクス値（float）、失敗時はNone
        """
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {'query': query}
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'success':
                self.logger.warning(f"Prometheus query failed: {data}")
                return None
            
            results = data['data']['result']
            if not results:
                self.logger.warning(f"No results for query: {query}")
                return None
            
            # 最初の結果を返す
            value = float(results[0]['value'][1])
            
            return value
        
        except Exception as e:
            self.logger.error(f"Prometheus query error: {e}")
            return None
    
    def _load_history(self) -> Dict:
        """履歴ファイルから読み込み"""
        if not self.history_path.exists():
            return {'records': []}
        
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            self.logger.info(f"Loaded threshold history: {len(history.get('records', []))} records")
            return history
        
        except Exception as e:
            self.logger.warning(f"Failed to load history: {e}")
            return {'records': []}
    
    def _save_threshold_to_history(
        self,
        metric_name: str,
        section: Optional[str],
        threshold: float,
        base_value: float
    ):
        """閾値決定履歴を保存"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'section': section or 'global',
            'threshold': threshold,
            'base_value': base_value,
            'ratio': self.base_ratio if metric_name == 'min_proba' else self.margin_ratio
        }
        
        self.history.setdefault('records', []).append(record)
        
        # 履歴が1000件を超えたら古いものを削除
        if len(self.history['records']) > 1000:
            self.history['records'] = self.history['records'][-1000:]
        
        # ファイル保存
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Failed to save history: {e}")
    
    def get_threshold_summary(self) -> Dict:
        """現在の適応的閾値サマリーを取得"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'global': {
                'min_proba': self.get_adaptive_threshold('min_proba'),
                'min_margin': self.get_adaptive_threshold('min_margin')
            },
            'per_section': {}
        }
        
        # セクション別
        for section in ['Chorus', 'Verse', 'Bridge']:
            summary['per_section'][section] = {
                'min_proba': self.get_adaptive_threshold('min_proba', section),
                'min_margin': self.get_adaptive_threshold('min_margin', section)
            }
        
        return summary
    
    def export_to_yaml(self, output_path: str):
        """
        適応的閾値をYAML形式でエクスポート（gate_prod.yaml更新用）
        
        Args:
            output_path: 出力先YAMLパス
        """
        summary = self.get_threshold_summary()
        
        yaml_content = f"""# Adaptive Safety Thresholds - Generated by AdaptiveThresholdManager
# Generated: {summary['timestamp']}
# Base Ratio: {self.base_ratio} (min_proba), {self.margin_ratio} (min_margin)

safety:
  # Global adaptive thresholds
  min_proba: {summary['global']['min_proba']:.3f}
  min_margin: {summary['global']['min_margin']:.3f}
  fallback_target: "safe-kit"
  
  # Section-specific adaptive thresholds
  by_section:
    Chorus:
      min_proba: {summary['per_section']['Chorus']['min_proba']:.3f}
      min_margin: {summary['per_section']['Chorus']['min_margin']:.3f}
    
    Verse:
      min_proba: {summary['per_section']['Verse']['min_proba']:.3f}
      min_margin: {summary['per_section']['Verse']['min_margin']:.3f}
    
    Bridge:
      min_proba: {summary['per_section']['Bridge']['min_proba']:.3f}
      min_margin: {summary['per_section']['Bridge']['min_margin']:.3f}
"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            self.logger.info(f"Exported adaptive thresholds to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export YAML: {e}")


# Example usage
if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create manager
    manager = AdaptiveThresholdManager(
        prometheus_url='http://localhost:9090',
        base_ratio=0.90,
        margin_ratio=0.90
    )
    
    # Get adaptive thresholds
    print("\n=== Adaptive Thresholds ===")
    summary = manager.get_threshold_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # Export to YAML
    output_path = 'data/adaptive_safety_thresholds.yaml'
    manager.export_to_yaml(output_path)
    print(f"\n✅ Exported to {output_path}")
