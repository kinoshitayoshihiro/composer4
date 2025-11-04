#!/usr/bin/env python3
"""
Safe-Kit SLO Monitor - Phase 24.4

Safe-Kit使用率をPrometheusから監視し、SLO違反時にアクションを実行。

SLO (Service Level Objectives):
- Warning: safe_kit_invocations_rate_5m > 8% が30分連続
  → Action: Exploration epsilonを50%に自動低減
  
- Critical: safe_kit_invocations_rate_5m > 12% が30分連続
  → Action: Exploration完全停止 + アラート

Usage:
    # 定期実行（cron推奨: 5分ごと）
    python monitoring/safekit_slo_monitor.py \
        --prometheus-url http://localhost:9090 \
        --check-window 30 \
        --warning-threshold 0.08 \
        --critical-threshold 0.12

Environment Variables:
    EXPLORATION_EPSILON_OVERRIDE: 探索率オーバーライド（0.0-1.0）
    EXPLORATION_DISABLED: 探索完全停止フラグ（"true"で停止）
"""

import argparse
import requests
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SafeKitSLOMonitor:
    """Safe-Kit使用率SLO監視"""
    
    def __init__(
        self,
        prometheus_url: str,
        warning_threshold: float = 0.08,
        critical_threshold: float = 0.12,
        check_window_minutes: int = 30,
        state_file: str = "data/safekit_slo_state.json"
    ):
        self.prometheus_url = prometheus_url
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_window_minutes = check_window_minutes
        self.state_file = Path(state_file)
        
        # 状態ファイル読み込み
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """前回チェックの状態を読み込み"""
        if not self.state_file.exists():
            return {
                'last_check': None,
                'warning_start': None,
                'critical_start': None,
                'exploration_reduced': False,
                'exploration_disabled': False
            }
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return {}
    
    def _save_state(self):
        """状態をファイルに保存"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def query_prometheus(self, query: str) -> Optional[float]:
        """
        Prometheusクエリ実行
        
        Args:
            query: PromQLクエリ文字列
        
        Returns:
            クエリ結果の値（単一メトリクス想定）
        """
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if data['status'] != 'success':
                logger.error(f"Prometheus query failed: {data}")
                return None
            
            result = data['data']['result']
            if not result:
                logger.warning(f"No data for query: {query}")
                return None
            
            # 単一値を返す（複数seriesがある場合は最大値）
            values = [float(r['value'][1]) for r in result]
            return max(values)
        
        except Exception as e:
            logger.error(f"Prometheus query error: {e}")
            return None
    
    def check_safekit_rate(self) -> Dict:
        """
        Safe-Kit使用率をチェック
        
        Returns:
            {
                'current_rate': float,
                'status': 'ok' | 'warning' | 'critical',
                'duration_minutes': int  # SLO違反継続時間
            }
        """
        # Safe-Kit使用率（5分平均）
        # rate(safe_kit_invocations_total[5m]) / rate(pattern_generation_total[5m])
        query = (
            'rate(guitar_safekit_fallback_total[5m]) / '
            '(rate(guitar_v3_filter_total[5m]) + rate(guitar_safekit_fallback_total[5m]))'
        )
        
        current_rate = self.query_prometheus(query)
        
        if current_rate is None:
            logger.error("Failed to get Safe-Kit rate from Prometheus")
            return {'current_rate': None, 'status': 'unknown', 'duration_minutes': 0}
        
        logger.info(f"Current Safe-Kit rate: {current_rate:.2%}")
        
        # 状態判定
        now = datetime.now()
        
        if current_rate >= self.critical_threshold:
            # Critical状態
            if self.state['critical_start'] is None:
                self.state['critical_start'] = now.isoformat()
                logger.warning(f"🔴 CRITICAL threshold exceeded: {current_rate:.2%} >= {self.critical_threshold:.2%}")
            
            critical_start = datetime.fromisoformat(self.state['critical_start'])
            duration = (now - critical_start).total_seconds() / 60
            
            return {
                'current_rate': current_rate,
                'status': 'critical',
                'duration_minutes': int(duration)
            }
        
        elif current_rate >= self.warning_threshold:
            # Warning状態
            if self.state['warning_start'] is None:
                self.state['warning_start'] = now.isoformat()
                logger.warning(f"⚠️  WARNING threshold exceeded: {current_rate:.2%} >= {self.warning_threshold:.2%}")
            
            warning_start = datetime.fromisoformat(self.state['warning_start'])
            duration = (now - warning_start).total_seconds() / 60
            
            return {
                'current_rate': current_rate,
                'status': 'warning',
                'duration_minutes': int(duration)
            }
        
        else:
            # OK状態 - リセット
            if self.state['warning_start'] or self.state['critical_start']:
                logger.info(f"✅ Safe-Kit rate back to normal: {current_rate:.2%}")
                self.state['warning_start'] = None
                self.state['critical_start'] = None
            
            return {
                'current_rate': current_rate,
                'status': 'ok',
                'duration_minutes': 0
            }
    
    def execute_action(self, status: str, duration_minutes: int):
        """
        SLO違反時のアクション実行
        
        Args:
            status: 'ok' | 'warning' | 'critical'
            duration_minutes: 継続時間（分）
        """
        if status == 'critical' and duration_minutes >= self.check_window_minutes:
            # Critical: Exploration完全停止
            if not self.state['exploration_disabled']:
                logger.critical(f"🚨 CRITICAL: Safe-Kit rate > {self.critical_threshold:.0%} for {duration_minutes}min")
                logger.critical("   Action: Disabling Exploration completely")
                
                os.environ['EXPLORATION_DISABLED'] = 'true'
                self.state['exploration_disabled'] = True
                self._save_state()
                
                # アラート送信（Slack/PagerDuty等）
                self._send_alert('critical', duration_minutes)
        
        elif status == 'warning' and duration_minutes >= self.check_window_minutes:
            # Warning: Exploration epsilon 50%低減
            if not self.state['exploration_reduced']:
                logger.warning(f"⚠️  WARNING: Safe-Kit rate > {self.warning_threshold:.0%} for {duration_minutes}min")
                logger.warning("   Action: Reducing Exploration epsilon to 50%")
                
                # 現在のepsilonを0.5倍に
                current_epsilon = float(os.environ.get('EXPLORATION_EPSILON_OVERRIDE', '0.10'))
                reduced_epsilon = current_epsilon * 0.5
                os.environ['EXPLORATION_EPSILON_OVERRIDE'] = str(reduced_epsilon)
                
                self.state['exploration_reduced'] = True
                self._save_state()
                
                # アラート送信
                self._send_alert('warning', duration_minutes)
        
        elif status == 'ok':
            # 正常復帰: フラグリセット
            if self.state['exploration_reduced'] or self.state['exploration_disabled']:
                logger.info("✅ Safe-Kit rate normalized, restoring Exploration settings")
                
                if 'EXPLORATION_EPSILON_OVERRIDE' in os.environ:
                    del os.environ['EXPLORATION_EPSILON_OVERRIDE']
                if 'EXPLORATION_DISABLED' in os.environ:
                    del os.environ['EXPLORATION_DISABLED']
                
                self.state['exploration_reduced'] = False
                self.state['exploration_disabled'] = False
                self._save_state()
    
    def _send_alert(self, level: str, duration_minutes: int):
        """
        アラート送信（Slack/PagerDuty等）
        
        実装例: Slack Webhook
        """
        # TODO: 実際のアラート送信実装
        logger.info(f"📢 Alert sent: level={level}, duration={duration_minutes}min")
    
    def run_check(self):
        """SLOチェック実行（メインエントリポイント）"""
        logger.info("=== Safe-Kit SLO Monitor Check ===")
        
        result = self.check_safekit_rate()
        
        if result['status'] == 'unknown':
            logger.error("⚠️  Unable to determine Safe-Kit rate")
            return
        
        logger.info(f"Status: {result['status'].upper()}, "
                   f"Rate: {result['current_rate']:.2%}, "
                   f"Duration: {result['duration_minutes']}min")
        
        # アクション実行
        self.execute_action(result['status'], result['duration_minutes'])
        
        # 状態保存
        self.state['last_check'] = datetime.now().isoformat()
        self._save_state()


def main():
    parser = argparse.ArgumentParser(description="Safe-Kit SLO Monitor")
    parser.add_argument('--prometheus-url', type=str, default='http://localhost:9090',
                       help='Prometheus server URL')
    parser.add_argument('--warning-threshold', type=float, default=0.08,
                       help='Warning threshold (default: 0.08 = 8%)')
    parser.add_argument('--critical-threshold', type=float, default=0.12,
                       help='Critical threshold (default: 0.12 = 12%)')
    parser.add_argument('--check-window', type=int, default=30,
                       help='Check window in minutes (default: 30)')
    parser.add_argument('--state-file', type=str, default='data/safekit_slo_state.json',
                       help='State file path')
    
    args = parser.parse_args()
    
    monitor = SafeKitSLOMonitor(
        prometheus_url=args.prometheus_url,
        warning_threshold=args.warning_threshold,
        critical_threshold=args.critical_threshold,
        check_window_minutes=args.check_window,
        state_file=args.state_file
    )
    
    monitor.run_check()


if __name__ == "__main__":
    main()
