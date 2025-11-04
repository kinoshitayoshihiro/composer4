#!/usr/bin/env python3
"""
Auto Fallback Logic for Shadow Testing

Monitors Prometheus metrics and automatically triggers fallback from v3 to v1
when degradation conditions are detected.

Fallback Conditions (any one triggers):
1. Accent Score Delta < -5pt
2. p95 Latency > 150ms
3. Error Rate > 1%

Features:
- Prometheus metrics polling every 30s
- Slack notification on fallback trigger
- Config file update (v3 → v1 switch)
- Graceful restart trigger
"""

import requests
import time
import json
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
import signal

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FallbackConditions:
    """Fallback条件判定結果"""
    accent_delta_critical: bool  # delta < -5pt
    latency_critical: bool        # p95 > 150ms
    error_rate_critical: bool     # error > 1%
    
    def is_triggered(self) -> bool:
        """いずれかの条件を満たすか"""
        return self.accent_delta_critical or self.latency_critical or self.error_rate_critical
    
    def get_reasons(self) -> list:
        """トリガー理由のリスト"""
        reasons = []
        if self.accent_delta_critical:
            reasons.append("Accent Score Delta < -5pt")
        if self.latency_critical:
            reasons.append("p95 Latency > 150ms")
        if self.error_rate_critical:
            reasons.append("Error Rate > 1%")
        return reasons


class AutoFallback:
    """自動フォールバック管理クラス"""
    
    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        config_path: str = "config/model_config.yaml",
        slack_webhook_url: Optional[str] = None,
        poll_interval: int = 30,
        accent_threshold: float = -0.05,
        latency_threshold: float = 150.0,
        error_threshold: float = 0.01
    ):
        """
        Args:
            prometheus_url: PrometheusのURL
            config_path: モデル設定ファイルパス
            slack_webhook_url: Slack Webhook URL (任意)
            poll_interval: ポーリング間隔(秒)
            accent_threshold: Accent Score Delta閾値
            latency_threshold: p95 Latency閾値(ms)
            error_threshold: Error Rate閾値
        """
        self.prometheus_url = prometheus_url
        self.config_path = config_path
        self.slack_webhook_url = slack_webhook_url
        self.poll_interval = poll_interval
        
        # 閾値
        self.accent_threshold = accent_threshold
        self.latency_threshold = latency_threshold
        self.error_threshold = error_threshold
        
        # 状態管理
        self.is_fallback_triggered = False
        self.fallback_time: Optional[datetime] = None
    
    def query_prometheus(self, query: str) -> Optional[float]:
        """
        Prometheusにクエリを送信して最新値を取得
        
        Args:
            query: PromQLクエリ
        
        Returns:
            クエリ結果の値 (取得失敗時はNone)
        """
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            response = requests.get(url, params={'query': query}, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                # 最初の結果の値を返す
                value = float(data['data']['result'][0]['value'][1])
                return value
            return None
        except Exception as e:
            logger.error(f"Prometheus query failed: {query}, error: {e}")
            return None
    
    def check_conditions(self) -> Tuple[FallbackConditions, Dict[str, Optional[float]]]:
        """
        フォールバック条件をチェック
        
        Returns:
            (条件判定結果, メトリクス値の辞書)
        """
        # メトリクス取得
        accent_delta = self.query_prometheus("guitar_shadow_accent_delta")
        latency_p95 = self.query_prometheus("guitar_v3_latency_p95_ms")
        error_rate = self.query_prometheus("guitar_v3_error_rate")
        
        metrics = {
            'accent_delta': accent_delta,
            'latency_p95': latency_p95,
            'error_rate': error_rate
        }
        
        # 条件判定
        conditions = FallbackConditions(
            accent_delta_critical=(accent_delta is not None and accent_delta < self.accent_threshold),
            latency_critical=(latency_p95 is not None and latency_p95 > self.latency_threshold),
            error_rate_critical=(error_rate is not None and error_rate > self.error_threshold)
        )
        
        return conditions, metrics
    
    def send_slack_notification(self, conditions: FallbackConditions, metrics: Dict[str, Optional[float]]):
        """
        Slackに通知を送信
        
        Args:
            conditions: フォールバック条件判定結果
            metrics: メトリクス値
        """
        if not self.slack_webhook_url:
            logger.info("Slack webhook URL not configured, skipping notification")
            return
        
        reasons = conditions.get_reasons()
        reason_text = "\n".join(f"• {r}" for r in reasons)
        
        # メトリクス値のフォーマット
        accent_str = f"{metrics['accent_delta']:.2%}" if metrics['accent_delta'] is not None else "N/A"
        latency_str = f"{metrics['latency_p95']:.1f}ms" if metrics['latency_p95'] is not None else "N/A"
        error_str = f"{metrics['error_rate']:.2%}" if metrics['error_rate'] is not None else "N/A"
        
        message = {
            "text": "🚨 *Shadow Testing Auto Fallback Triggered*",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚨 Shadow Testing Auto Fallback Triggered"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Trigger Conditions:*\n{reason_text}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Accent Delta:*\n{accent_str}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*p95 Latency:*\n{latency_str}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Error Rate:*\n{error_str}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Fallback Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Action:* Switching from v3 to v1 model"
                    }
                }
            ]
        }
        
        try:
            response = requests.post(
                self.slack_webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            response.raise_for_status()
            logger.info("Slack notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def update_config_file(self):
        """
        設定ファイルを更新 (v3 → v1 に切り替え)
        
        Note: この実装ではYAML設定ファイルの guitar_model_version を
        'v3' から 'v1' に書き換える想定
        """
        try:
            if not os.path.exists(self.config_path):
                logger.error(f"Config file not found: {self.config_path}")
                return False
            
            # ファイル読み込み
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # v3 → v1 置換
            # YAML形式想定: guitar_model_version: v3
            new_content = content.replace(
                'guitar_model_version: v3',
                'guitar_model_version: v1'
            )
            
            if new_content == content:
                logger.warning("No v3 configuration found in config file")
                return False
            
            # バックアップ作成
            backup_path = f"{self.config_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Config backup created: {backup_path}")
            
            # 新設定書き込み
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logger.info(f"Config file updated: v3 → v1")
            
            return True
        except Exception as e:
            logger.error(f"Failed to update config file: {e}")
            return False
    
    def trigger_graceful_restart(self):
        """
        グレースフルリスタートをトリガー
        
        Note: この実装では親プロセスにSIGHUPを送信してリロードをトリガー
        実際の運用環境ではkubectl rollout restartなどを使用
        """
        try:
            # 親プロセスのPIDを取得
            parent_pid = os.getppid()
            
            # SIGHUPシグナル送信
            os.kill(parent_pid, signal.SIGHUP)
            logger.info(f"Sent SIGHUP to parent process (PID: {parent_pid})")
            
            return True
        except Exception as e:
            logger.error(f"Failed to trigger graceful restart: {e}")
            return False
    
    def trigger_fallback(self, conditions: FallbackConditions, metrics: Dict[str, Optional[float]]):
        """
        フォールバックを実行
        
        Args:
            conditions: フォールバック条件判定結果
            metrics: メトリクス値
        """
        logger.warning("========================================")
        logger.warning("FALLBACK TRIGGERED")
        logger.warning("========================================")
        
        reasons = conditions.get_reasons()
        logger.warning(f"Reasons: {', '.join(reasons)}")
        logger.warning(f"Metrics: {metrics}")
        
        # 1. Slack通知
        self.send_slack_notification(conditions, metrics)
        
        # 2. 設定ファイル更新
        config_updated = self.update_config_file()
        
        # 3. グレースフルリスタート
        if config_updated:
            restart_triggered = self.trigger_graceful_restart()
            if restart_triggered:
                logger.info("Fallback completed successfully")
            else:
                logger.error("Fallback partial: config updated but restart failed")
        else:
            logger.error("Fallback failed: config update failed")
        
        # 状態更新
        self.is_fallback_triggered = True
        self.fallback_time = datetime.now()
    
    def run(self):
        """
        メインループ: 定期的にメトリクスをチェックしてフォールバックを監視
        """
        logger.info("Auto Fallback Monitor started")
        logger.info(f"Prometheus URL: {self.prometheus_url}")
        logger.info(f"Config path: {self.config_path}")
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info(f"Thresholds: accent={self.accent_threshold}, latency={self.latency_threshold}ms, error={self.error_threshold}")
        
        while not self.is_fallback_triggered:
            try:
                # 条件チェック
                conditions, metrics = self.check_conditions()
                
                # メトリクス値をログ出力
                logger.info(
                    f"Metrics - Accent Delta: {metrics['accent_delta']:.4f if metrics['accent_delta'] else 'N/A'}, "
                    f"p95 Latency: {metrics['latency_p95']:.1f if metrics['latency_p95'] else 'N/A'}ms, "
                    f"Error Rate: {metrics['error_rate']:.4f if metrics['error_rate'] else 'N/A'}"
                )
                
                # フォールバック判定
                if conditions.is_triggered():
                    self.trigger_fallback(conditions, metrics)
                    break
                
                # 待機
                time.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Auto Fallback Monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.poll_interval)
        
        if self.is_fallback_triggered:
            logger.info(f"Fallback triggered at {self.fallback_time}")
        logger.info("Auto Fallback Monitor finished")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Fallback Monitor for Shadow Testing")
    parser.add_argument(
        "--prometheus-url",
        default=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        help="Prometheus URL (default: http://localhost:9090)"
    )
    parser.add_argument(
        "--config-path",
        default=os.getenv("CONFIG_PATH", "config/model_config.yaml"),
        help="Model config file path (default: config/model_config.yaml)"
    )
    parser.add_argument(
        "--slack-webhook",
        default=os.getenv("SLACK_WEBHOOK_URL"),
        help="Slack Webhook URL for notifications"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Polling interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--accent-threshold",
        type=float,
        default=-0.05,
        help="Accent Score Delta threshold (default: -0.05)"
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        default=150.0,
        help="p95 Latency threshold in ms (default: 150.0)"
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.01,
        help="Error Rate threshold (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # Auto Fallback Monitor起動
    monitor = AutoFallback(
        prometheus_url=args.prometheus_url,
        config_path=args.config_path,
        slack_webhook_url=args.slack_webhook,
        poll_interval=args.poll_interval,
        accent_threshold=args.accent_threshold,
        latency_threshold=args.latency_threshold,
        error_threshold=args.error_threshold
    )
    
    monitor.run()


if __name__ == "__main__":
    main()
