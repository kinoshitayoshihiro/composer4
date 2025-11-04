# Auto Fallback Monitor for Shadow Testing

自動フォールバックモニター: Shadow Testingでv3モデルの劣化を検出し、自動的にv1へ切り替えます。

## 概要

Prometheusメトリクスを30秒間隔でポーリングし、以下のいずれかの条件を満たした場合に自動フォールバックを実行:

1. **Accent Score Delta < -5pt** - v3がv1より5ポイント以上劣化
2. **p95 Latency > 150ms** - v3のp95レイテンシが150ms超過
3. **Error Rate > 1%** - v3のエラー率が1%超過

## 機能

- ✅ **Prometheusメトリクス監視**: 定期的にKPIメトリクスをポーリング
- ✅ **3条件判定**: Accent Delta、Latency、Error Rateを自動評価
- ✅ **Slack通知**: フォールバックトリガー時にリッチフォーマット通知
- ✅ **設定ファイル自動更新**: `guitar_model_version: v3` → `v1` 書き換え
- ✅ **グレースフルリスタート**: SIGHUPシグナルでアプリケーションリロード
- ✅ **バックアップ作成**: 設定ファイル変更前に自動バックアップ

## 使用方法

### 基本起動

```bash
# デフォルト設定で起動
python monitoring/auto_fallback.py

# カスタム設定で起動
python monitoring/auto_fallback.py \
  --prometheus-url http://prometheus:9090 \
  --config-path config/model_config.yaml \
  --slack-webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  --poll-interval 30
```

### 環境変数

```bash
export PROMETHEUS_URL=http://localhost:9090
export CONFIG_PATH=config/model_config.yaml
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

python monitoring/auto_fallback.py
```

### コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--prometheus-url` | `http://localhost:9090` | PrometheusサーバーURL |
| `--config-path` | `config/model_config.yaml` | モデル設定ファイルパス |
| `--slack-webhook` | (環境変数) | Slack Webhook URL |
| `--poll-interval` | `30` | ポーリング間隔(秒) |
| `--accent-threshold` | `-0.05` | Accent Score Delta閾値 |
| `--latency-threshold` | `150.0` | p95 Latency閾値(ms) |
| `--error-threshold` | `0.01` | Error Rate閾値 |

## アーキテクチャ

### フォールバックフロー

```
┌─────────────────────┐
│ Auto Fallback       │
│ Monitor             │◄──── 30秒ポーリング
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Prometheus          │
│ Metrics Query       │
│ - accent_delta      │
│ - latency_p95       │
│ - error_rate        │
└──────────┬──────────┘
           │
           ▼
    ┌──────────┐
    │ 条件判定  │
    │ (いずれか)│
    └─────┬────┘
          │
          ▼
    ┌──────────┐
    │ Slack通知 │──────┐
    └──────────┘      │
          │            │
          ▼            │
    ┌──────────┐      │
    │ 設定更新  │      │
    │ v3 → v1  │      │
    └─────┬────┘      │
          │            │
          ▼            │
    ┌──────────┐      │
    │ SIGHUP   │      │
    │ 送信      │      │
    └──────────┘      │
          │            │
          ▼            ▼
    ┌─────────────────┐
    │ フォールバック   │
    │ 完了             │
    └─────────────────┘
```

### Prometheusメトリクス

監視対象メトリクス:

```promql
# Accent Score Delta
guitar_shadow_accent_delta

# p95 Latency
guitar_v3_latency_p95_ms

# Error Rate
guitar_v3_error_rate
```

### 設定ファイル形式

`config/model_config.yaml`:

```yaml
guitar_model_version: v3  # ← これがv1に書き換えられる

shadow_testing:
  fallback_conditions:
    accent_delta_threshold: -0.05
    latency_p95_threshold_ms: 150.0
    error_rate_threshold: 0.01
```

## Slack通知フォーマット

フォールバックトリガー時のSlack通知例:

```
🚨 Shadow Testing Auto Fallback Triggered

Trigger Conditions:
• Accent Score Delta < -5pt

Accent Delta: -7.32%
p95 Latency: 125.3ms
Error Rate: 0.15%
Fallback Time: 2025-01-16 14:32:15

Action: Switching from v3 to v1 model
```

## ログ出力

### 通常時

```
2025-01-16 14:30:00 - auto_fallback - INFO - Auto Fallback Monitor started
2025-01-16 14:30:00 - auto_fallback - INFO - Prometheus URL: http://localhost:9090
2025-01-16 14:30:00 - auto_fallback - INFO - Poll interval: 30s
2025-01-16 14:30:30 - auto_fallback - INFO - Metrics - Accent Delta: -0.0123, p95 Latency: 87.5ms, Error Rate: 0.0000
2025-01-16 14:31:00 - auto_fallback - INFO - Metrics - Accent Delta: -0.0234, p95 Latency: 91.2ms, Error Rate: 0.0000
```

### フォールバック時

```
2025-01-16 14:32:15 - auto_fallback - WARNING - ========================================
2025-01-16 14:32:15 - auto_fallback - WARNING - FALLBACK TRIGGERED
2025-01-16 14:32:15 - auto_fallback - WARNING - ========================================
2025-01-16 14:32:15 - auto_fallback - WARNING - Reasons: Accent Score Delta < -5pt
2025-01-16 14:32:15 - auto_fallback - WARNING - Metrics: {'accent_delta': -0.0732, 'latency_p95': 125.3, 'error_rate': 0.0015}
2025-01-16 14:32:15 - auto_fallback - INFO - Slack notification sent successfully
2025-01-16 14:32:15 - auto_fallback - INFO - Config backup created: config/model_config.yaml.backup_20250116_143215
2025-01-16 14:32:15 - auto_fallback - INFO - Config file updated: v3 → v1
2025-01-16 14:32:15 - auto_fallback - INFO - Sent SIGHUP to parent process (PID: 12345)
2025-01-16 14:32:15 - auto_fallback - INFO - Fallback completed successfully
```

## 運用

### Docker環境での起動

```bash
# Prometheusコンテナと同じネットワークで起動
docker run -d \
  --name auto-fallback \
  --network monitoring \
  -e PROMETHEUS_URL=http://prometheus:9090 \
  -e SLACK_WEBHOOK_URL=https://hooks.slack.com/... \
  -v $(pwd)/config:/app/config \
  composer-auto-fallback:latest
```

### Kubernetes環境

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: auto-fallback-config
data:
  PROMETHEUS_URL: "http://prometheus-service:9090"
  CONFIG_PATH: "/config/model_config.yaml"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auto-fallback-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: auto-fallback
  template:
    metadata:
      labels:
        app: auto-fallback
    spec:
      containers:
      - name: monitor
        image: composer-auto-fallback:latest
        envFrom:
        - configMapRef:
            name: auto-fallback-config
        - secretRef:
            name: slack-webhook-secret
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: model-config
```

### Systemd Service

```ini
[Unit]
Description=Auto Fallback Monitor for Shadow Testing
After=network.target prometheus.service

[Service]
Type=simple
User=composer
WorkingDirectory=/opt/composer2-3
Environment="PROMETHEUS_URL=http://localhost:9090"
Environment="CONFIG_PATH=config/model_config.yaml"
EnvironmentFile=/etc/composer/auto-fallback.env
ExecStart=/opt/composer2-3/.venv311/bin/python monitoring/auto_fallback.py
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

## テスト

### 手動テスト

1. **Prometheusメトリクスモック**:

```bash
# Prometheusが返すメトリクスを手動設定
curl -X POST http://localhost:9090/api/v1/admin/tsdb/delete_series \
  -d 'match[]=guitar_shadow_accent_delta'

# テスト値を挿入 (メトリクスをpushgateway経由で送信)
echo "guitar_shadow_accent_delta -0.08" | curl --data-binary @- \
  http://localhost:9091/metrics/job/test
```

2. **モニター起動**:

```bash
python monitoring/auto_fallback.py \
  --prometheus-url http://localhost:9090 \
  --accent-threshold -0.05 \
  --poll-interval 10
```

3. **期待動作**:
   - 10秒後にフォールバックトリガー
   - Slack通知送信
   - `config/model_config.yaml` が `v1` に更新
   - バックアップファイル作成

### ユニットテスト

```python
import unittest
from monitoring.auto_fallback import AutoFallback, FallbackConditions

class TestAutoFallback(unittest.TestCase):
    def test_fallback_conditions_triggered(self):
        conditions = FallbackConditions(
            accent_delta_critical=True,
            latency_critical=False,
            error_rate_critical=False
        )
        self.assertTrue(conditions.is_triggered())
        self.assertEqual(conditions.get_reasons(), ["Accent Score Delta < -5pt"])
    
    def test_query_prometheus_success(self):
        monitor = AutoFallback(prometheus_url="http://localhost:9090")
        # Mock Prometheus response
        # ...
```

## トラブルシューティング

### Prometheus接続エラー

```
ERROR - Prometheus query failed: guitar_shadow_accent_delta, error: Connection refused
```

**解決策**:
- Prometheusサーバーが起動しているか確認: `curl http://localhost:9090/metrics`
- `--prometheus-url` が正しいか確認

### 設定ファイル更新失敗

```
ERROR - Failed to update config file: [Errno 2] No such file or directory
```

**解決策**:
- `--config-path` のパスが正しいか確認
- ファイルの書き込み権限があるか確認: `ls -l config/model_config.yaml`

### Slack通知送信失敗

```
ERROR - Failed to send Slack notification: 400 Client Error: Bad Request
```

**解決策**:
- Slack Webhook URLが正しいか確認
- URLが期限切れでないか確認（Slackで再生成）

## 関連ドキュメント

- [SHADOW_TESTING_DESIGN.md](../SHADOW_TESTING_DESIGN.md) - Shadow Testing設計仕様
- [monitoring/grafana_shadow_dashboard.json](grafana_shadow_dashboard.json) - ダッシュボード定義
- [monitoring/prometheus/alerts/guitar_shadow_alerts.yml](prometheus/alerts/guitar_shadow_alerts.yml) - アラートルール

## ライセンス

Copyright (c) 2025 Composer Team
