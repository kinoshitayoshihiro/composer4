# Guitar v3 KPI Monitoring System

本番投入されたGuitar v3のリアルタイムKPI監視システムです。

## アーキテクチャ

```
CSV Logs → kpi_collector.py → Prometheus → Grafana
                             ↓
                        Alertmanager → Slack
```

## クイックスタート

### 1. Docker環境起動

```bash
cd monitoring/
docker-compose up -d
```

起動確認：
- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093
- Grafana: http://localhost:3000 (admin/admin)

### 2. KPI収集スクリプト実行

```bash
# 手動実行
./run_kpi_monitor.sh

# cron登録（1分ごと）
crontab -e
# 以下を追加：
# */1 * * * * /path/to/monitoring/run_kpi_monitor.sh >> /path/to/monitoring/cron.log 2>&1
```

### 3. Grafanaダッシュボード確認

1. ブラウザで http://localhost:3000 を開く
2. admin/admin でログイン
3. 左メニュー「Dashboards」→「Guitar v3」→「Guitar v3 KPI Dashboard」

## コンポーネント

### kpi_collector.py
- CSVログ自動検索（`**/*kpi*.csv`, `**/*canary*.csv`）
- 統計計算（mean/min/max、セクション別）
- Prometheusメトリクス出力（metrics.prom）
- KPIゲート判定（7項目、exit code 0/1）

### Prometheusメトリクス
- `guitar_v3_accent_score_mean`: 平均アクセントスコア（目標≥0.65）
- `guitar_v3_chord_fit_mean`: 平均コード適合率（目標≥0.60）
- `guitar_v3_ml_usage_rate`: ML採用率（目標≥0.70）
- `guitar_v3_safety_fallback_rate`: セーフティ発動率（目標≤0.10）
- `guitar_v3_section_accent_score{section="chorus"}`: セクション別アクセント

### Grafanaダッシュボード（9パネル）
1. **Accent Score (Mean)** - グラフ+アラート
2. **Chord Fit (Mean)** - グラフ+アラート
3. **ML Usage Rate** - グラフ+アラート
4. **Safety Fallback Rate** - グラフ+アラート
5. **Accent Score by Section** - Chorus/Verse/Bridge/Intro
6. **ML Usage by Section** - セクション別ML採用率
7. **Top-1 Probability (Mean)** - ML確信度
8. **Total Evaluated Cases** - 統計パネル
9. **KPI Status Summary** - テーブル（色付き）

### アラートルール（11種類）

**Critical（即座対応）**:
- `GuitarV3AccentScoreCritical` (<65%, 5分)
- `GuitarV3ChordFitCritical` (<60%, 5分)
- `GuitarV3MLUsageCritical` (<70%, 5分)
- `GuitarV3MetricsAbsent` (5分間データなし)

**Warning（監視強化）**:
- `GuitarV3AccentScoreWarning` (<70%, 5分)
- `GuitarV3ChordFitWarning` (<65%, 5分)
- `GuitarV3MLUsageWarning` (<80%, 5分)
- `GuitarV3HighSafetyFallback` (>10%, 10分)
- `GuitarV3SectionAccentScoreDrop` (<70%, 5分)
- `GuitarV3LowDataVolume` (収集率低下, 10分)

**Info（情報）**:
- `GuitarV3VeryHighSafetyFallback` (>15%, 10分)

## トラブルシューティング

### KPI収集が動かない
```bash
# Python環境確認
.venv311/bin/python monitoring/kpi_collector.py --help

# ログファイル確認
ls -la logs/*.csv data/*.csv

# 実行権限確認
chmod +x monitoring/kpi_collector.py monitoring/run_kpi_monitor.sh
```

### Prometheusにデータが入らない
```bash
# metrics.prom確認
cat monitoring/metrics.prom

# file_sd_configs確認
cat monitoring/filesd/guitar_v3_kpi.json

# Prometheus設定リロード
curl -X POST http://localhost:9090/-/reload
```

### Grafanaダッシュボードが表示されない
```bash
# プロビジョニング確認
docker exec guitar_v3_grafana ls /etc/grafana/dashboards/
docker exec guitar_v3_grafana ls /etc/grafana/provisioning/

# コンテナログ確認
docker logs guitar_v3_grafana
```

## Slack連携

環境変数 `SLACK_WEBHOOK_URL` を設定してSlack通知を有効化：

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
./run_kpi_monitor.sh
```

アラート例：
```
⚠️  WARNING: Guitar v3 Accent Score: 0.68 (warning <70%)
🔴 CRITICAL: Guitar v3 Chord Fit: 0.58 (critical <60%)
```

## メンテナンス

### データ保持期間
- Prometheus: 30日間
- Grafana: 無期限（ダッシュボード設定）

### バックアップ
```bash
# Prometheusデータ
docker exec guitar_v3_prometheus tar czf - /prometheus > prometheus_backup.tar.gz

# Grafanaデータ
docker exec guitar_v3_grafana tar czf - /var/lib/grafana > grafana_backup.tar.gz
```

### 停止・再起動
```bash
# 停止
docker-compose down

# 再起動
docker-compose restart

# ログ確認
docker-compose logs -f
```

## KPIゲート閾値

| KPI | Critical | Warning | 目標 |
|-----|----------|---------|------|
| **Accent Score** | <65% | <70% | ≥65% |
| **Chord Fit** | <60% | <65% | ≥60% |
| **ML Usage** | <70% | <80% | ≥70% |
| **Safety Fallback** | - | >10% | ≤10% |

## ファイル構成

```
monitoring/
├── docker-compose.yml           # Docker環境定義
├── prometheus.yml               # Prometheus設定
├── guitar_v3_alerts.yml         # アラートルール
├── alertmanager.yml             # Alertmanager設定
├── grafana_dashboard.json       # Grafanaダッシュボード
├── grafana_provisioning/        # Grafana自動プロビジョニング
│   ├── datasources/
│   │   └── prometheus.yml
│   └── dashboards/
│       └── dashboards.yml
├── filesd/
│   └── guitar_v3_kpi.json       # Prometheusサービスディスカバリ
├── kpi_collector.py             # KPI収集エンジン
├── run_kpi_monitor.sh           # 自動監視スクリプト（cron用）
├── metrics.prom                 # Prometheusメトリクス（出力）
├── kpi_stats.json               # JSON統計（出力）
├── alerts.log                   # アラートログ（出力）
└── README.md                    # 本ドキュメント
```

## 参考リンク

- [Prometheus公式](https://prometheus.io/)
- [Grafana公式](https://grafana.com/)
- [PromQL入門](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [本番投入レポート](../PRODUCTION_DEPLOYMENT_REPORT.md)
- [GitHub Release](../GITHUB_RELEASE_v3_GUITAR_ML.md)
