# Drums Production Ready Guide

**Phase 25.3完了**: Rhythm AI (Drums) 本番展開準備完了

---

## 📋 目次

1. [概要](#概要)
2. [使用方法](#使用方法)
3. [KPI Reference](#kpi-reference)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Canary Deployment](#canary-deployment)
6. [Troubleshooting](#troubleshooting)
7. [Rollback Procedures](#rollback-procedures)

---

## 概要

### Phase 25: Rhythm AI (Drums) 実装完了

**Phase 25の成果** (13タスク、4,250行):

- **Phase 25.0 (v3基盤統合)**: Drums KPIゲート、Safe-Kit、DrumPatternRecommender、DrumsGeneratorStage2拡張 (4タスク、964行)
- **Phase 25.1 (データセット構築)**: Stage1正規化、パターン抽出、位相正規化、Family分類、学習データセット (5タスク、1,920行)
- **Phase 25.2 (学習パイプライン)**: XGB/LogRegトレーニング、ML統合、スモークテスト (3タスク、850行)
- **Phase 25.3 (運用統合)**: Prometheus/Grafana監視、Canary展開設定、Auto-Recovery有効化、運用ドキュメント (4タスク、516行)

### 主要機能

#### 1. ML-Driven Pattern Recommendation
- **XGBoost**: リズムパターン選択（Kick/Snare/HiHat/Cymbal配置）
- **Logistic Regression**: Kit選択（ポップス・ロック・ジャズ等、15種類）
- **Top-1確率**: 平均60%以上（v3パターン信頼性）
- **Safety Fallback**: 確率15%未満でSafe-Kit発火

#### 2. 5-KPI Quality Gates
| KPI | 閾値 | 説明 |
|-----|------|------|
| `kick_downbeat_rate` | ≥ 0.80 | キックのダウンビート命中率（1拍目・3拍目） |
| `snare_backbeat_acc` | ≥ 0.85 | スネアのバックビート整合率（2&4拍目） |
| `hat_density_abs` | ≤ 2.0 | ハイハット密度許容誤差（QL単位、中央値） |
| `fill_placement_valid` | ≥ 0.95 | フィル配置妥当性（小節跨ぎ・休符崩壊なし） |
| `ml_used` | ≥ 0.90 | ML使用率（v3直採用率） |

#### 3. Auto-Recovery Mechanism
- **監視ウィンドウ**: 64バー
- **許容違反回数**: 10回/ウィンドウ
- **Recovery動作**: Safe-Kitへ自動フォールバック
- **クールダウン**: 16バー（再試行前の待機時間）

#### 4. Canary Deployment Strategy
- **4週間段階的ロールアウト**:
  - Week 1: Shadow (5% logging)
  - Week 2: Canary 5% (serving)
  - Week 3: Canary 20% (serving)
  - Week 4: Production 100% (full rollout)
- **Auto-Rollback Protection**: 4条件（Critical KPI, High error, Degradation, Latency）

---

## 使用方法

### 基本的な使用フロー

```python
from stage2.drums_generator_v3 import DrumsGeneratorStage2
from stage2.drum_pattern_recommender import DrumPatternRecommender

# 1. Recommender初期化（ML推論エンジン）
recommender = DrumPatternRecommender(
    model_path="data/patterns/drums_v3_ml_final.pickle",
    enable_ml=True,  # ML推論有効化
    use_kpi_gates=True,  # KPIゲート有効化
    auto_recovery=True,  # Auto-Recovery有効化
)

# 2. Generator初期化（ドラムMIDI生成）
generator = DrumsGeneratorStage2(
    pattern_recommender=recommender,
    kpi_gates_enabled=True,
)

# 3. MIDI生成
drums_events = generator.generate_drums(
    chords=[("C", "maj"), ("Am", "min"), ...],
    emotion="energetic",
    bpm=120,
    time_signature="4/4",
)

# 4. KPI検証
from quality.drums_kpi import compute_drums_kpi
kpi_results = compute_drums_kpi(drums_events)

print(f"Kick Downbeat Rate: {kpi_results['kick_downbeat_rate']:.2f}")
print(f"Snare Backbeat Acc: {kpi_results['snare_backbeat_acc']:.2f}")
print(f"ML Used: {kpi_results['ml_used']:.2f}")
```

### Feature Flags

Canary展開時にFeature Flagsで段階的に有効化:

```yaml
# config/canary_drums.yaml
feature_flags:
  use_ml_inference:
    shadow: true       # Shadow: ML推論のみ（logging）
    canary: true       # Canary: ML推論有効
    production: false  # Prod: 検証後に有効化
  
  auto_recovery:
    shadow: true       # Shadow: Recovery動作をログ記録
    canary: true       # Canary: Recovery有効
    production: false  # Prod: 検証後に有効化
  
  kpi_gates_strict:
    shadow: false      # Shadow: Strict gates無効
    canary: true       # Canary: Strict gates有効
    production: true   # Prod: Strict gates有効
```

---

## KPI Reference

### 1. Kick Downbeat Rate

**定義**: キックのダウンビート命中率（1拍目・3拍目への配置率）

**計算方法**:
```python
kick_downbeat_rate = (1拍目または3拍目のキック数) / (総小節数)
```

**閾値**:
- Production: ≥ 0.80
- Chorus: ≥ 0.85（安定重視）
- Intro/Outro: ≥ 0.70（柔軟）

**トラブルシューティング**:
- `< 0.80`: MLモデル再トレーニング検討
- `< 0.75`: Auto-Recovery発火（Safe-Kit発火）

---

### 2. Snare Backbeat Accuracy

**定義**: スネアのバックビート整合率（4/4で2&4拍目、6/8相応位置）

**計算方法**:
```python
snare_backbeat_acc = (バックビート位置のスネア数) / (総小節数)
```

**閾値**:
- Production: ≥ 0.85
- Chorus: ≥ 0.90（安定重視）
- Bridge: ≥ 0.80（変化許容）

**トラブルシューティング**:
- `< 0.85`: パターン抽出ルール見直し
- `< 0.80`: Auto-Recovery発火

---

### 3. HiHat Density Deviation

**定義**: ハイハット密度許容誤差（目標密度との絶対差の中央値、QL単位）

**計算方法**:
```python
hat_density_abs = median(|actual_density - target_density|)
```

**閾値**:
- Production: ≤ 2.0 QL
- Chorus: ≤ 1.5 QL（安定重視）
- Bridge: ≤ 2.5 QL（変化許容）

**トラブルシューティング**:
- `> 2.0`: 密度制御ロジック見直し
- `> 3.0`: Auto-Recovery発火

---

### 4. Fill Placement Validity

**定義**: フィル配置妥当性（小節跨ぎ・休符崩壊なし）

**計算方法**:
```python
fill_placement_valid = (妥当なフィル数) / (総フィル数)
```

**閾値**:
- Production: ≥ 0.95
- All sections: ≥ 0.95（統一）

**トラブルシューティング**:
- `< 0.95`: フィル配置ルール見直し
- `< 0.90`: Auto-Recovery発火

---

### 5. ML Usage Rate

**定義**: ML使用率（v3パターン直採用率）

**計算方法**:
```python
ml_used = (ML推論成功数) / (総パターン生成数)
```

**閾値**:
- Production: ≥ 0.90
- Shadow/Canary: ≥ 0.85

**トラブルシューティング**:
- `< 0.90`: MLモデル性能劣化の可能性
- `< 0.70`: Safe-Kit発火率が高すぎる（モデル再トレーニング推奨）

---

## Monitoring & Alerting

### Prometheus Metrics

**KPI Metrics**:
```prometheus
# Kick Downbeat Rate
drums_kick_downbeat_rate{section="chorus"}

# Snare Backbeat Accuracy
drums_snare_backbeat_acc{section="verse"}

# HiHat Density Deviation
drums_hat_density_actual{section="bridge"}
drums_hat_density_target{section="bridge"}

# Fill Placement Validity
drums_fill_placement_valid

# ML Usage
rate(drums_ml_used_total[5m])
rate(drums_patterns_total[5m])
```

**Performance Metrics**:
```prometheus
# Latency (p95)
histogram_quantile(0.95, rate(drums_recommend_duration_seconds_bucket[5m]))

# Cache Hit Rate
rate(drums_pattern_cache_hits_total[5m]) / rate(drums_pattern_cache_requests_total[5m])

# Error Rate
rate(drums_errors_total[5m]) / rate(drums_patterns_total[5m])
```

**Auto-Recovery Metrics**:
```prometheus
# Recovery Triggered Count
increase(drums_auto_recovery_triggered_total[10m])

# Recovery Frequency (per second)
rate(drums_auto_recovery_triggered_total[30m])
```

### Grafana Dashboards

**Dashboard: Drums KPI Overview**
- Location: `http://grafana.company.com/d/drums-kpi-overview`
- Panels:
  - Kick Downbeat Rate (timeseries)
  - Snare Backbeat Accuracy (timeseries)
  - HiHat Density Deviation (timeseries)
  - Fill Placement Validity (stat)
  - ML Usage Rate (gauge)

**Dashboard: Drums Performance**
- Location: `http://grafana.company.com/d/drums-performance`
- Panels:
  - Latency p95 (timeseries)
  - Cache Hit Rate (timeseries)
  - Error Rate (stat)
  - Auto-Recovery Events (timeseries)

**Dashboard: Canary Comparison**
- Location: `http://grafana.company.com/d/drums-canary-comparison`
- Panels:
  - Canary vs Prod: Kick Downbeat Rate (comparison)
  - Canary vs Prod: Snare Backbeat Accuracy (comparison)
  - Canary vs Prod: Latency (comparison)
  - Canary vs Prod: Error Rate (comparison)

### Alert Severity

| Severity | 説明 | 対応SLA |
|----------|------|---------|
| **critical** | 即座に対応が必要（MLモデル停止等） | 15分以内 |
| **warning** | 監視が必要（KPI劣化等） | 1時間以内 |
| **info** | 情報提供（Auto-Recovery発火等） | 24時間以内 |

### Alert Recipients

```yaml
# monitoring/prometheus/alerts/drums_kpi_alerts.yml
alerting:
  slack:
    - channel: "#alerts-drums"
      severity: [critical, warning]
    - channel: "#alerts-info"
      severity: [info]
  
  pagerduty:
    - integration_key: "drums_critical"
      severity: [critical]
  
  email:
    - to: "drums-team@company.com"
      severity: [critical, warning]
```

---

## Canary Deployment

### 4-Week Rollout Schedule

**Week 1: Shadow Deployment (5% logging)**
- **目的**: KPIベースライン収集、ML推論検証
- **Traffic Split**: Shadow 5%, Production 95%
- **Feature Flags**: ML inference enabled (logging only)
- **Success Criteria**:
  - Shadow KPI ≥ Production KPI
  - ML usage ≥ 85%
  - No critical errors

**Week 2: Canary 5% (serving)**
- **目的**: 小規模トラフィックでML推論提供開始
- **Traffic Split**: Shadow 5%, Canary 5%, Production 90%
- **Feature Flags**: ML inference enabled (serving), Auto-Recovery enabled
- **Success Criteria**:
  - Canary KPI ≥ Production KPI
  - Latency p95 < 100ms
  - Error rate < 1%
  - ML usage ≥ 90%

**Week 3: Canary 20% (serving)**
- **目的**: トラフィック増加、統計的有意性検証
- **Traffic Split**: Canary 20%, Production 80%
- **Feature Flags**: All enabled
- **Success Criteria**:
  - Canary KPI ≥ Production KPI
  - Statistical significance (1000+ samples, 95% confidence)
  - Latency p95 < 100ms
  - Error rate < 1%

**Week 4: Production 100% (full rollout)**
- **目的**: 完全ロールアウト、全v3機能有効化
- **Traffic Split**: Production 100%
- **Feature Flags**: All enabled permanently
- **Success Criteria**:
  - All KPIs maintained
  - No rollback events for 7 days

### Rollout Criteria

**KPI Thresholds**:
```yaml
rollout_criteria:
  kpi_thresholds:
    kick_downbeat_rate_min: 0.80
    snare_backbeat_acc_min: 0.85
    hat_density_abs_max: 2.0
    fill_placement_valid_min: 0.95
    ml_used_min: 0.90
```

**Statistical Validation**:
```yaml
statistical:
  min_samples: 1000           # 最小サンプル数
  confidence_level: 0.95      # 信頼水準
  max_p_value: 0.05           # 最大p値（有意性検定）
```

**Performance Limits**:
```yaml
performance:
  max_p95_latency_ms: 100     # p95レイテンシ上限
  max_error_rate: 0.01        # エラー率上限（1%）
  min_cache_hit_rate: 0.80    # キャッシュヒット率下限
```

### Rollout Commands

**Week 1 → Week 2 (Shadow → Canary 5%)**:
```bash
# Canary展開設定を確認
cat config/canary_drums.yaml

# Canary有効化（5%）
kubectl apply -f k8s/canary_drums_5pct.yaml

# トラフィック分配確認
kubectl get virtualservice drums-service -o yaml

# メトリクス確認（Canary vs Prod）
curl -s 'http://prometheus:9090/api/v1/query?query=drums_kick_downbeat_rate{deployment="canary"}'
curl -s 'http://prometheus:9090/api/v1/query?query=drums_kick_downbeat_rate{deployment="prod"}'
```

**Week 2 → Week 3 (Canary 5% → 20%)**:
```bash
# Canary比較（Week 2の結果）
python scripts/compare_canary_prod.py --metric kick_downbeat_rate --min_samples 1000

# 統計的有意性検証
python scripts/statistical_test.py --canary_data canary_week2.csv --prod_data prod_week2.csv

# Canary増加（20%）
kubectl apply -f k8s/canary_drums_20pct.yaml
```

**Week 3 → Week 4 (Canary 20% → Production 100%)**:
```bash
# Final validation（Week 3の結果）
python scripts/final_validation.py --kpi_gates config/gate_prod.yaml --canary_data canary_week3.csv

# 完全ロールアウト
kubectl apply -f k8s/drums_production_100pct.yaml

# Feature Flags永続化
python scripts/enable_feature_flags.py --flags use_ml_inference,auto_recovery,kpi_gates_strict

# 監視継続（7日間）
python scripts/monitor_production.py --duration_days 7
```

---

## Troubleshooting

### Issue 1: DrumsKickDownbeatRateLow

**症状**:
```
Alert: DrumsKickDownbeatRateLow
Severity: warning
Expr: drums_kick_downbeat_rate < 0.80
Current Value: 0.72
```

**原因**:
- MLモデル性能劣化
- パターン抽出ルールの問題
- 入力データ（コード進行）の異常

**対応手順**:
1. **KPI詳細確認**:
   ```bash
   python scripts/analyze_kpi.py --kpi kick_downbeat_rate --window_hours 24
   ```

2. **MLモデル確率分布確認**:
   ```python
   from stage2.drum_pattern_recommender import DrumPatternRecommender
   recommender = DrumPatternRecommender.load("data/patterns/drums_v3_ml_final.pickle")
   recommender.analyze_proba_distribution()
   # → Top-1確率が低い場合、モデル再トレーニング検討
   ```

3. **Safe-Kit発火率確認**:
   ```bash
   curl -s 'http://prometheus:9090/api/v1/query?query=rate(drums_safe_kit_fallback_total[5m])'
   # → 10%超過の場合、MLモデル再トレーニング推奨
   ```

4. **一時的対策（Auto-Recovery無効化）**:
   ```yaml
   # config/gate_prod.yaml
   drums:
     auto_recovery:
       enabled: false  # 一時的に無効化
   ```

5. **根本対策（モデル再トレーニング）**:
   ```bash
   # Phase 25.2のトレーニングスクリプト再実行
   python training/train_drums_ml.py --config training/drums_config.yaml
   ```

---

### Issue 2: DrumsAutoRecoveryFrequent

**症状**:
```
Alert: DrumsAutoRecoveryFrequent
Severity: warning
Expr: rate(drums_auto_recovery_triggered_total[30m]) > 0.05
Current Value: 0.12 events/sec
```

**原因**:
- KPI違反が頻発（MLモデル性能問題）
- 閾値設定が厳しすぎる
- 入力データ品質低下

**対応手順**:
1. **Recovery頻度分析**:
   ```bash
   python scripts/analyze_recovery_events.py --window_hours 24
   # → どのKPIで違反が多いか特定
   ```

2. **KPI閾値見直し**:
   ```yaml
   # config/gate_prod.yaml（一時的に緩和）
   drums:
     kpi_gates:
       kick_downbeat_rate_min: 0.75  # 0.80 → 0.75に緩和
   ```

3. **MLモデル再評価**:
   ```python
   from training.evaluate_drums_ml import evaluate_model
   results = evaluate_model("data/patterns/drums_v3_ml_final.pickle", test_data)
   print(results['kpi_summary'])
   # → KPI達成率が低い場合、モデル再トレーニング
   ```

4. **データ品質確認**:
   ```bash
   python scripts/check_input_quality.py --source production --window_hours 24
   # → 異常なコード進行やBPMが多い場合、データフィルタリング検討
   ```

---

### Issue 3: DrumsRecommendLatencyHigh

**症状**:
```
Alert: DrumsRecommendLatencyHigh
Severity: warning
Expr: histogram_quantile(0.95, rate(drums_recommend_duration_seconds_bucket[5m])) > 0.100
Current Value: 0.145 seconds
```

**原因**:
- キャッシュヒット率低下
- MLモデル推論時間増加
- データベースクエリ遅延

**対応手順**:
1. **キャッシュヒット率確認**:
   ```bash
   curl -s 'http://prometheus:9090/api/v1/query?query=rate(drums_pattern_cache_hits_total[5m]) / rate(drums_pattern_cache_requests_total[5m])'
   # → 80%未満の場合、キャッシュ戦略見直し
   ```

2. **ML推論時間プロファイリング**:
   ```python
   from stage2.drum_pattern_recommender import DrumPatternRecommender
   import time
   
   recommender = DrumPatternRecommender.load("data/patterns/drums_v3_ml_final.pickle")
   
   start = time.time()
   recommender.recommend(...)
   duration = time.time() - start
   print(f"Inference duration: {duration:.3f}s")
   # → 0.1s超過の場合、モデル軽量化検討
   ```

3. **キャッシュサイズ増加**:
   ```yaml
   # config/cache_config.yaml
   drums_pattern_cache:
     max_size: 10000  # 5000 → 10000に増加
     ttl_seconds: 3600
   ```

4. **モデル軽量化**:
   ```bash
   # XGBoostのmax_depth削減、n_estimators削減
   python training/train_drums_ml.py --config training/drums_config_lightweight.yaml
   ```

---

### Issue 4: DrumsMLModelUnavailable

**症状**:
```
Alert: DrumsMLModelUnavailable
Severity: critical
Expr: drums_ml_model_loaded == 0
Current Value: 0
```

**原因**:
- モデルファイルが見つからない
- モデルファイル破損
- デプロイメント設定ミス

**対応手順**:
1. **モデルファイル確認**:
   ```bash
   ls -lh data/patterns/drums_v3_ml_final.pickle
   # → ファイルが存在しない場合、バックアップから復元
   ```

2. **モデルロードテスト**:
   ```python
   from stage2.drum_pattern_recommender import DrumPatternRecommender
   try:
       recommender = DrumPatternRecommender.load("data/patterns/drums_v3_ml_final.pickle")
       print("Model loaded successfully")
   except Exception as e:
       print(f"Model load failed: {e}")
   ```

3. **バックアップから復元**:
   ```bash
   # S3バックアップから復元
   aws s3 cp s3://company-models/drums_v3_ml_final.pickle data/patterns/
   
   # Podを再起動してモデル再ロード
   kubectl rollout restart deployment drums-service
   ```

4. **Safe-Kit発火確認**:
   ```bash
   curl -s 'http://prometheus:9090/api/v1/query?query=rate(drums_safe_kit_fallback_total[5m])'
   # → Safe-Kitが100%発火している場合、緊急対応完了（モデル復元を優先）
   ```

---

## Rollback Procedures

### Auto-Rollback Triggers

Canary展開時、以下の条件で自動ロールバック:

1. **Critical KPI Failure**:
   ```yaml
   condition: "canary_kick_downbeat_rate < 0.75"
   description: "Kick Downbeat Rate が臨界値を下回った"
   ```

2. **High Error Rate**:
   ```yaml
   condition: "canary_error_rate > 0.05"
   description: "エラー率が5%を超過した"
   ```

3. **Significant Degradation**:
   ```yaml
   condition: "(canary_kick_downbeat_rate - prod_kick_downbeat_rate) < -0.10"
   description: "CanaryがProdより10%以上劣化した"
   ```

4. **Unacceptable Latency**:
   ```yaml
   condition: "canary_p95_latency_ms > 200"
   description: "p95レイテンシが200msを超過した"
   ```

### Manual Rollback (Canary → Production)

**Step 1: ロールバック決定**
```bash
# Canary vs Prod比較（直近24時間）
python scripts/compare_canary_prod.py --window_hours 24

# ロールバック理由を確認
# → KPI劣化、エラー増加、レイテンシ増加等
```

**Step 2: トラフィック停止**
```bash
# Canaryトラフィックを0%に削減
kubectl apply -f k8s/canary_drums_0pct.yaml

# トラフィック分配確認
kubectl get virtualservice drums-service -o yaml
# → Canary: 0%, Production: 100%を確認
```

**Step 3: Feature Flags無効化**
```bash
# Canary用Feature Flagsを無効化
python scripts/disable_feature_flags.py --flags use_ml_inference,auto_recovery --deployment canary

# 設定確認
python scripts/check_feature_flags.py --deployment canary
# → すべてfalseを確認
```

**Step 4: ロールバック後の検証**
```bash
# KPI正常化を確認（30分間）
python scripts/monitor_kpi.py --duration_minutes 30 --kpis kick_downbeat_rate,snare_backbeat_acc

# エラー率確認
curl -s 'http://prometheus:9090/api/v1/query?query=rate(drums_errors_total[5m])'
# → 1%未満を確認
```

**Step 5: ポストモーテム**
```bash
# ロールバックレポート作成
python scripts/generate_rollback_report.py --canary_data canary_failure.csv --output rollback_postmortem.md

# Issue登録（再発防止）
gh issue create --title "Canary Rollback: [reason]" --body-file rollback_postmortem.md
```

### Emergency Rollback (Production → Safe-Kit)

**緊急事態（MLモデル完全停止等）の場合**:

**Step 1: Safe-Kit強制発火**
```yaml
# config/gate_prod.yaml
drums:
  safety:
    min_proba: 1.00  # 実質的にMLを完全無効化（Safe-Kit 100%発火）
```

**Step 2: 設定反映**
```bash
# ConfigMapを更新
kubectl create configmap drums-config --from-file=config/gate_prod.yaml --dry-run=client -o yaml | kubectl apply -f -

# Podを再起動して設定反映
kubectl rollout restart deployment drums-service
```

**Step 3: Safe-Kit発火確認**
```bash
# Safe-Kit発火率確認（100%になることを確認）
curl -s 'http://prometheus:9090/api/v1/query?query=rate(drums_safe_kit_fallback_total[5m])'

# KPI安定化確認
python scripts/monitor_kpi.py --duration_minutes 15 --kpis kick_downbeat_rate,snare_backbeat_acc
```

**Step 4: 根本対応**
```bash
# MLモデル再トレーニング
python training/train_drums_ml.py --config training/drums_config.yaml

# モデル検証
python training/evaluate_drums_ml.py --model data/patterns/drums_v3_ml_retrained.pickle

# モデル再デプロイ
kubectl apply -f k8s/drums_ml_model_update.yaml
```

---

## Summary

**Phase 25.3完了**: Rhythm AI (Drums) 本番展開準備完了

### 実装内容

1. **Prometheus/Grafana監視**: 14種類のアラートルール（KPI violations, ML usage, performance, auto-recovery, critical, quality, canary）
2. **Canary展開設定**: 4週間段階的ロールアウト（Shadow→Canary→Production）、Auto-Rollback保護
3. **Auto-Recovery有効化**: KPI違反時の自動復旧機能（64バーウィンドウ、10回許容、16バークールダウン）
4. **運用ドキュメント**: 使用方法、KPI解説、トラブルシューティング、Canary展開手順、ロールバック手順

### Next Steps

1. **Canary展開開始**: Week 1 Shadow deployment（5% logging）
2. **KPI監視**: Grafanaダッシュボードで継続監視
3. **Phase 26検討**: 他の楽器へのv3展開（Guitar/Bass/Piano ML化）、またはStrings/Vocalsの強化

### Phase 25全体成果

- **Phase 25.0-25.3**: 13タスク、4,250行実装完了
- **Rhythm AI**: ML-Driven Pattern Recommendation、5-KPI Quality Gates、Auto-Recovery、Canary Deployment
- **本番展開準備**: 完了（監視・アラート・ロールアウト計画・運用ガイド整備済み）

---

**作成日**: 2025-01-XX  
**作成者**: Copilot  
**Phase**: 25.3 (運用統合)  
**Status**: 本番展開Ready ✅
