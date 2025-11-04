# Phase 23: Production Migration Procedures

**目的**: v3ギター生成システムを本番環境へ段階的に展開  
**Date**: 2025-01-XX  
**Status**: 🔧 READY FOR MIGRATION

---

## 事前準備チェックリスト

### 1. Configuration Freeze（設定凍結）

```bash
# 1. SHA256SUM生成（改変検知）
sha256sum data/patterns/stage2_guitar_v3_fixed.pickle > SHA256SUMS
sha256sum data/patterns/stage2_guitar.pickle >> SHA256SUMS
sha256sum monitoring/gate_prod.yaml >> SHA256SUMS
sha256sum config/controls.yaml >> SHA256SUMS

# 2. Git commit & tag
git add SHA256SUMS
git commit -m "Phase23: Production configuration freeze

- v3 pattern: $(sha256sum data/patterns/stage2_guitar_v3_fixed.pickle | cut -d' ' -f1)
- v1 pattern: $(sha256sum data/patterns/stage2_guitar.pickle | cut -d' ' -f1)
- gate_prod.yaml: $(sha256sum monitoring/gate_prod.yaml | cut -d' ' -f1)
- controls.yaml: $(sha256sum config/controls.yaml | cut -d' ' -f1)

Commit SHA: $(git rev-parse HEAD)"

git tag -a v3-guitar-prod-candidate -m "Phase 23 production candidate

Features:
- Chord Fit v3.1 (continuous value scoring)
- Auto-Recovery v2 (ratio-based judgment)
- Safety valve with margin criteria
- Distribution monitoring (p10/p50/p90)
- Time signature normalization

Gate Configuration:
- Auto-Recovery: 64/10/16 (window/threshold/cooldown)
- Fallback ratio: 20%, Recovery ratio: 5%
- Safety: min_proba=0.15, min_margin=0.08

Metrics:
- Accent p10/p50/p90: 0.50/0.73/0.90
- Chord Fit p10/p50/p90: 0.50/0.73/0.90
- Auto-Recovery false positive: <1%
"

git push origin main --tags
```

**合格基準**:
- ✅ SHA256SUMSに4ファイルの署名が記録されている
- ✅ Git tagがリモートにpushされている
- ✅ `git log`に設定凍結commitが残っている

---

### 2. Monitoring Infrastructure Deployment

#### 2-1. Prometheus Rules

```bash
# Recording rules配置
cp monitoring/prometheus/rules.d/guitar_drift.rules.yml /etc/prometheus/rules.d/
cp monitoring/prometheus/alerts/guitar_drift.alerts.yml /etc/prometheus/alerts/

# prometheus.yml編集
cat >> /etc/prometheus/prometheus.yml <<EOF

# Guitar v3 monitoring
rule_files:
  - "rules.d/guitar_drift.rules.yml"
  - "alerts/guitar_drift.alerts.yml"

# Scrape configs（既存のjob_nameに追加）
scrape_configs:
  - job_name: 'guitar-v3'
    static_configs:
      - targets: ['localhost:8000']  # FastAPI metrics endpoint
EOF

# Prometheus reload
curl -X POST http://localhost:9090/-/reload

# ルール確認
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name | contains("guitar"))'
```

**合格基準**:
- ✅ `guitar_drift_weekly` ルールグループが表示される
- ✅ `guitar_v3_accent_p10_7d` などの12個の recording rules が active
- ✅ 8個の alert rules が active（pending or firing）

#### 2-2. Grafana Dashboard

```bash
# Dashboard JSON配置（TODO: ダッシュボード作成後）
# cp monitoring/grafana/dashboards/guitar_drift.json /var/lib/grafana/dashboards/

# Grafanaでインポート確認
# http://localhost:3000/dashboard/import
```

**合格基準**:
- ✅ 4パネル表示: Accent分布, Chord Fit分布, Latency, Section別p10
- ✅ Drift ratio グラフで過去7日のトレンド表示

---

### 3. Pre-Migration Validation Tests

#### 3-1. Auto-Recovery Real-World Test

```bash
# 100曲テスト実行
python scripts/test_auto_recovery_real_world.py \
    --window 64 --breach 10 --cooldown 16 \
    --fallback-ratio 0.20 --recover-ratio 0.05 \
    --num-songs 100

# 合格基準チェック
# - Fallback rate ≤ 1%
# - Cooldown violations = 0
# - No invalid switches
```

**合格基準**:
- ✅ `総合判定: PASS` が出力される
- ✅ `data/auto_recovery_real_world.csv` にスイッチログが記録される
- ✅ フォールバック率 < 1%

#### 3-2. Chord Fit v3.1 Distribution Test

```bash
# 100曲テスト実行
python scripts/test_shadow_traffic_100songs.py --num-songs 100

# 分布確認
grep "chord_fit" data/shadow_traffic_100songs.csv | \
  python -c "
import sys, numpy as np
scores = [float(line.split(',')[5]) for line in sys.stdin]
print(f'p10={np.percentile(scores, 10):.3f}')
print(f'p50={np.percentile(scores, 50):.3f}')
print(f'p90={np.percentile(scores, 90):.3f}')
print(f'std={np.std(scores):.3f}')
"
```

**合格基準**:
- ✅ p10: 0.45-0.55（低品質パターンを識別）
- ✅ p50: 0.70-0.75（典型的なパターン）
- ✅ p90: 0.85-0.95（高品質パターン）
- ✅ std: ~0.15（弁別力あり）

#### 3-3. Safety Threshold Test（TODO: 実装後）

```bash
# 低確率注入テスト
FORCE_LOW_PROBA=1 python scripts/test_safety_threshold.py \
    --num-songs 20 --output data/safety_probe.csv

# 合格基準チェック
# - Safe fallback rate ≈100%
# - Chord Fit failures = 0
# - ログに safety_trigger=1, reason={low_p1|low_margin}
```

**合格基準**:
- ✅ 総合判定: PASS
- ✅ Chord Fit < 0.4 のケースがゼロ

---

## Canary Deployment（段階展開）

### Phase 1: 10% Traffic (24h monitoring)

```yaml
# monitoring/gate_prod.yaml編集
traffic:
  v3_ratio: 0.10  # 10%のみv3

auto_recovery:
  enabled: true
  threshold: 10  # 厳しめ（100曲中10曲失敗で退避）
```

```bash
# サービス再起動
systemctl restart composer-api

# ログ監視
tail -f /var/log/composer/guitar_v3.log | grep -E "route_and_compare|auto_recovery"

# Prometheusクエリ（24h経過後）
guitar_v3_accent_p10_24h  # 期待値: >0.50
guitar_v3_chord_p10_24h   # 期待値: >0.50
rate(auto_recovery_switches_v3_to_v1_total[24h])  # 期待値: <0.01 (1% fallback)
```

**合格基準**:
- ✅ 24時間でAuto-Recoveryフォールバック < 1%
- ✅ Accent p10 > 0.50（Drift alertが発火しない）
- ✅ Chord Fit p10 > 0.50
- ✅ ユーザーからの音楽的破綻報告ゼロ

**異常時対応**:
```bash
# 即座にロールバック
sed -i 's/v3_ratio: 0.10/v3_ratio: 0.00/' monitoring/gate_prod.yaml
systemctl restart composer-api

# 原因調査
grep "version=v1" /var/log/composer/guitar_v3.log | tail -20
# → pattern_id, chord_root, section を確認
# → learning/meta feedbackへフィードバック
```

---

### Phase 2: 30% Traffic (24h monitoring)

```yaml
# 10%が安定したら拡大
traffic:
  v3_ratio: 0.30  # 30%へ増加
```

**合格基準**（Phase 1と同じ）:
- ✅ Fallback rate < 1%
- ✅ p10 > 0.50（両方）
- ✅ Drift ratio > 0.90（警告なし）

---

### Phase 3: 70% Traffic (24h monitoring)

```yaml
traffic:
  v3_ratio: 0.70  # 70%へ増加
```

**合格基準**（Phase 1と同じ）

---

### Phase 4: 100% Rollout

```yaml
traffic:
  v3_ratio: 1.00  # 全量展開

auto_recovery:
  threshold: 10  # 本番値（100曲中10曲で退避）
  cooldown: 16   # 安定運用
```

```bash
# 最終タグ付け
git tag -a v3-guitar-prod-stable -m "Phase 23 full rollout complete

Deployment history:
- 10% → 30% → 70% → 100% (each step 24h monitoring)
- Auto-Recovery false positive rate: <1%
- Accent/Chord p10 maintained: >0.50
- No user-reported quality issues

Production metrics (7-day average):
- Accent p10/p50/p90: 0.52/0.74/0.91
- Chord Fit p10/p50/p90: 0.51/0.73/0.89
- Latency p99: <50ms
"

git push origin --tags
```

**合格基準**:
- ✅ 7日間連続でDrift alert発火ゼロ
- ✅ Auto-Recovery fallback rate < 1%
- ✅ ユーザー満足度調査でv1比較「同等以上」

---

## SRE Runbook（アラート対応手順）

### Alert: `GuitarAccentP10DriftWarning`

**トリガー条件**: `guitar_v3_accent_drift_ratio < 0.90` for 6h

**対応手順**:

1. **状況確認**:
   ```bash
   # 現在のp10値
   curl -s 'http://localhost:9090/api/v1/query?query=guitar_v3_accent_p10_24h' | jq '.data.result[0].value[1]'
   
   # 7日間ベースライン
   curl -s 'http://localhost:9090/api/v1/query?query=guitar_v3_accent_p10_7d' | jq '.data.result[0].value[1]'
   
   # Drift ratio
   curl -s 'http://localhost:9090/api/v1/query?query=guitar_v3_accent_drift_ratio' | jq '.data.result[0].value[1]'
   ```

2. **CSV抽出（低スコア曲の分析）**:
   ```bash
   # 過去24時間の低Accentスコア曲
   grep "accent" /var/log/composer/guitar_v3.csv | \
     awk -F',' '$6 < 0.50 {print $0}' | \
     sort -t',' -k6 -n | head -20 > /tmp/low_accent.csv
   
   # パターン分析
   cut -d',' -f3,4,5 /tmp/low_accent.csv | sort | uniq -c | sort -rn
   # → 特定のpattern_id, chord, sectionが集中していないか確認
   ```

3. **一時的対応**（drift ratio < 0.85の場合）:
   ```yaml
   # gate_prod.yaml一時調整
   auto_recovery:
     threshold: 8  # 10 → 8へ厳格化（100曲中8曲で退避）
   
   # または
   traffic:
     v3_ratio: 0.70  # 100% → 70%へ一時縮小
   ```

4. **根本原因分析**:
   - パターンデータの劣化？ → SHA256SUMS検証
   - 入力データの分布変化？ → chord_root, tempo, section分布を週次比較
   - モデルの過学習？ → learning/meta feedbackで再学習検討

---

### Alert: `GuitarAutoRecoveryFallbackStorm`

**トリガー条件**: `rate(auto_recovery_switches_v3_to_v1_total[1h]) > 3`

**対応手順**:

1. **スイッチログ確認**:
   ```bash
   grep "auto_recovery_switch" /var/log/composer/guitar_v3.log | tail -50
   # → breach_count, breach_ratio, trigger_reason を確認
   ```

2. **一時的対応**:
   ```yaml
   # gate_prod.yaml
   auto_recovery:
     enabled: false  # Auto-Recoveryを一時停止
   
   traffic:
     v3_ratio: 0.00  # v1へ完全退避
   ```

3. **原因調査後の復旧**:
   ```bash
   # パターンデータ整合性チェック
   sha256sum -c SHA256SUMS
   
   # OK → Auto-Recovery再有効化
   # NG → パターンファイル再配置 & git SHA確認
   ```

---

## Rollback Procedures（緊急ロールバック）

### Scenario 1: Critical quality degradation

```bash
# 1. v3を完全停止
sed -i 's/v3_ratio: [0-9.]\+/v3_ratio: 0.00/' monitoring/gate_prod.yaml
systemctl restart composer-api

# 2. Auto-Recovery無効化（v1固定）
sed -i 's/enabled: true/enabled: false/' monitoring/gate_prod.yaml

# 3. アラートサイレンス（24h）
curl -X POST http://localhost:9093/api/v1/silences \
  -d '{
    "matchers": [{"name": "alertname", "value": "Guitar.*", "isRegex": true}],
    "startsAt": "2025-01-XX",
    "endsAt": "2025-01-XX",
    "createdBy": "SRE-oncall",
    "comment": "v3 rollback due to quality degradation"
  }'

# 4. Post-mortem作成
# → Why発生? What検知できた? How防止?
```

---

## Post-Deployment Monitoring（展開後の継続監視）

### Weekly Review Checklist

```bash
# 1. Drift ratio確認（週次）
curl -s 'http://localhost:9090/api/v1/query?query=guitar_v3_accent_drift_ratio' | \
  jq '.data.result[0].value[1]'
# 期待値: >0.95（5%以内の変動）

# 2. Auto-Recovery履歴（週次）
grep "auto_recovery_switch" /var/log/composer/guitar_v3.log | \
  grep "$(date -d '7 days ago' +%Y-%m-%d)" | wc -l
# 期待値: <7（週1回未満のフォールバック）

# 3. p10トレンド（月次）
# Grafanaダッシュボードで30日間のp10推移を確認
# 期待値: 緩やかな上昇 or 横ばい（学習効果 or 安定運用）
```

### Quarterly Pattern Re-evaluation

```bash
# 3ヶ月ごとにパターンデータを再評価
python scripts/test_shadow_traffic_100songs.py --num-songs 500

# 分布変化チェック
# p10が0.50 → 0.55へ上昇 → 良い兆候（学習効果）
# p10が0.50 → 0.45へ下降 → 要調査（データdrift or モデル劣化）
```

---

## Next Phase: Continuous Improvement

1. **Learning/Meta Feedback**:
   - 低スコア曲を定期収集 → 人間レビュー → パターン再学習
   - Chord Fit < 0.40のケースを100件集めて原因分析

2. **A/B Testing Framework**:
   - v3.1 vs v3.2（新しいペナルティルール）の比較
   - Shadow traffic splitterで同時評価

3. **Safety Kit Enhancement**:
   - "safe-kit"パターンの品質向上（現在は汎用キット）
   - Section別の専用safe-kitパターン作成

4. **Distribution-based Gating**:
   - gate_prod.yamlの`default_p10`セクション有効化
   - p10 < 0.50でアラート → p10 < 0.45で自動フォールバック

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-XX  
**Owner**: Phase 23 Migration Team  
**Reviewers**: SRE, Product, ML Engineering
