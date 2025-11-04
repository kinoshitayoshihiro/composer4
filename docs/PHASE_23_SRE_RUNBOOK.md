# Phase 23.5: SRE Runbook - Operational Monitoring Guide

**Version**: 1.1 (Phase 23.5強化版)  
**Last Updated**: 2025-10-27  
**Owner**: Guitar v3 Production Team

---

## 🚦 Red/Yellow/Green 基準（Phase 23.5版）

### Green（通常運用）

すべての条件を満たす:

| 指標 | 基準 | 理由 |
|------|------|------|
| **Accent p95** | ≥ 0.85 | 上位95%が高品質を維持 |
| **Chord Fit p50** | ≥ 0.70 | 中央値が70%以上（典型的パターンが良好） |
| **Safety Trigger Rate** | < 5% | 安全閾値発火が5%未満（正常範囲） |
| **Drift Ratio (7d/24h)** | ≥ 0.95 | 品質ドリフトが5%以内 |
| **Safe-Kit Fallback Rate** | < 1% | Safe-Kitへのフォールバックが1%未満 |

**アクション**: 通常監視継続。週次レビューで長期トレンド確認。

---

### Yellow（要観察）

いずれかの条件に該当:

| 指標 | 基準 | アラート名 | 理由 |
|------|------|-----------|------|
| **Accent p10 (7d/24h)** | < 0.92 | `GuitarAccentP10DriftWarning` | 下位10%が週次比で8%以上劣化 |
| **Chord Fit p10 (7d/24h)** | < 0.92 | `GuitarChordP10DriftWarning` | 下位10%のコード適合が劣化 |
| **Safety Trigger Rate** | 5-10% | `GuitarSafetyStorm` | 安全閾値発火が増加傾向 |
| **Safe-Kit Fallback Rate** | 1-3% | `SafeKitFallbackIncrease` | Safe-Kit使用が増加（異常兆候） |
| **Latency p95** | 50-100ms | `GuitarLatencyWarning` | レイテンシがやや高い |

**アクション**:

1. **Drift検知時** (`GuitarAccentP10DriftWarning`):
   ```bash
   # CSV抽出（過去24時間の低スコアパターン）
   cd /opt/composer-v3
   tail -1000 data/shadow_traffic_log.csv | \
     awk -F',' '$6 < 0.50 {print $0}' | \
     column -t -s','
   
   # セクション別集計
   tail -1000 data/shadow_traffic_log.csv | \
     awk -F',' '{sum[$9]+=$6; count[$9]++} END {for(s in sum) print s, sum[s]/count[s]}' | \
     sort -k2 -n
   ```

2. **Safety Storm時** (`GuitarSafetyStorm`):
   ```bash
   # 安全発火ログ抽出
   grep "safety_triggered=1" data/shadow_traffic_log.csv | \
     tail -100 | \
     awk -F',' '{print $9, $10, $15, $16}' | \  # section, chord, top1_proba, margin
     column -t -s','
   
   # 発火理由別集計
   grep "safety_triggered=1" data/shadow_traffic_log.csv | \
     tail -100 | \
     awk -F',' '{count[$17]++} END {for(r in count) print r, count[r]}' | \
     sort -k2 -rn
   ```

3. **Safe-Kit Fallback増加時**:
   ```bash
   # Safe-Kit使用パターン確認
   grep "SAFE_KIT_" data/shadow_traffic_log.csv | \
     tail -50 | \
     awk -F',' '{print $9, $12}' | \  # section, pattern_id
     sort | uniq -c | sort -rn
   ```

4. **観察継続**:
   - 30分ごとにGrafanaダッシュボード確認
   - Slackアラートチャンネル監視
   - 1時間継続→ **Red判定に移行**

---

### Red（緊急対応）

いずれかの条件に該当:

| 指標 | 基準 | アラート名 | 影響 |
|------|------|-----------|------|
| **Accent p10 (7d/24h)** | < 0.85 | `GuitarAccentP10Critical` | 下位10%が15%以上劣化 |
| **Chord Fit p10 (7d/24h)** | < 0.85 | `GuitarChordP10Critical` | 重大なコード不一致 |
| **Safety Trigger Rate** | > 10% (30分継続) | `GuitarSafetyStormCritical` | v3モデル品質異常 |
| **Safe-Kit Fallback Rate** | > 5% (30分継続) | `SafeKitFallbackCritical` | Safe-Kitでも吸収できない |
| **Auto-Recovery v3→v1切替** | 発生 | `AutoRecoveryFallback` | 自動退避実行済み |

**即時アクション（5分以内）**:

#### Step 1: Safe-Kit強制モード（最優先）

```bash
# gate_prod.yamlでSafe-Kit強制
cd /opt/composer-v3
vi monitoring/gate_prod.yaml

# 以下に変更:
safety:
  min_proba: 0.50  # 閾値を大幅に上げてほぼ全てSafe-Kitへ
  min_margin: 0.20
  fallback_target: "safe-kit"

# サービス再起動
systemctl restart composer-api

# 5分待機して様子見
sleep 300
tail -100 data/shadow_traffic_log.csv | grep "SAFE_KIT_"
```

**期待結果**:
- Safe-Kit使用率が急増（50%+）
- Accent Score p10が0.70以上に回復
- ユーザー体験は維持（Safe-Kitは最低品質保証）

---

#### Step 2: Safe-Kitでも改善なし→ Canary巻き戻し（30分経過後）

```bash
# Canary比率を30%に巻き戻し
vi monitoring/gate_prod.yaml

# 変更:
traffic:
  v3_ratio: 0.30  # 100% → 30% へ巻き戻し

systemctl restart composer-api

# 監視継続（10分）
watch -n 60 'curl -s http://localhost:9090/api/v1/query?query=guitar_v3_accent_p10_24h | jq'
```

**期待結果**:
- v1トラフィック70% → 品質安定化
- 事象の切り分け（v3特有の問題か、データ起因か）

---

#### Step 3: 全面v1切替（最終手段、60分経過後）

```bash
# v3を完全停止
vi monitoring/gate_prod.yaml

traffic:
  v3_ratio: 0.00  # v3完全停止

systemctl restart composer-api

# Post-mortem準備
mkdir -p /tmp/guitar-v3-incident-$(date +%Y%m%d-%H%M)
cp data/shadow_traffic_log.csv /tmp/guitar-v3-incident-$(date +%Y%m%d-%H%M)/
cp monitoring/gate_prod.yaml /tmp/guitar-v3-incident-$(date +%Y%m%d-%H%M)/
```

**Slack通知**:
```
🚨 **CRITICAL: Guitar v3 Emergency Rollback** 🚨

Reason: <Drift/Safety Storm/Safe-Kit Failure>
Actions Taken:
  1. ✅ Safe-Kit forced mode (t+5m)
  2. ✅ Canary rollback to 30% (t+30m)
  3. ✅ Full v1 fallback (t+60m)

Current Status:
  - v3 traffic: 0%
  - v1 traffic: 100%
  - Accent p10: <current_value>
  - Next Steps: Post-mortem analysis

Incident ID: guitar-v3-incident-<timestamp>
```

---

## 📊 監視クエリ（Grafana/Prometheus）

### Drift Ratio監視（7d/24h比較）

```promql
# Accent Score Drift
guitar_v3_accent_p10_24h / guitar_v3_accent_p10_7d

# Chord Fit Drift
guitar_v3_chordfit_p10_24h / guitar_v3_chordfit_p10_7d

# セクション別Drift（Chorus）
guitar_v3_accent_p10_24h_chorus / guitar_v3_accent_p10_7d_chorus
```

**解釈**:
- **0.95-1.05**: 正常（5%以内の変動）
- **0.90-0.95**: 軽度劣化（Yellow）
- **< 0.90**: 重度劣化（Red）

---

### Safety Threshold監視

```promql
# Safety Trigger Rate（5分間）
rate(guitar_v3_safety_triggered_total[5m])

# top1_proba分布（p10/p50/p90）
histogram_quantile(0.10, guitar_v3_top1_proba_bucket)
histogram_quantile(0.50, guitar_v3_top1_proba_bucket)
histogram_quantile(0.90, guitar_v3_top1_proba_bucket)

# margin分布
histogram_quantile(0.10, guitar_v3_top12_margin_bucket)
histogram_quantile(0.50, guitar_v3_top12_margin_bucket)
```

**正常範囲**:
- `top1_proba_p10` > 0.20（p10が20%以上）
- `margin_p10` > 0.10（p10マージンが10%以上）
- `safety_trigger_rate` < 0.05（5%未満）

---

### Safe-Kit Fallback監視

```promql
# Safe-Kit使用率（5分間）
rate(guitar_v3_safe_kit_invocations_total[5m])

# パターン別使用率
sum by (pattern_name) (rate(guitar_v3_safe_kit_pattern_selection_by_section[5m]))
```

**正常範囲**:
- Safe-Kit使用率 < 0.01（1%未満）
- 異常時は急増→要調査

---

## 🔧 ローリング増加時のスクリーンショット手順

### 10% → 30% 移行判定時

```bash
# 1. p10スクリーンショット取得
open "http://localhost:3000/d/guitar-v3-drift?orgId=1&from=now-7d&to=now&var-metric=accent_score"

# 2. スナップショット保存（Grafana UI）
# 右上: Share → Snapshot → Create local snapshot
# URL: http://localhost:3000/dashboard/snapshot/<snapshot_id>

# 3. メトリクス数値記録
curl -s 'http://localhost:9090/api/v1/query?query=guitar_v3_accent_drift_ratio' | \
  jq '.data.result[0].value[1]' | \
  tee -a /tmp/phase23_rollout_metrics.txt

echo "Date: $(date), Stage: 10%→30%, Drift Ratio: <value>" >> /tmp/phase23_rollout_metrics.txt
```

### 30% → 70% 移行判定時

同様の手順で再度実行。

### 70% → 100% 移行判定時

同様の手順で再度実行 + **最終Go/No-Go判定**。

---

## 📝 Post-Mortem Template（インシデント発生時）

### 基本情報

| 項目 | 内容 |
|------|------|
| **Incident ID** | `guitar-v3-<date>-<time>` |
| **発生日時** | YYYY-MM-DD HH:MM (JST) |
| **検知アラート** | `GuitarAccentP10Critical` / `GuitarSafetyStormCritical` 等 |
| **影響範囲** | v3トラフィック比率（例: 70%） |
| **復旧完了時刻** | YYYY-MM-DD HH:MM (JST) |

### タイムライン

| 時刻 | イベント | アクション | 実施者 |
|------|---------|----------|--------|
| t+0 | Alert発火 | Slack通知 | Prometheus |
| t+5 | Safe-Kit強制 | `min_proba=0.50` | SRE Team |
| t+30 | 改善なし | Canary 30% | SRE Team |
| t+60 | 全面v1切替 | `v3_ratio=0.00` | SRE Team |
| t+90 | 安定確認 | Post-mortem開始 | SRE Team |

### 根本原因

- [ ] モデル劣化（データドリフト）
- [ ] 入力データ異常（異常なコード進行等）
- [ ] インフラ問題（レイテンシ、メモリ）
- [ ] 設定ミス（gate_prod.yaml誤編集）
- [ ] その他: ___________

### 再発防止策

1. **監視強化**:
   - Section別p10を1時間ごとチェック
   - Safe-Kit使用率のアラート閾値を2%に引き下げ

2. **テスト強化**:
   - Grid Search再実行（週次）
   - Chord Fit v3.1分布検証（毎日）

3. **Runbook更新**:
   - Safe-Kit強制手順を5分→3分に短縮
   - Rollback判定基準を明確化

---

## 📅 ローンチチェックリスト（Phase 23 Go-Live）

### Day 1-2: 10%トラフィック

- [ ] Prometheus/Grafana設定完了
- [ ] `v3_ratio=0.10` 設定
- [ ] 監視開始（1時間ごとチェック）
- [ ] p10スクリーンショット取得（初回）

### Day 3-7: 10%継続監視

- [ ] 24時間異常なし確認
- [ ] Drift Ratio ≥ 0.95 維持
- [ ] Safety Trigger < 5% 維持
- [ ] **Go判定** → 30%へ移行

### Day 8-10: 30%トラフィック

- [ ] `v3_ratio=0.30` 更新
- [ ] 監視継続（30分ごと）
- [ ] p10スクリーンショット取得（2回目）
- [ ] **Go判定** → 70%へ移行

### Day 11-14: 70%トラフィック

- [ ] `v3_ratio=0.70` 更新
- [ ] 重点監視（15分ごと初日のみ）
- [ ] p10スクリーンショット取得（3回目）
- [ ] **Go判定** → 100%へ移行

### Day 15: 100%ロールアウト

- [ ] `v3_ratio=1.00` 最終更新
- [ ] 48時間監視（1時間ごと）
- [ ] 安定確認

### Day 16-21: 安定運用確認

- [ ] 7日間安定運用
- [ ] `v3-guitar-prod-stable` タグ作成
- [ ] Phase 23完了報告

---

**Document Version**: 1.1  
**Status**: 🚀 PRODUCTION READY  
**Last Updated**: 2025-10-27
