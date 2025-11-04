# Shadow Testing Design Document

**作成日**: 2025年10月27日  
**目的**: v3本番投入後、v1と並行運用してリアルタイムKPI比較・自動フォールバック  
**対象**: Guitar Stage2 v3 (ML) vs v1 (Rule-based)

---

## 📋 概要

### 目的

1. **安全な本番投入**: v3を本番トラフィックに投入しつつ、v1をシャドウ実行して比較
2. **リアルタイムKPI監視**: v3 vs v1のKPI差分を継続監視
3. **自動フォールバック**: v3がデグレした場合、自動でv1に戻す
4. **統計的検証**: 統計的有意性のある改善確認

### アーキテクチャ

```
┌──────────────────┐
│  Input Request   │
│  (Chord, Tempo,  │
│   Section, etc)  │
└────────┬─────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌──────────────────┐
│  v3 Generator   │  │  v1 Generator    │ (Shadow)
│  (ML-based)     │  │  (Rule-based)    │
└────────┬────────┘  └────────┬─────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌──────────────────┐
│  v3 Pattern     │  │  v1 Pattern      │
└────────┬────────┘  └────────┬─────────┘
         │                    │
         ├────────────────────┤
         │   KPI Comparison   │
         │   ───────────────  │
         │   - Accent Score   │
         │   - Chord Fit      │
         │   - Density        │
         │   - ML Usage       │
         │   - Latency        │
         └────────┬───────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Decision      │
         │  ────────────  │
         │  Primary: v3   │
         │  Fallback: v1  │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Prometheus    │
         │  Metrics       │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Grafana       │
         │  Dashboard     │
         └────────────────┘
```

---

## 🎯 Shadow Testing戦略

### フェーズ1: 並行実行（1週間）

**目標**: v3とv1の性能差を定量的に検証

- **トラフィック**: 100%（全リクエスト）
- **Primary**: v3（ユーザーに返す）
- **Shadow**: v1（内部比較のみ）
- **KPI収集**: 両方のパターンでKPI計算
- **判定**: KPIゲート判定（v3が目標未達なら警告）

**成功基準**:
- v3 Accent Score ≥ v1 Accent Score（統計的有意）
- v3 KPIゲート全PASS（7/7）
- v3レイテンシ < 100ms p95

### フェーズ2: 自動フォールバック（2週間）

**目標**: v3デグレ時の自動切り戻し

- **Primary**: v3
- **Shadow**: v1
- **監視間隔**: 5分
- **フォールバック条件**:
  - v3 Accent Score < v1 - 5pt（5分間継続）
  - v3 KPIゲートFAIL（Critical）
  - v3エラー率 > 1%

**フォールバックアクション**:
1. Slack Critical Alert送信
2. Primary自動切り替え: v3 → v1
3. Post-Mortem Issueオープン
4. 手動承認後、v3再投入

### フェーズ3: カナリアリリース（継続）

**目標**: 新バージョンの段階的展開

- **カナリア**: v3新バージョン（10%トラフィック）
- **Stable**: v3現行版（90%トラフィック）
- **Shadow**: v1（比較用）

---

## 📊 KPI比較メトリクス

### 主要KPI（両方で計測）

| KPI | v3目標 | v1ベースライン | 比較方法 |
|-----|--------|----------------|----------|
| **Accent Score** | ≥65% | ~60% | Δ = v3 - v1 |
| **Chord Fit** | ≥60% | ~55% | Δ = v3 - v1 |
| **Density Abs** | ≤1.0 | ~0.8 | Δ = v3 - v1 |
| **ML Usage** | ≥70% | 0% | v3のみ |
| **Top-1 Proba** | - | - | v3のみ |
| **Safety Fallback** | ≤10% | 0% | v3のみ |

### 追加メトリクス

| メトリクス | 説明 | 目標 |
|-----------|------|------|
| **Latency p50** | 中央値レイテンシ | <50ms |
| **Latency p95** | 95パーセンタイルレイテンシ | <100ms |
| **Latency p99** | 99パーセンタイルレイテンシ | <200ms |
| **Error Rate** | エラー発生率 | <0.1% |
| **v3 Win Rate** | v3がv1より良い割合 | >70% |

### Prometheusメトリクス

```prometheus
# v3メトリクス
guitar_v3_accent_score_mean
guitar_v3_chord_fit_mean
guitar_v3_ml_usage_rate
guitar_v3_latency_seconds{quantile="0.5|0.95|0.99"}
guitar_v3_error_total

# v1メトリクス（Shadow）
guitar_v1_accent_score_mean
guitar_v1_chord_fit_mean
guitar_v1_latency_seconds{quantile="0.5|0.95|0.99"}
guitar_v1_error_total

# 比較メトリクス
guitar_shadow_accent_delta  # v3 - v1
guitar_shadow_chord_delta   # v3 - v1
guitar_shadow_v3_win_rate   # v3 > v1の割合
guitar_shadow_agreement_rate # v3とv1が同じパターン選択した割合
```

---

## 🔧 実装仕様

### 1. shadow_test.py（新規作成）

**責務**: 同一入力でv3/v1を並行実行、KPI比較

```python
class ShadowTester:
    def __init__(self, v3_pickle_path, v1_pickle_path):
        self.v3_recommender = PatternRecommender(v3_pickle_path)
        self.v1_recommender = PatternRecommender(v1_pickle_path)
    
    def run_shadow_test(self, chord, tempo, section, key, chord_type):
        """v3とv1を並行実行してKPI比較"""
        
        # v3実行（Primary）
        start_v3 = time.time()
        v3_pattern = self.v3_recommender.recommend(...)
        v3_latency = time.time() - start_v3
        
        # v1実行（Shadow）
        start_v1 = time.time()
        v1_pattern = self.v1_recommender.recommend(...)
        v1_latency = time.time() - start_v1
        
        # KPI計算
        v3_kpi = self.compute_kpi(v3_pattern, ...)
        v1_kpi = self.compute_kpi(v1_pattern, ...)
        
        # 比較レポート
        comparison = {
            'v3': v3_kpi,
            'v1': v1_kpi,
            'delta': {
                'accent': v3_kpi['accent'] - v1_kpi['accent'],
                'chord_fit': v3_kpi['chord_fit'] - v1_kpi['chord_fit']
            },
            'v3_wins': v3_kpi['accent'] > v1_kpi['accent'],
            'latency': {'v3': v3_latency, 'v1': v1_latency}
        }
        
        return v3_pattern, comparison
    
    def export_prometheus_metrics(self, comparison):
        """Prometheusメトリクス出力"""
        # guitar_shadow_accent_delta
        # guitar_shadow_v3_win_rate
        # ...
```

### 2. guitar_generator_stage2.py修正

**Shadow Testing対応**:

```python
class GuitarGeneratorStage2:
    def __init__(self, ..., shadow_mode=False, shadow_v1_pickle=None):
        self.shadow_mode = shadow_mode
        if shadow_mode:
            self.shadow_tester = ShadowTester(
                v3_pickle_path=pickle_path,
                v1_pickle_path=shadow_v1_pickle
            )
    
    def generate_pattern(self, ...):
        if self.shadow_mode:
            # Shadow Testing実行
            v3_pattern, comparison = self.shadow_tester.run_shadow_test(...)
            
            # メトリクス記録
            self.shadow_tester.export_prometheus_metrics(comparison)
            
            # CSVログ出力（v3/v1両方）
            self.log_shadow_comparison(comparison)
            
            # Primary返却（v3）
            return v3_pattern
        else:
            # 通常モード（v3のみ）
            return self.v3_recommender.recommend(...)
```

### 3. Grafana Shadow Testing Dashboard

**パネル構成**（12パネル）:

1. **v3 vs v1 Accent Score** [Time Series]
   - v3: 青線
   - v1: 赤線
   - デルタ: 緑塗りつぶし

2. **v3 vs v1 Chord Fit** [Time Series]
   - 同上

3. **Accent Score Delta** [Graph]
   - Δ = v3 - v1
   - 0ライン（基準）
   - 目標: Δ > 0

4. **v3 Win Rate** [Gauge]
   - v3がv1より良い割合
   - 目標: >70%

5. **Latency Comparison** [Time Series]
   - v3 p95: 青
   - v1 p95: 赤
   - 目標: v3 < 100ms

6. **Error Rate** [Graph]
   - v3 error rate
   - v1 error rate
   - 目標: <0.1%

7. **Agreement Rate** [Gauge]
   - v3とv1が同じパターン選択した割合

8. **KPI Gate Status** [Table]
   - v3: PASS/FAIL（7項目）
   - v1: 参考値

9. **Section-wise Delta** [Heatmap]
   - Chorus/Verse/Bridge/Intro × Accent/Chord

10. **Statistical Significance** [Stat]
    - p-value（t検定）
    - 有意水準: p < 0.05

11. **Fallback Trigger Count** [Counter]
    - 自動フォールバック発動回数

12. **Shadow Test Volume** [Graph]
    - 並行実行数/分

### 4. アラートルール拡張

**monitoring/guitar_v3_shadow_alerts.yml**:

```yaml
groups:
  - name: shadow_testing
    interval: 30s
    rules:
      # v3デグレ検出
      - alert: GuitarV3Degradation
        expr: guitar_shadow_accent_delta < -5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "v3 degraded vs v1"
          description: "Accent delta: {{ $value }}pt (5pt below v1)"
          runbook: "https://wiki/shadow-testing#degradation"
      
      # v3勝率低下
      - alert: GuitarV3LowWinRate
        expr: guitar_shadow_v3_win_rate < 0.60
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "v3 win rate below 60%"
      
      # レイテンシ劣化
      - alert: GuitarV3HighLatency
        expr: |
          histogram_quantile(0.95, 
            rate(guitar_v3_latency_seconds_bucket[5m])
          ) > 0.100
        for: 5m
        labels:
          severity: warning
      
      # フォールバック頻発
      - alert: GuitarV3FrequentFallback
        expr: |
          increase(guitar_shadow_fallback_total[1h]) > 3
        for: 5m
        labels:
          severity: critical
```

---

## 🔄 自動フォールバック仕様

### フォールバック条件（OR条件）

1. **KPIデグレ**: v3 Accent < v1 - 5pt（5分間継続）
2. **KPIゲートFAIL**: v3 Accent < 65%（Critical、5分間）
3. **エラー率**: v3 error rate > 1%（5分間）
4. **レイテンシ**: v3 p95 > 200ms（5分間）

### フォールバックアクション

```python
class AutoFallback:
    def check_conditions(self):
        """フォールバック条件チェック"""
        if self.is_degraded():
            self.trigger_fallback()
    
    def is_degraded(self):
        """デグレ判定"""
        # Prometheusからメトリクス取得
        v3_accent = get_metric('guitar_v3_accent_score_mean')
        v1_accent = get_metric('guitar_v1_accent_score_mean')
        delta = v3_accent - v1_accent
        
        if delta < -5:  # 5pt以上劣化
            return True
        
        if v3_accent < 0.65:  # KPIゲートFAIL
            return True
        
        return False
    
    def trigger_fallback(self):
        """フォールバック実行"""
        logger.critical("🔴 Auto Fallback Triggered!")
        
        # 1. Slack通知
        send_slack_alert(
            "🔴 CRITICAL: Guitar v3 Auto Fallback\n"
            f"Reason: Accent delta < -5pt\n"
            f"Action: Switching to v1"
        )
        
        # 2. 設定ファイル更新（v3 → v1）
        update_config({
            'primary_version': 'v1',
            'shadow_version': 'v3',
            'fallback_reason': 'accent_degradation',
            'fallback_timestamp': datetime.now().isoformat()
        })
        
        # 3. Prometheusメトリクス記録
        fallback_counter.inc()
        
        # 4. Post-Mortem Issue作成（GitHub API）
        create_github_issue(
            title=f"[Post-Mortem] v3 Auto Fallback at {datetime.now()}",
            body=self.generate_post_mortem()
        )
        
        # 5. プロセス再起動（graceful restart）
        os.kill(os.getpid(), signal.SIGHUP)
```

### Post-Mortem自動生成

```markdown
# Post-Mortem: v3 Auto Fallback

**発生日時**: 2025-10-27 14:30:00 JST
**影響範囲**: Guitar Stage2 v3
**対応**: v3 → v1 自動フォールバック

## Timeline

- 14:25:00 - v3 Accent Score低下開始（90% → 62%）
- 14:27:30 - アラート発火（GuitarV3AccentScoreCritical）
- 14:30:00 - 5分間継続、自動フォールバック実行
- 14:30:15 - v1に切り替え完了
- 14:35:00 - KPI正常化確認（Accent 89%）

## Root Cause

[自動分析結果]
- セクション別分析: Chorus -12pt, Verse -3pt
- 入力分布変化: Minor Chord増加（35% → 55%）
- MLモデル: Minor Chordパターン不足の可能性

## Action Items

- [ ] Minor Chord学習データ追加
- [ ] v3パターンDB拡張
- [ ] テストケース追加（Minor Chord重点）
```

---

## 📈 統計的検証

### A/Bテスト統計

**t検定（対応のある）**:
```python
from scipy import stats

def statistical_test(v3_scores, v1_scores):
    """統計的有意性検定"""
    t_stat, p_value = stats.ttest_rel(v3_scores, v1_scores)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': cohens_d(v3_scores, v1_scores),
        'conclusion': (
            'v3 significantly better than v1' if p_value < 0.05 and t_stat > 0
            else 'No significant difference'
        )
    }
```

**必要サンプルサイズ**:
- 検出力 80%
- 有意水準 5%
- 効果量 5pt（Accent Score）
- → **n ≈ 200ケース**

---

## 🔐 安全対策

### 1. レート制限

```python
# フォールバック頻度制限（1時間に最大3回）
if fallback_count_last_hour() >= 3:
    logger.warning("Fallback rate limit exceeded")
    send_slack_alert("⚠️  Fallback disabled (rate limit)")
    return  # フォールバック無効化
```

### 2. 手動オーバーライド

```bash
# 緊急時の手動フォールバック無効化
export DISABLE_AUTO_FALLBACK=true
```

### 3. Dry-Run モード

```python
# Dry-Run: フォールバック判定のみ、実行しない
AUTO_FALLBACK_DRY_RUN=true

if should_fallback() and not dry_run:
    trigger_fallback()
else:
    logger.info("Would trigger fallback (dry-run)")
```

---

## 📊 期待結果

### フェーズ1完了時（1週間後）

- **Shadow Test Volume**: 10,000+ cases
- **v3 Win Rate**: >70%
- **Statistical Significance**: p < 0.05
- **v3 KPI Gate**: 100% PASS rate
- **Latency p95**: <80ms

### フェーズ2完了時（3週間後）

- **Auto Fallback**: 0回（安定稼働）
- **v3 Uptime**: >99.9%
- **KPI Delta**: Accent +10pt, Chord +8pt

---

## 🛠️ 実装ファイル一覧

### 新規作成

- [ ] `scripts/shadow_test.py` - Shadow Testing実行スクリプト
- [ ] `generator/shadow_tester.py` - ShadowTesterクラス
- [ ] `monitoring/auto_fallback.py` - 自動フォールバック
- [ ] `monitoring/grafana_shadow_dashboard.json` - Grafanaダッシュボード
- [ ] `monitoring/guitar_v3_shadow_alerts.yml` - アラートルール
- [ ] `scripts/shadow_cron.sh` - cron用自動実行

### 修正

- [ ] `generator/guitar_generator_stage2.py` - Shadow対応
- [ ] `monitoring/kpi_collector.py` - v1メトリクス追加
- [ ] `monitoring/docker-compose.yml` - ボリューム追加

---

## 📅 実装スケジュール

### Week 1: 基盤実装

- Day 1-2: `shadow_test.py`, `shadow_tester.py`実装
- Day 3-4: Grafanaダッシュボード作成
- Day 5-7: テスト実行（100ケース）

### Week 2: 自動化

- Day 8-10: `auto_fallback.py`実装
- Day 11-12: アラートルール設定
- Day 13-14: フォールバックテスト（意図的デグレ）

### Week 3: 本番投入

- Day 15-16: Shadow Testing開始（10%トラフィック）
- Day 17-18: 50%トラフィック
- Day 19-21: 100%トラフィック、監視

---

## ✅ 完了基準

### Shadow Testing成功判定

- [ ] v3 Win Rate ≥70%（統計的有意）
- [ ] v3 KPI Gate 100% PASS（1週間）
- [ ] Auto Fallback 0回（安定稼働）
- [ ] Latency p95 <100ms
- [ ] Error Rate <0.1%

### v1完全廃止判定

- [ ] Shadow Testing 3週間成功
- [ ] v3 Accent Score ≥ v1 + 8pt（継続）
- [ ] Post-Mortemなし（デグレゼロ）
- [ ] ステークホルダー承認

---

**Next Action**: `scripts/shadow_test.py`実装開始
