# Guitar v3 Latency Monitoring

**作成日**: 2025年10月27日  
**目的**: Guitar v3の推論時間（レイテンシ）をリアルタイム監視

---

## 📊 遅延監視メトリクス

### 目標値

| メトリクス | 目標値 | 警告 | Critical |
|-----------|--------|------|----------|
| **Latency p50** | <50ms | 50-80ms | >80ms |
| **Latency p95** | <100ms | 100-150ms | >150ms |
| **Latency p99** | <200ms | 200-300ms | >300ms |
| **Latency max** | <500ms | 500-1000ms | >1000ms |

### 実装方法

#### 1. CSVログに推論時間追加

**新フォーマット**（latency_ms追加）:
```csv
song_id,section,chord_root,chord_quality,tempo,pattern_id,accent_score,density_abs,chord_fit,ml_used,top1_proba,phase_slots,latency_ms
c9d4543648229fbe,Intro,C,maj,120.0,unknown,0.957,0.0,1.0,1,0.516,0,45.2
```

**推論時間計測**（pattern_recommender.py）:
```python
import time

def recommend(self, query, ...):
    start_time = time.time()
    
    # ... 推薦処理 ...
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        'pattern': pattern,
        'latency_ms': latency_ms,
        ...
    }
```

#### 2. kpi_collector.pyに遅延統計追加

```python
class KPICollector:
    def compute_statistics(self) -> Dict:
        latencies = [m.latency_ms for m in self.metrics if hasattr(m, 'latency_ms')]
        
        stats = {
            'latency': {
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'max': np.max(latencies),
                'mean': np.mean(latencies)
            }
        }
```

#### 3. Prometheusメトリクス

```prometheus
# HELP guitar_v3_latency_seconds Guitar v3 inference latency
# TYPE guitar_v3_latency_seconds summary
guitar_v3_latency_seconds{quantile="0.5"} 0.0452
guitar_v3_latency_seconds{quantile="0.95"} 0.0876
guitar_v3_latency_seconds{quantile="0.99"} 0.1234
guitar_v3_latency_seconds_count 1280
guitar_v3_latency_seconds_sum 58.4
```

---

## 🎨 Grafanaダッシュボード拡張

### 新規パネル（3つ追加）

#### Panel 10: Latency Distribution (Heatmap)
- **タイプ**: Heatmap
- **Y軸**: レイテンシ範囲（0-200ms、20msごと）
- **X軸**: 時間
- **Query**:
  ```promql
  rate(guitar_v3_latency_seconds_bucket[5m])
  ```

#### Panel 11: Latency Percentiles (Graph)
- **タイプ**: Time Series
- **系列**:
  - p50: 青線（目標線 50ms）
  - p95: 黄線（目標線 100ms）
  - p99: 赤線（目標線 200ms）
- **Query**:
  ```promql
  histogram_quantile(0.5, rate(guitar_v3_latency_seconds_bucket[5m]))
  histogram_quantile(0.95, rate(guitar_v3_latency_seconds_bucket[5m]))
  histogram_quantile(0.99, rate(guitar_v3_latency_seconds_bucket[5m]))
  ```
- **Alert**: p95 > 100ms for 5m

#### Panel 12: Latency vs Throughput (Scatter)
- **タイプ**: Graph
- **X軸**: リクエスト数/秒
- **Y軸**: p95レイテンシ
- **目的**: スループット増加時の遅延監視

---

## 🚨 アラートルール

### guitar_v3_alerts.yml拡張

```yaml
groups:
  - name: latency_alerts
    interval: 30s
    rules:
      # Critical: p95 > 150ms
      - alert: GuitarV3HighLatencyP95
        expr: |
          histogram_quantile(0.95, 
            rate(guitar_v3_latency_seconds_bucket[5m])
          ) > 0.150
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Guitar v3 latency p95 critically high"
          description: "p95 latency: {{ $value | humanizeDuration }}"
          runbook: "https://wiki/runbooks/high-latency"
      
      # Warning: p95 > 100ms
      - alert: GuitarV3LatencyP95Warning
        expr: |
          histogram_quantile(0.95, 
            rate(guitar_v3_latency_seconds_bucket[5m])
          ) > 0.100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Guitar v3 latency p95 above target"
      
      # Critical: p99 > 300ms
      - alert: GuitarV3HighLatencyP99
        expr: |
          histogram_quantile(0.99, 
            rate(guitar_v3_latency_seconds_bucket[5m])
          ) > 0.300
        for: 5m
        labels:
          severity: critical
      
      # Warning: max > 500ms
      - alert: GuitarV3SlowRequest
        expr: |
          max_over_time(guitar_v3_latency_seconds[5m]) > 0.500
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Guitar v3 slow request detected"
      
      # Latency regression (vs 1h ago)
      - alert: GuitarV3LatencyRegression
        expr: |
          histogram_quantile(0.95, rate(guitar_v3_latency_seconds_bucket[5m]))
          /
          histogram_quantile(0.95, rate(guitar_v3_latency_seconds_bucket[5m] offset 1h))
          > 1.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Guitar v3 latency increased 50% vs 1h ago"
```

---

## 📈 期待レイテンシ（ベンチマーク）

### 実測値（予測）

| 処理 | 時間 | 備考 |
|-----|------|------|
| **Pickle Load** | 10ms | 初回のみ（キャッシュ後0ms） |
| **Pattern Search** | 20-40ms | 2,148パターン全探索 |
| **ML Inference** | 5-10ms | XGBoost予測 |
| **Re-ranking** | 5ms | Top-10再ランク |
| **Total p50** | **35-50ms** | 目標達成 |
| **Total p95** | **60-90ms** | 目標達成 |
| **Total p99** | **100-150ms** | ギリギリ達成 |

### 最適化案（p95 < 80ms達成）

1. **パターンインデックス化**
   - Tempo/Section/Chordでインデックス
   - 全探索 → 絞り込み探索（1/10削減）
   - 期待効果: -15ms

2. **ML予測バッチ化**
   - Top-100候補を一括予測
   - For-loop → Batch predict
   - 期待効果: -5ms

3. **キャッシュ拡張**
   - (Chord, Tempo, Section)でキャッシュ
   - ヒット率80%想定
   - 期待効果: -30ms（ヒット時）

**最適化後目標**:
- p50: <30ms
- p95: <70ms
- p99: <120ms

---

## 🔧 実装ステップ

### Phase 1: 計測追加（1日）

- [ ] pattern_recommender.py に time.time() 追加
- [ ] CSVログフォーマット拡張（latency_ms列）
- [ ] ab_test_guitar_v3.py 出力修正
- [ ] テスト実行（10曲、遅延計測）

### Phase 2: メトリクス実装（1日）

- [ ] kpi_collector.py に遅延統計追加
- [ ] Prometheusメトリクス出力（histogram）
- [ ] テスト実行（メトリクス確認）

### Phase 3: 可視化（1日）

- [ ] Grafanaパネル3つ追加
- [ ] アラートルール3つ追加
- [ ] ダッシュボード動作確認

### Phase 4: 最適化（1週間）

- [ ] パターンインデックス実装
- [ ] ML予測バッチ化
- [ ] キャッシュ機能拡張
- [ ] ベンチマーク（p95 < 80ms達成）

---

## 📊 モックデータ（テスト用）

現在のCSVログに遅延データがないため、モックデータ生成：

```python
# monitoring/generate_latency_mock.py
import pandas as pd
import numpy as np

df = pd.read_csv('data/canary_kpi_v3_production.csv')

# 正規分布でモック遅延生成（平均60ms、標準偏差20ms）
np.random.seed(42)
latencies = np.random.normal(60, 20, len(df))
latencies = np.clip(latencies, 10, 500)  # 10-500msに制限

df['latency_ms'] = latencies

df.to_csv('data/canary_kpi_v3_with_latency.csv', index=False)
print(f"Generated latency data: p50={np.percentile(latencies, 50):.1f}ms, p95={np.percentile(latencies, 95):.1f}ms")
```

**実行結果（期待）**:
```
Generated latency data: p50=60.2ms, p95=92.4ms
```

---

## ✅ 完了基準

- [ ] p50/p95/p99メトリクス収集
- [ ] Grafana遅延パネル3つ表示
- [ ] p95 > 100ms アラート動作確認
- [ ] 実測p95 < 100ms達成

---

**Next Action**: モックデータ生成 → kpi_collector.py拡張 → Grafanaパネル追加
