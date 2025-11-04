# Shadow Traffic 分布監視ガイド

## 概要

Shadow Traffic システムに分布ベース監視（p10/p50/p90 パーセンタイル）を追加しました。
平均値だけでなく、分布の裾（tail）の劣化を検出できるようになります。

## 追加されたメトリクス

### Accent Score分布
- `guitar_v3_accent_score_p10` - v3の10パーセンタイル（下位10%のボーダー）
- `guitar_v3_accent_score_p50` - v3の中央値
- `guitar_v3_accent_score_p90` - v3の90パーセンタイル（上位10%のボーダー）
- `guitar_v1_accent_score_p10/p50/p90` - v1の同様のメトリクス

### Chord Fit分布
- `guitar_v3_chord_fit_p10/p50/p90` - v3のChord Fit分布
- `guitar_v1_chord_fit_p10/p50/p90` - v1のChord Fit分布

### Latency分布
- `guitar_v3_latency_p50_ms` - v3の中央値レイテンシ
- `guitar_v3_latency_p95_ms` - v3の95パーセンタイル（上位5%のボーダー）
- `guitar_v1_latency_p50_ms/p95_ms` - v1の同様のメトリクス

## セクション別統計

`TrafficSplitter.get_section_statistics()` メソッドで各セクション（Chorus, Verse, Bridge等）の統計を取得できます：

```python
splitter = TrafficSplitter(...)
section_stats = splitter.get_section_statistics()

for section, stats in section_stats.items():
    print(f"{section}:")
    print(f"  Count: {stats['count']}")
    print(f"  v3 Accent Mean: {stats['v3_accent_mean']:.3f}")
    print(f"  v3 Accent Median: {stats['v3_accent_p50']:.3f}")
```

## Grafanaダッシュボード

`monitoring/grafana_dashboard_shadow_traffic.json` をGrafanaにインポートして使用できます。

### パネル構成
1. **v3 Accent Score Distribution** - p10/p50/p90の時系列推移
2. **v3 Chord Fit Distribution** - Chord Fitの分布推移
3. **Latency Distribution** - v3/v1のレイテンシ比較（p50/p95）
4. **Win Rates & Error Rates** - 勝率とエラー率

### アラート設定例

#### 下位10%の劣化検出
```yaml
# Accent Scoreの下位10%が閾値を下回った場合
- alert: AccentScoreTailDegradation
  expr: guitar_v3_accent_score_p10 < 0.50
  for: 5m
  annotations:
    summary: "v3 Accent Score下位10%が0.50を下回っています"
```

#### 上位95%のレイテンシ増加
```yaml
# p95レイテンシが10msを超えた場合
- alert: HighLatencyP95
  expr: guitar_v3_latency_p95_ms > 10
  for: 5m
  annotations:
    summary: "v3 p95レイテンシが10msを超えています"
```

## 分布監視のメリット

### 1. 平均に隠れた問題の検出
- 平均が正常でも、下位10%が大きく劣化している場合を検出
- 例: 平均0.80でも、p10が0.30なら一部のパターンで大きく失敗

### 2. レイテンシの裾の監視
- p50（中央値）は正常でも、p95が大きい場合、一部のリクエストで遅延
- SLO設定に重要（例: 95%のリクエストが5ms以内）

### 3. セクション別の問題特定
- Chorusだけ劣化、Verseは正常などの傾向を発見
- セクション別ゲート閾値の調整に活用

## 使用例

### シャドウテスト実行
```bash
python scripts/test_shadow_traffic.py --songs 100

# メトリクス確認
cat data/shadow_metrics.txt | grep p50
```

### 統計サマリー表示
```python
from ml.traffic_splitter import TrafficSplitter

splitter = TrafficSplitter(
    v3_pickle_path="data/patterns/stage2_guitar_v3_fixed.pickle",
    v1_pickle_path="data/patterns/stage2_guitar.pickle"
)

# テスト実行後
splitter.print_summary()  # セクション別統計も表示される
```

### CSV分析
```python
import pandas as pd

df = pd.read_csv('data/shadow_traffic_log.csv')

# セクション別のaccent score分布
df.groupby('section')['v3_accent_score'].describe()

# 時系列での推移確認
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp').resample('1H')['v3_accent_score'].agg(['mean', lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)])
```

## 次のステップ

1. **PrometheusとGrafanaのセットアップ**
   - Prometheusで `data/shadow_metrics.txt` を定期的にスクレイプ
   - Grafanaダッシュボードをインポート

2. **アラート設定**
   - p10/p90ベースのアラートルール作成
   - Slackやメール通知の設定

3. **長期運用**
   - 週次でセクション別統計をレビュー
   - 分布の変化傾向を分析してモデル改善に活用
