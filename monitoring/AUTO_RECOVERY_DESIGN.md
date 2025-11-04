# Shadow Auto-Recovery Design

## 概要

Shadow TrafficにおけるAuto-Recovery（自動復帰）機能の設計仕様。
v3→v1のフォールバックだけでなく、v1→v3への自動復帰も実現し、運用の自動化を図る。

## 目的

- **双方向フォールバック**: v3でKPI劣化 → v1に切替、v1で安定 → v3に復帰
- **ヒステリシス**: 頻繁な切り替えを防ぐため、連続的な成功/失敗を要求
- **運用自動化**: 手動介入なしでバージョン管理

## 用語定義

- **Breach（違反）**: KPIがgate閾値を下回ること
  - 例: accent_score < 0.60, chord_fit < 0.40
- **Window（ウィンドウ）**: 違反カウントの観測期間
  - デフォルト: 32バー（約8小節 x 4セクション）
- **Threshold（閾値）**: ウィンドウ内で許容される違反回数
  - デフォルト: 6回（約20%の失敗率）
- **Cooldown（クールダウン）**: バージョン切替後の猶予期間
  - デフォルト: 16バー（約4小節 x 4セクション）

## 動作仕様

### 1. 状態遷移

```
[State A: v3 Active]
  ↓ (32バーウィンドウで6回以上breach)
[State B: v1 Fallback + Cooldown(16バー)]
  ↓ (Cooldown終了後、32バーウィンドウで違反なし)
[State C: v3 Recovery + Cooldown(16バー)]
  ↓ (安定稼働継続)
[State A: v3 Active]
```

### 2. 違反検出ロジック

各リクエストで以下をチェック：

```python
def check_breach(result: ComparisonResult, gate_config: dict) -> bool:
    """KPIゲート違反を検出"""
    section = result.section
    gate = get_gate_threshold(gate_config, section)
    
    # v3のKPIをチェック
    if result.v3_error:
        return True  # エラーは違反扱い
    
    if result.v3_accent_score < gate['accent_min']:
        return True
    
    if result.v3_chord_fit < gate['chord_min']:
        return True
    
    return False
```

### 3. ウィンドウ管理

```python
from collections import deque

class AutoRecoveryManager:
    def __init__(self, window_size=32, threshold=6, cooldown=16):
        self.window_size = window_size
        self.threshold = threshold
        self.cooldown = cooldown
        
        self.window = deque(maxlen=window_size)  # 最新32件の結果
        self.cooldown_counter = 0  # クールダウン残りバー数
        self.current_version = 'v3'  # 現在のアクティブバージョン
```

### 4. バージョン切替判定

```python
def should_switch_version(self) -> Optional[str]:
    """バージョン切替が必要か判定"""
    # クールダウン中は切替しない
    if self.cooldown_counter > 0:
        return None
    
    # ウィンドウ内の違反回数をカウント
    breach_count = sum(1 for is_breach in self.window if is_breach)
    
    # v3アクティブ時: 違反が閾値超過 → v1へ切替
    if self.current_version == 'v3' and breach_count >= self.threshold:
        return 'v1'
    
    # v1アクティブ時: 違反なし（安定） → v3へ復帰
    if self.current_version == 'v1' and breach_count == 0:
        return 'v3'
    
    return None  # 切替不要
```

### 5. クールダウン処理

```python
def switch_version(self, new_version: str):
    """バージョン切替を実行し、クールダウンを開始"""
    self.current_version = new_version
    self.cooldown_counter = self.cooldown
    self.window.clear()  # ウィンドウをリセット
    
    logger.info(f"Auto-Recovery: Switched to {new_version}, cooldown={self.cooldown} bars")

def tick_cooldown(self):
    """クールダウンカウンターを減算"""
    if self.cooldown_counter > 0:
        self.cooldown_counter -= 1
```

## 統合方法

### TrafficSplitterへの組み込み

```python
class TrafficSplitter:
    def __init__(self, ..., enable_auto_recovery=True):
        # ...
        if enable_auto_recovery:
            self.auto_recovery = AutoRecoveryManager(
                window_size=32,
                threshold=6,
                cooldown=16
            )
        else:
            self.auto_recovery = None
    
    def route_and_compare(self, ...) -> ComparisonResult:
        # ...既存のroute処理...
        
        # Auto-Recovery判定
        if self.auto_recovery:
            is_breach = check_breach(result, self.gate_config)
            self.auto_recovery.add_result(is_breach)
            
            new_version = self.auto_recovery.should_switch_version()
            if new_version:
                self.auto_recovery.switch_version(new_version)
                # v3_ratioを動的に変更
                if new_version == 'v1':
                    self.v3_ratio = 0.0  # v1に完全切替
                elif new_version == 'v3':
                    self.v3_ratio = 0.9  # v3に復帰（90%）
            
            self.auto_recovery.tick_cooldown()
        
        return result
```

## Prometheusメトリクス

```prometheus
# HELP auto_recovery_switches_total バージョン切替回数（累積）
# TYPE auto_recovery_switches_total counter
auto_recovery_switches_total{from="v3",to="v1"} 5
auto_recovery_switches_total{from="v1",to="v3"} 3

# HELP auto_recovery_cooldown_active クールダウン中フラグ
# TYPE auto_recovery_cooldown_active gauge
auto_recovery_cooldown_active 1  # 1=クールダウン中, 0=通常

# HELP auto_recovery_breach_count ウィンドウ内の違反回数
# TYPE auto_recovery_breach_count gauge
auto_recovery_breach_count 4  # 現在のウィンドウ内の違反数

# HELP auto_recovery_window_size ウィンドウサイズ（バー数）
# TYPE auto_recovery_window_size gauge
auto_recovery_window_size 32

# HELP auto_recovery_current_version 現在のアクティブバージョン
# TYPE auto_recovery_current_version gauge
auto_recovery_current_version{version="v3"} 1
auto_recovery_current_version{version="v1"} 0
```

## テストシナリオ

### Scenario 1: v3 → v1 Fallback

```
Input:
  - 初期状態: v3 active
  - 32バーで8回breach（閾値6超過）

Expected:
  - v1に切替
  - cooldown=16バー開始
  - メトリクス: auto_recovery_switches_total{from="v3",to="v1"} += 1
```

### Scenario 2: v1 → v3 Recovery

```
Input:
  - 初期状態: v1 active, cooldown=0
  - 32バーで0回breach（完全安定）

Expected:
  - v3に復帰
  - cooldown=16バー開始
  - メトリクス: auto_recovery_switches_total{from="v1",to="v3"} += 1
```

### Scenario 3: Cooldown中の切替抑制

```
Input:
  - 初期状態: v3 active, cooldown=10バー
  - 新たに6回以上breach発生

Expected:
  - バージョン切替なし（クールダウン中のため）
  - cooldown_counterは通常通り減算継続
```

## パラメータチューニング指針

| パラメータ | デフォルト | 調整方針 |
|----------|---------|---------|
| window_size | 32バー | 小→応答性高、大→安定性高 |
| threshold | 6回 | 小→厳格、大→寛容 |
| cooldown | 16バー | 小→頻繁切替、大→慎重切替 |

### 調整例

- **高トラフィック環境**: window_size=64, threshold=12（より多くのデータで判断）
- **低レイテンシ要求**: cooldown=8（早期復帰）
- **安定性重視**: threshold=3（厳格なゲート）

## 実装優先度

1. **Phase 1** (HIGH): 基本的な双方向切替
   - AutoRecoveryManagerクラス
   - TrafficSplitter統合
   - 基本メトリクス

2. **Phase 2** (MEDIUM): 高度な制御
   - セクション別閾値対応
   - 動的パラメータ調整API
   - アラート連携

3. **Phase 3** (LOW): 可観測性強化
   - Grafanaダッシュボード
   - 切替履歴ログ
   - A/Bテスト統計分析

## 関連ドキュメント

- `DISTRIBUTION_MONITORING_GUIDE.md`: 分布ベース監視
- `gate_prod.yaml`: KPIゲート閾値定義
- `GRAFANA_LATENCY_PANELS_COMPLETE.md`: Grafana設定

## 更新履歴

- 2025-10-27: 初版作成（Phase 21完了後）
