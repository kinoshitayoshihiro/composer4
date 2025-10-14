# Phase 5: Parameter Application & ML Enhancement

**開始日**: 2025-10-14  
**予定期間**: 2-3週間  
**目標**: v1.1.0 リリース準備

---

## 🎯 Phase 5 概要

Phase 4で構築したEmotion Mapping Systemの**実際のパラメータ適用**を実装し、機械学習機能を強化します。

### Phase 4との違い

- **Phase 4.9**: Emotion adjustmentsを`section_data['_emotion_adjustments']`に**格納のみ**
- **Phase 5**: 格納された調整値を**実際の生成ロジックに適用**

---

## 📋 Phase 5 Sub-Phases

### Phase 5.0: 事前準備 (1日)
- [ ] Phase 5計画書作成 (このドキュメント)
- [ ] 現状分析・ベースライン確立
- [ ] テスト計画策定

### Phase 5.1: Piano Parameter Application (2-3日)
- [ ] velocity_std_multiplier適用
- [ ] notes_per_bar_multiplier適用
- [ ] 既存テストとの互換性確認
- [ ] Quality Gate検証

### Phase 5.2: Guitar Parameter Application (2-3日)
- [ ] strum_consistency_target適用
- [ ] velocity_boost適用
- [ ] ストローク生成への統合
- [ ] Quality Gate検証

### Phase 5.3: Bass Parameter Application (2-3日)
- [ ] notes_per_bar_multiplier適用
- [ ] root_emphasis適用
- [ ] ウォーキングベースへの統合
- [ ] Quality Gate検証

### Phase 5.4: Strings Parameter Application (2-3日)
- [ ] legato_rate_target適用
- [ ] chord_spread_multiplier適用
- [ ] レガート・アルペジオ生成への統合
- [ ] Quality Gate検証

### Phase 5.5: Drums Parameter Application (2-3日)
- [ ] hihat_density_multiplier適用
- [ ] kick_emphasis適用
- [ ] velocity_boost適用
- [ ] フィル・トランジションへの統合
- [ ] Quality Gate検証

### Phase 5.6: A/B Testing Framework (2-3日)
- [ ] 比較テストスクリプト作成
- [ ] メトリクス収集自動化
- [ ] 統計的有意性検証
- [ ] レポート生成

### Phase 5.7: ML Enhancement (3-5日)
- [ ] Velocity予測モデル統合
- [ ] Context-awareパラメータ選択
- [ ] Historical learning実装
- [ ] モデル評価

### Phase 5.8: Performance Optimization (2-3日)
- [ ] 生成速度最適化
- [ ] メモリ使用量削減
- [ ] キャッシング戦略
- [ ] ベンチマーク

### Phase 5.9: v1.1 Release Preparation (2-3日)
- [ ] 全楽器を100%完成度に
- [ ] ドキュメント更新
- [ ] Migration guide作成
- [ ] v1.1.0タグ作成

---

## 🔧 Phase 5.1詳細: Piano Parameter Application

### 実装対象

#### 1. velocity_std_multiplier

**現状**:
```python
# generator/piano_generator.py
velocity_std = 15  # ハードコード
```

**実装後**:
```python
# emotion adjustmentsから取得
emotion_adj = section_data.get('_emotion_adjustments', {}).get('piano', {})
base_velocity_std = 15
velocity_std = base_velocity_std * emotion_adj.get('velocity_std_multiplier', 1.0)
```

**影響範囲**:
- `_render_part()` メソッド
- Velocity計算ロジック

#### 2. notes_per_bar_multiplier

**現状**:
```python
# 固定のnotes_per_bar計算
notes_per_bar = calculate_notes_per_bar(...)
```

**実装後**:
```python
# Emotion調整を反映
base_notes_per_bar = calculate_notes_per_bar(...)
emotion_adj = section_data.get('_emotion_adjustments', {}).get('piano', {})
notes_per_bar = base_notes_per_bar * emotion_adj.get('notes_per_bar_multiplier', 1.0)
```

**影響範囲**:
- `compose()` メソッド
- Note density計算

### テスト戦略

1. **Unit Tests**: 各パラメータの適用確認
2. **Integration Tests**: Emotion profileごとの動作確認
3. **Quality Gate**: 既存メトリクスの維持
4. **A/B Test**: happy_high vs neutral_mediumの比較

### Success Criteria

- ✅ velocity_stdが調整値で変化する
- ✅ notes_per_barが調整値で変化する
- ✅ 既存のQuality Gateを通過
- ✅ Emotion profileの違いが生成に反映される

---

## 🔧 Phase 5.2詳細: Guitar Parameter Application

### 実装対象

#### 1. strum_consistency_target

**現状**:
```python
# ハードコードされたstrum consistency
strum_consistency = 0.8
```

**実装後**:
```python
emotion_adj = section_data.get('_emotion_adjustments', {}).get('guitar', {})
strum_consistency = emotion_adj.get('strum_consistency_target', 0.8)
```

#### 2. velocity_boost

**現状**:
```python
# 固定velocity
velocity = base_velocity
```

**実装後**:
```python
emotion_adj = section_data.get('_emotion_adjustments', {}).get('guitar', {})
velocity_boost = emotion_adj.get('velocity_boost', 0)
velocity = base_velocity + velocity_boost
```

### Success Criteria

- ✅ Strum consistencyが感情プロファイルで変化
- ✅ Velocityがboost値で調整される
- ✅ Guitar Quality Gateを通過

---

## 🔧 Phase 5.3詳細: Bass Parameter Application

### 実装対象

#### 1. notes_per_bar_multiplier

**用途**: energetic_high, melancholic_medium, calm_low

```python
emotion_adj = section_data.get('_emotion_adjustments', {}).get('bass', {})
base_notes_per_bar = 4  # 例
notes_per_bar = base_notes_per_bar * emotion_adj.get('notes_per_bar_multiplier', 1.0)
```

#### 2. root_emphasis

**用途**: ルート音の強調

```python
root_emphasis = emotion_adj.get('root_emphasis', 1.0)
if note_is_root:
    velocity *= root_emphasis
```

### Success Criteria

- ✅ 音符密度が感情で変化
- ✅ ルート音が適切に強調される
- ✅ Bass Quality Gateを通過

---

## 🔧 Phase 5.4詳細: Strings Parameter Application

### 実装対象

#### 1. legato_rate_target

```python
emotion_adj = section_data.get('_emotion_adjustments', {}).get('strings', {})
legato_rate = emotion_adj.get('legato_rate_target', 0.5)
```

#### 2. chord_spread_multiplier

```python
chord_spread = emotion_adj.get('chord_spread_multiplier', 1.0)
spread_time = base_spread_time * chord_spread
```

### Success Criteria

- ✅ Legato rateが感情で変化
- ✅ Chord spreadが調整される
- ✅ Strings Quality Gateを通過

---

## 🔧 Phase 5.5詳細: Drums Parameter Application

### 実装対象

#### 1. hihat_density_multiplier

```python
emotion_adj = section_data.get('_emotion_adjustments', {}).get('drums', {})
base_density = 0.75
hihat_density = base_density * emotion_adj.get('hihat_density_multiplier', 1.0)
```

#### 2. kick_emphasis

```python
kick_emphasis = emotion_adj.get('kick_emphasis', 1.0)
if note_is_kick:
    velocity *= kick_emphasis
```

#### 3. velocity_boost

```python
velocity_boost = emotion_adj.get('velocity_boost', 0)
velocity = base_velocity + velocity_boost
```

### Success Criteria

- ✅ ハイハット密度が感情で変化
- ✅ キックが適切に強調される
- ✅ 全体のvelocityが調整される
- ✅ Drums Quality Gateを通過

---

## 🔧 Phase 5.6詳細: A/B Testing Framework

### 実装内容

#### 1. 比較テストスクリプト

**scripts/ab_test_emotions.py**:
```python
def compare_emotion_profiles(
    instrument: str,
    profile1: str,
    profile2: str,
    section: str = "Chorus",
    num_samples: int = 10
):
    """2つの感情プロファイルを比較"""
    results1 = generate_with_emotion(instrument, section, profile1, num_samples)
    results2 = generate_with_emotion(instrument, section, profile2, num_samples)
    
    # メトリクス比較
    metrics_comparison = compare_metrics(results1, results2)
    
    # 統計的有意性検定
    significance = statistical_test(results1, results2)
    
    return {
        'profile1': profile1,
        'profile2': profile2,
        'metrics': metrics_comparison,
        'significance': significance
    }
```

#### 2. メトリクス収集

- velocity_std
- notes_per_bar
- grid_off_std_ms
- chord_tone_rate
- 楽器別メトリクス

#### 3. レポート生成

**Markdown形式**:
```markdown
# A/B Test Report: happy_high vs neutral_medium

## Summary
- Instrument: Piano
- Section: Chorus
- Samples: 10 each

## Metrics Comparison
| Metric | happy_high | neutral_medium | Diff | p-value |
|--------|-----------|----------------|------|---------|
| velocity_std | 18.5 | 15.0 | +3.5 | 0.001 |
| notes_per_bar | 6.2 | 4.8 | +1.4 | 0.005 |
...
```

### Success Criteria

- ✅ 2プロファイルの比較が自動化される
- ✅ 統計的有意性が計算される
- ✅ レポートが自動生成される

---

## 🔧 Phase 5.7詳細: ML Enhancement

### 実装内容

#### 1. Velocity予測モデル統合

**現状**: ランダムまたは固定velocity
**改善後**: Context-awareな予測

```python
from ml.velocity_predictor import VelocityPredictor

predictor = VelocityPredictor.load_checkpoint("phrase.best.ckpt")

# コンテキスト情報
context = {
    'chord': current_chord,
    'position': bar_position,
    'previous_notes': prev_notes,
    'emotion_profile': emotion_profile
}

# 予測
predicted_velocity = predictor.predict(context)
```

#### 2. Context-awareパラメータ選択

**自動emotion profile選択**:
```python
def auto_select_emotion(
    section: str,
    previous_section: str,
    song_context: dict
) -> str:
    """楽曲コンテキストから最適な感情プロファイルを選択"""
    
    # セクション遷移ルールを考慮
    transition_rule = get_transition_rule(previous_section, section)
    
    # 楽曲全体の雰囲気を考慮
    song_mood = song_context.get('overall_mood', 'neutral')
    
    # 最適プロファイルを決定
    return select_best_profile(section, transition_rule, song_mood)
```

#### 3. Historical Learning

**生成履歴から学習**:
```python
class HistoricalLearner:
    def __init__(self):
        self.history = []
    
    def record(self, generation_params, quality_metrics):
        """生成パラメータと品質メトリクスを記録"""
        self.history.append({
            'params': generation_params,
            'metrics': quality_metrics,
            'timestamp': datetime.now()
        })
    
    def suggest_params(self, target_metrics):
        """目標メトリクスに最適なパラメータを提案"""
        # 履歴から類似ケースを検索
        similar_cases = self.find_similar(target_metrics)
        
        # パラメータを推薦
        return self.recommend_params(similar_cases)
```

### Success Criteria

- ✅ Velocity予測が生成に統合される
- ✅ Emotion profileの自動選択が機能する
- ✅ 履歴ベースの学習が動作する
- ✅ 生成品質が向上する (QG metrics)

---

## 🔧 Phase 5.8詳細: Performance Optimization

### 実装内容

#### 1. 生成速度最適化

**ボトルネック分析**:
```python
import cProfile
import pstats

# プロファイリング
profiler = cProfile.Profile()
profiler.enable()

# 生成処理
part = generator.compose(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

**最適化候補**:
- numpy演算のベクトル化
- 不要な再計算の削除
- キャッシュ活用

#### 2. メモリ使用量削減

**メモリプロファイリング**:
```python
from memory_profiler import profile

@profile
def compose_optimized(self, ...):
    # メモリ効率的な実装
    ...
```

#### 3. キャッシング戦略

**Emotion adjustmentsのキャッシュ**:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_emotion_adjustments_cached(
    instrument: str,
    emotion_profile: str
) -> dict:
    config = load_emotion_mapping()  # 1回のみ
    return get_emotion_adjustments(instrument, emotion_profile, config)
```

### Success Criteria

- ✅ 生成速度が20%以上向上
- ✅ メモリ使用量が15%以上削減
- ✅ ベンチマーク結果が改善

---

## 🔧 Phase 5.9詳細: v1.1 Release Preparation

### 実装内容

#### 1. 全楽器を100%完成度に

**残りタスク**:
- Guitar: 95% → 100% (パラメータ適用完了)
- Bass: 90% → 100% (パラメータ適用 + エッジケース)
- Strings: 90% → 100% (パラメータ適用 + レガート改善)
- Drums: 90% → 100% (パラメータ適用 + フィル改善)

#### 2. ドキュメント更新

**更新対象**:
- README.md: v1.1新機能追加
- EMOTION_MAPPING_GUIDE.md: パラメータ適用例追加
- API_REFERENCE.md: 新規作成
- PERFORMANCE_GUIDE.md: 最適化ガイド

#### 3. Migration Guide

**docs/MIGRATION_v1.0_to_v1.1.md**:
```markdown
# Migration Guide: v1.0 → v1.1

## Breaking Changes

### 1. Emotion Parametersの挙動変更

**v1.0**: 格納のみ
**v1.1**: 実際に適用される

**移行方法**:
特になし（後方互換）

### 2. 新しいAPI

**自動emotion選択**:
```python
# v1.1新機能
part = piano.compose(
    section_data=section_data,
    section="Chorus",
    auto_emotion=True  # NEW
)
```

## New Features

- ✅ Emotion parameter完全適用
- ✅ A/Bテストツール
- ✅ ML予測統合
- ✅ Performance最適化

## Upgrade Steps

1. Requirements更新: `pip install -r requirements.txt`
2. 設定ファイル更新: emotion_mapping.yaml確認
3. テスト実行: `pytest tests/`
4. Quality Gate確認: `./scripts/ci_quality_gate.sh`
```

#### 4. v1.1.0タグ作成

```bash
git tag -a v1.1.0 -m "Release v1.1.0: Parameter Symphony

composer2-3 v1.1.0 - Feature Release

Codename: Parameter Symphony
Release Date: 2025-11-XX

Major Features:
✅ Emotion Parameter Application: All 5 instruments
✅ A/B Testing Framework: Statistical comparison
✅ ML Enhancement: Velocity prediction, auto-selection
✅ Performance Optimization: 20% faster, 15% less memory
✅ 100% Instrument Completion: All instruments fully implemented

Phase 5 Complete (9/9 phases)
...
"
```

### Success Criteria

- ✅ 全楽器が100%完成
- ✅ ドキュメントが完全
- ✅ Migration guideが明確
- ✅ v1.1.0タグ作成完了

---

## 📊 Phase 5 全体タイムライン

```
Week 1 (Day 1-7):
├── Phase 5.0: 計画・準備 (Day 1)
├── Phase 5.1: Piano適用 (Day 2-4)
└── Phase 5.2: Guitar適用 (Day 5-7)

Week 2 (Day 8-14):
├── Phase 5.3: Bass適用 (Day 8-10)
├── Phase 5.4: Strings適用 (Day 11-13)
└── Phase 5.5: Drums適用 (Day 14)

Week 3 (Day 15-21):
├── Phase 5.6: A/Bテスト (Day 15-17)
├── Phase 5.7: ML強化 (Day 18-20)
└── Phase 5.8: 最適化 (Day 21)

Week 4 (Day 22-24):
└── Phase 5.9: v1.1リリース (Day 22-24)
```

**Total**: 24日 (3-4週間)

---

## 🎯 Success Metrics

### Phase 5全体

1. **機能完成度**
   - ✅ 全楽器でemotion parameter適用
   - ✅ A/Bテスト自動化
   - ✅ ML機能統合
   - ✅ パフォーマンス20%向上

2. **品質メトリクス**
   - ✅ Quality Gate: 100% PASS (全楽器)
   - ✅ テストカバレッジ: 80%+
   - ✅ ドキュメント完全性: 100%

3. **コード品質**
   - ✅ Type hints: 100%
   - ✅ Docstrings: 100%
   - ✅ Pylint score: 9.0+

4. **リリース品質**
   - ✅ v1.1.0タグ作成
   - ✅ Migration guide完全
   - ✅ 後方互換性維持

---

## 🔄 リスク管理

### 潜在的リスク

1. **パラメータ適用の複雑性**
   - リスク: 既存ロジックとの衝突
   - 対策: 段階的実装、徹底的テスト

2. **Quality Gate違反**
   - リスク: 新パラメータでQG失敗
   - 対策: 調整値の慎重な設定、A/Bテスト

3. **パフォーマンス劣化**
   - リスク: ML統合で速度低下
   - 対策: プロファイリング、キャッシング

4. **後方互換性**
   - リスク: 既存コードが動作しない
   - 対策: デフォルト値設定、移行ガイド

### 対策

- ✅ 各Phase後にQuality Gate実行
- ✅ 段階的コミット（Phase単位）
- ✅ 継続的ドキュメント更新
- ✅ ユーザーフィードバック収集

---

## 📝 Phase 5.0 即時タスク

### 今日実施 (2025-10-14)

1. ✅ Phase 5計画書作成 (このドキュメント)
2. [ ] 現状分析スクリプト作成
3. [ ] ベースライン品質確立
4. [ ] Phase 5.1準備

### 現状分析

**scripts/analyze_current_state.py**:
```python
"""Phase 5開始時の現状分析"""

import json
from pathlib import Path

def analyze_emotion_integration():
    """Emotion統合状態を分析"""
    generators = [
        'generator/piano_generator.py',
        'generator/guitar_generator.py',
        'generator/bass_generator.py',
        'generator/strings_generator.py',
        'generator/drum_generator.py'
    ]
    
    for gen_file in generators:
        # emotion_loader importを確認
        # section/emotion_profile parametersを確認
        # _emotion_adjustments格納を確認
        # 実際の適用を確認（Phase 5で実装予定）
        ...
    
    return analysis_report

def run_baseline_quality_gates():
    """ベースライン品質を確立"""
    # 各楽器のQuality Gate実行
    # メトリクスを記録
    # Phase 5後との比較用
    ...

if __name__ == "__main__":
    print("=== Phase 5.0: Current State Analysis ===")
    
    # Emotion統合分析
    emotion_report = analyze_emotion_integration()
    print(f"\nEmotion Integration Status:")
    print(json.dumps(emotion_report, indent=2))
    
    # ベースライン品質
    baseline = run_baseline_quality_gates()
    print(f"\nBaseline Quality Metrics:")
    print(json.dumps(baseline, indent=2))
    
    # レポート保存
    with open('results/phase5_baseline.json', 'w') as f:
        json.dump({
            'emotion_integration': emotion_report,
            'baseline_quality': baseline,
            'timestamp': '2025-10-14'
        }, f, indent=2)
    
    print("\n✅ Phase 5.0 Analysis Complete!")
```

---

## 📚 参考ドキュメント

### Phase 4ドキュメント（基礎）

- [PHASE_4_9_COMPLETE.md](./PHASE_4_9_COMPLETE.md) - Emotion統合完了
- [EMOTION_MAPPING_GUIDE.md](./EMOTION_MAPPING_GUIDE.md) - 使用ガイド
- [RELEASE_NOTES_v1.0.md](./RELEASE_NOTES_v1.0.md) - v1.0詳細

### Phase 5新規ドキュメント（予定）

- docs/PHASE_5_1_PIANO_APPLICATION.md
- docs/PHASE_5_6_AB_TESTING.md
- docs/PHASE_5_7_ML_ENHANCEMENT.md
- docs/MIGRATION_v1.0_to_v1.1.md
- docs/API_REFERENCE.md
- docs/PERFORMANCE_GUIDE.md

---

## 🚀 Next Steps

### Phase 5.0 (今日)

```bash
# 1. 現状分析スクリプト作成
touch scripts/analyze_current_state.py

# 2. ベースライン確立
python scripts/analyze_current_state.py

# 3. Phase 5.1準備
# - Piano generator確認
# - velocity_std適用箇所特定
# - notes_per_bar適用箇所特定
```

### Phase 5.1 (明日〜)

- Piano parameter application実装開始
- テスト作成
- Quality Gate検証

---

**Phase 5計画完成!** 🎊

次は現状分析スクリプトを作成し、ベースラインを確立します。

---

**Version**: 1.0  
**Date**: 2025-10-14  
**Status**: Phase 5.0 Planning Complete
