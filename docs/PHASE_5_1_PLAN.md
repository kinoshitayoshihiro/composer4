# Phase 5.1 実装計画: Piano Parameter Application

**開始予定日**: 2025-10-15  
**予定期間**: 2-3日  
**目標**: Piano Generatorでemotion adjustmentsを実際の生成に適用

---

## 🎯 Phase 5.1 目標

Phase 4.9で**格納のみ**だったemotion adjustmentsを、Piano Generatorの**実際の生成ロジック**に適用する。

### 適用対象パラメータ

1. **velocity_std_multiplier**: velocity変動の標準偏差を調整
2. **notes_per_bar_multiplier**: 1小節あたりの音符数を調整

---

## 📍 現状分析結果 (Phase 5.0より)

### Piano Generator概要

- **ファイル**: `generator/piano_generator.py`
- **コード行数**: 857行
- **コメント行数**: 8行
- **Phase 4.9実装**: ✅ Complete
  - emotion_loader import: ✅
  - section/emotion_profile params: ✅
  - _emotion_adjustments storage: ✅
  - Parameter application: ⏳ Phase 5.1で実装

### composeメソッド (line 851-951)

**現状**:
```python
def compose(
    self,
    *,
    section_data: dict[str, Any],
    ...
    section: str = "Verse",
    emotion_profile: str | None = None,
) -> stream.Part | dict[str, stream.Part]:
    # Emotion adjustments storage (Phase 4.9)
    if emotion_profile is not None or section != "Verse":
        try:
            emotion_params = get_generation_params(
                "piano",
                section=section,
                emotion_profile=emotion_profile
            )
            # ⏳ 格納のみ (Phase 5.1で実際に適用)
            section_data.setdefault("_emotion_adjustments", {})
            section_data["_emotion_adjustments"]["piano"] = emotion_params
        except Exception as e:
            logging.warning(f"Failed to load emotion adjustments: {e}")
    
    # 既存のsuper().compose()呼び出し
    result = super().compose(...)
    
    # Echo機能（既存）
    ...
    
    return result
```

**問題点**: emotion_paramsを取得しているが、実際の生成には使用されていない

---

## 🔧 実装計画

### 1. velocity_std_multiplier適用

#### 現状のvelocity処理

**場所**: `_render_hand_part()` メソッド (line 220-337)

```python
base_velocity = params.get("velocity", 70)  # line 291
...
vel_factor = float(p_event.get("velocity_factor", 1.0))
velocity = scale_velocity(base_velocity, vel_factor)  # line 308-309
```

**問題**: velocityのバラつきは`velocity_factor`で決まっているが、これは**パターンファイル固定**

#### 実装方法

**Option A**: humanizerで適用 (推奨)

Piano Generatorは`utilities.humanizer.apply()`を使用してvelocityの人間的なバラつきを追加している (line 806-819):

```python
for part in (rh_part, lh_part):
    if profile_name:
        humanizer.apply(
            part,
            profile_name,
            global_settings=self.global_settings,
        )
```

**実装**:
```python
# composeメソッド内、humanizerを呼ぶ前
emotion_adj = section_data.get('_emotion_adjustments', {}).get('piano', {})
velocity_std_mult = emotion_adj.get('velocity_std_multiplier', 1.0)

# humanizerの設定を一時的に上書き
if velocity_std_mult != 1.0:
    # humanizer profileの velocity_std を調整
    # または、humanizer後に追加のvelocity調整
    ...
```

**Option B**: _render_hand_partで直接適用

```python
def _render_hand_part(self, ...):
    ...
    # Emotion adjustmentsを取得
    emotion_adj = section_data.get('_emotion_adjustments', {}).get('piano', {})
    velocity_std_mult = emotion_adj.get('velocity_std_multiplier', 1.0)
    
    base_velocity = params.get("velocity", 70)
    
    # velocity計算ループで、velocity_stdを考慮
    for p_event in pattern_events:
        ...
        velocity = scale_velocity(base_velocity, vel_factor)
        
        # Emotion調整を反映
        if velocity_std_mult != 1.0:
            # ランダムなバラつきを追加
            std_dev = 15 * velocity_std_mult  # base std = 15
            velocity_noise = self.rng.normal(0, std_dev)
            velocity = int(velocity + velocity_noise)
            velocity = max(1, min(127, velocity))
```

**推奨**: Option B (より直接的、制御しやすい)

---

### 2. notes_per_bar_multiplier適用

#### 現状のnote density処理

**問題**: Piano Generatorは**パターンファイル**でnote densityを制御している。

**パターン例**:
- `piano_rh_ambient_pad`: 少ない音符（pad）
- `piano_rh_arpeggio_sixteenths_up_down`: 多い音符（アルペジオ）

**pattern_eventsの例**:
```python
pattern_events = [
    {"offset": 0.0, "duration": 1.0, "type": "chord"},
    {"offset": 1.0, "duration": 1.0, "type": "chord"},
    {"offset": 2.0, "duration": 1.0, "type": "chord"},
    {"offset": 3.0, "duration": 1.0, "type": "chord"},
]  # 4 notes/bar
```

#### 実装方法

**Option A**: パターンイベントを間引く/追加する

```python
def _render_hand_part(self, ...):
    ...
    pattern_events = pattern_data.get("pattern") or []
    
    # Emotion adjustments
    emotion_adj = section_data.get('_emotion_adjustments', {}).get('piano', {})
    notes_mult = emotion_adj.get('notes_per_bar_multiplier', 1.0)
    
    if notes_mult < 1.0:
        # 音符を間引く
        target_count = int(len(pattern_events) * notes_mult)
        pattern_events = random.sample(pattern_events, target_count)
        pattern_events.sort(key=lambda e: e['offset'])
    elif notes_mult > 1.0:
        # 音符を追加（subdivision）
        # 既存イベントの間に新しいイベントを挿入
        ...
```

**Option B**: パターン選択時に考慮

現在のパターン選択ロジック (line 695-712):
```python
rh_key = piano_params.get("rhythm_key_rh") or piano_params.get("rhythm_key")
lh_key = piano_params.get("rhythm_key_lh") or piano_params.get("rhythm_key")

if not rh_key or not lh_key:
    def_rh, def_lh = self._get_pattern_keys(musical_intent, None)
    rh_key = rh_key or def_rh
    lh_key = lh_key or def_lh
```

**実装**:
```python
def _get_pattern_keys(self, musical_intent, emo_adj):
    ...
    # Emotion multiplierに基づいてパターンを選択
    notes_mult = emo_adj.get('notes_per_bar_multiplier', 1.0)
    
    if notes_mult < 0.8:
        # 音符が少ないパターンを優先
        # e.g., ambient_pad, block_chords
        ...
    elif notes_mult > 1.2:
        # 音符が多いパターンを優先
        # e.g., arpeggio_sixteenths, alberti_bass
        ...
```

**推奨**: Option A（既存パターンを維持しつつ調整可能）

---

## 📝 実装ステップ

### Day 1: velocity_std_multiplier実装

1. [ ] `_render_hand_part()`に emotion adjustments取得コードを追加
2. [ ] velocity計算ロジックに`velocity_std_multiplier`を統合
3. [ ] Unit testを作成
4. [ ] 手動テスト（happy_high vs neutral_medium）

### Day 2: notes_per_bar_multiplier実装

1. [ ] パターンイベント調整ロジックを実装
2. [ ] 音符間引き（mult < 1.0）機能
3. [ ] 音符追加（mult > 1.0）機能（オプション）
4. [ ] Unit testを作成
5. [ ] Integration test

### Day 3: Quality Gate & Documentation

1. [ ] Piano eval実行
2. [ ] Quality Gate検証
3. [ ] A/Bテスト実施
4. [ ] Phase 5.1完了レポート作成
5. [ ] Git commit

---

## ✅ Success Criteria

### 機能要件

- [ ] velocity_std_multiplierが実際のvelocityに反映される
- [ ] notes_per_bar_multiplierが実際の音符数に反映される
- [ ] 既存の機能が壊れていない
- [ ] Emotion profileの違いが明確に聞き取れる

### 品質要件

- [ ] Piano Quality Gateを通過
- [ ] velocity_std: 適切な範囲内
- [ ] notes_per_bar: 適切な範囲内
- [ ] grid_off_std_ms: 既存レベル維持

### テスト要件

- [ ] Unit tests: 2+ tests
- [ ] Integration tests: 2+ tests
- [ ] A/B tests: happy_high vs neutral_medium

---

## 🧪 テスト計画

### Unit Tests

**tests/test_piano_emotion_application.py**:

```python
import pytest
from generator.piano_generator import PianoGenerator
from utils.emotion_loader import load_emotion_mapping

def test_velocity_std_multiplier_applied():
    """velocity_std_multiplierが実際に適用されることを確認"""
    gen = PianoGenerator(...)
    
    # Section data with emotion adjustments
    section_data = {
        "chord_symbol_for_voicing": "C",
        "q_length": 4.0,
        "_emotion_adjustments": {
            "piano": {
                "velocity_std_multiplier": 1.5  # 1.5倍のバラつき
            }
        }
    }
    
    result = gen._render_part(section_data)
    
    # Velocityのバラつきを検証
    velocities = [n.volume.velocity for n in result['piano_rh'].flatten().notes]
    std_dev = np.std(velocities)
    
    # 期待: std_dev ≈ 15 * 1.5 = 22.5
    assert 18 < std_dev < 27  # ±20%の許容範囲

def test_notes_per_bar_multiplier_applied():
    """notes_per_bar_multiplierが音符数に反映されることを確認"""
    gen = PianoGenerator(...)
    
    section_data = {
        "chord_symbol_for_voicing": "C",
        "q_length": 4.0,
        "_emotion_adjustments": {
            "piano": {
                "notes_per_bar_multiplier": 0.5  # 半分の音符数
            }
        }
    }
    
    result = gen._render_part(section_data)
    
    # 音符数を検証
    note_count = len(result['piano_rh'].flatten().notes)
    
    # ベースライン比較（multiplier=1.0の時の音符数）
    baseline_count = 16  # 仮定
    expected_count = baseline_count * 0.5
    
    assert abs(note_count - expected_count) <= 2  # ±2音符の許容
```

### Integration Tests

**tests/test_piano_emotion_integration.py**:

```python
def test_happy_high_vs_neutral_medium():
    """happy_high と neutral_medium の違いを検証"""
    gen = PianoGenerator(...)
    
    # happy_high
    result_happy = gen.compose(
        section_data=base_section,
        section="Chorus",
        emotion_profile="happy_high"
    )
    
    # neutral_medium
    result_neutral = gen.compose(
        section_data=base_section,
        section="Verse",
        emotion_profile="neutral_medium"
    )
    
    # メトリクス比較
    happy_velocities = extract_velocities(result_happy)
    neutral_velocities = extract_velocities(result_neutral)
    
    happy_notes = count_notes(result_happy)
    neutral_notes = count_notes(result_neutral)
    
    # happy_highの方がvelocityのバラつきが大きい
    assert np.std(happy_velocities) > np.std(neutral_velocities)
    
    # happy_highの方が音符が多い
    assert happy_notes > neutral_notes
```

### A/B Tests

**scripts/ab_test_piano_emotions.py**:

```python
def compare_piano_emotions():
    """Piano emotion profilesのA/B comparison"""
    profiles = ["happy_high", "neutral_medium", "calm_low", "energetic_high"]
    
    results = {}
    for profile in profiles:
        samples = []
        for _ in range(10):
            part = generate_piano(section="Chorus", emotion_profile=profile)
            metrics = evaluate_piano(part)
            samples.append(metrics)
        
        results[profile] = {
            'velocity_std_mean': np.mean([s['velocity_std'] for s in samples]),
            'notes_per_bar_mean': np.mean([s['notes_per_bar'] for s in samples]),
            'velocity_mean': np.mean([s['velocity_mean'] for s in samples])
        }
    
    # Statistical significance test
    from scipy import stats
    happy_vs_neutral = stats.ttest_ind(
        [s['velocity_std'] for s in results['happy_high']],
        [s['velocity_std'] for s in results['neutral_medium']]
    )
    
    print(f"happy_high vs neutral_medium: p-value={happy_vs_neutral.pvalue}")
    
    # Report
    generate_report(results)
```

---

## 📊 期待される結果

### Emotion Profile別のメトリクス予測

| Profile | velocity_std_mult | notes_per_bar_mult | 期待velocity_std | 期待notes/bar |
|---------|------------------|-------------------|----------------|--------------|
| happy_high | 1.3 | 1.2 | ~19.5 | ~5.8 |
| happy_medium | 1.1 | 1.0 | ~16.5 | ~4.8 |
| happy_low | 0.9 | 0.8 | ~13.5 | ~3.8 |
| neutral_medium | 1.0 | 1.0 | ~15.0 | ~4.8 |
| calm_low | 0.7 | 0.6 | ~10.5 | ~2.9 |
| energetic_high | 1.5 | 1.5 | ~22.5 | ~7.2 |

**Base values**: velocity_std = 15, notes_per_bar = 4.8 (neutral_medium)

---

## 🚨 リスク & 対策

### Risk 1: Quality Gate違反

**リスク**: velocity_stdやnotes_per_barが範囲外になりQG失敗

**対策**:
- multiplierの範囲を制限 (0.5〜1.5)
- velocity clipping (1〜127)
- notes_per_bar最小値保証

### Risk 2: 既存機能の破壊

**リスク**: 新しいロジックで既存パターンが壊れる

**対策**:
- emotion_profile=Noneの場合は何もしない（後方互換）
- 既存テストの継続実行
- 段階的ロールアウト

### Risk 3: 聴感上の不自然さ

**リスク**: multiplierが極端で音楽的に不自然

**対策**:
- A/Bテストで聴感確認
- multiplier範囲の調整
- emotion_mapping.yamlの微調整

---

## 📚 参考実装

### emotion_loader.py (既存)

```python
def get_emotion_adjustments(
    instrument: str,
    emotion_profile: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """楽器別のemotion adjustmentsを取得"""
    if config is None:
        config = load_emotion_mapping()
    
    profiles = config.get('emotion_profiles', {})
    profile = profiles.get(emotion_profile, {})
    
    instruments = profile.get('instruments', {})
    adjustments = instruments.get(instrument, {})
    
    return adjustments
```

### emotion_mapping.yaml (既存)

```yaml
emotion_profiles:
  happy_high:
    instruments:
      piano:
        velocity_std_multiplier: 1.3
        notes_per_bar_multiplier: 1.2
```

---

## ⏭️ 次のステップ

### Phase 5.1実装完了後

- [ ] Phase 5.2: Guitar Parameter Application開始
- [ ] Guitar strum_consistency_target実装
- [ ] Guitar velocity_boost実装

---

**Phase 5.1 Implementation Plan Complete!** 📋

準備完了。明日から実装開始。

---

**Version**: 1.0  
**Date**: 2025-10-14  
**Status**: Planning Complete
