# Emotion Mapping Guide
# セクション&Emotion Profile 統合ガイド

**Version**: 1.0  
**Date**: 2025-10-14  
**Phase**: 4.9 - v1.0 Release Preparation

---

## 概要

Phase 4.9で導入されたEmotion Mappingシステムは、楽曲のセクション(Intro, Verse, Chorusなど)と感情プロファイル(happy_high, melancholic_mediumなど)に基づいて、各楽器の生成パラメータを自動調整します。

### 主要機能

- **10 Emotion Profiles**: 感情の強度とムードの組み合わせ
- **7 Section Types**: 楽曲構造に応じたデフォルトemotion
- **5 Instrument Adjustments**: 楽器ごとの最適化パラメータ
- **Transition Rules**: セクション間の移行ルール

---

## 基本的な使い方

### 1. シンプルな生成例

```python
from generator import PianoGenerator

# Pianoジェネレータを作成
piano = PianoGenerator(
    global_settings={},
    global_tempo=120,
    global_time_signature="4/4",
)

# セクションとemotion profileを指定して生成
section_data = {
    "section_name": "Chorus",
    "q_length": 16.0,
    "chord_symbol_for_voicing": "C",
}

# Chorus → デフォルトは happy_high
part = piano.compose(
    section_data=section_data,
    section="Chorus"  # デフォルトemotion (happy_high) が適用される
)
```

### 2. Emotion Profileを明示的に指定

```python
# Verseで melancholic_medium を使用
part = piano.compose(
    section_data=section_data,
    section="Verse",
    emotion_profile="melancholic_medium"  # デフォルトを上書き
)
```

### 3. 全楽器での使用例

```python
from generator import (
    PianoGenerator,
    GuitarGenerator,
    BassGenerator,
    StringsGenerator,
    DrumGenerator
)

# 共通設定
section_data = {
    "section_name": "Bridge",
    "q_length": 8.0,
    "chord_symbol_for_voicing": "Am",
}

# Piano - Bridgeのデフォルト (melancholic_medium)
piano_part = piano_gen.compose(
    section_data=section_data,
    section="Bridge"
)

# Guitar - 同じsectionで異なるemotion
guitar_part = guitar_gen.compose(
    section_data=section_data,
    section="Bridge",
    emotion_profile="calm_low"  # より穏やかに
)

# Bass - セクションデフォルトを使用
bass_part = bass_gen.compose(
    section_data=section_data,
    section="Bridge"
)

# Strings - 高いintensity
strings_parts = strings_gen.compose(
    section_data=section_data,
    section="Bridge",
    emotion_profile="sad_high"  # 悲しみを強調
)

# Drums - energetic
drum_part = drum_gen.compose(
    section_data=section_data,
    section="Bridge",
    emotion_profile="energetic_medium"  # リズムを活発に
)
```

---

## Emotion Profiles

### 10プロファイル一覧

| Profile | Intensity | Mood | Tension | Dynamics | 用途 |
|---------|-----------|------|---------|----------|------|
| `happy_low` | low | happy | low | soft | 軽快な導入、穏やかな明るさ |
| `happy_medium` | medium | happy | medium | moderate | 標準的な明るさ、Verse |
| `happy_high` | high | happy | high | loud | 盛り上がり、Chorus |
| `sad_low` | low | sad | low | soft | 静かな悲しみ、Outro |
| `melancholic_medium` | medium | melancholic | medium | moderate | 物憂げ、Bridge |
| `sad_high` | high | sad | high | loud | 激しい悲しみ、ドラマチック |
| `energetic_medium` | medium | energetic | medium | moderate | 活動的、Pre-Chorus |
| `energetic_high` | high | energetic | high | loud | 非常に活動的、サビ |
| `calm_low` | low | calm | low | soft | 穏やか、Intro/Outro |
| `neutral_medium` | medium | neutral | medium | moderate | 中立、Verse |

### Intensity Levels

- **low**: 控えめ、静的、疎なテクスチャ
- **medium**: 標準、バランス
- **high**: 激しい、密なテクスチャ、ダイナミック

### Mood Types

- **happy**: 明るい、陽気、軽快
- **sad**: 悲しい、深刻、重い
- **melancholic**: 物憂げ、感傷的
- **energetic**: 活動的、躍動的
- **calm**: 穏やか、平和
- **neutral**: 中立、標準

---

## Section-to-Emotion Mapping

### 7セクションタイプのデフォルトemotion

| Section | Default Emotion | 説明 | Alternative Emotions |
|---------|----------------|------|----------------------|
| **Intro** | `calm_low` | 控えめな導入 | neutral_medium, happy_low |
| **Verse** | `neutral_medium` | ストーリーを語る | melancholic_medium, happy_medium, calm_low |
| **Pre-Chorus** | `energetic_medium` | 盛り上がりへの準備 | happy_medium, melancholic_medium |
| **Chorus** | `happy_high` | 曲の核心、最も印象的 | energetic_high, sad_high, happy_medium |
| **Bridge** | `melancholic_medium` | 変化と対比 | neutral_medium, calm_low, energetic_medium |
| **Outro** | `calm_low` | 終わりに向けて | melancholic_medium, happy_low |
| **Fill** | `energetic_medium` | セクション間の移行 | energetic_high, neutral_medium |

### セクション選択ガイドライン

```python
# Intro - 穏やかに始める
section="Intro"
# → デフォルト: calm_low
# → 楽器: 疎なテクスチャ、低いvelocity

# Verse - ストーリー展開
section="Verse"
# → デフォルト: neutral_medium
# → 楽器: バランスの取れたダイナミクス

# Pre-Chorus - 盛り上がりへ
section="Pre-Chorus"
# → デフォルト: energetic_medium
# → 楽器: 密度増加、tension上昇

# Chorus - クライマックス
section="Chorus"
# → デフォルト: happy_high
# → 楽器: 最大のdynamics、密なテクスチャ

# Bridge - 対比・変化
section="Bridge"
# → デフォルト: melancholic_medium
# → 楽器: 異なるムード、テクスチャ変化

# Outro - 終わり
section="Outro"
# → デフォルト: calm_low
# → 楽器: 疎なテクスチャ、fade out
```

---

## Instrument-Specific Adjustments

各楽器は、emotion profileに応じて異なるパラメータが調整されます。

### Piano

| Emotion | velocity_std_multiplier | notes_per_bar_multiplier | 効果 |
|---------|------------------------|--------------------------|------|
| `happy_high` | 1.2 | 1.1 | 振れ幅大、密度高 |
| `melancholic_medium` | 0.9 | 0.9 | 控えめ、疎め |
| `calm_low` | 0.7 | 0.8 | 穏やか、疎め |

**適用例**:

```python
# 基準パラメータ
base_params = {
    "velocity_std": 15,  # velocity標準偏差
    "notes_per_bar": 8   # 1小節あたりの音符数
}

# happy_high適用後
# velocity_std: 15 × 1.2 = 18
# notes_per_bar: 8 × 1.1 = 8.8
```

### Guitar

| Emotion | strum_consistency_target | velocity_boost | 効果 |
|---------|-------------------------|----------------|------|
| `happy_high` | 0.80 | +10 | 高い一貫性、明るい |
| `melancholic_medium` | 0.75 | 0 | 標準的 |
| `calm_low` | 0.70 | -10 | アルペジオ的、穏やか |

**strum_consistency_target**: ストローク一貫性 (0.0-1.0)  
**velocity_boost**: velocity加算値 (-20 ~ +20)

### Bass

| Emotion | notes_per_bar_multiplier | root_emphasis | 効果 |
|---------|--------------------------|---------------|------|
| `energetic_high` | 1.2 | 0.75 | Walking bass的、活発 |
| `melancholic_medium` | 0.9 | 0.80 | 疎め |
| `calm_low` | 0.7 | 0.85 | 非常に疎め、root強調 |

**root_emphasis**: root音強調率 (0.0-1.0、高いほどroot重視)

### Strings

| Emotion | legato_rate_target | chord_spread_multiplier | 効果 |
|---------|-------------------|------------------------|------|
| `happy_high` | 0.65 | 1.1 | やや高めlegato、広い和声 |
| `melancholic_medium` | 0.70 | 0.9 | 高めlegato、狭めの和声 |
| `calm_low` | 0.75 | 0.8 | 非常に高いlegato、狭い和声 |

**legato_rate_target**: レガート率目標 (0.0-1.0)  
**chord_spread_multiplier**: 和音音域の倍率

### Drums

| Emotion | hihat_density_multiplier | kick_emphasis | velocity_boost | 効果 |
|---------|--------------------------|---------------|----------------|------|
| `energetic_high` | 1.2 | 1.1 | +10 | 密度高、kick強調、明るい |
| `melancholic_medium` | 0.9 | 1.0 | 0 | 疎め、標準kick |
| `calm_low` | 0.7 | 0.9 | -10 | 非常に疎め、穏やか |

**hihat_density_multiplier**: ハイハット密度倍率  
**kick_emphasis**: キック強調率  
**velocity_boost**: velocity加算値

---

## Transition Rules

セクション間の移行時の境界処理ルール。

### 基本ルール

```yaml
basic:
  max_overlap_ms: 50  # 最大50msまでの重複を許容
  min_gap_ms: 0      # 最小ギャップ（0=直接接続OK）
```

### 特別な移行ルール (4種)

| 移行 | ルール | 説明 | 効果 |
|------|--------|------|------|
| **Verse → Pre-Chorus** | `min_gap: 100ms` | Verseの余韻を残す | 100ms以上の空白 |
| **Pre-Chorus → Chorus** | `max_overlap: 100ms` | シームレスな移行 | 100msまで重複OK |
| **Chorus → Verse** | `min_gap: 200ms` | Chorusの余韻を残す | 200ms以上の空白 |
| **Bridge → Chorus** | `min_gap: 150ms` | 対比を保つ | 150ms以上の空白 |

**使用例**:

```python
from utils.emotion_loader import get_transition_rule

# Pre-Chorus から Chorus への移行ルール取得
rule = get_transition_rule("Pre-Chorus", "Chorus")
print(rule)
# {'max_overlap_ms': 100, 'min_gap_ms': 0, 'description': 'シームレスな移行'}

# 生成時に境界チェック
# (将来のバージョンで自動適用予定)
```

---

## Section Length Constraints

各セクションの推奨長さ。

| Section | Min Bars | Max Bars | 説明 |
|---------|----------|----------|------|
| Intro | 2 | 8 | 短めの導入 |
| Verse | 4 | 16 | ストーリー展開 |
| Pre-Chorus | 2 | 8 | 橋渡し |
| Chorus | 4 | 16 | 核心部分 |
| Bridge | 4 | 16 | 対比セクション |
| Outro | 2 | 8 | 終わり |
| Fill | 1 | 2 | ドラムフィル |

**検証例**:

```python
from utils.emotion_loader import validate_section_constraints

# Intro 4小節 - ✅ OK
is_valid = validate_section_constraints("Intro", 4)
# True

# Intro 20小節 - ❌ Too long
is_valid = validate_section_constraints("Intro", 20)
# False
```

---

## 高度な使い方

### 1. カスタムパラメータ調整

```python
from utils.emotion_loader import get_emotion_adjustments, apply_adjustments_to_params

# ベースパラメータ
base_params = {
    "velocity_std": 15,
    "notes_per_bar": 8,
    "legato_rate": 0.60
}

# Emotion調整を取得
adjustments = get_emotion_adjustments("piano", "happy_high")

# 調整を適用
final_params = apply_adjustments_to_params(base_params, adjustments)
print(final_params)
# {
#   'velocity_std': 18.0,  # 15 × 1.2
#   'notes_per_bar': 8.8,  # 8 × 1.1
#   'legato_rate': 0.60    # 変更なし
# }
```

### 2. ワンライナーで完全パラメータ取得

```python
from utils.emotion_loader import get_generation_params

# 楽器、セクション、emotionを指定して完全パラメータ取得
params = get_generation_params(
    instrument="guitar",
    section="Chorus",
    # emotion_profile省略時はChorusのデフォルト (happy_high) を使用
)
print(params)
# {'strum_consistency_target': 0.80, 'velocity_boost': 10}

# ベースパラメータに適用
base = {"strum_consistency": 0.70, "velocity": 80}
final = get_generation_params(
    "guitar",
    "Chorus",
    base_params=base
)
print(final)
# {
#   'strum_consistency': 0.80,  # target値で上書き
#   'velocity': 90              # 80 + 10 boost
# }
```

### 3. Emotion Profile情報取得

```python
from utils.emotion_loader import get_emotion_profile_info

# Profileの詳細情報取得
info = get_emotion_profile_info("happy_high")
print(info)
# {
#   'intensity': 'high',
#   'mood': 'happy',
#   'tension': 'high',
#   'dynamics': 'loud'
# }
```

### 4. セクションのalternative emotions取得

```python
from utils.emotion_loader import get_section_alternative_emotions

# Chorusで使用可能な代替emotions
alternatives = get_section_alternative_emotions("Chorus")
print(alternatives)
# ['energetic_high', 'sad_high', 'happy_medium']
```

---

## トラブルシューティング

### Q1: emotion_mapping.yamlが見つからない

**エラー**:
```
FileNotFoundError: Emotion mapping config not found: /path/to/config/emotion_mapping.yaml
```

**解決策**:
```bash
# emotion_mapping.yamlが存在するか確認
ls config/emotion_mapping.yaml

# 存在しない場合はPhase 4.7のファイルを確認
git status
```

### Q2: 不明なemotion profileエラー

**エラー**:
```
ValueError: Unknown emotion profile: happy_ultra_high
```

**解決策**:
```python
# 利用可能なemotion profiles一覧を確認
from utils.emotion_loader import load_emotion_mapping

config = load_emotion_mapping()
print(list(config['emotion_profiles'].keys()))
# ['happy_low', 'happy_medium', 'happy_high', ...]
```

### Q3: 不明なsectionエラー

**エラー**:
```
ValueError: Unknown section: Interlude
```

**解決策**:
```python
# 利用可能なsection types一覧を確認
config = load_emotion_mapping()
print(list(config['section_emotion_mapping'].keys()))
# ['Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Bridge', 'Outro', 'Fill']

# カスタムsectionの場合はVerse等で代用
section="Verse"  # Interludeの代わり
```

### Q4: Adjustmentsが適用されない

**症状**: emotion_profileを指定しても音が変わらない

**原因**: 現在のバージョン(v1.0)では、調整値は`section_data['_emotion_adjustments']`に格納されますが、実際の生成パラメータへの適用は各Generatorの内部実装に依存します。

**対策**:
```python
# section_dataに格納された調整値を確認
section_data = {"section_name": "Chorus", ...}
part = piano.compose(
    section_data=section_data,
    section="Chorus",
    emotion_profile="happy_high"
)

# 調整値が格納されているか確認
if "_emotion_adjustments" in section_data:
    print(section_data["_emotion_adjustments"]["piano"])
    # {'velocity_std_multiplier': 1.2, 'notes_per_bar_multiplier': 1.1}
```

**将来の改善**: Phase 5で実際のパラメータ適用を実装予定

---

## ベストプラクティス

### 1. セクションデフォルトを活用

```python
# Good: セクション指定のみ（デフォルトemotion使用）
part = piano.compose(section_data=section_data, section="Chorus")

# OK: 特定のムードが必要な場合のみ明示的に指定
part = piano.compose(
    section_data=section_data,
    section="Chorus",
    emotion_profile="sad_high"  # 悲しいChorusの場合
)
```

### 2. 楽曲全体で一貫性を保つ

```python
# 全楽器で同じsectionを使用
common_section = "Chorus"

piano_part = piano_gen.compose(section_data=sd, section=common_section)
guitar_part = guitar_gen.compose(section_data=sd, section=common_section)
bass_part = bass_gen.compose(section_data=sd, section=common_section)
```

### 3. Intensityを段階的に変化

```python
# Intro: low intensity
intro_part = piano.compose(sd_intro, section="Intro")  # calm_low

# Verse: medium intensity
verse_part = piano.compose(sd_verse, section="Verse")  # neutral_medium

# Pre-Chorus: medium-high intensity
prechorus_part = piano.compose(sd_pc, section="Pre-Chorus")  # energetic_medium

# Chorus: high intensity
chorus_part = piano.compose(sd_chorus, section="Chorus")  # happy_high
```

### 4. 対比を作る

```python
# Verse: neutral
verse = piano.compose(sd, section="Verse")

# Bridge: 対比的なemotion
bridge = piano.compose(sd, section="Bridge")  # melancholic_medium (デフォルト)

# または明示的に対比を作る
bridge = piano.compose(sd, section="Bridge", emotion_profile="calm_low")
```

---

## 完全な使用例

### フルソング生成

```python
from generator import PianoGenerator, GuitarGenerator, BassGenerator

# Generator初期化
piano = PianoGenerator(global_settings={}, global_tempo=120)
guitar = GuitarGenerator(global_settings={}, global_tempo=120)
bass = BassGenerator(global_settings={}, global_tempo=120)

# セクション定義
sections = [
    {"name": "Intro", "section": "Intro", "bars": 4, "chord": "C"},
    {"name": "Verse1", "section": "Verse", "bars": 8, "chord": "C"},
    {"name": "PreChorus1", "section": "Pre-Chorus", "bars": 4, "chord": "F"},
    {"name": "Chorus1", "section": "Chorus", "bars": 8, "chord": "G"},
    {"name": "Verse2", "section": "Verse", "bars": 8, "chord": "C"},
    {"name": "Bridge", "section": "Bridge", "bars": 8, "chord": "Am"},
    {"name": "Chorus2", "section": "Chorus", "bars": 8, "chord": "G"},
    {"name": "Outro", "section": "Outro", "bars": 4, "chord": "C"},
]

# 各セクションを生成
song_parts = {"piano": [], "guitar": [], "bass": []}

for sec in sections:
    section_data = {
        "section_name": sec["name"],
        "q_length": sec["bars"] * 4.0,
        "chord_symbol_for_voicing": sec["chord"],
    }
    
    # Piano
    piano_part = piano.compose(
        section_data=section_data,
        section=sec["section"]
    )
    song_parts["piano"].append(piano_part)
    
    # Guitar
    guitar_part = guitar.compose(
        section_data=section_data,
        section=sec["section"]
    )
    song_parts["guitar"].append(guitar_part)
    
    # Bass
    bass_part = bass.compose(
        section_data=section_data,
        section=sec["section"]
    )
    song_parts["bass"].append(bass_part)

# 楽曲全体を結合
from music21 import stream

full_score = stream.Score()
for instrument_name, parts_list in song_parts.items():
    instrument_part = stream.Part()
    offset = 0.0
    for p in parts_list:
        for element in p:
            instrument_part.insert(offset, element)
        offset += p.duration.quarterLength
    full_score.insert(0, instrument_part)

# MIDIエクスポート
full_score.write('midi', fp='my_song_with_emotions.mid')
```

---

## まとめ

Emotion Mappingシステムは、楽曲のセクション構造と感情表現を自動的に楽器パラメータに反映します。

**主要な利点**:

- ✅ **簡潔なAPI**: section="Chorus" だけで適切なemotionが適用
- ✅ **一貫性**: 全楽器で統一されたemotion表現
- ✅ **柔軟性**: デフォルトを使いつつ、必要に応じてカスタマイズ
- ✅ **YAML駆動**: コード変更不要で設定調整可能

**v1.0での制限**:

- 調整値は格納されるが、実際の適用は各Generatorの実装に依存
- 一部のパラメータは将来のバージョンで完全適用予定

**次のステップ**:

- Phase 5: パラメータ適用の完全実装
- A/Bテスト: emotion profileの効果検証
- ユーザーフィードバック: 実際の楽曲制作での改善点収集

---

**参考リンク**:

- [emotion_mapping.yaml](../config/emotion_mapping.yaml): 完全な設定ファイル
- [utils/emotion_loader.py](../utils/emotion_loader.py): ヘルパー関数実装
- [PHASE_4_7_COMPLETE.md](./PHASE_4_7_COMPLETE.md): Phase 4.7レポート
- [PHASE_4_9_COMPLETE.md](./PHASE_4_9_COMPLETE.md): Phase 4.9レポート

---

**Version History**:

- v1.0 (2025-10-14): 初版リリース、全5楽器対応
