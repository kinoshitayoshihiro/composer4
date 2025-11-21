# OtobonAI Phase 2.0 Implementation Summary

## 📋 実装完了コンポーネント

### 1. LyricAnchorIndex ✅
**ファイル**: `otobonAI/lyric_index.py`

**機能**:
- lyric_anchors.json（time-based）をbar単位でインデックス化
- phrase_role自動検出（start/mid/end）
- stress_level計算（0.0-1.0）
- vocal有無判定

**テスト結果**（song_004, 120 BPM）:
```
Total bars with anchors: 63
Bar  0: role=mid   stress=0.80 vocal=True
Bar  5: role=start stress=0.80 vocal=True  ← フレーズ開始検出
```

### 2. RulebookEngine 統一インターフェース ✅
**ファイル**: `otobonAI/rulebook_engine.py`

**拡張内容**:
- `query(context, domain)`: 統一contextからdomain別アクション取得
- `list_matched_rules(context, domain)`: マッチしたルールリスト（デバッグ用）
- `_matches_context()`: Phase 2.0 context対応マッチング
  - `emotion.energy_gte`
  - `lyric.phrase_role`
  - `role`（strings/piano/bass）

### 3. EmotionAI v2 ✅
**ファイル**: `otobonAI/emotion_ai_v2.py`

**新機能**:
- `EmotionParams` dataclass
  - energy, tension, brightness, valence
  - velocity_scale, duration_scale, density_scale
  - phrase_role（lyric由来）
  - tags

- `get_params(context)` メソッド
  - Rulebook query統合
  - Lyric phrase_role統合

### 4. GuideToneAI v2 ✅
**ファイル**: `otobonAI/guide_tone_ai_v2.py`

**新機能**:
- `GuideTonePlan` dataclass
  - notes_per_bar
  - preferred_degrees / avoid_degrees
  - register, motion, phrase_shape
  - phrase_role（lyric由来）

- `get_plan(context)` メソッド
  - Rulebook query統合
  - Lyric phrase_role統合
  - phrase_start → uphill + notes+2
  - phrase_end → downhill + notes-1

### 5. Rulebook v2 ルール追加 ✅
**ファイル**: `configs/otobonAI/rulebook.yaml`

**追加ルール**:
```yaml
LYRIC_001: Phrase Start - Uplifting Motion
  - phrase_roles: [start]
  - notes_per_bar: 6, priority_tones: [3rd, 5th, 9th]
  - phrase_shape: uphill

LYRIC_002: Phrase End - Descending Resolution
  - phrase_roles: [end]
  - notes_per_bar: 2, priority_tones: [3rd, root]
  - phrase_shape: downhill

EMOTION_001: High Energy Chorus
  - sections: [chorus], energy_gte: 0.6
  - energy_delta: +0.1, density_delta: +0.2

GUIDETONE_001: Chorus Strings - High Register
  - sections: [chorus], roles: [strings]
  - priority_tones: [9th, 11th, 13th], register: high
```

---

## 🎯 統一Context構造

```python
context = {
    "song_id": "song_004",
    "bar_index": 12,
    "section": "chorus",
    "role": "strings",
    "key_center": "C#m",
    "chord_symbol": "Emaj7(9)",
    "tempo_bpm": 120,
    
    # Emotion base（emotion_profile.jsonから）
    "emotion": {
        "energy": 0.52,
        "tension": 0.61,
    },
    
    # Lyric info（LyricAnchorIndexから）
    "lyric": {
        "has_anchor": True,
        "phrase_role": "start",  # "start" | "mid" | "end" | "none"
        "stress_level": 0.8,     # 0.0-1.0
        "is_silent": False,
    },
    
    # Slots（bars_with_slots.parquetから）
    "slots": {
        "has_fill": True,
        "has_riff": False,
    },
}
```

---

## 📝 次のステップ：generate_strings_plan_v2.py Phase 2.0統合

### 修正箇所

#### 1. Import追加
```python
from otobonAI.lyric_index import LyricAnchorIndex
from otobonAI.emotion_ai_v2 import EmotionAI, EmotionParams
from otobonAI.guide_tone_ai_v2 import GuideToneAI, GuideTonePlan
from otobonAI.rulebook_engine import Rulebook
```

#### 2. main()で初期化
```python
def main(...):
    # Rulebook
    rulebook = Rulebook.load("configs/otobonAI/rulebook.yaml")
    
    # Lyric Index
    lyric_index = LyricAnchorIndex.from_file(
        "analysis/lyric_anchors.json",
        tempo_bpm=120
    )
    
    # EmotionAI v2
    emotion_ai = EmotionAI.from_files(
        "analysis/emotion_profile.json",
        "configs/otobonAI/rulebook.yaml"
    )
    
    # GuideToneAI v2
    guide_ai = GuideToneAI.from_files(
        "analysis/guide_tone_hints.json",
        "configs/otobonAI/rulebook.yaml"
    )
```

#### 3. Bar loop内でcontext構築
```python
for bar_idx in range(total_bars):
    # Lyric info取得
    lyric_info = lyric_index.get_bar_info(bar_idx)
    
    # Context構築
    context = {
        "bar_index": bar_idx,
        "section": section_for_bar(bar_idx, sections),
        "role": "strings",
        "key_center": key_center,
        "chord_symbol": chord["symbol"],
        "tempo_bpm": 120,
        "lyric": lyric_info,
        "slots": {
            "has_fill": bars_df.loc[bar_idx, "fill_slot"] > 0.0,
            "has_riff": bars_df.loc[bar_idx, "riff_slot"] > 0.0,
        },
    }
    
    # AI取得
    emo = emotion_ai.get_params(context)
    guide = guide_ai.get_plan(context)
    
    # Density/Notes調整
    density_scale = emo.density_scale
    notes_per_bar = guide.notes_per_bar
```

#### 4. make_countermelody()にphrase_shape統合
```python
def make_countermelody(..., guide: GuideTonePlan):
    # Phrase shape適用
    if guide.phrase_shape == "uphill":
        # 上昇パターン
        melody = sorted(chord_tones)[:guide.notes_per_bar]
    elif guide.phrase_shape == "downhill":
        # 下降パターン
        melody = sorted(chord_tones, reverse=True)[:guide.notes_per_bar]
    
    # Preferred degrees filter
    filtered_tones = [
        t for t in all_tones
        if (t % 12) in guide.preferred_degrees
    ]
```

---

## 🎵 期待される効果

### Before（v1.5）
- 和声的に正しいが平板なメロディ
- セクション間の変化が少ない
- フレーズ構造が不明確

### After（v2.0）
- **Phrase start**: 上昇モーション、高エネルギー
  - `notes_per_bar`: 4 → 6（+2）
  - `phrase_shape`: uphill
  - `preferred_degrees`: [3, 5, 9]（明るい音）

- **Phrase mid**: 安定
  - `notes_per_bar`: 4（デフォルト）
  - `phrase_shape`: None（stepモーション）

- **Phrase end**: 下降解決
  - `notes_per_bar`: 4 → 2（-2）
  - `phrase_shape`: downhill
  - `preferred_degrees`: [1, 3]（安定音）

- **Chorus高エネルギー**:
  - `velocity_scale`: 1.2（+20%）
  - `density_scale`: 1.5（+50%）
  - `register`: high（1オクターブ上）

---

## ✅ 完了状況

| コンポーネント | ステータス | ファイル |
|---------------|-----------|---------|
| LyricAnchorIndex | ✅ 完了 | `otobonAI/lyric_index.py` |
| RulebookEngine v2 | ✅ 完了 | `otobonAI/rulebook_engine.py` |
| EmotionAI v2 | ✅ 完了 | `otobonAI/emotion_ai_v2.py` |
| GuideToneAI v2 | ✅ 完了 | `otobonAI/guide_tone_ai_v2.py` |
| Rulebook Phase 2.0ルール | ✅ 完了 | `configs/otobonAI/rulebook.yaml` |
| generate_strings_plan_v2.py統合 | 🔄 準備完了 | 次のステップ |

---

## 🚀 次の実行推奨コマンド

### オプション1: Phase 2.0テスト統合スクリプト作成
新しいスクリプト `generate_strings_plan_v3.py` を作成して、Phase 2.0統合を実装。

### オプション2: 既存v2に段階的統合
`generate_strings_plan_v2.py` を直接修正してPhase 2.0対応。

---

**推奨**: オプション1（新規スクリプト）で最初にテストし、動作確認後にv2をマージ。
