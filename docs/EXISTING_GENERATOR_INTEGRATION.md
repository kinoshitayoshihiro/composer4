# 既存Generator統合計画

## 🔍 現状確認

### 既存Generator一覧（`/generator`ディレクトリ）

| Generator | ファイル | 状態 | 機能 |
|-----------|---------|------|------|
| **BasePartGenerator** | `base_part_generator.py` (714 lines) | ✅ 完成 | 全Generator基底クラス、Control適用（CC11/CC64/Vibrato） |
| **PianoGenerator** | `piano_generator.py` (986 lines) | ✅ 完成 | RH/LH pattern、Emotion→Pattern mapping、ML velocity model対応 |
| **BassGenerator** | `bass_generator.py` (2,487 lines) | ✅ 完成 | Walking/Pick/Slap/Fingerstyle、Kick同期、Approach note |
| **GuitarGenerator** | `guitar_generator.py` | ✅ 完成 | Strum/Arpeggio/Fingerpicking、Guitar controls |
| **StringsGenerator** | `strings_generator.py` | ✅ 完成 | Legato/Staccato/Spiccato、Bowing expression |
| **DrumGenerator** | `drum_generator.py` | ✅ 完成 | Pattern-based、LAMDA統合可能 |
| **DrumAdapter** | `drum/adapter.py` (143 lines) | ✅ 完成 | Stage3 v1.1パイプライン接続 |

### サポートモジュール

| モジュール | 機能 |
|-----------|------|
| `arranger.py` | セクション構成・楽器配置 |
| `chord_voicer.py` | コードボイシング |
| `articulation.py` | Articulation適用 |
| `melody_generator.py` | Melody生成 |
| `vocal_generator.py` | Vocal生成 |
| `obligato_generator.py` | オブリガート生成 |
| `riff_generator.py` | Riff生成 |

### Utilities統合

| Utility | 機能 |
|---------|------|
| `utilities/humanizer.py` | Humanization (timing/velocity variation) |
| `utilities/groove_profile.py` | Groove profile (Kick同期) |
| `utilities/bass_transformer.py` | Bass transformer (ML model) |
| `utilities/controls_bundle.py` | Piano/Bass/Drum controls |
| `utilities/guitar_controls.py` | Guitar controls |
| `utilities/duv_apply.py` | DUV適用 |
| `utilities/emotion_profile_loader.py` | Emotion profile読み込み |

---

## ❌ 私の新規実装は不要

作成してしまった以下のファイルは**既存実装と重複**：

- ❌ `generators/base.py` → 既存: `generator/base_part_generator.py`
- ❌ `generators/piano.py` → 既存: `generator/piano_generator.py`
- ❌ `tests/test_piano_generator.py` → 既存実装をテストすべき

**理由:**
- 既存Generatorは**986-2,487行**の完全実装
- Emotion mapping、ML model、Controls、Humanization完備
- Stage2データ（3,559 files）を活用する設計ではない（pattern-based）

---

## ✅ 正しいアプローチ: 既存Generator + Stage2データ統合

### 統合戦略

**既存Generator（Pattern-based）+ Stage2データ（Learning-based）のハイブリッド**

```
既存Generator（ルールベース）
    ↓
    Pattern selection（Emotion → Pattern）
    ↓
    ML enhancement（Stage2データで学習したモデル）
    ↓
    Humanization（既存utilities活用）
```

---

## 🎯 Phase 1: Piano Generator強化（既存活用）

### 1.1 既存PianoGeneratorの機能確認

```python
# generator/piano_generator.py の主要機能

class PianoGenerator(BasePartGenerator):
    # Emotion → Pattern mapping
    EMO_TO_BUCKET_PIANO = {
        "quiet_pain": "calm",
        "emotional_realization": "groove",
        "love_and_resolution": "energetic",
    }
    
    # Pattern library
    BUCKET_TO_PATTERN_PIANO = {
        ("calm", "low"): ("piano_rh_ambient_pad", "piano_lh_roots_whole"),
        ("groove", "medium"): ("piano_rh_syncopated_chords_pop", "piano_lh_octaves_quarters"),
        ("energetic", "high"): ("piano_rh_arpeggio_sixteenths_up_down", "piano_lh_alberti_bass_eighths"),
    }
    
    # ML velocity model対応
    def __init__(self, ml_velocity_model_path=None, velocity_model=None):
        self.velocity_model = velocity_model
    
    # Compose method
    def compose(self, section_data: dict) -> stream.Part:
        # 1. Pattern選択
        # 2. RH/LH生成
        # 3. Velocity適用
        # 4. Humanization
        pass
```

### 1.2 Stage2データ統合ポイント

**既存: Pattern-based**
```python
# 現状: 固定パターンライブラリ
pattern = BUCKET_TO_PATTERN_PIANO[("energetic", "high")]
# → ("piano_rh_arpeggio_sixteenths_up_down", "piano_lh_alberti_bass_eighths")
```

**強化: Stage2 Learning-based**
```python
# 提案: Stage2データから学習したパターン推薦
from ml.piano_pattern_recommender import PianoPatternRecommender

recommender = PianoPatternRecommender(
    model_path="models/piano_pattern_recommender.pt",
    stage2_data="output/pop909/stage2/melody+chords"  # 554 files
)

# Emotion + Chord progression → 最適Pattern推薦
recommended_pattern = recommender.recommend(
    emotion="joy",
    chord_progression=["C", "Am", "F", "G"],
    tempo=120,
)
# → Stage2データから類似パターンを検索・生成
```

### 1.3 具体的な統合実装

#### Step 1: Stage2データからPattern抽出

```python
# scripts/extract_piano_patterns_from_stage2.py

import pickle
from pathlib import Path
import pretty_midi as pm

def extract_patterns_from_stage2():
    """Stage2 POP909データからPianoパターンを抽出"""
    
    stage2_dir = Path("output/pop909/clean")
    melody_files = list((stage2_dir / "melody").glob("*.mid"))
    chords_files = list((stage2_dir / "chords").glob("*.mid"))
    
    patterns = []
    
    for melody_file in melody_files:
        # 対応するchordsファイル検索
        chords_file = stage2_dir / "chords" / melody_file.name.replace("v1", "v2")
        
        if not chords_file.exists():
            continue
        
        # MIDI読み込み
        melody_midi = pm.PrettyMIDI(str(melody_file))
        chords_midi = pm.PrettyMIDI(str(chords_file))
        
        # Pattern特徴抽出
        pattern = {
            "melody_notes": extract_note_sequence(melody_midi),
            "chords_notes": extract_note_sequence(chords_midi),
            "tempo": melody_midi.estimate_tempo(),
            "duration": melody_midi.get_end_time(),
            "metrics": load_stage2_metrics(melody_file),  # Stage2スコア
        }
        
        patterns.append(pattern)
    
    # Pickle保存
    with open("data/piano_patterns_stage2.pkl", "wb") as f:
        pickle.dump(patterns, f)
    
    print(f"Extracted {len(patterns)} patterns from Stage2 data")
```

#### Step 2: Pattern Recommender実装

```python
# ml/piano_pattern_recommender.py

import pickle
from typing import List, Dict
import numpy as np

class PianoPatternRecommender:
    """Stage2データからPianoパターンを推薦"""
    
    def __init__(self, patterns_path: str):
        with open(patterns_path, "rb") as f:
            self.patterns = pickle.load(f)
    
    def recommend(
        self,
        emotion: str,
        chord_progression: List[str],
        tempo: float,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        類似パターンを推薦
        
        Args:
            emotion: 感情（joy/sorrow/tension等）
            chord_progression: コード進行
            tempo: テンポ
            top_k: 推薦数
        
        Returns:
            類似パターンのリスト（スコア降順）
        """
        # 1. Tempo範囲でフィルタ
        tempo_candidates = [
            p for p in self.patterns
            if abs(p["tempo"] - tempo) < 20
        ]
        
        # 2. Chord progression類似度計算
        scores = []
        for pattern in tempo_candidates:
            # Chord progressionの類似度（簡易版）
            similarity = self._chord_similarity(
                chord_progression,
                pattern.get("chord_progression", [])
            )
            
            # Stage2スコアも考慮
            quality_score = pattern["metrics"]["score"]
            
            # 総合スコア
            total_score = similarity * 0.7 + quality_score * 0.3
            
            scores.append((pattern, total_score))
        
        # 3. Top-K選択
        scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, s in scores[:top_k]]
    
    def _chord_similarity(self, chords1: List[str], chords2: List[str]) -> float:
        """コード進行の類似度（0.0-1.0）"""
        if not chords1 or not chords2:
            return 0.0
        
        # 簡易: 共通コード数 / 全コード数
        common = set(chords1) & set(chords2)
        total = set(chords1) | set(chords2)
        
        return len(common) / len(total) if total else 0.0
```

#### Step 3: 既存PianoGeneratorに統合

```python
# generator/piano_generator.py に追加

from ml.piano_pattern_recommender import PianoPatternRecommender

class PianoGenerator(BasePartGenerator):
    def __init__(self, *args, use_stage2_patterns=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Stage2パターン推薦システム
        if use_stage2_patterns:
            self.pattern_recommender = PianoPatternRecommender(
                patterns_path="data/piano_patterns_stage2.pkl"
            )
        else:
            self.pattern_recommender = None
    
    def compose(self, section_data: dict) -> stream.Part:
        """既存composeメソッドを拡張"""
        
        # 既存のEmotion → Pattern mapping
        emotion = section_data.get("emotion", "default")
        bucket = EMO_TO_BUCKET_PIANO.get(emotion, "default")
        
        # Stage2パターン推薦を試行
        if self.pattern_recommender:
            recommended = self.pattern_recommender.recommend(
                emotion=emotion,
                chord_progression=section_data.get("chord_progression", []),
                tempo=section_data.get("tempo", 120),
                top_k=1,
            )
            
            if recommended:
                # Stage2パターンを使用
                return self._apply_stage2_pattern(recommended[0], section_data)
        
        # Fallback: 既存のPattern-based生成
        default_pattern = BUCKET_TO_PATTERN_PIANO.get(
            (bucket, "medium"),
            ("piano_rh_block_chords_quarters", "piano_lh_roots_whole")
        )
        return self._apply_default_pattern(default_pattern, section_data)
```

---

## 🎯 Phase 2: Bass Generator強化

### 既存BassGeneratorの機能

```python
# generator/bass_generator.py の主要機能

class BassGenerator(BasePartGenerator):
    # Technique selection
    def select_technique(self, style: str, tempo: float):
        if style == "jazz":
            return "walking"
        elif tempo > 140:
            return "pick"
        elif "funk" in style:
            return "slap"
        else:
            return "fingerstyle"
    
    # Kick drum同期
    def sync_with_kick(self, bass_notes, kick_pattern):
        # Bass root noteをkickタイミングに同期
        pass
    
    # Approach note生成
    def add_approach_notes(self, bass_line, scale):
        # Chromatic/Diatonic approach
        pass
```

### Stage2統合ポイント

**既存: ルールベースTechnique選択**
```python
technique = self.select_technique(style="pop", tempo=120)
# → "fingerstyle"
```

**強化: Stage2データから学習**
```python
# SLAKH Bass 584 files (100% pass, 平均76.9%)から学習
from ml.bass_technique_classifier import BassTechniqueClassifier

classifier = BassTechniqueClassifier(
    model_path="models/bass_technique.pt",
    stage2_data="output/slakh/stage2/bass"
)

# Chord + Tempo → 最適Technique推薦
technique = classifier.predict(
    chord_progression=["C", "Am", "F", "G"],
    tempo=120,
    style="pop",
)
# → Stage2データのwalking/pick/slap/fingerstyle分布から推薦
```

---

## 🎯 実装スケジュール（修正版）

### Week 1-2: Piano Generator強化
- [x] 既存PianoGenerator機能確認 ✅
- [ ] Stage2データからPattern抽出スクリプト作成
- [ ] PianoPatternRecommender実装
- [ ] 既存PianoGeneratorに統合
- [ ] テスト実行（既存 vs Stage2強化版）

### Week 3-4: Bass Generator強化
- [ ] 既存BassGenerator機能確認
- [ ] Stage2データからTechnique分類モデル学習
- [ ] BassTechniqueClassifier実装
- [ ] 既存BassGeneratorに統合
- [ ] Kick同期テスト

### Week 5-6: Guitar/Strings Generator強化
- [ ] 既存Generator機能確認
- [ ] Stage2データ統合
- [ ] Suno AI補完データ統合

### Week 7-8: Drums Generator強化
- [ ] DrumAdapter拡張
- [ ] LAMDA Stage2データ統合
- [ ] Bass同期強化

### Week 9: 全楽器統合テスト
- [ ] modular_composer_stub.py統合
- [ ] 完全楽曲生成デモ（3-5曲）

---

## 📊 既存 vs 新規実装の比較

| 項目 | 既存Generator | 私の新規実装（❌不要） |
|------|--------------|-------------------|
| **行数** | 986-2,487行/Generator | ~500行/Generator |
| **機能** | ✅ Pattern library完備 | ❌ 基本機能のみ |
| **Emotion** | ✅ 詳細mapping（12種類） | ⚠️ 簡易mapping（6種類） |
| **Controls** | ✅ CC11/CC64/Vibrato完備 | ❌ なし |
| **Humanization** | ✅ utilities統合 | ❌ なし |
| **ML Model** | ✅ Velocity model対応 | ❌ なし |
| **Kick同期** | ✅ Bass完全実装 | ❌ スタブのみ |
| **Stage2統合** | ⚠️ 未実装（これを追加） | ⚠️ 想定したが実装なし |

**結論:**
- ✅ **既存Generatorを活用**
- ✅ **Stage2データ統合を追加**
- ❌ **新規実装は不要**

---

## 🔧 次のアクション

**Immediate (今日):**
1. ❌ 新規実装ファイルを削除/無視
2. ✅ 既存Generator機能確認完了
3. ⏸️ Stage2パターン抽出スクリプト作成開始

**Short-term (今週):**
1. Stage2データからPianoパターン抽出
2. PianoPatternRecommender実装
3. 既存PianoGeneratorに統合
4. テスト実行

**作曲デモ計画:**
次回、既存Generator + Stage2統合での実際の作曲ワークフローを提案します。

---

## 💡 重要な気づき

**既存実装は非常に成熟:**
- BasePartGenerator (714行): Controls/Humanization完備
- PianoGenerator (986行): RH/LH pattern、ML velocity
- BassGenerator (2,487行): Walking/Pick/Slap、Kick同期、Approach note

**必要なのは:**
- ✅ Stage2データ統合（Learning-based強化）
- ✅ Pattern推薦システム
- ✅ Technique分類器

**不要なのは:**
- ❌ Generator再実装
- ❌ 基本機能の再発明

**正しいアプローチ:**
既存の成熟したGenerator + Stage2高品質データ = 最強の組み合わせ！
