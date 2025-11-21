# OtobonAI 統合ロードマップ（v1 → v1.5 → v2）

**作成日**: 2025年11月15日  
**目的**: EmotionAI / GuideToneAI を段階的に拡張し、lyric_anchors / CREPE / OaF との統合を実現

---

## 📊 現状分析（v1）

### ✅ 実装済み
- **Rulebook Engine**: `otobonAI/rulebook_engine.py`（280行）
- **生成スクリプト**: `scripts/generate_guidetone_and_emotion_from_rulebook.py`（700行）
- **Rulebook定義**: `configs/otobonAI/rulebook.yaml`（10ルール）

### 📥 v1 の入力
```
bars_with_slots.parquet   → Bar構造
manual_chordmap.json      → Chord情報
sections.json             → Section定義
tempo_map.json            → Tempo情報（動的BPM）
```

### 📤 v1 の出力
```json
// emotion_profile.json
{
  "unit": "bar",
  "meta": {"key_center": "C#m", "base": {...}},
  "events": [
    {
      "bar": 0,
      "energy": 0.45,
      "tension": 0.55,
      "brightness": 0.40,
      "valence": 0.35,
      "density": 0.5,
      "rule_ids": ["HRM_001"],
      "tags": ["bittersweet"]
    }
  ]
}

// guide_tone_hints.json
{
  "unit": "bar",
  "events": [
    {
      "bar": 0,
      "scale_degree": 3,
      "register": "mid",
      "approx_pitch": 65,
      "motion": "step",
      "notes_per_bar": 1.2,
      "rule_ids": ["HRM_001"]
    }
  ]
}
```

### 🎯 v1 の設計哲学
> **「曲の骨格（ハーモニーと構成）だけから Emotion / GuideTone を決める層」**
>
> - Rulebook = 和声・セクション・テンポによる基礎判断
> - lyric_anchors は**意図的に含めていない**（堅実な設計）
> - 次のステップで「味付け」として追加する余地を残している

---

## 🚀 拡張ロードマップ

### Phase 1.5: Lyric Anchors 統合（優先度: 高）

#### 📥 追加入力
```json
// lyric_anchors.json（既存ファイル）
{
  "anchors": [
    {
      "bar": 23,
      "beat": 1.0,
      "syllable": "こ",
      "word": "恋",
      "stress": true,           // ★ 強勢音節
      "phrase_boundary": "end", // ★ フレーズ境界
      "vowel_type": "o",
      "duration_hint": 0.8
    }
  ]
}
```

#### 📤 拡張出力（v1.5）
```json
// emotion_profile.json v1.5
{
  "events": [
    {
      "bar": 23,
      "energy": 0.68,
      "tension": 0.72,
      "brightness": 0.65,
      "valence": 0.70,
      "density": 0.8,
      
      // ★ v1.5 で追加
      "anchor_weight": 0.85,        // どれだけ歌の主役バーか（0-1）
      "has_lyric_stress": true,     // 強勢音節あり
      "phrase_position": "end",     // begin/mid/end
      "vocal_focus": true,          // ボーカル重点バー
      
      "rule_ids": ["HRM_001", "MLT_001"],
      "tags": ["peak", "vocal_climax", "lyric_focus"]
    }
  ]
}

// guide_tone_hints.json v1.5
{
  "events": [
    {
      "bar": 23,
      "preferred_degrees": [3, 7, 9],  // v1の scale_degree を拡張
      "register": "mid_high",
      "approx_pitch": 72,
      "motion": "step",
      "notes_per_bar": 1.6,            // 密度上昇
      
      // ★ v1.5 で追加
      "lyric_anchor_weight": 0.85,    // anchor の強度
      "phrase_role": "climax",        // begin/build/climax/release
      "stress_alignment": true,       // 強勢とガイドトーンを同期
      "vowel_rich": true,             // 伸ばしやすい母音
      
      "rule_ids": ["HRM_001", "MLT_001"]
    }
  ]
}
```

#### 🔧 実装タスク
1. **lyric_anchors.json パーサー追加**
   - `scripts/generate_guidetone_and_emotion_from_rulebook.py` に統合
   - Bar毎の anchor 情報を集約（stress_level, phrase_pos, vowel_type）

2. **Emotion補正ロジック**
   ```python
   # stress == true → energy += 0.1, tension += 0.1
   # phrase_boundary == "end" → tension_peak = True
   # vowel_rich == true → valence += 0.05（伸ばしやすい = 表現力）
   ```

3. **GuideTone補正ロジック**
   ```python
   # stress位置 → preferred_degrees に 9th/11th 追加
   # phrase_end → motion = "leap_to_resolution"
   # vowel_rich → notes_per_bar *= 0.8（伸ばす = 音数減）
   ```

4. **Rulebook拡張（optional）**
   ```yaml
   # 新ルール例
   - id: MLT_003
     name: "強勢音節へのテンションノート配置"
     when:
       lyric_anchor:
         has_stress: true
         phrase_pos: ["mid", "end"]
     guide_tone:
       priority_tones: [9, 11, 13]
       motion: "leap_ok"
   ```

---

### Phase 2.0: CREPE / OaF 統合（優先度: 中）

#### 📥 追加入力
```json
// crepe_pitch.json（ボーカル抽出結果）
{
  "frames": [
    {
      "time": 12.5,
      "bar": 23,
      "pitch_midi": 72.3,
      "confidence": 0.92,
      "energy": 0.78
    }
  ]
}

// oaf_features.json（Open Audio Features）
{
  "bars": [
    {
      "bar": 23,
      "rms_energy": 0.65,
      "spectral_centroid": 2500,
      "vocal_presence": 0.88
    }
  ]
}
```

#### 📤 拡張出力（v2.0）
```json
// emotion_profile.json v2.0
{
  "events": [
    {
      "bar": 23,
      "energy": 0.72,           // CREPE energy で微調整
      "tension": 0.75,
      
      // ★ v2.0 で追加
      "vocal_pitch_avg": 72.3,  // CREPE 平均ピッチ
      "vocal_energy_avg": 0.78, // CREPE エネルギー
      "actual_vocal_peak": true,// 実際に声が盛り上がった
      
      "sources": {
        "harmony": 0.50,        // 和声由来の割合
        "lyric": 0.30,          // 歌詞由来の割合
        "vocal_actual": 0.20    // 実ボーカル由来の割合
      }
    }
  ]
}
```

#### 🔧 実装タスク
1. **CREPE統合**
   - Bar毎の平均ピッチ・エネルギー抽出
   - `emotion_profile.json` の energy に反映（重み 0.2）

2. **OaF統合**
   - Spectral features → brightness 補正
   - RMS energy → density 補正

3. **三層統合アルゴリズム**
   ```python
   final_energy = (
       harmony_energy * 0.50 +
       lyric_energy * 0.30 +
       vocal_energy * 0.20
   )
   ```

---

## 🏗️ 統一インターフェース設計

### Context 構造（EmotionAI / GuideToneAI 共通）

```python
@dataclass
class BarContext:
    """Rulebook クエリ用の統一コンテキスト"""
    
    # Basic
    song_id: str
    bar_index: int
    section: str                    # intro/verse/chorus/bridge
    role: str                       # strings/bass/piano/drums
    
    # Harmony
    key_center: str                 # "C#m"
    chord_symbol: str               # "C#m7(9)"
    scale_degree: Optional[int]     # 1-7（GuideTone用）
    function: str                   # tonic/subdominant/dominant
    tempo_bpm: float                # 90.0
    
    # Slots
    slots: Dict[str, bool]          # {"fill": True, "riff": False}
    
    # Emotion (v1 base)
    emotion: Dict[str, Any]
    # {
    #   "global_mood": "hopeful_dark",
    #   "local_energy": 0.55,
    #   "local_tension": 0.65
    # }
    
    # Lyric Anchor (v1.5)
    lyric_anchor: Optional[Dict[str, Any]]
    # {
    #   "has_anchor": True,
    #   "stress_level": 0.8,
    #   "phrase_pos": "end",  # begin/mid/end
    #   "vowel_rich": True
    # }
    
    # Harmony Detail
    harmony: Dict[str, Any]
    # {
    #   "tension_flags": ["9"],
    #   "cadence_type": "V7-I"
    # }
    
    # Vocal Actual (v2.0)
    vocal_actual: Optional[Dict[str, Any]]
    # {
    #   "pitch_avg": 72.3,
    #   "energy_avg": 0.78,
    #   "spectral_brightness": 0.65
    # }
```

### EmotionAI クラス（v1.5 API）

```python
# otobonAI/emotion_ai.py（新規作成）
from pathlib import Path
from typing import Dict, Any
import json
from .rulebook_engine import Rulebook

class EmotionAI:
    """
    Emotion Profile を管理し、Bar毎の感情パラメータを提供
    
    v1:   harmony + section + tempo
    v1.5: + lyric_anchors
    v2:   + CREPE/OaF
    """
    
    def __init__(
        self,
        profile_path: Path,
        rulebook_path: Path,
        lyric_anchors_path: Optional[Path] = None,  # v1.5
        crepe_path: Optional[Path] = None            # v2.0
    ):
        self.profile = self._load_profile(profile_path)
        self.engine = Rulebook.load(rulebook_path)
        self.anchors = self._load_anchors(lyric_anchors_path) if lyric_anchors_path else {}
        self.crepe = self._load_crepe(crepe_path) if crepe_path else {}
    
    def get_bar_emotion(
        self,
        bar_index: int,
        role: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        指定Barの感情パラメータを取得
        
        Returns:
            {
                "energy": 0.68,
                "tension": 0.72,
                "brightness": 0.65,
                "valence": 0.70,
                "density": 0.80,
                "anchor_weight": 0.85,  # v1.5
                "tags": ["peak", "vocal_climax"]
            }
        """
        # v1 base
        base = self.profile["events"][bar_index]
        
        # v1.5: lyric_anchor 補正
        if bar_index in self.anchors:
            anchor = self.anchors[bar_index]
            if anchor.get("stress"):
                base["energy"] += 0.1
                base["tension"] += 0.1
            base["anchor_weight"] = anchor.get("stress_level", 0.0)
        
        # v2.0: CREPE 補正（未実装）
        # if bar_index in self.crepe:
        #     base["energy"] = 0.7 * base["energy"] + 0.3 * self.crepe[bar_index]["energy"]
        
        # Rulebook query (optional refinement)
        if context:
            full_context = self._build_context(bar_index, role, context)
            actions = self.engine.find_matching(full_context, "emotion")
            base = self._apply_emotion_actions(base, actions)
        
        return base
    
    def _build_context(self, bar_index: int, role: str, extra: Dict) -> Dict:
        """BarContext 構築"""
        base = self.profile["events"][bar_index]
        return {
            "bar_index": bar_index,
            "section": base.get("section", "unknown"),
            "role": role,
            "emotion": {
                "local_energy": base["energy"],
                "local_tension": base["tension"]
            },
            "lyric_anchor": self.anchors.get(bar_index),
            **extra
        }
    
    def _apply_emotion_actions(self, base: Dict, actions: List) -> Dict:
        """Rulebook actions を base に適用"""
        for action in actions:
            emo_action = action.get_emotion_action()
            if emo_action:
                base["energy"] += emo_action.energy_delta
                base["tension"] += emo_action.tension_delta
                base["brightness"] += emo_action.brightness_delta
                base["valence"] += emo_action.valence_delta
                base["tags"].extend(emo_action.tags_add)
        return base
```

### GuideToneAI クラス（v1.5 API）

```python
# otobonAI/guide_tone_ai.py（新規作成）
from pathlib import Path
from typing import Dict, Any, List
import json
from .rulebook_engine import Rulebook

class GuideToneAI:
    """
    Guide Tone Hints を管理し、Bar毎のガイドトーン推奨を提供
    
    v1:   harmony + section + tempo
    v1.5: + lyric_anchors
    v2:   + CREPE/OaF
    """
    
    def __init__(
        self,
        hints_path: Path,
        rulebook_path: Path,
        lyric_anchors_path: Optional[Path] = None
    ):
        self.hints = self._load_hints(hints_path)
        self.engine = Rulebook.load(rulebook_path)
        self.anchors = self._load_anchors(lyric_anchors_path) if lyric_anchors_path else {}
    
    def suggest_for_bar(
        self,
        bar_index: int,
        role: str,
        chord_symbol: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        指定Barのガイドトーンヒントを取得
        
        Returns:
            {
                "preferred_degrees": [3, 7, 9],
                "register": "mid_high",
                "approx_pitch": 72,
                "motion": "step",
                "notes_per_bar": 1.6,
                "lyric_anchor_weight": 0.85,  # v1.5
                "phrase_role": "climax",      # v1.5
                "stress_alignment": True      # v1.5
            }
        """
        # v1 base
        base = self.hints["events"][bar_index]
        
        # v1.5: lyric_anchor 補正
        if bar_index in self.anchors:
            anchor = self.anchors[bar_index]
            
            # 強勢音節 → テンションノート追加
            if anchor.get("stress"):
                if "preferred_degrees" not in base:
                    base["preferred_degrees"] = [base.get("scale_degree", 3)]
                base["preferred_degrees"].extend([9, 11])
                base["stress_alignment"] = True
            
            # フレーズ末 → 解決的な動き
            if anchor.get("phrase_boundary") == "end":
                base["motion"] = "leap_to_resolution"
                base["phrase_role"] = "release"
            
            # 母音豊か → 音数減（伸ばす）
            if anchor.get("vowel_rich"):
                base["notes_per_bar"] *= 0.8
            
            base["lyric_anchor_weight"] = anchor.get("stress_level", 0.0)
        
        # Rulebook query (optional refinement)
        if context:
            full_context = self._build_context(bar_index, role, chord_symbol, context)
            actions = self.engine.find_matching(full_context, "guide_tone")
            base = self._apply_guidetone_actions(base, actions)
        
        return base
    
    def _build_context(self, bar_index: int, role: str, chord_symbol: str, extra: Dict) -> Dict:
        """BarContext 構築"""
        base = self.hints["events"][bar_index]
        return {
            "bar_index": bar_index,
            "section": base.get("section", "unknown"),
            "role": role,
            "chord_symbol": chord_symbol,
            "scale_degree": base.get("scale_degree"),
            "lyric_anchor": self.anchors.get(bar_index),
            "slots": extra.get("slots", {}),
            **extra
        }
    
    def _apply_guidetone_actions(self, base: Dict, actions: List) -> Dict:
        """Rulebook actions を base に適用"""
        for action in actions:
            gt_action = action.get_guidetone_action()
            if gt_action:
                if gt_action.priority_tones:
                    base["preferred_degrees"] = gt_action.priority_tones
                if gt_action.default_register:
                    base["register"] = gt_action.default_register
                if gt_action.motion:
                    base["motion"] = gt_action.motion
                if gt_action.notes_per_bar:
                    base["notes_per_bar"] = gt_action.notes_per_bar
        return base
```

---

## 🔌 V2 Generator 統合

### strings_plan_v2.py 統合例

```python
# scripts/generate_strings_plan_v2.py（既存）

from otobonAI.emotion_ai import EmotionAI
from otobonAI.guide_tone_ai import GuideToneAI

# 初期化
emotion_ai = EmotionAI(
    profile_path=song_dir / "analysis/emotion_profile.json",
    rulebook_path=Path("configs/otobonAI/rulebook.yaml"),
    lyric_anchors_path=song_dir / "analysis/lyric_anchors.json"  # v1.5
)

guidetone_ai = GuideToneAI(
    hints_path=song_dir / "analysis/guide_tone_hints.json",
    rulebook_path=Path("configs/otobonAI/rulebook.yaml"),
    lyric_anchors_path=song_dir / "analysis/lyric_anchors.json"  # v1.5
)

# Bar毎の生成ループ
for bar_idx, bar_row in bars.iterrows():
    # Emotion取得
    emotion = emotion_ai.get_bar_emotion(bar_idx, role="strings")
    
    # GuideTone取得
    chord_symbol = get_chord_at_bar(bar_idx)
    guide = guidetone_ai.suggest_for_bar(bar_idx, role="strings", chord_symbol=chord_symbol)
    
    # Density調整
    base_density = policy["strings"]["density"]
    density_scale = 0.5 + emotion["energy"]  # energy 0.5 → density 100%, energy 1.0 → density 150%
    actual_density = base_density * density_scale
    
    # Notes per bar調整
    target_notes = guide["notes_per_bar"]
    
    # Register調整
    register_hint = guide["register"]  # "mid_high" → octave 4-5
    
    # Preferred degrees使用
    preferred_degrees = guide["preferred_degrees"]  # [3, 7, 9]
    
    # 生成
    events = generate_strings_for_bar(
        bar_idx=bar_idx,
        density=actual_density,
        target_notes=target_notes,
        register=register_hint,
        preferred_degrees=preferred_degrees,
        emotion=emotion
    )
```

---

## 🎯 生成AI連携（Phase 2.5）

### Rulebook = 審査員、AI = 候補生成者

```python
# scripts/generate_melody_with_ai.py（新規・構想）

from otobonAI.emotion_ai import EmotionAI
from otobonAI.guide_tone_ai import GuideToneAI
from some_melody_ml import MelodyGenerator  # CREPE/OaF/Melody ML

# AI初期化
melody_gen = MelodyGenerator(model_path="models/melody_v1.ckpt")
emotion_ai = EmotionAI(...)
guidetone_ai = GuideToneAI(...)

# Bar毎の生成
for bar_idx in range(num_bars):
    # コンテキスト構築
    emotion = emotion_ai.get_bar_emotion(bar_idx, role="vocal")
    guide = guidetone_ai.suggest_for_bar(bar_idx, role="vocal", chord_symbol=chord)
    
    # AI候補生成（N本）
    candidates = melody_gen.generate_candidates(
        bar_idx=bar_idx,
        num_candidates=10,
        temperature=0.8
    )
    
    # Rulebook審査
    scores = []
    for candidate in candidates:
        score = evaluate_candidate(
            candidate=candidate,
            guide=guide,
            emotion=emotion,
            rulebook=rulebook
        )
        scores.append(score)
    
    # 最高スコア採用
    best_idx = np.argmax(scores)
    final_melody[bar_idx] = candidates[best_idx]

def evaluate_candidate(candidate, guide, emotion, rulebook):
    """候補をRulebookに照らして採点"""
    score = 0.0
    
    # Preferred degrees チェック
    if candidate["scale_degree"] in guide["preferred_degrees"]:
        score += 10.0
    
    # Register チェック
    if candidate["register"] == guide["register"]:
        score += 5.0
    
    # Emotion alignment チェック
    if abs(candidate["energy"] - emotion["energy"]) < 0.2:
        score += 8.0
    
    # Avoid notes チェック（減点）
    if candidate["has_avoid_notes"]:
        score -= 15.0
    
    return score
```

### 三層統合の役割分担

| Layer | 役割 | 例 |
|-------|------|-----|
| **Harmony & Section** | 基礎判断（rulebook主入力） | "chorus + V7 → energy 0.7" |
| **Lyric Anchors** | 重点配置（どこを目立たせるか） | "強勢音節 → tension +0.1, degree 9追加" |
| **CREPE/OaF** | 実際の勢い（実ボーカル補正） | "実際のpitch高い → brightness +0.1" |
| **EmotionAI** | 司令塔（三層を統合） | `final_energy = 0.5*harm + 0.3*lyric + 0.2*vocal` |
| **GuideToneAI** | 司令塔（三層を統合） | `preferred_degrees = [3,7] + lyric_stress:[9,11]` |
| **生成AI** | 候補生成 | "N本のメロディ案を出す" |
| **Rulebook** | 審査員 | "AIの候補から音楽的に正しいものだけ通す" |

---

## 📋 実装優先度

### Phase 1.5（優先度: 高）
- [ ] `otobonAI/emotion_ai.py` 作成
- [ ] `otobonAI/guide_tone_ai.py` 作成
- [ ] `generate_guidetone_and_emotion_from_rulebook.py` に lyric_anchors パーサー追加
- [ ] emotion_profile.json v1.5 フォーマット対応
- [ ] guide_tone_hints.json v1.5 フォーマット対応
- [ ] `generate_strings_plan_v2.py` に EmotionAI/GuideToneAI 統合
- [ ] song_004 でテスト実行

### Phase 2.0（優先度: 中）
- [ ] CREPE pitch 抽出スクリプト作成
- [ ] OaF features 抽出スクリプト作成
- [ ] EmotionAI に CREPE/OaF 統合
- [ ] emotion_profile.json v2.0 フォーマット対応
- [ ] 三層統合重み調整実験

### Phase 2.5（優先度: 低）
- [ ] MelodyGenerator ラッパー作成
- [ ] Rulebook評価関数実装
- [ ] AI候補生成＋Rulebook審査パイプライン構築

---

## 📚 設計原則

### 1. 段階的拡張
> v1 → v1.5 → v2 と段階的に機能追加し、各段階で動作検証

### 2. 後方互換性
> v1.5 は v1 の出力フォーマットを拡張（既存フィールドはそのまま）

### 3. 疎結合
> EmotionAI / GuideToneAI は独立クラスとして、V2 Generator から呼び出し可能

### 4. 安全弁としての Rulebook
> 生成AIが暴走しても、Rulebookで「人間らしい範囲」に抑制

### 5. 三層統合の重み
> Harmony(50%) + Lyric(30%) + Vocal(20%) でバランス調整

---

**次のアクション**: Phase 1.5 実装開始
- `otobonAI/emotion_ai.py` 作成
- `otobonAI/guide_tone_ai.py` 作成
- lyric_anchors.json パーサー統合
