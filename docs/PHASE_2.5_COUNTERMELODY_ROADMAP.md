# Phase 2.5: Counter-Melody Spec実装ロードマップ

## 📋 現状確認（2025-11-15）

### ✅ Phase 2.0完了項目
- **generate_strings_plan_v2.py Phase 2.0統合**: 169イベント生成成功（+5.0%）
- **LyricAnchorIndex**: time-based lyric_anchors.json対応
  - 構造: `{unit: "sec", anchors: [{time, time_ql, classes, windows_ms}]}`
  - bar変換: tempo_bpm使用、phrase_role推定（start/mid/end）
- **EmotionAI v2**: velocity/duration/density調整動作確認
- **GuideToneAI v2**: phrase_shape/notes_per_bar調整動作確認
- **Rulebook細分化ルール**: section × phrase_role対応（10ルール）

### 🔍 lyric_anchors.json実構造
```json
{
  "unit": "sec",
  "anchors": [
    {
      "time": 0.06965986394557823,
      "time_ql": 0.0,
      "token": null,
      "classes": ["sibilant"],
      "section": null,
      "windows_ms": {"pre": 30.0, "post": 20.0}
    },
    {
      "time": 0.5804988662131519,
      "classes": ["stress"],
      "windows_ms": {"pre": 0.0, "post": 80.0}
    }
  ]
}
```

**重要**: 
- ❌ bar/beatフィールド無し（time-basedのみ）
- ❌ phrase_boundaryフィールド無し
- ✅ classes配列: `["sibilant", "stress", "plosive"]`
- ✅ LyricAnchorIndexがtime→bar変換、phrase_role推定を担当

---

## 🎯 Phase 2.5ゴール: 実戦投入可能なOtobonAI

### 核心3要素
1. **和声の説得力**: すべてのパートがmanual_chordmapを尊重
2. **ボーカルとの一体感**: lyric_anchors/CREPEによる「歌との会話」
3. **ルールブック駆動**: EmotionAI/GuideToneAIがrulebook.yaml経由で一元制御

---

## 📅 フェーズ別実装計画

### Phase A: Phase 2.0の仕上げと可視化（1-2週間）

#### A-1. Strings/Piano挙動の固定化 ⏳
**目的**: policy presets確立、再現性確保

**作業内容**:
1. **バリエーション生成**（済: strings_v20）
   ```bash
   # Policy変更版（density調整）
   python3 scripts/generate_strings_plan_v2.py \
     --policy configs/policy_presets/ballad_dense.yaml \
     --out plans/strings_ballad_dense.json
   
   # Register変更版（high重視）
   python3 scripts/generate_strings_plan_v2.py \
     --policy configs/policy_presets/ballad_high_register.yaml \
     --out plans/strings_ballad_high.json
   ```

2. **DAWでの選定**
   - strings_v20.mid vs バリエーション聴き比べ
   - 「採用セット」決定（例: ballad_guide_strings.yaml）

3. **Policy Preset化**
   ```yaml
   # configs/policy_presets/ballad_guide_strings.yaml
   sections:
     chorus:
       strings:
         density_floor: 0.6
         density_ceil: 0.9
         register: "mid_high"
         countermelody_strength: 0.8
     verse:
       strings:
         density_floor: 0.3
         density_ceil: 0.6
         register: "mid"
         countermelody_strength: 0.5
   ```

**成果物**:
- ✅ Policy presets: `ballad_guide_strings.yaml`
- ✅ 採用MIDI: `strings_ballad_guide.mid`

---

#### A-2. GuideTone/Emotionデバッグ表示 ⏳
**目的**: 「耳で気になったbarが数値的にどうなっているか」可視化

**実装**:

1. **GuideToneAI v2デバッグ出力**
   ```python
   # otobonAI/guide_tone_ai_v2.py
   class GuideToneAIv2:
       def __init__(self, guide_tone_hints, rulebook, debug_csv=None):
           self.debug_csv = debug_csv
           self.debug_rows = []
       
       def get_plan(self, context):
           plan = ...
           if self.debug_csv:
               self.debug_rows.append({
                   "bar": context["bar"],
                   "section": context["section"],
                   "phrase_role": context.get("lyric", {}).get("phrase_role", "none"),
                   "preferred_degrees": plan.preferred_degrees,
                   "register": plan.register,
                   "phrase_shape": plan.phrase_shape,
                   "notes_per_bar": plan.notes_per_bar
               })
           return plan
       
       def save_debug(self):
           if self.debug_csv and self.debug_rows:
               import csv
               with open(self.debug_csv, "w") as f:
                   writer = csv.DictWriter(f, fieldnames=self.debug_rows[0].keys())
                   writer.writeheader()
                   writer.writerows(self.debug_rows)
   ```

2. **EmotionAI v2デバッグ出力**（同様）
   ```python
   # analysis/emotion_debug.csv
   # bar, section, energy, tension, vel_scale, dur_scale, density_scale
   ```

3. **generate_strings_plan_v2.py統合**
   ```python
   guidetone_ai = GuideToneAIv2(
       guide_tone_hints=guide_hints_data,
       rulebook=rulebook,
       debug_csv="analysis/guide_tone_debug.csv"
   )
   emotion_ai = EmotionAIv2(
       emotion_profile=emotion_profile_data,
       rulebook=rulebook,
       debug_csv="analysis/emotion_debug.csv"
   )
   
   # 処理後
   guidetone_ai.save_debug()
   emotion_ai.save_debug()
   ```

**成果物**:
- ✅ `analysis/guide_tone_debug.csv`
- ✅ `analysis/emotion_debug.csv`
- ✅ 問題bar特定→rulebook調整フロー確立

---

### Phase B: Counter-Melody仕様の明文化（1週間）

#### B-1. Counter-Melody出自の仕様化 ✅
**宣言**: OtobonAIは**伴奏側カウンターメロディ**に専念

**材料3種**:
1. **manual_chordmap**: コード＋テンション
2. **guide_tone_hints**: 推奨スケール度数/notes_per_bar/register
3. **lyric_anchors**: phrase_role（start/mid/end）、stress

**成果物**: ✅ `docs/COUNTER_MELODY_SPEC.md`（本ドキュメント下部）

---

#### B-2. Counter-Melody生成アルゴリム明文化 ⏳
**実装**: `scripts/generate_strings_plan_v2.py`のmake_countermelody系関数整理

**4ステップ**:

1. **骨組み生成**
   ```python
   def make_countermelody_skeleton(chord, guide_plan, phrase_role):
       """
       guide_plan.preferred_degreesから1-2音選択
       phrase_role=start → 上昇優先
       phrase_role=end → 下降優先
       """
       degrees = guide_plan.preferred_degrees
       if phrase_role == "start":
           # 3rd → 5th → 7th → 9th (上昇)
           return [degrees[0], degrees[min(1, len(degrees)-1)]]
       elif phrase_role == "end":
           # 9th → 7th → 5th → 3rd (下降)
           return [degrees[-1], degrees[max(0, len(degrees)-2)]]
       else:
           return [degrees[0]]
   ```

2. **スケール補完**
   ```python
   def fill_scale_passing_tones(skeleton, chord, notes_per_bar):
       """
       notes_per_barを満たすまでスケール音で補完
       非和声音は前後がchord_toneの場合のみ、16分-8分に限定
       """
       scale = get_scale_from_chord(chord)
       filled = skeleton.copy()
       while len(filled) < notes_per_bar:
           # Stepwise (±2度) に移動
           last = filled[-1]
           next_note = stepwise_move(last, scale)
           if is_non_chord_tone(next_note, chord):
               # 16分-8分に限定
               duration = min(0.5, QL_16TH)
           filled.append(next_note)
       return filled
   ```

3. **ボイスリーディング**
   ```python
   def apply_voice_leading(notes, prev_note=None):
       """
       直前ノートとの距離最小化（5度以上跳躍は例外処理）
       跳躍後は次音でstepwise埋め戻し
       """
       optimized = [notes[0]]
       for i, note in enumerate(notes[1:], 1):
           interval = abs(note - optimized[-1])
           if interval > 7:  # 完全5度以上
               # 1オクターブ調整
               if note > optimized[-1]:
                   note -= 12
               else:
                   note += 12
           optimized.append(note)
       return optimized
   ```

4. **Emotion反映**
   ```python
   def apply_emotion_to_countermelody(notes, emotion_params):
       """
       energy高 → 音数・velocity増
       tension高 → テンションノート（9/11/13）増、duration延長
       """
       if emotion_params.energy > 0.7:
           notes = increase_note_density(notes, factor=1.2)
       
       velocities = [base_vel * emotion_params.velocity_scale 
                     for base_vel in base_velocities]
       
       durations = [base_dur * emotion_params.duration_scale
                    for base_dur in base_durations]
       
       return notes, velocities, durations
   ```

**成果物**:
- ✅ `docs/COUNTER_MELODY_ALGORITHM.md`
- ✅ `scripts/countermelody_lib.py`（共通関数ライブラリ）

---

### Phase C: Lyric Anchorsをルールブック側から活用（1週間）

#### C-1. rulebook.yamlにlyric系ルール拡張 ⏳

**現状**: LYRIC_001/002（基本ルール2個）

**拡張案**（10ルール追加）:

```yaml
# configs/otobonAI/rulebook.yaml

rules:
  # ========== Chorus × Phrase Role ==========
  - id: LYRIC_101
    name: "Chorus phrase_start → Strings 上昇motion"
    domain: "guidetone"
    when:
      section: ["chorus"]
      instrument: ["strings"]
      phrase_role: ["start"]
    params:
      pattern: "arpeggio_up"
      register: "mid_high"
      notes_per_bar: {min: 3, max: 6}
      preferred_degrees: [3, 7, 9, 11]
      phrase_shape: "uphill"
      weight: 0.8

  - id: LYRIC_102
    name: "Chorus phrase_end → Strings 着地感"
    domain: "guidetone"
    when:
      section: ["chorus"]
      instrument: ["strings"]
      phrase_role: ["end"]
    params:
      pattern: "cadential_hold"
      register: "mid_high"
      preferred_degrees: [7, 3, 1]  # leading→3rd→tonic
      sustain_ratio: 0.6
      phrase_shape: "downhill"
      weight: 0.7

  - id: LYRIC_103
    name: "Chorus phrase_mid → Strings sustained"
    domain: "guidetone"
    when:
      section: ["chorus"]
      instrument: ["strings"]
      phrase_role: ["mid"]
    params:
      pattern: "sustained_pad"
      register: "mid_high"
      notes_per_bar: {min: 1, max: 3}
      preferred_degrees: [3, 5, 7]
      weight: 0.5

  # ========== Verse × Phrase Role ==========
  - id: LYRIC_104
    name: "Verse phrase_start → Piano broken chord"
    domain: "guidetone"
    when:
      section: ["verse"]
      instrument: ["piano"]
      phrase_role: ["start"]
    params:
      pattern: "broken_chord"
      register: "mid"
      notes_per_bar: {min: 2, max: 4}
      preferred_degrees: [1, 3, 5, 7]
      phrase_shape: "uphill"
      weight: 0.6

  - id: LYRIC_105
    name: "Verse phrase_end → Piano sustained"
    domain: "guidetone"
    when:
      section: ["verse"]
      instrument: ["piano"]
      phrase_role: ["end"]
    params:
      pattern: "sustained_pad"
      register: "mid"
      notes_per_bar: {min: 1, max: 2}
      sustain_ratio: 0.7
      weight: 0.5

  # ========== No Vocal (has_vocal=false) ==========
  - id: LYRIC_106
    name: "No Vocal bar → density抑制"
    domain: "emotion"
    when:
      has_vocal: false
    params:
      density_scale: 0.7
      energy_scale: 0.9
      weight: 0.6

  - id: LYRIC_107
    name: "No Vocal bar → Strings sustained pad優先"
    domain: "guidetone"
    when:
      instrument: ["strings"]
      has_vocal: false
    params:
      pattern: "sustained_pad"
      notes_per_bar: {min: 1, max: 2}
      weight: 0.5

  # ========== Stress (classes含む["stress"]) ==========
  - id: LYRIC_108
    name: "Stress bar → velocity強調"
    domain: "emotion"
    when:
      stress_level: {min: 0.5}
    params:
      velocity_scale: 1.15
      energy_scale: 1.1
      weight: 0.5

  - id: LYRIC_109
    name: "Sibilant bar → Strings裏拍優先"
    domain: "guidetone"
    when:
      instrument: ["strings"]
      lyric_classes: ["sibilant"]
    params:
      rhythm_offset: 0.25  # 裏拍寄り
      weight: 0.4

  # ========== Emotion × Phrase Role ==========
  - id: LYRIC_110
    name: "phrase_end → duration延長（締め感）"
    domain: "emotion"
    when:
      phrase_role: ["end"]
    params:
      duration_scale: 1.2
      tension_scale: 0.85
      weight: 0.6
```

**成果物**:
- ✅ `configs/otobonAI/rulebook.yaml`（+10ルール）

---

#### C-2. EmotionAI/GuideToneAIのcontext統一 ⏳

**現状**: context["lyric"]でphrase_role/stress_levelを渡している

**拡張**:

1. **LyricAnchorIndex API拡張**
   ```python
   # otobonAI/lyric_index.py
   class LyricAnchorIndex:
       def get_bar_context(self, bar_idx: int) -> Dict[str, Any]:
           """
           Rulebook照会用の統一context返却
           """
           bar_info = self.get_bar_info(bar_idx)
           anchors = [a for a in self.anchors if a["bar"] == bar_idx]
           
           all_classes = []
           for a in anchors:
               all_classes.extend(a.get("classes", []))
           
           return {
               "has_vocal": bar_info.get("has_anchor", False),
               "phrase_role": bar_info.get("phrase_role", "none"),
               "stress_level": bar_info.get("stress_level", 0.0),
               "is_silent": bar_info.get("is_silent", False),
               "lyric_classes": list(set(all_classes)),  # ["stress", "sibilant"]
               "anchor_count": len(anchors)
           }
   ```

2. **generate_strings_plan_v2.py統合**
   ```python
   # Bar loop内
   lyric_ctx = lyric_index.get_bar_context(bar_idx)
   
   context = {
       "bar_index": bar_idx,
       "bar": bar_idx,
       "section": section_label,
       "role": "strings",
       "chord_symbol": chord.get("symbol", "C"),
       "slots": {"riff": riff_slot},
       **lyric_ctx  # has_vocal, phrase_role, stress_level, lyric_classes
   }
   
   emotion_params = emotion_ai.get_params(context)
   guide_params = guidetone_ai.get_plan(context)
   ```

**成果物**:
- ✅ `otobonAI/lyric_index.py` (get_bar_context追加)
- ✅ `scripts/generate_strings_plan_v2.py` (context統一)

---

### Phase D: CREPE/OaF導入（2-3週間）

#### D-1. CREPE役割の限定化 ⏳

**方針**: 全部に使わず、**解析器**として役割限定

**導入ステップ**:

1. **Vocal F0抽出**（register/motion検出）
   ```python
   # scripts/analyze_vocal_f0.py
   import crepe
   import numpy as np
   from pathlib import Path
   
   def analyze_vocal_f0(vocal_wav, output_csv):
       """
       CREPEでVocal F0抽出
       → bar単位でregister (low/mid/high) 集計
       → motion (uphill/downhill/flat) 推定
       """
       sr, audio = wavfile.read(vocal_wav)
       time, frequency, confidence, activation = crepe.predict(
           audio, sr, viterbi=True
       )
       
       # Bar単位集計
       bars_f0 = aggregate_f0_by_bar(time, frequency, confidence, tempo_bpm=120)
       
       # Register推定 (MIDI note基準)
       for bar in bars_f0:
           median_note = hz_to_midi(bar["median_f0"])
           if median_note < 60:
               bar["vocal_register"] = "low"
           elif median_note < 72:
               bar["vocal_register"] = "mid"
           else:
               bar["vocal_register"] = "high"
           
           # Motion推定 (前barとの差分)
           if bar_idx > 0:
               prev_median = bars_f0[bar_idx-1]["median_f0"]
               diff_semitones = 12 * np.log2(bar["median_f0"] / prev_median)
               if diff_semitones > 2:
                   bar["vocal_motion"] = "uphill"
               elif diff_semitones < -2:
                   bar["vocal_motion"] = "downhill"
               else:
                   bar["vocal_motion"] = "flat"
       
       # CSV出力
       pd.DataFrame(bars_f0).to_csv(output_csv, index=False)
   ```

2. **bars.parquetへのマージ**
   ```python
   # scripts/merge_vocal_f0_to_bars.py
   import pandas as pd
   
   bars = pd.read_parquet("analysis/bars_with_slots.parquet")
   vocal_f0 = pd.read_csv("analysis/vocal_f0.csv")
   
   bars = bars.merge(
       vocal_f0[["bar", "vocal_register", "vocal_motion", "median_f0"]],
       on="bar",
       how="left"
   )
   
   bars.to_parquet("analysis/bars_with_slots_v2.parquet")
   ```

3. **Rulebook VOCAL_系ルール追加**
   ```yaml
   # configs/otobonAI/rulebook.yaml
   rules:
     - id: VOCAL_001
       name: "Strings register → Vocal より3-5度上"
       domain: "guidetone"
       when:
         instrument: ["strings"]
         vocal_register: ["mid"]
       params:
         register: "mid_high"
         min_interval_from_vocal: 3  # 半音
         weight: 0.7
     
     - id: VOCAL_002
       name: "Vocal flat時 → Strings動き許可"
       domain: "guidetone"
       when:
         instrument: ["strings"]
         vocal_motion: ["flat"]
       params:
         notes_per_bar: {min: 3, max: 6}
         phrase_shape: "uphill"
         weight: 0.6
     
     - id: VOCAL_003
       name: "Vocal uphill時 → Strings水平-下降"
       domain: "guidetone"
       when:
         instrument: ["strings"]
         vocal_motion: ["uphill"]
       params:
         phrase_shape: "downhill"
         notes_per_bar: {min: 1, max: 3}
         weight: 0.5
   ```

4. **Context統合**
   ```python
   # generate_strings_plan_v2.py
   context = {
       ...,
       "vocal_register": bar_data.get("vocal_register", "mid"),
       "vocal_motion": bar_data.get("vocal_motion", "flat"),
       "vocal_f0": bar_data.get("median_f0", 261.63)
   }
   ```

**成果物**:
- ✅ `scripts/analyze_vocal_f0.py`
- ✅ `analysis/vocal_f0.csv`
- ✅ `analysis/bars_with_slots_v2.parquet` (vocal列追加)
- ✅ `configs/otobonAI/rulebook.yaml` (VOCAL_系ルール追加)

---

#### D-2. 衝突回避（±100cents帯回避） ⏳

**実装**: countermelody生成時にvocal_f0との距離チェック

```python
def avoid_vocal_collision(note_midi, vocal_f0, threshold_cents=100):
    """
    ±100cents以内の場合、3度上/下にシフト
    """
    note_f0 = midi_to_hz(note_midi)
    cents_diff = abs(1200 * np.log2(note_f0 / vocal_f0))
    
    if cents_diff < threshold_cents:
        # 3度シフト（4半音）
        if note_midi > hz_to_midi(vocal_f0):
            return note_midi + 4  # 長3度上
        else:
            return note_midi - 3  # 短3度下
    return note_midi
```

**成果物**:
- ✅ `scripts/countermelody_lib.py` (avoid_vocal_collision追加)

---

### Phase E: 他パートへの水平展開とQA（2-3週間）

#### E-1. Piano/Bass Phase 2.0統合 ⏳

**Piano**:
- 上声: Stringsと同じカウンターメロディ仕様共有
- 下声: ガイドトーン（3rd/7th）＋ルート/5th分担

```python
# scripts/generate_piano_plan_v2.py
def make_piano_upper_voice(chord, guide_plan, emotion_params):
    """Stringsのcountermelody_lib再利用"""
    return make_countermelody_skeleton(chord, guide_plan, emotion_params)

def make_piano_lower_voice(chord):
    """ガイドトーン（3rd/7th）優先"""
    return [chord.third, chord.seventh, chord.root]
```

**Bass**:
- EmotionAI → ウォーキング度合い（passing tone数）制御
- GuideToneAI → direction hint（上行/下行/リピート）

```python
# scripts/generate_bass_plan_v2.py
def make_bass_line(chord, emotion_params, guide_plan):
    walking_degree = emotion_params.energy * 0.5  # 0-0.5
    
    if walking_degree > 0.3:
        # Passing tones追加
        return make_walking_bass(chord, guide_plan.direction)
    else:
        # Root中心
        return make_root_bass(chord)
```

**成果物**:
- ✅ `scripts/generate_piano_plan_v2.py` Phase 2.0統合
- ✅ `scripts/generate_bass_plan_v2.py` Phase 2.0統合

---

#### E-2. QAゲート拡張 ⏳

**追加チェック項目**:

```python
# scripts/qa_countermelody.py
def qa_countermelody(plan_json):
    """
    Counter-Melody QAチェック
    """
    checks = {
        "chord_tone_ratio": check_chord_tone_ratio(plan_json),  # ≥60%
        "non_chord_consecutive": check_non_chord_consecutive(plan_json),  # ≤2連続
        "large_leap_count": check_large_leap_count(plan_json),  # ≤3回/chorus
        "register_collision": check_register_collision(plan_json),  # vocal±100cents回避
        "vocal_unison_ratio": check_vocal_unison_ratio(plan_json),  # ≤20%
    }
    
    failed = [k for k, v in checks.items() if not v]
    
    if failed:
        print(f"❌ QA FAILED: {failed}")
        return False
    else:
        print("✅ QA PASSED")
        return True
```

**成果物**:
- ✅ `scripts/qa_countermelody.py`
- ✅ CI統合（GitHub Actions）

---

### Phase F: 学習器との接続（中-長期、3-6ヶ月）

**方針**: ルール＋解析ベースを固めた上で、ML補助

**実装案**:

1. **context → パラメータペア収集**
   ```python
   # scripts/collect_training_data.py
   # {section, phrase_role, vocal_motion} → {notes_per_bar, register, phrase_shape}
   ```

2. **XGBoost学習**
   ```python
   import xgboost as xgb
   
   # 特徴: section, phrase_role, vocal_register, emotion.energy
   # ターゲット: notes_per_bar, register, phrase_shape
   
   model = xgb.XGBRegressor()
   model.fit(X_train, y_train)
   ```

3. **Rulebook defaultsをML提案**
   ```yaml
   # rulebook.yaml
   rules:
     - id: LYRIC_101
       params:
         notes_per_bar: {min: 3, max: 6}  # ML提案値
         ml_suggestion: true
         ml_confidence: 0.85
   ```

**成果物**:
- 🔄 Phase F（長期計画）

---

## 📊 優先順位マトリクス

| Phase | 優先度 | 期間 | 依存関係 | リスク |
|-------|--------|------|----------|--------|
| **A-1** Strings/Piano挙動固定 | 🔥 HIGH | 1週間 | Phase 2.0完了 | LOW |
| **A-2** Debug表示 | 🔥 HIGH | 3日 | A-1 | LOW |
| **B-2** Algorithm明文化 | 🔥 HIGH | 1週間 | A-2 | LOW |
| **C-1** Rulebook拡張 | 🔶 MED | 1週間 | B-2 | MED |
| **C-2** Context統一 | 🔶 MED | 3日 | C-1 | LOW |
| **D-1** CREPE導入 | 🔶 MED | 2週間 | C-2 | MED |
| **D-2** 衝突回避 | 🔷 LOW | 1週間 | D-1 | MED |
| **E-1** Piano/Bass統合 | 🔶 MED | 2週間 | C-2 | LOW |
| **E-2** QA拡張 | 🔷 LOW | 1週間 | E-1 | LOW |
| **F** 学習器接続 | 🔷 LOW | 3ヶ月+ | E-2 | HIGH |

---

## 🎯 即実行推奨（Next Steps）

### 1. 視聴テスト完了後 → A-1開始 ⏳
```bash
# Policy変更版生成
python3 scripts/generate_strings_plan_v2.py \
  --policy configs/policy_presets/ballad_dense.yaml \
  --out plans/strings_ballad_dense.json

# DAWで聴き比べ
open plans/strings_v20.mid
open plans/strings_ballad_dense.mid
```

### 2. A-2 Debug表示実装 ⏳
```bash
# GuideToneAI v2デバッグ出力追加
code otobonAI/guide_tone_ai_v2.py

# EmotionAI v2デバッグ出力追加
code otobonAI/emotion_ai_v2.py
```

### 3. B-2 Algorithm明文化 ⏳
```bash
# Counter-Melody Algorithm MD作成
code docs/COUNTER_MELODY_ALGORITHM.md

# 共通ライブラリ作成
code scripts/countermelody_lib.py
```

---

## 📝 重要な設計判断

### 1. lyric_anchors.json構造の扱い
**決定**: LyricAnchorIndexが**time→bar変換＋phrase_role推定**を担当

**理由**:
- lyric_anchors.jsonはtime-basedのみ（bar/beatフィールド無し）
- phrase_boundaryフィールド無し
- **LyricAnchorIndex.get_bar_info()が推定アルゴリズム実装済み**
- 他システム（CREPE、OaF）との統合時も、この層で吸収可能

**影響**:
- ✅ Rulebook側はbar単位のphrase_roleを期待（変更不要）
- ✅ LyricAnchorIndex内部でtime集計→phrase推定（柔軟性確保）

---

### 2. CREPE導入タイミング
**決定**: Phase D（Phase C完了後）

**理由**:
- Phase C（Lyric Anchors深化）完了まで、lyric_anchorsのみで十分
- CREPEは**解析器**として役割限定（生成はRulebook＋OtobonAI）
- vocal_register/motionを追加contextとして扱う（オプション扱い）

**安全弁**:
- vocal_f0データ無し時もエラーにしない（default値使用）
- Rulebook側でvocal_系条件は"あれば使う"（必須にしない）

---

### 3. Counter-Melody vs Main Melody
**決定**: OtobonAIは**Counter-Melodyのみ**担当

**理由**:
- Main MelodyはVocals（歌）が担当
- 伴奏側のメロディ（カウンターメロディ、オブリガート）に専念
- Strings/Piano上声がこの役割

**実装**:
- guide_tone_hints: カウンターメロディの骨組み
- lyric_anchors: ボーカルとの時間軸調整
- CREPE: ボーカルregister/motion参照（衝突回避）

---

## 📚 参考ドキュメント

- ✅ **Phase 2.0完了報告**: `test_phase2_pipeline_summary.md`
- 🔄 **Counter-Melody Spec**: `docs/COUNTER_MELODY_SPEC.md`（作成予定）
- 🔄 **Algorithm詳細**: `docs/COUNTER_MELODY_ALGORITHM.md`（作成予定）
- ✅ **Rulebook Engine**: `otobonAI/rulebook_engine.py`
- ✅ **LyricAnchorIndex**: `otobonAI/lyric_index.py`

---

**作成日**: 2025-11-15  
**ステータス**: Phase A準備完了、視聴テスト待ち  
**次のアクション**: strings_v20.mid視聴テスト → Policy preset決定
