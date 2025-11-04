# Phase 31 拡張実装完了レポート

## 📊 実装サマリ

### ✅ 完了項目

1. **Bass への Phase 31 実装**
   - `BassGeneratorStage2._apply_bass_scale_constraint()` 追加
   - マルチステム対応の生成後に自動適用
   - デフォルト強度: 0.5（控えめ、和声基盤優先）

2. **修正強度調整機能 (`scale_constraint_strength`)**
   - 範囲: 0.0-1.0
   - 0.0 = 修正なし（NO-OP）
   - 0.5 = 50%の確率で修正
   - 1.0 = 必ず修正（厳密なスケール遵守）
   - 楽器別にYAMLで設定可能

3. **コード情報の活用**
   - `chord_root` / `chord_quality` を `scale_mask_for_point()` に渡す
   - chordmap から現在のコード情報を自動抽出
   - コード相対プリセット（ionian_citypop等）でテンション補正

---

## 🎛️ 楽器別推奨強度

| 楽器 | 推奨強度 | 理由 |
|------|---------|------|
| **Piano** | 0.8 | 和声の中心、しっかり修正 |
| **Strings** | 0.85 | レガート優先、より強く修正 |
| **Guitar** | 0.75 | テンション許容、やや控えめ |
| **Bass** | 0.5 | 和声基盤、控えめに修正 |
| **Melody** | 0.9 | 歌メロ、最優先で修正 |

---

## 📁 変更ファイル

### 1. `generator/instrument_stage2_base.py`

**変更内容:**
- `_apply_mode_scale_constraint()` に `strength` パラメータ追加
- コード情報取得ロジック追加（chordmap連携）
- 確率的修正実装（`random.random() > strength` でスキップ）
- `_voice_leading_smooth()` から `cfg.get("scale_constraint_strength", 1.0)` を渡す

**主な変更:**
```python
def _apply_mode_scale_constraint(
    self, 
    part: Any, 
    section_meta: Dict[str, Any],
    strength: float = 1.0  # 新規追加
):
    # 強度0.0ならスキップ
    if strength <= 0.0:
        return
    
    # chordmap から chord_root/chord_quality 取得
    chordmap = self._overrides["mix_context"].get("chordmap")
    
    for n in (part.get("notes") or []):
        # コード情報パース
        chord_root = _parse_chord_root_pc(chord_symbol)
        chord_quality = "maj7" | "min7" | "7" | "maj" | "min"
        
        # Scale Mask取得（コード情報活用）
        mask = scale_mask_for_point(
            t_ql=off_ql,
            sections=sections,
            chord_root=chord_root,      # 新規
            chord_quality=chord_quality  # 新規
        )
        
        # 確率的修正
        if random.random() > strength:
            continue  # 修正しない
```

### 2. `generator/bass_generator_stage2.py`

**変更内容:**
- `_apply_bass_scale_constraint()` メソッド追加（95行）
- Stage2生成後に自動適用
- デフォルト強度: 0.5

**主な追加:**
```python
def _apply_bass_scale_constraint(
    self,
    part: stream.Part,
    section_data: Dict[str, Any],
    strength: float = 0.5  # Bass用デフォルト
):
    """Phase 31: Bass専用 Mode/Scale制約"""
    # Bassは和声の基礎なので控えめに適用
    # chordmap → chord_root/chord_quality 取得
    # scale_mask_for_point() でマスク取得
    # スケール外音を最近接音に修正（確率的）
```

### 3. `PHASE_30_31_YAML_EXAMPLES.yaml`

**変更内容:**
- 各楽器に `scale_constraint_strength` 追加
- 楽器別推奨強度の説明追加
- Bass用の新設定追加

**例:**
```yaml
# Piano
voice_leading:
  enable: true
  max_leap: 7
  scale_constraint_strength: 0.8   # しっかり修正

# Guitar
voice_leading:
  enable: true
  max_leap: 9
  scale_constraint_strength: 0.75  # やや控えめ

# Strings
voice_leading:
  enable: true
  max_leap: 5
  scale_constraint_strength: 0.85  # より強く修正

# Bass（新規）
voice_leading:
  enable: false                     # Voice-Leading不要
  scale_constraint_strength: 0.5    # 控えめに修正
```

---

## 🔧 使用方法

### 1. YAMLプリセットで設定

```yaml
# configs/piano_style_presets.yaml
piano_moderate:
  voice_leading:
    enable: true
    max_leap: 7
    scale_constraint_strength: 0.8
```

### 2. Pythonコードで動的設定

```python
from generator.piano_params_stage2 import PianoParamsStage2

gen = PianoParamsStage2()
params = {
    "voice_leading": {
        "enable": True,
        "max_leap": 7,
        "scale_constraint_strength": 0.8
    }
}

part = gen.compose(section_data=section, overrides=params)
```

### 3. Bass用の設定

```python
from generator.bass_generator_stage2 import BassGeneratorStage2

gen = BassGeneratorStage2(use_stage2=True)

# Bass用デフォルト: strength=0.5
# mix_context に sections/chordmap があれば自動適用
part = gen.compose(section_data=section, shared_tracks=shared_tracks)
```

---

## 🎯 動作フロー

### Piano/Guitar/Strings（既存）

```
generate() 
  → Phase 26 (Pitch Distribution)
  → Phase 31: _voice_leading_smooth()
      ├─ Voice-Leading Guard (max_leap制限)
      └─ _apply_mode_scale_constraint(strength=cfg["scale_constraint_strength"])
          ├─ chordmap から chord_root/chord_quality 取得
          ├─ scale_mask_for_point() で12音マスク取得
          ├─ スケール外音検出（threshold=avg*0.70）
          ├─ random.random() > strength → スキップ
          └─ 最近接スケール内音に修正（±1, ±2半音）
```

### Bass（新規実装）

```
compose()
  → _compose_with_stage2()  # Stage2パターン推薦
  → _apply_existing_processing()  # Humanization/Kick sync
  → _apply_bass_scale_constraint(strength=0.5)  # 新規
      ├─ chordmap から chord_root/chord_quality 取得
      ├─ scale_mask_for_point() でマスク取得
      ├─ スケール外音検出
      ├─ 50%の確率で修正（strength=0.5）
      └─ 最近接音に修正
```

---

## 🧪 テスト方法

### 1. 基本動作テスト

```python
# ops/scale_modes.py の既存テストが全てパス
docker run --rm -v "$(pwd)":/app -w /app composer2 python ops/scale_modes.py

# 出力例:
[Test 1] D Ionian: Mask OK ✓
[Test 2] G Mixolydian: Mask OK ✓
[Test 5] Preset: lydian_shimmer OK ✓
```

### 2. Bass Phase 31 テスト

```python
# Bass生成テスト
docker run --rm -v "$(pwd)":/app -w /app composer2 python generator/bass_generator_stage2.py \
  --tempo 120 --section Verse --measures 4 --output test_bass_phase31.mid

# ログ確認:
# [Bass] Phase 31: 48 → 50 (strength=0.50)  # E → F# に修正
```

### 3. 強度変更テスト

```python
# 強度0.0: 修正なし
gen = PianoParamsStage2()
params = {"voice_leading": {"scale_constraint_strength": 0.0}}
part = gen.compose(section_data=section, overrides=params)

# 強度1.0: 必ず修正
params = {"voice_leading": {"scale_constraint_strength": 1.0}}
part = gen.compose(section_data=section, overrides=params)
```

---

## 📊 コード情報活用の効果

### Before（コード情報なし）

```python
mask = scale_mask_for_point(
    t_ql=16.0,
    sections=sections,
    chord_root=None,      # キーのみ
    chord_quality=None
)
# → Ionian スケールのみ適用
```

### After（コード情報活用）

```python
# chordmap: {"bar": 4, "chord": "Gmaj7"}
mask = scale_mask_for_point(
    t_ql=16.0,
    sections=sections,
    chord_root=7,         # G = PC 7
    chord_quality="maj7"
)
# → Ionian + コード相対9th/13thブースト
# → プリセット "ionian_citypop" の効果適用
```

**効果:**
- コードトーン（1, 3, 5, 7）が自然に強調される
- テンション（9th, 11th, 13th）がコード相対で適切に配置
- プリセット（citypop等）のコード相対ブースト機能が発動

---

## 🎨 実用例

### J-POP Chorus（強めの修正）

```yaml
sections:
  - bar: 43
    label: chorus
    key_hint: D
    preset: ionian_citypop
    blues: 0.12
    code_offsets_mode: chord

piano_moderate:
  voice_leading:
    scale_constraint_strength: 0.85  # 強め
```

**効果:**
- Chorusでしっかりとスケール内に収まる
- コード相対9th/13thブーストでシティポップ感
- blues=0.12で適度な非ダイアトニック許容

### Ballad Verse（控えめの修正）

```yaml
sections:
  - bar: 21
    label: verse
    key_hint: A
    preset: aeolian_dream
    blues: 0.15

strings_simple:
  voice_leading:
    scale_constraint_strength: 0.6  # 控えめ
```

**効果:**
- Verseでは表情豊かな外れ音も許容
- blues=0.15で温かみのあるバラード感
- strength=0.6で60%の確率で修正（適度なバランス）

### Bass（和声基盤優先）

```yaml
bass_tight_pop:
  voice_leading:
    scale_constraint_strength: 0.5  # 控えめ
```

**効果:**
- Bassは和声基盤なので、50%の確率で修正
- ルート音優先、リズム優先の特性を尊重
- スケール外音も半分は許容（クロマチックアプローチ等）

---

## 🚀 今後の拡張

### 1. 楽器別デフォルト強度の自動設定

```python
DEFAULT_STRENGTHS = {
    "piano": 0.8,
    "strings": 0.85,
    "guitar": 0.75,
    "bass": 0.5,
    "melody": 0.9
}

strength = cfg.get("scale_constraint_strength", 
                   DEFAULT_STRENGTHS.get(self.instrument_name, 1.0))
```

### 2. セクション別の動的強度

```yaml
sections:
  - bar: 0
    label: intro
    scale_strength: 0.6  # Intro: 控えめ
  - bar: 43
    label: chorus
    scale_strength: 0.9  # Chorus: 強め
```

### 3. リアルタイム強度調整（UI）

```python
# 再生中に強度スライダーで調整
slider_value = 0.7  # 0.0-1.0
gen.set_scale_constraint_strength(slider_value)
```

---

## ✅ チェックリスト

- [x] Bass への Phase 31 実装
- [x] 修正強度調整機能 (scale_constraint_strength)
- [x] コード情報の活用 (chord_root/chord_quality)
- [x] YAMLプリセット更新
- [x] ドキュメント作成
- [ ] 実曲テスト（song_001）
- [ ] 楽器別デフォルト強度の自動設定
- [ ] セクション別動的強度（将来実装）

---

## 📚 関連ドキュメント

- `PHASE_26_31_INTEGRATION_REPORT.md` - Phase 31基本実装
- `CHORDMAP_WORKFLOW.md` - ChordMap生成ワークフロー
- `README_MODE_SCALE_INTEGRATION.md` - Mode/Scale機能詳細
- `ops/scale_modes.py` - Scale Mask実装（692行、10プリセット）
- `PHASE_30_31_YAML_EXAMPLES.yaml` - YAML設定例

---

**実装完了日**: 2025年10月21日  
**Status**: ✅ Ready for Production Testing
