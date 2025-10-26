# Composer2-3 クイックスタートガイド

このガイドでは、Composer2-3の基本的な使い方から高度な機能まで、段階的に説明します。

## 目次

1. [基本的な使い方](#基本的な使い方)
2. [Stage2 生成（全楽器）](#stage2-生成全楽器)
3. [スタイルプリセットの使用](#スタイルプリセットの使用)
4. [Phase 22-24: 高度な表現制御](#phase-22-24-高度な表現制御)
5. [Suno AI Stem統合](#suno-ai-stem統合)

---

## 基本的な使い方

### 1. 環境セットアップ

```bash
# 依存関係のインストール
bash install_deps.sh --dev

# または Python仮想環境で
python -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

### 2. シンプルな生成例

```python
from generator.bass_params_stage2 import BassParamsStage2

# 基本パラメータ
params = {
    "chords": ["C", "Am", "F", "G"],  # コード進行
    "tempo": 120,                      # BPM
    "seed": 42                         # 再現性のためのシード
}

# Bass生成
bass = BassParamsStage2()
bass.apply(params)
track = bass.generate()

# MIDI保存
track.write("midi", "output/bass.mid")
```

---

## Stage2 生成（全楽器）

### 対応楽器

- **Bass**: `BassParamsStage2`
- **Piano**: `PianoParamsStage2`
- **Guitar**: `GuitarParamsStage2`
- **Strings**: `StringsParamsStage2`
- **Drums**: `DrumsParamsStage2`

### 全楽器を統合生成

```python
from generator.bass_params_stage2 import BassParamsStage2
from generator.piano_params_stage2 import PianoParamsStage2
from generator.guitar_params_stage2 import GuitarParamsStage2
from generator.strings_params_stage2 import StringsParamsStage2
from generator.drums_params_stage2 import DrumsParamsStage2
import music21 as m21

# 共通パラメータ
chords = ["C", "Am", "F", "G"]
tempo = 120

# 各楽器を生成
bass = BassParamsStage2()
bass.apply({"chords": chords, "tempo": tempo})

piano = PianoParamsStage2()
piano.apply({"chords": chords, "tempo": tempo})

guitar = GuitarParamsStage2()
guitar.apply({"chords": chords, "tempo": tempo})

strings = StringsParamsStage2()
strings.apply({"chords": chords, "tempo": tempo})

drums = DrumsParamsStage2()
drums.apply({"chords": chords, "tempo": tempo})

# Score統合
score = m21.stream.Score()
score.insert(0, bass.generate())
score.insert(0, piano.generate())
score.insert(0, guitar.generate())
score.insert(0, strings.generate())
score.insert(0, drums.generate())

# MIDI保存
score.write("midi", "output/full_arrangement.mid")
```

---

## スタイルプリセットの使用

### プリセットファイル

各楽器に4つのスタイルプリセットが用意されています：

- `configs/bass_style_presets.yaml`
- `configs/piano_style_presets.yaml`
- `configs/guitar_style_presets.yaml`
- `configs/strings_style_presets.yaml`
- `configs/drums_style_presets.yaml`

### スタイル

- **simple**: シンプル・基本的（密度低、装飾少）
- **moderate**: 標準的（バランス良好）
- **complex**: 複雑（高密度、装飾多）
- **intense**: 激しい（最高密度、強いアクセント）

### プリセットの読み込み例

```python
import yaml
from generator.bass_params_stage2 import BassParamsStage2

# プリセット読み込み
with open("configs/bass_style_presets.yaml") as f:
    presets = yaml.safe_load(f)

# "moderate"スタイルを使用
moderate_preset = presets["moderate"]

# パラメータ設定
params = {
    "chords": ["C", "Am", "F", "G"],
    "tempo": 120,
    **moderate_preset  # プリセットを展開
}

# 生成
bass = BassParamsStage2()
bass.apply(params)
track = bass.generate()
track.write("midi", "output/bass_moderate.mid")
```

### スタイル比較

```python
import yaml

with open("configs/piano_style_presets.yaml") as f:
    presets = yaml.safe_load(f)

for style in ["simple", "moderate", "complex", "intense"]:
    params = {
        "chords": ["C", "F", "G", "C"],
        "tempo": 120,
        "seed": 42,
        **presets[style]
    }
    
    piano = PianoParamsStage2()
    piano.apply(params)
    track = piano.generate()
    track.write("midi", f"output/piano_{style}.mid")
```

---

## Phase 22-24: 高度な表現制御

### Phase 22: Emotion Mapping（感情連続写像）

感情カーブ `E(t) ∈ [0..1]` を音楽パラメータに連続写像：

```python
params = {
    "chords": ["C", "F", "G", "C"],
    "emotion_curve": [0.3, 0.5, 0.7, 0.9],  # 徐々に感情上昇
    "tempo": 120,
    
    # Emotion mapping設定
    "emotion_map": {
        "density_gain": 0.6,      # 感情に応じた密度増加（0.0-1.0）
        "register_shift": 2,      # 感情に応じた音域シフト（0-4半音）
        "staccato_bias": 0.15,    # 感情に応じたスタッカート確率（0.0-0.3）
        "smooth_ms": 180          # 平滑化ウィンドウ（100-300ms）
    }
}

bass = BassParamsStage2()
bass.apply(params)
track = bass.generate()

# E(t)が高い後半で密度増加・Velocity上昇・音域上昇
```

**効果**:
- **E(t)高い** → Velocity↑（±12）、密度↑、音域↑（±4半音）、スタッカート増加
- **E(t)低い** → Velocity↓、密度↓、音域↓、レガート優先

### Phase 24: Controls Unified（CC/RPN/PB統一）

統一されたMIDI制御実装：

```python
params = {
    "chords": ["C", "Am", "F", "G"],
    "tempo": 120,
    
    # Controls統一設定
    "controls": {
        "expression_curve": "arch",     # CC11表情カーブ: arch | linear | flat
        "sustain_policy": "pad_only",   # CC64ペダル: off | pad_only | always
        "bend_range": 2                 # Pitch Bend範囲（1-12半音）
    }
}

piano = PianoParamsStage2()
piano.apply(params)
track = piano.generate()

# CC11: arch（クレッシェンド→デクレッシェンド）
# CC64: pad_only（パッド系和音のみペダル使用）
# RPN: Pitch Bend Sensitivity=2（1回のみ書き込み）
```

**expression_curve**:
- `arch`: クレッシェンド→デクレッシェンド（劇的）
- `linear`: 徐々にクレッシェンド（自然）
- `flat`: 一定（静的）

**sustain_policy**:
- `off`: ペダルなし
- `pad_only`: パッド系和音のみペダル（Piano推奨）
- `always`: 常時ペダル（Stringsなど）

### Phase 23: Prosody Alignment（韻律整合）

ボーカルの韻律（子音・強勢）に楽器を同期：

```python
params = {
    "chords": ["C", "F", "G", "C"],
    "tempo": 120,
    
    # Prosody整合設定
    "prosody": {
        "enable": True,            # 有効化
        "stress_boost": 8,         # 強勢でVelocityブースト（0-15）
        "sibilant_duck_db": -3,    # 歯擦音で高域減衰（-6～0 dB）
        "plosive_gap_ms": 40,      # 破裂音で隙間作成（20-80ms）
        "window_ms": 120           # 韻律ウィンドウサイズ（80-200ms）
    }
}

strings = StringsParamsStage2()
strings.apply(params)
track = strings.generate()

# ボーカルの"s"音でStrings Vel↓、"p"音で隙間、強勢でVel↑
```

**効果**:
- **stress（強勢）**: Velocity↑（+10程度）
- **sibilant（歯擦音 "s"/"sh"）**: 高域Velocity↓（-6程度）
- **plosive（破裂音 "p"/"t"/"k"）**: ノート長短縮（40ms隙間）

### 統合例: Phase 22-24すべて使用

```python
params = {
    "chords": ["C", "Am", "F", "G"],
    "emotion_curve": [0.3, 0.6, 0.8, 0.9],
    "tempo": 120,
    
    # Phase 22: Emotion mapping
    "emotion_map": {
        "density_gain": 0.6,
        "register_shift": 2,
        "staccato_bias": 0.15,
        "smooth_ms": 180
    },
    
    # Phase 24: Controls
    "controls": {
        "expression_curve": "arch",
        "sustain_policy": "pad_only",
        "bend_range": 2
    },
    
    # Phase 23: Prosody
    "prosody": {
        "enable": True,
        "stress_boost": 8,
        "sibilant_duck_db": -3,
        "plosive_gap_ms": 40,
        "window_ms": 120
    }
}

# 全楽器に適用可能
for cls in [BassParamsStage2, PianoParamsStage2, GuitarParamsStage2, StringsParamsStage2]:
    gen = cls()
    gen.apply(params)
    track = gen.generate()
    instrument_name = cls.__name__.replace("ParamsStage2", "").lower()
    track.write("midi", f"output/{instrument_name}_advanced.mid")
```

### プリセットでのPhase 22-24設定

すべてのスタイルプリセットにPhase 22-24設定が含まれています：

```yaml
# configs/bass_style_presets.yaml の例
moderate:
  # ... (Phase 11-20の設定)
  
  # Phase 22: Emotion mapping
  emotion_map:
    density_gain: 0.6
    register_shift: 2
    staccato_bias: 0.15
    smooth_ms: 180
  
  # Phase 24: Controls
  controls:
    expression_curve: linear
    sustain_policy: off
    bend_range: 2
  
  # Phase 23: Prosody
  prosody:
    enable: true
    stress_boost: 8
    sibilant_duck_db: -3
    plosive_gap_ms: 40
    window_ms: 120
```

**NO-OP安全**: これらの設定を省略すると、過去のバージョンと完全に同じ動作になります（後方互換性）。

---

## Suno AI Stem統合

### Suno生成Stemの解析

```python
from analysis.stem_harmony import (
    make_beat_grid, estimate_activity,
    estimate_chords_per_stem, aggregate_stem_chords
)

# Stem WAVファイル
stems = {
    "drums": "stems/drums.wav",
    "bass": "stems/bass.wav",
    "guitar": "stems/guitar.wav",
    "vocals": "stems/vocals.wav"
}

# 1. ビートグリッド生成
beat_grid = make_beat_grid(stems, default_bpm=120.0, time_sig=(4, 4))

# 2. 活動マスク抽出（各小節の活動度 0..1）
activity = {
    role: estimate_activity(path, beat_grid)
    for role, path in stems.items() if role != "vocals"
}

# 3. コード推定（各Stemからコード候補抽出）
stem_votes = {
    role: estimate_chords_per_stem(path, beat_grid, role, key_hint="C:maj")
    for role, path in stems.items() if role not in ("vocals", "backing_vocals")
}

# 4. コード集約（重み付き投票）
audio_chordmap = aggregate_stem_chords(
    stem_votes, activity, key_hint="C:maj",
    sections=[], cfg={"weights": {"bass": 0.35, "guitar": 0.35, "piano": 0.2}}
)

# 5. 新規MIDI生成（元のVocalを保持しながら楽器を再生成）
overrides = {
    "mix_context": {
        "beat_grid": beat_grid,
        "activity": activity
    },
    "audio_chordmap": audio_chordmap
}

bass = BassParamsStage2()
bass.apply({"chords": [], "tempo": 120, **overrides})
track = bass.generate()
track.write("midi", "output/bass_from_stems.mid")
```

### ハーモニーソースモード

```yaml
harmony:
  source: audio          # audio | text | hybrid
  fallback: text         # オーディオ解析失敗時のフォールバック
  keep_audio_root: true  # 元のベース音を優先
  prefer_root5: true     # ルート/5thを優先
```

- **audio**: Suno Stemのコード進行をそのまま使用（推奨）
- **text**: ChatGPT生成コードマップを使用
- **hybrid**: オーディオのルート + テキストのテンション

---

## テスト実行

### Phase 22-24統合テスト

```bash
# Phase 22-24の動作確認
python scripts/test_phase_22_24_23.py

# 出力例：
# Phase 22: Emotion mapping tests...
# Phase 24: Controls unified tests...
# Phase 23: Prosody alignment tests...
# NO-OP regression tests...
# 実行: 15, 成功: 15, 失敗: 0
```

### Stage2生成テスト

```bash
# 全楽器の統合生成テスト
source .venv311/bin/activate
python scripts/stage2_production_test.py \
    --output data/stage2_test_output \
    --bars 8 \
    --tempo 120
```

---

## トラブルシューティング

### Phase 22-24が有効にならない

**原因**: `emotion_map`/`controls`/`prosody`キーが設定されていない

**解決**:
```python
# これらのキーを明示的に設定
params = {
    "chords": ["C", "F", "G", "C"],
    "tempo": 120,
    "emotion_map": {...},  # 必須
    "controls": {...},     # 必須
    "prosody": {...}       # オプション
}
```

### RPN/CC11が重複する

**原因**: Phase 24が複数回実行されている

**解決**: Phase 24は自動的に重複防止フラグ（`self._rpn_written`）を使用します。通常は問題ありません。

### Prosodyが効果を発揮しない

**原因**: `prosody.enable: false` または `emotion_curve` 未設定

**解決**:
```python
params = {
    "chords": ["C", "F", "G", "C"],
    "emotion_curve": [0.3, 0.5, 0.7, 0.9],  # 必須
    "tempo": 120,
    "prosody": {
        "enable": True,  # 必ずTrueに設定
        "stress_boost": 8,
        # ...
    }
}
```

---

## 次のステップ

- **[README.md](README.md)**: システム全体のアーキテクチャ
- **[PHASE_22_24_23_IMPLEMENTATION.md](docs/PHASE_22_24_23_IMPLEMENTATION.md)**: Phase 22-24実装詳細
- **[STEM_HARMONY_IMPLEMENTATION.md](STEM_HARMONY_IMPLEMENTATION.md)**: Stem解析実装詳細
- **[configs/](configs/)**: スタイルプリセットYAMLファイル

---

**Happy Composing! 🎵**
