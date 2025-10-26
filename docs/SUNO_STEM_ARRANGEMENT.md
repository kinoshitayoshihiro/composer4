# 🎵 Suno AI Stem分離データからのアレンジ生成ガイド

## 📋 現在の実現可能性まとめ

### ✅ **実装済み（すぐ使える）**

1. **Stem WAV → MIDI変換**
   - `audio_to_midi_batch.py` でPiano/Bass/Guitar等を自動MIDI化
   - `drumstem_to_midi.py` でDrum専用の精密変換

2. **5つの楽器ジェネレーター**
   - Drums: `DrumsGeneratorStage2` (パターンベース、品質ゲート完備)
   - Bass: `BassGenerator`
   - Piano: `PianoGenerator`
   - Guitar: `GuitarGenerator`
   - Strings: `StringsGenerator`

### ⚠️ **実装途中（手動作業が必要）**

1. **統合制御システム** (`scripts/suno_stem_arranger.py` - 今作成)
   - 現在: Drumsのみ自動生成可能
   - 残作業: Bass/Piano/Guitar/Stringsの統合

2. **コード進行自動推定**
   - 現在: 手動でコード指定が必要
   - 残作業: Piano/Guitar stemからコード推定ロジック

---

## 🚀 クイックスタート

### **ステップ1: Stem WAV準備**

```bash
# Sunoから生成したstemがあることを確認
ls /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/suno_ai/suno_themesong/song_001/stemswav_001

# 出力例:
# stem_wav_001_(Bass).wav
# stem_wav_001_(Drums).wav
# stem_wav_001_(Guitar).wav
# stem_wav_001_(Keyboard).wav
# stem_wav_001_(Vocals).wav
# ...
```

### **ステップ2: 自動アレンジ実行（Drums生成）**

```bash
# 仮想環境アクティベート
source .venv311/bin/activate

# Drumsのみ自動生成（現在実装済み）
python scripts/suno_stem_arranger.py \
    --input data/suno_ai/suno_themesong/song_001/stemswav_001 \
    --output data/arranged_midi \
    --tempo 120 \
    --emotion energetic \
    --bars 16
```

**出力:**
```
INFO: Found 10 stems: ['drums', 'bass', 'guitar', 'keyboard', 'vocals', ...]
WARNING: Chord extraction not fully implemented - using C-G-Am-F
INFO: Generating drums...
✅ Drums: 256 notes
💾 Saved to: data/arranged_midi/stemswav_001_arranged.mid
```

---

## 📝 詳細ワークフロー

### **方法A: 統合スクリプト（簡易版）**

現在の `suno_stem_arranger.py` でできること:

```python
# 1. Stem自動検出
stem_files = {
    'drums': 'stem_wav_001_(Drums).wav',
    'bass': 'stem_wav_001_(Bass).wav',
    'guitar': 'stem_wav_001_(Guitar).wav',
    # ...
}

# 2. Drumsのみ自動生成
drum_part = DrumsGeneratorStage2().generate(
    bars=16,
    chords=["C", "G", "Am", "F"] * 4,  # 仮コード
    tempo=120,
    emotion="energetic"
)
```

**制限事項:**
- ✅ Drums: 完全自動
- ⚠️ Bass/Piano/Guitar/Strings: ジェネレーター呼び出し部分がコメントアウト済み

---

### **方法B: 手動制御（全5楽器使用）**

より詳細な制御が必要な場合:

```python
#!/usr/bin/env python3
"""手動制御サンプル"""

from pathlib import Path
from music21 import stream, tempo as m21tempo

# ジェネレーターインポート
from generator.drums_generator_stage2 import DrumsGeneratorStage2
from generator.bass_generator import BassGenerator
from generator.piano_generator import PianoGenerator
from generator.guitar_generator import GuitarGenerator
from generator.strings_generator import StringsGenerator

# 設定読み込み
from utilities.config_loader import load_main_cfg
main_cfg = load_main_cfg(Path("configs/main_config.yaml"))

# ジェネレーター初期化（要パラメータ設定）
drums = DrumsGeneratorStage2()
# bass = BassGenerator(
#     global_settings=main_cfg['global_settings'],
#     main_cfg=main_cfg,
#     # ... 他のパラメータ
# )

# コード進行（手動設定）
chords = ["Cm", "Ab", "Eb", "Bb"] * 4  # 16小節

# Score作成
score = stream.Score()
score.insert(0, m21tempo.MetronomeMark(number=110))

# 1. Drums生成
drum_part = drums.generate(
    bars=16,
    chords=chords,
    tempo=110,
    emotion="melancholic",
    section="Verse"
)
score.insert(0, drum_part)

# 2. Bass生成
# bass_part = bass.compose(
#     section_data={
#         'section_name': 'Verse',
#         'processed_chord_events': [...],
#         'musical_intent': {'emotion': 'melancholic'}
#     }
# )
# score.insert(0, bass_part)

# 3-5. Piano/Guitar/Strings生成
# ... (同様)

# MIDI出力
score.write('midi', fp='out/manual_arrangement.mid')
```

---

## 🔧 完全自動化のために必要な作業

### **Todo #6: Bass/Piano/Guitar/Strings統合**

`scripts/suno_stem_arranger.py` の以下の部分を実装:

```python
# 現在コメントアウト中
def arrange_with_generators(self, chords, tempo, emotion, bars):
    # ... (Drums実装済み)
    
    # TODO: Bass生成
    logger.info("Generating bass...")
    bass_part = self.generators['bass'].compose(
        section_data={
            'section_name': 'Verse',
            'processed_chord_events': self._build_chord_events(chords, tempo),
            'musical_intent': {'emotion': emotion},
            'part_params': {
                'bass': {
                    'pattern_type': 'walking' if emotion == 'energetic' else 'root',
                    'density': 'medium',
                }
            }
        }
    )
    score.insert(0, bass_part)
    
    # TODO: Piano生成
    # TODO: Guitar生成
    # TODO: Strings生成
```

**課題:**
- 各ジェネレーターの初期化パラメータが複雑
- `section_data` の構造が各ジェネレーターで異なる
- `main_cfg` の完全な設定が必要

---

### **Todo #7: コード進行自動推定**

Piano/Guitar stemからコード推定:

```python
def extract_chords_from_stems(self, stem_files: Dict[str, Path], bars: int):
    """Piano/Guitar stemからコード推定"""
    
    # 1. Piano/Guitar WAV → MIDI変換
    if 'piano' in stem_files:
        piano_midi = self._transcribe_to_midi(stem_files['piano'])
    elif 'guitar' in stem_files:
        guitar_midi = self._transcribe_to_midi(stem_files['guitar'])
    else:
        return self._default_progression(bars)
    
    # 2. MIDIからコード推定（music21 or madmom使用）
    chords = []
    # TODO: 実装
    # - 和音検出
    # - コードラベル推定
    # - 小節単位で整理
    
    return chords
```

**実装オプション:**
- `music21.analysis.chordify()` (簡易)
- `madmom.features.chords` (高精度、要インストール)
- LAMDa統合 (`lamda_unified_analyzer.py`)

---

## 📊 現在の制約

### **1. ジェネレーターの初期化が複雑**

各ジェネレーターは `main_cfg` の完全な設定が必要:

```yaml
# configs/main_config.yaml (一部)
global_settings:
  tempo_bpm: 120
  time_signature: "4/4"
  key_tonic: "C"
  key_mode: "major"

part_defaults:
  drums:
    role: "drums"
    part_parameters:
      # ... 詳細設定
  
  bass:
    role: "bass"
    part_parameters:
      # ... 詳細設定
```

**解決策:**
- デフォルト設定ファイル作成
- または `GenFactory.build_from_config()` 使用

### **2. Stem→コード推定が未実装**

現在は手動でコード進行を指定:

```python
# 仮実装
chords = ["C", "G", "Am", "F"] * 4
```

**理想:**
```python
# Piano stemから自動推定
chords = extract_chords_from_piano_stem("piano.wav")
# → ["Cm", "Ab", "Eb", "Bb", ...]
```

---

## 🎯 推奨アプローチ

### **現時点で最も実用的な方法**

```bash
# ステップ1: Drumsのみ自動生成
python scripts/suno_stem_arranger.py \
    --input data/suno_ai/suno_themesong/song_001/stemswav_001 \
    --output data/arranged_midi \
    --tempo 120 \
    --emotion energetic \
    --bars 16

# ステップ2: 他の楽器は既存ツールで生成
python modular_composer.py \
    --config configs/my_arrangement.yaml \
    --chordmap configs/my_chords.yaml \
    --output data/full_arrangement.mid
```

**メリット:**
- Drumsは品質ゲート完備の高精度生成
- Bass/Piano/Guitar/Stringsは既存の安定した生成パイプライン使用
- 段階的に統合可能

---

## 📚 関連ドキュメント

- [COMPLETE_MUSIC_ARCHITECTURE.md](COMPLETE_MUSIC_ARCHITECTURE.md) - 全体アーキテクチャ
- [INSTRUMENT_GENERATOR_PLAN.md](INSTRUMENT_GENERATOR_PLAN.md) - 5楽器ジェネレーター計画
- [LAMDA_README.md](LAMDA_README.md) - LAMDaデータセット統合
- [TODO5_QUALITY_GATE_SUCCESS.md](TODO5_QUALITY_GATE_SUCCESS.md) - Drums品質ゲート

---

## 🔮 将来の拡張

### **Phase 3: Suno完全統合**

```python
class SunoFullIntegration:
    """Suno 12ステム完全統合"""
    
    def process_suno_export(self, suno_dir: Path):
        """12ステム全自動処理"""
        
        stems = {
            'vocals': self._process_vocal_stem(),
            'drums': self._process_drum_stem(),
            'bass': self._process_bass_stem(),
            'guitar': self._process_guitar_stem(),
            'keyboard': self._process_keyboard_stem(),
            'strings': self._process_strings_stem(),
            'synth': self._process_synth_stem(),
            'fx': self._process_fx_stem(),
            # ... 全12ステム
        }
        
        # 統合アレンジ
        return self._unified_arrangement(stems)
```

---

## ❓ FAQ

### Q1: 今すぐ5楽器全部使える？

**A:** Drumsのみフル自動。他は手動統合が必要（上記「方法B」参照）

### Q2: コード進行は自動推定できる？

**A:** 未実装。現在は手動指定またはデフォルトコード（C-G-Am-F）使用

### Q3: Sunoの12ステム全部対応してる？

**A:** Drums/Bass/Guitar/Piano/Stringsの5種のみ。Vocals/FX等は未対応

### Q4: 既存の `modular_composer.py` との違いは？

**A:** 
- `modular_composer.py`: 完全な楽曲生成（コード進行から全自動）
- `suno_stem_arranger.py`: Suno stemを活用したアレンジ追加

---

## 📞 サポート

問題が発生した場合:

1. ログ確認: `--verbose` フラグで詳細ログ
2. テストデータで検証: `data/suno_ai/test_stems` (作成推奨)
3. Issue報告: 再現手順とログを添付

---

**最終更新:** 2025-10-18  
**ステータス:** Drums自動生成完了 (50%)、Bass/Piano/Guitar/Strings統合待ち
