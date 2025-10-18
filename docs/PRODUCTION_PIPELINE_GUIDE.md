# Complete Production Pipeline Guide

ChatGPT提案の完全実装：Suno AI → 構造抽出 → MIDI生成 → WAV変換

---

## 📋 ワークフロー概要

```
① Suno AI で作詞 → Vocal + 伴奏生成
          ↓
② 伴奏をstem wavに分離 → システムに流す
          ↓
③ 構造抽出（テンポ、セクション、コード、ドラム、ベース）
          ↓
④ 高精度MIDI生成（奏法差し替え対応）
          ↓
⑤ WAV変換（pretty_midi + FluidSynth or VST）
          ↓
⑥ Vocal stemと合成 → 新曲完成
          ↓
⑦ TuneCore Japanで配信
```

---

## 🚀 基本使用方法

### 1. 必要なファイル準備

```bash
# Suno AIからダウンロードしたstem
stems/
  ├── vocal.wav          # ボーカルトラック
  └── accompaniment.wav  # 伴奏トラック
```

### 2. 完全パイプライン実行

```bash
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01 \
  --technique-map configs/technique_map_example.yaml \
  --soundfont soundfonts/GeneralUser_GS.sf2 \
  --seed 42
```

### 3. 出力ファイル

```
output/song01/
├── structure/
│   └── structure.yaml      # 抽出した構造データ
├── midi/
│   ├── guitar_strum.mid    # ギターMIDI
│   └── strings_legato.mid  # ストリングスMIDI
├── audio/
│   ├── guitar.wav          # ギターWAV
│   └── strings.wav         # ストリングスWAV
└── reports/
    ├── vocal_sync.json     # Vocal同期レポート
    └── pipeline_report.json # 全体レポート
```

---

## 🎸 奏法差し替え例

### 例1: Chorusでアルペジオ→ストラムに変更

`configs/my_technique_map.yaml`を作成：

```yaml
sections:
  Chorus:
    guitar:
      technique: "strum"           # 原曲がarpeggioでもstrumに
      rhythm_key: "strum_8ths_updown"
      strum_spread_ms: 12
      strum_direction_bias: 0.6
      emotion: "energetic"
```

実行：

```bash
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01 \
  --technique-map configs/my_technique_map.yaml
```

### 例2: Stringsをlegato→pizzicatoに変更

```yaml
sections:
  Verse:
    strings:
      technique: "pizzicato"      # legatoからpizzicatoへ
      rhythm_key: "pizz_8ths"
      notes_per_bar_multiplier: 0.7
      emotion: "playful"
```

---

## 📊 構造抽出の詳細

### 抽出される5つの要素

1. **tempo_map**: テンポ変化（BPM列）
   ```yaml
   tempo_map:
     - {beat: 0.0, bpm: 120.0}
     - {beat: 64.0, bpm: 125.0}
   ```

2. **sections**: セクション境界
   ```yaml
   sections:
     - {name: Intro, start: 0.0, end: 8.0, bars: 4}
     - {name: Verse, start: 8.0, end: 24.0, bars: 8}
   ```

3. **chords**: コード進行
   ```yaml
   chords:
     Verse: [C, G, Am, F]
     Chorus: [F, G, Em, Am]
   ```

4. **drums_hits**: ドラム打点
   ```yaml
   drums_hits:
     kick: [0.0, 2.0, 4.0, 6.0]
     snare: [2.0, 6.0, 10.0, 14.0]
   ```

5. **bass_contour**: ベース輪郭
   ```yaml
   bass_contour:
     - {start: 0.0, f0_hz: 65.4, dur_beats: 1.0}
   ```

---

## 🔍 Vocal Sync Guard（同期検証）

### 自動実行（デフォルト）

```bash
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01 \
  --max-drift-ms 30.0
```

### スキップ（構造を変更しない場合）

```bash
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01 \
  --skip-vocal-sync
```

### レポート確認

```json
{
  "detected_offset_ms": -12.4,
  "max_drift_ms": 18.7,
  "p95_drift_ms": 14.2,
  "section_outliers": [],
  "action": "offset_shift_only",
  "status": "OK"
}
```

---

## 🎛️ 高度な設定

### 1. 特定の抽出メソッドのみ実行

```bash
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01 \
  --extraction-methods tempo_map sections chords
```

### 2. SoundFont指定

```bash
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01 \
  --soundfont /path/to/YourSoundFont.sf2
```

### 3. 再現性確保（seed固定）

```bash
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01 \
  --seed 42
```

---

## 📈 品質ゲート（ChatGPT提案実装）

### Guitar

- **strum_consistency** ≥ 0.75（ストラム一貫性）
- **bar_violation_rate** ≤ 0.02（小節境界逸脱）
- **velocity_std** ∈ [12, 35]（ダイナミクス）

### Strings

- **legato_connection_rate** ≥ 0.65（レガート連結率）
- **chord_spread_semitones** ≤ 24（コード音域）
- **bar_violation_rate** ≤ 0.02

### 共通

- **max_drift_ms** ≤ 30.0（Vocal同期ドリフト）
- **grid_off_std_ms** ≤ 12.0（グリッドずれ）

---

## 🔧 トラブルシューティング

### Q1: MIDIファイルが生成されない

```bash
# ログを詳細表示
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01 \
  --verbose
```

### Q2: WAV変換が失敗する

```bash
# SoundFontなしでfallback合成を使用
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01
  # --soundfont を指定しない
```

### Q3: Vocal同期がずれる

```bash
# しきい値を緩和
python scripts/full_pipeline.py \
  --vocal stems/vocal.wav \
  --accompaniment stems/accompaniment.wav \
  --output output/song01 \
  --max-drift-ms 50.0
```

---

## 📝 実装済み機能一覧

### ✅ 完全実装済み

- [x] 構造抽出（5メソッド）
- [x] MIDI生成（Guitar/Strings Stage2）
- [x] 奏法差し替え（strum ↔ arpeggio等）
- [x] WAV変換（pretty_midi + FluidSynth）
- [x] Vocal Sync Guard（同期検証）
- [x] レポート生成（JSON）
- [x] 品質ゲート（メトリクス検証）

### 🔧 今後の拡張

- [ ] Piano/Bass Stage2統合
- [ ] Drums Stage2統合
- [ ] VST統合（DAWdreamer完全活用）
- [ ] 並列レンダリング（マルチプロセス）
- [ ] リアルタイムプレビュー

---

## 📚 関連ドキュメント

- [FINAL_INTEGRATION_REPORT.md](../FINAL_INTEGRATION_REPORT.md) - 完全統合レポート
- [tests/test_e2e_yaml_to_wav.py](../tests/test_e2e_yaml_to_wav.py) - E2Eテスト実装
- [tests/test_technique_switch.py](../tests/test_technique_switch.py) - 奏法差し替えテスト

---

## 🎉 使用例：実際の楽曲製作

### シナリオ: ポップソング制作

```bash
# 1. Suno AIでボーカル＋伴奏生成
# → stems/my_song_vocal.wav
# → stems/my_song_accompaniment.wav

# 2. 完全パイプライン実行
python scripts/full_pipeline.py \
  --vocal stems/my_song_vocal.wav \
  --accompaniment stems/my_song_accompaniment.wav \
  --output output/my_song \
  --technique-map configs/technique_map_pop.yaml \
  --soundfont soundfonts/GeneralUser_GS.sf2 \
  --seed 12345

# 3. 出力確認
ls output/my_song/audio/
# → guitar.wav
# → strings.wav

# 4. DAWで最終ミックス
# - output/my_song/audio/*.wav をインポート
# - stems/my_song_vocal.wav を追加
# - マスタリング
# - 完成！
```

---

**作成日**: 2025年10月18日  
**プロジェクト**: composer2-3  
**ChatGPT提案**: 完全実装済み
