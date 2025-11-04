# Composer2-3 作曲コマンドガイド（midi_writer統一版）

## 重要な設計方針

### 全楽器midi_writer統一（Thread決定事項）

Composer2-3では、**全楽器（Drums含む）をPlan JSON → midi_writer.py経由で書き出す設計**に統一されています。

**理由**:
- テンポ/拍子メタのTrack 0限定書き込み（956s問題解決）
- 絶対tick→delta変換の統一
- 曲末クリップの自動処理
- Vocal Sync Guard統合

### 非推奨パターン

以下の古いパターンは**使用しないでください**:

```python
# ❌ 非推奨（旧方式）
drum_part = DrumsGeneratorStage2().generate(bars=16, chords=chords, tempo=120)
score.insert(0, drum_part)
score.write("midi", fp="drums.mid")
```

### 推奨パターン

```python
# ✅ 推奨（現行方式）
python scripts/full_pipeline.py \
    --vocal data/suno_ai/.../stem_wav_001_\(Vocals\).wav \
    --accompaniment data/suno_ai/.../stem_wav_001_\(Other\).wav \
    --output output/full_pipeline \
    --tempo 120
```

---

## コマンド一覧

### カテゴリ別機能

| カテゴリ | コマンド数 | 説明 |
|---------|----------|------|
| **Full Pipeline** | 1 | E2E（分析→生成→検証）**最推奨** |
| **Stage1分析** | 2 | chordmap/sections/lyric_anchors生成 |
| **YAML → MIDI** | 1 | 構造YAML → MIDI生成 |
| **Tempo修復** | 1 | 956s問題対策ツール |
| **レンダリング** | 1 | MIDI → WAV |
| **バッチ** | 1 | 一括処理 |
| **A/B比較** | 1 | 品質比較 |
| **デモ** | 1 | クイックスタート |

---

## 詳細リファレンス

### Full Pipeline（最推奨）

**`compose_full_pipeline`**

**フロー**:
```
Suno Stem → Stage1分析 → Plan生成 → midi_writer → 
Vocal Sync Guard → KPI Gate → WAVレンダリング
```

**パラメータ**:
```bash
compose_full_pipeline \
    <vocal_wav> \          # Vocal WAV
    <accompaniment_wav> \  # Accompaniment WAV
    <output_dir> \         # Output directory
    <tempo>                # BPM (default: 120)
```

**実行例**:
```bash
compose_full_pipeline \
    data/suno_ai/suno_themesong/song_001/stemswav_001/stem_wav_001_\(Vocals\).wav \
    data/suno_ai/suno_themesong/song_001/stemswav_001/stem_wav_001_\(Other\).wav \
    output/full_pipeline \
    120
```

---

### Stage1分析

**`analyze_suno_song`**

**出力**:
- `chordmap.json` - コード進行（7th chords対応）
- `sections.json` - セクション構造 + tempo_map
- `lyric_anchors.json` - 歌詞タイムライン

**実行例**:
```bash
analyze_suno_song \
    data/suno_ai/suno_themesong/song_001 \
    C
```

---

### Tempo修復ツール

**`inject_tempo_track`**

**目的**: 既存MIDIの956s問題（Track 0以外のテンポメタ混入）を修復

**固定BPM版**:
```bash
inject_tempo_track \
    output/full_pipeline/full_arrangement.mid \
    output/full_pipeline/full_arrangement_fixed.mid \
    bpm \
    120
```

**sections.json使用版**:
```bash
inject_tempo_track \
    output/full_pipeline/full_arrangement.mid \
    output/full_pipeline/full_arrangement_fixed.mid \
    map \
    "" \
    data/suno_ai/suno_themesong/song_001/analysis/sections.json
```

---

## トラブルシューティング

### DrumsGeneratorStage2 generate() エラー

**原因**: 旧方式のジェネレーター直接呼び出し

**解決策**: `compose_full_pipeline`を使用

```bash
# ❌ 直接呼び出し（エラー）
python scripts/suno_stem_arranger.py --input ... --output ...

# ✅ Full Pipeline経由
compose_full_pipeline <vocal_wav> <accompaniment_wav> <output_dir> <tempo>
```

---

### 956s問題（MIDI長さが異常）

**原因**: Track 0以外にテンポメタが混入

**解決策**: `inject_tempo_track`で修復

```bash
inject_tempo_track \
    output/full_pipeline/full_arrangement.mid \
    output/full_pipeline/full_arrangement_fixed.mid \
    bpm \
    120
```

---

### music21.instrument.StringEnsemble エラー

**原因**: music21に存在しないクラス

**解決策**: midi_writer.pyがGM Program番号で楽器指定（自動処理）

---

## 関連ドキュメント

- [`docs/PRODUCTION_PIPELINE_GUIDE.md`](PRODUCTION_PIPELINE_GUIDE.md ) - Full Pipeline詳細
- [`docs/RELEASE_v1.0.0_COMPLETE.md`](RELEASE_v1.0.0_COMPLETE.md ) - Emotion AI
- [`docs/FINAL_INTEGRATION_REPORT.md`](FINAL_INTEGRATION_REPORT.md ) - Stage2統合