# Stage1 ファイル一覧と用途

## 📋 概要

**Stage1の目的**: Suno AIステムWAV → メタデータJSON生成（chordmap/sections/mix_context）

**Stage2の目的**: メタデータ + rhythm_library → 各楽器MIDI生成（modular_composer経由）

---

## 🎵 既存のStage1関連ファイル

### 1. コア解析モジュール

#### **analysis/stem_harmony.py**
- **目的**: ステムWAVから音楽的特徴を抽出
- **主要機能**:
  - `make_beat_grid()` - ビートグリッド生成（BPM/拍子検出）
  - `estimate_activity()` - 楽器別活動マスク（bar単位のenergy 0..1）
  - `estimate_chords_per_stem()` - 各ステムからコード候補抽出
  - `aggregate_stem_chords()` - 複数ステム投票→最終chordmap統合
  - `extract_accent_grid()` - キック/スネア/ハイハット位置抽出
  - `export_guide_midi()` - テンポ/マーカー/ブロックコードMIDI出力
- **出力形式**:
  ```python
  beat_grid: {"bpm": 120, "time_sig": [4,4], "ql_per_bar": 4.0, "beats": [...], "bars": [...]}
  activity: {role: [(bar, energy), ...]}
  chordmap: {time_ql: {"root": "C", "quality": "major", "tones_midi": [60,64,67]}}
  ```

### 2. 統合スクリプト

#### **scripts/suno_stem_arranger.py** (1617行)
- **目的**: ステムWAV → MIDI一括生成（Stage1+Stage2統合）
- **使用例**:
  ```bash
  python scripts/suno_stem_arranger.py \
    --input data/suno_ai/suno_themesong/song_001/stemswav_001 \
    --output data/arranged_midi \
    --tempo 120 \
    --emotion energetic \
    --bars 16
  ```
- **内部フロー**:
  1. ステム解析（stem_harmony使用）
  2. 各楽器Generator呼び出し（Drums/Bass/Piano/Guitar/Strings）
  3. Stage2統合（Phase 13-32適用）
  4. MIDI出力

#### **scripts/suno_importer.py**
- **目的**: Sunoプロジェクト一括インポート

#### **scripts/stage2_suno_stem_test.py**
- **目的**: Stage2機能の単体テスト

---

## 📁 既存の生成済みデータ例

### **data/suno_ai/suno_themesong/song_001/analysis/**

#### **mix_context.json**
```json
{
  "beat_grid": {"bpm": 120.0, "time_signature": [4,4], "total_bars": 16},
  "activity": {
    "piano": [{"bar": 0, "energy": 0.3}, {"bar": 4, "energy": 0.6}, ...],
    "guitar": [...],
    "bass": [...],
    "drums": [...]
  },
  "emotion_curve": [[0.0, 0.3], [4.0, 0.5], [8.0, 0.8], ...],
  "sections": [
    {"bar": 0, "label": "intro", "start_ql": 0.0},
    {"bar": 4, "label": "verse", "start_ql": 16.0},
    ...
  ]
}
```

#### **sections.json**
```json
[
  {
    "label": "intro",
    "bar": 0,
    "beat": 0,
    "tempo": 120.0,
    "ql_per_bar": 4.0,
    "index": 1,
    "chordmap": {
      "0.0": {"root": "C", "quality": "major", "tones_midi": [60,64,67]},
      "2.0": {"root": "G", "quality": "major", "tones_midi": [55,59,62]},
      ...
    }
  },
  ...
]
```

**⚠️ 注意**: 現行版は**chordmapがsections内に埋め込み型**だが、新方針では**chordmap.jsonとして独立**させる。

#### **{role}_style_presets.yaml** (piano/bass/guitar/drums/strings)
```yaml
styles:
  simple:
    rh_pattern: piano_rh_gentle_pads_whole
    lh_pattern: piano_lh_roots_half
    density_scale: 0.85
    register: {lo: 50, hi: 76}
  moderate: {...}
  complex: {...}
  intense: {...}
```

---

## 🔧 Stage1で必要な新規ファイル（今後作成）

### **ops/generate_stage1_jsons.py** ✅ 作成完了
- **目的**: 統一インターフェース（WAV → 3つのJSON）
- **出力**:
  1. `chordmap.json` - 和声進行（独立ファイル）
  2. `sections.json` - セクション区間のみ（chordmap持たせない）
  3. `lyric_anchors.json` - 歌詞タイムライン
- **内部**: `stem_harmony.py` のラッパー

### **lyric_anchors.json** (新規フォーマット)
```json
[
  {"time_ql": 0.0, "line_id": 1, "token": "君", "stress": true, "phoneme": "ki"},
  {"time_ql": 0.5, "line_id": 1, "token": "の", "stress": false, "phoneme": "no"},
  {"time_ql": 1.0, "line_id": 1, "token": "笑", "stress": true, "phoneme": "wa"},
  ...
]
```

---

## 🎼 Stage2関連ファイル（既存）

### **modular_composer.py**
- **目的**: セクション単位でジェネレーター呼び出し・統合
- **入力**:
  - `chordmap.json` (外部) ← 新方針
  - `sections.json`
  - `mix_context.json`
  - `rhythm_library.yml`
- **出力**: 統合されたMIDIスコア

### **data/rhythm_library.yml**
- **目的**: パターン語彙集（piano/bass/guitar/drums）
- **使用例**:
  ```yaml
  piano_rh_block_chords_quarters:
    pattern: [0, 1, 2, 3]
    grid: quarters
    articulation: staccato
  ```

### **utilities/rhythm_library_loader.py**
- **目的**: YAML読み込みユーティリティ

### **utilities/chordmap_merge.py**
- **目的**: ベース進行 × ナラティブ（歌詞/感情）の安全マージ
- **使用例**:
  ```bash
  python -m utilities.chordmap_merge \
    --base base_chordmap.yaml \
    --narr narrative_chords.yaml \
    --out chordmap.final.yaml
  ```

---

## 📊 データフロー全体像

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: WAV → メタデータ生成                          │
└─────────────────────────────────────────────────────┘
Suno WAV stems (vocals/drums/bass/guitar/piano/strings)
  ↓
analysis/stem_harmony.py
  ├─ make_beat_grid()        → beat_grid
  ├─ estimate_activity()     → activity (per role)
  ├─ estimate_chords()       → chord candidates
  └─ aggregate_stem_chords() → chordmap
  ↓
ops/generate_stage1_jsons.py (統合スクリプト)
  ├─ chordmap.json          - 和声進行（独立）
  ├─ sections.json          - セクション区間のみ
  └─ lyric_anchors.json     - 歌詞タイムライン（ボーカルから抽出）

┌─────────────────────────────────────────────────────┐
│ Stage 2: メタデータ → MIDI生成                         │
└─────────────────────────────────────────────────────┘
chordmap.json + sections.json + mix_context.json
  ↓
modular_composer.py
  ├─ rhythm_library.yml 読み込み
  ├─ GenFactory.build_from_config()
  │   ├─ PianoGenerator.compose()    → ノート生成
  │   ├─ GuitarGenerator.compose()   → ノート生成
  │   ├─ BassGenerator.compose()     → ノート生成
  │   └─ DrumsGenerator.compose()    → ノート生成
  ↓
(オプション) Stage2 Params適用
  ├─ Phase 11-12: 密度・音域調整
  ├─ Phase 13-19: 和声・遷移・表情
  ├─ Phase 20-24: Humanize・感情・コントロール
  └─ Phase 25-32: 最終調整・エクスポート
  ↓
MIDI出力
```

---

## 🎯 次のアクション

### ✅ 完了
1. Stage1解析システム存在確認（`stem_harmony.py`）
2. 既存データ形式確認（`mix_context.json`, `sections.json`）
3. Stage2統合スクリプト確認（`suno_stem_arranger.py`）

### 🔄 作業中
4. **ops/generate_stage1_jsons.py** - 統一インターフェース作成 ✅

### ⏳ 未着手
5. **chordmap.json独立化** - sections.jsonからchordmap分離
6. **lyric_anchors.json生成** - ボーカルステムから歌詞抽出
7. **modular_composer.py更新** - chordmap外部読み込み対応
8. **実行テスト** - 実際のSunoステムで動作確認

---

## 📝 重要な設計判断

### 現行版（旧）
- `sections.json` に chordmap を埋め込み
- 各セクションごとに重複したコード情報

### 新方針（推奨）
- **chordmap.json を独立ファイル化**
  - 利点1: 複数セクションでコード共有
  - 利点2: 転調・進行変更が容易
  - 利点3: chordmap_merge.py による安全な編集
- **sections.json は区間情報のみ**
  - bar, label, tempo, ql_per_bar のみ
  - chordmap への参照は modular_composer で結合

### 歌詞情報の扱い
- **lyric_anchors.json** で時間軸管理
- Phase 23 (Prosody Alignment) で活用
- 子音窓・強勢・間の自動調整

---

## 🔗 関連ドキュメント

- `README.md` - プロジェクト全体概要（Suno統合セクション含む）
- `IMPLEMENTATION_REPORT_20251011.md` - Stage2実装レポート
- `DUV_IMPLEMENTATION_STATUS.md` - Phase 13-32実装状況
- `utilities/chordmap_merge.py` - コードマージ仕様
