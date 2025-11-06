# Composer4 - AI音楽アレンジメント自動生成システム

SunoAI楽曲から完全な楽器編成（Bass, Guitar, Piano, Strings, Drums）を自動生成するエンドツーエンドシステム。

## 概要

Composer4は、SunoAIで生成された楽曲（stems WAV）から以下を自動生成します:

1. **音楽分析**
   - テンポマップ抽出（小節単位BPM）
   - セクション構造分析（Intro, Verse, Chorus等）
   - コード進行抽出（chordmap.json）
   - リリックアンカー生成（ボーカルタイミング）

2. **完全編成MIDI生成**
   - Bass（ウォーキングベース、ボイスリーディング対応）
   - Guitar（ストローク、アルペジオ）
   - Piano（コードボイシング、メロディ補強）
   - Strings（パッド、カウンターメロディ）
   - Drums（ハイブリッドv2: WAV×MIDIフュージョン）

3. **品質保証**
   - Magenta Groove統合（ヒューマナイズ）
   - CI/CD自動検証（11項目）
   - KPI Gate（90%以上パス率）

## 主な機能

### 🎵 完全自動E2Eワークフロー

```bash
# song_packageディレクトリ作成（21カラム完全版bars.parquet自動生成）
bash scripts/make_song_package_from_sources.sh \
  data/suno_ai/suno_themesong/song_001 \
  --stems-dir "data/suno_ai/suno_themesong/song_001/stemswav_001"

# E2E統合処理（full_arrangement.mid生成）
./scripts/e2e_suno_arrangement.sh data/suno_ai/suno_themesong/song_001
```

### 🎹 高精度コード解析

- **エンハーモニック正規化**: C#とDb、F#とGbを楽曲キーに応じて統一
- **Bar情報自動付加**: 各コードに小節番号を自動追加
- **Cadence検証**: ドミナント→トニック解決を100%達成

### 🥁 ハイブリッドドラム生成（v2）

- **WAVステム分析**: 元ドラムトラックから密度・エネルギー抽出
- **MIDIパターンマッチング**: 30,000+パターンからTop-K選出
- **Groove Polish**: ハイハットブースト、フィル挿入、フラム追加

### 📊 bars.parquet完全版（21カラム）

すべてのカラムが`make_song_package_from_sources.sh`で自動生成されます:

**必須カラム（10個）**:
- `bar_index`, `tempo_bpm`, `time_signature`
- `start_sec`, `end_sec`, `start_beat`, `end_beat`
- `density_target`, `swing_target`, `section_label`

**stem_features由来（11個）**:
- `drums_active`, `energy_curve`, `hat_density`, `kick_peak_db`
- `snare_backbeat`, `fill_likelihood`, `loudness_db`, `vocal_stress`
- `guitar_activity`, `piano_activity`, `strings_activity`

## セットアップ

### 前提条件

- Python 3.11+
- FFmpeg（audio処理用）
- PortAudio（pyaudio用）

### インストール

```bash
# リポジトリクローン
git clone https://github.com/kinoshitayoshihiro/composer4.git
cd composer4

# Python仮想環境作成
python3.11 -m venv .venv311
source .venv311/bin/activate

# 依存関係インストール
pip install -r requirements.txt
pip install -r requirements-magenta.txt  # Magenta Groove用

# FFmpegインストール（macOS）
brew install ffmpeg portaudio
```

## 使用方法

### 1. SunoAI楽曲準備

```
data/suno_ai/suno_themesong/song_001/
├── stemswav_001/
│   ├── vocals.wav
│   ├── bass.wav
│   ├── drums.wav
│   ├── guitar.wav
│   ├── piano.wav
│   └── other.wav
└── original.wav
```

### 2. Song Package生成

```bash
bash scripts/make_song_package_from_sources.sh \
  data/suno_ai/suno_themesong/song_001 \
  --stems-dir "data/suno_ai/suno_themesong/song_001/stemswav_001"
```

**生成されるファイル**:
- `tempo_map.json`: 小節単位BPM
- `bars.parquet`: 完全版（21カラム）
- `sections.json`: セクション構造
- `chordmap.json`: コード進行
- `lyric_anchors.json`: ボーカルタイミング
- `stem_features.parquet`: ステム特徴量
- `analysis/`: 上記すべてのコピー

### 3. E2E統合処理

```bash
./scripts/e2e_suno_arrangement.sh data/suno_ai/suno_themesong/song_001
```

**処理フロー**:
1. Stem Features生成（240 bars, 12 columns）
2. Pattern Matching（Top-K=5リズムパターン選出）
3. Drums Recommendations（ルールベース推薦）
4. Instrument MIDI生成（Bass, Guitar, Piano, Strings）
5. Drums Plan（ハイブリッドv2）
6. Full Arrangement統合（12,000+イベント）
7. Plan Validation（ドラムch10正規化）
8. MIDI Generation（`full_arrangement.mid`）
9. MIDI Statistics
10. Groove Polish（tomfills挿入、フラム追加）
11. CI Verification（11項目検証）

**成果物**:
- `full_arrangement.mid`: 完成MIDI（6 tracks, PPQ=480）
- `bass_plan.json`, `guitar_plan.json`, `piano_plan.json`, `strings_plan.json`, `drums_plan.json`
- `ci_verify_report.json`: CI検証レポート

### 4. Mix Variants生成（オプション）

```yaml
# mix_variants.yaml
variants:
  - id: soft
    params:
      bass_velocity_scale: 0.85
      drums_velocity_scale: 0.80
      
  - id: standard
    params:
      bass_velocity_scale: 1.0
      drums_velocity_scale: 1.0
      
  - id: bright
    params:
      bass_velocity_scale: 1.1
      drums_velocity_scale: 1.15
```

```bash
python3 scripts/generate_suno_song_package_v1_1.py \
  --base song_001 \
  --variants mix_variants.yaml \
  --output song_package.yaml
```

## プロジェクト構造

```
composer2-3/
├── scripts/
│   ├── make_song_package_from_sources.sh   # 完全版bars.parquet生成（根本治療）
│   ├── e2e_suno_arrangement.sh             # E2E統合処理メインスクリプト
│   ├── recommend_drums.py                  # ドラムパターン推薦（schema v1.0/v1.1対応）
│   ├── adapt_drums_to_plan.py              # ドラムプラン生成（ハイブリッドv2）
│   ├── instrument_midi_to_plan_real.py     # 楽器MIDI生成（Stage2）
│   └── midi_writer.py                      # MIDIファイル書き出し
├── ops/
│   ├── chordmap_to_music21.py              # コード→music21変換（bar情報付加）
│   ├── normalize_enharmonic.py             # エンハーモニック正規化
│   ├── sections_normalize.py               # セクション正規化
│   ├── stems_features.py                   # ステム特徴量抽出（21カラム）
│   ├── ci_verify_music_package.py          # CI検証（11項目）
│   └── magenta_groove.py                   # Magenta Groove統合
├── data/suno_ai/suno_themesong/
│   ├── song_001/
│   ├── song_002/
│   └── song_003/
└── docs/
    ├── PHASE_H_PRODUCTION_COMPLETE.md      # Phase 117完了記録
    ├── HARMONY_QA_CRITERIA.md              # コード品質基準
    ├── MAGENTA_INTEGRATION_PATCHES.md      # Magenta統合パッチ
    └── PHASE_113_SYMBOL_FIRST_PATCH.md     # Symbol優先パッチ
```

## 技術仕様

### bars.parquet完全版（21カラム）生成ロジック

**STEP 1.5: 初期拡張**
```python
# テンポマップから中央値BPM取得
MEDIAN_BPM = statistics.median([p['bpm'] for p in tempo_map['tempo_points']])

# 時刻計算
bars['start_sec'] = bars.index * (60.0 / MEDIAN_BPM * 4)
bars['end_sec'] = (bars.index + 1) * (60.0 / MEDIAN_BPM * 4)

# 拍数計算
bars['start_beat'] = bars.index * 4.0
bars['end_beat'] = (bars.index + 1) * 4.0

# デフォルト値設定
bars['density_target'] = 0.7
bars['swing_target'] = 0.0
```

**STEP 2.5: section_label + セクション別カスタマイズ**
```python
section_defaults = {
    'intro': {'density': 0.5, 'swing': 0.0},
    'verse': {'density': 0.6, 'swing': 0.0},
    'chorus': {'density': 0.9, 'swing': 0.0},
    'bridge': {'density': 0.7, 'swing': 0.1},
    'outro': {'density': 0.4, 'swing': 0.0},
}

for sec in sections:
    mask = (bars.index >= start_bar) & (bars.index <= end_bar)
    bars.loc[mask, 'density_target'] = defaults['density']
    bars.loc[mask, 'swing_target'] = defaults['swing']
    bars.loc[mask, 'section_label'] = label
```

**STEP 5: stem_features.parquet生成 + 全カラムマージ**
```bash
python3 ops/stems_features.py \
  --stems "$STEMS_DIR" \
  --bars "$BARS_FILE" \
  --output "$STEM_FEATURES_FILE" \
  --tempo-bpm "$MEDIAN_BPM" \
  --inst-activity

# 11個のカラムすべてをマージ
merge_columns = [
    'drums_active', 'energy_curve', 'hat_density', 'kick_peak_db',
    'snare_backbeat', 'fill_likelihood', 'loudness_db', 'vocal_stress',
    'guitar_activity', 'piano_activity', 'strings_activity'
]
```

### CI検証項目（11項目）

1. **Tempo meta on Track>0**: set_tempoがTrack 0のみに存在
2. **PPQ consistency**: PPQ=480
3. **Drums channel=9**: ドラムトラックがMIDI ch10（index 9）
4. **Downbeats vs bars**: ダウンビート数がbars数±1
5. **Total duration**: 総演奏時間が期待値±5秒
6. **Track duration (Bass)**: Bassトラック長が期待値±5秒
7. **Track duration (Guitar)**: Guitarトラック長が期待値±5秒
8. **Track duration (Piano)**: Pianoトラック長が期待値±5秒
9. **Track duration (Strings)**: Stringsトラック長が期待値±5秒
10. **Track duration (Drums)**: Drumsトラック長が期待値±5秒
11. **Hard clip over-end**: 終端超過ノートが極小（1個以下許容）

### Magenta Groove統合

```python
from note_seq.protobuf import music_pb2
from magenta.models.drums_rnn import drums_rnn_sequence_generator

# GrooveVAEでヒューマナイズ
temperature = 1.0
humanized_sequence = model.generate(
    input_sequence,
    generator_options=generator_pb2.GeneratorOptions(
        args=['temperature:%f' % temperature]
    )
)
```

## 実装記録

### Phase 117（2025年11月5日〜7日）

**目標**: song_001完全作り直し + E2E統合テスト成功

**達成内容**:
1. ✅ song_001基本5ツール生成（tempo_map, bars, sections, chordmap, lyric_anchors）
2. ✅ Cadence改善（50% → 100%達成）
3. ✅ chordmap.json修正（エンハーモニック、bar情報、symbol追加）
4. ✅ **根本治療完了**（make_song_package_from_sources.sh完全版実装）
5. ✅ E2E統合処理成功（full_arrangement.mid生成）

**技術詳細**:
- `make_song_package_from_sources.sh`に3つのSTEP追加
  - STEP 1.5: bars.parquet初期拡張（start_sec/end_sec/density/swing）
  - STEP 2.5: section_label + セクション別カスタマイズ
  - STEP 5: stem_features.parquet生成 + 全11カラムマージ
- `recommend_drums.py`: schema v1.0/v1.1互換性対応
- `e2e_suno_arrangement.sh`: bash配列修正（引用符問題解決）
- `ci_verify_music_package.py`: インデントエラー修正

**成果物**:
- `full_arrangement.mid`: 85KB、6 tracks、12,126 notes、8分（480.0秒）
- CI検証: 11項目中10項目PASS（1項目警告のみ）

詳細は[PHASE_H_PRODUCTION_COMPLETE.md](PHASE_H_PRODUCTION_COMPLETE.md)を参照。

### Phase 116（2025年11月4日〜5日）

**目標**: song_002にsong_003システム適用 + song_001仕様更新開始

**達成内容**:
1. ✅ song_002にsong_003システム適用
2. ✅ song_001基本5ツール生成開始
3. ✅ chordmap.json修正（エンハーモニック対応）
4. ✅ Cadence改善開始（50% → 100%への道筋確立）

詳細は`PHASE_F_G_H_COMPLETE.md`を参照。

## トラブルシューティング

### bars.parquetにカラムが足りない

**原因**: 古い`make_song_package_from_sources.sh`を使用

**解決策**:
```bash
# 最新版スクリプト使用
git pull origin main
bash scripts/make_song_package_from_sources.sh <song_dir> --stems-dir <stems_dir>
```

### CI Verification失敗

**原因**: パスに日本語が含まれる場合、bash変数展開エラー

**解決策**: 最新版`e2e_suno_arrangement.sh`（配列形式対応）を使用

### Magenta Grooveでエラー

**原因**: note-seq, magentaバージョン不一致

**解決策**:
```bash
pip install -r requirements-magenta.txt
# note-seq==0.0.3, magenta==2.1.4が必要
```

## ライセンス

MIT License

## クレジット

- **Magenta Groove**: Google Magenta Team
- **music21**: Michael Scott Cuthbert
- **SunoAI**: 楽曲生成プラットフォーム

## 参考資料

- [Magenta Groove Documentation](https://github.com/magenta/magenta/tree/main/magenta/models/drums_rnn)
- [music21 Documentation](https://web.mit.edu/music21/)
- [Phase 117完了記録](PHASE_H_PRODUCTION_COMPLETE.md)
- [Harmony QA Criteria](docs/HARMONY_QA_CRITERIA.md)
