# Song Package 自動生成ツール（統合レイアウト準拠版）

## 📦 概要

**「入口は二刀流、出口は一本」**を実現するSong Package自動生成ツール一式です。

- **入口（二刀流）**: WAV（MoisesDB/MUSDB18等のstem）+ MIDI（Stage1クリーニング済み）
- **出口（一本化）**: `song_package.yaml`で論理統合（bars.parquet＋楽曲仕様三点で束ねる）

### 方針

「**正本＝JSON/YAML/Parquet、DB＝索引、pickleは使わない。キャッシュは任意・短命・再計算可能**」

- **ハブ**: `{song_id}.bars.parquet` — すべての指標をbarキーで左結合できる唯一の土台
- **楽曲仕様三点**（Stage3の"真"）:
  - `sections.json`: Verse/Pre/Chorus…（QL境界・拍子・テンポヒント）
  - `chordmap.json`: 小節単位のコード（music21準拠）＋転調情報
  - `lyric_anchors.json`: 読み/歌詞のタイムアンカー

---

## 🗂️ フォルダ構成（統合レイアウト）

```
LOCAL_LAMDA/
├── Local_Lamda_midi/
│   ├── CLEANED_MIDI/                  # 入口（元データ）
│   ├── midi_guide/                    # 成果物（Stage1＋パート別MIDI）
│   │   └── {song_id}/
│   │       ├── stage1_clean.mid       # OK::meta注入済み
│   │       ├── stage1_clean.json      # ID/拍子/テンポ/統計
│   │       ├── piano.mid / guitar.mid / bass.mid / drums.mid / vocal.mid
│   │       └── song_package.yaml      # ★ 出口の一本化ファイル
│   └── stats/                         # LAMDA先験データ
│       ├── LAMDA_TOTALS.parquet
│       └── LAMDA_SIGNATURES.json
│
├── Local_Lamda_wav/
│   ├── CLEANED_WAV/                   # 入口（元データ）
│   │   ├── moisesdb_original/
│   │   └── musdb18_decoded/
│   └── wav_guide/                     # 成果物（Stage2）
│       ├── moisesdb/
│       │   ├── {song_id}/
│       │   │   ├── beat_grid.json
│       │   │   ├── accent_grid.json
│       │   │   ├── audio_chordmap.yaml
│       │   │   └── {song_id}.bars.parquet  # ★ ハブ
│       │   ├── vocal_features.parquet
│       │   └── mix_diagnostics.parquet
│       └── musdb18/...
│
├── Local_Lamda_specs/                 # 楽曲仕様三点（Stage3）
│   └── {song_id}/
│       ├── sections.json
│       ├── chordmap.json
│       └── lyric_anchors.json
│
├── renders/                           # レンダー成果物（Stage4）
│   └── {dataset}/{song_id}/
│       ├── piano.wav / guitar.wav / bass.wav / drums.wav / vocal.wav
│       ├── render_config.yaml
│       └── render_report.json
│
├── qa/                                # QA成果物
│   └── {dataset}/
│       ├── {song_id}_qa.json
│       └── {song_id}_qa.csv
│
└── local_lamda_registry.db            # DB索引（パス/IDのみ）
```

---

## 🛠️ ツール一式

### 1. **generate_song_package_v2.py** — Song Package自動生成

各曲の`song_package.yaml`を生成（MIDI側のフォルダに出力、相対パスで束ねる）。

#### 依存

```bash
pip install pyyaml
```

#### 使い方

```bash
python scripts/generate_song_package_v2.py \
  --base "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/Los-Angeles-MIDI/LOCAL_LAMDA" \
  --dataset moisesdb --dataset musdb18 \
  --include-dataset-level --add-audio-chordmap \
  --code-version "local_lamda_moises_integration.py@<git-hash>" \
  --index-out "/tmp/song_packages_index.csv"
```

#### オプション

- `--dataset`: 複数指定可（`--dataset moisesdb --dataset musdb18` or `--dataset moisesdb,musdb18`）
- `--include-dataset-level`: dataset-level diagnostics（vocal_features/mix_diagnostics）を含める
- `--add-audio-chordmap`: per-song `audio_chordmap.yaml`を含める
- `--index-out`: 生成したパッケージのCSVインデックス出力
- `--dry-run`: 書き込みなしの確認実行

#### 出力例: `song_package.yaml`

```yaml
version: 1.0
ids:
  song_id: "9653a690-c28c-4e8f-962e-ff7ed18b8ee9"
  run_id: "local-2025-10-25T12:34:56"
  code_version: "local_lamda_moises_integration.py@abc123"
  midi_content_id: "9f0e1d2c3b4a5f6e"
  wav_file_id: "2ead80e890c4"
  dataset: "moisesdb"

spec:  # 楽曲仕様三点（任意）
  sections: "../../../../Local_Lamda_specs/9653a690.../sections.json"
  chordmap: "../../../../Local_Lamda_specs/9653a690.../chordmap.json"
  anchors:  "../../../../Local_Lamda_specs/9653a690.../lyric_anchors.json"

hub:  # bars.parquet（必須のハブ）
  bars_parquet: "../../../../Local_Lamda_wav/wav_guide/moisesdb/9653a690.../9653a690....bars.parquet"

guides:
  midi:
    stage1_clean: "stage1_clean.mid"
    piano:  "piano.mid"
    guitar: "guitar.mid"
    bass:   "bass.mid"
    drums:  "drums.mid"
    vocal:  "vocal.mid"

diagnostics:
  wav_beat_grid:   "../../../../Local_Lamda_wav/wav_guide/moisesdb/9653a690.../beat_grid.json"
  wav_accent_grid: "../../../../Local_Lamda_wav/wav_guide/moisesdb/9653a690.../accent_grid.json"
  wav_audio_chordmap: "../../../../Local_Lamda_wav/wav_guide/moisesdb/9653a690.../audio_chordmap.yaml"
  dataset_level:
    vocal_features:  "../../../../Local_Lamda_wav/wav_guide/moisesdb/vocal_features.parquet"
    mix_diagnostics: "../../../../Local_Lamda_wav/wav_guide/moisesdb/mix_diagnostics.parquet"
```

---

### 2. **render_from_package.py** — クイック試聴stems生成

`song_package.yaml`からMIDIガイドをFluidsynth+SF2で簡易stems化。

#### 依存

```bash
pip install pyyaml mido
# 任意: Fluidsynth CLI + SF2サウンドフォント（例: GeneralUser.sf2）
```

#### 使い方

```bash
python scripts/render_from_package.py \
  --package "/.../midi_guide/SONG123/song_package.yaml" \
  --soundfont "/path/to/GeneralUser.sf2" \
  --outdir "/.../renders/SONG123" \
  --preset-map '{"piano":0, "guitar":24, "bass":32, "drums":128, "vocal":0}'
```

- `drums=128`: チャンネル10（ドラム）固定の簡易指定
- Fluidsynth/SF2が無い場合は`render_config.yaml`のみ出力

#### 出力

- `piano.wav`, `guitar.wav`, `bass.wav`, `drums.wav`, `vocal.wav`
- `render_config.yaml`
- `render_report.json`

---

### 3. **qa_from_package.py** — 軽量QA

`song_package.yaml`の整合性チェック（bars.parquet/spec三点/diagnostics/MIDIパートの有無確認）。

#### 依存

```bash
pip install pyyaml mido pandas pyarrow
```

#### 使い方

```bash
python scripts/qa_from_package.py \
  --package "/.../midi_guide/SONG123/song_package.yaml" \
  --out "/.../qa/SONG123_qa.json" \
  --csv "/.../qa/SONG123_qa.csv"
```

#### 出力

- `qa_report.json`: 詳細レポート
- `qa.csv`: 簡易サマリー（--csv指定時）

---

### 4. **batch_from_packages.py** — 一括レンダー＆QA

複数曲の`song_package.yaml`を並列処理（render/qa）。

#### 依存

```bash
pip install pyyaml mido pandas pyarrow
# 任意: Fluidsynth + SF2（レンダー実行時）
```

#### 使い方

```bash
python scripts/batch_from_packages.py \
  --base "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/Los-Angeles-MIDI/LOCAL_LAMDA" \
  --tasks render,qa \
  --dataset moisesdb,musdb18 \
  --soundfont "/path/to/GeneralUser.sf2" \
  --render-out "/.../LOCAL_LAMDA/renders" \
  --qa-out "/.../LOCAL_LAMDA/qa" \
  --workers 4 \
  --index-out "/.../LOCAL_LAMDA/batch_index.csv"
```

#### オプション

- `--tasks`: `render,qa` or `render` or `qa`
- `--dataset`: フィルタ（複数指定可）
- `--workers`: 並列数（デフォルト2）
- `--force`: 既存成果物を無視して再実行
- `--index-out`: バッチ結果のCSVインデックス

#### 出力

- `batch_index.csv`: 成功/失敗/出力パス一覧

---

## 📋 運用フロー

### 1. **Stage1–2実行**（WAV/MIDI処理）

```bash
# MIDI Stage1（クリーニング＋ID付与）
python scripts/stage1_lamda_plus_v2.py --config config/stage1_config.yaml

# WAV Stage2（beat/chord/bars抽出）
python scripts/local_lamda_moises_integration.py \
  --input-dir data/Los-Angeles-MIDI/LOCAL_LAMDA/Local_Lamda_wav/CLEANED_WAV/moisesdb_original \
  --output-db data/moisesdb_wav_unified.db \
  --source-name moisesdb \
  --policy-yaml config/stem_policy.yaml \
  --verbose
```

### 2. **Song Package生成**（出口一本化）

```bash
python scripts/generate_song_package_v2.py \
  --base "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/Los-Angeles-MIDI/LOCAL_LAMDA" \
  --dataset moisesdb --dataset musdb18 \
  --include-dataset-level --add-audio-chordmap \
  --code-version "local_lamda_moises_integration.py@$(git rev-parse --short HEAD)" \
  --index-out "data/Los-Angeles-MIDI/LOCAL_LAMDA/song_packages_index.csv"
```

### 3. **レンダー＆QA実行**（一括）

```bash
python scripts/batch_from_packages.py \
  --base "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/Los-Angeles-MIDI/LOCAL_LAMDA" \
  --tasks render,qa \
  --dataset moisesdb,musdb18 \
  --soundfont "/path/to/GeneralUser.sf2" \
  --render-out "data/Los-Angeles-MIDI/LOCAL_LAMDA/renders" \
  --qa-out "data/Los-Angeles-MIDI/LOCAL_LAMDA/qa" \
  --workers 4 \
  --index-out "data/Los-Angeles-MIDI/LOCAL_LAMDA/batch_index.csv"
```

### 4. **結果確認**

```bash
# CSVインデックス確認
cat data/Los-Angeles-MIDI/LOCAL_LAMDA/batch_index.csv

# 個別QAレポート確認
cat data/Los-Angeles-MIDI/LOCAL_LAMDA/qa/moisesdb/SONG123_qa.json
```

---

## 🎯 重要ポイント

1. **bars.parquet は必須**  
   ハブがない曲はスキップされます。Stage2を先に実行してください。

2. **相対パス設計**  
   `song_package.yaml`内はすべて相対パスなので、`LOCAL_LAMDA`ツリーを移動してもリンク切れしません。

3. **ID体系**  
   - WAV系: `file_id = sha256(canonical_manifest)[:12]`
   - MIDI系: `content_id = md5(bar_fingerprint+duration_ticks)[:16]`, `source_mid_id = md5(input_bytes)[:16]`

4. **並列処理**  
   `batch_from_packages.py`の`--workers`で高速化可能（Fluidsynth/QAは並列安全）。

5. **Dry-run**  
   `--dry-run`で書き込みなしの確認実行ができます。

---

## 📚 参考

- **マニフェスト**: `data/Los-Angeles-MIDI/LOCAL_LAMDA/local_lamda_master_manifest_v2.yaml`
- **Stage1設定**: `config/stage1_config.yaml`
- **キャッシュ**: `utils/cache_util.py`
- **DB索引**: `utils/db_registry.py`

---

## 🆘 トラブルシューティング

### bars.parquet が見つからない

Stage2（WAV処理）を先に実行してください：

```bash
python scripts/local_lamda_moises_integration.py ...
```

### Fluidsynthが見つからない

レンダーは任意です。Fluidsynth未導入でも`render_config.yaml`は生成されます：

```bash
python scripts/render_from_package.py --package ... --outdir ... # --soundfont省略
```

### CSVインデックスが空

`--index-out`を指定し、`--dry-run`を外してください。

---

**以上で「入口は二刀流、出口は一本」の運用体制が完成しました！** 🎉
