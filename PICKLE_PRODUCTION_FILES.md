# Production Pickle Files Inventory

**作成日**: 2025年10月22日  
**目的**: Docker イメージに含める正規版 pickle ファイルのみを特定

---

## ✅ 正規版 (Docker に含めるべきファイル)

### 1. Slakh Dataset (365KB total)

楽器別の学習済みパターン:

```
output/slakh/shards/
├── bass/
│   ├── bass_00000.pkl       (27KB)  ← Bass パターンデータ
│   └── bass_index.pkl       (359B)  ← インデックス
├── drums/
│   ├── drums_00000.pkl      (210KB) ← Drums パターンデータ (最大)
│   └── drums_index.pkl      (366B)  ← インデックス
├── guitar/
│   ├── guitar_00000.pkl     (78KB)  ← Guitar パターンデータ
│   └── guitar_index.pkl     (367B)  ← インデックス
└── strings/
    ├── strings_00000.pkl    (50KB)  ← Strings パターンデータ
    └── strings_index.pkl    (368B)  ← インデックス
```

**合計**: 8 files, ~365KB

---

### 2. POP909 Dataset (1.3MB total)

ポップミュージックの学習済みパターン:

```
output/pop909/shards/
├── bass/
│   ├── bass_00000.pkl       (399KB) ← Bass パターンデータ (最大)
│   └── bass_index.pkl       (364B)  ← インデックス
├── chords/
│   ├── piano_00000.pkl      (309KB) ← Piano コードパターン
│   └── piano_index.pkl      (368B)  ← インデックス
└── melody/
    ├── piano_00000.pkl      (310KB) ← Piano メロディパターン
    └── piano_index.pkl      (368B)  ← インデックス
```

**合計**: 6 files, ~1.3MB

---

### 3. Drums Metadata - LAMDA V2 (30.2MB total) ⭐️ **正規版**

LAMDA (Large-scale MIDI Dataset Analyzer) V2 による Drums メタデータ:

```
output/drums_metadata/
├── drums_index.pkl      (1.2KB)   ← マスターインデックス
├── drums_00000.pkl      (2.7MB)   ← Shard 0: 5,000 loops
├── drums_00001.pkl      (2.8MB)   ← Shard 1: 5,000 loops
├── drums_00002.pkl      (2.7MB)   ← Shard 2: 5,000 loops
├── drums_00003.pkl      (2.8MB)   ← Shard 3: 5,000 loops
├── drums_00004.pkl      (3.0MB)   ← Shard 4: 5,000 loops
├── drums_00005.pkl      (3.0MB)   ← Shard 5: 5,000 loops
├── drums_00006.pkl      (3.0MB)   ← Shard 6: 5,000 loops
├── drums_00007.pkl      (3.3MB)   ← Shard 7: 5,000 loops
├── drums_00008.pkl      (3.0MB)   ← Shard 8: 5,000 loops
├── drums_00009.pkl      (3.1MB)   ← Shard 9: 5,000 loops
└── drums_00010.pkl      (754KB)   ← Shard 10: 1,248 loops
```

**合計**: 12 files, 30.2MB, **51,248 loops**

**作成日**: 2025年10月17日 02:41 (最新!)

**メタデータ詳細** (各ループに19個のフィールド):
- `bpm`, `time_signature`, `duration_ms/ticks`, `note_count`, `avg_velocity`
- `filename`, `md5`, `genre`, `input_path`, `cleaned_file`, `output_path`
- `pitches` (distribution, counts, sum), `patches_counts`, `ms_chords_counts`
- `statistics` (average/median/mode for time, duration, velocity)
- `pitches_times_sum_ms`, `total_number_of_chords`

**備考**: 
- Version: `lamda_v2_index` (index) / `lamda_v2_shard` (shards)
- Shard size: 5,000 loops per file (最終シャードは 1,248 loops)
- Total notes across all loops: 21,967,916 notes
- Average BPM: 109.93 (range: 1-260)
**確認済み**: 
- **V2 が正規版**: `configs/lamda/drums_stage2.yaml` で使用
- **V2 構造**: Version 2.0, 2025-10-07 生成, 1,929 unique loops
- **シャード0**: 実ループデータ (md5, filename, genre, bpm, metrics など)
- **旧版は除外**: `drumloops_metadata.pickle` (9.5MB) は V1 の古いフォーマット

**使用箇所**:
- `configs/lamda/drums_stage2.yaml` → `metadata_index: "output/drumloops_metadata/drumloops_metadata_v2.pickle"`
- `scripts/lamda_stage2_extractor.py` → `DEFAULT_METADATA_INDEX`

---

## 🔴 テスト版・バックアップ (Docker から除外すべき)

### 削除候補 (合計 ~106MB)

| ディレクトリ | ファイル数 | 容量 | 備考 |
|------------|----------|------|------|
| `drumloops_metadata` | 4 files | 20.3MB | ⚠️ **古い開発版** (1,929 loops のみ、10月7日作成) |
| `drums_metadata_backup` | 12 files | 28MB | 明示的なバックアップ |
| `drums_metadata_backup2` | 12 files | 28MB | 2回目のバックアップ |
| `slakh_train_shards` | 2 files | 216KB | トレーニング用シャード |
| `test_slakh_shards` | ? files | ? | テスト用 |
| `single_shards` | 2 files | 8KB | 単一シャードテスト |
| `test_drums_pkl` | 3 files | 64KB | Drums テストデータ |

**合計削除可能容量**: ~106MB

**重要な発見**:
- `drumloops_metadata` は **古い開発版** (1,929 loops、10月7日作成)
- `drums_metadata` が **最新の正規版** (51,248 loops、10月17日作成、26倍のデータ量!)
- バックアップディレクトリは正規版の複製なので削除可能

---

## 📋 Docker `.dockerignore` 推奨設定

```dockerignore
# Pickle files - 正規版のみホワイトリスト
output/**/*.pkl
output/**/*.pickle

# 正規版をホワイトリストで許可
!output/slakh/shards/**/*.pkl
!output/pop909/shards/**/*.pkl
!output/drums_metadata/*.pkl

# 古い drumloops_metadata は除外 (drums_metadata が正規版)
output/drumloops_metadata/**

# テスト版・バックアップも除外
output/drums_metadata_backup/**
output/drums_metadata_backup2/**
output/slakh_train_shards/**
output/test_*/**
output/single_shards/**
```

---

## 🔍 drums_metadata vs drumloops_metadata 比較結果

### ✅ drums_metadata (最新正規版) - 採用

| 項目 | 詳細 |
|------|------|
| **ループ数** | **51,248 loops** ✅ |
| **容量** | 30.2 MB (12 files) |
| **作成日** | **2025-10-17 02:41** (最新!) |
| **Version** | lamda_v2_index / lamda_v2_shard |
| **メタデータ** | 19 fields/loop (詳細!) |
| **シャード** | 11 shards (5000/shard) + 1 partial (1248) |
| **Total notes** | 21,967,916 notes |

### ⚠️ drumloops_metadata (古い開発版) - 除外

| 項目 | 詳細 |
|------|------|
| **ループ数** | 1,929 loops のみ ⚠️ |
| **容量** | 20.3 MB (4 files) |
| **作成日** | 2025-10-07 19:25 (10日古い) |
| **Version** | 2.0 |
| **メタデータ** | 10 fields/loop (限定的) |
| **シャード** | 1 shard のみ |

**結論**: 
- `drums_metadata` は **26倍以上のデータ量** (51,248 vs 1,929 loops)
- より詳細なメタデータ (19 vs 10 fields)
- 10日新しい (10/17 vs 10/07)
- **Docker には drums_metadata のみを含めるべき**

---

## 🔍 V2 検証結果 (参考: drumloops_metadata の内容)

### drumloops_metadata_v2.pickle (362KB)

⚠️ **古い開発版メタデータ** (参考情報)

- **Version**: 2.0
- **Generated**: 2025-10-07T10:25:39
- **Total scanned**: 76,165 files
- **Unique loops**: 1,929 loops (重複除去率: 2.5%)
- **Genre distribution**: 81 ジャンル (funk, rock, jazz, latin など)
- **BPM range**: 1-260 BPM (mean: 37.7)
- **Metrics**: note_count, swing_ratio, ghost_rate, accent_rate など詳細メトリクス
- **シャード情報**: shard_0000.pickle へのポインタ

### drumloops_metadata_v2_shard_0000.pickle (1.4MB)

✅ **実ループデータ**

- **Loops**: 1,929 ループの詳細データ
- **各ループ構造**:
  - `md5`: ハッシュ値
  - `filename`: ファイル名
  - `input_path`: 元ファイルパス
  - `output_path`: クリーン後のパス
  - `genre`: ジャンル
  - `bpm`: テンポ
  - `note_count`: ノート数
  - `duration_ticks`: 長さ
  - `pitches`: 使用ピッチ
  - `metrics`: 詳細メトリクス

### drumloops_filez.pickle (9.2MB)

✅ **ファイルリスト**

- **Total files**: 76,165 ファイルパス
- **用途**: 全 MIDI ファイルのインデックス

---

## 📊 サマリー

| カテゴリ | ファイル数 | 合計容量 | 用途 |
|---------|----------|---------|------|
| **正規版** | **26 files** | **~32MB** | **Docker に含める** |
| Slakh | 8 files | 365KB | 楽器別パターン (4楽器) |
| POP909 | 6 files | 1.3MB | ポップ音楽パターン |
| **Drums Metadata** | **12 files** | **30.2MB** | **LAMDA V2 正規版 (51,248 loops)** |
| **テスト版** | **47+ files** | **~106MB** | **Docker から除外** |

**重要な変更**:
- ✅ `drums_metadata` (30.2MB, 51,248 loops) を正規版として採用
- ❌ `drumloops_metadata` (20.3MB, 1,929 loops) を古い開発版として除外
- ❌ バックアップディレクトリ (`drums_metadata_backup*`) も除外

**容量削減効果**:
- 正規版のみ: 26 files, ~32MB
- 全体: 333 files
- **削減率: 92% のファイルを除外、容量は ~24% に削減**

---

## 🚀 次のアクション

1. ✅ **drums_metadata 検証完了**: 最新の LAMDA V2 正規版 (51,248 loops) を確認
2. ⏳ **`.dockerignore` 更新**: 正規版のみホワイトリスト (26 files, ~32MB)
3. ⏳ **Docker ビルド**: `docker compose build --no-cache`
4. ⏳ **サイズ確認**: ビルド後のイメージサイズが削減されているか確認
5. ⏳ **不要ファイル削除** (任意): バックアップディレクトリを削除してディスク節約 (~106MB)

---

## 参考: 正規版 pickle ファイルの完全リスト

```bash
# Slakh (8 files, 365KB)
output/slakh/shards/bass/bass_00000.pkl
output/slakh/shards/bass/bass_index.pkl
output/slakh/shards/drums/drums_00000.pkl
output/slakh/shards/drums/drums_index.pkl
output/slakh/shards/guitar/guitar_00000.pkl
output/slakh/shards/guitar/guitar_index.pkl
output/slakh/shards/strings/strings_00000.pkl
output/slakh/shards/strings/strings_index.pkl

# POP909 (6 files, 1.3MB)
output/pop909/shards/bass/bass_00000.pkl
output/pop909/shards/bass/bass_index.pkl
output/pop909/shards/chords/piano_00000.pkl
output/pop909/shards/chords/piano_index.pkl
output/pop909/shards/melody/piano_00000.pkl
output/pop909/shards/melody/piano_index.pkl

# Drums Metadata - LAMDA V2 (12 files, 30.2MB, 51,248 loops)
output/drums_metadata/drums_index.pkl
output/drums_metadata/drums_00000.pkl ~ drums_00010.pkl (11 shards)

# 合計: 26 files, ~32MB
```

---

## 参考: 全 pickle ファイル統計

```bash
find output -name "*.pkl" -o -name "*.pickle" | wc -l
# 結果: 333 files
```

**正規版のみ**: 26 files (**92% 削減!**)
