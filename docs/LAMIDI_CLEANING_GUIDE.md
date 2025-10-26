# LAMDA (Los-Angeles-MIDI) クリーニングガイド

## 概要
**LAMDA (Los-Angeles-MIDI Dataset)** は約40万ファイル（404,714個）のマルチトラック楽曲データセットです。
このガイドでは、LAMDAを**楽器別に分離してクリーニング**する方法を説明します。

### LAMDAの特徴
- **約40万ファイル**: 大規模な一般MIDI楽曲コレクション
- **マルチトラック**: piano, strings, drums, guitar, bass等が混在
- **Pitch範囲**: 22-86（通常の楽器範囲）
- **コンテンツ**: メロディ、ハーモニー、コード進行を含む完全な楽曲

### クリーニング方針
LAMDAは**単一楽器データセットではない**ため、楽器別に分離して処理します：
- **LAMDA_PIANO**: ピアノパート抽出 + クリーニング
- **LAMDA_STRINGS**: ストリングスパート抽出 + クリーニング
- **LAMDA_GUITAR**: ギターパート抽出 + クリーニング
- **LAMDA_BASS**: ベースパート抽出 + クリーニング
- **LAMDA_DRUMS**: ドラムパート抽出 + クリーニング（オプション）

## 前提条件
- POP909、SLAKH、loopsフォルダは既にクリーニング済み（Stage2まで完了）
- Los-Angeles-MIDIデータは `/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/Los-Angeles-MIDI/MIDIs` に配置済み
- `scripts/cleaners/` に楽器別クリーナー (`piano.py`, `strings.py`, `guitar.py`, `bass.py`, `drums.py`) が配置済み

## クリーニングプロセス

### 1. データセット情報確認
まず、MIDIファイル数を確認します:

```bash
./scripts/check_lamidi_dataset.sh
```

### 2. ドライラン（コマンド確認）
実行せずにコマンドを確認:

```bash
# 全楽器（piano/strings/guitar/bass）
./scripts/run_lamda_full.sh --dry-run

# 特定の楽器のみ確認
./scripts/run_lamda_full.sh --dry-run piano
./scripts/run_lamda_full.sh --dry-run guitar bass
```

### 3. クリーニング実行

#### オプションA: 推奨4楽器を一括実行（drumsを除く）
```bash
./scripts/run_lamda_full.sh
```
デフォルトで `piano`, `strings`, `guitar`, `bass` を処理します。
drumsは母数が巨大なため、明示的に指定した場合のみ実行されます。

#### オプションB: 特定の楽器のみ実行
```bash
# pianoのみ
./scripts/run_lamda_full.sh piano

# guitarとbassのみ
./scripts/run_lamda_full.sh guitar bass

# 全楽器（drumsを含む）
./scripts/run_lamda_full.sh piano strings guitar bass drums
```

### 4. 進捗モニター（別ターミナル）
別のターミナルウィンドウで進捗を監視:

```bash
./scripts/monitor_lamda.sh
```

全楽器の処理状況を一覧表示します：
- 各楽器の処理済みファイル数
- クリーニング成功率
- 隔離ファイル数
- Pickleインデックス状態
- 最新ログ

## Stage 1 の処理内容

### 共通クリーニング (`cleaners/common.py`)
- テンポの正規化
- トラック数チェック
- ノート数チェック
- 空トラックの削除

### Piano専用クリーニング (`cleaners/piano.py`)
- ピアノ楽器チェック
- 音域チェック
- ポリフォニーチェック
- ダイナミクスチェック

### 出力

#### 成功ファイル
- 保存先: `data/cleaned/lamidi/`
- Pickleメタデータ: `data/lamidi_metadata/`

#### 隔離ファイル
- 保存先: `data/quarantine/lamidi/`
- 品質基準を満たさないファイル

### ログ
- 保存先: `logs/clean_LAMIDI_piano_YYYYMMDD_HHMMSS.log`

## Stage 2: メタデータ抽出

Stage 1が完了したら、Stage 2を実行します:

```bash
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamidi_metadata/piano_metadata_v2.pickle \
  --output data/lamidi_stage2_scored.jsonl
```

## トラブルシューティング

### SSD停止・中断からの再開
クリーニング中にSSDが停止した場合、`--resume` オプションで再開できます（既に `run_lamidi_full.sh` に組み込み済み）:

```bash
./scripts/run_lamidi_full.sh
```

既存のシャードから自動的に再開されます。

### ファイル数が想定と異なる場合
`scripts/monitor_lamidi.sh` の `EXPECTED_TOTAL` を実際のファイル数に更新してください:

```bash
# 例: 11823ファイルの場合
EXPECTED_TOTAL=11823
```

### エラーログ確認
最新のログファイルを確認:

```bash
ls -lt logs/clean_LAMIDI_*.log | head -1
tail -50 logs/clean_LAMIDI_*.log
```

## 並列度調整

デフォルトは8並列ですが、CPUコア数に応じて調整できます:

`scripts/run_dataset_full.sh` の LAMIDI行を編集:

```bash
# 4並列に変更する場合
LAMIDI|piano|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamidi|data/quarantine/lamidi|data/lamidi_metadata|5000|4|lamidi-v1
```

## 参考: 他のデータセット

### POP909
```bash
./scripts/run_pop909_full.sh        # 実行
./scripts/monitor_pop909.sh         # 進捗確認
```

### 全データセット一括実行
```bash
./scripts/run_dataset_full.sh
```

## データセット設定

`scripts/run_dataset_full.sh` の `DATASETS` セクション:

```
フォーマット: name|instrument|in_dir|clean_dir|quarantine_dir|pickle_dir|shard_size|jobs|seed
```

- **name**: データセット識別名
- **instrument**: 楽器タイプ (piano/guitar/bass/strings/drums)
- **in_dir**: 入力MIDIディレクトリ
- **clean_dir**: クリーニング済みMIDI保存先
- **quarantine_dir**: 隔離ファイル保存先
- **pickle_dir**: Pickleメタデータ保存先
- **shard_size**: 1シャードあたりのファイル数
- **jobs**: 並列度
- **seed**: 乱数シード（再現性確保）
