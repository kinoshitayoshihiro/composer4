# Multi-Dataset Runner Guide

複数データセット（POP909/SLAKH/LAMDA/Strings等）を統一的に処理する共通ランナーの使用ガイド

## 概要

従来は各データセット用の個別スクリプト（`run_pop909_full.sh`など）を管理していましたが、新しい共通ランナーでは：

- **1つのスクリプトで全データセットを処理**
- **ブラッシュアップフラグ（streaming/resume/dual-threshold）を全面採用**
- **データセット追加が容易**（表に1行追加するだけ）

## アーキテクチャ

### Stage1: クリーニング & Sharded Pickle 生成

**スクリプト:** `scripts/run_stage1_clean_multi.sh`

**処理内容:**
- 生MIDIファイルのクリーニング（invalid note/tempo除去等）
- クリーンファイルの出力
- 不適格ファイルの隔離（quarantine）
- Sharded pickle 生成（メタデータ用）

**対象データセット:**
| Dataset | Instrument | Input | Output |
|---------|-----------|-------|--------|
| POP909  | drums     | `data/pop909/raw/drums` | `output/pop909/clean/drums` |
| POP909  | strings   | `data/pop909/raw/strings` | `output/pop909/clean/strings` |
| SLAKH   | drums     | `data/slakh2100_midi/raw/drums` | `output/slakh/clean/drums` |
| LAMDA   | drums     | `data/lamda/raw/drumloops` | `output/lamda/clean/drumloops` |

### Stage2: スコアリング & 選抜

**スクリプト:** `scripts/run_stage2_multi.sh`

**処理内容:**
- LAMDAメトリクスによる品質スコアリング
- Dual-threshold（soft/hard）による選抜
- Streaming出力でメモリ節約
- Batch manifest による冪等実行

**対象データセット:**
| Dataset | Instrument | Input | Output | Config |
|---------|-----------|-------|--------|--------|
| POP909  | drums     | `output/pop909/clean/drums` | `output/pop909/stage2/drums` | `configs/lamda/drums_stage2.yaml` |
| SLAKH   | drums     | `output/slakh/clean/drums` | `output/slakh/stage2/drums` | `configs/lamda/drums_stage2.yaml` |
| LAMDA   | drums     | `output/lamda/clean/drumloops` | `output/lamda/stage2/drumloops` | `configs/lamda/drums_stage2.yaml` |

## 使用方法

### 基本実行

```bash
# Stage1: 全データセット一括クリーニング
bash scripts/run_stage1_clean_multi.sh

# Stage2: 全データセット一括スコアリング
bash scripts/run_stage2_multi.sh
```

### カスタマイズ実行

#### Stage1 環境変数

```bash
# シャードサイズを10000に、並列度を4に設定
SHARD_SIZE=10000 JOBS=4 bash scripts/run_stage1_clean_multi.sh

# 冪等実行を無効化（毎回全処理）
RESUME_FLAG="" bash scripts/run_stage1_clean_multi.sh

# JSON出力を有効化（pickle + JSON両方）
EMIT_META_JSON=on bash scripts/run_stage1_clean_multi.sh
```

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `SHARD_SIZE` | 5000 | Pickle シャードサイズ |
| `JOBS` | 8 | 並列処理数 |
| `EMIT_META_JSON` | off | JSON出力の有無 |
| `RESUME_FLAG` | --resume | 冪等実行フラグ |

#### Stage2 環境変数

```bash
# 学習用に緩めの閾値、バッチサイズ大きめ
LIMIT=10000 TH_SOFT=60 TH_HARD=75 bash scripts/run_stage2_multi.sh

# メモリ制約が厳しい環境（バッチサイズ小）
LIMIT=2000 ROW_GROUP=4096 bash scripts/run_stage2_multi.sh

# Streaming/Resume無効化（テスト用）
STREAMING_FLAG="" RESUME_FLAG="" bash scripts/run_stage2_multi.sh
```

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `LIMIT` | 5000 | 1バッチあたりのループ数 |
| `TH_SOFT` | 65.0 | Soft threshold（学習候補） |
| `TH_HARD` | 70.0 | Hard threshold（公開用） |
| `ROW_GROUP` | 8192 | Parquet行グループサイズ |
| `MANIFEST_FLUSH` | 200 | Manifestフラッシュ間隔 |
| `STREAMING_FLAG` | --streaming | Streaming出力フラグ |
| `RESUME_FLAG` | --resume | 冪等実行フラグ |

## データセット追加方法

### Stage1 にデータセット追加

`scripts/run_stage1_clean_multi.sh` の `DATASETS` 変数に行を追加：

```bash
DATASETS="$(cat <<'EOF'
POP909   drums       data/pop909/raw/drums           output/pop909/clean/drums        ...
POP909   strings     data/pop909/raw/strings         output/pop909/clean/strings      ...
SLAKH    drums       data/slakh2100_midi/raw/drums   output/slakh/clean/drums         ...
LAMDA    drums       data/lamda/raw/drumloops        output/lamda/clean/drumloops     ...
# ↓ 新しいデータセットを追加
MYDATA   piano       data/mydata/raw/piano           output/mydata/clean/piano        output/mydata/quarantine/piano  output/mydata/shards/piano
EOF
)"
```

**カラム:**
1. Dataset名
2. Instrument種別（drums/strings/piano等）
3. 入力ディレクトリ（生MIDI）
4. 出力ディレクトリ（クリーンMIDI）
5. Quarantine ディレクトリ（不適格MIDI）
6. Pickle 出力ディレクトリ

### Stage2 にデータセット追加

`scripts/run_stage2_multi.sh` の `DATASETS` 変数に行を追加：

```bash
DATASETS="$(cat <<'EOF'
POP909   drums       output/pop909/clean/drums      output/pop909/stage2/drums        ...
SLAKH    drums       output/slakh/clean/drums       output/slakh/stage2/drums         ...
LAMDA    drums       output/lamda/clean/drumloops   output/lamda/stage2/drumloops     ...
# ↓ 新しいデータセットを追加（Strings用設定ファイルも用意）
POP909   strings     output/pop909/clean/strings    output/pop909/stage2/strings      output/strings_metadata  output/strings_metadata/index.pkl  configs/lamda/strings_stage2.yaml
EOF
)"
```

**カラム:**
1. Dataset名
2. Instrument種別
3. 入力ディレクトリ（クリーンMIDI）
4. 出力ディレクトリ（Stage2結果）
5. メタデータディレクトリ
6. メタデータインデックスpickle
7. Stage2設定YAML

## 出力ファイル構造

### Stage1 出力

```
output/
├── pop909/
│   ├── clean/
│   │   ├── drums/          # クリーンMIDI
│   │   └── strings/
│   ├── quarantine/
│   │   ├── drums/          # 不適格MIDI
│   │   └── strings/
│   └── shards/
│       ├── drums/          # Pickleシャード
│       │   ├── shard_0000.pkl
│       │   ├── shard_0001.pkl
│       │   └── ...
│       └── strings/
├── slakh/
│   └── ...
└── lamda/
    └── ...
```

### Stage2 出力

```
output/
├── pop909/
│   └── stage2/
│       └── drums/
│           ├── batch_0/
│           │   ├── BATCH_MANIFEST.json        # 冪等性メタ
│           │   ├── metrics_score.jsonl        # スコア結果
│           │   ├── loop_summary.csv           # ループサマリー
│           │   ├── canonical_events.parquet   # イベントデータ
│           │   ├── stage2_summary.json        # 統計情報
│           │   └── velocity_coverage.json     # Velocity分布
│           ├── batch_5000/
│           │   └── ...
│           ├── metrics_score.ALL.jsonl        # 全バッチ結合
│           └── loop_summary.ALL.csv           # 全バッチ結合
├── slakh/
│   └── ...
└── lamda/
    └── ...
```

## 利点

### 1. 一元管理

- 全データセットを1つのスクリプトで制御
- フラグ・閾値・バッチサイズを統一的に管理
- 個別スクリプトの重複・齟齬がない

### 2. 再実行に強い

- `--resume` により中断から安全に再開
- `BATCH_MANIFEST.json` で処理済みファイルをトラッキング
- 手動でのバッチ番号管理が不要

### 3. 省メモリ

- `--streaming` で逐次書き出し（メモリ蓄積なし）
- `--limit/--offset` による小分け処理
- 大規模データ（LAMDA 51,248ループ等）でも安定動作

### 4. 拡張容易

- データセット追加は表に1行追加するだけ
- Instrument種別（drums/strings/piano）も簡単に対応
- 特殊処理が必要な場合はフック関数を追加

## 既存スクリプトとの関係

### 置き換え対象

以下のデータセット別スクリプトは共通ランナーで代替可能：

- `run_pop909_full.sh` → `run_stage1_clean_multi.sh` + `run_stage2_multi.sh`
- `run_slakh_full.sh` → 同上
- `run_lamda_full.sh` → 同上

### 残すべきスクリプト

データセット固有の特殊処理がある場合は残してOK：

- 独自の前処理（例：POP909 の chord annotation 抽出）
- 独自の検証（例：SLAKH の multi-track 整合性チェック）
- 独自のポストプロセス（例：LAMDA の velocity histogram 分析）

**推奨:** 共通ランナーから "特殊フック" として呼ぶ形に整理すると運用がきれい

## トラブルシューティング

### Stage1 でエラーが出る

**症状:** `scripts/clean_midi.py` が見つからない

**解決:**
```bash
# PYTHONPATH を明示
PYTHONPATH=. bash scripts/run_stage1_clean_multi.sh
```

**症状:** Pickle 書き込みでエラー

**解決:**
```bash
# 出力ディレクトリの権限確認
ls -ld output/*/shards/*

# 権限がない場合は修正
chmod -R u+w output/
```

### Stage2 でメタデータが見つからない

**症状:** `Metadata index not found: output/xxx_metadata/xxx_index.pkl`

**解決:**
```bash
# Stage1 を先に実行
bash scripts/run_stage1_clean_multi.sh

# または該当データセットのみ実行（スクリプトを一時的に編集）
```

### OOM（メモリ不足）が発生

**解決:**
```bash
# バッチサイズを減らす
LIMIT=2000 bash scripts/run_stage2_multi.sh

# Parquet行グループサイズも減らす
LIMIT=2000 ROW_GROUP=2048 bash scripts/run_stage2_multi.sh

# Streamingが無効になっていないか確認
echo $STREAMING_FLAG  # --streaming が表示されるべき
```

### 再実行で重複処理される

**症状:** `--resume` を指定しても再処理される

**原因:** `BATCH_MANIFEST.json` のSHA1不一致（メタデータインデックスが更新された）

**解決:**
```bash
# Stage1 からやり直す（メタデータ再生成）
bash scripts/run_stage1_clean_multi.sh
bash scripts/run_stage2_multi.sh

# または手動でマニフェストを削除
rm -rf output/*/stage2/*/batch_*/BATCH_MANIFEST.json
```

## 性能目安

### Stage1（8並列）

| Dataset | Files | Time | Throughput |
|---------|-------|------|-----------|
| POP909 drums | ~800 | ~2分 | 400 files/min |
| SLAKH drums | ~8,000 | ~15分 | 530 files/min |
| LAMDA drums | ~51,000 | ~90分 | 560 files/min |

### Stage2（LIMIT=5000, streaming有効）

| Dataset | Loops | Time | Throughput |
|---------|-------|------|-----------|
| POP909 drums | ~800 | ~3分 | 250 loops/min |
| SLAKH drums | ~8,000 | ~30分 | 260 loops/min |
| LAMDA drums | ~51,000 | ~180分 | 280 loops/min |

**メモリピーク:**
- Streaming有効: ~500MB（一定）
- Streaming無効: ~8GB（ループ数に比例）

## 今後の拡張

### 優先度1: Parquet 統合マージ

現在は JSONL/CSV のみ結合。Parquet も `pyarrow` で統合する：

```python
# scripts/merge_stage2_parquets.py（TODO実装）
import pyarrow.parquet as pq

def merge_parquets(input_pattern, output_path):
    tables = [pq.read_table(f) for f in glob(input_pattern)]
    merged = pa.concat_tables(tables)
    pq.write_table(merged, output_path)
```

### 優先度2: 自動シャーディング

システムメモリを読み取って最適な `LIMIT` を自動計算：

```bash
# 利用可能メモリに応じて自動調整
LIMIT=auto bash scripts/run_stage2_multi.sh
```

### 優先度3: データセット別フック

特殊処理を共通ランナーから呼び出す：

```bash
# run_stage2_multi.sh に追加
if [[ "${NAME}" == "POP909" ]]; then
  bash scripts/hooks/pop909_post_process.sh "${OUT_DIR}"
fi
```

## 関連ドキュメント

- **Stage2ブラッシュアップ詳細:** `STAGE2_BRUSHUP_GUIDE.md`
- **LAMDA個別実行:** `run_stage2_batches.sh`（LAMDA専用スクリプト、今後は非推奨）
- **完了レポート:** `STAGE2_COMPLETION_REPORT.md`

## まとめ

共通ランナーにより：

- ✅ **1つのコマンドで全データセット処理**
- ✅ **ブラッシュアップフラグを全面採用**（streaming/resume/dual-threshold）
- ✅ **データセット追加が容易**（表に1行追加）
- ✅ **メモリ効率的**（省メモリ＆OOM回避）
- ✅ **再実行に強い**（冪等性保証）

従来の個別スクリプト管理から脱却し、運用コストを大幅に削減できます。
