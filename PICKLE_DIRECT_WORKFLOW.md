# Pickle直書き運用ガイド

## 概要

`clean_midi.py` が `.meta.json` を経由せず、直接 sharded pickle に書き出す運用に完全対応しました。

## 主要な変更点

### 1. スキップ判定の改善

**変更前:**
- `--emit-meta-json` の設定に関わらず、常に `.meta.json` の存在でスキップ判定
- 過去の `.meta.json` が残っていると、pickle に登録されない問題

**変更後:**
- `--emit-meta-json off` の時は、クリーニング済み `.mid` の存在でスキップ判定
- `.meta.json` が残っていても pickle 運用が止まらない

### 2. スキップ時もshardに登録

**新機能:**
- スキップされたファイルも、既存の `.mid` を再パースして LAMDA エントリを生成
- shard に自動追加されるため、レジューム時も完全な pickle が作られる

### 3. レジューム対応の強化

- SSD接続トラブルなどで中断しても、`--resume` で途中から再開可能
- 既存ファイルは shard に追加され、未処理ファイルのみクリーニング

## 推奨使用方法

### 基本コマンド（Piano例）

```bash
python -m scripts.clean_midi \
  --in data/lamda/raw/piano \
  --out data/lamda/clean/piano \
  --quarantine data/lamda/quarantine/piano \
  --instrument piano \
  --pickle-out data/lamda/shards/piano \
  --shard-size 5000 \
  --resume \
  --emit-meta-json off \
  --jobs 8
```

### パラメータ説明

- `--emit-meta-json off` : **推奨** - .meta.json を書かない（pickle直書き）
- `--shard-size 5000` : **推奨** - 5,000件/シャード（バランス最適）
  - メモリとI/O効率のバランスが良い
  - SSDトラブル時の再開距離も適切
  - 必要に応じて 1,000 〜 10,000 で調整可能
- `--resume` : 既存シャードから続行（SSD事故対策）
- `--jobs 8` : 並列処理数（CPUコア数に応じて調整）

### 中断からの復旧

```bash
# まったく同じコマンドを再実行（--resume が重要）
python -m scripts.clean_midi \
  --in data/lamda/raw/piano \
  --out data/lamda/clean/piano \
  --quarantine data/lamda/quarantine/piano \
  --instrument piano \
  --pickle-out data/lamda/shards/piano \
  --shard-size 5000 \
  --resume \
  --emit-meta-json off \
  --jobs 8
```

**動作:**
1. 既存の shard を走査して、次のインデックスを決定
2. 既存のクリーニング済み `.mid` はスキップしつつ、shard に登録
3. 未処理ファイルのみクリーニング実行

## 出力ファイル構造

```
data/lamda/shards/piano/
├── piano_00000.pkl   # 0-4,999件
├── piano_00001.pkl   # 5,000-9,999件
├── piano_00002.pkl   # 10,000-14,999件
└── piano_index.pkl   # 全シャードのインデックス

data/lamda/clean/piano/
├── file1.mid         # クリーニング済みMIDI
├── file2.mid
└── ...

data/lamda/quarantine/piano/
├── bad1.mid          # 隔離されたMIDI
├── bad1.meta.json    # エラー情報（--emit-meta-json off でも記録）
└── ...
```

**注意:**
- `.meta.json` は成功ファイルには出力されない（`--emit-meta-json off`）
- 隔離ファイルのみ `.meta.json` が残る（デバッグ用）

## 検証方法

### 1. スモークテスト（10〜50ファイル）

```bash
# 小規模ディレクトリで試す
python -m scripts.clean_midi \
  --in data/lamda/raw/piano_test \
  --out data/lamda/clean/piano_test \
  --quarantine data/lamda/quarantine/piano_test \
  --instrument piano \
  --pickle-out data/lamda/shards/piano_test \
  --shard-size 5000 \
  --resume \
  --emit-meta-json off \
  --jobs 4
```

### 2. 確認ポイント

```bash
# 1. pickle が作成されている
ls -la data/lamda/shards/piano_test/

# 2. .meta.json が出ていない（成功ファイル）
find data/lamda/clean/piano_test -name "*.meta.json" | wc -l
# → 0 であること

# 3. pickle の件数を確認
python scripts/status_metadata.sh data/lamda/shards/piano_test
# → クリーニング済みファイル数と一致すること

# 4. 再実行してスキップ動作確認
python -m scripts.clean_midi ... --resume
# → "Skipped: XX" が表示され、pickle 件数が維持されること
```

## トラブルシューティング

### 問題: 既存 .meta.json が残っていて shard に追加されない

**原因:** 古いバージョンで実行した結果

**解決策:**
```bash
# 1. 古い .meta.json を削除（オプション）
find data/lamda/clean/piano -name "*.meta.json" -delete

# 2. --resume で再実行
python -m scripts.clean_midi ... --resume --emit-meta-json off
```

新バージョンでは `.meta.json` の有無に関わらず、`.mid` の存在で判定するため、
削除しなくても動作しますが、混乱を避けるため削除推奨。

### 問題: SSD接続が切れて途中で停止した

**解決策:**
```bash
# まったく同じコマンドで再実行（--resume が重要）
python -m scripts.clean_midi ... --resume
```

- 既存シャードの続きから処理が再開されます
- 処理済みファイルは自動スキップされます

### 問題: シャードサイズを変更したい

**小さくする場合（1,000件/シャード）:**
```bash
python -m scripts.clean_midi ... --shard-size 1000
```

**大きくする場合（10,000件/シャード）:**
```bash
python -m scripts.clean_midi ... --shard-size 10000
```

**既存シャードの結合（事後変更）:**
```bash
# TODO: shard結合ツールを実装予定
# python scripts/merge_shards.py --in shards/piano --shard-size 10000
```

## レガシースクリプトとの比較

### 非推奨（旧方式）

```bash
# ❌ 非推奨: JSON経由の2ステップ方式
python scripts/build_drumloops_metadata.py
python scripts/build_index_from_json.py
python scripts/append_to_index.py
```

**問題点:**
- .meta.json が大量に生成される
- 2ステップ必要で手間がかかる
- レジューム対応が不完全

### 推奨（新方式）

```bash
# ✅ 推奨: pickle直書き1ステップ方式
python -m scripts.clean_midi \
  --pickle-out data/lamda/shards/piano \
  --emit-meta-json off \
  --resume
```

**利点:**
- .meta.json を生成しない（ディスク節約）
- 1ステップで完結
- レジューム対応完全
- SSD事故に強い

## FAQ

### Q: --emit-meta-json の使い分けは？

**A:**
- `off` : **推奨** - pickle直書き運用。ディスク節約。
- `auto` : 隔離/警告ファイルのみ .meta.json 出力（デバッグ用）
- `on` : 全ファイルに .meta.json 出力（レガシー互換、非推奨）

### Q: シャードサイズの推奨値は？

**A:**
- **5,000** : デフォルト推奨（バランス最適）
- 1,000 : 小規模・頻繁な保存が必要な場合
- 10,000 : 大規模・高速処理優先の場合

### Q: 既存の .meta.json を削除すべき？

**A:**
新バージョンでは削除不要（`.mid` で判定）。
ただし、ディスク容量節約・混乱回避のため削除推奨:

```bash
find data/lamda/clean -name "*.meta.json" -delete
```

### Q: レガシースクリプトは削除すべき？

**A:**
互換性のため残しても良いが、README に「レガシー・非推奨」と明記推奨:

```markdown
## レガシースクリプト（非推奨）

以下は後方互換性のためのみ残されています。新規利用は推奨しません:
- `build_drumloops_metadata.py`
- `build_index_from_json.py`
- `append_to_index.py`

→ 代わりに `clean_midi.py --pickle-out ... --emit-meta-json off` を使用してください。
```

## まとめ

1. **推奨コマンド:** `--emit-meta-json off --shard-size 5000 --resume`
2. **SSD事故対策:** `--resume` で同じコマンド再実行
3. **検証:** pickle件数 = クリーニング済みファイル数
4. **レガシー:** 旧スクリプトは非推奨化

これで完全な pickle 直書き運用が可能になりました！
