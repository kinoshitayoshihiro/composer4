# Drumloops処理ワークフロー（更新版）

**更新日:** 2025-10-16  
**対応:** Stage2互換 Pickle直書き方式

---

## 📋 概要

Drumloopsの完全処理パイプラインを、Stage2互換のPickle直書き方式で実行します。

### 主な変更点

1. **Pickle直書き**: clean_midi.py が直接 Stage2互換の sharded pickle を生成
2. **`.meta.json` 廃止**: ディスク容量節約のため JSON 出力をオフ
3. **レジューム対応**: SSD接続トラブル時も安全に途中から再開
4. **genreフィールド追加**: Stage2互換性のため楽器名を自動設定

---

## 🚀 実行手順

### ステップ1: 小規模テスト（推奨）

77,346ファイル全体を処理する前に、小規模データでテストしてください。

```bash
# 100ファイル程度のテスト
python -m scripts.clean_midi \
  --in data/loops \
  --out output/test_drums_100 \
  --quarantine output/test_drums_100_q \
  --instrument drums \
  --pickle-out output/test_drums_pkl \
  --shard-size 50 \
  --emit-meta-json off \
  --jobs 1
```

**確認:**
```bash
# 結果確認
find output/test_drums_100 -name "*.mid" | wc -l
find output/test_drums_pkl -name "*.pkl" | wc -l

# Stage2互換性チェック
python verify_stage2_compat.py output/test_drums_pkl
```

---

### ステップ2: 本番実行

テストが成功したら、本番実行します。

#### 方法1: シェルスクリプト使用（推奨）

```bash
# クリーニング + Pickle生成
./scripts/run_drumloops_full.sh
```

**内容:**
- 入力: data/loops (77,346ファイル)
- 出力: output/drumloops_v3
- 隔離: output/drumloops_v3_q
- Pickle: output/drums_metadata
- Shard Size: 5,000件
- 並列処理: 8 jobs
- ログ: logs/drumloops_cleaning_YYYYMMDD_HHMMSS.log

#### 方法2: 直接コマンド

```bash
# 仮想環境アクティベート
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3
source .venv311/bin/activate

# 実行
python -m scripts.clean_midi \
  --in data/loops \
  --out output/drumloops_v3 \
  --quarantine output/drumloops_v3_q \
  --instrument drums \
  --pickle-out output/drums_metadata \
  --shard-size 5000 \
  --resume \
  --emit-meta-json off \
  --jobs 8
```

---

### ステップ3: 進捗確認

```bash
# リアルタイム監視
watch -n 5 ./scripts/status_drumloops.sh

# または手動確認
./scripts/status_drumloops.sh
```

**出力例:**
```
🥁 Drumloops Cleaning Status
Stage2互換 Pickle直書き方式
===========================

✅ Status: RUNNING
   PID: 12345

📊 Progress: 15432 / 77346 files
   ✅ Cleaned:      14876
   🗑️  Quarantined:  556
   📦 Pickle Shards: 3
   Progress:     20.0%
   Success Rate: 96.4%

📦 Pickle Status:
   ✅ Index: drums_index.pkl exists
   Shards: 3 files
   ✅ Shard count looks good

Commands:
  Watch:    watch -n 5 ./scripts/status_drumloops.sh
  Log:      tail -f logs/drumloops_cleaning_*.log
  Stop:     kill 12345
  Verify:   python verify_stage2_compat.py output/drums_metadata
```

---

### ステップ4: Stage2処理

クリーニング完了後、Stage2でメトリクス計算を実行します。

```bash
# Stage2実行
./scripts/run_drumloops_stage2.sh
```

**内容:**
- 入力: output/drums_metadata/drums_index.pkl
- メタデータDir: output/drums_metadata
- MIDI入力: output/drumloops_v3
- 出力: output/drumloops_v3_stage2
- 設定: configs/lamda/drums_stage2.yaml
- 閾値: 70.0

---

## 📁 出力ファイル構造

### クリーニング後

```
output/
├── drumloops_v3/              # クリーニング済みMIDI
│   ├── file001.mid
│   ├── file002.mid
│   └── ...
├── drumloops_v3_q/            # 隔離MIDI
│   ├── bad001.mid
│   ├── bad001.meta.json       # エラー情報（隔離のみ）
│   └── ...
└── drums_metadata/            # Stage2入力（Pickle）
    ├── drums_00000.pkl        # Shard 0 (0-4,999件)
    ├── drums_00001.pkl        # Shard 1 (5,000-9,999件)
    ├── ...
    └── drums_index.pkl        # インデックス ★Stage2入力
```

### Stage2処理後

```
output/
└── drumloops_v3_stage2/       # Stage2出力
    ├── canonical_events.parquet
    ├── loop_summary.csv
    ├── metrics_score.jsonl
    └── stage2_summary.json
```

---

## 🔧 トラブルシューティング

### SSD接続が切れた場合

```bash
# まったく同じコマンドで再実行（--resume が重要）
python -m scripts.clean_midi \
  --in data/loops \
  --out output/drumloops_v3 \
  --quarantine output/drumloops_v3_q \
  --instrument drums \
  --pickle-out output/drums_metadata \
  --shard-size 5000 \
  --resume \
  --emit-meta-json off \
  --jobs 8
```

**動作:**
- 既存 shard を検出して続きから処理
- 処理済みファイルは自動スキップ
- Pickle に追加されるため、データ欠損なし

### 並列処理でエラーが発生する場合

```bash
# jobs=1 で直列実行
python -m scripts.clean_midi \
  ... \
  --jobs 1
```

### Pickle互換性の確認

```bash
# 検証スクリプト実行
python verify_stage2_compat.py output/drums_metadata

# 期待される出力:
# ✅ Stage2互換性チェック完了 - すべてOK！
```

---

## 📊 期待される処理時間

### テスト（100ファイル）
- **時間:** ~1-2分
- **目的:** 動作確認

### 本番（77,346ファイル）
- **推定時間:** 6-12時間（jobs=8の場合）
- **並列度に依存:**
  - jobs=1: ~24時間
  - jobs=4: ~12時間
  - jobs=8: ~6時間

---

## ✅ チェックリスト

### 実行前
- [ ] 仮想環境アクティベート済み
- [ ] SSDマウント確認
- [ ] ディスク容量確認（100GB以上推奨）
- [ ] 小規模テスト成功

### 実行中
- [ ] プロセス実行中確認
- [ ] 進捗定期確認
- [ ] ログ監視

### 実行後
- [ ] 処理完了確認（77,346ファイル）
- [ ] Pickle生成確認（~15 shards期待）
- [ ] Stage2互換性チェック成功
- [ ] Stage2実行可能

---

## 🎯 主要コマンドまとめ

```bash
# 1. 小規模テスト
python -m scripts.clean_midi \
  --in data/loops --out output/test_drums_100 \
  --quarantine output/test_drums_100_q --instrument drums \
  --pickle-out output/test_drums_pkl --shard-size 50 \
  --emit-meta-json off --jobs 1

# 2. 本番実行
./scripts/run_drumloops_full.sh

# 3. 進捗確認
./scripts/status_drumloops.sh

# 4. 互換性チェック
python verify_stage2_compat.py output/drums_metadata

# 5. Stage2実行
./scripts/run_drumloops_stage2.sh
```

---

## 📝 レガシースクリプトについて

以下のスクリプトは **非推奨** です:

- `build_drumloops_metadata.py` - JSON経由の古い方式
- `build_index_from_json.py` - 不要（Pickle直書き）
- `append_to_index.py` - 不要（Pickle直書き）

新規利用は推奨しません。後方互換性のためのみ残されています。

---

**すべて準備完了！** 🚀

Stage2互換のPickle直書き方式で、安全かつ効率的にDrumloopsを処理できます。
