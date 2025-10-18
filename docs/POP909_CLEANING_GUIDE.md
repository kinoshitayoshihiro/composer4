# POP909 クリーニング実行ガイド

## 📋 概要

POP909データセット（2,898 MIDIファイル）の完全クリーニング手順

**推定所要時間**: 1-3時間（マシンスペック依存）

---

## 🚀 実行方法

### 1️⃣ バックグラウンド実行（推奨）

```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

# バックグラウンドで実行
nohup ./scripts/run_pop909_full.sh > /dev/null 2>&1 &

# プロセスID確認
echo $!

# または
ps aux | grep run_pop909_full.sh | grep -v grep
```

**現在実行中**: PID 76175

---

## 📊 進行状況の確認方法

### 方法1: リアルタイムモニター（別ターミナル推奨）

```bash
./scripts/monitor_pop909.sh
```

**表示内容**:
- プログレスバー
- 処理済み/隔離ファイル数
- リアルタイムログ
- 10秒ごと自動更新

### 方法2: ログファイル確認

```bash
# 最新ログをtail
tail -f logs/pop909_cleaning_$(date +%Y%m%d)*.log

# ログ一覧
ls -lht logs/
```

### 方法3: ファイル数カウント

```bash
# クリーニング済み
find data/cleaned/pop909 -name "*.mid" | wc -l

# 隔離済み
find data/quarantine/pop909 -name "*.mid" | wc -l

# 合計
echo $(($(find data/cleaned/pop909 -name "*.mid" | wc -l) + $(find data/quarantine/pop909 -name "*.mid" | wc -l)))
```

### 方法4: メタインデックス確認

```bash
# 処理済みエントリ数
wc -l data/cleaned/pop909/meta_index.jsonl

# 最新5件
tail -5 data/cleaned/pop909/meta_index.jsonl | jq .
```

---

## ⏸️ 処理の一時停止/再開

### 一時停止

```bash
# プロセスID確認
ps aux | grep run_pop909_full.sh | grep -v grep

# 一時停止 (SIGSTOPシグナル)
kill -STOP <PID>
```

### 再開

```bash
kill -CONT <PID>
```

### 完全停止

```bash
kill <PID>
```

**注意**: `--force`フラグなしで再実行すれば、既存の`.meta.json`はスキップされます（再入可能）

---

## ✅ 完了後の確認

### 統計確認

```bash
# 最終統計（ログファイル末尾）
tail -30 logs/pop909_cleaning_*.log

# ファイル数集計
echo "Cleaned: $(find data/cleaned/pop909 -name '*.mid' | wc -l)"
echo "Quarantined: $(find data/quarantine/pop909 -name '*.mid' | wc -l)"
```

### 出力ファイル確認

```bash
# ディレクトリ構造
tree -L 2 data/cleaned/pop909
tree -L 2 data/quarantine/pop909

# サンプルファイル
ls data/cleaned/pop909/*.mid | head -5
ls data/cleaned/pop909/*.meta.json | head -5
```

---

## 🎯 次のステップ

### Step 1: Quality Gates検証

```bash
python3 scripts/validate_and_gate.py \
  --in data/cleaned/pop909 \
  --gates configs/quality_gates/quality_gates.yaml \
  --report reports/pop909_validation.json \
  --summary reports/pop909_summary.jsonl \
  --fail-on-critical
```

### Step 2: Train/Val/Test分割

```bash
python3 scripts/prepare_splits.py \
  --in data/cleaned/pop909 \
  --out data/cleaned/pop909_splits \
  --seed 42 \
  --min-bucket 3
```

### Step 3: 統計レポート確認

```bash
# クリーニングレポート
cat data/cleaned/piano_clean_report.json | jq .

# 検証サマリ
cat reports/pop909_summary.jsonl | jq -s 'group_by(.passed) | map({passed: .[0].passed, count: length})'

# 分割統計
cat data/cleaned/pop909_splits/split_summary.json | jq .
```

---

## 🔧 トラブルシューティング

### 問題1: プロセスが止まっている

```bash
# プロセス確認
ps aux | grep python3 | grep clean_midi

# Pythonプロセスのスタックトレース
kill -USR1 <PYTHON_PID>
```

### 問題2: ディスク容量不足

```bash
# ディスク使用量確認
df -h /Volumes/SSD-SCTU3A

# POP909データサイズ確認
du -sh data/POP909
du -sh data/cleaned/pop909
du -sh data/quarantine/pop909
```

### 問題3: メモリ不足

```bash
# メモリ使用量確認
top -pid $(pgrep -f clean_midi.py)

# --jobs 1 で直列実行中のはず（メモリ使用量は最小）
```

### 問題4: 再実行したい

```bash
# 既存出力を削除
rm -rf data/cleaned/pop909
rm -rf data/quarantine/pop909

# または --force フラグで上書き
python3 scripts/clean_midi.py \
  --in data/POP909 \
  --out data/cleaned/pop909 \
  --instrument piano \
  --quarantine data/quarantine/pop909 \
  --jobs 1 \
  --seed "pop909-v1" \
  --force
```

---

## 📈 予想される結果

### ファイル分布（テストから推定）

- **Total**: 2,898 files
- **Cleaned**: ~2,700-2,800 files (93-97%)
- **Quarantined**: ~100-200 files (3-7%)

### 主な隔離理由（推定）

1. `pedal_excessive` - ペダル使用過多
2. `velocity_variation_low` - ベロシティ変化が少ない
3. `hand_separation_low` - 左右の手が分離されていない
4. `note_count_low` - ノート数が少ない

### 処理時間の目安

- **ファイル列挙**: 5-10分（2,898ファイル）
- **クリーニング**: ~1秒/ファイル = 48分
- **メタデータ保存**: 含まれる
- **合計**: 約1-1.5時間

---

## 📝 現在の状況

**開始時刻**: 2025-10-15 16:55:16  
**プロセスID**: 76175  
**ログファイル**: `logs/pop909_cleaning_20251015_165516.log`  
**ステータス**: ファイル列挙中（`stable_list_midis`実行中）

**確認コマンド**:
```bash
# プロセス確認
ps aux | grep 76175

# ログ確認
tail -f logs/pop909_cleaning_20251015_165516.log

# 進行状況（ファイル数）
find data/cleaned/pop909 -name "*.mid" | wc -l
```

---

**時間をかけて問題ありません！処理が完了するまで待ちましょう** 🎵✨
