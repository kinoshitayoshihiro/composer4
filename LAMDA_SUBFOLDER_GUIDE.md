# LAMDA サブフォルダ別処理ガイド

## 概要

Los-Angeles-MIDI データセット（~40万曲）を効率的に処理するため、16サブフォルダ（0-9, a-f）単位で pickle ファイルを生成します。

### メリット
- ✅ メモリ効率が良い（一度に約25,000曲のみ処理）
- ✅ 途中停止からの再開が簡単
- ✅ SSD接続不良時の影響が最小限
- ✅ 並列処理数を制御可能（`--jobs 4`でメモリ使用量を抑制）

## ディレクトリ構造

```
data/Los-Angeles-MIDI/MIDIs/
├── 0/      → lamda_piano_metadata/piano_shard_0.pickle (約25,000曲)
├── 1/      → lamda_piano_metadata/piano_shard_1.pickle
├── ...
└── f/      → lamda_piano_metadata/piano_shard_f.pickle

合計: 16 pickles × 5楽器 = 80 pickleファイル
```

## 使い方

### 1. Piano 全サブフォルダ処理（16個）
```bash
./scripts/run_lamda_by_subfolder.sh piano
```

### 2. Piano 特定サブフォルダのみ処理
```bash
# サブフォルダ 0, 1, 2 のみ
./scripts/run_lamda_by_subfolder.sh piano 0 1 2

# サブフォルダ 0 のみ
./scripts/run_lamda_by_subfolder.sh piano 0
```

### 3. 全楽器、全サブフォルダ処理（80個のpickle）
```bash
./scripts/run_lamda_by_subfolder.sh
```

### 4. 特定楽器のみ処理
```bash
./scripts/run_lamda_by_subfolder.sh strings     # Strings全16サブフォルダ
./scripts/run_lamda_by_subfolder.sh guitar 0 1  # Guitar サブフォルダ 0, 1のみ
./scripts/run_lamda_by_subfolder.sh bass        # Bass全16サブフォルダ
./scripts/run_lamda_by_subfolder.sh drums       # Drums全16サブフォルダ
```

## 実行順序の推奨

```bash
# 1. Piano から開始（最も重要）
./scripts/run_lamda_by_subfolder.sh piano

# 2. Strings
./scripts/run_lamda_by_subfolder.sh strings

# 3. Guitar
./scripts/run_lamda_by_subfolder.sh guitar

# 4. Bass
./scripts/run_lamda_by_subfolder.sh bass

# 5. Drums（既に drums_metadata があるため任意）
./scripts/run_lamda_by_subfolder.sh drums
```

## 進捗監視

### ログファイル確認
```bash
# 最新のログをリアルタイム監視
tail -f logs/lamda_piano_*_$(ls -t logs/lamda_piano_* | head -1 | cut -d_ -f4).log

# エラー確認
grep -i "error\|fail\|exception" logs/lamda_piano_*.log
```

### 生成済みpickleファイル確認
```bash
# Piano
ls -lh data/lamda_piano_metadata/*.pickle

# 全楽器
ls -lh data/lamda_*_metadata/*.pickle

# 統計
find data/lamda_*_metadata -name "*.pickle" | wc -l
```

### Pickle内容確認
```bash
python3 -c "
import pickle
with open('data/lamda_piano_metadata/piano_shard_0.pickle', 'rb') as f:
    data = pickle.load(f)
print(f'Version: {data[\"version\"]}')
print(f'Instrument: {data[\"instrument\"]}')
print(f'Entries: {data[\"count\"]}')
print(f'Total notes: {data[\"summary\"][\"total_notes\"]}')
print(f'Avg BPM: {data[\"summary\"][\"avg_bpm\"]:.1f}')
"
```

## 処理設定

### デフォルト設定
- **並列処理**: `--jobs 4` (4並列、メモリ節約)
- **シャードサイズ**: `--shard-size 100000` (サブフォルダ全体を1ファイルにまとめる)
- **レジューム**: `--resume` 有効（既存pickleはスキップ）
- **メタJSON**: `--emit-meta-json off` (pickle直書き、JSON不要)

### メモリ不足時の対策
スクリプト内の`--jobs 4`を変更：
```bash
# run_lamda_by_subfolder.sh の Line 87付近
--jobs 2 \       # 並列数を2に削減
# または
--jobs 1 \       # 直列処理（最もメモリ効率良い）
```

## 再開機能

既存のpickleファイルがある場合、そのサブフォルダはスキップされます：
```bash
# サブフォルダ 0 の pickle が既にある場合
$ ./scripts/run_lamda_by_subfolder.sh piano 0
✅ SKIP: Pickle already exists: data/lamda_piano_metadata/piano_shard_0.pickle
```

## トラブルシューティング

### 問題: すべてのファイルが隔離（quarantine）される
**原因**: 並列処理でグローバル変数が正しく初期化されていない  
**解決**: 最新版では`ProcessPoolExecutor`の`initializer`で修正済み

### 問題: SSD接続が切れた
**解決**: 再接続後、同じコマンドを再実行（既存pickleはスキップされる）
```bash
./scripts/run_lamda_by_subfolder.sh piano
```

### 問題: メモリ不足で停止
**解決**: 並列数を削減
1. `scripts/run_lamda_by_subfolder.sh`を編集
2. Line 87: `--jobs 4` → `--jobs 1`
3. 再実行

### 問題: 特定サブフォルダだけ失敗
**解決**: そのサブフォルダのみ再実行
```bash
# サブフォルダ 5 だけ失敗した場合
./scripts/run_lamda_by_subfolder.sh piano 5
```

## 処理時間の目安

- **1サブフォルダ（約25,000曲）**: 5-15分
- **1楽器（16サブフォルダ）**: 1.5-4時間
- **全5楽器（80 pickles）**: 7.5-20時間

※ 処理時間はMacのスペック、SSD速度、並列数（--jobs）に依存

## 次のステップ

全pickleファイルが生成されたら：

1. **検証**: 全pickleファイルが存在するか確認
   ```bash
   find data/lamda_*_metadata -name "*.pickle" | wc -l
   # 期待値: 80 (5楽器 × 16サブフォルダ)
   ```

2. **Stage2学習**: 各楽器の学習データとして使用
   ```bash
   # 例: Piano
   python scripts/train_stage2.py \
       --instrument piano \
       --data data/lamda_piano_metadata/*.pickle
   ```

3. **Docker統合**: .dockerignoreに追加
   ```
   !data/lamda_piano_metadata/*.pickle
   !data/lamda_strings_metadata/*.pickle
   !data/lamda_guitar_metadata/*.pickle
   !data/lamda_bass_metadata/*.pickle
   !data/lamda_drums_metadata/*.pickle
   ```
