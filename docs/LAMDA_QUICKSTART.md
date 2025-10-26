# LAMDA (Los-Angeles-MIDI) クリーニング - クイックスタート

## 🚀 すぐに始める

### 1. データセット確認
```bash
./scripts/check_lamidi_dataset.sh
```
実行後、総ファイル数が自動保存されます（`data/lamda_expected_total.txt`）。

### 2. ドライラン
```bash
./scripts/run_lamda_full.sh --dry-run
```

### 3. 実行（推奨: 楽器別に順次実行）
```bash
# Piano
./scripts/run_lamda_full.sh piano

# Strings
./scripts/run_lamda_full.sh strings

# Guitar
./scripts/run_lamda_full.sh guitar

# Bass
./scripts/run_lamda_full.sh bass
```

### 4. 進捗監視（別ターミナル）
```bash
./scripts/monitor_lamda.sh
```

---

## 📊 LAMDAとは？

- **総ファイル数**: 約40万件
- **タイプ**: マルチトラック楽曲データセット
- **楽器**: piano, strings, guitar, bass, drums等

---

## 🎯 処理内容

### 楽器別クリーニング
1. **LAMDA_PIANO** - ピアノパート抽出 + 品質チェック
2. **LAMDA_STRINGS** - ストリングスパート抽出
3. **LAMDA_GUITAR** - ギターパート抽出（ストラム偏重）
4. **LAMDA_BASS** - ベースパート抽出
5. **LAMDA_DRUMS** - ドラムパート抽出（オプション）

### 品質ゲート
- 楽器判定（GM Program Number）
- 音域チェック
- ノート密度チェック
- ベロシティ分布
- 楽器別特性チェック（ストラム/アルペジオ等）

---

## 📂 出力

```
data/
├── cleaned/
│   ├── lamda_piano/      # ✅ クリーニング済み
│   ├── lamda_strings/
│   ├── lamda_guitar/
│   └── lamda_bass/
├── quarantine/
│   └── lamda_*/          # 🗑️ 品質不適合
└── lamda_*_metadata/     # 📦 Pickleメタデータ
```

---

## 📚 詳細ドキュメント

- **実行手順**: `docs/LAMDA_EXECUTION_GUIDE.md`
- **詳細ガイド**: `docs/LAMIDI_CLEANING_GUIDE.md`
- **セットアップ**: `docs/LAMIDI_SETUP_SUMMARY.md`

---

## ⚙️ 設定ファイル

- **データセット定義**: `scripts/run_dataset_full.sh`
- **実行スクリプト**: `scripts/run_lamda_full.sh`
- **モニター**: `scripts/monitor_lamda.sh`
- **楽器別クリーナー**: `scripts/cleaners/*.py`

---

## 🔧 よくある質問

**Q: 全楽器を一度に実行できますか？**
A: 可能ですが、40万ファイルの大規模処理のため、楽器別に順次実行を推奨します。

**Q: drumsが自動実行されないのはなぜ？**
A: 母数が巨大なため、明示的に指定した場合のみ実行されます。
```bash
./scripts/run_lamda_full.sh drums
```

**Q: 中断から再開できますか？**
A: はい。`--resume`オプションが既定で有効です。同じコマンドを再実行するだけで、既存シャードから自動再開します。

**Q: 処理時間の目安は？**
A: 楽器・並列度・ファイル数により変動します。Piano（8並列）で数時間～1日程度を想定してください。

---

## 🎯 次のステップ

Stage 1完了後:

```bash
# Stage 2: メタデータ抽出
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamda_piano_metadata/piano_metadata_v2.pickle \
  --output data/lamda_piano_stage2_scored.jsonl
```

---

## 🐛 トラブルシューティング

```bash
# ログ確認
tail -50 logs/clean_LAMDA_*.log

# プロセス確認
ps aux | grep clean_midi.py

# ディスク容量確認
df -h /Volumes/SSD-SCTU3A
```
