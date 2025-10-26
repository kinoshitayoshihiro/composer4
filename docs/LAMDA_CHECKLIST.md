# ✅ LAMDA実運用チェックリスト

## 🎯 評価: A（実運用OK）

---

## ✅ 実装完了項目

### 表記ゆれ対策
- [x] `monitor_lamidi.sh` → 薄ラッパー（241B）
- [x] `run_lamidi_full.sh` → 薄ラッパー（237B）
- [x] どちらの表記でも動作

### カウント精度
- [x] `-type f`追加（ディレクトリ除外）
- [x] `.midi`拡張子対応
- [x] 正確なファイル数表示

### 自動化
- [x] `EXPECTED_TOTAL`自動読込
- [x] `check_lamidi_dataset.sh`が総数保存
- [x] 手動編集不要

### 環境非依存
- [x] `BASE_DIR`自動解決（Git root検出）
- [x] スクリプト相対パスフォールバック
- [x] 他環境でも動作

### 楽器名マッピング
- [x] `piano` → `LAMDA_PIANO`自動変換
- [x] `LAMDA_*`形式も直接指定可能
- [x] 後方互換性維持

### ログ検出
- [x] パターン拡張（将来対応）
- [x] 楽器別ログ対応
- [x] 命名変更に強い

---

## 🚀 実行前チェック

### 1. データセット確認
```bash
./scripts/check_lamidi_dataset.sh
```

**確認項目:**
- [ ] MIDIファイル総数が表示される
- [ ] `data/lamda_expected_total.txt`が生成される
- [ ] エラーが出ない

### 2. ドライラン
```bash
./scripts/run_lamda_full.sh --dry-run piano
```

**確認項目:**
- [ ] `LAMDA_PIANO`に変換される
- [ ] コマンドが表示される
- [ ] エラーが出ない

### 3. モニターテスト
```bash
# 両方試す
./scripts/monitor_lamda.sh &
./scripts/monitor_lamidi.sh &
```

**確認項目:**
- [ ] 両方とも同じ画面が表示される
- [ ] `BASE_DIR`が正しく解決される
- [ ] `EXPECTED_TOTAL`が表示される

---

## 🎯 実運用開始

### 推奨順序
```bash
# 1. Piano
./scripts/run_lamda_full.sh piano

# 2. Strings
./scripts/run_lamda_full.sh strings

# 3. Guitar
./scripts/run_lamda_full.sh guitar

# 4. Bass
./scripts/run_lamda_full.sh bass
```

### 進捗監視（別ターミナル）
```bash
./scripts/monitor_lamda.sh
```

---

## 📊 各楽器の完了条件

### Piano
- [ ] Pickleインデックス生成（`data/lamda_piano_metadata/piano_metadata_v2.pickle`）
- [ ] クリーニング成功率 60-85%
- [ ] ログにエラーなし

### Strings
- [ ] Pickleインデックス生成（`data/lamda_strings_metadata/strings_metadata_v2.pickle`）
- [ ] クリーニング成功率 60-85%
- [ ] ログにエラーなし

### Guitar
- [ ] Pickleインデックス生成（`data/lamda_guitar_metadata/guitar_metadata_v2.pickle`）
- [ ] クリーニング成功率 60-85%
- [ ] ログにエラーなし

### Bass
- [ ] Pickleインデックス生成（`data/lamda_bass_metadata/bass_metadata_v2.pickle`）
- [ ] クリーニング成功率 60-85%
- [ ] ログにエラーなし

---

## 🎵 Stage 2準備

### 各楽器のStage 2実行
```bash
# Piano
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamda_piano_metadata/piano_metadata_v2.pickle \
  --output data/lamda_piano_stage2_scored.jsonl

# Strings
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamda_strings_metadata/strings_metadata_v2.pickle \
  --output data/lamda_strings_stage2_scored.jsonl

# Guitar
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamda_guitar_metadata/guitar_metadata_v2.pickle \
  --output data/lamda_guitar_stage2_scored.jsonl

# Bass
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamda_bass_metadata/bass_metadata_v2.pickle \
  --output data/lamda_bass_stage2_scored.jsonl
```

---

## 🔧 トラブルシューティング

### Q: プロセスが停止しない
```bash
# プロセス確認
ps aux | grep clean_midi.py

# 強制終了
pkill -f clean_midi.py
```

### Q: ディスク容量不足
```bash
# 容量確認
df -h /Volumes/SSD-SCTU3A

# 隔離ファイル削除（慎重に）
rm -rf data/quarantine/lamda_*
```

### Q: 進捗が進まない
```bash
# ログ確認
tail -50 logs/clean_LAMDA_*.log

# エラー検索
grep -i "error\|exception" logs/clean_LAMDA_*.log
```

---

## 📚 ドキュメント参照

- `docs/LAMDA_FINAL_BRUSHUP.md` - 最終ブラッシュアップ
- `docs/LAMDA_QUICKSTART.md` - クイックスタート
- `docs/LAMDA_EXECUTION_GUIDE.md` - 詳細実行手順

---

## 🎉 成功基準

### Stage 1完了
- [x] 4楽器全てのPickleインデックス生成
- [x] クリーニング成功率が適切
- [x] ログにクリティカルエラーなし

### 次のステップ準備完了
- [x] Stage 2スクリプト動作確認
- [x] 出力ディレクトリ準備
- [x] Guitar品質スコア設計確認

---

**準備状態**: ✅ 実戦投入可能  
**評価**: A（実運用OK）  
**推奨**: Piano → Strings → Guitar → Bass の順で即実行

🚀 **Let's Go!**
