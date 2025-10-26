# SSD切断対策ガイド

## 🔴 発生した問題

**症状**: 外付けSSD (`/Volumes/SSD-SCTU3A/`) が処理中に切断
- `OSError: [Errno 6] Device not configured`
- `Input/output error`
- `Bus error: 10`

**影響**: 54,561/56,598曲処理時点（96.4%）でクラッシュ

## ✅ 対策（実行前に必須）

### 1. **Macのスリープ防止**
```bash
# スリープ無効化（処理中のみ）
caffeinate -i bash scripts/run_stage2_resume.sh

# または
sudo pmset -a disablesleep 1
```

### 2. **USBケーブル確認**
- ✅ USB-Cケーブルが奥まで挿さっているか確認
- ✅ 他のUSBデバイスを外す（電力不足回避）
- ✅ Macに直接接続（ハブ経由は避ける）

### 3. **SSD電力管理の無効化**
```bash
# ディスクのスリープを無効化
sudo pmset -a disksleep 0

# AutoFS（自動マウント）の無効化（オプション）
sudo automount -vc
```

### 4. **Resume機能で再開**
```bash
# 処理済みファイルをスキップして残りを処理
bash scripts/run_stage2_resume.sh
```

## 📊 現在の状況

```
処理済み: 32,924曲 (JSONファイル保存済み)
残り:     ~23,674曲
成功率:   99.2%
```

## 🚀 再開手順

### ステップ1: スリープ防止
```bash
# ターミナルで実行（別ウィンドウで開いておく）
caffeinate -d
```

### ステップ2: SSD接続確認
```bash
# SSDがマウントされているか確認
df -h | grep SSD-SCTU3A
```

### ステップ3: Resume実行
```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3
bash scripts/run_stage2_resume.sh
```

## ⚠️ トラブルシューティング

### Q: また切断された場合は？
A: Resumeスクリプトを再実行すれば、前回までの処理をスキップして続きから再開します。

### Q: 処理速度が遅くなった場合は？
A: SSDの温度上昇によるサーマルスロットリングの可能性。
```bash
# ファン速度を上げる（smcFanControlなどのアプリを使用）
# または、一時停止して冷却後に再開
```

### Q: Resumeが動かない場合は？
A: 処理済みファイルリストを手動で確認：
```bash
# 処理済みファイル数
find output/stage2_production/json -name "*.json" | wc -l

# Resumeファイルの行数（一致するはず）
wc -l logs/stage2_processed_files.txt
```

## 📈 予想所要時間

- 残り約23,674曲
- 速度: 約3.8 files/s（安定時）
- 所要時間: 約1.7時間

## 💡 今後の改善

1. **バッチサイズ縮小**: 5,000曲ごとにチェックポイント保存
2. **ローカルSSD使用**: 内蔵SSDへの一時保存後、外付けにコピー
3. **並列処理制限**: CPU使用率を下げてSSD負荷軽減
