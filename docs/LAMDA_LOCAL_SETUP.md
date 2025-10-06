# LAMDa ローカルテスト - クイックセットアップ
# ===========================================

## 📦 依存関係インストール

```bash
# music21のみインストール (最小構成)
/opt/homebrew/bin/python3 -m pip install --user music21

# または requirements-lamda.txtから
/opt/homebrew/bin/python3 -m pip install --user -r requirements-lamda.txt
```

**注意**: macOS Homebrew Python環境では `--user` フラグが必要です。

## 🧪 テスト実行手順

### Step 1: テストサンプル作成

```bash
/opt/homebrew/bin/python3 scripts/create_test_sample.py
```

**出力**: `data/Los-Angeles-MIDI/TEST_SAMPLE/` (100サンプル)

### Step 2: ローカルビルド

```bash
/opt/homebrew/bin/python3 scripts/test_local_build.py
```

**推定時間**: 2-5分  
**出力**: `data/test_lamda.db`

## ✅ 成功時の出力例

```
🧪 LAMDa Local Test - Small Sample (100 entries)
================================================================================

📁 Test data directory: data/Los-Angeles-MIDI/TEST_SAMPLE
💾 Test database path: data/test_lamda.db

📊 Initializing LAMDaUnifiedAnalyzer...

🔨 Building test database...
   (This should take 2-5 minutes for 100 samples)

📁 Processing CHORDS_DATA...
  Analyzing sample_100.pickle...

📁 Processing KILO_CHORDS_DATA...
  Loading sequences...

📁 Processing SIGNATURES_DATA...
  Loading signatures...

✅ Database built successfully in 142.3 seconds!
   Database size: 245.7 KB

🔍 Validating database...
   Tables found: progressions, kilo_sequences, signatures
   • progressions: 87 records
   • kilo_sequences: 100 records
   • signatures: 100 records
   • Linked records (progressions ↔ kilo): 81

✅ Database validation passed!

📈 Performance Estimation:
   Test build time: 142.3s for 100 samples
   Time per sample: 1.423s

🔮 Full build estimation (assuming ~180,000 samples):
   Estimated time: 4267.4 minutes (71.1 hours)
   Estimated cost: ¥1635 (at ¥23/hour on Vertex AI)

🎉 Local test completed successfully!
```

## 🐛 トラブルシューティング

### ModuleNotFoundError: No module named 'music21'

```bash
# 解決策: userインストール
/opt/homebrew/bin/python3 -m pip install --user music21
```

### externally-managed-environment エラー

```bash
# macOS Homebrew Python では --user フラグ必須
/opt/homebrew/bin/python3 -m pip install --user パッケージ名
```

### Permission Denied

```bash
# システムPythonを使わず、Homebrew Pythonを使用
which python3  # /opt/homebrew/bin/python3 を確認
```

## 📊 データベース検証

テスト成功後、以下で検証:

```bash
# SQLiteで開く
sqlite3 data/test_lamda.db

# レコード数確認
SELECT COUNT(*) FROM progressions;
SELECT COUNT(*) FROM kilo_sequences;
SELECT COUNT(*) FROM signatures;

# サンプルデータ表示
SELECT * FROM progressions LIMIT 1;
```

## 🚀 次のステップ

ローカルテスト成功後:

1. **コードレビュー**: 動作確認完了
2. **Vertex AI実行**: [LAMDA_EXECUTION_CHECKLIST.md](LAMDA_EXECUTION_CHECKLIST.md) 参照
3. **フルデータベース構築**: 180,000サンプル × 90-120分

詳細は [LAMDA_README.md](LAMDA_README.md) を参照してください。
