# KenLM 言語モデル セットアップガイド

## 現在の状態

### ✅ 完了済み
1. **KenLM Pythonバインディングのインストール**
   - `kenlm`パッケージがインストール済み
   - Pythonから言語モデルを読み込み可能

2. **コーパス生成スクリプトの改善**
   - `make_lm_corpus_plus_v2.py` を作成
   - エラーハンドリング強化、進捗表示追加、詳細レポート機能追加
   - 26,107文のコーパスを生成済み

3. **Google Cloud Vision API設定**
   - サービスアカウント認証完了
   - OCR処理が正常に動作

### ✅ すべて完了済み！
1. **KenLMバイナリツールのビルド**
   - `lmplz`および`build_binary`コマンドのビルド完了
   - Boost@1.85を使用してビルド成功

2. **言語モデルファイルの作成**
   - `modern_ja.bin`ファイルの作成完了（88MB）
   - `modern_ja.arpa`ファイルも生成済み（143MB）
   - 5-gram言語モデル、4,819種類のユニグラム
   - OCRスクリプトで使用可能な状態

### ⚠️ 重要な問題と解決策

**ocr2novel_integrated.py の問題点:**
- レイアウト再構築処理（列分割・行グループ化）が語順を破壊
- 改行を強制削除して読みにくいテキストを生成
- KenLM文脈補正が不要な語順変更を引き起こす

**解決策:**
- `ocr_clean.py` を使用（シンプルで正確なOCR）
- Google Vision APIの `full_text_annotation.text` をそのまま使用
- 旧字→新字変換のみ実施、レイアウト破壊処理は一切なし

## KenLM言語モデルの作成方法

### 方法1: ビルドが完了後（推奨）

```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/kenlm-master

# ビルドが完了したか確認
ls -la build/bin/lmplz
ls -la build/bin/build_binary

# ARPAフォーマットでN-gram言語モデルを学習（5-gram）
build/bin/lmplz -o 5 \
  < /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/ocr_output/corpus_v2.char.txt \
  > /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/models/modern_ja.arpa

# バイナリ形式に変換（高速読み込み用）
build/bin/build_binary \
  /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/models/modern_ja.arpa \
  /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/models/modern_ja.bin
```

### 方法2: KenLMリポジトリの手動ビルド

```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/kenlm-master

# Boostの問題を回避してビルド
mkdir -p build
cd build
cmake .. -DCOMPILE_TESTS=OFF
make -j 4

# 成功したらバイナリを確認
ls -la bin/lmplz
ls -la bin/build_binary
```

### 方法3: Pythonから直接学習（実験的）

```python
# KenLMのPython APIを使用（詳細は公式ドキュメント参照）
import kenlm

# ARPAファイルから読み込み
model = kenlm.LanguageModel('/path/to/model.arpa')

# スコア計算
score = model.score('これ は テスト です', bos=True, eos=True)
print(f'Log probability: {score}')
```

## OCRスクリプトの使用

### 推奨: ocr_clean.py（シンプルで正確）

Google Vision APIの結果をそのまま使用する、シンプルで正確なOCRスクリプト:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/marutakesyobou-555e0d7946df.json"

python ocr_clean.py \
  --input-dir '/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/ocr_targets/化物五十人力' \
  --output '/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/ocr_output/化物五十人力_clean.txt' \
  --kyujitai-csv '/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/maps/kyujitai_map.csv' \
  --blocklist '/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/maps/blocklist.txt'
```

**特徴:**
- 語順を保持（Vision APIの結果をそのまま使用）
- 改行を保持（原文のレイアウトを維持）
- 旧字→新字変換のみ実施
- レイアウト破壊処理なし

### 非推奨: ocr2novel_integrated.py（レイアウト再構築あり）

⚠️ **このスクリプトには重大な問題があります:**
- 座標ベースの列分割・行グループ化が語順を破壊
- 改行を強制削除して読みにくいテキストを生成
- KenLM文脈補正が不要な語順変更を引き起こす

**使用は推奨しません。**

## トラブルシューティング

### KenLMビルドがBoostエラーで失敗する場合

1. **Boostのバージョンを確認:**
   ```bash
   brew info boost
   ```

2. **古いバージョンのBoostをインストール:**
   ```bash
   # KenLMは古いBoost APIに依存している可能性があります
   brew install boost@1.76  # 例
   ```

3. **CMakeでBoostのパスを指定:**
   ```bash
   cmake .. -DBOOST_ROOT=/opt/homebrew/opt/boost@1.76
   ```

### KenLMなしでOCRを使用

KenLMモデルがなくても、OCR処理は正常に動作します:
- Google Cloud Vision APIによる文字認識
- 旧字体→新字体変換
- ルビ除去
- レイアウト補正

KenLMは主に**文脈に基づく補正**に使用されますが、必須ではありません。

## 生成されたファイル

```
ocr_data/
├── ocr_output/
│   ├── corpus.char.txt          # 元のコーパス (28,212文)
│   ├── corpus.word.txt          # 元のコーパス (単語分割)
│   ├── corpus_v2.char.txt       # 改善版コーパス (26,107文)
│   ├── corpus_v2.word.txt       # 改善版コーパス (単語分割)
│   ├── corpus_v2_report.json    # 詳細レポート
│   └── novel.txt                # OCR結果 (12,381文字)
├── models/
│   └── (modern_ja.bin を配置予定)
└── kenlm-master/
    └── (KenLMソースコード)
```

## 次のステップ

1. KenLMのビルドが完了するまで待つ
2. `lmplz`でコーパスから言語モデルを学習
3. `build_binary`でバイナリ形式に変換
4. OCRスクリプトで`--kenlm`オプション付きで実行し、精度を確認

## 参考リンク

- [KenLM公式サイト](https://kheafield.com/code/kenlm/)
- [KenLM GitHub](https://github.com/kpu/kenlm)
- [KenLM言語モデルの学習](https://kheafield.com/code/kenlm/estimation/)
