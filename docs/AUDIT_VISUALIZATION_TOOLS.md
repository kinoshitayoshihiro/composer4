# LAMDA監査・可視化ツール群（v1.2+）

**目的**: LAMDA Stage2拡張メタの品質保証と一貫性監視のための5つの監査・可視化スクリプト

---

## 📦 ツール一覧

### 1. `auto_file_id_map.py` - ファイルID自動マッピング

**機能**:
- GOLD（Stage2 JSON）とEXT（外部chordmap）のファイル名不一致を自動解決
- MD5ハッシュ完全一致（Exact Pass）
- 小節長許容（±2小節）+ ハミング距離近似（Approximate Pass）

**出力**: `mappings/file_map.csv`
- カラム: `file_id, ext_base, gold_base`

**使用例**:
```bash
python scripts/auto_file_id_map.py \
  --gold-dir data/GOLD_stage2_json \
  --ext-dir  data/lamda_chordmaps \
  --out-csv  mappings/file_map.csv \
  --bars-tol 2
```

**効き所**: 
- すべてのA/B監査の土台
- Top-Nサムネ生成の前提
- 長期リング監視の一貫性保証

---

### 2. `audit_kilo_vs_chords.py` - KILO vs CHORDS一貫性監査

**機能**:
- LAMDAの統合KILO（1ファイル）と分割CHORDS（162シャード）の整合性検証
- 小節単位のコード進行比較

**出力**: `analysis/kilo_vs_chords_audit.csv`
- カラム: `file_id, bars, match_rate, n_diff, diff_bars, kilo_first5, chords_first5`

**使用例**:
```bash
python scripts/audit_kilo_vs_chords.py \
  --kilo data/Los-Angeles-MIDI/KILO_CHORDS_DATA/LAMDa_KILO_CHORDS_DATA.pickle \
  --chords-dir data/Los-Angeles-MIDI/CHORDS_DATA \
  --out-csv analysis/kilo_vs_chords_audit.csv \
  --tpq 480 \
  --max-files 0  # 0=全件
```

**判断基準**:
- `match_rate ≥ 0.95` → KILOを正として採用
- `match_rate < 0.85` → CHORDSを優先、またはマニュアル確認

---

### 3. `ringbuffer_append.py` - リングバッファ追記（時系列監視）

**機能**:
- 監査CSVの平均一致率をJSONL形式のリングバッファに追記
- タイムスタンプ + タグ + メトリック値

**出力**: `analysis/consistency_ring.jsonl`
- フォーマット: `{timestamp, tag, metric, value, count}`

**使用例**:
```bash
python scripts/ringbuffer_append.py \
  --csv  analysis/ext_vs_gold.csv \
  --ring analysis/consistency_ring.jsonl \
  --tag  stage2_patch_v1.2
```

**運用**:
- CI/定期実行で一致率の推移を監視
- 閾値（例: 0.85）を下回ったら警告

---

### 4. `ringbuffer_report.py` - リングバッファ→CSV変換

**機能**:
- JSONL形式のリングバッファをCSVに変換
- 時系列グラフ・トレンド分析用

**出力**: `analysis/consistency_ring.csv`
- カラム: `timestamp, tag, metric, value, count`

**使用例**:
```bash
python scripts/ringbuffer_report.py \
  --ring    analysis/consistency_ring.jsonl \
  --out-csv analysis/consistency_ring.csv
```

**応用**:
- Excelやグラフツールで時系列プロット
- 回帰検出（一致率の急落）

---

### 5. `build_topn_thumbs.py` - Top-N差分サムネ一括生成

**機能**:
- 監査CSVから不一致率が高い上位N件を抽出
- 各ファイルの差分を1行ヒートマップPNGとして可視化

**依存**: `visualize_chord_diff.py`（ヒートマップ生成）

**出力**: `analysis/thumbs/*.png`

**使用例**:
```bash
python scripts/build_topn_thumbs.py \
  --audit-csv analysis/ext_vs_gold.csv \
  --gold-dir  data/GOLD_stage2_json \
  --ext-dir   data/lamda_chordmaps \
  --out-dir   analysis/thumbs \
  --top-n 50 \
  --mapping-csv mappings/file_map.csv
```

**可視化仕様**:
- 横軸: 小節番号
- 色: 緑=一致（0.0）、赤=不一致（1.0）
- サイズ: 1行 × N列（N=曲の総小節数）

---

## 🔄 推奨ワークフロー

### パターンA: 一貫性監視（CI/定期実行）

```bash
# 1. ファイルマッピング生成（初回のみ）
python scripts/auto_file_id_map.py \
  --gold-dir data/GOLD_stage2_json \
  --ext-dir  data/lamda_chordmaps \
  --out-csv  mappings/file_map.csv

# 2. KILO vs CHORDS監査
python scripts/audit_kilo_vs_chords.py \
  --kilo data/LAMDa_KILO_CHORDS_DATA.pickle \
  --chords-dir data/CHORDS_DATA \
  --out-csv analysis/kilo_vs_chords.csv

# 3. A/B監査（既存ツール）
python scripts/ab_chord_audit.py \
  --ext-dir data/lamda_chordmaps \
  --int-dir output/stage2/json \
  --out-csv analysis/ext_vs_gold.csv

# 4. リング追記（時系列記録）
python scripts/ringbuffer_append.py \
  --csv  analysis/ext_vs_gold.csv \
  --ring analysis/consistency_ring.jsonl \
  --tag  daily_audit

# 5. CSV変換（グラフ用）
python scripts/ringbuffer_report.py \
  --ring    analysis/consistency_ring.jsonl \
  --out-csv analysis/consistency_ring.csv

# 6. Top-N可視化（任意）
python scripts/build_topn_thumbs.py \
  --audit-csv analysis/ext_vs_gold.csv \
  --gold-dir  data/GOLD_stage2_json \
  --ext-dir   data/lamda_chordmaps \
  --out-dir   analysis/thumbs \
  --top-n 20 \
  --mapping-csv mappings/file_map.csv
```

### パターンB: 新規データ投入時の品質ゲート

```bash
# 1. Stage2拡張実行
python scripts/lamda_stage2_extractor.py \
  --input-dir  output/stage1/clean \
  --output-dir output/stage2/new_batch \
  --lamda-chords-dir data/lamda_chordmaps \
  --whitelist-validate 1

# 2. 即時監査
python scripts/ab_chord_audit.py \
  --ext-dir data/lamda_chordmaps \
  --int-dir output/stage2/new_batch/json \
  --out-csv analysis/new_batch_audit.csv

# 3. 品質ゲート判定
awk -F, 'NR>1 {sum+=$3; n++} END {if(sum/n < 0.85) exit 1}' \
  analysis/new_batch_audit.csv && echo "✅ PASS" || echo "❌ FAIL"

# 4. FAIL時: Top-N可視化で原因特定
python scripts/build_topn_thumbs.py \
  --audit-csv analysis/new_batch_audit.csv \
  --gold-dir  output/stage2/new_batch/json \
  --ext-dir   data/lamda_chordmaps \
  --out-dir   analysis/new_batch_thumbs \
  --top-n 10
```

---

## 📊 メトリクス解釈ガイド

### `match_rate`（一致率）
- **≥ 0.95**: 優秀（SILVER昇格候補）
- **0.85–0.94**: 合格（GOLD維持）
- **0.70–0.84**: 要確認（手動監査推奨）
- **< 0.70**: 不合格（再抽出または除外）

### `n_diff`（不一致小節数）
- 絶対値として評価
- 例: 32小節曲で `n_diff=2` → `match_rate=0.9375`
- Top-Nソート基準: `(match_rate昇順, n_diff降順)`

### `diff_bars`（不一致小節番号）
- セミコロン区切りの小節インデックス
- 例: `0;15;31` → イントロ・ブリッジ・アウトロで不一致
- 可視化で赤マーカー箇所を確認

---

## 🛠️ 依存関係

### Python標準ライブラリ
- `argparse`, `csv`, `json`, `os`, `glob`, `pickle`, `hashlib`, `datetime`, `subprocess`

### 外部ライブラリ
- `numpy` (visualize_chord_diff.py用)
- `matplotlib` (PNG生成用)

### インストール
```bash
pip install numpy matplotlib
```

---

## 🎯 合格基準（品質ゲート）

| 段階               | 一致率閾値       | 用途                  |
| ---------------- | ----------- | ------------------- |
| GOLD（最小基準）      | ≥ 0.70      | Stage2拡張の入力として許容   |
| PRODUCTION（推奨基準） | ≥ 0.85      | Teacher v1学習データとして推奨 |
| SILVER（昇格基準）    | ≥ 0.95      | 高品質データとしてティア昇格      |
| PLATINUM（理想）    | ≥ 0.98      | リファレンス用（手動検証済み）     |

---

## 📝 運用ヒント

### 1. マッピングCSVの更新タイミング
- 新規データ追加時
- ファイル名変更時
- 外部chordmap更新時

### 2. リングバッファの回転
- 最大1万レコードで古いものを削除（任意）
- タグで運用フェーズを区別（例: `v1.2_pilot`, `v1.3_prod`）

### 3. Top-N可視化の活用
- 不一致パターンの分類（イントロ系／転調系／誤検出系）
- ホワイトリスト拡張の判断材料
- 外部chordmapの品質フィードバック

### 4. KILO vs CHORDS の使い分け
- 統合処理: KILO（高速）
- 分散処理: CHORDS（並列化容易）
- 一貫性監査で定期的に検証

---

## 🔗 関連ドキュメント

- `docs/LAMDA_V12_IMPLEMENTATION_GUIDE.md` - v1.2完全実装ガイド
- `docs/LAMDA_INTEGRATION_PLAN.md` - 統合計画
- `docs/LAMDA_IMPLEMENTATION_STATUS.md` - 実装状況
- `scripts/ab_chord_audit.py` - 既存A/B監査スクリプト

---

## ✅ クイックスタート（3ステップ）

```bash
# 1. マッピング生成
python scripts/auto_file_id_map.py \
  --gold-dir data/GOLD_stage2_json \
  --ext-dir data/lamda_chordmaps \
  --out-csv mappings/file_map.csv

# 2. 監査実行
python scripts/audit_kilo_vs_chords.py \
  --kilo data/LAMDa_KILO_CHORDS_DATA.pickle \
  --chords-dir data/CHORDS_DATA \
  --out-csv analysis/kilo_vs_chords.csv

# 3. 可視化
python scripts/build_topn_thumbs.py \
  --audit-csv analysis/kilo_vs_chords.csv \
  --gold-dir data/GOLD_stage2_json \
  --ext-dir data/lamda_chordmaps \
  --out-dir analysis/thumbs \
  --top-n 10 \
  --mapping-csv mappings/file_map.csv
```

---

**実装完了率**: 100%（5スクリプト + 1可視化補助）

**スキーマバージョン**: lamda_v2.3+対応

**最終更新**: 2025-10-23
