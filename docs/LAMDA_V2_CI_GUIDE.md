# LAMDA v2.6+ CI/CD ガイド

## 概要

LAMDA v2.6+では、品質保証のためのCI/CDパイプラインを提供しています。

## 品質ゲート

### 1. Match Rate Gate
**コード認識の一致率を検証**

- **閾値**: ≥ 0.85 (85%)
- **対象**: A/B chord audit結果
- **データ**: `analysis/ab_chords_audit.csv`

```bash
# A/B audit実行
python scripts/ab_chord_audit.py \
  --ext-dir data/lamda_chordmaps \
  --int-dir output/stage2/json \
  --out-csv analysis/ab_chords_audit.csv
```

### 2. Controls Integrity Gate
**MIDI制御情報の健全性を検証**

- **閾値**: ≥ 0.99 (99%)
- **検証項目**:
  - Pitch Bend範囲: [-8191, 8191]
  - CC値範囲: [0, 127]
  - 値の整合性 (min ≤ max)

---

## GitHub Actions Workflow

`.github/workflows/lamda_v2_ci.yml`が自動実行されます。

### トリガー
- `main`, `develop`ブランチへのpush
- Pull Request作成/更新
- 対象パス: `scripts/lamda_v2/**`, `tests/lamda_v2/**`

### ワークフロー
1. Python 3.11セットアップ
2. 依存関係インストール
3. **テスト実行** (38/38 tests)
4. A/B chord audit (オプション)
5. **品質ゲート実行**
6. PR結果コメント

---

## ローカル実行

### 全テスト実行
```bash
pytest tests/lamda_v2/ -v
```

### 品質ゲート実行
```bash
# デフォルト閾値
python scripts/ci/metrics_gate.py \
  --ab-csv analysis/ab_chords_audit.csv \
  --stage2-json-dir output/stage2/json

# カスタム閾値
MATCH_RATE_MIN=0.90 CONTROLS_INTEGRITY_MIN=1.0 \
python scripts/ci/metrics_gate.py \
  --ab-csv analysis/ab_chords_audit.csv \
  --stage2-json-dir output/stage2/json
```

---

## 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `MATCH_RATE_MIN` | 0.85 | Match rate最小閾値 |
| `CONTROLS_INTEGRITY_MIN` | 0.99 | Controls integrity最小閾値 |

---

## トラブルシューティング

### ゲート失敗時の対処

#### Match Rate < 0.85
1. コード認識精度を確認
2. `scripts/lamda_v2/chord_analyzer.py`のロジック確認
3. テストMIDIファイルの品質確認

#### Controls Integrity < 0.99
1. `scripts/lamda_v2/controls_analyzer.py`のバリデーション確認
2. 出力JSONの`controls`フィールド確認
3. 異常値を含むMIDIファイルを特定

### テスト失敗時
```bash
# 詳細ログ付きテスト
pytest tests/lamda_v2/ -vv --tb=long

# 特定モジュールのみ
pytest tests/lamda_v2/test_groove_analyzer.py -v
```

---

## Badge

[![LAMDA v2 CI](https://github.com/kinoshitayoshihiro/composer4/actions/workflows/lamda_v2_ci.yml/badge.svg)](https://github.com/kinoshitayoshihiro/composer4/actions/workflows/lamda_v2_ci.yml)

---

## 関連ファイル

```
.github/workflows/
  └── lamda_v2_ci.yml       # GitHub Actions設定

scripts/ci/
  └── metrics_gate.py       # 品質ゲートスクリプト

tests/lamda_v2/
  ├── test_groove_analyzer.py
  ├── test_controls_analyzer.py
  └── ... (38 tests total)

scripts/lamda_v2/
  ├── groove_analyzer.py
  ├── controls_analyzer.py
  └── stage2_extractor.py
```

---

## 次のステップ

1. **初回実行**: PRを作成してCIが動作することを確認
2. **データ準備**: A/B auditとStage2 JSONを準備
3. **閾値調整**: プロジェクトに合わせて環境変数を調整
4. **監視**: CI結果を定期的にチェック

---

**更新履歴**
- 2025-10-23: CI/CDパイプライン初版リリース (v2.6+)
