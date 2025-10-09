# Stage2 A/B テスト運用テンプレート

## 目的
Stage2 の設定変更（velocity_targets.yaml、retry_presets.yaml 等）による効果を定量化し、改善判断を迅速化する。

## 事前準備
- 仮想環境を有効化し、`PYTHONPATH` をプロジェクトルートに通す。
- 比較する設定ファイルを準備する（例：`configs/lamda/drums_stage2.yaml`、`configs/lamda/retry_presets.yaml`）。
- 100 ループ前後の代表サンプルを `data/drumloops_cleaned` などに用意。
- 出力先ディレクトリ `outputs/` に書き込み権限があることを確認。

## 実行手順
### 1. 基準（A）ラン
> **Note:** `OUTPUT_DIR` は `configs/lamda/drums_stage2.yaml` の `paths.output_dir` に連動する。既定値は `output/drumloops_stage2`。
```bash
# A: 既存設定
OUTPUT_DIR=output/drumloops_stage2

PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --config configs/lamda/drums_stage2.yaml \
  --print-summary

mv "${OUTPUT_DIR}/stage2_summary.json" "${OUTPUT_DIR}/stage2_summary.A.json"
mv "${OUTPUT_DIR}/metrics_score.jsonl" "${OUTPUT_DIR}/metrics_score.A.jsonl"
```

### 2. 変更後（B）ラン
設定ファイルを更新したうえで再実行する。
```bash
# B: 変更後設定
OUTPUT_DIR=output/drumloops_stage2

PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --config configs/lamda/drums_stage2.yaml \
  --print-summary

mv "${OUTPUT_DIR}/stage2_summary.json" "${OUTPUT_DIR}/stage2_summary.B.json"
mv "${OUTPUT_DIR}/metrics_score.jsonl" "${OUTPUT_DIR}/metrics_score.B.jsonl"
```

### 3. 差分集計
```bash
python scripts/analysis/ab_compare.py \
  output/drumloops_stage2/metrics_score.A.jsonl \
  output/drumloops_stage2/metrics_score.B.jsonl \
  --threshold 50 \
  --tempo-bins "95,130"

# 集計結果をメモ化したい場合は以下のように tee で保存する
python scripts/analysis/ab_compare.py \
  output/drumloops_stage2/metrics_score.A.jsonl \
  output/drumloops_stage2/metrics_score.B.jsonl \
  --threshold 50 \
  --tempo-bins "95,130" | tee output/drumloops_stage2/ab_velocity_structure.md
```

## 最低限チェックする指標
| 指標 | 目的 |
| --- | --- |
| `pass_rate` | 合格率の改善度合い |
| `score.p50 / p75 / max` | 総合スコア分布を把握 |
| `axes_raw.velocity.p50` | Velocity の中央値 |
| `axes_raw.velocity.<0.3` 件数 | 低スコアの残存数 |
| `axes_raw.structure.p50` | Structure 安定度の中央値 |
| `axes_raw.structure.<0.3` 件数 | 不安定サンプルの残存数 |
| `retry_ops` 非空率 | 処方適用率の増減 |
| `retry_ops` 内訳 | 処方チェーンの頻度解析 |

### 楽器 × テンポ帯 KPI ミニテーブル
| 楽器カテゴリ | テンポ帯 | KPI | 期待レンジ | 観察ポイント |
| --- | --- | --- | --- | --- |
| Drums | < 90 BPM | `axes_raw.velocity.p50` | 0.30 以上 | キックの腰抜けがないか、ゴーストノートの残存率 |
| Drums | 90-120 BPM | `score.p75` | +2 以上 (対 A) | スネアの粒立ちとハットのダイナミクス |
| Drums | > 120 BPM | `axes_raw.structure.<0.3` 件数 | 横這い or 減少 | フィル崩れ・クラッシュ過多の頻度 |
| Piano (if present) | < 100 BPM | `axes_raw.velocity.p50` | 0.35 以上 | レガート時のベロシティ押し潰しを監視 |
| Piano | ≥ 100 BPM | `retry_ops` 内 `humanize_pedal` 適用率 | +5pt 程度 | ペダル過多による濁り増加に注意 |

> **メモ:** テンポ帯は `--tempo-bins "95,110,130,150"` のように 95/110/130/150 の 4 分割を使うと、EDM／歌モノ／高速ロックといった帯域差が見えやすい。各カテゴリに十分なサンプル数（最低 10 件）を確保して母数依存のノイズを除去する。

### 閾値運用のヒント
- `retry_round_guard.py` の `--max-regression` と `--min-improvement` を用いて、スコア改善が 0.02 未満のケースを除外しつつ、悪化を 0.05 以内に抑制する。
- `--strict` を付与すると、`max_regression` / `min_improvement_not_met` / `score_threshold` の理由別集計を確認でき、どの条件で弾かれているかが把握しやすい。
- KPI が閾値を割り込んだ場合は、理由別カウントから優先度の高い失敗カテゴリ（例：`max_regression` 多発）を特定し、処方や閾値の見直しに繋げる。

### Retry 処方の検証メモ
- `scripts/retry_apply.py` に `--dry-run` を指定すると、実ファイルを書き換えずに適用予定のプリセットと対象サンプルを `stdout` に確認できる。
- `--explain /path/to/explain.jsonl` を追加すると、条件式の評価結果やスコア変化を JSONL で記録でき、レビュー時の添付資料として再利用できる。
- 新しいプリセットを試す際は `--round-limit` で最大ラウンド数を 1 に固定し、副作用の切り分けをしやすくする。

## レポート雛形
```
# Stage2 A/B レポート

## 概要
- 実行日: YYYY-MM-DD
- サンプル数: XXX
- 設定差分: <例> velocity_targets.yaml 高速帯 phase_adjust_db +0.2dB

## メトリクス比較
| 指標 | A | B | Δ |
| --- | --- | --- | --- |
| pass_rate | 0.55 | 0.60 | +0.05 |
| score.p50 | 42.1 | 44.8 | +2.7 |
| velocity.p50 | 0.36 | 0.41 | +0.05 |
| structure.p50 | 0.38 | 0.40 | +0.02 |

## 所感
- Velocity の底上げが耳感とも一致。高速帯の粒立ち改善を確認。
- Structure 軸の変化は軽微。さらなる微調整候補: density 上限 +1。
- retry_ops の適用率は 18% → 22%。追加処方の副作用は未検出。

## 次アクション
- normalize_dynamic_range OFFで再検証（高速帯のみ）。
- retry チェーンの 2 ラウンド目閾値を 0.05 → 0.07 へ試験。
```

## サンプルランの確認ポイント
- `output/drumloops_stage2/stage2_summary.A.json` / `.B.json` が生成されているか。
- `output/drumloops_stage2/metrics_score.*.jsonl` のレコード数がサンプル数と一致しているか。
- `output/drumloops_stage2/ab_velocity_structure.md` に主要指標が揃っているか。

## CI 連携のヒント
- 代表 100 ループのデータサブセットを用意し、Pull Request ごとに A/B 実行。
- `pass_rate`、`velocity.p50`、`structure.p50` が基準値を下回った場合は CI を fail させる。
- レポートを Artifacts に添付して、レビュー時に耳チェックの優先度を判断。

---
このテンプレートを用いて、改善アイデアの検証を定型化し、小さく速いフィードバックサイクルを維持する。