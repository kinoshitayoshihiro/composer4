# Guard Retry Accept

## 概要

`guard_retry_accept.py` はリトライ処方の**最終ゲート**です。リトライ適用前後のメトリクスを比較し、control blockの `min_delta` 条件を満たすかを判定します。

## 使い方

```bash
# 基本形
python scripts/guard_retry_accept.py \
  --before outputs/metrics_before.jsonl \
  --after outputs/metrics_after.jsonl \
  --out outputs/guard_decisions.jsonl
```

## min_delta フォーマット

### 辞書形式（推奨）

```yaml
control:
  min_delta:
    score_total: 1.0         # 総合スコア +1.0以上
    axes_raw:
      velocity: 0.05         # velocity軸 +0.05以上
      structure: 0.04        # structure軸 +0.04以上
```

### レガシー float 形式

```yaml
control:
  min_delta: 2.0  # スコア +2.0以上（後方互換）
```

## 出力形式

```jsonl
{
  "loop_id": "loop_001",
  "accepted_by_guard": true,
  "guard_meta": {
    "delta_total": 2.5,
    "deltas_axes": {"velocity": 0.06, "structure": 0.05},
    "ok_total": true,
    "ok_axes": true,
    "criteria": {"score_total": 1.0, "axes_raw": {"velocity": 0.05}}
  },
  "_retry_state": {...},
  "_retry_control": {...}
}
```

## 運用フロー

```bash
# 1) 初回評価
python scripts/lamda_stage2_run.py --out outputs/run1/

# 2) リトライ計画（control付きプリセット）
python scripts/retry_apply.py outputs/run1/metrics_score.jsonl \
  --presets configs/lamda/retry_presets.yaml \
  --out outputs/retry_plan.jsonl

# 3) リトライ適用 + 再評価
python scripts/lamda_stage2_run.py \
  --retry-from outputs/retry_plan.jsonl \
  --out outputs/run2/

# 4) ガード判定
python scripts/guard_retry_accept.py \
  --before outputs/run1/metrics_score.jsonl \
  --after outputs/run2/metrics_score.jsonl \
  --out outputs/guard_decisions.jsonl

# 5) A/B比較レポート
python scripts/ab_summarize_v2.py \
  --a outputs/run1 --b outputs/run2 \
  --out reports/ab_summary.md
```

## テスト

```bash
pytest tests/scripts/test_guard_retry_accept.py -v
```

## 関連ファイル

- `scripts/retry_apply.py` - control block解釈とリトライ計画
- `configs/lamda/retry_presets.yaml` - priority + min_delta設定
- `scripts/ab_summarize_v2.py` - 層別A/B比較レポート
