# scripts/ab_summarize.py

Stage2 の A/B 比較レポートを Markdown で自動生成します。

## 使い方

### 基本形

```bash
python scripts/ab_summarize.py \
  --a <run_A のパス> \
  --b <run_B のパス> \
  --threshold 50 \
  --out report.md
```

### 例: 実際のディレクトリから比較

```bash
# outputs ディレクトリが存在し、中に metrics_score.jsonl がある場合
python scripts/ab_summarize.py \
  --a outputs/stage2_output_baseline \
  --b outputs/stage2_output_audio_adaptive \
  --threshold 50 \
  --out outputs/ab_report.md
```

### 例: JSONL ファイルを直接指定

```bash
python scripts/ab_summarize.py \
  --a outputs/baseline/metrics_score.jsonl \
  --b outputs/adaptive/metrics_score.jsonl \
  --threshold 50 \
  --out ab_report.md
```

### 標準出力へ出力

```bash
python scripts/ab_summarize.py \
  --a outputs/run_A \
  --b outputs/run_B \
  --threshold 50 \
  --out -
```

## 出力内容

- **スコア概況**: pass_rate (推定)、p50/p90/mean、低スコア件数
- **主要軸**: timing, velocity, structure の中央値
- **音声適応**: audio 付き件数、適応発動率、ルール上位

## エラーの場合

パスが存在しない場合、明確なエラーメッセージが表示されます：

```
FileNotFoundError: Path does not exist: outputs/run_A
Please provide a valid directory containing metrics_score.jsonl or a direct path to the JSONL file.
```
