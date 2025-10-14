# Piano Transformer External Benchmark (Phase 4.3)

## Overview

Piano Transformer の外部ベンチマーク評価システムです。MAESTRO データセットのサブセットを使用して、モデルの品質を客観的に測定します。

## Features

- **外部ベンチマーク評価**: MAESTRO サブセット (10-20 samples)
- **トレンド追跡**: 履歴 JSONL による時系列データ保存
- **可視化**: Markdown レポート + ASCII チャート
- **Nightly CI 統合**: 自動評価・レポート生成

## Metrics

### Core Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Chord Tone Rate** | Harmonic consistency (pitch class diversity) | >0.70 |
| **Hand Separation** | Left/right hand independence (pitch range) | >0.60 |
| **Velocity Std** | Dynamic expression diversity | 15-25 |
| **Bar Violation Rate** | Notes spanning multiple bars | <0.02 |
| **Notes Per Bar** | Average density | 8-16 |

### Why These Metrics?

- **Chord Tone Rate**: 和音構成音の一致率の代理指標（現在はピッチクラス多様性で近似）
  - ⚠️ **現在の実装**: ピッチクラス（0-11）の多様性を7音階で正規化（簡易版）
  - 🎯 **将来計画**: music21統合により"真の和音一致率"に置換予定（Phase 4.4+）
- **Hand Separation**: 左右手の独立性の代理指標（音域の広がりで近似）
- **Velocity Std**: ダイナミクスの表現力
- **Bar Violation Rate**: 小節境界の尊重度（構造の理解度）
- **Notes Per Bar**: 密度の適切性

> **Naming Note**: "Chord Tone Rate" という名称は将来の完全実装を見据えたものです。現時点では簡易版（pitch class diversity）ですが、メトリクス名は据え置きのまま、将来的に実装を高度化します（互換性維持）。

## Setup

### 1. MAESTRO サブセットの準備

```bash
# Option A: Manual download (recommended for CI)
mkdir -p data/maestro_subset
# Download 10-20 MIDI files from MAESTRO v3.0.0
# https://magenta.tensorflow.org/datasets/maestro

# Option B: Full MAESTRO dataset (for comprehensive evaluation)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip -d data/
# Sample 20 files
find data/maestro-v3.0.0 -name "*.midi" | head -20 | xargs -I {} cp {} data/maestro_subset/
```

### 2. Dependencies

すでに `requirements.txt` に含まれています:
- `pretty_midi`: MIDI I/O
- Python 3.11+

## Usage

### Single Evaluation

```bash
python scripts/eval_piano_external.py \
  --maestro-dir data/maestro_subset \
  --out-json output/reports/piano_external_bench.json \
  --n-samples 10 \
  --seed 42
```

### Nightly CI Integration

```bash
# Run evaluation with history tracking
bash scripts/run_piano_external_bench.sh

# Environment variables (optional)
export MAESTRO_DIR="data/maestro_subset"
export N_SAMPLES=10
bash scripts/run_piano_external_bench.sh
```

### Trend Visualization

```bash
# After running evaluations multiple times
python scripts/visualize_piano_trends.py \
  --history output/reports/piano_external_bench_history.jsonl \
  --out-dir output/reports/trends
```

## Output Structure

```
output/reports/
├── piano_external_bench_20251014_120000.json  # Timestamped result
├── piano_external_bench_latest.json           # Symlink to latest
├── piano_external_bench_history.jsonl         # Time series data
└── trends/
    └── piano_external_trends.md               # Markdown report
```

### JSON Output Schema

```json
{
  "benchmark": "maestro_subset",
  "n_samples": 10,
  "seed": 42,
  "summary": {
    "total_samples": 10,
    "valid_samples": 10,
    "chord_tone_rate": {
      "mean": 0.7234,
      "median": 0.7150
    },
    "hand_separation": {
      "mean": 0.6543,
      "median": 0.6500
    },
    "velocity_std": {
      "mean": 18.45,
      "median": 17.80
    },
    "bar_violation_rate": {
      "mean": 0.0123,
      "median": 0.0100
    },
    "notes_per_bar": {
      "mean": 12.34,
      "median": 11.50
    }
  },
  "per_file": [
    {
      "file": "MIDI-Unprocessed_01_R1_2004_01-05_ORIG_MID--AUDIO_01_R1_2004_05_Track05_wav.midi",
      "valid": true,
      "tempo": 120.5,
      "bars": 64,
      "chord_tone_rate": 0.7143,
      "hand_separation": 0.6458,
      ...
    }
  ]
}
```

## CI Integration Example

### .github/workflows/nightly-piano-bench.yml (例)

```yaml
name: Nightly Piano External Benchmark

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Download MAESTRO subset
        run: |
          mkdir -p data/maestro_subset
          # Add your MAESTRO download logic here
      
      - name: Run external benchmark
        run: bash scripts/run_piano_external_bench.sh
      
      - name: Generate trend report
        run: |
          python scripts/visualize_piano_trends.py \
            --history output/reports/piano_external_bench_history.jsonl \
            --out-dir output/reports/trends
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: piano-external-bench
          path: output/reports/
```

## Thresholds & Alerts

推奨しきい値（アラート条件）:

```python
THRESHOLDS = {
    "chord_tone_rate": {"min": 0.70, "direction": "higher_is_better"},
    "hand_separation": {"min": 0.60, "direction": "higher_is_better"},
    "velocity_std": {"min": 15.0, "max": 25.0, "direction": "within_range"},
    "bar_violation_rate": {"max": 0.02, "direction": "lower_is_better"},
    "notes_per_bar": {"min": 8.0, "max": 16.0, "direction": "within_range"},
}
```

## Design Principles

1. **最小依存**: `pretty_midi` のみ使用、外部ツール不要
2. **互換性**: 既存の `eval_drum_batch_stratified.py` のロジックを再利用
3. **決定論**: seed 固定で再現可能な評価
   - **Deterministic sampling**: SHA1ソートによるglob順序非依存 + seedベースshuffle
   - 同一seedで完全に再現可能なサンプル選択
4. **軽量**: サブセット (10-20 samples) で高速実行
5. **拡張性**: JSONL 形式で時系列データ蓄積
6. **監査性**: Provenance情報（git commit, branch, maestro_dir）をJSON出力に記録

## Troubleshooting

### MAESTRO directory not found

```bash
# Check directory structure
ls -la data/maestro_subset/

# Expected: At least 10 MIDI files
# If empty, download files manually
```

### No valid samples

- MIDI ファイルが破損している可能性
- Piano トラック（非ドラム）が存在しない
- `per_file` で `"valid": false` のエントリを確認

### Metrics seem off

- **Chord Tone Rate**: ピッチクラス多様性の近似なので、完全な和音分析ではありません
- **Hand Separation**: 音域の広がりの近似なので、実際の左右手分離とは異なります
- より精密な評価が必要な場合は music21 等の統合を検討してください

## Future Enhancements

### High Priority
- [ ] **Chord Tone Rate 高度化**: music21統合による真の和音一致率計算
  - Roman numeral analysis での和音推定
  - 各音符が和音構成音かを判定
  - コード進行の妥当性評価
- [ ] **PNG チャート生成**: matplotlib による可視化（Markdown 埋め込み対応）
- [ ] **Threshold Flags**: 逸脱方向の記録（例: `["velocity_std:low", "bar_violation_rate:high"]`）

### Medium Priority
- [ ] ペダル利用率の評価
- [ ] フレーズ構造の評価（句読点の適切性）
- [ ] リズムパターンの多様性

### Low Priority
- [ ] ASAP データセット対応（パフォーマンス解釈の評価）
- [ ] 欠損時の取り扱い明確化（`null` metrics + `missing_*` カウント）

## Related

- Phase 4.1: Training robustness improvements
- Phase 4.2: Data quality & generation improvements
- Phase 4.5: A/B evaluation mini-patch
- Phase 4.2-polish: Stratified split stability

---

**Phase 4.3 Complete** ✅
