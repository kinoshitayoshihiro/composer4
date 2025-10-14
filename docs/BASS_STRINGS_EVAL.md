# Bass & Strings Evaluation Scripts

Phase 4.6で追加されたBass/Strings評価スクリプトのドキュメント。

## 概要

- `scripts/eval_bass.py`: ベースライン評価
- `scripts/eval_strings.py`: ストリングス評価
- 両スクリプトとも`config/quality_gates.yaml`の閾値定義に準拠

## eval_bass.py

### 評価指標

| 指標名 | 説明 | 閾値 |
|--------|------|------|
| `root_or_chord_tone_rate` | ルート/和声音ヒット率 | ≥ 0.70 |
| `leap_rate` | 跳躍の割合 (3度以上) | ≤ 0.20 |
| `max_leap_semitones` | 最大跳躍幅 | ≤ 12 (1オクターブ) |
| `grid_off_std_ms` | タイミング安定度 (標準偏差) | ≤ 18 ms |
| `notes_per_bar` | 音符密度 | 4〜12 |
| `velocity_std` | ベロシティ分散 | 10〜22 |

### 使用方法

```bash
python scripts/eval_bass.py \
  --input "output/bass/*.mid" \
  --out-json "output/reports/bass_eval.json" \
  --out-csv "output/reports/bass_eval.csv" \
  --chord-root "C" \
  --mode "major"
```

### パラメータ

- `--input`: 入力MIDIファイル (globパターン可)
- `--out-json`: 出力JSON (必須)
- `--out-csv`: 出力CSV (オプション)
- `--chord-root`: 和音ルート (デフォルト: "C")
- `--mode`: スケールモード ("major" or "minor", デフォルト: "major")

### 出力フォーマット

```json
{
  "schema_version": "1.1",
  "instrument": "bass",
  "evaluation_date": null,
  "fileset_hash": "abc123...",
  "chord_root": "C",
  "mode": "major",
  "provenance": {
    "git_commit": "abc123...",
    "git_branch": "main"
  },
  "thresholds": { ... },
  "threshold_flags": ["root_or_chord_tone_rate:low"],
  "aggregated": {
    "n_files": 10,
    "n_valid": 10,
    "summary": {
      "root_or_chord_tone_rate": {
        "mean": 0.75,
        "median": 0.76,
        "std": 0.05
      }
    }
  },
  "per_file": [ ... ]
}
```

## eval_strings.py

### 評価指標

| 指標名 | 説明 | 閾値 |
|--------|------|------|
| `legato_connection_rate` | レガート連結率 (50ms以内) | ≥ 0.60 |
| `leap_rate` | 跳躍の割合 (3度以上) | ≤ 0.15 |
| `max_leap_semitones` | 最大跳躍幅 | ≤ 12 (1オクターブ) |
| `chord_spread_semitones` | 和声音の広がり | ≤ 24 (2オクターブ) |
| `velocity_std` | ベロシティ分散 | ≥ 12 |
| `bar_violation_rate` | 小節境界逸脱率 | ≤ 0.02 |

### 使用方法

```bash
python scripts/eval_strings.py \
  --input "output/strings/*.mid" \
  --out-json "output/reports/strings_eval.json" \
  --out-csv "output/reports/strings_eval.csv"
```

### パラメータ

- `--input`: 入力MIDIファイル (globパターン可)
- `--out-json`: 出力JSON (必須)
- `--out-csv`: 出力CSV (オプション)

### 出力フォーマット

```json
{
  "schema_version": "1.1",
  "instrument": "strings",
  "evaluation_date": null,
  "fileset_hash": "abc123...",
  "provenance": {
    "git_commit": "abc123...",
    "git_branch": "main"
  },
  "thresholds": { ... },
  "threshold_flags": ["legato_connection_rate:low"],
  "aggregated": {
    "n_files": 10,
    "n_valid": 10,
    "summary": {
      "legato_connection_rate": {
        "mean": 0.65,
        "median": 0.64,
        "std": 0.08
      }
    }
  },
  "per_file": [ ... ]
}
```

## CI統合

両スクリプトは`scripts/ci_quality_gate.sh`に統合済み:

```bash
# CI実行
./scripts/ci_quality_gate.sh

# または個別チェック
python scripts/quality_gate_checker.py --check bass --json output/reports/bass_eval.json
python scripts/quality_gate_checker.py --check strings --json output/reports/strings_eval.json
```

## 品質ゲート判定

### Bass

- ✅ PASS: 全6指標が閾値内
- ❌ FAIL: 1つ以上の違反あり (threshold_flags に記録)

**スタイル別オーバーライド** (from quality_gates.yaml):

- `walking`: 音符密度 8〜16, ルート音率 ≥0.60 (パッシング許容)
- `ballad`: タイミング余裕 ≤22ms

### Strings

- ✅ PASS: 全6指標が閾値内
- ❌ FAIL: 1つ以上の違反あり (threshold_flags に記録)

**スタイル別オーバーライド** (from quality_gates.yaml):

- `staccato`: レガート率 ≥0.30 (低めOK), ベロシティ分散 ≥8

## テスト実行例

### Bass評価テスト

```bash
$ python scripts/eval_bass.py \
  --input "out/samples/bass_sample.mid" \
  --out-json "output/reports/bass_eval_test.json" \
  --chord-root "C" \
  --mode "major"

🎸 Evaluating 1 bass MIDI files...
   Chord root: C major
  ✅ bass_sample.mid

📊 Aggregated metrics:
   root_or_chord_tone_rate: 0.609 (±0.000)
   leap_rate: 0.714 (±0.000)
   max_leap_semitones: 13.000 (±0.000)
   grid_off_std_ms: 0.000 (±0.000)
   notes_per_bar: 8.000 (±0.000)
   velocity_std: 0.000 (±0.000)

🚦 Quality Gates:
   ❌ FAIL - Violations: root_or_chord_tone_rate:low, leap_rate:high, 
                         max_leap_semitones:high, velocity_std:low
```

### Strings評価テスト

```bash
$ python scripts/eval_strings.py \
  --input "output/stringsgen_B/pad_mid_90bpm_8bars/*.mid" \
  --out-json "output/reports/strings_eval_test.json"

🎻 Evaluating 2 strings MIDI files...
  ✅ strings_pad_90bpm_8bars_seed42.mid
  ✅ strings_pad_90bpm_8bars_seed43.mid

📊 Aggregated metrics:
   legato_connection_rate: 0.200 (±0.000)
   leap_rate: 1.000 (±0.000)
   max_leap_semitones: 29.000 (±0.000)
   chord_spread_semitones: 24.000 (±0.000)
   velocity_std: 2.270 (±0.255)
   bar_violation_rate: 0.066 (±0.000)

🚦 Quality Gates:
   ❌ FAIL - Violations: legato_connection_rate:low, leap_rate:high,
                         max_leap_semitones:high, velocity_std:low,
                         bar_violation_rate:high
```

## 実装詳細

### 共通機能

両スクリプトとも以下の機能を実装:

1. **Schema versioning**: `"schema_version": "1.1"`
2. **Fileset hash**: SHA1ハッシュで評価対象ファイルセットを一意に識別
3. **Threshold flags**: 違反指標を自動検出 (`["metric_name:low", ...]`)
4. **Provenance**: Git commit/branch情報を記録
5. **Exit codes**: 0=PASS, 1=FAIL (CI統合用)

### Bass特有の実装

- **Chord tone detection**: `CHORD_ROOTS` + `MAJOR_INTERVALS/MINOR_INTERVALS`
- **Grid quantization**: 16分音符グリッドベース
- **Leap detection**: 3度以上の跳躍を検出

### Strings特有の実装

- **Legato detection**: 50ms以内のノート連結を検出
- **Chord window analysis**: 同時発音ノート (50ms以内) のピッチ広がり測定
- **Range filtering**: STRINGS_RANGE_MIN (G2) 〜 STRINGS_RANGE_MAX (E6)

## 次のステップ

Phase 4.7での統合:

1. Section alignment tests 追加
2. Emotion profile mapping 統合
3. GitHub Actions workflow 更新 (.github/workflows/quality_gate.yml)
4. 既存generator実装との整合性確認

## 関連ファイル

- `config/quality_gates.yaml`: 閾値定義
- `scripts/quality_gate_checker.py`: 品質ゲートチェッカー (CLI)
- `scripts/ci_quality_gate.sh`: CI統合スクリプト
- `.github/workflows/quality_gate.yml`: GitHub Actions workflow
- `docs/INSTRUMENT_COMPLETION_STATUS.md`: 楽器別完成度レポート
- `docs/PHASE_4_ROADMAP.md`: Phase 4ロードマップ

## 参考実装

- `scripts/eval_piano_external.py`: Piano評価 (Phase 4.3)
- `scripts/eval_drum_batch_stratified.py`: Drum評価 (Phase 4.6統合済み)
