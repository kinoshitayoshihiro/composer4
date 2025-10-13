# Generator Stage3 Implementation Plan

## 🎯 目標
drumgenerator を Stage3 (生成→評価→CI) パイプラインに統合

---

## ✅ Phase 1.1: ベースライン完成 (完了)

### 成果物
- ✅ `scripts/run_stage3_drum_eval.sh` - ワンボタン評価スクリプト
- ✅ `scripts/generate_drum_samples.py` - サンプル生成スクリプト
- ✅ プレースホルダーMIDI生成で**パイプライン疎通確認**完了

### 実行ログ
```bash
./scripts/run_stage3_drum_eval.sh --n-samples 3 --length-bars 8 --seed 42
# ✅ 3 MIDI files generated
# ✅ metadata.json created
# ⚠️  Stage2/A/B report は未実装 (expected)
```

---

## 🔄 Phase 1.2: DrumGenerator統合 (次のタスク)

### 問題
`scripts/generate_drum_samples.py` は現在プレースホルダーMIDIを生成:
```python
# generator/drum_generator.py の初期化に失敗
except Exception as e:
    logger.error(f"Failed to initialize DrumGenerator: {e}")
    # → プレースホルダーMIDI生成に fallback
```

### 原因分析
`generator/drum_generator.py` の `DrumGenerator.__init__()` は以下を要求:
- `global_settings`: tempo, time_signature など
- `main_cfg`: style, density, swing など
- **パターンライブラリ**: `data/drum_patterns/*.yaml` などのファイル

### 解決策A: 最小限のconfig提供 (推奨)
```python
# scripts/generate_drum_samples.py 修正
drum_gen = DrumGenerator(
    part_name="drum",
    global_settings={
        "tempo": tempo_bpm,
        "time_signature": "4/4",
        "patterns_dir": "data/drum_patterns",  # 追加
    },
    main_cfg={
        "style": style,
        "density": density,
        "swing": swing,
    }
)
```

**必要な調査**:
1. `generator/drum_generator.py` の `__init__()` を読み、必須パラメータを特定
2. `data/drum_patterns/` の構造を確認
3. 最小限の設定で動作するパターンを1つ用意 (`pop_straight.yaml` など)

**所要時間**: 2-3時間

### 解決策B: ラッパークラス作成
```python
# generator/drum_generator_standalone.py (新規作成)
class StandaloneDrumGenerator:
    """DrumGenerator の最小限ラッパー"""
    
    def __init__(self, tempo: int = 120, style: str = "pop_straight"):
        # デフォルト設定を内部で構築
        self.drum_gen = DrumGenerator(
            part_name="drum",
            global_settings=self._default_global_settings(tempo),
            main_cfg=self._default_main_cfg(style),
        )
    
    def generate(self, length_bars: int, seed: int) -> stream.Part:
        section_data = self._build_section_data(length_bars)
        return self.drum_gen.compose(section_data=section_data)
```

**所要時間**: 3-4時間

---

## 📝 Phase 1.3: Stage2評価連携 (2-3時間)

### 問題
```bash
quick_eval_stage2.py: error: unrecognized arguments: --input-dir
```

### 解決策
`scripts/quick_eval_stage2.py` の引数を確認し、既存の Stage2 評価スクリプトと整合:
```bash
# 現状の引数形式を確認
python scripts/quick_eval_stage2.py --help

# 期待される形式に修正
python scripts/quick_eval_stage2.py \
  --midi-dir "$OUT/generated" \  # --input-dir ではない?
  --output-dir "$OUT/stage2"
```

**または**: 新規に `scripts/batch_eval_stage2.py` を作成

---

## 📊 Phase 1.4: A/Bレポート生成 (3-4時間)

### 現状
```bash
ab_summarize_v2.py: error: the following arguments are required: --a, --b, --out
```

### 必要な機能
1. **生成パターンの評価メトリクス抽出**:
   - Bar violation rate (小節線のずれ)
   - Hat grid alignment (ハイハットのグリッド精度)
   - Kick/Snare pattern coherence
   
2. **ベースラインとの比較**:
   - Option A: 過去の生成結果と比較
   - Option B: ゴールドスタンダード (手動作成パターン) と比較

3. **レポート生成**:
   ```markdown
   # Stage3 Drum Evaluation Report
   
   ## Config
   - Tempo: 120 BPM
   - Style: pop_straight
   - Samples: 10
   
   ## Metrics
   | Metric | Current | Baseline | Status |
   |--------|---------|----------|--------|
   | Bar violation | 0.0% | 0.0% | ✅ PASS |
   | Hat grid | 0.87 | 0.85 | ✅ PASS |
   | Pass rate | 0.70 | 0.65 | ✅ PASS |
   ```

---

## 🧪 Phase 1.5: CI統合 (1-2時間)

### 受け入れ判定スクリプト
```python
# scripts/check_acceptance.py
def check_acceptance(report_path: Path, thresholds: dict) -> bool:
    """Check if generated samples meet acceptance criteria."""
    report = parse_report(report_path)
    
    checks = {
        "bar_violations": report["bar_violations"] <= thresholds["bar_violations"],
        "hat_grid": report["hat_grid"] >= thresholds["hat_grid"],
        "pass_rate": report["pass_rate"] >= thresholds["pass_rate"],
    }
    
    for name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    return all(checks.values())
```

### GitHub Actions ワークフロー
```yaml
# .github/workflows/stage3_drum_validation.yml
name: Stage3 Drum Validation

on:
  push:
    paths:
      - 'generator/drum_generator.py'
      - 'data/drum_patterns/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11
      - name: Run Stage3 evaluation
        run: |
          ./scripts/run_stage3_drum_eval.sh \
            --n-samples 10 \
            --style pop_straight \
            --seed 42
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: stage3-results
          path: output/drumgen_eval_*/
```

---

## 🎯 最終的な成果物

### ユーザーストーリー
```bash
# 開発者がdrumgeneratorを変更
git commit -m "feat: improve fill pattern diversity"

# CI が自動実行
# → 10サンプル生成
# → Stage2評価
# → A/Bレポート生成
# → 受け入れ判定 (bar_violations=0, hat_grid≥0.85, pass_rate≥0.65)
# ✅ PASS → マージ可能
# ❌ FAIL → レビューが必要
```

### 手動テスト
```bash
# 新しいスタイルをテスト
./scripts/run_stage3_drum_eval.sh --style shuffle --tempo 140

# 長尺パターンをテスト
./scripts/run_stage3_drum_eval.sh --length-bars 128 --n-samples 5

# 異なるdensityを比較
./scripts/run_stage3_drum_eval.sh --density low --seed 1
./scripts/run_stage3_drum_eval.sh --density high --seed 1
```

---

## 📅 スケジュール

| Phase | タスク | 所要時間 | 完了 |
|-------|--------|---------|------|
| 1.1 | ベースライン (スクリプト作成) | 2時間 | ✅ |
| 1.2 | DrumGenerator統合 | 3-4時間 | 🔄 次 |
| 1.3 | Stage2評価連携 | 2-3時間 | 📋 |
| 1.4 | A/Bレポート生成 | 3-4時間 | 📋 |
| 1.5 | CI統合 | 1-2時間 | 📋 |
| **合計** | | **11-15時間** | |

---

## 🚀 次のアクション

### Immediate (今すぐ)
1. `generator/drum_generator.py` の `__init__()` を読む
2. `data/drum_patterns/` の構造を確認
3. 最小限の設定で DrumGenerator を初期化する方法を特定

### コマンド
```bash
# DrumGenerator の初期化要件を確認
grep -A 50 "def __init__" generator/drum_generator.py | head -60

# パターンライブラリの構造を確認
ls -R data/drum_patterns/ 2>/dev/null || echo "パターンディレクトリが存在しません"

# 既存の使用例を検索
grep -r "DrumGenerator(" --include="*.py" | head -10
```

**実行しますか?** 🔍
