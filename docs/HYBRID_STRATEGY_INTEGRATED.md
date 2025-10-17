# Hybrid Learning Strategy: Integrated Configuration

## 統合方針

ChatGPTのmanifest駆動 + 私のハイブリッド戦略を統合

---

## データソース構成

### 1. 実データ（Primary）- 80%

**POP909:**
- v1 (melody): 909曲 → 全採用
- v2 (chords): 566曲 → **343曲不足** → manifest自動発注
- v3 (bass): 279曲 → **630曲不足** → manifest自動発注

**SLAKH:**
- Guitar: 1,422曲 (67.7%合格 = 963曲) → **不足奏法を補完**
- Bass: 584曲 (100%合格) → 十分、augmentationのみ
- Strings: 999曲 (~70%合格 = ~700曲) → **legato不足を補完**
- Drums: 557曲 (99.3%合格) → 十分

**LAMDA:**
- Drums: 51,248ループ → 極めて豊富

### 2. 合成データ（Supplementary）- 20%

**Sunoパイプライン:**
```
Suno WAV export
  ↓
WAV→MIDI ensemble変換 (Basic Pitch + Omnizart + MT3)
  ↓
投票で信頼性の高い音符のみ採用
  ↓
反復改善 (cleanup_midi × N回)
  ↓
Stage2品質ゲート (+5%厳しく)
  ↓
合格データのみ採用
```

**Emotion Humanizer:**
```
Manifest駆動生成
  ↓
GuitarGenerator/BassGenerator/StringsGenerator
  ↓
clean_midi.py (shard pickle直書き)
  ↓
Stage2品質ゲート
```

---

## Targets設定（統合版）

### configs/targets_hybrid.yaml

```yaml
# POP909完全版目標
pop909_trio_target: 500  # v1+v2+v3セットを500曲に

# 楽器別目標（実データ + 合成データ統合）
instruments:
  # === Guitar ===
  guitar:
    # 実データ: 963曲（67.7%合格）
    # 不足: ストラムパターン、フィンガーピッキング
    slow:
      total: 800
      technique:
        strum: 480       # 実: 200 → 不足: 280 (合成)
        arpeggio: 240    # 実: 500 → 十分（実データのみ）
        fingerpicking: 80 # 実: 50 → 不足: 30 (合成)
    mid:
      total: 1200
      technique:
        strum: 720       # 実: 300 → 不足: 420 (合成)
        arpeggio: 360    # 実: 600 → 十分
        mixed: 120       # 実: 63 → 不足: 57 (合成)
    fast:
      total: 600
      technique:
        strum: 360       # 不足分を合成
        arpeggio: 180
        shred: 60        # 高速フレーズ（合成）
  
  # === Bass ===
  bass:
    # 実データ: 584曲（100%合格）
    # 戦略: 実データ85% + augmentation 10% + 合成5%
    slow:
      total: 400
      technique:
        walking: 160     # 実: 100 → 不足: 60 (合成)
        sustained: 240   # 実: 300 → 十分
    mid:
      total: 1000
      technique:
        pick: 600        # 実: 500 → 不足: 100 (augmentation優先)
        slap: 100        # 実: 30 → 不足: 70 (合成)
        walking: 300     # 実: 50 → 不足: 250 (合成)
    fast:
      total: 400
      technique:
        pick: 320        # 実: 250 → 不足: 70
        slap: 80         # 実: 20 → 不足: 60 (合成)
  
  # === Strings ===
  strings:
    # 実データ: ~700曲（70%合格）
    # 不足: legato（レガート不足大）
    slow:
      total: 700
      technique:
        legato: 560      # 実: 200 → 不足: 360 (合成優先)
        staccato: 105    # 実: 400 → 過多（削減）
        spiccato: 35     # 実: 100 → 過多
    mid:
      total: 900
      technique:
        legato: 540      # 実: 250 → 不足: 290 (合成)
        staccato: 270    # 実: 350 → 過多
        tremolo: 90      # 実: 100 → 十分
    fast:
      total: 400
      technique:
        spiccato: 240    # 実: 150 → 不足: 90
        staccato: 120    # 実: 200 → 過多
        tremolo: 40      # 合成
  
  # === Piano ===
  piano:
    # 実データ: 554曲（100%合格、極めて高品質）
    # 戦略: 実データ90% + 合成10%（最小限）
    slow:
      total: 300
      technique:
        ballad: 180      # 実: 200 → 十分
        classical: 120   # 実: 50 → 不足: 70 (合成)
    mid:
      total: 500
      technique:
        pop: 350         # 実: 400 → 十分
        jazz: 150        # 実: 50 → 不足: 100 (合成)
    fast:
      total: 300
      technique:
        pop: 180         # 実: 150 → 不足: 30
        rock: 120        # 実: 50 → 不足: 70 (合成)
  
  # === Drums ===
  drums:
    # 実データ: 51,248ループ（LAMDA）+ 557曲（SLAKH）
    # 極めて豊富、合成不要
    slow:
      total: 1000       # 実データで十分
    mid:
      total: 2000       # 実データで十分
    fast:
      total: 1000       # 実データで十分

# === 合成データ品質基準 ===
synthetic_quality:
  # 実データより厳しい閾値
  guitar:
    threshold: 45.0     # 実データ: 40.0
    min_pass_rate: 0.75
  
  bass:
    threshold: 45.0     # 実データ: 40.0
    min_pass_rate: 0.80
  
  strings:
    threshold: 50.0     # 実データ: 45.0
    min_pass_rate: 0.70
  
  piano:
    threshold: 50.0     # 実データ: 45.0
    min_pass_rate: 0.85

# === データソース優先順位 ===
data_source_priority:
  # 優先順位: 実データ > Augmentation > Emotion合成 > Suno改善
  
  guitar:
    - source: "slakh_real"
      weight: 0.70
    - source: "emotion_synthetic"
      weight: 0.25
    - source: "suno_improved"
      weight: 0.05
  
  bass:
    - source: "slakh_real"
      weight: 0.85
    - source: "augmentation"
      weight: 0.10
    - source: "emotion_synthetic"
      weight: 0.05
  
  strings:
    - source: "slakh_real"
      weight: 0.70
    - source: "emotion_synthetic"
      weight: 0.25
    - source: "suno_improved"
      weight: 0.05
  
  piano:
    - source: "pop909_real"
      weight: 0.90
    - source: "emotion_synthetic"
      weight: 0.10

# === Augmentation設定 ===
augmentation:
  bass:
    timing_jitter: 15   # ±15ms
    velocity_range: 5   # ±5
    enable: true
  
  guitar:
    velocity_curve: "humanize"
    strum_variation: true
    enable: true
  
  strings:
    legato_enhance: true  # レガート接続強化
    enable: true

# === 外部ベンチマーク ===
external_benchmarks:
  piano:
    - name: "MAESTRO"
      path: "data/external/maestro"
      metrics: ["pitch_accuracy", "rhythm_consistency", "dynamics"]
  
  guitar:
    - name: "GuitarSet"
      path: "data/external/guitarset"
      metrics: ["technique_coherence", "chord_accuracy"]
  
  strings:
    - name: "ASAP"
      path: "data/external/asap"
      metrics: ["articulation_clarity", "legato_quality"]
```

---

## 実行フロー

### Step 1: 分布分析（現状把握）

```bash
# 全楽器の奏法分布を分析
for instrument in guitar bass strings piano; do
  python scripts/analyze_technique_distribution.py \
    --instrument $instrument \
    --input output/test_results/${instrument}_full.json \
    --output reports/${instrument}_technique_dist.json
done

# 統合レポート生成
python scripts/generate_distribution_report.py \
  --inputs reports/*_technique_dist.json \
  --targets configs/targets_hybrid.yaml \
  --output reports/integrated_distribution_report.json
```

### Step 2: Manifest生成（不足量推定）

```bash
# ChatGPTのスクリプトを使用
python scripts/estimate_gaps_and_emit_manifest.py \
  --pop909-root data/POP909 \
  --inventory roots:output/slakh/clean/guitar,output/slakh/clean/bass,output/slakh/clean/strings,output/pop909/clean/melody \
  --targets configs/targets_hybrid.yaml \
  --out manifests/manifest_$(date +%Y%m%d).jsonl \
  --infer-technique off
```

### Step 3: データ準備

**3a. Suno WAV→MIDI改善**
```bash
python scripts/improve_suno_midi.py \
  --input suno_exports/raw \
  --output improved/suno \
  --method both \
  --iterations 3 \
  --config configs/suno_improvement.yaml
```

**3b. 実データAugmentation**
```bash
# Bass ヒューマナイズ
python scripts/augment_midi_variations.py \
  --instrument bass \
  --input output/slakh/clean/bass \
  --output output/slakh/augmented/bass \
  --timing-jitter 15 \
  --velocity-range 5

# Guitar ベロシティ多様化
python scripts/augment_midi_variations.py \
  --instrument guitar \
  --input output/slakh/clean/guitar \
  --output output/slakh/augmented/guitar \
  --velocity-curve humanize \
  --strum-variation on
```

### Step 4: Manifest実行（合成データ生成）

```bash
# ChatGPTのスクリプトを使用
python scripts/run_manifest.py \
  --manifest manifests/manifest_$(date +%Y%m%d).jsonl \
  --pickle-out data/shards/hybrid \
  --shard-size 5000 \
  --jobs 4
```

### Step 5: データ統合

```bash
# 実データ + Augmentation + 合成データをブレンド
python scripts/merge_real_synthetic.py \
  --real output/slakh/augmented \
  --synthetic output/synthetic \
  --suno improved/suno/clean \
  --output output/training/hybrid \
  --targets configs/targets_hybrid.yaml \
  --tag-source on
```

### Step 6: 層化分割

```bash
# Train/Val/Test分割（合成はTrainのみ）
python scripts/prepare_stratified_splits.py \
  --input output/training/hybrid \
  --output output/splits/hybrid \
  --strata technique,tempo,style \
  --ratio 80:10:10 \
  --synthetic-only-train on
```

### Step 7: KPI監視

```bash
# Nightly実行
python scripts/monitor_kpis.py \
  --training-dir output/training/hybrid \
  --splits-dir output/splits/hybrid \
  --benchmarks configs/targets_hybrid.yaml \
  --output reports/kpi_dashboard.html
```

---

## 品質保証ルール

### 1. Sunoデータの厳格ゲート

```yaml
# configs/suno_improvement.yaml
quality_gate:
  stage2_threshold_boost: 5.0  # +5%
  ensemble_vote_threshold: 2   # 2エンジン以上の同意
  max_synthetic_ratio: 0.05    # 全体の5%まで
```

### 2. 反復改善の停止条件

```python
# improve_suno_midi.py内
if score >= threshold + boost:
    break  # 合格
if iterations >= max_iterations:
    reject  # 不合格として廃棄
```

### 3. タグ付け必須

```json
{
  "file": "synthetic_guitar_strum_001.mid",
  "source": "emotion_synthetic",
  "generator": "GuitarGenerator",
  "technique": "strum_downup_16th",
  "quality_score": 0.72,
  "stage2_threshold": 45.0
}

{
  "file": "suno_improved_bass_001.mid",
  "source": "suno_improved",
  "wav_source": "suno_export_123.wav",
  "ensemble_engines": ["basic_pitch", "omnizart"],
  "iterations": 3,
  "quality_score": 0.68,
  "stage2_threshold": 45.0
}
```

---

## リスク管理

### 1. セルフトレーニング崩壊回避

- 合成比率上限: 楽器別に5-30%
- Val/Testは実データのみ
- 外部ベンチマーク継続監視

### 2. Suno品質のリスク

- 複数エンジンアンサンブルで信頼性向上
- 反復改善で段階的に品質UP
- 最終的に5%以下に抑制

### 3. 奏法偏りの継続監視

- Nightly分布分析
- Manifestの自動再生成
- A/B評価で効果測定

---

## まとめ

### ✅ 統合の利点

1. **Manifest駆動**: 不足量を自動推定→自動発注
2. **Suno改善**: WAVアンサンブル→反復改善で品質向上
3. **ハイブリッド**: 実80% + 合成20%で最適バランス
4. **品質保証**: 合成データは+5%厳しいゲート
5. **冪等性**: resume対応で安全な追加実行

### 🎯 次のアクション

1. **Strings完了待ち** 🔄
2. **分布分析実行** → 不足量の定量化
3. **targets_hybrid.yaml調整** → 実際の数字に基づく
4. **Manifest生成** → 自動発注開始
5. **A/B評価** → 効果測定
