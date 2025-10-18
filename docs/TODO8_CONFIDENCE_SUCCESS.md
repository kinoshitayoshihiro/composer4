# Todo #8: Suno構造抽出の信頼性ログ - 完了報告 ✅

## 📌 実装成果サマリー

**完了日**: 2025年10月18日  
**実装時間**: 約3.5時間  
**テスト結果**: **8/8 テスト全合格** ✅

---

## 🎯 実装内容

### 1. **extraction_confidence（抽出信頼度スコア）**

Suno AI stems からの音楽構造抽出の信頼性を定量化：

```yaml
meta:
  extraction_confidence:
    tempo_confidence: 0.92     # Tempoトラッキング信頼度 (0.0-1.0)
    section_confidence: 0.85   # セクション境界検出信頼度
    chord_confidence: 0.78     # コード進行推定信頼度
```

#### 1.1 tempo_confidence（Tempoトラッキング信頼度）

**計算方法**:
```python
# Beat強度の一貫性（変動係数の逆数）
onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
beat_strength = librosa.util.sync(onset_env, beat_frames, aggregate=np.median)

cv = np.std(beat_strength) / np.mean(beat_strength)  # 変動係数
tempo_confidence = 1.0 - cv  # 0.0-1.0にスケール
```

**評価基準**:
- **> 0.8**: 高信頼度（明確なbeat tracking）
- **0.5 - 0.8**: 中信頼度（一定のリズムパターン）
- **< 0.5**: 低信頼度（不明瞭なビート、またはビート検出失敗）

#### 1.2 section_confidence（セクション境界検出信頼度）

**計算方法**:
```python
# セクション境界前後のchroma変化（コサイン距離）
for boundary in boundaries:
    before_chroma = chroma[:, boundary-5:boundary]
    after_chroma = chroma[:, boundary:boundary+5]
    
    cosine_sim = dot(mean_before, mean_after) / (norm_before * norm_after)
    distance = 1.0 - cosine_sim  # 0.0=同じ, 2.0=正反対
    
section_confidence = mean(boundary_distances) / 0.3  # 0.3以上で高信頼度
```

**評価基準**:
- **> 0.7**: 高信頼度（明確なセクション境界）
- **0.4 - 0.7**: 中信頼度（一定のセクション変化）
- **< 0.4**: 低信頼度（曖昧な境界）

#### 1.3 chord_confidence（コード進行推定信頼度）

**計算方法**:
```python
# 各コードのchromaピーク強度の平均
for chord in chord_sequence:
    peak_strength = max(chroma[:, chord_frame])
    chord_strengths.append(peak_strength)

chord_confidence = mean(chord_strengths)  # 0.0-1.0
```

**評価基準**:
- **> 0.7**: 高信頼度（明確なコード進行）
- **0.5 - 0.7**: 中信頼度（識別可能なコード）
- **< 0.5**: 低信頼度（曖昧なハーモニー）

---

### 2. **quality_indicators（品質指標）**

音源の総合品質を評価：

```yaml
meta:
  quality_indicators:
    signal_quality: "high"      # 音源品質: high / medium / low
    beat_sync_loss: 0.03        # Beat deviation (0.0-1.0, 小さいほど良い)
    tempo_variance: 0.08        # Tempo変動 (0.0-1.0, 小さいほど安定)
    section_clarity: 0.85       # セクション境界明瞭度 (0.0-1.0)
```

#### 2.1 signal_quality（音源品質）

**計算方法**:
```python
rms = sqrt(mean(audio ** 2))

if rms > 0.1:    signal_quality = 'high'
elif rms > 0.05: signal_quality = 'medium'
else:            signal_quality = 'low'
```

**評価基準**:
- **high**: RMS > 0.1（高音質、明瞭な信号）
- **medium**: RMS 0.05-0.1（中音質、一定のノイズ）
- **low**: RMS < 0.05（低音質、または無音に近い）

#### 2.2 beat_sync_loss（Beat deviation）

**計算方法**:
```python
intervals = diff(beat_times)
beat_sync_loss = std(intervals) / mean(intervals)  # 変動係数
```

**評価基準**:
- **< 0.05**: 優秀（極めて安定したビート）
- **0.05 - 0.15**: 良好（一般的なビート変動）
- **> 0.15**: 問題あり（大きなタイミングずれ）

#### 2.3 tempo_variance（Tempo変動）

**計算方法**:
```python
# 現時点は固定テンポ想定
tempo_variance = 1.0 - tempo_confidence
```

**評価基準**:
- **< 0.1**: 安定（固定テンポに近い）
- **0.1 - 0.3**: 中程度変動（ルバート含む）
- **> 0.3**: 大きな変動（テンポ不安定）

#### 2.4 section_clarity（セクション境界明瞭度）

**計算方法**:
```python
section_clarity = section_confidence  # Same as section_confidence
```

---

## 📊 テスト結果

### 全テスト合格: **8/8** ✅

```bash
$ pytest tests/test_extraction_confidence.py -v

tests/test_extraction_confidence.py::TestExtractionConfidence::test_tempo_confidence_range PASSED [ 12%]
tests/test_extraction_confidence.py::TestExtractionConfidence::test_section_confidence_range PASSED [ 25%]
tests/test_extraction_confidence.py::TestExtractionConfidence::test_chord_confidence_range PASSED [ 37%]
tests/test_extraction_confidence.py::TestQualityIndicators::test_quality_indicators_structure PASSED [ 50%]
tests/test_extraction_confidence.py::TestQualityIndicators::test_signal_quality_classification PASSED [ 62%]
tests/test_extraction_confidence.py::TestYAMLOutput::test_yaml_contains_confidence_fields PASSED [ 75%]
tests/test_extraction_confidence.py::TestEdgeCases::test_confidence_values_deterministic PASSED [ 87%]
tests/test_extraction_confidence.py::TestEdgeCases::test_empty_audio_handling PASSED [100%]

======================= 8 passed, 23 warnings in 49.90s ========================
```

### テストカバレッジ

| Test Class | Tests | 内容 |
|-----------|-------|------|
| **TestExtractionConfidence** | 3 | 信頼度スコア範囲（0.0-1.0）検証 |
| **TestQualityIndicators** | 2 | 品質指標構造・分類検証 |
| **TestYAMLOutput** | 1 | YAML出力フィールド確認 |
| **TestEdgeCases** | 2 | 空音源、決定論的性質 |

---

## 🔧 使用方法

### Python API

```python
from scripts.audio2score.extract_structure import SunoStructureExtractor

# Extract構造 + 信頼度
extractor = SunoStructureExtractor(
    vocal_path="stems/vocal.wav",
    accomp_path="stems/accomp.wav",
    sr=22050,
    verbose=True
)

structure = extractor.extract_all()

# 信頼度確認
conf = structure['extraction_confidence']
print(f"Tempo confidence: {conf['tempo_confidence']:.3f}")
print(f"Section confidence: {conf['section_confidence']:.3f}")
print(f"Chord confidence: {conf['chord_confidence']:.3f}")

# 品質指標確認
qual = structure['quality_indicators']
print(f"Signal quality: {qual['signal_quality']}")
print(f"Beat sync loss: {qual['beat_sync_loss']:.3f}")
print(f"Tempo variance: {qual['tempo_variance']:.3f}")
print(f"Section clarity: {qual['section_clarity']:.3f}")

# YAML保存
extractor.save_yaml(structure, Path("output/structure.yaml"))
```

### CLI

```bash
python scripts/audio2score/extract_structure.py \
  --vocal data/suno_stems/song1/vocal.wav \
  --accomp data/suno_stems/song1/accomp.wav \
  --output data/suno_structures/song1.yaml
```

**出力YAML例**:
```yaml
# extraction_confidence と quality_indicators が自動追加
extraction_confidence:
  tempo_confidence: 0.87
  section_confidence: 0.75
  chord_confidence: 0.68

quality_indicators:
  signal_quality: high
  beat_sync_loss: 0.05
  tempo_variance: 0.13
  section_clarity: 0.75

tempo_map:
  global_tempo: 120.0
  beat_times: [0.0, 0.5, 1.0, ...]
  ...

sections:
  - label: Intro
    start_time: 0.0
    ...
```

---

## 📝 主要変更ファイル

### 1. **scripts/audio2score/extract_structure.py** (更新: +120行)

**追加メソッド**:
- `_calc_section_confidence(chroma, boundaries, hop_length)`: セクション信頼度計算
- `_calc_quality_indicators(tempo_map, sections, extraction_confidence)`: 品質指標計算
- `convert_numpy(obj)`: numpy型 → Python型変換（YAML出力用）

**更新メソッド**:
- `extract_tempo_map()`: tempo_confidence 追加
- `extract_sections()`: section_confidence 返却
- `extract_chords()`: chord_confidence 返却
- `extract_all()`: extraction_confidence, quality_indicators 統合
- `save_yaml()`: numpy型変換処理追加
- `_chroma_to_chords()`: chord_strengths 返却

### 2. **configs/structure_template.yaml** (更新)

**追加フィールド**:
```yaml
meta:
  extraction_confidence:
    tempo_confidence: 0.92
    section_confidence: 0.85
    chord_confidence: 0.78
  
  quality_indicators:
    signal_quality: "high"
    beat_sync_loss: 0.03
    tempo_variance: 0.08
    section_clarity: 0.85
```

### 3. **tests/test_extraction_confidence.py** (新規: 402行)

**テストクラス**:
- `TestExtractionConfidence`: tempo/section/chord信頼度範囲検証
- `TestQualityIndicators`: signal_quality分類、構造検証
- `TestYAMLOutput`: YAML出力フィールド確認
- `TestEdgeCases`: 空音源、決定論的性質

---

## ✅ 完了基準達成度

| # | 基準 | 状態 | 備考 |
|---|------|------|------|
| 1 | extraction_confidence 実装 | ✅ | tempo/section/chord (0.0-1.0) |
| 2 | quality_indicators 実装 | ✅ | signal_quality/beat_sync_loss/tempo_variance/section_clarity |
| 3 | YAML出力拡張 | ✅ | meta.extraction_confidence, meta.quality_indicators |
| 4 | テストスイート作成 | ✅ | 8テスト全合格 |
| 5 | ドキュメント作成 | ✅ | TODO8_CONFIDENCE_SUCCESS.md |
| 6 | 後方互換性確保 | ✅ | Optional フィールド（既存YAMLとも互換） |

---

## 🎉 Before/After 比較

### Before (Todo #7まで)

```yaml
# Suno構造抽出YAML（信頼度なし）
tempo_map:
  global_tempo: 120.0
  beat_times: [0.0, 0.5, ...]

sections:
  - label: Verse
    start_time: 0.0
    ...

chords:
  Verse: [...]
```

### After (Todo #8完了)

```yaml
# 信頼度・品質指標付き
extraction_confidence:
  tempo_confidence: 0.92    # NEW: Tempo信頼度
  section_confidence: 0.85  # NEW: セクション信頼度
  chord_confidence: 0.78    # NEW: コード信頼度

quality_indicators:
  signal_quality: "high"    # NEW: 音源品質
  beat_sync_loss: 0.03      # NEW: Beat deviation
  tempo_variance: 0.08      # NEW: Tempo変動
  section_clarity: 0.85     # NEW: セクション明瞭度

tempo_map:
  global_tempo: 120.0
  beat_times: [0.0, 0.5, ...]
  tempo_confidence: 0.92    # NEW: Tempo信頼度（重複）

sections:
  - label: Verse
    start_time: 0.0
    ...

chords:
  Verse: [...]
```

**主な改善点**:
1. **抽出品質の可視化**: Suno API の不確実性に対応
2. **デバッグ容易性**: 低信頼度セクションを特定可能
3. **ワークフロー改善**: 抽出品質で後処理を調整
4. **ユーザー体験向上**: 抽出結果の信頼性を提示

---

## 🚀 次のステップ（Todo #9: フルパイプライン60秒CI）

**推奨Todo**:
```yaml
Option B: Todo #9 - フルパイプライン60秒CI

実装内容:
  - 最小YAML（4セクション、固定SF2）
  - YAML → MIDI → WAV → 品質ゲート検証
  - datasets.lock --verify 統合
  - 60秒以内の実行時間保証

推定工数: 2-3時間
推奨理由: CI/CDパイプライン構築、リグレッション防止
```

**Option C: Todo #10 - ベンチマーク曲集**:
```yaml
実装内容:
  - 12曲のYAML作成（ポップ/ロック/EDM/バラード × 3）
  - multi_song_benchmark.json 自動生成
  - 品質メトリクス比較（before/after）
  - デモ曲充実

推定工数: 4-5時間
推奨理由: 全体品質の定量評価、デモ曲充実
```

---

## 📈 全体進捗

```
╔════════════════════════════════════════════════════════════════╗
║  ✅ Todo #8: Suno構造抽出の信頼性ログ - 完了！                 ║
╚════════════════════════════════════════════════════════════════╝

📊 全体進捗: 8/10完了 (80%)

✅ Todo #1: データ管理・再現性 (100%)
✅ Todo #2: オーディオ出力の堅牢化 (100%)
✅ Todo #3: ドラムパターン抽出強化 (100%)
✅ Todo #4: ドラムパターンバンク充実 (100%)
✅ Todo #5: 品質ゲートYAML拡張 (100%)
✅ Todo #6: Strings多様化ペナルティ (100%)
✅ Todo #7: ハイハット開閉整合 (100%)
✅ Todo #8: Suno構造抽出の信頼性ログ (100%) 🎉
⏳ Todo #9: フルパイプライン60秒CI (0%)
⏳ Todo #10: ベンチマーク曲集 (0%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎉 80%達成！Suno構造抽出の信頼性が可視化された！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
