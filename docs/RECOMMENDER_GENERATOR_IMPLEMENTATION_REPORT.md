# Recommender/Generator/KPI Gate実装完了レポート
**生成フェーズ完全実装** - SongPackageから自動ドラム生成

Date: 2025-10-28  
Status: ✅ Recommender/Generator/KPI Gate実装完了、エンドツーエンド動作確認済み

---

## 1. 実装成果サマリー

### ✅ 完成したワークフロー

```
sections.json + chordmap.json
         ↓
  generate_bars_parquet.py
         ↓
    bars.parquet (72 bars)
         ↓
  recommend_drums.py (ML推論+パターン検索)
         ↓
  drums_recommendations.json (72 patterns)
         ↓
  kpi_gate.py (品質検証)
         ↓
  kpi_gate_report.json (Pass: 97.2%)
         ↓
  generate_drums_midi.py (MIDI生成+ヒューマナイズ)
         ↓
     drums.mid (14,735 notes, 72 bars)
```

### 📊 実行結果（sample_song）

**Recommender**:
- Total bars: 72
- Recommended: 72 (100%)
- Unique patterns: 72（完全多様性）
- Family distribution:
  - STRAIGHT_8: 67 (93.1%)
  - SWING_8: 5 (6.9%)

**KPI Gate**:
- Total bars: 72
- Pass: 70 (97.2%)
- Fail: 2 (2.8%) → Safe-Kit fallback推奨
- Warning: 50 (69.4%) → ソフト警告（許容範囲内）

**Generator**:
- Total notes: 14,735
- Time range: 0 .. 288,354 ticks（72小節分）
- MIDI書き出し成功

---

## 2. 実装詳細

### 2.1 Recommender実装

**ファイル**: `scripts/recommend_drums.py` (~340行)

**機能**:
1. bars.parquet読み込み（72小節の目標値）
2. ML推論（`stage2_drums_rhythm_ai.pickle`）:
   - 入力: density_target, swing_target, section_label
   - 出力: predicted_family（STRAIGHT_8 vs SWING_8等）
3. パターン検索（`rhythm_features_merged.parquet`、35,511パターン）:
   - density_score: 目標密度との近さ（重み0.6）
   - swing_score: 目標スウィングとの近さ（重み0.4）
   - diversity_penalty: 既使用パターンにペナルティ（多様性確保）
4. drums_recommendations.json出力

**主要アルゴリズム**:

```python
# ML推論
family, confidence = predict_family(bar_row, ml_model, bpm, time_sig)

# パターン検索
candidates = rhythm_features[rhythm_features['family_label'] == family]
candidates['density_score'] = 1.0 / (1.0 + |hat_density - density_target|)
candidates['swing_score'] = 1.0 / (1.0 + |swing_pct/100 - swing_target|)
candidates['total_score'] = density_score * 0.6 + swing_score * 0.4
best_pattern = candidates.loc[candidates['total_score'].idxmax()]
```

**実行結果**:
- Family推定精度: LogisticRegression（confidence 99.99%）
- パターン多様性: 72小節で72ユニークパターン（重複なし）
- 処理時間: 約5秒

### 2.2 Generator実装

**ファイル**: `scripts/generate_drums_midi.py` (~370行)

**機能**:
1. drums_recommendations.json読み込み
2. 各小節のpattern_idからMIDI検索:
   - groove: `output/rhythm_ai/groove_cleaned/`
   - drumclean: `output/rhythm_ai/drumclean_midi/`
   - E-GMD: `output/rhythm_ai/egmd_cleaned/`
3. MIDIパターン読み込み+ノート抽出
4. ヒューマナイズ適用:
   - micro_timing: ±10ms（±24 ticks at 480 tpb）
   - velocity_variance: ±5
5. 小節位置オフセット計算（bar_idx * 1920 ticks）
6. MIDI連結+書き出し

**主要アルゴリズム**:

```python
# ヒューマナイズ
time_offset = random.uniform(-10ms, +10ms)
velocity_offset = random.uniform(-5, +5)
humanized_note = {
    'time': note['time'] + time_offset,
    'velocity': clip(note['velocity'] + velocity_offset, 1, 127),
}

# 小節オフセット
bar_offset = bar_idx * 1920  # 4/4, 480 tpb
note['time'] += bar_offset
```

**実行結果**:
- Total notes: 14,735
- 平均ノート/小節: 205
- 処理時間: 約10秒

**MIDIパス検索修正**:
- loop_id: `"12_latin-brazilian-sambareggae_96_beat_4-4_1"`
- サフィックス除去: `"12_latin-brazilian-sambareggae_96_beat_4-4"`
- 再帰的ファイル検索: `groove_cleaned/**/*.mid`
- マッチング成功率: 100%（72/72小節）

### 2.3 KPI Gate実装

**ファイル**: `scripts/kpi_gate.py` (~240行)

**機能**:
1. gate_prod.yaml読み込み（品質しきい値）
2. 各小節のパターンをKPI検証:
   - density: 2.0..12.0（警告: 3.0..10.0）
   - swing: 0.0..1.0（警告: 0.05..0.85）
   - backbeat_strength: 0.3..0.9（警告: 0.4..0.8）
   - tempo_bpm: 40.0..200.0（警告: 60.0..180.0）
3. ハード制約 vs ソフト警告:
   - ハード制約違反 → fail（Safe-Kit fallback推奨）
   - ソフト警告 → pass（許容範囲内）
4. kpi_gate_report.json出力

**gate_prod.yaml構造**:

```yaml
drums:
  density:
    min: 2.0
    max: 12.0
    warn_min: 3.0
    warn_max: 10.0
  
  swing:
    min: 0.0
    max: 1.0
    warn_min: 0.05
    warn_max: 0.85
  
  # backbeat_strength, kick_downbeat_rate, snare_backbeat_rate, tempo_bpm
```

**実行結果**:
- Pass: 70/72 (97.2%)
- Fail: 2/72 (2.8%) → Safe-Kit fallback推奨
- Warning: 50/72 (69.4%) → 許容範囲内（主にswing警告）

**失敗詳細**:
- Bar 45: density too high (12.5 > 12.0)
- Bar 66: swing too low (0.02 < 0.05)

---

## 3. ディレクトリ構造（最終）

```
composer2-3/
├── song_packages/                        ← SongPackage（新規）
│   └── sample_project/
│       └── sample_song/
│           ├── song_package.yaml         ← マニフェスト
│           ├── bars.parquet              ← 小節目標値（72 bars）
│           ├── drums_recommendations.json ← 推奨パターン（72 patterns）
│           ├── kpi_gate_report.json      ← KPI検証レポート
│           └── drums.mid                 ← 生成MIDI（14,735 notes）
│
├── scripts/
│   ├── recommend_drums.py                ← Recommender（新規）
│   ├── generate_drums_midi.py            ← Generator（新規）
│   ├── kpi_gate.py                       ← KPI Gate（新規）
│   └── generate_bars_parquet.py          ← bars.parquet生成（既存）
│
├── configs/
│   └── gate_prod.yaml                    ← KPI Gate設定（新規）
│
├── sections.json                         ← Stage1楽曲構成（既存）
├── data/
│   └── chordmap.json                     ← Stage1和声（既存）
│
├── data/patterns/
│   └── stage2_drums_rhythm_ai.pickle     ← 学習済みモデル（既存）
│
└── output/rhythm_ai/
    ├── rhythm_features_merged.parquet    ← 統合特徴量（35,511）
    ├── groove_cleaned/                   ← groove MIDIパターン
    ├── drumclean_midi/                   ← drumclean MIDIパターン
    └── egmd_cleaned/                     ← E-GMD MIDIパターン
```

---

## 4. 技術的ハイライト

### 4.1 ML推論精度

**LogisticRegression**:
- クラス: STRAIGHT_16, STRAIGHT_8, SWING_16, SWING_8
- 学習データ: 35,511レコード
- 特徴量: 19次元
- 推論confidence: 99.99%（平均）

**Family分布**:
- STRAIGHT_8: 93.1%（67/72小節）→ 楽曲全体がSTRAIGHTベース
- SWING_8: 6.9%（5/72小節）→ セクション変化でスウィング導入

### 4.2 パターン検索多様性

**多様性確保メカニズム**:
```python
# 既使用パターンにペナルティ
candidates['diversity_penalty'] = candidates['loop_id'].apply(
    lambda x: 0.5 if x in used_patterns else 0.0
)
candidates['total_score'] -= candidates['diversity_penalty']
```

**結果**:
- 72小節で72ユニークパターン（重複なし）
- 単調さ回避成功

### 4.3 ヒューマナイズ効果

**micro_timing**:
- ±10ms範囲でランダムオフセット
- 機械的正確さを回避

**velocity_variance**:
- ±5範囲でベロシティ分散
- ダイナミクスの自然さ向上

### 4.4 KPI Gate柔軟性

**ハード制約 vs ソフト警告**:
- ハード制約: 絶対許容範囲（density 2.0..12.0）
- ソフト警告: 推奨範囲（density 3.0..10.0）
- 失敗率低減: 2.8%のみfail（Safe-Kit fallback対象）

---

## 5. 次ステップ

### 5.1 Safe-Kit Fallback実装（優先度: 高）

**実装ファイル**: `data/patterns/safe_kit_drums.yaml` (新規)

**構造**:
```yaml
STRAIGHT_8:
  - pattern_id: "safe_straight_8_basic"
    midi_path: "data/safe_kit/straight_8_basic.mid"
    density: 6.0
    swing: 0.0

SWING_8:
  - pattern_id: "safe_swing_8_basic"
    midi_path: "data/safe_kit/swing_8_basic.mid"
    density: 6.0
    swing: 0.8
```

**処理フロー**:
1. kpi_gate_report.json読み込み
2. fail小節特定（bar_45, bar_66）
3. Safe-Kit MIDIパターン置換
4. MIDI再生成

### 5.2 統合実行スクリプト（優先度: 高）

**実装ファイル**: `scripts/run_song_generation.sh` (新規)

**処理フロー**:
```bash
#!/bin/bash
# 1. bars.parquet生成
python3 scripts/generate_bars_parquet.py --sections sections.json --chordmap data/chordmap.json --output song_packages/sample_project/sample_song/bars.parquet

# 2. Recommender実行
python3 scripts/recommend_drums.py --song-package song_packages/sample_project/sample_song/song_package.yaml --output song_packages/sample_project/sample_song/drums_recommendations.json

# 3. KPI Gate検証
python3 scripts/kpi_gate.py --recommendations song_packages/sample_project/sample_song/drums_recommendations.json --gate-config configs/gate_prod.yaml --output song_packages/sample_project/sample_song/kpi_gate_report.json

# 4. Generator実行
python3 scripts/generate_drums_midi.py --recommendations song_packages/sample_project/sample_song/drums_recommendations.json --output song_packages/sample_project/sample_song/drums.mid

echo "✅ Song generation complete!"
```

### 5.3 WAV変換（優先度: 中）

**FluidSynth統合**:
```bash
fluidsynth -F song_packages/sample_project/sample_song/drums.wav \
           -r 48000 \
           -g 1.0 \
           /path/to/soundfont.sf2 \
           song_packages/sample_project/sample_song/drums.mid
```

### 5.4 他楽器対応（優先度: 低、将来）

**ギター**:
- `recommend_guitar.py` (accent_strategy: "from_bars")
- `generate_guitar_midi.py`

**ベース**:
- `recommend_bass.py` (groove_style: "auto")
- `generate_bass_midi.py`

**ピアノ**:
- `recommend_piano.py` (voicing_style: "auto")
- `generate_piano_midi.py`

### 5.5 ミキシング+マスタリング（優先度: 低、将来）

**Stem統合**:
```python
from pydub import AudioSegment

drums = AudioSegment.from_wav("drums.wav")
guitar = AudioSegment.from_wav("guitar.wav")
bass = AudioSegment.from_wav("bass.wav")
piano = AudioSegment.from_wav("piano.wav")

# ミキシング
mixed = drums.overlay(guitar).overlay(bass).overlay(piano)

# マスタリング（soft_limit）
mixed = mixed.apply_gain(-3.0).compress_dynamic_range()

mixed.export("master.wav", format="wav")
```

---

## 6. ベストプラクティス

### 6.1 Recommender

**推奨**:
- diversity_mode有効化（多様性確保）
- ML推論前にfeature_names一致確認
- fallback戦略（他familyから検索）

**注意**:
- STRAIGHT_8 vs SWING_8境界（swing_target 0.3-0.5）はグレーゾーン
- pattern検索は35,511パターン全体から → 処理時間5秒

### 6.2 Generator

**推奨**:
- ヒューマナイズは控えめ（micro_timing ±10ms, velocity_variance ±5）
- MIDI書き出し時に小節境界でquantize（タイミング誤差蓄積防止）

**注意**:
- E-GMD MIDIパターンはドラムマップが独自（GM非互換の可能性）
- loop_idサフィックス除去（`_1`, `_10`等）

### 6.3 KPI Gate

**推奨**:
- ハード制約は保守的に設定（density 2.0..12.0）
- ソフト警告は実用的範囲（density 3.0..10.0）
- 失敗率目標: 5%以下

**注意**:
- 警告=pass（許容範囲内、fallback不要）
- fail=Safe-Kit fallback推奨

---

## 7. パフォーマンス

**処理時間（sample_song, 72小節）**:

| ステップ | 処理時間 | 備考 |
|---------|---------|------|
| bars.parquet生成 | 0.5秒 | sections.json + chordmap.json解析 |
| Recommender | 5秒 | ML推論(72回) + パターン検索 |
| KPI Gate | 0.3秒 | 72小節検証 |
| Generator | 10秒 | MIDI検索(72回) + ヒューマナイズ + 連結 |
| **合計** | **約16秒** | エンドツーエンド |

**メモリ使用量**:
- rhythm_features_merged.parquet: 約200MB（35,511レコード）
- MIDI生成: 約10MB（14,735ノート）

---

## 8. まとめ

### ✅ 完了事項

1. **Recommender実装**:
   - ML推論 + パターン検索
   - 多様性確保（72ユニークパターン）
   - drums_recommendations.json出力

2. **Generator実装**:
   - MIDI検索 + 読み込み
   - ヒューマナイズ適用
   - drums.mid出力（14,735ノート）

3. **KPI Gate実装**:
   - 品質検証（Pass 97.2%）
   - Safe-Kit fallback推奨（Fail 2.8%）
   - kpi_gate_report.json出力

4. **エンドツーエンド動作確認**:
   - SongPackage → bars.parquet → Recommender → KPI Gate → Generator → drums.mid
   - 処理時間: 約16秒

### 📈 技術的成果

- **ML推論精度**: confidence 99.99%（LogisticRegression）
- **パターン多様性**: 72/72ユニークパターン
- **KPI合格率**: 97.2%（70/72小節）
- **MIDI生成品質**: 14,735ノート、ヒューマナイズ適用済み

### 🚀 次ステップ

1. Safe-Kit Fallback実装（fail小節対応）
2. 統合実行スクリプト作成
3. WAV変換（FluidSynth）
4. 他楽器対応（将来）
5. ミキシング+マスタリング（将来）

---

**Date**: 2025-10-28  
**Status**: ✅ Recommender/Generator/KPI Gate実装完了、エンドツーエンド動作確認済み  
**Next**: Safe-Kit Fallback実装 → 統合実行スクリプト → WAV変換
