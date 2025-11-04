# Stem WAV + ChordMap ハイブリッド統合ガイド

## 📋 Executive Summary

**結論**: **Stem WAV + ChordMap ハイブリッド方式が最も有効**

| アプローチ | 強み | 弱み | KPI Pass率 |
|-----------|------|------|-----------|
| **ChordMap主導**（現行） | 和声安定・軽量・再現性高 | 密度・Fill検出が画一的 | **80.7%** |
| **Stem主導** | 実グルーヴ・抑揚反映 | 前処理重い・ノイズ影響 | - |
| **ハイブリッド**（推奨） | 和声はChordMap・リズムはStem | 実装ひと手間 | **90%台前半期待** |

---

## 🎯 現状分析

### Stem Hybrid Composer実行結果

```bash
# 実行コマンド
.venv311/bin/python scripts/stem_hybrid_composer.py \
  --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
  --analysis data/suno_ai/suno_themesong/song_001/analysis \
  --output output/hybrid_test \
  --bars 32 \
  --emotion energetic

# 結果: 正常動作確認（Exit Code: 0）
```

### ChordMap vs Stem WAVの役割分担

**ChordMap (chordmap.json)**:
- **役割**: 和声骨格（Chord Progression）
- **生成方法**: `ops/stem_harmony.py` → WAV分析 → 手動ブラッシュアップ
- **強み**: 安定・軽量・調整容易
- **課題**: リズム・ダイナミクス欠落

**Stem WAV (stemswav_001/*.wav)**:
- **役割**: 音響特徴（Onset/Rhythm/Dynamics）
- **利用可能**: 10トラック（Vocals/Drums/Bass/Guitar/Keyboard/Percussion/Strings/Synth/FX/Backing Vocals）
- **強み**: 実グルーヴ・抑揚・Fill位置反映
- **課題**: 前処理コスト・ノイズ影響

---

## 🔧 opsフォルダファイル関連性

### カテゴリ別整理

#### 1️⃣ Stem WAV分析系（音響処理）

| ファイル | 機能 | Hybridでの役割 |
|---------|------|---------------|
| `stem_harmony.py` | **WAV → Chord Recognition** | ChordMap生成（手動補正前） |
| `audio_safe.py` | **NumPy/SciPy版オーディオ処理** | librosa回避・numba問題解決 |
| `anchors_from_vocal.py` | **Vocal → Lyric Anchors抽出** | Fill/リフ同期ポイント |
| `sections_from_audio.py` | **WAV → Section分割** | Energy Curve生成 |

**統合ポイント**:
```python
# Stem特徴抽出（新規 stems_features.py）
def extract_bar_features(stem_drums, stem_mix, bars_df, sr):
    for bar in bars_df:
        seg = y[bar.start:bar.end]
        # Drums特徴
        hat_density = onset_count(seg, band=[6k,12k]) / bar_beats
        kick_peak_db = peak_in_band(seg, [30,120])
        snare_backbeat = energy_at_beats(seg, beats=[2,4], band=[180,4k])
        # Mix特徴
        loudness = rms(seg)
        yield {bar_index, hat_density, kick_peak_db, snare_backbeat, loudness}
```

#### 2️⃣ ChordMap処理系（構造情報）

| ファイル | 機能 | Hybridでの役割 |
|---------|------|---------------|
| `normalize_chordmap_format.py` | **ChordMap統一スキーマ** | 形式正規化 |
| `chordmap_expand_bars.py` | **小節単位展開** | 粗い→細かい変換 |
| `scale_modes.py` | **Key/Mode → PCマスク** | Chord制約生成 |

**統合ポイント**:
```python
# ChordMap + Stem特徴の融合
def blend_features(chordmap, stem_features, blend_alpha=0.5):
    for bar in bars:
        # ChordMap: 和声骨格
        chord = chordmap[bar]
        # Stem: 密度・ダイナミクス
        density = stem_features[bar]['hat_density']
        loudness = stem_features[bar]['loudness']
        # ブレンド
        target_density = max(bars.parquet[bar].target, density * boost_factor)
        energy_scale = (1-blend_alpha) * bars.energy + blend_alpha * loudness
```

#### 3️⃣ Section/Tempo処理系

| ファイル | 機能 | Hybridでの役割 |
|---------|------|---------------|
| `finalize_sections.py` | **Sections仕上げ** | Tempo合議・命名正規化 |
| `enrich_anchors.py` | **Anchors連携** | Section/time_ql付与 |
| `anchors_to_midi.py` | **Chorus MIDI生成** | Vocal Sync参照 |

**統合ポイント**:
```python
# Fill推奨バー検出
def detect_fill_bars(anchors, stem_drums, sections):
    fill_bars = []
    for anchor in anchors:
        if anchor['class'] == 'stress':  # 強勢アンカー
            bar = anchor['bar']
            # Drums Stemのピーク確認
            if stem_drums.peak_at_bar(bar) > threshold:
                fill_bars.append(bar)
    
    # セクション境界も追加
    for i, sec in enumerate(sections[:-1]):
        fill_bars.append(sec['bar'] + sec['length'] - 1)
    
    return fill_bars
```

#### 4️⃣ Stage2/Plan生成系

| ファイル | 機能 | Hybridでの役割 |
|---------|------|---------------|
| `stage2_batch_export.py` | **Stage2 MIDI一括生成** | Plan → MIDI変換 |
| `cache_utils.py` | **キャッシュユーティリティ** | Stem特徴キャッシュ |

---

## 🧠 mlフォルダ統合戦略

### 即戦力ファイル（小パッチで統合可）

#### ✅ Pattern Recommender系

| ファイル | 機能 | 統合方法 |
|---------|------|---------|
| `pattern_recommender.py` | **汎用パターン推薦** | Top-K再スコア層 |
| `drum_pattern_recommender.py` | **Drums専用推薦** | ML推論統合 |
| `bass_pattern_recommender.py` | **Bass専用推薦** | ヒューリスティク移植 |
| `guitar_pattern_recommender.py` | **Guitar専用推薦** | ヒューリスティク移植 |
| `piano_pattern_recommender.py` | **Piano専用推薦** | ヒューリスティク移植 |

**統合実装例**:
```python
# scripts/pattern_matcher.py への追加
if cfg.ml_rescore.enabled:
    from ml.pattern_recommender import PatternRecommender
    
    model = PatternRecommender(
        role="drums",
        pickle_path=cfg.ml_rescore.model_path
    )
    
    # Top-K候補の再スコア
    for i, candidate in enumerate(candidates):
        ml_score = model.score(candidate)
        candidates[i]['final_score'] = (
            (1 - cfg.ml_rescore.weight) * candidate['rule_score'] +
            cfg.ml_rescore.weight * ml_score
        )
    
    candidates = candidates.sort_values('final_score', ascending=False)
```

#### ✅ Config/品質系

| ファイル | 機能 | 統合方法 |
|---------|------|---------|
| `pattern_quality_config.py` | **品質閾値設定** | arranger_weights.yaml統合 |
| `v3_filter_config.py` | **v3フィルタ設定** | arranger_weights.yaml統合 |
| `chord_fit_config.py` | **Chord適合度設定** | arranger_weights.yaml統合 |

**arranger_weights.yaml拡張例**:
```yaml
ml_rescore:
  enabled: true
  model_path: output/rhythm_ai/stage2_drums_rhythm_ai.pickle
  feature_cols:
    - tempo_bpm
    - swing_pct
    - hat_density
    - backbeat_strength
  weight: 0.35  # ルール:ML = 0.65:0.35

exploration:
  epsilon: 0.10  # 10% exploration
  cap_by_section:
    Chorus: 0.05  # Chorus: 探索5%に抑制
    Verse: 0.15   # Verse: 探索15%
    Bridge: 0.20  # Bridge: 探索20%
```

### 要アダプタファイル（中パッチで統合）

| ファイル | 機能 | 統合方法 |
|---------|------|---------|
| `pattern_quality_learner.py` | **ML学習器** | Quality再スコア |
| `exploration_manager.py` | **探索管理** | Diversity強化 |

**Quality Learner統合例**:
```python
# Pattern MatcherへのML再スコア追加
from ml.pattern_quality_learner import load_quality_model

quality_model = load_quality_model(cfg.ml_rescore.model_path)

# Top-K候補の品質予測
proba = quality_model.predict_proba(
    candidates[cfg.ml_rescore.feature_cols].to_numpy()
)[:, 1]

candidates['ml_score'] = proba
candidates['final_score'] = (
    (1 - cfg.ml_rescore.weight) * candidates['rule_score'] +
    cfg.ml_rescore.weight * candidates['ml_score']
)
```

### アーカイブファイル（参考・温存）

| ファイル | 機能 | 備考 |
|---------|------|------|
| `stage3_generator.py` | **Stage3生成器** | Plan→midi_writer統一で非推奨 |
| `stage3_infer.py` | **Stage3推論** | 同上 |
| `attention_*.py` | **Attention機構** | Transformer系（将来用） |
| `tokenizer_remi.py` | **REMIトークナイザ** | Transformer系（将来用） |

---

## 🚀 統合実装ロードマップ

### Phase 1: Stem特徴抽出（最小パッチ）

**目的**: Stem WAVから小節別特徴を抽出してPlan生成に反映

**実装ステップ**:

1. **stems_features.py作成**（新規）
```python
#!/usr/bin/env python3
"""
Stem WAV → Bar-level Features Extraction

Usage:
    python ops/stems_features.py \
        --stems data/suno_ai/.../stemswav_001 \
        --bars data/suno_ai/.../bars.parquet \
        --output data/suno_ai/.../stem_features.parquet
"""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

def extract_bar_features(stem_drums, stem_mix, bars_df, sr=22050):
    features = []
    
    for idx, bar in bars_df.iterrows():
        start = int(bar['start_sec'] * sr)
        end = int(bar['end_sec'] * sr)
        
        # Drums特徴
        drums_seg = stem_drums[start:end]
        hat_density = onset_count(drums_seg, sr, band=[6000, 12000]) / bar['beats']
        kick_peak_db = peak_db(drums_seg, band=[30, 120])
        snare_backbeat = backbeat_score(drums_seg, sr, bar['beats'])
        
        # Mix特徴
        mix_seg = stem_mix[start:end]
        loudness = rms_db(mix_seg)
        
        features.append({
            'bar': idx,
            'hat_density': hat_density,
            'kick_peak_db': kick_peak_db,
            'snare_backbeat': snare_backbeat,
            'loudness': loudness
        })
    
    return pd.DataFrame(features)

def onset_count(y, sr, band):
    # バンドパスフィルタ → Onset検出
    y_filt = librosa.effects.preemphasis(y)
    onset_env = librosa.onset.onset_strength(y=y_filt, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    return len(onsets)

def peak_db(y, band):
    # ピーク振幅（dB）
    return 20 * np.log10(np.max(np.abs(y)) + 1e-9)

def backbeat_score(y, sr, beats):
    # 2拍目・4拍目のエネルギー比
    hop = len(y) // beats
    beat_energies = [rms_db(y[i*hop:(i+1)*hop]) for i in range(beats)]
    backbeat = (beat_energies[1] + beat_energies[3]) / 2
    return backbeat / (np.mean(beat_energies) + 1e-9)

def rms_db(y):
    return 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)
```

2. **scripts/recommend_drums.py修正**
```python
# --stems-features引数追加
parser.add_argument('--stems-features', type=Path, help='Stem features parquet')

# 密度ブースト
if args.stems_features:
    stem_df = pd.read_parquet(args.stems_features)
    for idx, bar in bars_df.iterrows():
        stem_hat = stem_df.loc[idx, 'hat_density']
        target_density = max(
            bar['density_target'],
            stem_hat * cfg.stems.drums.density_boost
        )
        bars_df.loc[idx, 'density_target'] = target_density
```

3. **scripts/generate_piano_strings_plans.py修正**
```python
# Energy Curveブレンド
if args.stems_features:
    stem_df = pd.read_parquet(args.stems_features)
    for idx, bar in bars_df.iterrows():
        stem_loudness = stem_df.loc[idx, 'loudness']
        # 正規化
        stem_loudness_norm = (stem_loudness - stem_df['loudness'].min()) / \
                             (stem_df['loudness'].max() - stem_df['loudness'].min())
        
        # ブレンド
        alpha = cfg.stems.piano.loudness_blend
        blended_energy = (1 - alpha) * bar['energy'] + alpha * stem_loudness_norm
        bars_df.loc[idx, 'energy'] = blended_energy
```

### Phase 2: ML再スコア統合

**目的**: pattern_recommender.pyを使ってTop-K候補の品質再評価

**実装ステップ**:

1. **arranger_weights.yaml拡張**
```yaml
ml_rescore:
  enabled: true
  model_path: ml/stage2_drums_rhythm_ai.pickle
  feature_cols:
    - tempo_bpm
    - swing_pct
    - hat_density
    - backbeat_strength
  weight: 0.35
```

2. **scripts/pattern_matcher.py修正**
```python
from ml.pattern_recommender import PatternRecommender

if cfg.ml_rescore.enabled:
    recommender = PatternRecommender(
        role=role,
        pickle_path=cfg.ml_rescore.model_path
    )
    
    # Top-K再スコア
    for i, row in candidates.iterrows():
        ml_score = recommender.score_pattern(row)
        candidates.loc[i, 'ml_score'] = ml_score
    
    candidates['final_score'] = (
        (1 - cfg.ml_rescore.weight) * candidates['rule_score'] +
        cfg.ml_rescore.weight * candidates['ml_score']
    )
    
    candidates = candidates.sort_values('final_score', ascending=False)
```

### Phase 3: Exploration Manager統合

**目的**: セクション別探索上限で多様性向上

**実装ステップ**:

1. **config/exploration_config.yaml作成**
```yaml
exploration:
  epsilon: 0.10
  cap_by_section:
    Chorus: 0.05  # 探索5%
    Verse: 0.15
    Bridge: 0.20
  min_exploration_samples: 10
  quality_threshold: 0.70
```

2. **scripts/pattern_matcher.py修正**
```python
from ml.exploration_manager import ExplorationManager

exp_mgr = ExplorationManager(
    epsilon=cfg.exploration.epsilon,
    config_path='config/exploration_config.yaml'
)

# セクション別探索判定
if exp_mgr.should_explore_section(section=section):
    pattern = exp_mgr.select_exploration_pattern(
        exploration_pool=low_rank_candidates,
        section=section
    )
else:
    pattern = candidates.iloc[0]  # Top-1採用

# 結果記録
exp_mgr.record_exploration_result(
    pattern_id=pattern['loop_id'],
    quality_score=pattern['final_score'],
    section=section
)
```

---

## 📊 期待効果（KPI Pass率）

### 現行（ChordMap主導）

| 指標 | Pass率 |
|------|--------|
| **Overall** | **80.7%** |
| Drums密度 | 75.0% |
| Fill検出 | 70.0% |
| セクション強弱 | 78.0% |

### Phase 1実装後（Stem特徴追加）

| 指標 | 期待Pass率 |
|------|-----------|
| **Overall** | **~88%** |
| Drums密度 | **85%** ← Stem hat_density反映 |
| Fill検出 | **82%** ← anchor同期 |
| セクション強弱 | **86%** ← loudnessブレンド |

### Phase 2実装後（ML再スコア）

| 指標 | 期待Pass率 |
|------|-----------|
| **Overall** | **~92%** |
| Pattern品質 | **90%** ← ML質スコア |
| Groove一貫性 | **88%** ← Top-K再評価 |

### Phase 3実装後（Exploration）

| 指標 | 期待Pass率 |
|------|-----------|
| **Overall** | **~94%** |
| 多様性 | **92%** ← セクション別探索 |
| 新Pattern発見 | **85%** ← epsilon-greedy |

---

## 🎯 推奨実装順序

### 最小パッチ（1-2日）

```bash
# 1. Stem特徴抽出スクリプト作成
# ops/stems_features.py

# 2. recommend_drums.py修正
# --stems-features引数追加 + 密度ブースト

# 3. generate_piano_strings_plans.py修正
# Energy Curveブレンド

# 4. 動作確認
python scripts/full_pipeline.py \
    --vocal data/suno_ai/.../Vocals.wav \
    --accompaniment data/suno_ai/.../Other.wav \
    --output output/hybrid_v1 \
    --stems-features data/suno_ai/.../stem_features.parquet
```

### 中パッチ（3-5日）

```bash
# 1. arranger_weights.yaml拡張
# ml_rescore設定追加

# 2. pattern_matcher.py修正
# ML再スコア統合

# 3. KPI比較
python scripts/batch_kpi_compare.py \
    --before output/baseline \
    --after output/hybrid_v2
```

### フルパッチ（1週間）

```bash
# 1. exploration_config.yaml作成

# 2. Exploration Manager統合

# 3. Production運用移行
```

---

## 🔍 関連ファイル一覧

### Stem WAV関連

- `ops/stem_harmony.py` - WAV → Chord Recognition
- `ops/audio_safe.py` - NumPy版オーディオ処理
- `ops/anchors_from_vocal.py` - Vocal → Anchors
- `ops/sections_from_audio.py` - WAV → Sections

### ChordMap関連

- `ops/normalize_chordmap_format.py` - 統一スキーマ
- `ops/chordmap_expand_bars.py` - 小節展開
- `ops/scale_modes.py` - Key/Modeマスク

### Section/Tempo関連

- `ops/finalize_sections.py` - Sections仕上げ
- `ops/enrich_anchors.py` - Anchors連携
- `ops/anchors_to_midi.py` - Chorus MIDI生成

### ML関連

- `ml/pattern_recommender.py` - 汎用推薦
- `ml/drum_pattern_recommender.py` - Drums推薦
- `ml/exploration_manager.py` - 探索管理
- `ml/pattern_quality_config.py` - 品質設定

---

## 💡 まとめ

### 最重要ポイント

1. ✅ **ChordMap = 和声骨格**、**Stem WAV = リズム/ダイナミクス**
2. ✅ **ハイブリッド方式が最も効果的**（KPI 80.7% → 94%期待）
3. ✅ **最小パッチで即効果**（Stem特徴抽出 → 密度ブースト）
4. ✅ **ML再スコアで品質向上**（pattern_recommender統合）
5. ✅ **既存Plan→midi_writer統一は維持**（suno_stem_arranger.py不要）

### 次のアクション

```bash
# 1. Stem特徴抽出実装
vim ops/stems_features.py

# 2. Drums/Piano/Strings Plan修正
vim scripts/recommend_drums.py
vim scripts/generate_piano_strings_plans.py

# 3. 動作確認
python scripts/full_pipeline.py --stems-features ...

# 4. KPI比較
python scripts/batch_kpi_compare.py --before baseline --after hybrid
```

**準備完了！ハイブリッド統合を開始してください🎵**
