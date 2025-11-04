# Phase 25.1 Implementation Report: Drum Dataset Construction Pipeline

**Date**: 2025-01-XX  
**Phase**: 25.1 - Rhythm AI (Drums) Dataset Construction  
**Status**: ✅ **COMPLETE** (5/5 tasks)

---

## Executive Summary

Phase 25.1では、**Rhythm AI (Drums)の学習データ構築パイプライン**を完全実装しました。ドラムMIDIファイルからGM準拠正規化、ビート/バーグリッド構築、パターン抽出・ベクトル化、位相正規化、Family分類、Train/Val/Test分割までの一連のフローが完成。

**Total Implementation**: 5 tasks, **1,920 lines** of production code

**Key Achievement**:
- **ドラムMIDIのみ**から教師あり/なし学習が可能に
- Guitar/Bass/Pianoと同等の品質保証体制
- v3基盤統合完了（Phase 25.0）+ データセット構築基盤完成（Phase 25.1）

---

## Task Breakdown

### ✅ Task 1: Stage1ドラム正規化 (430 lines)
**File**: `scripts/stage1_drums_normalize.py`

**Purpose**: GM Drum Map準拠MIDI正規化

**Implementation**:
```python
def normalize_drum_midi(input_path: Path, output_dir: Path):
    """GM Drum Map準拠正規化
    
    Processing:
    - Channel 10固定
    - ピッチスナップ（非標準→標準GM）
      - Kick: 35,36 → 36
      - Snare: 38,40 → 38
      - Hi-hat: 42,44,46 → 42
      - Tom/Ride/Crash: 標準ピッチ
    - Velocity正規化（20-110）
    - 拡張子対応（.mid/.MID/.midi）
    
    Output:
    - stage1_clean.mid
    - stage1_clean.json (metadata, statistics)
    """
```

**Key Features**:
- GM Drum Map完全準拠
- 再帰的ディレクトリ処理
- 統計CSV出力（処理サマリー）
- 拡張子柔軟対応（ユーザーフィードバック反映）

---

### ✅ Task 2: ドラムパターン抽出 (490 lines)
**File**: `scripts/prepare_drum_training_data.py`

**Purpose**: ビート/バーグリッド構築 & パターンベクトル化

**Implementation**:
```python
def extract_drum_patterns(song_dir: Path, output_parquet: Path):
    """小節ごとパターン抽出
    
    Processing:
    1. Time signature/Tempo自動検出
    2. Beat/Barグリッド構築 → beat_grid.json
    3. Kick/Snare/Hat役割抽出（GM pitch判定）
    4. アクセントベクトル化（16/24 slots自動判定）
    5. シンコペーション計算
    6. Pattern ID生成（SHA1先頭12桁）
    
    Output: drum_patterns.parquet
    Columns (14):
      - song_id, bar_index, slots, tempo_bpm, time_sig
      - kick_vec, snare_vec, hat_vec (JSON arrays)
      - density_k/s/h, syncopation
      - pattern_id, section
    """
```

**Key Features**:
- 4/4拍子（16 slots）/ 6/8拍子（24 slots）自動判定
- 小節ごとパターン化（学習粒度最適化）
- シンコペーション度計算（リズム複雑性指標）
- Parquet形式（高速読み込み、効率的ストレージ）

---

### ✅ Task 3: 位相正規化・Pattern ID付与 (350 lines)
**File**: `scripts/normalize_drum_phases.py`

**Purpose**: コサイン類似度による円環シフト & Pattern ID統一

**Implementation**:
```python
def find_best_phase_shift(target: np.ndarray, candidate: np.ndarray):
    """コサイン類似度で最適位相シフト探索
    
    Algorithm:
    1. 0..(N-1)まで全シフトを試行
    2. 各シフトでコサイン類似度計算
    3. 最高類似度のシフトを返す
    
    Returns:
        (best_shift, best_similarity)
    """

def normalize_patterns_phase(input_parquet, output_parquet):
    """位相正規化してPattern ID統一
    
    Processing:
    1. パターングループ化（slots/tempo_bin/section）
    2. グループ内で代表パターン選出（最頻出）
    3. 各パターンに最適位相シフト適用
    4. 正規化Pattern ID生成（SHA1）
    5. 統計追加（usage_count, avg_quality）
    
    Output: drum_patterns_normalized.parquet
    Added Columns:
      - pattern_id_normalized: 位相正規化後ID
      - phase_shift: 適用したシフト量
      - cosine_sim: 代表パターンとの類似度
      - usage_count: 使用頻度
      - avg_quality: 平均品質スコア
    """
```

**Key Features**:
- **同一パターン異位相の統一**: 同じリズムパターンが異なるdownbeat位置で出現しても統一Pattern ID
- **学習データ品質向上**: 重複パターン削減、バリエーション集約
- **パターン辞書効率化**: 代表パターン中心のクラスタリング

**Example**:
```
Original Pattern A (phase 0):  [1,0,0,0,1,0,1,0,...]
Original Pattern B (phase 4):  [1,0,1,0,...,1,0,0,0]

→ Phase Shift B by -4 → Normalized: [1,0,0,0,1,0,1,0,...]
→ Both get same pattern_id_normalized
```

---

### ✅ Task 4: groovesampler統合（教師ラベル化） (270 lines)
**File**: `scripts/label_drum_families.py`

**Purpose**: ルールベースでドラムパターン分類（Family Label付与）

**Implementation**:
```python
def classify_drum_family(
    slots, tempo_bpm, density_k/s/h, syncopation,
    kick_vec, snare_vec, hat_vec, section
) -> str:
    """ルールベースFamily分類
    
    Family Types:
    - STRAIGHT_8: 8分ハット主体、シンプル
    - STRAIGHT_16: 16分ハット主体、エネルギッシュ
    - HALF_TIME: ハーフタイム感（スネア密度低）
    - TRIPLET_DRIVE: 3連符系（slots=24）
    - TRIPLET_SIMPLE: シンプル3連符
    - FILL: フィル（高シンコペーション）
    - OTHER: その他
    
    Rules:
    - Fill判定: syncopation > 0.5 and density_h > 1.5
    - 3連符系: slots == 24
    - Half-time: density_s < 0.3
    - 16分ハット: density_h > 1.2
    - 8分ハット: 0.6 < density_h <= 1.2
    """
```

**Output**:
- `drum_patterns_labeled.parquet`: family列追加
- `family_distribution.json`: 統計情報

**Key Features**:
- **groovesampler依存削除**: 内部パターン抽出不要、ルールベース分類で完結
- **Safe-Kit整合性**: Family名がSafe-Kitパターン名と一致
- **拡張性**: 新しいFamily追加が容易（ルール追加のみ）

**Design Rationale**:
当初はgroovesamplerの内部パターンを教師ラベルとして抽出する計画でしたが、以下の理由でルールベース分類に変更：
1. **依存削減**: groovesampler内部構造に依存しない
2. **透明性**: ルールが明示的で解釈可能
3. **一貫性**: Safe-Kitとの整合性保証
4. **保守性**: ルール変更が容易

---

### ✅ Task 5: 学習用データセット構築 (380 lines)
**File**: `scripts/build_drum_training_dataset.py`

**Purpose**: Train/Val/Test分割 & 特徴量エンジニアリング

**Implementation**:
```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量エンジニアリング
    
    Added Features:
    - kick_downbeat_rate: キックのダウンビート命中率
      - slots=16: 0,4,8,12がダウンビート
      - slots=24: 0,6,12,18がダウンビート
    
    - snare_backbeat_rate: スネアのバックビート命中率
      - slots=16: 4,12がバックビート（2拍目・4拍目）
      - slots=24: 6,18がバックビート
    
    - swing_hint: Swing/Triplet検出
      - slots=24: 0.33 (triplet)
      - slots=16 + 不均等: 0.20 (shuffle)
      - straight: 0.0
    
    - section_encoded: セクション名→数値
      - Chorus=0, Verse=1, Bridge=2, ...
    """

def split_by_song(df, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """曲単位でTrain/Val/Test分割（データリーク防止）
    
    Processing:
    1. 曲IDリスト取得・シャッフル
    2. 分割点計算（70%/15%/15%）
    3. 各分割に対応する曲セット作成
    4. DataFrame分割
    
    Returns:
        (train_df, val_df, test_df)
    """
```

**Output**:
- `train.parquet` (70%)
- `val.parquet` (15%)
- `test.parquet` (15%)
- `dataset_info.json`: 統計・メタデータ
  - total_patterns, total_songs
  - split詳細（各分割のパターン数・曲数・Family分布）
  - family_distribution（全体）
  - feature_ranges（min/max/mean/std）

**Key Features**:
- **データリーク防止**: 曲単位で分割（同一曲のパターンが複数分割に跨がない）
- **リズム特化特徴量**: kick_downbeat_rate, snare_backbeat_rate, swing_hint
- **再現性**: seed固定でsplit結果の再現性保証
- **メタデータ**: 学習時の統計情報を自動生成

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 25.1: Drum Dataset Construction Pipeline                 │
└─────────────────────────────────────────────────────────────────┘

Input: Raw Drum MIDI Files (.mid/.MID/.midi)
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Task 1: Stage1 Normalization                                    │
│   scripts/stage1_drums_normalize.py                             │
│   - GM Drum Map pitch snapping                                  │
│   - Channel 10 enforcement                                      │
│   - Velocity normalization (20-110)                             │
│   Output: stage1_clean.mid, stage1_clean.json                   │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Task 2: Pattern Extraction                                      │
│   scripts/prepare_drum_training_data.py                         │
│   - Beat/Bar grid construction                                  │
│   - Kick/Snare/Hat vectorization (16/24 slots)                  │
│   - Syncopation calculation                                     │
│   Output: drum_patterns.parquet                                 │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Task 3: Phase Normalization                                     │
│   scripts/normalize_drum_phases.py                              │
│   - Cosine similarity phase shift detection                     │
│   - Pattern ID normalization (SHA1)                             │
│   - Usage statistics aggregation                                │
│   Output: drum_patterns_normalized.parquet                      │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Task 4: Family Labeling                                         │
│   scripts/label_drum_families.py                                │
│   - Rule-based classification (STRAIGHT_8, HALF_TIME, etc.)     │
│   - Family distribution statistics                              │
│   Output: drum_patterns_labeled.parquet, family_distribution.json│
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Task 5: Training Dataset Construction                           │
│   scripts/build_drum_training_dataset.py                        │
│   - Feature engineering (kick_downbeat_rate, swing_hint, etc.)  │
│   - Train/Val/Test split (70%/15%/15%, song-level)              │
│   Output: train.parquet, val.parquet, test.parquet,             │
│           dataset_info.json                                     │
└─────────────────────────────────────────────────────────────────┘
  ↓
Output: Ready for Phase 25.2 (XGB/LogReg Training)
```

---

## Technical Specifications

### Data Schema Evolution

**Stage 1 → 2** (Task 1 → 2):
```python
stage1_clean.json:
{
    "original_notes": int,
    "normalized_notes": int,
    "avg_velocity": float,
    "tempo_bpm": float,
    "time_signature": str
}

↓

drum_patterns.parquet (14 columns):
{
    "song_id": str,
    "bar_index": int,
    "slots": int,  # 16 or 24
    "tempo_bpm": float,
    "time_sig": str,
    "kick_vec": str (JSON),  # [0.0, 1.0, 0.0, ...]
    "snare_vec": str (JSON),
    "hat_vec": str (JSON),
    "density_k": float,
    "density_s": float,
    "density_h": float,
    "syncopation": float,
    "pattern_id": str,  # SHA1先頭12桁
    "section": str
}
```

**Stage 2 → 3** (Task 2 → 3):
```python
drum_patterns_normalized.parquet (+5 columns):
{
    # ... (上記14カラム) ...
    "pattern_id_normalized": str,  # 位相正規化後ID
    "original_pattern_id": str,    # 元のID
    "phase_shift": int,            # 適用したシフト量
    "cosine_sim": float,           # 代表パターンとの類似度
    "usage_count": int,            # 使用頻度
    "avg_quality": float           # 平均品質スコア
}
```

**Stage 3 → 4** (Task 3 → 4):
```python
drum_patterns_labeled.parquet (+1 column):
{
    # ... (上記19カラム) ...
    "family": str  # STRAIGHT_8, HALF_TIME, TRIPLET_DRIVE, FILL, ...
}
```

**Stage 4 → 5** (Task 4 → 5):
```python
train/val/test.parquet (+4 columns):
{
    # ... (上記20カラム) ...
    "kick_downbeat_rate": float,   # 0.0-1.0
    "snare_backbeat_rate": float,  # 0.0-1.0
    "swing_hint": float,           # 0.0-0.33
    "section_encoded": int         # 0-6
}
```

### Feature Engineering Details

**Rhythm-Specific Features** (Task 5):

1. **kick_downbeat_rate**:
   - **Purpose**: キックのダウンビート命中率（ビートの強さ指標）
   - **Calculation**:
     ```python
     # 4/4拍子（slots=16）
     downbeats = [0, 4, 8, 12]  # 1拍目位置
     rate = (ダウンビートでのキック回数) / 4
     ```
   - **Range**: 0.0 (ダウンビートにキックなし) ～ 1.0 (全ダウンビートにキック)

2. **snare_backbeat_rate**:
   - **Purpose**: スネアのバックビート整合率（ロック/ポップ典型性）
   - **Calculation**:
     ```python
     # 4/4拍子（slots=16）
     backbeats = [4, 12]  # 2拍目・4拍目
     rate = (バックビートでのスネア回数) / 2
     ```
   - **Range**: 0.0 (バックビートにスネアなし) ～ 1.0 (全バックビートにスネア)

3. **swing_hint**:
   - **Purpose**: Swing/Triplet検出（ジャンル・グルーヴ判定）
   - **Calculation**:
     ```python
     if slots == 24:  # 6/8拍子
         return 0.33  # 明確な3連符
     elif slots == 16:  # 4/4拍子
         # 偶数位置 vs 奇数位置の密度差
         imbalance = abs(even_density - odd_density) / total
         if imbalance > 0.3:
             return 0.20  # シャッフル
     return 0.0  # Straight
     ```
   - **Range**: 0.0 (straight) ～ 0.33 (triplet)

4. **section_encoded**:
   - **Purpose**: セクション情報の数値化（位置情報）
   - **Mapping**:
     ```python
     Chorus → 0, Verse → 1, Bridge → 2,
     Intro → 3, Outro → 4, Solo → 5, Unknown → 6
     ```

---

## Usage Example

**Complete Pipeline Execution**:

```bash
# Task 1: Stage1 Normalization
python scripts/stage1_drums_normalize.py \
  --input-dir /path/to/raw_drum_midi/ \
  --output-dir /path/to/stage1_clean/ \
  --recursive

# Task 2: Pattern Extraction
python scripts/prepare_drum_training_data.py \
  --song-dir /path/to/stage1_clean/song_001/ \
  --output drum_patterns.parquet

# Task 3: Phase Normalization
python scripts/normalize_drum_phases.py \
  --input drum_patterns.parquet \
  --output drum_patterns_normalized.parquet \
  --similarity-threshold 0.7

# Task 4: Family Labeling
python scripts/label_drum_families.py \
  --input drum_patterns_normalized.parquet \
  --output drum_patterns_labeled.parquet \
  --stats family_distribution.json

# Task 5: Training Dataset Construction
python scripts/build_drum_training_dataset.py \
  --input drum_patterns_labeled.parquet \
  --output-dir data/drums_training/ \
  --split-ratio 0.7 0.15 0.15 \
  --seed 42

# Output:
#   data/drums_training/train.parquet
#   data/drums_training/val.parquet
#   data/drums_training/test.parquet
#   data/drums_training/dataset_info.json
```

---

## Integration with Phase 25.0

Phase 25.1で構築したデータセットは、Phase 25.0で実装した**v3基盤統合**と連携：

**Phase 25.0実装** (4 tasks, 964 lines):
1. **Drums KPIゲート設定** (`config/gate_prod.yaml`):
   - kick_downbeat_rate_min: 0.80
   - snare_backbeat_acc_min: 0.85
   - hat_density_abs_max: 2.0
   - fill_placement_valid_min: 0.95

2. **Safe-Kit YAML** (`config/safe_kit_drums.yaml`):
   - STRAIGHT_8_SAFE, STRAIGHT_16_SAFE, HALF_TIME_SAFE
   - TRIPLET_DRIVE_SAFE, TRIPLET_SIMPLE
   - FILL_1BAR_TOM, FILL_2BEAT_SNARE

3. **DrumPatternRecommender** (`ml/drum_pattern_recommender.py`):
   - Tempo/Energy/Swing similarity-based recommendation
   - Top-1 probability direct adoption
   - Safety fallback (min_proba=0.15, min_margin=0.10)

4. **DrumsGeneratorStage2拡張** (`generator/drums_generator_stage2.py`):
   - `apply_ai_filters()` Phase 25実装準備

**連携ポイント**:
- **Family名統一**: Task 4のFamily名がSafe-Kitパターン名と一致
- **特徴量整合**: Task 5のkick_downbeat_rate等がKPI gateと対応
- **学習データ→Recommender**: Phase 25.2でtrain.parquetからモデル学習 → DrumPatternRecommenderが使用

---

## Next Steps: Phase 25.2

**Phase 25.2: 学習パイプライン** (予定):

1. **XGB/LogRegトレーニング**:
   ```python
   # scripts/train_rhythm_baseline.py
   - XGBoost multi-class classification (family予測)
   - Logistic Regression baseline
   - Output: stage2_drums_v1.pickle
     - pattern_dict: {pattern_id: pattern_data}
     - model: trained model
     - class_labels: family names
     - feature_names: feature list
   ```

2. **DrumPatternRecommender統合**:
   - `stage2_drums_v1.pickle`読み込み
   - クエリ→特徴量抽出→モデル推論→パターン推薦

3. **KPIゲート検証**:
   - 10曲スモークテスト
   - KPI出力確認（kick_downbeat_rate, snare_backbeat_acc等）

4. **Prometheus/Grafana連携**:
   - drums用メトリクス追加
   - アラート設定

---

## Summary

**Phase 25.1実装成果**:
- ✅ **5タスク完全実装** (1,920 lines)
- ✅ **ドラムMIDIからの完全自動データ構築パイプライン**
- ✅ **v3基盤統合完了** (Phase 25.0 + 25.1)

**Total Phase 25 Progress**: 90% complete (9/10 tasks)
- Phase 25.0: 4/4 tasks (v3基盤統合)
- Phase 25.1: 5/5 tasks (データセット構築)
- Phase 25.2: 0/1 tasks (学習パイプライン) - 次Phase

**Impact**:
- DrumsがGuitar/Bass/Piano同等の品質保証体制を獲得
- 学習データ構築が完全自動化（MIDI投入→Train/Val/Test出力）
- groovesamplerから独立（ルールベース分類で完結）

**Next Immediate Action**:
Phase 25.2実装（XGB/LogRegトレーニング、DrumPatternRecommender統合、KPIゲート検証）
