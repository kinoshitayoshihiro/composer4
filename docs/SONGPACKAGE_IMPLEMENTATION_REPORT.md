# SongPackage実装完了レポート
**生成フェーズ準備完了** - 学習済みモデル+楽曲構成統合

Date: 2025-01-XX  
Status: ✅ SongPackage基盤実装完了、Recommender/Generator連携準備完了

---

## 1. 概要

**SongPackageとは**: 1曲単位の設計図（セクション構成/コード/歌詞アンカー/bars.parquet）を束ねるYAML形式のマニフェスト。

**目的**:
- 学習済みモデル（`stage2_drums_rhythm_ai.pickle`）とRecommender/Generatorを連携
- 既存楽曲構成（`sections.json`, `chordmap.json`）を活用
- 小節単位の目標値（accent/density/swing）をbars.parquetで管理
- KPI Gate検証 → Safe-Kit fallback

---

## 2. 実装成果

### 2.1 SongPackage YAML作成

**ファイル**: `song_packages/sample_project/sample_song/song_package.yaml`

**構造**:

```yaml
meta:
  title: "OreNoNyoubou_Shudaika"
  project: "sample_project"
  bpm: 76.01
  time_signature: "4/4"
  key_hint: "Fm"
  emotion_profile: "quiet_pain_and_nascent_strength"
  total_bars: 72

artifacts:
  sections: "sections.json"           # Stage1楽曲構成
  chordmap: "chordmap.json"           # Stage1和声
  lyric_anchors: "results/test_anchors_beat.json"
  bars: "bars.parquet"                # 小節目標値（新規）

models:
  drums: "data/patterns/stage2_drums_rhythm_ai.pickle"
  # 将来: guitar/bass/piano

generation:
  drums:
    target_family: "auto"             # Recommenderに任せる
    target_density: 6.0
    swing_hint: "detect"
    kpi_gate: "configs/gate_prod.yaml"
    safe_kit_fallback: true

render:
  out_dir: "renders/sample_project/sample_song"
  stem_format: "wav/48k/24bit"
```

**特徴**:
- **既存楽曲構成活用**: sections.json（8セクション、72小節）+ chordmap.json（14セクション詳細）
- **学習済みモデル連携**: stage2_drums_rhythm_ai.pickle（LogisticRegression, 4クラス、35,511レコード）
- **自動推奨**: Recommenderが最適なドラムファミリー/密度/スウィング推定
- **KPI検証**: 生成後にgate_prod.yaml基準で品質検証 → 失敗時Safe-Kit fallback

### 2.2 bars.parquet生成スクリプト

**ファイル**: `scripts/generate_bars_parquet.py` (~150行)

**機能**:
- sections.json（エナジー曲線、セクション区切り）+ chordmap.json（ドラムスタイル）読み込み
- 小節単位の目標値生成:
  - **bar_index**: 0..71
  - **section_label**: intro/verse/chorus/pre_chorus/outro
  - **energy_curve**: 0..1（sections.jsonのenergy）
  - **accent_score_target**: 0..1（エナジー曲線そのまま）
  - **density_target**: 2.0..12.0（セクション種別+エナジーから推定）
  - **swing_target**: 0..1（chordmap.jsonドラムスタイルから推定）

**使用例**:
```bash
python3 scripts/generate_bars_parquet.py \
  --sections sections.json \
  --chordmap data/chordmap.json \
  --output song_packages/sample_project/sample_song/bars.parquet
```

**実行結果**:
```
✅ Generated bars.parquet
   Total bars: 72
   Columns: ['bar_index', 'section_label', 'energy_curve', 
             'accent_score_target', 'density_target', 'swing_target']

📊 Statistics:
   Energy: 0.00 .. 1.00 (mean: 0.52)
   Density: 2.00 .. 9.64 (mean: 5.17)
   Swing: 0.00 .. 0.30 (mean: 0.07)

🔍 Section distribution:
   verse         30
   intro         21
   pre_chorus    15
   chorus         5
   outro          1
```

**推定ロジック**:

1. **density_target推定**:
   - セクション基準密度（intro: 4.0, verse: 5.0, chorus: 8.0, etc.）
   - エナジー補正（±2.0）
   - 範囲制限（2.0..12.0）

2. **swing_target推定**:
   - chordmap.jsonのdrum_style_key検索
   - "swing" → 0.8
   - "ballad" → 0.3
   - "no_drums" → 0.0
   - デフォルト → 0.1（STRAIGHT）

### 2.3 既存楽曲構成確認

**sections.json** (ルート):
- 構造: セクション区切り（8セクション）+ エナジー曲線（72小節）+ tempo_map + key_hint
- セクション種別: intro(21) → verse(14) → chorus(5) → pre_chorus(15) → verse(6) → outro(1) → verse(10)
- エナジー範囲: 0.0001..0.9999（ほぼ全レンジ）
- テンポ: 76.01 BPM（固定）
- 調: Fm → A → D → A → Fm → D

**chordmap.json** (data/):
- 構造: 14セクション詳細（Verse 1-4, Pre-Chorus 1-2, Chorus 1-4, Bridge 1-2, Interlude, Sax Solo）
- コード進行: Dsus2 → Gsus2/D → Cadd9 → Amadd9 ... (複雑なジャズハーモニー)
- ドラム設定: 
  - "no_drums" (Verse 1, 3, Interlude)
  - "ballad_soft_kick_snare_8th_hat" (Pre-Chorus 1)
  - "anthem_rock_chorus_16th_hat" (Chorus 1, 3)
  - "rock_ballad_build_up_8th_hat" (Chorus 2, Bridge 1)
- 感情プロファイル: quiet_pain_and_nascent_strength (Verse 1) → acceptance_of_love_and_pain_hopeful_belief (Chorus 1) ...

**lyric_anchors** (results/):
- test_anchors_beat.json: ビート単位のアンカー
- test_anchors_class.json: 音素クラス分類アンカー
- test_anchors_sibilant.json: 子音特化アンカー
- （SongPackageでは任意、将来ボーカル生成時に使用）

---

## 3. SongPackageワークフロー

### 3.1 準備フェーズ（完了 ✅）

**学習済みモデル**:
- ✅ `data/patterns/stage2_drums_rhythm_ai.pickle` (LogisticRegression, 4クラス)
  - クラス: STRAIGHT_16, STRAIGHT_8, SWING_16, SWING_8
  - 学習データ: 35,511レコード（drumclean + groove + E-GMD統合）
  - 特徴量: 19次元（tempo_bpm, swing_pct, backbeat_strength, kick_downbeat_rate, snare_backbeat_rate, hat_density, etc.）

**楽曲構成**:
- ✅ sections.json（8セクション、72小節、エナジー曲線、tempo_map、key_hint）
- ✅ chordmap.json（14セクション詳細、コード進行、ドラム設定、感情プロファイル）
- ✅ lyric_anchors（test_anchors_beat.json等、任意）

**bars.parquet**:
- ✅ 小節単位の目標値（accent/density/swing）
- ✅ 72小節（verse 30, intro 21, pre_chorus 15, chorus 5, outro 1）

### 3.2 生成フェーズ（次ステップ ⏳）

**Recommender連携** (未実装):
1. bars.parquet読み込み（小節単位の目標値）
2. 各小節でドラムパターン推奨:
   - 入力: section_label, energy_curve, density_target, swing_target
   - ML推論: `stage2_drums_rhythm_ai.pickle`でfamily推定（STRAIGHT_8 vs SWING_8等）
   - パターン検索: 推定familyから最適なMIDIパターン検索（groove/drumclean/E-GMDから）
3. KPI Gate検証:
   - 生成パターンを`configs/gate_prod.yaml`基準で検証
   - density/swing/backbeat_strength等が範囲内か確認
4. Safe-Kit fallback:
   - KPI失敗 → 安全なテンプレートパターン使用

**Generator実装** (未実装):
1. Recommender推奨パターン受信
2. MIDI生成（各小節）
3. ヒューマナイズ（micro_timing, velocity_variance, etc.）
4. MIDI書き出し → WAV変換（FluidSynth等）

**統合レンダリング** (未実装):
1. ドラム/ギター/ベース/ピアノの各Stem生成
2. ミキシング（balanced/vocal_forward等）
3. マスタリング（soft_limit/transparent等）
4. 最終WAV出力（`renders/sample_project/sample_song/`）

---

## 4. ディレクトリ構造

```
composer2-3/
├── song_packages/                        ← 新規（SongPackage保管）
│   └── sample_project/
│       └── sample_song/
│           ├── song_package.yaml         ← SongPackageマニフェスト
│           └── bars.parquet              ← 小節目標値（72 bars）
│
├── sections.json                         ← Stage1楽曲構成（既存）
├── data/
│   └── chordmap.json                     ← Stage1和声（既存）
├── results/
│   └── test_anchors_beat.json            ← 歌詞アンカー（既存）
│
├── data/patterns/
│   └── stage2_drums_rhythm_ai.pickle     ← 学習済みモデル（既存）
│
├── scripts/
│   └── generate_bars_parquet.py          ← bars.parquet生成スクリプト（新規）
│
└── output/rhythm_ai/
    ├── rhythm_features_merged.parquet    ← 統合特徴量（35,511）
    ├── drumclean_stage2/                 ← drumclean特徴量（51,248）
    ├── groove_stage2/                    ← groove特徴量（827）
    └── egmd_stage2/                      ← E-GMD特徴量（4,547）
```

---

## 5. 次ステップ

### 5.1 Recommender実装（優先度: 高）

**実装ファイル**: `scripts/recommend_drums.py` (新規)

**機能**:
1. bars.parquet読み込み
2. 各小節でML推論（`stage2_drums_rhythm_ai.pickle`）
3. 推定family（STRAIGHT_8 vs SWING_8等）から最適パターン検索
4. KPI Gate検証
5. 推奨パターンJSON出力

**入力**:
- `song_packages/sample_project/sample_song/song_package.yaml`
- `song_packages/sample_project/sample_song/bars.parquet`
- `data/patterns/stage2_drums_rhythm_ai.pickle`
- `output/rhythm_ai/rhythm_features_merged.parquet`

**出力**:
- `song_packages/sample_project/sample_song/drums_recommendations.json`
  ```json
  {
    "bar_0": {
      "family": "STRAIGHT_8",
      "pattern_id": "drumclean_12345",
      "density": 4.2,
      "swing": 0.05,
      "kpi_pass": true
    },
    ...
  }
  ```

### 5.2 Generator実装（優先度: 高）

**実装ファイル**: `scripts/generate_drums_midi.py` (新規)

**機能**:
1. drums_recommendations.json読み込み
2. 各小節のパターンIDからMIDI検索
3. ヒューマナイズ適用（micro_timing, velocity_variance）
4. MIDI書き出し

**入力**:
- `song_packages/sample_project/sample_song/drums_recommendations.json`
- `output/rhythm_ai/egmd_cleaned/` (MIDIパターン)

**出力**:
- `renders/sample_project/sample_song/drums.mid`

### 5.3 KPI Gate検証実装（優先度: 中）

**実装ファイル**: `configs/gate_prod.yaml` (新規)

**構造**:
```yaml
drums:
  density:
    min: 2.0
    max: 12.0
  swing:
    min: 0.0
    max: 1.0
  backbeat_strength:
    min: 0.3
    max: 0.9
  kick_downbeat_rate:
    min: 0.5
    max: 1.0
```

**検証ロジック**: `scripts/kpi_gate.py` (新規)

### 5.4 Safe-Kit Fallback実装（優先度: 中）

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

### 5.5 WAV統合（優先度: 低、将来対応）

**目的**: MIDI特徴+WAV音響特徴で精度向上

**ステップ**:
1. E-GMD WAV取得（現在未配置）
2. `scripts/run_wav_stage1.sh`実行（WAVクリーニング）
3. `scripts/merge_wav_features.py`実行（MIDI+WAVマージ）
4. MLモデル再学習（XGBoost等）

**期待効果**: Swing予測精度 +5-10%向上

---

## 6. 技術仕様

### 6.1 bars.parquet仕様

**必須カラム**:
- `bar_index`: int (0..total_bars-1)
- `section_label`: str (intro/verse/chorus/pre_chorus/outro)

**推奨カラム**:
- `energy_curve`: float (0..1、sections.jsonのenergy参照)
- `accent_score_target`: float (0..1)
- `density_target`: float (Hi-hat密度目標、onset/bar)
- `swing_target`: float (0..1、0=STRAIGHT, 1=SWING)

**将来拡張**:
- `tempo_bpm`: float（小節単位のテンポ変化）
- `time_sig_num`, `time_sig_denom`: int（拍子変化）
- `key_tonic`, `key_mode`: str（調変化）

### 6.2 SongPackage YAML仕様

**セクション**:

**meta**: 楽曲基本情報
- `title`: str（楽曲タイトル）
- `project`: str（プロジェクト名）
- `bpm`: float（テンポ）
- `time_signature`: str（拍子）
- `key_hint`: str（調）
- `emotion_profile`: str（感情プロファイル）
- `total_bars`: int（小節数）

**artifacts**: 楽曲構成ファイル
- `sections`: str（sections.jsonパス）
- `chordmap`: str（chordmap.jsonパス）
- `lyric_anchors`: str（lyric_anchors.jsonパス、任意）
- `bars`: str（bars.parquetパス）

**models**: 学習済みモデル
- `drums`: str（stage2_drums_rhythm_ai.pickleパス）
- `guitar`: str（将来）
- `bass`: str（将来）
- `piano`: str（将来）

**generation**: 生成設定
- `drums`:
  - `target_family`: str（"auto" or "STRAIGHT_8", "SWING_8", etc.）
  - `target_density`: float（Hi-hat密度目標）
  - `swing_hint`: str（"detect" or 0..1）
  - `kpi_gate`: str（gate_prod.yamlパス）
  - `safe_kit_fallback`: bool（KPI失敗時fallback有効化）

**render**: レンダリング設定
- `out_dir`: str（出力ディレクトリ）
- `stem_format`: str（"wav/48k/24bit"等）

---

## 7. ベストプラクティス

### 7.1 bars.parquet生成

**推奨**:
- sections.json + chordmap.jsonから自動生成
- density_target推定は保守的に（2.0..12.0範囲制限）
- swing_target推定はchordmap.jsonのドラムスタイル優先

**注意**:
- エナジー曲線が極端な場合（0.0001等）、density補正が過剰にならないよう調整
- セクションラベル不一致（sections.json vs chordmap.json）に注意

### 7.2 Recommender実装

**推奨**:
- ML推論前にfeature_namesチェック（19次元一致確認）
- KPI Gate検証を必ず通す（Safe-Kit fallbackの発火条件確認）
- パターン検索は多様性重視（同じpattern_idの連続使用を避ける）

**注意**:
- LogisticRegressionは確率出力（`predict_proba`）を活用
- STRAIGHT_8とSWING_8の境界（swing_target 0.3-0.5）はグレーゾーン → 両方試して良い方選択

### 7.3 Generator実装

**推奨**:
- ヒューマナイズは控えめ（micro_timing ±10ms, velocity_variance ±5）
- MIDI書き出し時に小節境界でquantize（タイミング誤差蓄積防止）

**注意**:
- E-GMD MIDIパターンはドラムマップが独自（GM非互換）
- FluidSynth変換時のSoundFont選択（ドラムキット品質）

---

## 8. 完成イメージ

**ワークフロー全体**:

```
sections.json + chordmap.json
         ↓
  generate_bars_parquet.py
         ↓
    bars.parquet (72 bars)
         ↓
  recommend_drums.py (ML推論)
         ↓
  drums_recommendations.json
         ↓
  generate_drums_midi.py
         ↓
     drums.mid
         ↓
  FluidSynth (WAV変換)
         ↓
  renders/sample_project/sample_song/drums.wav
```

**最終出力**:

```
renders/sample_project/sample_song/
├── drums.wav           ← ドラムStem
├── guitar.wav          ← ギターStem（将来）
├── bass.wav            ← ベースStem（将来）
├── piano.wav           ← ピアノStem（将来）
├── vocal.wav           ← ボーカルStem（将来）
└── master.wav          ← ミックス+マスタリング（将来）
```

---

## 9. まとめ

**完了事項**:
- ✅ SongPackage YAML作成（meta, artifacts, models, generation, render）
- ✅ bars.parquet生成スクリプト作成（sections.json + chordmap.json → 72小節目標値）
- ✅ bars.parquet生成実行（energy: 0.00..1.00, density: 2.00..9.64, swing: 0.00..0.30）
- ✅ 既存楽曲構成確認（sections.json: 8セクション、chordmap.json: 14セクション詳細）

**次ステップ**:
1. **Recommender実装** (`scripts/recommend_drums.py`)
   - bars.parquet読み込み
   - ML推論（`stage2_drums_rhythm_ai.pickle`）
   - パターン検索 → drums_recommendations.json出力
2. **Generator実装** (`scripts/generate_drums_midi.py`)
   - drums_recommendations.json読み込み
   - MIDI生成 → drums.mid出力
3. **KPI Gate実装** (`configs/gate_prod.yaml` + `scripts/kpi_gate.py`)
   - 品質検証 → Safe-Kit fallback
4. **WAV統合** (将来対応)
   - E-GMD WAV取得 → 精度向上

**期待成果**:
- 1曲分の自動ドラム生成（72小節、8セクション）
- KPI検証済み品質保証
- 学習済みモデル（35,511レコード）の実戦投入

**技術的ハイライト**:
- 学習フェーズ（E-GMD統合+ML学習）完了
- 生成フェーズ（SongPackage基盤）実装完了
- Recommender/Generator連携準備完了
- 既存楽曲構成（sections.json, chordmap.json）活用
- bars.parquet（小節単位の目標値）自動生成

---

**Date**: 2025-01-XX  
**Status**: ✅ SongPackage基盤実装完了、Recommender/Generator連携準備完了  
**Next**: Recommender実装 → Generator実装 → KPI Gate実装
