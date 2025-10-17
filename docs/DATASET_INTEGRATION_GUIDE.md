# Dataset Integration Guide

**目的:** Suno合成に頼らず、既存の高品質データセットで不足奏法を補完

**戦略:** Manifest分析で特定された7,888ファイルの不足を、外部データセット統合で効率的に埋める

---

## 📊 Critical Gaps (優先度順)

Manifest分析から特定された主要な不足奏法:

| 優先度 | 奏法 | 不足数 | 推奨データセット | ステータス |
|-------|------|-------|----------------|----------|
| 🔴 **1** | **strings_legato** | 1,117 | URMP, MAESTRO | ⏸️ 調査中 |
| 🔴 **2** | **guitar_strum** | 1,554 | GuitarSet, MusicNet | ⏸️ 調査中 |
| 🟡 **3** | guitar_arpeggio | 1,007 | GuitarSet, SMD | ⏸️ 調査中 |
| 🟡 **4** | bass_pick | 900 | PHENICX, SMD | ⏸️ 調査中 |
| 🟢 **5** | strings_spiccato | 600 | URMP | ⏸️ 調査中 |
| 🟢 **6** | piano_pop_comping | 400 | Lakh MIDI, SMD | ⏸️ 調査中 |

**Total Target:** 既存データセットで3,000-4,000ファイル補完 → 合成データは残り4,000-5,000に削減

---

## 🎯 推奨データセット (楽器別)

### 1. Guitar Datasets

#### **GuitarSet** (最優先)
- **URL:** https://github.com/marl/guitarset
- **概要:** 360曲、6人のギタリスト、アノテーション付き
- **内容:**
  - MIDI + Audio + Annotations (note-level)
  - 奏法アノテーション: picking, hammer-on, pull-off, slide
  - 多様なジャンル: Rock, Jazz, Bossa Nova, Funk
- **Target:**
  - ✅ guitar_strum: ~150ファイル推定
  - ✅ guitar_arpeggio: ~80ファイル推定
  - ✅ guitar_fingerpicking: ~130ファイル推定
- **ダウンロード:**
  ```bash
  cd data/
  git clone https://github.com/marl/guitarset.git
  ```
- **統合方法:** `scripts/import_guitarset.py`（作成予定）

#### **MusicNet**
- **URL:** https://zenodo.org/record/5120004
- **概要:** 330クラシック楽曲、MIDI + Audio
- **Target:** guitar_arpeggio (クラシックギター編曲)
- **ファイル数:** ~50-100推定

---

### 2. Strings Datasets

#### **URMP (University of Rochester Multi-Modal Performance)** (最優先)
- **URL:** http://www2.ece.rochester.edu/projects/air/projects/URMP.html
- **概要:** 44曲、室内楽アンサンブル、楽器別分離済み
- **内容:**
  - Violin, Viola, Cello, Double Bass tracks
  - MIDI + Audio (高品質録音)
  - アノテーション: bowings, articulations
- **Target:**
  - ✅ strings_legato: ~200ファイル推定（最優先）
  - ✅ strings_spiccato: ~150ファイル推定
  - ✅ strings_staccato: ~100ファイル推定
- **ダウンロード:**
  ```bash
  cd data/
  wget http://www2.ece.rochester.edu/projects/air/resource/urmp_dataset.tar.gz
  tar -xzf urmp_dataset.tar.gz
  ```
- **統合方法:** `scripts/import_urmp.py`（作成予定）

#### **MAESTRO (Piano & Strings)**
- **URL:** https://magenta.tensorflow.org/datasets/maestro
- **概要:** 200時間以上のクラシックピアノ演奏
- **内容:** MIDI + Audio、国際ピアノコンクール録音
- **Target:**
  - strings_legato: ~50ファイル（オーケストラ編曲版）
  - piano_ballad: ~200ファイル
- **ダウンロード:**
  ```bash
  cd data/
  wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
  unzip maestro-v3.0.0-midi.zip
  ```

---

### 3. Bass Datasets

#### **PHENICX (Symphonic Orchestra)**
- **URL:** https://phenicx.upf.edu/
- **概要:** オーケストラ演奏、楽器別分離
- **Target:**
  - bass_pick: ~100ファイル（コントラバス）
  - bass_walking: ~80ファイル
- **特記:** クラシック中心、ジャズスタイルは少ない

#### **SMD (Synthetic MIDI Dataset)**
- **URL:** https://github.com/bytedance/SMD
- **概要:** 約15万MIDI、ジャンル多様
- **Target:**
  - bass_pick: ~300ファイル
  - bass_slap: ~100ファイル
  - bass_walking: ~150ファイル

---

### 4. Piano Datasets

#### **Lakh MIDI Dataset**
- **URL:** https://colinraffel.com/projects/lmd/
- **概要:** 176,581 MIDI files
- **Target:**
  - piano_pop_comping: ~400ファイル
  - piano_jazz_voicing: ~200ファイル
- **注意:** 品質バラつき大 → Stage2フィルタリング必須

#### **PiJAMA (Piano Jazz)**
- **URL:** https://github.com/SonyCSLParis/PiJAMA
- **概要:** ジャズピアノ演奏、アノテーション付き
- **Target:**
  - piano_jazz_voicing: ~150ファイル
  - piano_comping: ~100ファイル

---

## 🔧 統合ワークフロー

### Phase 1: ダウンロード & 前処理

```bash
# 1. データセットダウンロード
bash scripts/download_external_datasets.sh

# 2. 統一フォーマット変換 (→ data/external/{dataset}/raw/)
python scripts/import_guitarset.py
python scripts/import_urmp.py
python scripts/import_maestro.py
```

### Phase 2: Stage1 統合

```bash
# 3. 既存のStage1パイプラインに統合
# MULTI_DATASET_RUNNER_GUIDE.md の DATASETS テーブルに追加

DATASETS="$(cat <<'EOF'
# 既存データセット
POP909   melody      data/POP909                     output/pop909/clean/melody       ...
SLAKH    guitar      data/slakh2100_midi/guitar      output/slakh/clean/guitar        ...

# 新規外部データセット
GUITARSET guitar     data/external/guitarset/raw     output/guitarset/clean/guitar    output/guitarset/quarantine/guitar  output/guitarset/shards/guitar
URMP     strings     data/external/urmp/raw          output/urmp/clean/strings        output/urmp/quarantine/strings      output/urmp/shards/strings
MAESTRO  piano       data/external/maestro/raw       output/maestro/clean/piano       output/maestro/quarantine/piano     output/maestro/shards/piano
SMD      bass        data/external/smd/raw/bass      output/smd/clean/bass            output/smd/quarantine/bass          output/smd/shards/bass
EOF
)"

# 4. Stage1実行
bash scripts/run_stage1_clean_multi.sh
```

### Phase 3: Stage2 スコアリング & 選抜

```bash
# 5. 外部データセットもStage2適用（同一メトリクス）
bash scripts/run_stage2_multi.sh

# 6. 統合分布再分析
python scripts/generate_distribution_counts.py \
  --results-dir output/test_results \
  --include-external \
  --output reports/integrated_distribution_with_external.json
```

### Phase 4: Gap再評価

```bash
# 7. 外部データセット統合後のGap再計算
python scripts/estimate_gaps_and_emit_manifest.py \
  --targets configs/targets_hybrid.yaml \
  --current reports/integrated_distribution_with_external.json \
  --out manifests/manifest_after_external.jsonl

# 8. 削減確認
python scripts/analyze_manifest.py manifests/manifest_after_external.jsonl
# Expected: 7,888 → 4,000-5,000に削減
```

---

## 📋 データセット評価基準

外部データセット取り込み前の品質確認:

### 必須条件
- ✅ **MIDI形式** または MIDI変換可能
- ✅ **Stem分離済み** または楽器別Track明示
- ✅ **ライセンス:** 商用利用可能 or 学術利用可能
- ✅ **最低ファイル数:** 50件以上

### 推奨条件
- 🟢 奏法アノテーション付き (GuitarSet, URMP)
- 🟢 高品質録音由来のMIDI (MAESTRO)
- 🟢 多様なジャンル (SMD, Lakh)
- 🟢 既存Stage2メトリクスで評価可能

### 除外基準
- ❌ Stem分離なし（Los-Angeles-MIDIなど）
- ❌ 低品質・エラー率高（個人アップロードサイト）
- ❌ ライセンス不明確
- ❌ MIDIファイルが極端に短い（<4秒）

---

## 🎯 統合目標

### 短期目標 (Phase 1-2)

| データセット | 楽器 | 推定取得数 | 優先度 |
|------------|------|----------|-------|
| GuitarSet | guitar | 360 | 🔴 High |
| URMP | strings | 350 | 🔴 High |
| MAESTRO | piano | 200 | 🟡 Medium |
| SMD | bass | 400 | 🟡 Medium |

**Total:** ~1,310ファイル

### 中期目標 (Phase 3-4)

| データセット | 楽器 | 推定取得数 | 優先度 |
|------------|------|----------|-------|
| MusicNet | guitar | 100 | 🟢 Low |
| Lakh MIDI | piano | 600 | 🟢 Low |
| PHENICX | bass | 150 | 🟢 Low |

**Total:** ~850ファイル

### 統合後の予想分布

```
Real Data (Stage2 pass): 3,559
External Datasets:       2,160 (GuitarSet + URMP + MAESTRO + SMD + others)
-------------------------------------------
Total Real/External:     5,719

Remaining Synthetic:     7,888 - 2,160 = 5,728
```

**戦略転換効果:**
- ✅ 合成データ依存度: 100% → 73% (5,728/7,888)
- ✅ 高品質Real Data比率向上: 62% → 74% (5,719/7,719)
- ✅ 特にstrings_legato/guitar_strumの不足を大幅改善

---

## 🚀 次のステップ

### Immediate (今週)
1. ✅ このガイド作成 (完了)
2. ⏸️ GuitarSet ダウンロード & 前処理スクリプト作成
3. ⏸️ URMP ダウンロード & 前処理スクリプト作成

### Short-term (来週)
4. ⏸️ GuitarSet/URMP を Stage1/Stage2 パイプラインに統合
5. ⏸️ 統合後のGap再評価
6. ⏸️ Manifest更新 (manifest_after_external.jsonl)

### Mid-term (今月)
7. ⏸️ MAESTRO/SMD 統合
8. ⏸️ 完全統合後のTraining Dataset準備
9. ⏸️ Hybrid Data Strategy完成 (Real + External + Synthetic)

---

## 📚 参考資料

### Dataset Papers
- **GuitarSet:** Xi et al., "GuitarSet: A Dataset for Guitar Transcription", ISMIR 2018
- **URMP:** Li et al., "URMP: A Dataset for Multi-Modal Music Performance", ICASSP 2019
- **MAESTRO:** Hawthorne et al., "Enabling Factorized Piano Music Modeling", ISMIR 2019

### Related Work
- Stem分離: Slakh2100, MedleyDB
- 奏法アノテーション: NSynth, RWC Music Database
- ジャンル多様性: Lakh MIDI, Million Song Dataset

---

## ⚠️ 注意事項

### ライセンス確認
- **必須:** 各データセット使用前にライセンス確認
- **推奨:** 学術利用許諾を明示的に取得
- **禁止:** 商用利用不可データセットの誤用

### 品質管理
- **Stage2フィルタリング:** 外部データも同一基準で評価
- **閾値:** Real+5%の品質ゲート適用
- **Quarantine:** 不適格ファイルは適切に隔離

### データ管理
- **バージョン管理:** データセットバージョンを記録
- **トレーサビリティ:** 各MIDIファイルの出所を追跡可能に
- **バックアップ:** 前処理前の生データを保存

---

**最終更新:** 2025年10月17日  
**ステータス:** Phase 1 準備完了、実装開始待ち
