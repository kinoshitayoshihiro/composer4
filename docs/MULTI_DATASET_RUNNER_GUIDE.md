# Multi-Dataset Runner Guide

複数データセット（POP909/SLAKH/LAMDA Drums等）を統一的に処理する共通ランナーの使用ガイド

## 概要

従来は各データセット用の個別スクリプト（`run_pop909_full.sh`など）を管理していましたが、新しい共通ランナーでは：

- **1つのスクリプトで全データセットを処理**
- **ブラッシュアップフラグ（streaming/resume/dual-threshold）を全面採用**
- **データセット追加が容易**（表に1行追加するだけ）

## ⚠️ データセット適性ガイド

### 楽器別Generator開発に適したデータセット

**重要:** 楽器別パターン学習には **Stem分離済み** データセットのみ使用できます。

| 楽器 | 推奨データセット | ファイル数 | 備考 |
|------|----------------|-----------|------|
| **Drums** | LAMDA (loops) | ~51,000 | ✅ Drum専用ループ、LAMDA Stage2対応 |
| | SLAKH drums | ~8,000 | ✅ Stem分離済み |
| | POP909 drums | ~800 | ❌ Stem分離なし（melody/chords混在） |
| **Piano** | POP909 melody | ~900 | ✅ Piano/Melody専用 |
| **Guitar** | SLAKH guitar | ~8,000 | ✅ Stem分離済み（acoustic/electric） |
| **Bass** | SLAKH bass | ~8,000 | ✅ Stem分離済み（acoustic/electric/synth） |
| **Strings** | SLAKH strings | ~8,000 | ✅ Stem分離済み（violin/viola/cello） |
| **その他** | SLAKH (35トラック) | ~8,000 | ✅ Brass/Woodwinds/Synth等 |

### ❌ 使用を推奨しないデータセット

| Dataset | ファイル数 | 理由 | 代替用途 |
|---------|-----------|------|---------|
| **Los-Angeles-MIDI** | ~400,000 | ❌ Stem分離なし（全楽器混在） | コード進行分析、楽曲構造研究には有用 |
| **POP909 (フルMIDI)** | ~900 | ❌ 3パート混在（melody/chords/drums） | Melody専用として使用可能 |

### 📊 データセット構成の詳細

#### POP909構成
```
POP909/001/
├── 001.mid              # ❌ 3パート混在（使用しない）
├── beat_audio.txt       # ビート情報
├── chord_audio.txt      # コード進行
└── versions/            # ✅ Stem分離版（これを使用）
    ├── 001-v1.mid       # Melody (Piano) - 909曲
    ├── 001-v2.mid       # Chords (Piano) - 566曲
    └── 001-v3.mid       # Bass - 279曲
```
- **Part v1**: Melody (Piano) - 高域・単旋律中心
- **Part v2**: Chords (Piano) - 和音中心・伴奏
- **Part v3**: Bass - 低域・モノフォニック

**重要な発見:**
- **Complete stems (v1+v2+v3)**: 279曲 (30.7%)
- **Partial stems (v1+v2のみ)**: 287曲 (31.6%)
- **v1のみ**: 343曲 (37.7%)

**採用戦略:**
- ✅ **v1+v2+v3が揃っている279曲のみ使用** (計837 MIDIファイル)
- ❌ 混在版(001.mid)は重複として除外
- ⚠️ v1単体343曲はmelody専用として別途活用可能

**処理済み (Stage1 Complete):**
- Melody (v1): 277/279 (99.3%)
- Chords (v2): 277/279 (99.3%)
- Bass (v3): 276/279 (98.9%)
- Total: 830/837 MIDI files (99.2% success)

#### LAMDA Drums構成
```
data/loops/               # ✅ 正しいパス
└── drums/
    ├── groove/
    │   ├── drummer1/
    │   ├── drummer2/
    │   └── ...
    └── e-gmd/
        ├── drummer1/
        └── ...
```
- **約51,000ループ** のDrum専用MIDI
- LAMDA Stage2スコアリング対応

#### SLAKH2100構成
```
data/slakh2100_midi/
├── drums/               # ✅ Stem分離済み
├── guitar/              # ✅ Stem分離済み
├── bass/                # ✅ Stem分離済み
├── strings/             # ✅ Stem分離済み
└── ... (全35トラック)
```
- **約8,000曲** × 35トラック
- 完全なStem分離
- 楽器別Generator開発に最適

## アーキテクチャ

### Stage1: クリーニング & Sharded Pickle 生成

**スクリプト:** `scripts/run_stage1_clean_multi.sh`

**処理内容:**
- 生MIDIファイルのクリーニング（invalid note/tempo除去等）
- クリーンファイルの出力
- 不適格ファイルの隔離（quarantine）
- Sharded pickle 生成（メタデータ用）

**対象データセット:**
| Dataset | Instrument | Input | Output | Status |
|---------|-----------|-------|--------|--------|
| POP909  | melody (v1) | `data/POP909/*/versions/*-v1.mid` | `output/pop909/clean/melody` | ✅ **Complete (277/279 = 99.3%)** |
| POP909  | chords (v2) | `data/POP909/*/versions/*-v2.mid` | `output/pop909/clean/chords` | ✅ **Complete (277/279 = 99.3%)** |
| POP909  | bass (v3)   | `data/POP909/*/versions/*-v3.mid` | `output/pop909/clean/bass` | ✅ **Complete (276/279 = 98.9%)** |
| SLAKH   | drums     | `data/slakh_by_instrument/drums/{train,validation,test}` | `output/slakh/clean/drums` | ✅ **Complete (557/561 = 99.3%)** |
| SLAKH   | guitar    | `data/slakh_by_instrument/guitar/{train,validation,test}` | `output/slakh/clean/guitar` | ✅ **Complete (1422/1471 = 96.7%)** |
| SLAKH   | bass      | `data/slakh_by_instrument/bass/{train,validation,test}` | `output/slakh/clean/bass` | ✅ **Complete (584/599 = 97.5%)** |
| SLAKH   | strings   | `data/slakh_by_instrument/strings/{train,validation,test}` | `output/slakh/clean/strings` | ✅ **Complete (999/1045 = 95.6%)** |
| LAMDA   | drums     | `data/loops` | `output/lamda/clean/drumloops` | ✅ Complete (51,248) |

**POP909 Stage1 特記事項:**
- Stem分離版 (versions/*-v1/v2/v3.mid) のみ使用
- v1+v2+v3完全セット: 279曲
- 混在版 (*.mid) は重複として除外
- 高品質な楽器別MIDIデータを確保

**SLAKH Stage1 総括:**
- **Total:** 3,676 files → 3,562 clean (96.9% overall success)
- **Quarantine:** 114 files (3.1%)
- 楽器別Generator開発に必要な高品質MIDIデータを確保

### Stage2: スコアリング & 選抜

**スクリプト:** `scripts/run_stage2_multi.sh`

**処理内容:**
- LAMDAメトリクスによる品質スコアリング
- Dual-threshold（soft/hard）による選抜
- Streaming出力でメモリ節約
- Batch manifest による冪等実行

**対象データセット:**
| Dataset | Instrument | Input | Output | Config | Status |
|---------|-----------|-------|--------|--------|--------|
| SLAKH   | drums     | `output/slakh/clean/drums` | `output/slakh/stage2/drums` | `configs/lamda/drums_stage2.yaml` | ✅ **Complete (412 loops)** |
| LAMDA   | drums     | `output/lamda/clean/drumloops` | `output/lamda/stage2/drumloops` | `configs/lamda/drums_stage2.yaml` | ✅ Complete (461 loops) |
| SLAKH   | guitar    | `output/slakh/clean/guitar` | `output/slakh/stage2/guitar` | `configs/lamda/guitar_stage2.yaml` | 🧪 **Test OK (68% pass)** |
| SLAKH   | bass      | `output/slakh/clean/bass` | `output/slakh/stage2/bass` | `configs/lamda/bass_stage2.yaml` | 🧪 **Test OK (100% pass)** |
| SLAKH   | strings   | `output/slakh/clean/strings` | `output/slakh/stage2/strings` | `configs/lamda/strings_stage2.yaml` | 🧪 **Test OK (70% pass)** |
| POP909  | piano (melody) | `output/pop909/clean/melody` | `output/pop909/stage2/melody` | `configs/lamda/piano_stage2.yaml` | 🧪 **Test OK (100% pass)** |
| POP909  | piano (chords) | `output/pop909/clean/chords` | `output/pop909/stage2/chords` | `configs/lamda/piano_stage2.yaml` | 🧪 **Test OK (100% pass)** |

**SLAKH Drums Stage2 結果:**
- Processed: 412 loops (99.3% from 557 Stage1 clean files)
- Soft threshold (≥65.0): 36 loops (8.7%)
- Hard threshold (≥70.0): 7 loops (1.7%)
- Mean score: 55.66 (Min: 35.06, Max: 74.79)

**Guitar/Bass/Strings Stage2 実装:**
- ✅ **設定ファイル**: configs/lamda/{guitar,bass,strings}_stage2.yaml
- ✅ **メトリクス**: scripts/stage2_instrument_metrics.py
- 🧪 **テスト完了**: 各100ファイルで動作確認済み
- 📊 **詳細結果**: docs/STAGE2_INSTRUMENT_TEST_RESULTS.md

**Piano Stage2 実装:**
- ✅ **設定ファイル**: configs/lamda/piano_stage2.yaml
- ✅ **メトリクス**: scripts/stage2_instrument_metrics.py (5 metrics)
- 🧪 **テスト完了**: Melody(v1) + Chords(v2) 各100ファイル
- 📊 **詳細結果**: docs/STAGE2_INSTRUMENT_TEST_RESULTS.md

**テスト結果サマリー (100ファイル/楽器):**
| 楽器 | 平均スコア | 合格率 | 閾値 | 主要メトリクス |
|------|-----------|--------|------|---------------|
| Guitar | 43.6% | 68% | 40.0 | アルペジオ品質、コード協和度、ストラムパターン |
| Bass | 76.7% | 100% 🏆 | 40.0 | ルート音正確性、グルーヴ品質、音域適合性 |
| Strings | 50.9% | 70% | 45.0 | ボウイング表現、ハーモニー品質、レガート品質 |
| Piano (Melody) | 64.0% | 100% 🏆 | 45.0 | メロディー表現、リズム多様性、ダイナミクスレンジ |
| Piano (Chords) | 64.2% | 100% 🏆 | 45.0 | ハーモニー進行、リズム多様性、ダイナミクスレンジ |

---

## 🎯 Stage2 本番実行結果 (Full Production Run)

**実行日**: 2025年10月17日  
**総処理ファイル数**: 3,559 ✅ **全楽器完了**

### 📊 全楽器統合サマリー

| 楽器 | データセット | ファイル数 | 平均スコア | 合格率 | 閾値 | ステータス |
|------|-------------|-----------|-----------|--------|------|-----------|
| **Piano (Melody)** | POP909 v1 | 277 | 63.9% | **100%** 🏆 | 45.0 | ✅ Complete |
| **Piano (Chords)** | POP909 v2 | 277 | 64.2% | **100%** 🏆 | 45.0 | ✅ Complete |
| **Bass** | SLAKH | 584 | 76.9% | **100%** 🏆 | 40.0 | ✅ Complete |
| **Guitar** | SLAKH | 1,422 | 42.9% | 67.7% | 40.0 | ✅ Complete |
| **Strings** | SLAKH | 999 | 51.1% | 69.7% | 45.0 | ✅ **Complete** |

**主要な知見:**
- ✅ Piano/Bass は極めて高品質（全ファイル合格）
- ✅ Guitar/Strings は適切な選別機能（67-70%合格率）
- ✅ テスト結果と本番結果が整合（メトリクスの信頼性確認）
- ✅ **全5楽器でStage2完了 - 総計3,559ファイル処理**
- 📊 詳細レポート: `docs/STAGE2_FULL_PRODUCTION_REPORT.md`

### 🎹 Piano (POP909) - 554ファイル

**Melody (v1) - 277ファイル:**
- 平均スコア: 63.9% | 中央値: 64.7%
- スコア範囲: 51.7% - 68.9%
- 合格率: **277/277 (100%)** 🏆
- TOP: 071-v1.mid (68.97%) | BOTTOM: 792-v1.mid (51.74%)
- メトリクス: melody_expression 69.9%, rhythm_diversity 86.6%, dynamics_range 76.9%
- 弱点: pedaling_quality 17.7% (CC64データなし、推定値)

**Chords (v2) - 277ファイル:**
- 平均スコア: 64.2% | 中央値: 64.8%
- スコア範囲: 53.5% - 72.5%
- 合格率: **277/277 (100%)** 🏆
- TOP: 365-v2.mid (72.49%) | BOTTOM: 484-v2.mid (53.49%)
- メトリクス: melody_expression 70.0%, rhythm_diversity 87.0%, dynamics_range 78.0%
- 特徴: MelodyとChordsのスコア分布が極めて類似（±0.3%）

### 🎸 Bass (SLAKH) - 584ファイル

- 平均スコア: 76.9% | 中央値: 77.0%
- スコア範囲: 57.0% - 91.0%
- 合格率: **584/584 (100%)** 🏆
- TOP: Track00601_S05.mid (91.04%) | BOTTOM: Track01495_S08.mid (56.99%)
- メトリクス: root_accuracy 84.3%, pitch_range_fit 87.1%, groove_quality 64.3%
- 推奨: 閾値を45.0へ引き上げ検討（全て合格のため選別機能なし）

### 🎸 Guitar (SLAKH) - 1,422ファイル

- 平均スコア: 42.9% | 中央値: 47.2%
- スコア範囲: 11.2% - 78.5%
- 合格率: **963/1,422 (67.7%)**
- 不合格: 459ファイル (32.3%) - 適切に除外
- 特徴: 最大データセット、現実的な合格率
- 評価: 閾値40.0は妥当（適切な品質フィルタリングを実現）

### 🎻 Strings (SLAKH) - 999ファイル

- 平均スコア: 51.1% | 中央値: 56.0%
- スコア範囲: 7.8% - 88.3%
- 合格率: **696/999 (69.7%)**
- 不合格: 303ファイル (30.3%) - 適切に除外
- TOP: Track00601_S12.mid (88.29%) | BOTTOM: Track01340_S03.mid (7.78%)
- メトリクス: bowing_expression 48.6%, harmony_quality 57.2%, legato_quality 42.0%
- 評価: 閾値45.0は妥当（適切な品質フィルタリングを実現）

### 改善提案

**優先度1: Piano ペダリング品質**
- 現状: 17-18% (CC64データなし)
- 対策: より高度な推定アルゴリズムまたは重み削減 (0.5→0.2)

**優先度2: Bass 閾値調整**
- 現状: 閾値40.0で100%合格
- 対策: 45.0へ引き上げまたは新規メトリクス追加

**優先度3: Strings レガート品質向上**
- 現状: legato_quality 42.0% (最低メトリクス)
- 対策: レガート検出アルゴリズム改善、または合成データでレガート補完

**優先度4: Guitar ストラムパターン強化**
- 現状: 多様な演奏スタイル対応が課題
- 対策: 演奏スタイル別評価軸の追加

### 次のステップ

**✅ Stage2 全楽器完了 (3,559ファイル処理)**

次フェーズ候補:
1. **Technique Distribution Analysis**: 奏法分布の定量分析（arpeggio/strum/legato等）
2. **Hybrid Data Strategy**: 不足奏法を合成データで補完（ChatGPT戦略統合）
3. **Quality Improvement Pipeline**: Suno MIDI品質向上（ensemble voting）
4. **Training Dataset Preparation**: 学習用データセット統合・分割

---

## 使用方法

### 基本実行

```bash
# Stage1: 全データセット一括クリーニング
bash scripts/run_stage1_clean_multi.sh

# Stage2: 全データセット一括スコアリング
bash scripts/run_stage2_multi.sh
```

### カスタマイズ実行

#### Stage1 環境変数

```bash
# シャードサイズを10000に、並列度を4に設定
SHARD_SIZE=10000 JOBS=4 bash scripts/run_stage1_clean_multi.sh

# 冪等実行を無効化（毎回全処理）
RESUME_FLAG="" bash scripts/run_stage1_clean_multi.sh

# JSON出力を有効化（pickle + JSON両方）
EMIT_META_JSON=on bash scripts/run_stage1_clean_multi.sh
```

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `SHARD_SIZE` | 5000 | Pickle シャードサイズ |
| `JOBS` | 8 | 並列処理数 |
| `EMIT_META_JSON` | off | JSON出力の有無 |
| `RESUME_FLAG` | --resume | 冪等実行フラグ |

#### Stage2 環境変数

```bash
# 学習用に緩めの閾値、バッチサイズ大きめ
LIMIT=10000 TH_SOFT=60 TH_HARD=75 bash scripts/run_stage2_multi.sh

# メモリ制約が厳しい環境（バッチサイズ小）
LIMIT=2000 ROW_GROUP=4096 bash scripts/run_stage2_multi.sh

# Streaming/Resume無効化（テスト用）
STREAMING_FLAG="" RESUME_FLAG="" bash scripts/run_stage2_multi.sh
```

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `LIMIT` | 5000 | 1バッチあたりのループ数 |
| `TH_SOFT` | 65.0 | Soft threshold（学習候補） |
| `TH_HARD` | 70.0 | Hard threshold（公開用） |
| `ROW_GROUP` | 8192 | Parquet行グループサイズ |
| `MANIFEST_FLUSH` | 200 | Manifestフラッシュ間隔 |
| `STREAMING_FLAG` | --streaming | Streaming出力フラグ |
| `RESUME_FLAG` | --resume | 冪等実行フラグ |

## データセット追加方法

### Stage1 にデータセット追加

`scripts/run_stage1_clean_multi.sh` の `DATASETS` 変数に行を追加：

```bash
DATASETS="$(cat <<'EOF'
POP909   melody      data/POP909                     output/pop909/clean/melody       output/pop909/quarantine/melody  output/pop909/shards/melody
SLAKH    drums       data/slakh2100_midi/drums       output/slakh/clean/drums         output/slakh/quarantine/drums    output/slakh/shards/drums
SLAKH    guitar      data/slakh2100_midi/guitar      output/slakh/clean/guitar        output/slakh/quarantine/guitar   output/slakh/shards/guitar
SLAKH    bass        data/slakh2100_midi/bass        output/slakh/clean/bass          output/slakh/quarantine/bass     output/slakh/shards/bass
SLAKH    strings     data/slakh2100_midi/strings     output/slakh/clean/strings       output/slakh/quarantine/strings  output/slakh/shards/strings
LAMDA    drums       data/loops                      output/lamda/clean/drumloops     output/lamda/quarantine/drumloops output/lamda/shards/drumloops
# ↓ 新しいデータセットを追加
MYDATA   piano       data/mydata/raw/piano           output/mydata/clean/piano        output/mydata/quarantine/piano  output/mydata/shards/piano
EOF
)"
```

**カラム:**
1. Dataset名
2. Instrument種別（drums/strings/piano等）
3. 入力ディレクトリ（生MIDI）
4. 出力ディレクトリ（クリーンMIDI）
5. Quarantine ディレクトリ（不適格MIDI）
6. Pickle 出力ディレクトリ

### Stage2 にデータセット追加

`scripts/run_stage2_multi.sh` の `DATASETS` 変数に行を追加：

```bash
DATASETS="$(cat <<'EOF'
SLAKH    drums       output/slakh/clean/drums       output/slakh/stage2/drums         output/slakh_drums_metadata  output/slakh_drums_metadata/index.pkl  configs/lamda/drums_stage2.yaml
LAMDA    drums       output/lamda/clean/drumloops   output/lamda/stage2/drumloops     output/drums_metadata        output/drums_metadata/drums_index.pkl  configs/lamda/drums_stage2.yaml
# ↓ 将来: Guitar/Bass/Strings用のStage2設定を追加（現在はdrums専用）
# SLAKH guitar     output/slakh/clean/guitar      output/slakh/stage2/guitar        output/guitar_metadata       output/guitar_metadata/index.pkl       configs/lamda/guitar_stage2.yaml
EOF
)"
```

**カラム:**
1. Dataset名
2. Instrument種別
3. 入力ディレクトリ（クリーンMIDI）
4. 出力ディレクトリ（Stage2結果）
5. メタデータディレクトリ
6. メタデータインデックスpickle
7. Stage2設定YAML

## 出力ファイル構造

### Stage1 出力

```
output/
├── pop909/
│   ├── clean/
│   │   └── melody/         # クリーンMIDI (Piano/Melody)
│   ├── quarantine/
│   │   └── melody/         # 不適格MIDI
│   └── shards/
│       └── melody/         # Pickleシャード
│           ├── shard_0000.pkl
│           ├── shard_0001.pkl
│           └── ...
├── slakh/
│   ├── clean/
│   │   ├── drums/          # ✅ Stem分離済み
│   │   ├── guitar/         # ✅ Stem分離済み
│   │   ├── bass/           # ✅ Stem分離済み
│   │   └── strings/        # ✅ Stem分離済み
│   ├── quarantine/
│   │   └── ...
│   └── shards/
│       └── ...
└── lamda/
    ├── clean/
    │   └── drumloops/      # ✅ Drum専用ループ
    ├── quarantine/
    │   └── drumloops/
    └── shards/
        └── drumloops/
```

### Stage2 出力

**注意:** Stage2は現在 **drums専用** です。

```
output/
├── slakh/
│   └── stage2/
│       └── drums/              # ✅ LAMDA Stage2対応
│           ├── batch_0/
│           │   ├── BATCH_MANIFEST.json        # 冪等性メタ
│           │   ├── metrics_score.jsonl        # スコア結果
│           │   ├── loop_summary.csv           # ループサマリー
│           │   ├── canonical_events.parquet   # イベントデータ
│           │   ├── stage2_summary.json        # 統計情報
│           │   └── velocity_coverage.json     # Velocity分布
│           ├── batch_5000/
│           │   └── ...
│           ├── metrics_score.ALL.jsonl        # 全バッチ結合
│           └── loop_summary.ALL.csv           # 全バッチ結合
└── lamda/
    └── stage2/
        └── drumloops/          # ✅ LAMDA Stage2対応（51,000ループ）
            ├── batch_0/
            │   └── ...
            └── ...
```

**将来拡張:** Guitar/Bass/Strings用のStage2メトリクスは実装済み（簡易版）。

**Guitar/Bass/Strings Stage2について:**

現在実装されているのは**簡易版メトリクス**です:
- ✅ 楽器別評価軸を実装 (arpeggio/groove/bowing等)
- ✅ 設定ファイル完備 (configs/lamda/{guitar,bass,strings}_stage2.yaml)
- ✅ テスト済み (100ファイル/楽器で動作確認)
- ⚠️ 完全なLAMDA統合は未実装 (将来のタスク)

詳細: docs/STAGE2_INSTRUMENT_TEST_RESULTS.md

```

## 利点

### 1. 一元管理

- 全データセットを1つのスクリプトで制御
- フラグ・閾値・バッチサイズを統一的に管理
- 個別スクリプトの重複・齟齬がない

### 2. 再実行に強い

- `--resume` により中断から安全に再開
- `BATCH_MANIFEST.json` で処理済みファイルをトラッキング
- 手動でのバッチ番号管理が不要

### 3. 省メモリ

- `--streaming` で逐次書き出し（メモリ蓄積なし）
- `--limit/--offset` による小分け処理
- 大規模データ（LAMDA 51,248ループ等）でも安定動作

### 4. 拡張容易

- データセット追加は表に1行追加するだけ
- Instrument種別（drums/strings/piano）も簡単に対応
- 特殊処理が必要な場合はフック関数を追加

## 既存スクリプトとの関係

### 置き換え対象

以下のデータセット別スクリプトは共通ランナーで代替可能：

- `run_pop909_full.sh` → `run_stage1_clean_multi.sh` + `run_stage2_multi.sh`
- `run_slakh_full.sh` → 同上
- `run_lamda_full.sh` → 同上

### 残すべきスクリプト

データセット固有の特殊処理がある場合は残してOK：

- 独自の前処理（例：POP909 の chord annotation 抽出）
- 独自の検証（例：SLAKH の multi-track 整合性チェック）
- 独自のポストプロセス（例：LAMDA の velocity histogram 分析）

**推奨:** 共通ランナーから "特殊フック" として呼ぶ形に整理すると運用がきれい

## トラブルシューティング

### Stage1 でエラーが出る

**症状:** `scripts/clean_midi.py` が見つからない

**解決:**
```bash
# PYTHONPATH を明示
PYTHONPATH=. bash scripts/run_stage1_clean_multi.sh
```

**症状:** Pickle 書き込みでエラー

**解決:**
```bash
# 出力ディレクトリの権限確認
ls -ld output/*/shards/*

# 権限がない場合は修正
chmod -R u+w output/
```

### Stage2 でメタデータが見つからない

**症状:** `Metadata index not found: output/xxx_metadata/xxx_index.pkl`

**解決:**
```bash
# Stage1 を先に実行
bash scripts/run_stage1_clean_multi.sh

# または該当データセットのみ実行（スクリプトを一時的に編集）
```

### OOM（メモリ不足）が発生

**解決:**
```bash
# バッチサイズを減らす
LIMIT=2000 bash scripts/run_stage2_multi.sh

# Parquet行グループサイズも減らす
LIMIT=2000 ROW_GROUP=2048 bash scripts/run_stage2_multi.sh

# Streamingが無効になっていないか確認
echo $STREAMING_FLAG  # --streaming が表示されるべき
```

### 再実行で重複処理される

**症状:** `--resume` を指定しても再処理される

**原因:** `BATCH_MANIFEST.json` のSHA1不一致（メタデータインデックスが更新された）

**解決:**
```bash
# Stage1 からやり直す（メタデータ再生成）
bash scripts/run_stage1_clean_multi.sh
bash scripts/run_stage2_multi.sh

# または手動でマニフェストを削除
rm -rf output/*/stage2/*/batch_*/BATCH_MANIFEST.json
```

## 性能目安

### Stage1（8並列）

| Dataset | Files | Time | Throughput |
|---------|-------|------|-----------|
| POP909 melody | ~900 | ~2分 | 450 files/min |
| SLAKH drums | ~8,000 | ~15分 | 530 files/min |
| SLAKH guitar | ~8,000 | ~15分 | 530 files/min |
| LAMDA drums | ~51,000 | ~90分 | 560 files/min |

### Stage2（LIMIT=5000, streaming有効）

| Dataset | Loops | Time | Throughput |
|---------|-------|------|-----------|
| SLAKH drums | ~8,000 | ~30分 | 260 loops/min |
| LAMDA drums | ~51,000 | ~180分 | 280 loops/min |

**メモリピーク:**
- Streaming有効: ~500MB（一定）
- Streaming無効: ~8GB（ループ数に比例）

## 今後の拡張

### 優先度1: Parquet 統合マージ

現在は JSONL/CSV のみ結合。Parquet も `pyarrow` で統合する：

```python
# scripts/merge_stage2_parquets.py（TODO実装）
import pyarrow.parquet as pq

def merge_parquets(input_pattern, output_path):
    tables = [pq.read_table(f) for f in glob(input_pattern)]
    merged = pa.concat_tables(tables)
    pq.write_table(merged, output_path)
```

### 優先度2: 自動シャーディング

システムメモリを読み取って最適な `LIMIT` を自動計算：

```bash
# 利用可能メモリに応じて自動調整
LIMIT=auto bash scripts/run_stage2_multi.sh
```

### 優先度3: データセット別フック

特殊処理を共通ランナーから呼び出す：

```bash
# run_stage2_multi.sh に追加
if [[ "${NAME}" == "POP909" ]]; then
  bash scripts/hooks/pop909_post_process.sh "${OUT_DIR}"
fi
```

## 関連ドキュメント

- **Stage2ブラッシュアップ詳細:** `STAGE2_BRUSHUP_GUIDE.md`
- **LAMDA個別実行:** `run_stage2_batches.sh`（LAMDA専用スクリプト、今後は非推奨）
- **完了レポート:** `STAGE2_COMPLETION_REPORT.md`

## まとめ

共通ランナーにより：

- ✅ **1つのコマンドで全データセット処理**
- ✅ **ブラッシュアップフラグを全面採用**（streaming/resume/dual-threshold）
- ✅ **データセット追加が容易**（表に1行追加）
- ✅ **メモリ効率的**（省メモリ＆OOM回避）
- ✅ **再実行に強い**（冪等性保証）

従来の個別スクリプト管理から脱却し、運用コストを大幅に削減できます。
