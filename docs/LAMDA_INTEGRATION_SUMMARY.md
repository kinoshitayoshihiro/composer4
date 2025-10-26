# LAMDA統合 & 自己循環学習 実装完了サマリー

## ✅ 実装完了（2025-10-23）

### 1. LAMDAのCHORDSデータアダプタ

**ファイル**: `adapters/lamda_chords_to_chordmap.py`

```bash
# 40万曲分のコードデータを chordmap.json に変換
python3 adapters/lamda_chords_to_chordmap.py \
  data/Los-Angeles-MIDI/CHORDS_DATA/LAMDa_CHORDS_DATA_5000.pickle \
  output/chordmap.json

# ✅ 動作確認済み: 85イベント変換成功
```

**機能**:
- LAMDA独自エンコーディング → 標準chordmap.json
- コード推定（音程解析: maj/min/7/maj7/m7/dim/aug）
- 時間正規化（beats/ticks/sec/bar_index対応）
- 重複除去（2.0QL未満の同一コードを間引き）

---

### 2. 階層化データ管理スキーマ

**ファイル**: `schemas/tiered_data_schema.py`

```python
# GOLD / SILVER / BRONZE の3ランク管理
from schemas.tiered_data_schema import TieredDataManager, TieredDataEntry

manager = TieredDataManager("data/tiered_corpus.jsonl")

# GOLD追加（Suno生成）
entry = TieredDataEntry(
    id="song_000001",
    tier="GOLD",
    audio=AudioStemPaths(mix="...", drums="...", bass="..."),
    key="C", tempo_bpm=120.0, genre="rock"
)
manager.add(entry)

# SILVER追加（LAMDA高信頼）
silver = TieredDataEntry(
    id="song_lamda_123",
    tier="SILVER",
    confidence=ConfidenceScores(chord=0.93, sections=0.88, ...)
)
manager.add(silver)

manager.save()
```

**機能**:
- GOLD/SILVER/BRONZE の品質階層化
- 信頼度スコア管理（chord/sections/roles/groove/controls/key/tempo）
- カバレッジヒートマップ（Key×Tempo×Genre×Emotion）
- 出所情報（suno/lamda/dawdreamer/manual）

---

### 3. 統合テストスクリプト

**ファイル**: `scripts/test_lamda_integration.sh`

```bash
chmod +x scripts/test_lamda_integration.sh
./scripts/test_lamda_integration.sh
```

**テスト項目**:
1. ✅ LAMDA CHORDS → chordmap 変換
2. ✅ 階層化スキーマ動作確認
3. ✅ fluidsynth存在確認
4. ⚠️ DawDreamer存在確認（オプショナル）
5. ✅ カバレッジ格子計算
6. ✅ LAMDA統計情報

---

## 📊 発見事項

### 1. LAMDAの公式CHORDSデータ

```
構造: List[[file_id, chord_sequence]]
総数: 162ファイル × 2,500曲 = 40.5万曲

エンコーディング:
[delta_time, duration, ?, pitch, velocity, ...]
例: [0, 39, 0, 66, 96, 39, 0, 62, 96, ...]
```

**活用方法**:
- ✅ **即座に使える**: 40万曲分のコード進行データ
- ✅ **楽器を超えた普遍的構造**: ステム分離なしで各楽器generatorに適用可能
- ✅ **Stage2統合**: `extended.chordmap` に公式データを優先使用

---

### 2. fluidsynthの存在

```
場所: data/Los-Angeles-MIDI/CODE/fluidsynth-master
用途: DawDreamerでの高品質レンダリング
```

**活用方法**:
```bash
# ビルド
cd data/Los-Angeles-MIDI/CODE/fluidsynth-master
mkdir build && cd build
cmake ..
make -j8

# DawDreamerと連携
python -c "
import dawdreamer as daw
engine = daw.RenderEngine(44100, 512)
# fluidsynthをVSTとして使用
"
```

---

## 🎯 ChatGPT提案の戦略（完全同意）

### ❌ 避けるべき
```
LAMDA 40万曲を全てSunoでstem化
→ ストレージ: ~72TB
→ コスト爆発
```

### ✅ 推奨フライホイール
```
Phase A: Suno GOLD (5-10k)
  ↓ カバレッジ格子設計
  ↓ 12 keys × 6 tempos × 3 sigs × 10 genres × 8 emotions
  ↓ = 17,280 cells → 1-2曲/cell = 5,760-11,520曲

Phase B: 教師器 v1 学習
  ↓ GOLD → 7つの推定器（key/chord/sections/roles/groove/controls/tempo）
  ↓ 信頼度スコア付き

Phase C: LAMDA → SILVER化
  ↓ 象徴的MIDI分離（音声デミックスなし）
  ↓ 教師器v1で推定 → 高信頼のみ採用（confidence ≥ 0.85）
  ↓ SILVER: 30,000曲（40万曲中）

Phase D: 再学習 → v2
  ↓ GOLD (30%) + SILVER (70%)
  ↓ 自己蒸留の劣化防止

Phase E: 継続運用
  ↓ カバレッジ格子の穴を監視
  ↓ Suno追加生成（必要セルのみ）
```

---

## 📁 ファイル構成

```
composer2-3/
├── adapters/
│   ├── lamda_chords_to_chordmap.py  ✅ 完成
│   ├── lamda_symbolic_demix.py      🔄 次に実装
│   └── __init__.py
├── schemas/
│   ├── tiered_data_schema.py        ✅ 完成
│   └── __init__.py
├── scripts/
│   ├── test_lamda_integration.sh    ✅ 完成
│   ├── dawdreamer_gold_generator.py 🔄 次に実装
│   ├── lamda_to_silver.sh           🔄 次に実装
│   └── filter_by_confidence.py      🔄 次に実装
├── models/
│   └── teacher_v1.py                🔄 次に実装
├── docs/
│   ├── LAMDA_INTEGRATION_PLAN.md    ✅ 完成
│   └── LAMDA_INTEGRATION_SUMMARY.md ✅ このファイル
└── data/
    ├── Los-Angeles-MIDI/
    │   ├── CHORDS_DATA/             ✅ 40万曲
    │   └── CODE/fluidsynth-master/  ✅ 発見
    └── tiered_corpus.jsonl          (GOLD/SILVER管理)
```

---

## 🚀 次のアクション

### 優先度1: 象徴的MIDI分離

```python
# adapters/lamda_symbolic_demix.py
def symbolic_demix(midi_path) -> Dict[str, MidiTrack]:
    """
    LAMDAの統合MIDIを役割別に疑似分離
    - Bass: 最低音域 < E3
    - Melody: 最高音域
    - Harmony: 和音密度
    - Drums: チャンネル10
    """
```

### 優先度2: 教師器v1実装

```python
# models/teacher_v1.py
class TeacherV1:
    def predict_with_confidence(self, midi) -> Dict[str, Any]:
        """
        7つの推定器 + 信頼度スコア
        - key, chord, sections, roles, groove, controls, tempo
        """
```

### 優先度3: fluidsynthビルド

```bash
cd data/Los-Angeles-MIDI/CODE/fluidsynth-master
mkdir build && cd build
cmake ..
make -j8
```

### 優先度4: Suno GOLD生成（手動）

```yaml
# 5-10k曲をカバレッジ格子に基づいて生成
# → audio/suno_gold/*.wav（6 stems）
```

---

## 📊 成功指標（KPI）

```yaml
Phase A (GOLD生成):
  カバレッジスコア: ≥ 0.30
  ステム品質: SNR ≥ 20dB

Phase B (教師器v1):
  Key F1: ≥ 0.95
  Chord F1: ≥ 0.90
  Sections F1: ≥ 0.85

Phase C (SILVER化):
  高信頼サンプル: ≥ 30,000曲
  平均信頼度: ≥ 0.88

Phase D (v2学習):
  v2 > v1: 全指標で+2%以上
  Controls integrity: ≥ 0.90
```

---

## 🎉 重要な洞察

### 1. コード進行は楽器を超えた普遍的構造

```
LAMDAのコードデータ（楽器混在）→ 各楽器generatorで活用可能

例: C → Am → F → G という進行があれば...

Guitar Generator:
  C  → [60, 64, 67] をアルペジオ/ストラム
  Am → [57, 60, 64] をアルペジオ/ストラム

Piano Generator:
  C  → [60, 64, 67, 72] をvoicing変換
  Am → [57, 60, 64, 69] をvoicing変換

Bass Generator:
  C  → [36] (ルート音のみ)
  Am → [33] (ルート音のみ)
```

### 2. 全データstem化は不要

```
❌ 400k × 6 stems = 2.4M ファイル（~72TB）
✅ 5-10k GOLD + 30k SILVER（高信頼）で十分

理由:
- 教師器v1で40万曲をラベリング
- 信頼度でフィルタ（≥ 0.85）
- 高品質だけを学習に使用
```

### 3. 自己蒸留の劣化防止

```
GOLD (30%以上) を常時混合
DAWDreamer完全教師でPB/CC規律維持
カバレッジヒートマップで分布監視
```

---

## 📖 参考ドキュメント

- `docs/LAMDA_INTEGRATION_PLAN.md`: 詳細な実装計画
- `adapters/lamda_chords_to_chordmap.py`: コード変換アダプタ
- `schemas/tiered_data_schema.py`: 階層化スキーマ
- `scripts/test_lamda_integration.sh`: 統合テスト

---

## 💡 結論

**ChatGPTの提案は極めて正確で実用的です。**

1. ✅ LAMDA公式CHORDSデータは即座に使える
2. ✅ fluidsynthはDawDreamerと連携可能
3. ✅ 全データstem化は避けるべき（5-10k GOLDで教師器を作る）
4. ✅ 自己循環学習でLAMDA 40万曲を活用（高信頼のみ）

**次のステップ**: 象徴的MIDI分離 → 教師器v1実装 → LAMDA適用
