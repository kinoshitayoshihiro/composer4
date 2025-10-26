# LAMDA統合 & 自己循環学習 実装計画

## 📊 戦略概要（ChatGPT提案に基づく）

```
❌ 避けるべき: LAMDA 40万曲を全てSunoでstem化
   → ストレージ: ~72TB、コスト爆発

✅ 推奨フライホイール:
   Phase A: Suno GOLD (5-10k)
   Phase B: 教師器 v1 学習
   Phase C: LAMDA → SILVER化（高信頼のみ）
   Phase D: 再学習 → v2（自己蒸留）
   Phase E: 継続運用
```

---

## 🎯 Phase A: Seed-GOLD 生成（5-10k曲）

### カバレッジ格子設計

```python
COVERAGE_GRID = {
    "keys": 12種 (C, C#, D, ...)
    "tempo_bins": 6種 (60-80, 80-100, ...)
    "time_signatures": 3種 (4/4, 3/4, 6/8)
    "genres": 10種 (rock, jazz, pop, ...)
    "emotions": 8種 (happy, sad, energetic, ...)
}

総セル数: 12 × 6 × 3 × 10 × 8 = 17,280 cells
目標: 1-2クリップ/セル → 約5,760-11,520曲
```

### Suno生成プロンプト例

```yaml
# configs/suno/coverage_grid.yaml
generation_matrix:
  - key: C
    tempo: 120
    time_sig: 4/4
    genre: rock
    emotion: energetic
    prompt: "Energetic rock song in C major, 120 BPM, 4/4 time"
    stems: [vocals, drums, bass, guitar, piano, other]
  
  - key: Am
    tempo: 90
    time_sig: 3/4
    genre: jazz
    emotion: calm
    prompt: "Calm jazz waltz in A minor, 90 BPM, 3/4 time"
    stems: [vocals, drums, bass, guitar, piano, other]
  
  # ... 17,280 configurations
```

### DAWDreamer合成GOLD（補完用）

```python
# scripts/dawdreamer_gold_generator.py
"""
完全教師データ生成: PB/CC/RPN/奏法を確定的に付与
"""
from dawdreamer import RenderEngine

def generate_gold_with_controls(midi_path, vst_path, output_wav):
    """
    MIDI → VSTレンダリング（完全制御）
    
    - Pitch Bend: ±8191
    - CC: Expression(11), Modulation(1), Sustain(64)
    - RPN: Pitch Bend Sensitivity
    - 奏法: キースイッチで確定
    """
    engine = RenderEngine(44100, 512)
    synth = engine.make_plugin_processor("vst", vst_path)
    
    # MIDIロード + 制御イベント追加
    synth.load_midi(midi_path)
    synth.add_midi_note(60, 100, 0.0, 1.0)
    synth.add_midi_cc(1, 64, 0.0)  # Modulation
    synth.add_midi_pitch_bend(8191, 0.5)  # Max pitch up
    
    engine.render()
    engine.save_audio(output_wav)
```

---

## 🧠 Phase B: 教師器 v1 学習

### 推定器の7つのモジュール

```python
# models/teacher_v1.py
"""
GOLD (5-10k) から学習する7つの推定器
"""

class TeacherV1:
    def __init__(self):
        self.key_estimator = KeyEstimator()        # key推定
        self.chord_estimator = ChordEstimator()    # chord推定
        self.section_estimator = SectionEstimator()  # sections推定
        self.role_estimator = RoleEstimator()      # 役割分解
        self.groove_estimator = GrooveEstimator()  # グルーヴ推定
        self.control_estimator = ControlEstimator()  # PB/CC推定
        self.tempo_estimator = TempoEstimator()    # テンポ推定
    
    def predict_with_confidence(self, midi_path) -> Dict[str, Any]:
        """
        各モジュールの推定 + 信頼度スコア
        
        Returns:
            {
                "key": {"value": "C", "confidence": 0.95},
                "chordmap": {"events": [...], "confidence": 0.90},
                "sections": {"events": [...], "confidence": 0.88},
                "roles": {"bass": [...], "confidence": 0.92},
                "groove": {"swing": 0.3, "confidence": 0.85},
                "controls": {"pb_range": 2, "confidence": 0.80},
                "tempo": {"bpm": 120, "confidence": 0.93}
            }
        """
        pass
```

### 信頼度スコアの算出方法

```python
def calculate_confidence(prediction, ground_truth):
    """
    - Key: 完全一致 → 1.0, 不一致 → 0.0
    - Chord: bar単位のF1スコア
    - Sections: セクション境界のIoU
    - Roles: 楽器分離のSNR
    - Groove: スイング量の誤差率
    - Controls: PB/CCの正規化誤差
    - Tempo: BPM誤差率
    """
    pass
```

---

## 🔄 Phase C: LAMDA → SILVER化

### 象徴的MIDI分離（音声デミックスなし）

```python
# adapters/lamda_symbolic_demix.py
"""
LAMDAの統合MIDIを役割別に疑似分離
"""

def symbolic_demix(midi_path) -> Dict[str, MidiTrack]:
    """
    記譜上の特徴で役割を推定
    
    - Bass: 最低音域 < E3, 5度運動比率
    - Melody: 最高音域, 最長フレーズ線
    - Harmony: 同時発音数, 和音密度
    - Drums: チャンネル10 or 無音程ノート
    - Ornament: 短音, トリル, 装飾音
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    tracks = {
        "bass": [],
        "melody": [],
        "harmony": [],
        "drums": [],
        "ornament": [],
    }
    
    for inst in pm.instruments:
        if inst.is_drum:
            tracks["drums"] = inst.notes
        else:
            # 音域・密度・フレーズ長で分類
            avg_pitch = np.mean([n.pitch for n in inst.notes])
            if avg_pitch < 48:  # < C3
                tracks["bass"].extend(inst.notes)
            elif avg_pitch > 72:  # > C5
                tracks["melody"].extend(inst.notes)
            else:
                tracks["harmony"].extend(inst.notes)
    
    return tracks
```

### LAMDA適用パイプライン

```bash
# scripts/lamda_to_silver.sh
#!/bin/bash

# 1) LAMDA 40万曲に教師器v1を適用
for midi in data/lamda/**/*.mid; do
  # 象徴的分離
  python adapters/lamda_symbolic_demix.py "$midi" --out demixed/
  
  # 教師器v1で推定
  python models/teacher_v1.py \
    --midi "$midi" \
    --demixed demixed/ \
    --out predictions/
  
  # 信頼度でフィルタ
  python scripts/filter_by_confidence.py \
    --predictions predictions/ \
    --min-overall 0.85 \
    --min-chord 0.90 \
    --min-sections 0.85 \
    --tier SILVER \
    --out silver_corpus.jsonl
done
```

---

## 🔁 Phase D: 再学習 → v2（自己蒸留）

### 学習データの混合比率

```python
# GOLD: 常に30%以上混ぜる（劣化防止）
# SILVER: 70%まで使用（高信頼のみ）

train_data = {
    "GOLD": load_gold_corpus(),      # 5-10k
    "SILVER": load_silver_corpus(),  # 高信頼サンプル（最大30k）
}

# 混合
mixed = []
mixed.extend(random.sample(train_data["GOLD"], int(len(train_data["GOLD"]) * 1.0)))  # 100% GOLD
mixed.extend(random.sample(train_data["SILVER"], int(len(train_data["SILVER"]) * 0.7)))  # 70% SILVER

# v2学習
teacher_v2 = TeacherV2()
teacher_v2.train(mixed)
```

### 品質監視ダッシュボード

```python
# scripts/quality_dashboard.py
"""
自己蒸留の劣化を監視
"""

def monitor_quality(version: str):
    """
    - Controls integrity: PB±8191・RPN順序成立率
    - Role separability: 各ロールの帯域/密度分離度
    - Coverage: カバレッジ格子のヒートマップ
    - Confidence distribution: 信頼度分布の推移
    """
    metrics = {
        "controls_integrity": calculate_controls_integrity(),
        "role_separability": calculate_role_separability(),
        "coverage_score": calculate_coverage_score(),
        "confidence_dist": get_confidence_distribution(),
    }
    
    # 可視化
    plot_heatmap(metrics["coverage_score"])
    plot_distribution(metrics["confidence_dist"])
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
├── models/
│   ├── teacher_v1.py                🔄 次に実装
│   ├── key_estimator.py
│   ├── chord_estimator.py
│   ├── section_estimator.py
│   ├── role_estimator.py
│   ├── groove_estimator.py
│   ├── control_estimator.py
│   └── tempo_estimator.py
├── scripts/
│   ├── dawdreamer_gold_generator.py 🔄 次に実装
│   ├── lamda_to_silver.sh           🔄 次に実装
│   ├── filter_by_confidence.py      🔄 次に実装
│   └── quality_dashboard.py         🔄 次に実装
├── configs/
│   └── suno/
│       └── coverage_grid.yaml       🔄 次に実装
├── data/
│   ├── tiered_corpus.jsonl          (GOLD/SILVER/BRONZEの管理)
│   ├── Los-Angeles-MIDI/
│   │   └── CHORDS_DATA/             ✅ 既存40万曲
│   └── suno_gold/                   (5-10k Sunoステム)
└── docs/
    └── LAMDA_INTEGRATION_PLAN.md    ✅ このファイル
```

---

## 🚀 最短の実装ステップ

### Step 1: LAMDA CHORDSデータの統合（✅ 完了）

```bash
# 40万曲分のCHORDSデータをchordmap.jsonに変換
python adapters/lamda_chords_to_chordmap.py \
  data/Los-Angeles-MIDI/CHORDS_DATA/LAMDa_CHORDS_DATA_5000.pickle \
  /tmp/test_chordmap.json

# ✅ 出力: 85イベント、D#/G/E などのコード進行
```

### Step 2: fluidsynthのビルド（🔄 次に実行）

```bash
cd data/Los-Angeles-MIDI/CODE/fluidsynth-master
mkdir build && cd build
cmake ..
make -j8

# DawDreamerと連携確認
python -c "import dawdreamer; print(dawdreamer.__version__)"
```

### Step 3: Suno GOLD生成（手動 or API）

```bash
# Sunoで5-10k曲を生成（カバレッジ格子に基づく）
# → audio/suno_gold/*.wav（6 stems）
```

### Step 4: Stage0/2パイプライン（既存）

```bash
# Suno GOLD → Stage0
python ops/sections_from_audio.py \
  --audio audio/suno_gold/song_001_mix.wav \
  --out stage0/suno_gold/song_001/sections.json

# Stage0 → Stage2
python scripts/lamda_stage2_extractor.py \
  --input-dir stage0/suno_gold/ \
  --output-dir stage2/suno_gold/ \
  --emit-csv aggregate
```

### Step 5: 教師器v1学習（🔄 新規実装）

```bash
# GOLD (5-10k) → 教師器v1
python models/teacher_v1.py \
  --train-data data/tiered_corpus.jsonl \
  --tier GOLD \
  --out models/teacher_v1.ckpt
```

### Step 6: LAMDA → SILVER（🔄 新規実装）

```bash
# LAMDA 40万曲に適用
bash scripts/lamda_to_silver.sh \
  --lamda-dir data/Los-Angeles-MIDI/ \
  --teacher models/teacher_v1.ckpt \
  --min-confidence 0.85 \
  --out data/tiered_corpus.jsonl
```

---

## ⚠️ リスク対策

### 1. 自己蒸留の劣化防止

```python
# GOLD常時混合（30%以上）
# DAWDreamer完全教師でPB/CC規律維持
```

### 2. 分布偏り回避

```python
# カバレッジヒートマップ監視
# 薄いセルを優先的にSuno生成
```

### 3. ライセンス確認

```bash
# Suno出力の利用許諾範囲を事前確認
# 不明な場合はDAWDreamer合成に寄せる
```

---

## 📊 成功指標（KPI）

```yaml
Phase A (GOLD生成):
  - カバレッジスコア: ≥ 0.30 (17,280セル中5,184セル)
  - ステム品質: SNR ≥ 20dB

Phase B (教師器v1):
  - Key F1: ≥ 0.95
  - Chord F1: ≥ 0.90
  - Sections F1: ≥ 0.85

Phase C (SILVER化):
  - 高信頼サンプル: ≥ 30,000曲（40万曲中）
  - 平均信頼度: ≥ 0.88

Phase D (v2学習):
  - v2 > v1: 全指標で+2%以上
  - Controls integrity: ≥ 0.90
```

---

## 🎯 次のアクション

1. ✅ **LAMDA CHORDSアダプタ**: 完成・テスト済み
2. ✅ **階層化スキーマ**: 完成
3. 🔄 **fluidsynthビルド**: 次に実行
4. 🔄 **象徴的MIDI分離**: 実装待ち
5. 🔄 **教師器v1**: 実装待ち
6. 🔄 **Suno GOLD生成**: 手動実行待ち

**どれから始めますか？**
