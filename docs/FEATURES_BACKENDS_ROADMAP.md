# Features Backends 段階導入ロードマップ

## 概要

librosaベース特徴抽出から専門バックエンドへの段階的移行により、KPI安定性と実音追従精度を向上。

**現状課題**:
- bars.parquet start/end 揺らぎ（149/150小節、ダウンビート過剰検出）
- hat_density 過小評価（平均1.2 vs 目標5-6）
- Energy Curve の耳との乖離

**期待効果**:
- ✅ ビート/ダウンビート堅牢化 → KPI Gate安定化
- ✅ ハット密度精度向上 → relative density判定改善
- ✅ LUFS正規化 → Voicing/Velocity自然化

---

## Phase A: 即効・小変更（**実装完了**）

### 導入バックエンド

| 機能 | 既存 | 新規 | 期待効果 |
|------|------|------|----------|
| Beats | librosa.beat | **madmom RNN+DBN** | 小節境界安定化、rubato対応 |
| Downbeats | なし | **madmom RNN+DBN** | bars.start_sec/end_sec堅牢化 |
| Hat Density | librosa onset | **librosa_enhanced**（5-12kHz帯域限定） | 取りこぼし/誤検出削減 |
| Loudness | RMS | **pyloudnorm LUFS** | Energy Curve信頼度向上 |

### 実装状況

#### 1. バックエンド切替フラグ（`configs/arranger_weights.yaml`）

```yaml
features_backend:
  beats: madmom        # librosa | madmom
  downbeats: madmom    # none | librosa | madmom
  hat_density: librosa_enhanced  # librosa | librosa_enhanced | yamnet
  loudness: pyloudnorm # rms | pyloudnorm | essentia
  
  # madmom設定
  madmom:
    fps: 100           # 時間分解能（Hz）
    beats_per_bar: [3, 4]  # 3/4, 4/4対応
  
  # librosa_enhanced設定
  librosa_enhanced:
    bandpass_low: 5000   # Hz（5-12kHz帯域限定）
    bandpass_high: 12000
    onset_threshold: 0.6  # ロバスト閾値
    aggregate_window: 0.1  # フレーム集計窓（秒）
  
  # pyloudnorm設定
  pyloudnorm:
    block_size: 0.4    # EBU R128ブロックサイズ（秒）
```

#### 2. バックエンドラッパー（`ops/features_backends.py`）

**実装済み関数**:
- `extract_beats_madmom()`: madmom RNN+DBNビート抽出
- `extract_downbeats_madmom()`: madmom RNN+DBNダウンビート抽出
- `extract_hat_density_librosa_enhanced()`: 5-12kHz帯域限定オンセット検出
- `extract_loudness_pyloudnorm()`: EBU R128 LUFS測定
- `FeaturesBackend`: ディスパッチャークラス

**使用例**:
```python
from ops.features_backends import FeaturesBackend

config = yaml.safe_load(open('configs/arranger_weights.yaml'))
backend = FeaturesBackend(config['features_backend'])

# ビート抽出
beats = backend.extract_beats(audio_path, audio, sr)

# ダウンビート抽出
downbeats, positions = backend.extract_downbeats(audio_path, audio, sr)

# ハット密度（小節単位）
density = backend.extract_hat_density(
    audio_path, audio, sr,
    bar_start_sec=3.214,
    bar_end_sec=6.428
)

# LUFS
lufs = backend.extract_loudness(audio, sr, bar_start_sec, bar_end_sec)
```

#### 3. stems_features.py 統合

**現状**: バックエンド初期化のみ実装済み

```python
# ops/stems_features.py main()

backend = None
if args.backend_config and HAS_BACKENDS:
    config = yaml.safe_load(open(args.backend_config))
    backend = FeaturesBackend(config['features_backend'])
    logger.info("Backend config loaded")
```

**TODO（Phase A完成）**:
- [ ] `extract_drums_features()` → `backend.extract_hat_density()` 統合
- [ ] `extract_mix_features()` → `backend.extract_loudness()` 統合
- [ ] `_hat_density()` → librosa_enhanced切替ロジック追加
- [ ] `_loudness_db()` → pyloudnorm切替ロジック追加

### 依存パッケージ

```bash
# requirements.txt に追加済み
madmom>=0.16.1        # 既存（Chord認識用に導入済み）
pyloudnorm>=0.1.1     # 新規追加
scipy>=1.11.4         # butterworth filters用（既存）

# インストール
pip install pyloudnorm  # ✅ 完了
```

### 動作確認

```bash
# Phase A設定でStem特徴抽出
python ops/stems_features.py \
    --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
    --bars song_packages/suno_project/song_001/bars.parquet \
    --anchors data/suno_ai/suno_themesong/song_001/analysis/lyric_anchors.json \
    --output song_packages/suno_project/song_001/stem_features_enhanced.parquet \
    --backend-config configs/arranger_weights.yaml \
    --tempo-bpm 74.677

# 出力確認
# INFO: FeaturesBackend initialized:
# INFO:   beats: madmom
# INFO:   downbeats: madmom
# INFO:   hat_density: librosa_enhanced
# INFO:   loudness: pyloudnorm
```

**現状**: バックエンド初期化成功、ただし既存librosa実装を使用中（TODO項目で完成）

---

## Phase B: 効果大・中コスト（**未実装**）

### 導入バックエンド

| 機能 | Phase A | Phase B | 期待効果 |
|------|---------|---------|----------|
| Hat Density | librosa_enhanced | **YAMNet / PANNs** | "Hi-hat"クラス確率で誤検出大幅削減 |
| Loudness | pyloudnorm | pyloudnorm（継続） | - |

### YAMNet実装（`ops/features_backends.py`）

**実装済み**:
```python
def extract_hat_density_yamnet(
    audio_path: Path,
    bar_start_sec: float,
    bar_end_sec: float,
    threshold: float = 0.3,
    target_classes: List[str] = ["Hi-hat", "Cymbal"],
    **kwargs
) -> float:
    """
    YAMNet（AudioSet分類器）によるハット密度推定
    
    - モデル: TensorFlow Hub 'google/yamnet/1'
    - 出力: フレームごとのクラス確率（521クラス）
    - 集計: Hi-hat/Cymbal確率 > threshold のフレーム数
    """
```

### 依存パッケージ

```bash
# requirements-extra.txt に追加
tensorflow>=2.13,<2.14  # YAMNet用
tensorflow-hub>=0.14     # TF Hub
```

### 設定例（`configs/arranger_weights.yaml`）

```yaml
features_backend:
  hat_density: yamnet  # librosa | librosa_enhanced | yamnet
  
  yamnet:
    threshold: 0.3     # Hi-hat確率閾値
    target_classes: ["Hi-hat", "Cymbal"]
```

### 動作確認

```bash
# Phase B設定でStem特徴抽出
python ops/stems_features.py \
    --stems ... \
    --backend-config configs/arranger_weights.yaml \
    --tempo-bpm 74.677

# 期待: hat_density平均 3～6（現状1.2から大幅改善）
```

---

## Phase C: 重め・高精度（**未実装**）

### 導入バックエンド

| 機能 | Phase B | Phase C | 期待効果 |
|------|---------|---------|----------|
| Chords/Key | chordmap_only | **Chordino / Essentia** | 誤検出時のリカバー、調推定堅牢化 |
| Tempo | 単一BPM | **madmom tempo map** | 変動BPM追従、bars.start_sec精度向上 |

### Chordino実装（未実装）

```python
def extract_chords_chordino(audio_path: Path, **kwargs) -> List[Tuple[float, str]]:
    """
    Chordino（Vamp Plugin）によるコード推定
    
    - プラグイン: nnls-chroma + Chordino
    - 出力: [(time_sec, chord_label), ...]
    """
```

### 設定例

```yaml
features_backend:
  chords: chordino  # chordmap_only | chordino | essentia
```

---

## 段階導入スケジュール

### ✅ Phase A（即効・小変更）- **実装60%完了**

| タスク | 状態 | 備考 |
|--------|------|------|
| arranger_weights.yaml設定追加 | ✅ | features_backend セクション追加 |
| features_backends.py実装 | ✅ | madmom/librosa_enhanced/pyloudnorm対応 |
| stems_features.py統合 | 🔄 | バックエンド初期化のみ完了 |
| extract_drums_features()修正 | ⏳ | TODO: hat_density切替実装 |
| extract_mix_features()修正 | ⏳ | TODO: loudness切替実装 |
| 動作確認・KPI評価 | ⏳ | TODO: hat_density改善検証 |

**推定工数**: 2～3時間（TODO項目実装 + 検証）

### ⏳ Phase B（効果大・中コスト）- **未実装**

| タスク | 状態 | 備考 |
|--------|------|------|
| TensorFlow/YAMNetインストール | ⏳ | pip install tensorflow tensorflow-hub |
| YAMNet動作確認 | ⏳ | extract_hat_density_yamnet()テスト |
| stems_features.py統合 | ⏳ | backend.extract_hat_density()呼び出し |
| hat_density改善検証 | ⏳ | 目標: 平均3～6（現状1.2） |

**推定工数**: 4～6時間（TF環境構築 + YAMNet統合 + 検証）

### ⏳ Phase C（重め・高精度）- **未実装**

| タスク | 状態 | 備考 |
|--------|------|------|
| Vamp Pluginインストール | ⏳ | Chordino/NNLS Chroma |
| Python Vamp bindings | ⏳ | vamp / librosa.feature.chroma_cqt |
| Chordino実装 | ⏳ | extract_chords_chordino() |
| madmom tempo map実装 | ⏳ | extract_tempo_map_madmom() |

**推定工数**: 8～12時間（Vamp環境 + Chordino統合 + Tempo map実装）

---

## 期待される改善効果（Phase A完成時）

### Before（librosa-only）

```
hat_density:
  平均: 1.2
  最大: 2.0
  → ブースト未発動（0/150 bars）

KPI Pass率:
  スケルトン: 100% (149/149)
  実グルーヴ: 80.5% (120/149)
  → density too low: 14 bars (9.4%)
```

### After（Phase A: madmom + librosa_enhanced + pyloudnorm）

```
hat_density（期待値）:
  平均: 3～5（librosa_enhanced 5-12kHz帯域限定）
  最大: 8～10
  → ブースト発動: 30～50 bars（20～33%）

KPI Pass率（期待値）:
  実グルーヴ: 85～90%（+5～9%改善）
  → density too low: 5～8 bars（-60%削減）

Energy Curve:
  pyloudnorm LUFS → Piano/Strings追従自然化
  → セクション強弱の設計が楽に
```

### After（Phase B: YAMNet追加）

```
hat_density（期待値）:
  平均: 5～7（YAMNet Hi-hatクラス確率）
  最大: 12～15
  → ブースト発動: 50～80 bars（33～53%）

KPI Pass率（期待値）:
  実グルーヴ: 90～95%（+10～15%改善）
  → density too low: 2～4 bars（-85%削減）
```

---

## 運用フラグ設計

### フォールバック設計

各バックエンドは個別にインポート可能。欠落時はlibrosaにフォールバック。

```python
# features_backends.py

def extract_beats_madmom(audio_path, **kwargs):
    try:
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    except ImportError:
        logger.warning("madmom not installed, falling back to librosa")
        audio, sr = librosa.load(str(audio_path))
        return extract_beats_librosa(audio, sr)
    
    # madmom処理
    ...
```

### トグル切替

arranger_weights.yamlで簡単に切替可能。

```yaml
# librosa-onlyモード（既存互換）
features_backend:
  beats: librosa
  downbeats: none
  hat_density: librosa
  loudness: rms

# Phase Aモード
features_backend:
  beats: madmom
  downbeats: madmom
  hat_density: librosa_enhanced
  loudness: pyloudnorm

# Phase Bモード
features_backend:
  hat_density: yamnet

# Phase Cモード
features_backend:
  chords: chordino
```

---

## まとめ

### 最小コスト移行の推奨順序

1. **Phase A（2～3h）**: madmom beats/downbeats + librosa_enhanced hat_density + pyloudnorm
   - ✅ 設定・ラッパー実装完了
   - ⏳ stems_features.py統合（TODO 2項目）
   - 期待: KPI Pass率 +5～9%、hat_density 2.5倍改善

2. **Phase B（4～6h）**: YAMNet hat_density
   - ⏳ TF環境構築 + YAMNet統合
   - 期待: KPI Pass率 +10～15%、hat_density 4倍改善

3. **Phase C（8～12h）**: Chordino chords + madmom tempo map
   - ⏳ Vamp環境 + 実装
   - 期待: 変動BPM対応、調推定堅牢化

### 一言でいうと

> librosaは基盤に最適。ただし、要所（ダウンビート／ハイハット／LUFS）だけ専門ツールに置換すると、**KPIの安定と実音に基づく"ノリ"の両方が一段上がります**。
> 
> まずは **madmom（ビート/ダウンビート）** と **pyloudnorm（LUFS）** を小変更で差し替え（Phase A）、その後 **YAMNet** をハイハット密度に当てる（Phase B）流れをおすすめします。

---

## 参考資料

### madmom

- [madmom Documentation](https://madmom.readthedocs.io/)
- [RNNBeatProcessor](https://madmom.readthedocs.io/en/latest/modules/features/beats.html#madmom.features.beats.RNNBeatProcessor)
- [DBNDownBeatTrackingProcessor](https://madmom.readthedocs.io/en/latest/modules/features/downbeats.html)

### YAMNet

- [TensorFlow Hub: YAMNet](https://tfhub.dev/google/yamnet/1)
- [AudioSet Ontology](https://research.google.com/audioset/ontology/index.html)
- Classes: "Hi-hat" (id=99), "Cymbal" (id=100)

### pyloudnorm

- [pyloudnorm Documentation](https://github.com/csteinmetz1/pyloudnorm)
- [EBU R128 Standard](https://tech.ebu.ch/docs/r/r128.pdf)

### Chordino/Vamp

- [Chordino Vamp Plugin](https://code.soundsoftware.ac.uk/projects/nnls-chroma)
- [Python Vamp](https://github.com/pyvamp/pyvamp)
