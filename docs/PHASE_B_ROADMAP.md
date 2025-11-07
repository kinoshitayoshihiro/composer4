# Phase B: GMD/E-GMD学習ロードマップ

Phase A（Magenta統合）、Phase C（CREPE/OaF受け口）完了後の次フェーズ。
ドラムの**人間味向上**を目的とした学習パイプライン構築。

---

## 背景と目的

**現状の課題**:
- Magenta GrooVAEは汎用モデル（学習データ: GMD約1,000曲）
- プロジェクト固有のグルーヴ傾向に最適化されていない
- 学習済みモデルの再利用のみ（fine-tune未実施）

**Phase Bの目標**:
1. **軽学習**: MIDI-onlyでGMD学習ループ動作確認（SSD増設不要）
2. **本学習**: E-GMD WAV+MIDI導入、アタック/マイクロタイミング改善
3. **5万曲統合**: 自前ドラムMIDIデータ追加、教師多様性向上

**期待効果**:
- プロジェクト特化のGroove特徴抽出
- 人間味（ms単位のズレ、アタック形状）の再現精度向上
- GrooVAE fine-tune用の自前checkpoint生成

---

## Phase B-1: 軽学習（MIDI-only、SSD増設不要）

### 概要
- **データ**: GMD MIDIのみ（約1,110ファイル、既存SSD内）
- **目的**: 学習ループ動作確認、PyTorch配線検証
- **所要時間**: 数時間（1-2エポック、MOCKモード卒業）

### 前提条件
```bash
# PyTorchインストール（まだの場合）
pip install torch torchvision torchaudio

# GMD前処理完了確認
ls -la data/GMD_processed/
# 期待: index.parquet, groove_stats.json, train.txt, val.txt, test.txt
```

### 実行手順

**Step 1: GMD前処理（完了済み）**
```bash
python3 ops/gmd_preprocess.py \
  --gmd-root /Volumes/SSD-SCTU3A/.../GMD/groove \
  --out-dir data/GMD_processed \
  --seed 42
```

**出力確認**:
```
data/GMD_processed/
├── index.parquet       # 1,110ファイル、Groove指標付き
├── groove_stats.json   # 統計サマリー
├── train.txt           # 1,030ファイル（93%）
├── val.txt             # 70ファイル（6%）
└── test.txt            # 10ファイル（1%）
```

**Step 2: ダッシュボード生成（完了済み）**
```bash
python3 ops/gmd_dashboard.py \
  --index data/GMD_processed/index.parquet \
  --out data/GMD_processed/dashboard.html
```

**確認ポイント**:
- Velocity std: 29.89±7.26（人間味の指標）
- IOI std: 0.09±0.06（マイクロタイミング）
- Genre分布: rock(281), hiphop(91), funk(77)

**Step 3: 学習ループ本実行（PyTorchインストール後）**
```bash
# MOCKモード卒業、本学習開始
python3 ops/gmd_train_loop.py \
  --gmd-root /Volumes/SSD-SCTU3A/.../GMD/groove \
  --out-dir data/GMD_ckpts \
  --train-list data/GMD_processed/train.txt \
  --val-list data/GMD_processed/val.txt \
  --epochs 2 \
  --batch-size 32 \
  --lr 0.001 \
  --device cpu  # GPU利用可能なら: --device cuda
```

**期待出力**:
```
data/GMD_ckpts/
├── epoch_1.pt          # Epoch 1 checkpoint
├── epoch_2.pt          # Epoch 2 checkpoint
├── best_model.pt       # Validation loss最小モデル
└── training_history.json  # Loss推移
```

**検証**:
```bash
# Loss推移確認
python3 - <<'PY'
import json
hist = json.load(open("data/GMD_ckpts/training_history.json"))
for ep in hist:
    print(f"Epoch {ep['epoch']}: Train Loss={ep['train_loss']:.4f}, Val Loss={ep['val_loss']:.4f}")
PY
```

---

## Phase B-2: 本学習（E-GMD WAV+MIDI、SSD 1TB推奨）

### 概要
- **データ**: E-GMD（Expanded GMD、WAV+MIDI同時学習）
- **目的**: アタック/マイクロタイミング改善、人間味の"間"を学習
- **所要時間**: 数日〜1週間（10-20エポック、GPU推奨）

### E-GMDとは
- GMDの拡張版、各MIDIに対応するWAVも提供
- WAV波形から**アタック形状**、**マイクロタイミング**を直接学習可能
- プロドラマーの演奏ニュアンス（ms単位のズレ、ベロシティ変化）を捉える

### 前提条件
```bash
# SSD増設（1TB推奨、WAVは容量大）
# E-GMDダウンロード（約200GB）
# ダウンロード先: https://magenta.tensorflow.org/datasets/e-gmd

# PyTorchインストール（GPU推奨）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 実行手順

**Step 1: E-GMD前処理**
```bash
python3 ops/gmd_preprocess.py \
  --gmd-root /Volumes/SSD-EXPANDED/.../E-GMD \
  --out-dir data/EGMD_processed \
  --with-wav  # WAVパスも記録
  --seed 42
```

**出力確認**:
```
data/EGMD_processed/
├── index.parquet       # MIDI+WAVパス、Groove指標
├── groove_stats.json
├── train.txt
├── val.txt
└── test.txt
```

**Step 2: E-GMD学習ループ拡張（WAV+MIDI同時学習）**

現状の `ops/gmd_train_loop.py` はMIDI-onlyなので、WAV対応版に拡張：

```python
# ops/egmd_train_loop.py（新規作成）
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pretty_midi

class EGMDDataset(Dataset):
    def __init__(self, index_df, sample_rate=16000, n_mels=128):
        self.index_df = index_df
        self.sample_rate = sample_rate
        self.n_mels = n_mels
    
    def __len__(self):
        return len(self.index_df)
    
    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        
        # MIDI読み込み（Groove特徴）
        pm = pretty_midi.PrettyMIDI(row["midi_path"])
        midi_features = extract_midi_features(pm)  # velocity_std, ioi_std, etc.
        
        # WAV読み込み（Mel-spectrogram）
        wav, sr = librosa.load(row["wav_path"], sr=self.sample_rate)
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        return {
            "midi_features": torch.tensor(midi_features, dtype=torch.float32),
            "mel_spectrogram": torch.tensor(mel_db, dtype=torch.float32),
        }

# 学習ループはgmd_train_loop.pyと同様
```

**Step 3: 本学習実行**
```bash
python3 ops/egmd_train_loop.py \
  --gmd-root /Volumes/SSD-EXPANDED/.../E-GMD \
  --out-dir data/EGMD_ckpts \
  --train-list data/EGMD_processed/train.txt \
  --val-list data/EGMD_processed/val.txt \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0005 \
  --device cuda  # GPU必須（WAV処理は重い）
```

**期待出力**:
```
data/EGMD_ckpts/
├── epoch_1.pt
├── epoch_5.pt
├── epoch_10.pt
├── epoch_20.pt
├── best_model.pt
└── training_history.json
```

---

## Phase B-3: 5万曲ドラムMIDI追加

### 概要
- **データ**: ユーザー保有の5万曲ドラムMIDI
- **目的**: 教師データの多様性向上、プロジェクト特化
- **注意**: クリーン度A/Bのみ絞り込み推奨（ノイズ混入防止）

### クリーン度絞り込み基準

**クリーン度A（最優先）**:
- ドラム専用トラック（ch=9 or ch=10）
- Velocity分布が自然（mean: 70-100、std: 15-30）
- IOI（Inter-Onset Interval）が人間的（std < 0.15秒）
- ノート密度が適正（4-12 notes/bar）

**クリーン度B（準優先）**:
- 若干のノイズあり（他楽器混在、Velocity極端）
- 手動確認で修正可能レベル

**クリーン度C/D（除外推奨）**:
- ドラム以外の楽器混入
- 機械的すぎるベロシティ（Velocity全て同じ）
- テンポ不安定、小節境界崩れ

### 前処理手順

**Step 1: クリーン度自動判定**
```bash
python3 ops/assess_drum_cleanness.py \
  --drum-midi-dir /path/to/50k_drums \
  --out-index data/50k_drums_index.parquet \
  --out-stats data/50k_drums_stats.json
```

**出力**:
```
data/50k_drums_index.parquet
# columns: midi_path, cleanness_grade, velocity_std, ioi_std, note_density
```

**Step 2: クリーン度A/Bフィルタリング**
```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("data/50k_drums_index.parquet")
clean_df = df[df["cleanness_grade"].isin(["A", "B"])]
clean_df.to_parquet("data/50k_drums_clean.parquet")
print(f"✅ Filtered: {len(clean_df)} / {len(df)} files (Grade A/B only)")
PY
```

**Step 3: GMD/E-GMDとマージ**
```bash
python3 ops/merge_drum_datasets.py \
  --gmd-index data/GMD_processed/index.parquet \
  --egmd-index data/EGMD_processed/index.parquet \
  --custom-index data/50k_drums_clean.parquet \
  --out data/merged_drums/index.parquet
```

**出力**:
```
data/merged_drums/
├── index.parquet       # GMD(1,110) + E-GMD(?) + 50k_clean(?) = 合計?万ファイル
├── train.txt
├── val.txt
└── test.txt
```

**Step 4: 統合学習**
```bash
python3 ops/egmd_train_loop.py \
  --train-list data/merged_drums/train.txt \
  --val-list data/merged_drums/val.txt \
  --epochs 30 \
  --batch-size 16 \
  --device cuda
```

---

## GrooVAE Fine-tune（Phase B完了後）

### 概要
Phase B学習済みモデルをMagenta GrooVAEのfine-tuneに使用。

### 手順

**Step 1: Phase B checkpoint → GrooVAE互換形式変換**
```bash
python3 ops/convert_to_groovae_ckpt.py \
  --input-ckpt data/EGMD_ckpts/best_model.pt \
  --output-ckpt data/groovae_finetune_init.ckpt
```

**Step 2: GrooVAE fine-tune**
```bash
# Magenta専用venvで実行
source .venv_magenta/bin/activate

python3 ops/groovae_finetune.py \
  --init-ckpt data/groovae_finetune_init.ckpt \
  --train-list data/merged_drums/train.txt \
  --val-list data/merged_drums/val.txt \
  --epochs 10 \
  --out-ckpt data/groovae_finetuned.ckpt

deactivate
```

**Step 3: E2Eで自前checkpointテスト**
```bash
# ops/magenta_groove.py に --ckpt オプション追加
bash scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --drums-mode magenta \
  --magenta-ckpt data/groovae_finetuned.ckpt  # 自前checkpoint
```

**Step 4: ABテスト（既存 vs 学習版）**
```bash
# A: 既存Magentaモデル
bash scripts/e2e_suno_arrangement.sh song_001 --drums-mode magenta
mv song_001/full_arrangement.mid song_001/full_arrangement_magenta_default.mid

# B: 学習版モデル
bash scripts/e2e_suno_arrangement.sh song_001 --drums-mode magenta --magenta-ckpt data/groovae_finetuned.ckpt
mv song_001/full_arrangement.mid song_001/full_arrangement_magenta_finetuned.mid

# 比較（DAWで聴き比べ、KPI測定）
python3 scripts/kpi_gate_enhanced.py --midi song_001/full_arrangement_magenta_default.mid ...
python3 scripts/kpi_gate_enhanced.py --midi song_001/full_arrangement_magenta_finetuned.mid ...
```

---

## KPI拡張（Groove指標追加）

### 新規KPI項目（Phase B完了後）

**1. Velocity Spread**
```python
# drums各ピッチのVelocity分布の標準偏差
velocity_std = np.std([note.velocity for note in drum_notes])
# 期待: 15-30（人間的）、<10（機械的）、>40（不安定）
```

**2. IOI Std（Inter-Onset Interval）**
```python
# 連続ノート間のマイクロタイミングばらつき
onsets = sorted([note.start for note in drum_notes])
ioi = np.diff(onsets)
ioi_std = np.std(ioi)
# 期待: 0.05-0.15（人間的）、<0.02（機械的）
```

**3. Humanization Score**
```python
# 総合人間味スコア（Velocity std + IOI std + Note density変動）
humanization_score = (
    normalize(velocity_std, 15, 30) * 0.4 +
    normalize(ioi_std, 0.05, 0.15) * 0.3 +
    normalize(note_density_variation, 1.5, 3.0) * 0.3
)
# 期待: 0.7-1.0（人間的）、<0.5（機械的）
```

### KPI Gate拡張
```yaml
# configs/gate_prod.yaml
drums:
  kpi:
    velocity_std_min: 15
    velocity_std_max: 40
    ioi_std_min: 0.02
    ioi_std_max: 0.20
    humanization_score_min: 0.65
```

---

## Phase B完了チェックリスト

- [ ] **B-1軽学習**: GMD MIDI-only学習ループ動作確認（2 epochs、Loss減少）
- [ ] **B-1検証**: best_model.pt生成、training_history.json保存
- [ ] **B-2前処理**: E-GMD前処理完了（WAVパス記録）
- [ ] **B-2本学習**: E-GMD WAV+MIDI学習（20 epochs、GPU推奨）
- [ ] **B-3絞り込み**: 5万曲ドラムMIDIクリーン度A/B抽出
- [ ] **B-3統合学習**: GMD+E-GMD+50k統合データで30 epochs
- [ ] **GrooVAE fine-tune**: 自前checkpoint生成、E2E動作確認
- [ ] **ABテスト**: 既存 vs 学習版モデル比較
- [ ] **KPI拡張**: Groove指標（Velocity std, IOI std, Humanization score）追加

---

## 次のステップ（Phase B完了後）

### Phase D: CREPE/OaF本格統合
- CREPE: vocal_f0_crepe.parquet → lyric_anchors_crepe.json → ±80ms duck/accent boost
- OaF: piano_oaf.mid → リズム評価（ベロシティ参照）+ ハーモニー補助

### Phase E: KPI/CI拡張
- CI項目追加（11項目以上）
- CREPE/OaF効果測定指標追加
- Groove指標の統合レポート

### Phase F: 本番運用
- 5万曲バッチ処理
- 品質モニタリング（KPI Gate Pass率追跡）
- 継続的改善（ユーザーフィードバック反映）

---

## トラブルシューティング

### Q1: PyTorchインストール失敗
```bash
# CPU版（軽量）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU版（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q2: GMD前処理で「ファイルが見つからない」
```bash
# パス確認
ls -la /Volumes/SSD-SCTU3A/.../GMD/groove/drummer1/session1/

# 相対パス→絶対パスに変更
python3 ops/gmd_preprocess.py \
  --gmd-root "$(realpath /Volumes/SSD-SCTU3A/.../GMD/groove)" \
  --out-dir data/GMD_processed
```

### Q3: 学習中にメモリ不足
```bash
# バッチサイズ削減
python3 ops/gmd_train_loop.py ... --batch-size 8  # 32→8

# データローダーのワーカー削減
# gmd_train_loop.py内: DataLoader(..., num_workers=0)
```

### Q4: GPU利用不可
```bash
# CPU学習に切り替え（遅いが動作可能）
python3 ops/gmd_train_loop.py ... --device cpu
```

---

## 参考資料

- [Magenta GrooVAE論文](https://arxiv.org/abs/1905.06118)
- [GMDデータセット](https://magenta.tensorflow.org/datasets/groove)
- [E-GMDデータセット](https://magenta.tensorflow.org/datasets/e-gmd)
- [PyTorch公式ドキュメント](https://pytorch.org/docs/stable/index.html)

---

**Phase B完了の証跡**:
- `data/EGMD_ckpts/best_model.pt`（本学習checkpoint）
- `data/groovae_finetuned.ckpt`（GrooVAE fine-tune済みモデル）
- ABテスト結果（KPI比較レポート）
