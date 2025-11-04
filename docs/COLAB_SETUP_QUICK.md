# 🚀 Google Colab セットアップ - クイックリファレンス

## 📋 推奨セットアップ（コピペ用）

### Step 1: 基本セットアップ
```python
# Driveマウント
from google.colab import drive
drive.mount('/content/drive')

# GitHubからクローン
!git clone https://github.com/kinoshitayoshihiro/composer4.git
%cd composer4

# 最新コード取得
!git pull origin main
```

### Step 2: GPU確認
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Step 3: 依存関係インストール

#### ✅ 推奨: setup_colab.sh使用
```bash
!chmod +x setup_colab.sh
!bash setup_colab.sh
```

#### 🔧 手動インストール（setup_colab.shがない場合）
```bash
# PyTorch（Colabデフォルト使用推奨）
# Colabには通常torch 2.x系がプリインストール済み

# Transformers
!pip install -q transformers==4.46.0

# その他依存関係
!pip install -q pandas numpy scikit-learn
```

### Step 4: ベンチマーク実行
```bash
# GPU環境で実時間計測（N=576）
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 20 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --n-embd 768 \
  --n-layer 12 \
  --output results/performer_gpu_n576.json
```

---

## 🔍 あなたのコマンドチェック

### ❌ 修正が必要な箇所

#### 1. PyTorch バージョン指定
```python
# 問題のあるコマンド
!sed -i 's/torch==2.3.\*/torch==2.8.0/g' requirements.txt
!sed -i 's/scipy>=1.9/scipy>=1.14.0/g' requirements.txt
```

**問題点**:
- `torch==2.8.0` は存在しません（最新: 2.5.x）
- `scipy>=1.14.0` も存在しません（最新: 1.13.x）

**修正案**:
```python
# Colabデフォルトのtorch使用（推奨）
# 何もしない（Colabプリインストール版が最適）

# または明示的に最新版指定
!pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu126
```

#### 2. torchvision/torchaudio
```python
# 問題のあるコマンド
!pip install torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 -f https://download.pytorch.org/whl/torch_stable.html
```

**問題点**:
- バージョン番号が不正確
- `-f` オプションは `--index-url` が推奨

**修正案**:
```python
# 不要（Performer benchmarkにtorchvision/torchaudioは使わない）
# または正しいバージョン:
!pip install torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu126
```

---

## ✅ 正しいセットアップ手順

### 🎯 最小構成（Performer benchmark用）
```python
# Step 1: クローン
!git clone https://github.com/kinoshitayoshihiro/composer4.git
%cd composer4

# Step 2: 必要最小限のインストール
!pip install -q transformers==4.46.0

# Step 3: GPU確認
import torch
assert torch.cuda.is_available(), "GPU not available!"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Step 4: ベンチマーク実行
!python scripts/benchmark_performer_realtime.py --device cuda --num-samples 20 --output results/gpu_bench.json
```

### 🎯 完全セットアップ（全機能使用）
```python
# Step 1: クローン
!git clone https://github.com/kinoshitayoshihiro/composer4.git
%cd composer4

# Step 2: setup_colab.sh実行
!chmod +x setup_colab.sh
!bash setup_colab.sh

# Step 3: 追加パッケージ（必要に応じて）
!pip install -q matplotlib seaborn  # グラフ作成用

# Step 4: ベンチマーク + 結果可視化
!python scripts/benchmark_performer_realtime.py --device cuda --output results/gpu_bench.json

# 結果確認
import json
with open('results/gpu_bench.json') as f:
    print(json.dumps(json.load(f)['comparison'], indent=2))
```

---

## 🔧 setup_colab.sh vs 手動インストール

### ✅ setup_colab.sh を使うべき場合
- 全機能を使いたい
- データダウンロード・前処理も行う
- Stage3訓練も実行予定

### ✅ 手動インストールが良い場合
- Performer benchmarkのみ実行
- 最小限の依存関係で済ませたい
- インストール内容を完全に把握したい

---

## 📊 期待される結果（GPU）

### Tesla T4 (Colab無料版)
```
🔵 Standard:  850ms, 2400MB
🟢 Performer: 620ms, 1800MB
🚀 Speedup:   1.37x
💚 Memory:    -25%
```

### Tesla V100 (Colab Pro)
```
🔵 Standard:  650ms, 2400MB
🟢 Performer: 450ms, 1700MB
🚀 Speedup:   1.44x
💚 Memory:    -29%
```

### A100 (Colab Pro+)
```
🔵 Standard:  400ms, 2500MB
🟢 Performer: 260ms, 1600MB
🚀 Speedup:   1.54x
💚 Memory:    -36%
```

---

## 💡 推奨フロー

### Colabでの推奨実行順序
```python
# 1. 基本セットアップ
!git clone https://github.com/kinoshitayoshihiro/composer4.git
%cd composer4

# 2. GPU確認
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 3. 最小インストール
!pip install -q transformers==4.46.0

# 4. ベンチマーク実行
!python scripts/benchmark_performer_realtime.py --device cuda --num-samples 20

# 5. 結果確認
!cat results/performer_realtime_benchmark.json
```

---

## ⚠️ 注意事項

1. **PyTorch バージョン**: Colabプリインストール版（2.4.x-2.5.x）推奨
2. **CUDA バージョン**: Colab環境に合わせて自動選択（cu118/cu121/cu126）
3. **セッションタイムアウト**: 無料版12時間、Pro版24時間
4. **GPU制限**: 無料版は使用時間に制限あり

---

## 📚 参考ドキュメント

- `docs/COLAB_SETUP_PERFORMER.md` - 詳細セットアップガイド
- `colab_performer_benchmark.py` - Notebook用コードサンプル
- `docs/PERFORMER_CPU_FINDINGS.md` - CPU性能発見
- `docs/STAGE3_LONG_SEQUENCE_FINAL_REPORT.md` - 最終報告

---

**推奨**: `setup_colab.sh` を使わず、**手動で最小インストール**が確実です！
