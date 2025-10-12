# 🚀 Google Colab セットアップガイド（最新版）

**Date**: 2025年10月12日  
**Target**: Performer Linear Attention ベンチマーク（GPU）  
**Environment**: Google Colab（GPU T4/V100/A100）

---

## ✅ 推奨セットアップ手順

### 📋 Step 1: Driveマウント + クローン
```python
# Driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# GitHubからクローン（最新版取得）
!git clone https://github.com/kinoshitayoshihiro/composer4.git
%cd composer4

# 最新のコミットを取得（Performer実装含む）
!git pull origin main
!git log --oneline -5
```

**確認ポイント**:
- `e0d1e9b39` Stage3長系列ベンチマーク最終報告
- `3a2b9125c` Stage3 v1.1 Sprint完了報告
- `6c457189a` Performer Linear Attention実装

---

### 📋 Step 2: GPU確認
```python
import torch
import sys

print("=" * 50)
print("🔍 Environment Check")
print("=" * 50)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("✅ GPU Ready!")
else:
    print("⚠️ GPU not available - CPU mode")
```

**期待される出力**:
```
GPU: Tesla T4 / V100 / A100
CUDA Version: 12.6
VRAM: 15.0 GB (T4) / 16.0 GB (V100) / 40.0 GB (A100)
✅ GPU Ready!
```

---

### 📋 Step 3: 依存関係インストール

#### 🎯 推奨: setup_colab.sh使用
```bash
# Colab専用セットアップスクリプト実行
!chmod +x setup_colab.sh
!bash setup_colab.sh
```

`setup_colab.sh`の内容（自動実行）:
- PyTorch GPU版インストール
- transformers、その他ML依存関係
- Colab最適化設定

#### 🔧 手動インストール（setup_colab.shがない場合）
```bash
# PyTorch GPU版（CUDA 12.6対応）
!pip install -q torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu126

# Transformers（GPT-2）
!pip install -q transformers==4.46.0

# 基本依存関係
!pip install -q -r requirements.txt

# 追加ML/Audio依存関係
!pip install -q numpy pandas scikit-learn librosa mido music21 pytorch-lightning
```

**注意**: Colabのデフォルトtorch 2.5.x使用を推奨

---

### 📋 Step 4: インストール確認
```python
# 重要パッケージの確認
import torch
import transformers
import numpy as np
import librosa
from ml.attention_performer import PerformerAttention, replace_attention_layers
from ml.performance_monitor import PerformanceMonitor

print("=" * 50)
print("✅ Package Verification")
print("=" * 50)
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"numpy: {np.__version__}")
print(f"librosa: {librosa.__version__}")
print("✅ Performer modules loaded successfully!")
```

---

## 🚀 Performer GPU ベンチマーク実行

### 📊 Method 1: 実時間計測版（推奨）
```python
# GPU環境で実時間計測ベンチマーク
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 20 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --n-embd 768 \
  --n-layer 12 \
  --num-random-features 256 \
  --output results/performer_gpu_n576.json
```

**期待される結果（GPU）**:
```
🎯 Performer Linear Attention Real-time Benchmark
Device: cuda
GPU: Tesla V100-SXM2-16GB

🔵 Standard Attention:
   Latency (mean):     850.00 ms
   Memory (peak):      2400.00 MB

🟢 Performer Attention:
   Latency (mean):     620.00 ms
   Memory (peak):      1800.00 MB

🚀 Speedup:          1.37x
💚 Memory reduction: +25.0%
```

### 📊 Method 2: 長系列ベンチマーク（N=1024）
```python
# 超長系列でメモリ効率検証
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 1024 \
  --n-embd 768 \
  --n-layer 12 \
  --output results/performer_gpu_n1088.json
```

**期待される改善**:
- Speedup: **1.5-2.0x**（長系列で顕著）
- Memory reduction: **30-40%**

---

## 📈 結果の確認

### JSON結果ファイル
```python
import json

# 結果を読み込み
with open('results/performer_gpu_n576.json') as f:
    results = json.load(f)

print("=" * 50)
print("📊 Benchmark Results")
print("=" * 50)

standard = results['standard']
performer = results['performer']
comparison = results['comparison']

print(f"\n🔵 Standard:")
print(f"   Latency: {standard['latency_mean']:.2f} ms")
print(f"   Memory:  {standard['peak_memory_mean']:.2f} MB")

print(f"\n🟢 Performer:")
print(f"   Latency: {performer['latency_mean']:.2f} ms")
print(f"   Memory:  {performer['peak_memory_mean']:.2f} MB")

print(f"\n🎯 Comparison:")
print(f"   Speedup:         {comparison['speedup']:.2f}x")
print(f"   Memory reduction: {comparison['memory_reduction_pct']:.1f}%")
```

### グラフ作成（オプション）
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Latency comparison
models = ['Standard', 'Performer']
latencies = [standard['latency_mean'], performer['latency_mean']]
ax1.bar(models, latencies, color=['blue', 'green'])
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Inference Latency')
ax1.grid(True, alpha=0.3)

# Memory comparison
memories = [standard['peak_memory_mean'], performer['peak_memory_mean']]
ax2.bar(models, memories, color=['blue', 'green'])
ax2.set_ylabel('Memory (MB)')
ax2.set_title('Peak Memory Usage')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/performer_comparison.png', dpi=150)
plt.show()

print("✅ Chart saved to results/performer_comparison.png")
```

---

## 🔍 トラブルシューティング

### Issue 1: CUDA Out of Memory
```python
# バッチサイズ削減
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 10 \  # 20→10に削減
  --max-new-tokens 256  # 512→256に削減
```

### Issue 2: PyTorch version mismatch
```python
# PyTorchバージョン確認
!pip show torch

# 必要に応じて再インストール
!pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

### Issue 3: transformers import error
```python
# transformers再インストール
!pip install transformers==4.46.0 --force-reinstall
```

---

## 📁 重要ファイル

### ベンチマークツール
- `scripts/benchmark_performer_realtime.py` - 実時間計測版（GPU対応）
- `scripts/benchmark_performer.py` - 旧版（ダミーメトリクス）

### Performer実装
- `ml/attention_performer.py` - Performer Linear Attention
- `ml/performance_monitor.py` - パフォーマンス監視

### ドキュメント
- `docs/PERFORMER_CPU_FINDINGS.md` - CPU性能発見
- `docs/STAGE3_LONG_SEQUENCE_BENCHMARK.md` - 長系列ベンチマーク
- `docs/STAGE3_LONG_SEQUENCE_FINAL_REPORT.md` - 最終報告

---

## 🎯 次のステップ

### 1. GPU実測完了後
```python
# 結果をローカルにダウンロード
from google.colab import files
files.download('results/performer_gpu_n576.json')
files.download('results/performer_comparison.png')
```

### 2. drumgeneratorへの適用
```python
# GPUで実測確認後、drumgeneratorに適用
from ml.attention_performer import replace_attention_layers

# drumgeneratorモデル読み込み
drum_model = load_drum_model(...)

# Performer適用
replace_attention_layers(drum_model, num_random_features=256)

# 長系列生成テスト
output = drum_model.generate(..., max_new_tokens=1024)
```

---

## 📚 参考コマンド集

### Git操作
```bash
# 最新コード取得
!git pull origin main

# 変更確認
!git log --oneline -10

# ブランチ確認
!git branch -a
```

### Python環境
```bash
# Python情報
!python --version
!which python

# インストール済みパッケージ
!pip list | grep -E "(torch|transform|numpy)"
```

### ファイル確認
```bash
# プロジェクト構造
!ls -lh ml/
!ls -lh scripts/

# 結果ディレクトリ
!mkdir -p results
!ls -lh results/
```

---

## ⚠️ 重要な注意点

### 1. GPU Runtime選択
- **Runtime** → **Change runtime type** → **GPU (T4/V100/A100)**

### 2. セッションタイムアウト
- Colab無料版: 12時間制限
- 長時間ベンチマークは注意

### 3. ファイル保存
```python
# 重要な結果はDriveに保存
!cp results/*.json /content/drive/MyDrive/composer4_results/
```

---

**最終更新**: 2025年10月12日  
**作成者**: GitHub Copilot  
**Status**: GPU検証待ち

**重要**: CPUでは0.71xと遅いため、**GPU環境での実測が必須**です！
