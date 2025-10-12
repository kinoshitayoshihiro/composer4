"""
Google Colab セットアップ - Performer GPU Benchmark
====================================================

📋 Step 1: 基本セットアップ
"""

# Driveマウント
from google.colab import drive
drive.mount('/content/drive')

# リポジトリクローン
!git clone https://github.com/kinoshitayoshihiro/composer4.git
%cd composer4

# 最新コード取得
!git pull origin main
!git log --oneline -5

"""
📋 Step 2: GPU確認
"""

import torch
import sys

print("=" * 50)
print("🔍 Environment Check")
print("=" * 50)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("✅ GPU Ready!")
else:
    print("⚠️ GPU not available")

"""
📋 Step 3: 依存関係インストール（推奨）
"""

# Colab用セットアップスクリプト実行
!chmod +x setup_colab.sh
!bash setup_colab.sh

# または手動インストール:
# !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# !pip install -q transformers==4.46.0
# !pip install -q -r requirements.txt

"""
📋 Step 4: インストール確認
"""

# パッケージ確認
import torch
import transformers
from ml.attention_performer import PerformerAttention, replace_attention_layers
from ml.performance_monitor import PerformanceMonitor

print("=" * 50)
print("✅ Package Verification")
print("=" * 50)
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print("✅ All packages loaded!")

"""
🚀 Step 5: Performer GPU ベンチマーク実行
"""

# GPU環境で実時間計測（N=576）
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 20 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --n-embd 768 \
  --n-layer 12 \
  --num-random-features 256 \
  --output results/performer_gpu_n576.json

"""
📊 Step 6: 結果確認
"""

import json
import matplotlib.pyplot as plt

# 結果読み込み
with open('results/performer_gpu_n576.json') as f:
    results = json.load(f)

standard = results['standard']
performer = results['performer']
comparison = results['comparison']

# 結果表示
print("=" * 50)
print("📊 Benchmark Results (GPU)")
print("=" * 50)
print(f"\n🔵 Standard:")
print(f"   Latency: {standard['latency_mean']:.2f} ms")
print(f"   Memory:  {standard['peak_memory_mean']:.2f} MB")

print(f"\n🟢 Performer:")
print(f"   Latency: {performer['latency_mean']:.2f} ms")
print(f"   Memory:  {performer['peak_memory_mean']:.2f} MB")

print(f"\n🎯 Comparison:")
print(f"   Speedup:          {comparison['speedup']:.2f}x")
print(f"   Memory reduction: {comparison['memory_reduction_pct']:.1f}%")

# グラフ作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

models = ['Standard', 'Performer']
latencies = [standard['latency_mean'], performer['latency_mean']]
memories = [standard['peak_memory_mean'], performer['peak_memory_mean']]

ax1.bar(models, latencies, color=['blue', 'green'])
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Inference Latency (GPU)')
ax1.grid(True, alpha=0.3)

ax2.bar(models, memories, color=['blue', 'green'])
ax2.set_ylabel('Memory (MB)')
ax2.set_title('Peak Memory Usage')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/performer_gpu_comparison.png', dpi=150)
plt.show()

print("✅ Chart saved!")

"""
💾 Step 7: 結果ダウンロード
"""

from google.colab import files
files.download('results/performer_gpu_n576.json')
files.download('results/performer_gpu_comparison.png')

"""
🎯 オプション: 長系列ベンチマーク（N=1024）
"""

# 超長系列でメモリ効率検証
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 1024 \
  --n-embd 768 \
  --n-layer 12 \
  --output results/performer_gpu_n1088.json

# 結果確認
with open('results/performer_gpu_n1088.json') as f:
    results_long = json.load(f)
    
print(f"\n📈 Long Sequence (N=1088):")
print(f"   Speedup: {results_long['comparison']['speedup']:.2f}x")
print(f"   Memory:  {results_long['comparison']['memory_reduction_pct']:.1f}%")
