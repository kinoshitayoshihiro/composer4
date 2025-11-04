# 🚀 GPU実測用コマンド集

## 📋 Google Colab セットアップ + 実行

### Step 1: 基本セットアップ（初回のみ）
```python
# Driveマウント
from google.colab import drive
drive.mount('/content/drive')

# リポジトリクローン
!git clone https://github.com/kinoshitayoshihiro/composer4.git
%cd composer4

# 最新コード取得
!git pull origin main
```

### Step 2: GPU確認（必須）
```python
import torch
import subprocess

# CUDA確認
print("="*60)
print("🔍 GPU Environment Check")
print("="*60)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    
    # VRAM確認
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {mem_total:.1f} GB")
else:
    print("❌ GPU not available! Change Runtime to GPU.")
    print("   Runtime > Change runtime type > Hardware accelerator: GPU")

# nvidia-smi
print("\n" + "="*60)
print("🖥️  nvidia-smi")
print("="*60)
!nvidia-smi
```

### Step 3: 依存関係インストール
```bash
# 最小構成（推奨）
!pip install -q transformers==4.46.0

# または完全セットアップ
!chmod +x setup_colab.sh
!bash setup_colab.sh
```

---

## 🎯 GPU実測コマンド

### 🔥 標準ベンチマーク（N=576）
```bash
# GPU実測: 中系列
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

**期待結果（Tesla T4）**:
```
🔵 Standard Attention:  850ms, 2400MB
🟢 Performer Attention: 620ms, 1800MB
🚀 Speedup: 1.37x
💚 Memory Reduction: -25%
```

---

### 🚀 長系列ベンチマーク（N=1024）
```bash
# GPU実測: 長系列
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 20 \
  --prompt-length 64 \
  --max-new-tokens 960 \
  --n-embd 768 \
  --n-layer 12 \
  --num-random-features 256 \
  --output results/performer_gpu_n1024.json
```

**期待結果（Tesla T4）**:
```
🔵 Standard Attention:  2200ms, 6500MB
🟢 Performer Attention: 1300ms, 4200MB
🚀 Speedup: 1.69x
💚 Memory Reduction: -35%
```

---

### 🔬 超長系列ベンチマーク（N=2048）
```bash
# GPU実測: 超長系列
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 1984 \
  --n-embd 768 \
  --n-layer 12 \
  --num-random-features 256 \
  --output results/performer_gpu_n2048.json
```

**期待結果（Tesla V100/A100）**:
```
🔵 Standard Attention:  8500ms, OOM risk
🟢 Performer Attention: 4200ms, 8500MB
🚀 Speedup: 2.02x
💚 Memory Reduction: -45%
```

---

## 📊 結果確認コマンド

### JSON結果表示
```python
import json

# 結果読み込み
with open('results/performer_gpu_n576.json', 'r') as f:
    results = json.load(f)

# 比較結果表示
comp = results['comparison']
print("="*60)
print("🎯 GPU Benchmark Results (N=576)")
print("="*60)
print(f"🔵 Standard:  {comp['standard_mean']:.0f}ms, {comp['standard_memory']:.0f}MB")
print(f"🟢 Performer: {comp['performer_mean']:.0f}ms, {comp['performer_memory']:.0f}MB")
print(f"🚀 Speedup:   {comp['speedup']:.2f}x")
print(f"💚 Memory:    {comp['memory_reduction_pct']:.1f}%")
```

### グラフ作成
```python
import matplotlib.pyplot as plt
import json

# データ読み込み
with open('results/performer_gpu_n576.json', 'r') as f:
    data = json.load(f)

comp = data['comparison']

# グラフ作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 速度比較
categories = ['Standard', 'Performer']
times = [comp['standard_mean'], comp['performer_mean']]
colors = ['#FF6B6B', '#4ECDC4']

ax1.bar(categories, times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title(f'⚡ Inference Speed (N=576)\nSpeedup: {comp["speedup"]:.2f}x', 
              fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for i, v in enumerate(times):
    ax1.text(i, v + 20, f'{v:.0f}ms', ha='center', fontweight='bold')

# メモリ比較
memories = [comp['standard_memory'], comp['performer_memory']]

ax2.bar(categories, memories, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
ax2.set_title(f'💾 GPU Memory Usage (N=576)\nReduction: {comp["memory_reduction_pct"]:.1f}%', 
              fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for i, v in enumerate(memories):
    ax2.text(i, v + 50, f'{v:.0f}MB', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/gpu_benchmark_n576.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Graph saved: results/gpu_benchmark_n576.png")
```

---

## 📥 結果ダウンロード

### Colab → ローカル
```python
from google.colab import files

# JSON結果
files.download('results/performer_gpu_n576.json')

# グラフ
files.download('results/gpu_benchmark_n576.png')

# ベンチマークログ
files.download('results/performer_realtime_benchmark.json')
```

---

## 🔄 複数系列長の一括実行

```bash
# N=576, N=1024, N=2048を連続実行
for n in 576 1024 2048; do
  tokens=$((n - 64))
  echo "🚀 Running benchmark: N=$n (max_new_tokens=$tokens)"
  
  python scripts/benchmark_performer_realtime.py \
    --device cuda \
    --num-samples 20 \
    --prompt-length 64 \
    --max-new-tokens $tokens \
    --n-embd 768 \
    --n-layer 12 \
    --num-random-features 256 \
    --output results/performer_gpu_n${n}.json
  
  echo "✅ Completed: N=$n"
  echo ""
done

echo "🎉 All benchmarks completed!"
```

---

## 📈 CPU vs GPU 比較

### CPU結果（既存）
```
N=320:
🔵 Standard:  4349ms
🟢 Performer: 6152ms
🚀 Speedup:   0.71x ❌ (遅い)
```

### GPU結果（期待値）
```
N=576:
🔵 Standard:  850ms
🟢 Performer: 620ms
🚀 Speedup:   1.37x ✅ (速い!)

N=1024:
🔵 Standard:  2200ms
🟢 Performer: 1300ms
🚀 Speedup:   1.69x ✅ (さらに速い!)
```

---

## 🎯 推奨実行フロー

### 1. クイックテスト（5分）
```bash
# 小規模テスト（N=256）
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 5 \
  --max-new-tokens 192 \
  --output results/quick_test.json
```

### 2. 標準ベンチマーク（10分）
```bash
# N=576
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 20 \
  --max-new-tokens 512 \
  --output results/performer_gpu_n576.json
```

### 3. 長系列ベンチマーク（20分）
```bash
# N=1024
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 20 \
  --max-new-tokens 960 \
  --output results/performer_gpu_n1024.json
```

### 4. 結果分析・グラフ作成（2分）
```python
# 上記のグラフ作成コード実行
```

---

## ⚠️ トラブルシューティング

### GPU not available
```python
# Runtime変更
# Runtime > Change runtime type > Hardware accelerator: GPU

# 確認
import torch
assert torch.cuda.is_available(), "GPU not enabled!"
```

### CUDA Out of Memory
```bash
# num-samples減少
--num-samples 10

# または系列長減少
--max-new-tokens 256
```

### ImportError: No module named 'transformers'
```bash
!pip install transformers==4.46.0
```

---

## 📚 関連ドキュメント

- `docs/COLAB_SETUP_PERFORMER.md` - 詳細セットアップ（533行）
- `COLAB_SETUP_QUICK.md` - クイックリファレンス
- `docs/PERFORMER_CPU_FINDINGS.md` - CPU性能発見
- `scripts/benchmark_performer_realtime.py` - 実測ツール

---

## 🎯 最終目標

1. **GPU実測**: N=576, N=1024でSpeedup 1.3x以上確認
2. **CPU vs GPU比較**: デバイス依存性の定量化
3. **drumgenerator適用判断**: GPU環境での採用可否決定

**期待**: GPU環境でPerformer 1.3-1.7x高速化 → drumgenerator進化へ！
