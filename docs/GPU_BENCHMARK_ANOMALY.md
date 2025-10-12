# 🚨 GPU Benchmark Anomaly Investigation

## 問題の症状

```
🚀 Speedup:   0.18x  ← 期待値: 1.37x
⏱️ Latency Delta: -27905ms
💾 Memory Delta: -1197MB
💚 Memory Reduction: -277.3%
```

**期待値と真逆**: CPU (0.71x) よりさらに遅い！

---

## 考えられる原因

### 1. デバイス配置ミス
- モデルがGPUに移動していない
- 入力データがCPUに残っている
- GPU ↔ CPU間のデータ転送オーバーヘッド

### 2. CUDA同期の問題
- `torch.cuda.synchronize()` が適切に呼ばれていない
- 非同期実行の計測ミス

### 3. num_random_features設定
- デフォルト値（256）が大きすぎる可能性
- L4の特性に合っていない

### 4. メモリスワップ
- VRAM不足でスワップ発生
- バッチサイズが大きすぎる

---

## デバッグ手順

### Step 1: デバイス確認
```python
import torch

# モデルのデバイス確認
print("Model device:", next(model.parameters()).device)

# 入力のデバイス確認
print("Input device:", input_ids.device)

# CUDA確認
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))
```

### Step 2: 詳細プロファイリング
```python
import time
import torch

# GPU同期確認
torch.cuda.synchronize()
start = time.time()

# 処理
output = model.generate(input_ids, max_new_tokens=512, use_cache=False)

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Actual time: {elapsed*1000:.0f}ms")
```

### Step 3: num_random_features調整
```bash
# 小さい値で再実行
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 5 \
  --max-new-tokens 512 \
  --num-random-features 64 \
  --output results/test_rf64.json

# 中間値
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 5 \
  --max-new-tokens 512 \
  --num-random-features 128 \
  --output results/test_rf128.json
```

### Step 4: メモリ確認
```python
import torch

# VRAM使用量
print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Reserved:", torch.cuda.memory_reserved() / 1e9, "GB")
print("Max allocated:", torch.cuda.max_memory_allocated() / 1e9, "GB")

# nvidia-smi
!nvidia-smi
```

---

## 修正版ベンチマーク実行

### クイックテスト（num_samples=5）
```bash
# 小規模で動作確認
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 5 \
  --prompt-length 64 \
  --max-new-tokens 256 \
  --n-embd 768 \
  --n-layer 12 \
  --num-random-features 128 \
  --output results/quick_test.json
```

### デバッグモード追加版
```python
# scripts/benchmark_performer_realtime.py の先頭に追加
import torch
print("="*60)
print("🔍 Device Check")
print("="*60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")
print("="*60)
```

---

## 期待される正常値

### NVIDIA L4 (24GB VRAM)

| 系列長 | Standard | Performer | Speedup | Memory |
|--------|----------|-----------|---------|--------|
| N=256  | ~400ms   | ~320ms    | 1.25x   | -15%   |
| N=512  | ~800ms   | ~600ms    | 1.33x   | -22%   |
| N=576  | ~900ms   | ~650ms    | 1.38x   | -25%   |
| N=1024 | ~2000ms  | ~1200ms   | 1.67x   | -33%   |

---

## 緊急対応コマンド

### 1. シンプルなテスト
```bash
# 最小構成で動作確認
!python -c "
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# GPU確認
assert torch.cuda.is_available()
device = torch.device('cuda')

# モデル作成
config = GPT2Config(vocab_size=1000, n_embd=768, n_layer=2, n_head=12)
model = GPT2LMHeadModel(config).to(device)

# 推論テスト
input_ids = torch.randint(0, 1000, (1, 64)).to(device)
torch.cuda.synchronize()

import time
start = time.time()
output = model.generate(input_ids, max_new_tokens=100, use_cache=False)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f'✅ GPU inference OK: {elapsed*1000:.0f}ms')
"
```

### 2. Standard vs Performer 個別テスト
```bash
# Standard Attentionのみ
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 5 \
  --max-new-tokens 256 \
  --benchmark-type standard \
  --output results/standard_only.json

# Performer Attentionのみ
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 5 \
  --max-new-tokens 256 \
  --benchmark-type performer \
  --output results/performer_only.json
```

---

## ローカルで結果確認

```bash
# Colabからダウンロードした結果を確認
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

# JSON確認
cat results/performer_gpu_n576.json | python -m json.tool

# 異常値の詳細確認
python -c "
import json
with open('results/performer_gpu_n576.json') as f:
    data = json.load(f)
    
print('Standard Stats:')
print(f\"  Mean: {data['standard']['mean_latency_ms']:.0f}ms\")
print(f\"  Median: {data['standard']['median_latency_ms']:.0f}ms\")
print(f\"  Memory: {data['standard']['peak_memory_mb']:.0f}MB\")

print('Performer Stats:')
print(f\"  Mean: {data['performer']['mean_latency_ms']:.0f}ms\")
print(f\"  Median: {data['performer']['median_latency_ms']:.0f}ms\")
print(f\"  Memory: {data['performer']['peak_memory_mb']:.0f}MB\")

print('Comparison:')
print(f\"  Speedup: {data['comparison']['speedup']:.2f}x\")
"
```

---

## 次のアクション

1. **即座**: デバイス配置確認（モデル・入力がGPUにあるか）
2. **短期**: num_random_features=64/128で再実行
3. **中期**: プロファイリング追加版で詳細分析
4. **長期**: L4専用の最適化パラメータ探索

---

**重要**: 0.18xは明らかに異常。デバイス配置ミスの可能性が最も高い。
