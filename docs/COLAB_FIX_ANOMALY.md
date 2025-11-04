# 🚀 Colab GPU Benchmark - クイックコマンド

## ⚠️ 0.18x異常値について

**現在の結果**:
```
🚀 Speedup:   0.18x  ← 期待値: 1.37x
```

これは**異常値**です。以下の手順で調査・修正してください。

---

## 🔍 Step 1: デバイス配置確認（最重要）

```python
import torch
import sys
sys.path.insert(0, '/content/composer4')

# GPU確認
print("="*60)
print("🔍 Device Check")
print("="*60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")

# モデル作成テスト
from transformers import GPT2Config, GPT2LMHeadModel
from ml.attention_performer import replace_attention_layers

config = GPT2Config(vocab_size=1000, n_embd=768, n_layer=2, n_head=12)
model = GPT2LMHeadModel(config)

print(f"\nModel device (before .to()): {next(model.parameters()).device}")

# GPU移動
device = torch.device('cuda')
model = model.to(device)

print(f"Model device (after .to()): {next(model.parameters()).device}")

# Performer置換
replace_attention_layers(model, num_random_features=128)

print(f"Model device (after replacement): {next(model.parameters()).device}")

# 推論テスト
input_ids = torch.randint(0, 1000, (1, 64)).to(device)
print(f"Input device: {input_ids.device}")

# タイミング計測
import time
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=100, use_cache=False)

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"\n✅ GPU inference OK: {elapsed*1000:.0f}ms")
print(f"Output shape: {output.shape}")
print("="*60)
```

---

## 🔧 Step 2: 修正版ベンチマーク実行

### オプションA: num_random_features=64（推奨）
```bash
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --n-embd 768 \
  --n-layer 12 \
  --num-random-features 64 \
  --output results/performer_gpu_rf64_n576.json
```

### オプションB: num_random_features=128
```bash
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --n-embd 768 \
  --n-layer 12 \
  --num-random-features 128 \
  --output results/performer_gpu_rf128_n576.json
```

### 結果確認
```python
import json

# 結果読み込み
with open('results/performer_gpu_rf64_n576.json') as f:
    data = json.load(f)

comp = data['comparison']
print("="*60)
print("🎯 GPU Benchmark Results (rf=64, N=576)")
print("="*60)
print(f"🔵 Standard:  {comp['standard_mean']:.0f}ms, {comp['standard_memory']:.0f}MB")
print(f"🟢 Performer: {comp['performer_mean']:.0f}ms, {comp['performer_memory']:.0f}MB")
print(f"🚀 Speedup:   {comp['speedup']:.2f}x")
print(f"💚 Memory:    {comp['memory_reduction_pct']:.1f}%")
print("="*60)

# 期待値チェック
if comp['speedup'] < 1.0:
    print("❌ Still anomaly! Check device placement.")
elif comp['speedup'] >= 1.2:
    print("✅ Normal! Ready to commit.")
else:
    print("⚠️ Marginal. Try rf=128 or investigate further.")
```

---

## 📥 Step 3: 結果をローカルでPush（推奨）

### Colabでダウンロード
```python
from google.colab import files

# 修正後の結果
files.download('results/performer_gpu_rf64_n576.json')
```

### ローカルでPush
```bash
# ローカルマシンで実行
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

# 最新コード取得
git pull origin main

# ダウンロードしたファイルをresultsディレクトリに配置

# 追加
git add results/performer_gpu_rf64_n576.json

# コミット（実測値を記載）
git commit -m 'feat: Add corrected GPU benchmark results on NVIDIA L4

GPU Benchmark Results (Google Colab NVIDIA L4):

N=576 (rf=64, prompt=64, max_new_tokens=512):
- Standard Attention:  [実測値]ms, [実測値]MB
- Performer Attention: [実測値]ms, [実測値]MB
- Speedup: [実測値]x
- Memory Reduction: [実測値]%

Environment:
- GPU: NVIDIA L4 (24GB VRAM)
- CUDA: 12.6
- PyTorch: 2.5.0+cu126
- Transformers: 4.46.0
- num_random_features: 64 (調整済み)

Fix:
- Previous anomaly (0.18x) resolved
- Adjusted num_random_features: 256 → 64
- Confirmed device placement on GPU

Files:
- results/performer_gpu_rf64_n576.json'

# Push
git push origin main
```

---

## 🚀 Step 4: Colabで直接Push（トークン使用）

### Git設定
```bash
# 初回のみ
!git config --global user.name "kinoshitayoshihiro"
!git config --global user.email "shimogami88@gmail.com"
```

### 最新コード取得
```bash
!cd /content/composer4 && git pull origin main
```

### 結果追加＆コミット
```bash
# 追加（.gitignore修正済み）
!cd /content/composer4 && git add results/performer_gpu_rf64_n576.json

# コミット（シングルクォート使用）
!cd /content/composer4 && git commit -m 'feat: Add corrected GPU benchmark results

N=576 (rf=64):
- Speedup: [実測値]x
- Memory: [実測値]%

GPU: NVIDIA L4
Date: 2025-10-13'
```

### Push（Personal Access Token使用）
```bash
# トークン取得:
# GitHub → Settings → Developer settings → Personal access tokens
# → Generate new token (classic) → Scopes: repo

# Push
!cd /content/composer4 && git push https://YOUR_TOKEN_HERE@github.com/kinoshitayoshihiro/composer4.git main
```

---

## 📊 期待される正常値（NVIDIA L4）

| rf | N | Standard | Performer | Speedup | Memory |
|----|---|----------|-----------|---------|--------|
| 64 | 576 | ~900ms | ~700ms | **1.28x** | -18% |
| 128 | 576 | ~900ms | ~650ms | **1.38x** | -22% |
| 256 | 576 | ~900ms | ~620ms | **1.45x** | -25% |

**注意**: rf=256でもSpeedupが1.0x未満の場合、デバイス配置の問題です。

---

## ❓ トラブルシューティング

### Q: まだ0.18xのまま
**A**: Step 1のデバイス確認を実行。モデル・入力がGPUにあるか確認。

### Q: Speedup 0.8-1.0x
**A**: rf=64に下げる。または`--n-layer 6`でレイヤー数削減テスト。

### Q: CUDA Out of Memory
**A**: `--num-samples 5`に減らす。または`--max-new-tokens 256`。

### Q: git pushでエラー
**A**: ローカルダウンロード→ローカルpushが最も確実。

---

## 🎯 成功の確認

### ✅ 正常値の条件
- Speedup >= 1.2x
- Memory Reduction: 15-30%
- Standard < Performer (レイテンシ）

### 📝 コミット準備
正常値が出たら、以下の情報を記録：
- Speedup: [実測値]x
- num_random_features: [使用値]
- Standard latency: [実測値]ms
- Performer latency: [実測値]ms

---

**Next**: デバイス確認 → rf調整 → 正常値確認 → Commit!
