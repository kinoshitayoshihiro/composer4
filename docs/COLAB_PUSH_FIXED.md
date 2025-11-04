# 🔧 修正版 Colab Push コマンド

## 問題点の修正

### ❌ 元のコマンド（エラーあり）
```bash
# シェルエスケープエラー
!git commit -m "feat: Add GPU benchmark results on NVIDIA L4

# ダブルクォートが閉じていない
!git push origin main
```

### ✅ 修正版コマンド

---

## 📊 結果をローカルでPush（推奨）

### Step 1: Colabで結果ダウンロード
```python
from google.colab import files

# JSON結果（.gitignore修正済み）
files.download('results/performer_gpu_n576.json')

# グラフ（もし生成していれば）
# files.download('results/gpu_benchmark_n576.png')
```

### Step 2: ローカルでPush
```bash
# ローカルマシンで実行
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

# 最新の.gitignore取得
git pull origin main

# ダウンロードしたファイルをresultsディレクトリに配置
# （ブラウザのダウンロードフォルダから移動）

# Git追加（.gitignore修正済みで追加可能）
git add .gitignore
git add results/performer_gpu_n576.json

# 結果を確認してからコミット
cat results/performer_gpu_n576.json | python -m json.tool | head -50

# コミット
git commit -m 'feat: Add GPU benchmark results on NVIDIA L4

GPU Benchmark Results (Google Colab NVIDIA L4):

N=576 (prompt=64, max_new_tokens=512):
- Standard Attention:  [実測値]ms, [実測値]MB
- Performer Attention: [実測値]ms, [実測値]MB
- Speedup: 0.18x (異常値 - 要調査)

Environment:
- GPU: NVIDIA L4 (24GB VRAM)
- CUDA: 12.6
- PyTorch: 2.5.0+cu126
- Transformers: 4.46.0

Issue:
- 期待値1.37xに対して0.18xと異常に遅い
- 原因調査中（デバイス配置、CUDA同期、num_random_features）

Files:
- .gitignore (allow results/*.json, results/*.png)
- results/performer_gpu_n576.json
- docs/GPU_BENCHMARK_ANOMALY.md'

# Push
git push origin main
```

---

## 🚀 Colabで直接Push（Personal Access Token使用）

### 準備
```bash
# GitHub Personal Access Token取得
# 1. GitHub → Settings → Developer settings → Personal access tokens
# 2. Generate new token (classic)
# 3. Scopes: repo (全選択)
# 4. トークンをコピー（ghp_xxxxxxxxxxxxxxxxxxxx）
```

### 実行
```bash
# Git設定（初回のみ）
!git config --global user.name "kinoshitayoshihiro"
!git config --global user.email "shimogami88@gmail.com"

# 最新コード取得
!git pull origin main

# .gitignore確認
!cat .gitignore | grep -A2 "認証情報"

# 結果追加（.gitignore修正済み）
!git add results/performer_gpu_n576.json

# コミット（シングルクォート使用）
!git commit -m 'feat: Add GPU benchmark results on NVIDIA L4

N=576 Results:
- Speedup: 0.18x (anomaly)
- Investigation required

GPU: NVIDIA L4
Date: 2025-10-13

Files:
- results/performer_gpu_n576.json'

# Push（トークン使用）
!git push https://YOUR_TOKEN_HERE@github.com/kinoshitayoshihiro/composer4.git main
```

---

## 🔍 0.18x 異常値の調査コマンド

### デバイス確認
```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel
import sys
sys.path.insert(0, '/content/composer4')
from ml.attention_performer import replace_attention_layers

# GPU確認
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

# モデル作成
config = GPT2Config(vocab_size=1000, n_embd=768, n_layer=12, n_head=12)
model = GPT2LMHeadModel(config)

# デバイス確認（移動前）
print("Model device (before):", next(model.parameters()).device)

# GPU移動
device = torch.device('cuda')
model = model.to(device)

# デバイス確認（移動後）
print("Model device (after):", next(model.parameters()).device)

# Performer置換
replace_attention_layers(model, num_random_features=256)

# デバイス確認（置換後）
print("Model device (after replacement):", next(model.parameters()).device)

# 推論テスト
input_ids = torch.randint(0, 1000, (1, 64)).to(device)
print("Input device:", input_ids.device)

# タイミング計測
import time
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=100, use_cache=False)

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Inference time: {elapsed*1000:.0f}ms")
print(f"Output shape: {output.shape}")
```

### 修正版ベンチマーク（num_random_features調整）
```bash
# 小さい値でテスト
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 5 \
  --prompt-length 64 \
  --max-new-tokens 256 \
  --n-embd 768 \
  --n-layer 12 \
  --num-random-features 64 \
  --output results/test_rf64.json

# 結果確認
!python -c "
import json
with open('results/test_rf64.json') as f:
    data = json.load(f)
    print(f\"Speedup (rf=64): {data['comparison']['speedup']:.2f}x\")
"
```

---

## 📝 コミットメッセージテンプレート（異常値版）

```
feat: Add GPU benchmark results on NVIDIA L4

GPU Benchmark Results (Google Colab NVIDIA L4):

N=576 (prompt=64, max_new_tokens=512, num_random_features=256):
- Standard Attention:  [実測値]ms, [実測値]MB
- Performer Attention: [実測値]ms, [実測値]MB
- Speedup: 0.18x ⚠️ ANOMALY

Environment:
- GPU: NVIDIA L4 (24GB VRAM)
- CUDA: 12.6
- PyTorch: 2.5.0+cu126
- Transformers: 4.46.0
- Python: 3.12

Issue Analysis:
- Expected: 1.37x speedup (theoretical)
- Actual: 0.18x speedup (異常値)
- Possible causes:
  1. Device placement issue (model/input not on GPU)
  2. CUDA synchronization problem
  3. num_random_features=256 too large for L4
  4. Memory swapping

Next Steps:
- Verify device placement
- Test with num_random_features=64/128
- Add detailed profiling
- Investigate L4-specific optimization

Files:
- .gitignore (allow results/*.json)
- results/performer_gpu_n576.json
- docs/GPU_BENCHMARK_ANOMALY.md

Status: 🚧 Investigation in progress
```

---

## 🎯 推奨アクション

### 即座に実行
1. **デバイス確認**: 上記のデバイス確認コードを実行
2. **num_random_features調整**: 64または128で再実行
3. **結果確認**: `speedup`が1.0x以上になるか確認

### 結果が正常になったら
```bash
# Colabでダウンロード
from google.colab import files
files.download('results/performer_gpu_fixed.json')

# ローカルでPush
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3
git add results/performer_gpu_fixed.json
git commit -m 'feat: Add corrected GPU benchmark results

Speedup: [正常値]x
num_random_features: [調整後の値]'
git push origin main
```

---

**重要**: 0.18xは明らかに異常。まずデバイス配置を確認してください。
