# Google Colab: Adaptive Attention Selector クイックリファレンス

**最小コピペで実行**: GPU長系列だけPerformer自動ON

---

## 🚀 セットアップ（1コマンド）

```bash
# Google Colab: リポジトリクローン＆依存インストール
!git clone https://github.com/kinoshitayoshihiro/composer4.git /content/composer4
%cd /content/composer4
!pip install -q torch transformers pytest
```

---

## ✅ テスト実行（GPU不要）

```bash
# Adaptive Attention Selectorテスト（10/10全合格を確認）
!pytest tests/test_attn_selector.py -v
```

**期待結果**: `10 passed, 1 warning in 0.35s` ✅

---

## 🧪 ベンチマーク実行

### パターン1: 自動モード（推奨）

```bash
# GPU + 長系列(N=576) → 閾値1024なのでSTANDARD選択
!python scripts/benchmark_performer_adaptive.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --n-embd 768 \
  --n-layer 12 \
  --attn auto \
  --attn-threshold 1024 \
  --num-random-features 128 \
  --output results/adaptive_auto_n576.json
```

**期待**: `✅ Attention selected: STANDARD` （576 < 1024）

### パターン2: 閾値を下げてPerformer試行

```bash
# 閾値512 → N=576でPerformer選択
!python scripts/benchmark_performer_adaptive.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --attn auto \
  --attn-threshold 512 \
  --num-random-features 256 \
  --output results/adaptive_performer_n576.json
```

**期待**: `✅ Attention selected: PERFORMER` （576 >= 512）

### パターン3: 強制Standard（本番推奨）

```bash
# 常にStandard（安全）
!python scripts/benchmark_performer_adaptive.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --attn standard \
  --output results/forced_standard.json
```

### パターン4: 強制Performer（実験用）

```bash
# 常にPerformer（比較実験用）
!python scripts/benchmark_performer_adaptive.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --attn performer \
  --num-random-features 256 \
  --output results/forced_performer.json
```

---

## 📊 結果確認

```python
# 結果JSONを読み込み
import json
from pathlib import Path

result = json.loads(Path("results/adaptive_auto_n576.json").read_text())

# 選択された注意機構を確認
print(f"Selected attention: {result['adaptive_meta']['attn']}")
print(f"Device: {result['adaptive_meta']['device']}")
print(f"Sequence length: {result['adaptive_meta']['seq_len']}")
print(f"Threshold: {result['adaptive_meta']['threshold']}")

# パフォーマンス確認
print(f"\nLatency (mean): {result['results']['latency_mean']:.2f} ms")
print(f"Latency (p95): {result['results']['latency_p95']:.2f} ms")
print(f"Per-token: {result['results']['per_token_mean']:.2f} ms")
print(f"Peak memory: {result['results']['peak_memory_mean']:.2f} MB")
```

**出力例**:
```
Selected attention: standard
Device: cuda
Sequence length: 576
Threshold: 1024

Latency (mean): 1234.56 ms
Latency (p95): 1456.78 ms
Per-token: 2.41 ms
Peak memory: 1024.00 MB
```

---

## 💾 結果をGitHub Pushする

```bash
# 環境変数でPATを設定（セッション変数推奨）
export GH_PAT="ghp_xxxxxxxxxxxxxxxxxxxx"

# リモートURL更新
git -C /content/composer4 remote set-url origin \
  https://$GH_PAT@github.com/kinoshitayoshihiro/composer4.git

# 最新を取得
git -C /content/composer4 pull --ff-only

# 結果を強制追加（.gitignoreを無視）
git -C /content/composer4 add -f results/adaptive_*.json results/forced_*.json

# コミット
git -C /content/composer4 commit -m "chore: Add Adaptive Attention benchmark results from Colab

- Auto mode (threshold=1024): N=576 → STANDARD
- Auto mode (threshold=512): N=576 → PERFORMER
- Forced Standard: baseline reference
- Forced Performer: experimental comparison

GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Push
git -C /content/composer4 push origin main

# 完了後、PATをクリア（セキュリティ）
unset GH_PAT
```

---

## 🎯 Python API使用例

### 基本的な使い方

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from ml.attn_selector import apply_adaptive_attention, AttnAutoConfig
from ml.attention_performer import replace_attention_layers

# モデル作成
config = GPT2Config(vocab_size=1000, n_embd=768, n_layer=12, n_head=12)
model = GPT2LMHeadModel(config)

# デバイス
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 系列長
seq_len = 64 + 512  # prompt + max_new_tokens

# 自動選択
cfg = AttnAutoConfig(threshold=1024, num_random_features=128, idempotent=True)
kind = apply_adaptive_attention(
    model,
    device=device,
    seq_len=seq_len,
    replace_fn=replace_attention_layers,
    cfg=cfg
)

print(f"Selected: {kind}")  # → "standard" (N=576 < 1024)
print(f"Device: {device}")
print(f"Model has _attn_kind: {hasattr(model, '_attn_kind')}")
```

### 強制指定

```python
# 常にPerformer（実験）
kind = apply_adaptive_attention(
    model,
    device="cuda",
    seq_len=128,
    replace_fn=replace_attention_layers,
    cfg=AttnAutoConfig(num_random_features=256),
    force="performer"
)
# → "performer"

# 常にStandard（安全）
kind = apply_adaptive_attention(
    model,
    device="cuda",
    seq_len=2048,
    replace_fn=replace_attention_layers,
    force="standard"
)
# → "standard"
```

### 冪等性確認

```python
# 1回目: Performer適用
kind1 = apply_adaptive_attention(
    model, device="cuda", seq_len=2048,
    replace_fn=replace_attention_layers,
    cfg=AttnAutoConfig(threshold=1024, idempotent=True)
)
print(f"1st: {kind1}, _attn_kind={model._attn_kind}")
# → 1st: performer, _attn_kind=performer

# 2回目: 再適用されない（冪等性）
kind2 = apply_adaptive_attention(
    model, device="cuda", seq_len=2048,
    replace_fn=replace_attention_layers,
    cfg=AttnAutoConfig(threshold=1024, idempotent=True)
)
print(f"2nd: {kind2}, _attn_kind={model._attn_kind}")
# → 2nd: performer, _attn_kind=performer (replace_fn呼ばれない)
```

---

## 📋 パラメータ早見表

### `benchmark_performer_adaptive.py`

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--device` | cuda | cpu/cuda |
| `--attn` | auto | auto（自動）/standard（強制）/performer（強制） |
| `--attn-threshold` | 1024 | 自動モード時の閾値 |
| `--num-random-features` | 128 | Performerのrf値（64/128/256推奨） |
| `--num-samples` | 10 | ベンチマーク実行回数 |
| `--prompt-length` | 64 | プロンプト長 |
| `--max-new-tokens` | 512 | 生成トークン数 |
| `--n-embd` | 768 | 埋め込み次元 |
| `--n-layer` | 12 | レイヤー数 |
| `--output` | （必須） | 出力JSONファイル |

### `AttnAutoConfig`

| フィールド | デフォルト | 説明 |
|----------|----------|------|
| `threshold` | 1024 | GPU時にPerformerを使う最小系列長 |
| `num_random_features` | 128 | Performerのランダム特徴数 |
| `idempotent` | True | 重複適用を防止 |

---

## 🎓 実測データ（参考）

**NVIDIA L4 GPU**での実測結果:

| 条件 | Speedup | メモリ | 判定 |
|------|---------|--------|------|
| N=576, rf=256 | **0.43x** | +277% | ❌ Performer遅い |
| N=576, rf=128 | **0.30x** | +175% | ❌ 最も遅い |
| N=576, rf=64 | **0.45x** | +124% | ❌ Performer遅い |
| N=1024, rf=256 | **0.34x** | +407% | ❌ さらに遅い |

**結論**: 現状、**Standard Attentionが2-3倍高速**

→ デフォルトthreshold=1024は実測に基づく現実的な設定

---

## 🔍 トラブルシューティング

### GPUメモリ不足

```bash
# モデルサイズを縮小
!python scripts/benchmark_performer_adaptive.py \
  --device cuda \
  --n-embd 384 \
  --n-layer 6 \
  --num-samples 5 \
  --max-new-tokens 256 \
  --output results/small_test.json
```

### CUDA利用不可

```bash
# CPUで実行（自動的にStandard選択）
!python scripts/benchmark_performer_adaptive.py \
  --device cpu \
  --num-samples 3 \
  --n-embd 256 \
  --n-layer 4 \
  --max-new-tokens 64 \
  --output results/cpu_test.json
```

### テスト失敗

```bash
# 詳細ログ付きでテスト
!pytest tests/test_attn_selector.py -v -s
```

---

## 📚 関連ドキュメント

- [Adaptive Attention Guide](./docs/ADAPTIVE_ATTENTION_GUIDE.md) - 完全ガイド
- [Performer Final Evaluation](./docs/PERFORMER_FINAL_EVALUATION.md) - 詳細実測結果
- [GPU Benchmark Analysis](./docs/GPU_BENCHMARK_ANALYSIS.md) - 根本原因分析

---

## ✅ チェックリスト

実行前:
- [ ] Google Colab GPU有効化（ランタイム → ランタイムのタイプを変更 → GPU）
- [ ] リポジトリクローン完了
- [ ] 依存パッケージインストール完了（torch, transformers, pytest）

実行中:
- [ ] テスト全合格確認（10/10 passed）
- [ ] ベンチマーク正常終了（✅ Benchmark complete!）
- [ ] 結果JSON生成確認（results/*.json）

実行後:
- [ ] adaptive_meta.attn が期待値（"standard" or "performer"）
- [ ] latency_mean, peak_memory_mean が記録されている
- [ ] GitHub Pushでエラーなし（オプション）

---

**Status**: ✅ Colab Ready  
**Tests**: 10/10 Passed  
**Recommendation**: `--attn auto --attn-threshold 1024` で安全運用
