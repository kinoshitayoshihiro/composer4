# Adaptive Attention Selector Guide

**自動切替システム**: GPU&長系列のみPerformer、それ以外はStandard Attention

## 📋 概要

Adaptive Attention Selectorは、デバイスと系列長に基づいて最適な注意機構を**自動選択**します。

### 選択ロジック

```python
if force:
    return force  # 強制指定
elif device == "cuda" and seq_len >= threshold:
    return "performer"  # GPU + 長系列
else:
    return "standard"  # その他
```

### デフォルト設定

| パラメータ | デフォルト値 | 説明 |
|----------|------------|------|
| `threshold` | 1024 | GPU時にPerformerを使う最小系列長 |
| `num_random_features` | 128 | Performerのランダム特徴数（64/128/256推奨） |
| `idempotent` | True | 重複適用を防止 |

**重要**: 実測結果（N=576でPerformer 0.30-0.45x）に基づき、デフォルトthresholdは1024に設定されています。

---

## 🚀 使い方

### 基本的な使い方

```python
from ml.attn_selector import apply_adaptive_attention, AttnAutoConfig
from ml.attention_performer import replace_attention_layers

# デフォルト設定で自動選択
cfg = AttnAutoConfig(threshold=1024, num_random_features=128)
kind = apply_adaptive_attention(
    model,
    device="cuda",
    seq_len=576,
    replace_fn=replace_attention_layers,
    cfg=cfg
)

print(f"Selected: {kind}")  # → "standard" (N=576 < 1024)
```

### カスタム閾値

```python
# 長系列専用（閾値を下げる）
cfg = AttnAutoConfig(threshold=512, num_random_features=256)
kind = apply_adaptive_attention(
    model,
    device="cuda",
    seq_len=768,
    replace_fn=replace_attention_layers,
    cfg=cfg
)
# → "performer" (768 >= 512)
```

### 強制指定

```python
# 常にPerformer（実験用）
kind = apply_adaptive_attention(
    model,
    device="cuda",
    seq_len=128,
    replace_fn=replace_attention_layers,
    force="performer"  # 閾値無視
)
# → "performer"

# 常にStandard（安全）
kind = apply_adaptive_attention(
    model,
    device="cuda",
    seq_len=2048,
    replace_fn=replace_attention_layers,
    force="standard"  # 閾値無視
)
# → "standard"
```

---

## 🧪 ベンチマーク実行

### 新しいスクリプト: `benchmark_performer_adaptive.py`

```bash
# 自動モード（閾値ベース）
python scripts/benchmark_performer_adaptive.py \
  --device cuda \
  --num-samples 10 \
  --prompt-length 64 \
  --max-new-tokens 512 \
  --attn auto \
  --attn-threshold 1024 \
  --num-random-features 128 \
  --output results/adaptive_attn_n576.json

# Performer強制
python scripts/benchmark_performer_adaptive.py \
  --device cuda \
  --attn performer \
  --output results/forced_performer.json

# Standard強制
python scripts/benchmark_performer_adaptive.py \
  --device cuda \
  --attn standard \
  --output results/forced_standard.json
```

### パラメータ

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--device` | cuda | cpu/cuda |
| `--attn` | auto | auto（自動）/standard（強制）/performer（強制） |
| `--attn-threshold` | 1024 | 自動モード時の閾値 |
| `--num-random-features` | 128 | Performerのrf値 |
| `--num-samples` | 10 | ベンチマーク実行回数 |
| `--prompt-length` | 64 | プロンプト長 |
| `--max-new-tokens` | 512 | 生成トークン数 |

---

## 📊 実測結果（参考）

**NVIDIA L4 GPU**での実測:

| 条件 | Speedup | メモリ | 推奨 |
|------|---------|--------|------|
| N=576, rf=256 | 0.43x | +277% | ❌ Standard使用 |
| N=576, rf=128 | 0.30x | +175% | ❌ Standard使用 |
| N=576, rf=64 | 0.45x | +124% | ❌ Standard使用 |
| N=1024, rf=256 | 0.34x | +407% | ❌ Standard使用 |

**結論**: 現状、**全条件でStandard Attentionが2-3倍高速**

したがって、デフォルト設定は事実上「常にStandard」として動作します。

---

## 🔧 冪等性保証

```python
# 1回目: Performer適用
kind1 = apply_adaptive_attention(
    model, device="cuda", seq_len=2048,
    replace_fn=replace_attention_layers,
    cfg=AttnAutoConfig(threshold=1024, idempotent=True)
)
# → "performer"

# 2回目: 再適用されない（idempotent=True）
kind2 = apply_adaptive_attention(
    model, device="cuda", seq_len=2048,
    replace_fn=replace_attention_layers,
    cfg=AttnAutoConfig(threshold=1024, idempotent=True)
)
# → "performer" (replace_fn呼ばれない)

# モデルに _attn_kind 属性が記録される
assert model._attn_kind == "performer"
```

---

## 🎯 推奨設定

### drumgenerator（本番）

```python
# 安全第一: 常にStandard
cfg = AttnAutoConfig(
    threshold=float('inf'),  # 事実上無効化
    idempotent=True
)
kind = apply_adaptive_attention(
    model,
    device=device,
    seq_len=seq_len,
    replace_fn=replace_attention_layers,
    cfg=cfg
)
# → 常に "standard"
```

### 実験用（長系列）

```python
# GPU + 超長系列でPerformer試行
cfg = AttnAutoConfig(
    threshold=512,  # 閾値を下げる
    num_random_features=256,  # rf増やす
    idempotent=True
)
```

### A/Bテスト

```python
# 明示的に比較
standard_model = create_model()
apply_adaptive_attention(standard_model, ..., force="standard")

performer_model = create_model()
apply_adaptive_attention(performer_model, ..., force="performer")

# ベンチマーク比較
...
```

---

## ✅ テスト

```bash
# 全テスト実行（GPU不要）
pytest tests/test_attn_selector.py -v

# 10/10テスト全合格を確認
```

### テスト内容

1. **CPU常にStandard**: CPU環境では必ずStandard
2. **GPU閾値動作**: seq_len >= threshold でPerformer
3. **強制オーバーライド**: `force="performer"/"standard"`
4. **冪等性**: 重複適用を防止
5. **replace_fn呼出確認**: Performerの時だけ呼ばれる
6. **設定デフォルト/カスタム**: AttnAutoConfig検証

---

## 📦 ファイル構成

```
ml/
  attn_selector.py          # 本体（171行）
    - AttnAutoConfig
    - select_attention()
    - apply_adaptive_attention()

scripts/
  benchmark_performer_adaptive.py  # ベンチマークツール
    - 自動/強制切替
    - 詳細メトリクス記録

tests/
  test_attn_selector.py     # テスト（10ケース）
    - GPU不要で実行可能
    - 全機能カバー

docs/
  ADAPTIVE_ATTENTION_GUIDE.md  # 本ドキュメント
  PERFORMER_FINAL_EVALUATION.md  # 実測詳細
```

---

## 🔍 Next Steps

### FlashAttention v2検討

```bash
pip install flash-attn
```

**期待効果**:
- 理論: 2-3倍高速化
- メモリ: 大幅削減（O(N) I/O）
- GPU最適化: CUDA kernelレベル

**注意**: Performerの教訓（理論≠実装）を踏まえ、**必ず実測ベンチマーク**を実施すること。

---

## 📚 関連ドキュメント

- [Performer Final Evaluation](./PERFORMER_FINAL_EVALUATION.md) - 詳細実測結果
- [GPU Benchmark Analysis](./GPU_BENCHMARK_ANALYSIS.md) - 根本原因分析
- [Colab Setup](../COLAB_SETUP_PERFORMER.md) - Google Colab実行手順

---

## 🎓 学んだこと

1. **理論的複雂度 ≠ 実装性能**
   - O(N·r) < O(N²)でも定数係数で逆転
   - GPU BLAS最適化（cuBLAS）が圧倒的

2. **実測の絶対的必要性**
   - 論文の理論値を盲信しない
   - ハードウェア・実装で結果は変わる

3. **柔軟な設計の価値**
   - Adaptive Selectorで将来対応可能
   - 実験・切替が容易

---

**Status**: ✅ Production Ready  
**Tests**: 10/10 Passed  
**Recommendation**: Standard Attention継続、FlashAttention v2検討
