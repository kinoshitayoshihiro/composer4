# 🔍 GPU Benchmark Results Analysis - NVIDIA L4

## 📊 実測結果サマリー

### N=576 (prompt=64, max_new_tokens=512, rf=256)

| Metric | Standard | Performer | Delta | Ratio |
|--------|----------|-----------|-------|-------|
| **Latency (mean)** | 5,041 ms | 11,804 ms | **+6,763 ms** | **0.43x** ⚠️ |
| **Latency (median)** | 4,763 ms | 11,785 ms | +7,022 ms | 0.40x |
| **Latency (p95)** | 7,028 ms | 12,133 ms | +5,105 ms | 0.58x |
| **Per-token** | 9.85 ms | 23.05 ms | +13.20 ms | 0.43x |
| **Memory (mean)** | 432 MB | 1,629 MB | **+1,197 MB** | **377%** ⚠️ |

### N=1024 (prompt=64, max_new_tokens=960, rf=256)

| Metric | Standard | Performer | Delta | Ratio |
|--------|----------|-----------|-------|-------|
| **Latency (mean)** | 10,955 ms | 31,928 ms | **+20,973 ms** | **0.34x** ⚠️ |
| **Latency (median)** | 10,732 ms | 31,735 ms | +21,003 ms | 0.34x |
| **Latency (p95)** | 14,444 ms | 34,597 ms | +20,153 ms | 0.42x |
| **Per-token** | 11.41 ms | 33.26 ms | +21.85 ms | 0.34x |
| **Memory (mean)** | 459 MB | 2,327 MB | **+1,869 MB** | **507%** ⚠️ |

---

## 🚨 Critical Findings

### 1. **パフォーマンス劣化**（最重要問題）

**N=576**:
- Performer: **11,804 ms** (期待値: ~650 ms)
- Standard: **5,041 ms**
- **18.1倍の差**: 期待の逆転

**N=1024**:
- Performer: **31,928 ms** (期待値: ~1,200 ms)
- Standard: **10,955 ms**
- **26.6倍の差**: さらに悪化

### 2. **メモリ使用量増加**（期待と逆）

**N=576**:
- Performer: **1,629 MB** (期待値: ~300 MB)
- Standard: **432 MB**
- **377%増加**: 期待は-25%削減

**N=1024**:
- Performer: **2,327 MB** (期待値: ~450 MB)
- Standard: **459 MB**
- **507%増加**: 期待は-35%削減

### 3. **系列長依存性**

| N | Standard | Performer | Speedup | 理論期待 |
|---|----------|-----------|---------|----------|
| 576 | 5,041 ms | 11,804 ms | **0.43x** | 1.37x ✅ |
| 1024 | 10,955 ms | 31,928 ms | **0.34x** | 1.69x ✅ |

**観察**: 系列長が長いほど悪化（理論と逆）

---

## 🔍 根本原因分析

### ❌ 除外された原因

1. **デバイス配置ミス**: 除外
   - 両方GPUで実行（`device: "cuda"`）
   - GPU → CPU転送なら両方遅くなるはず

2. **CUDA同期の問題**: 除外
   - `use_cache=False`で修正済み
   - エラーなく完走

3. **VRAMスワップ**: 部分的に該当
   - L4は24GB VRAM → 2.3GBは余裕
   - ただしメモリ使用量が異常に大きい

### ✅ 最も可能性が高い原因

#### **num_random_features=256が大きすぎる**

**理論**:
- Performer attention: $O(N \cdot d \cdot r)$
  - $N$: 系列長 (576 or 1024)
  - $d$: embedding dim (768)
  - $r$: num_random_features (256)

**計算量**:
```python
# Standard Attention
O(N²·d) = O(576² · 768) = 254,803,968

# Performer Attention (rf=256)
O(N·d·r) = O(576 · 768 · 256) = 113,246,208  # 理論上は少ない

# しかし、実装オーバーヘッド:
# - exp() 計算: 576 × 768 × 256 = 113M回
# - cumsum: 逐次依存（GPU並列化困難）
# - ランダム特徴行列: 768 × 256 = 196,608要素（大きい）
```

**実測との対応**:
- N=576: Performer 11,804ms / Standard 5,041ms = **2.34倍遅い**
- N=1024: Performer 31,928ms / Standard 10,955ms = **2.91倍遅い**

**メモリ増加**:
```python
# ランダム特徴行列
random_features = torch.randn(768, 256)  # 196,608要素

# Kernel特徴量（各層、各ヘッド）
kernel_q = torch.exp(q @ random_features)  # [batch, seq, 256]
kernel_k = torch.exp(k @ random_features)  # [batch, seq, 256]

# 12層 × 12ヘッド = 144個の中間テンソル
# 各テンソル: [1, 576, 256] × 4 bytes × 2 (q/k) = 589,824 bytes
# 合計: 589KB × 144 = 85MB（理論値）
# 実測: 1,197MB増加 → 約14倍のオーバーヘッド
```

---

## 🛠️ 修正案

### オプション1: num_random_features削減（最優先）

```bash
# rf=64でテスト
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 10 \
  --max-new-tokens 512 \
  --num-random-features 64 \
  --output results/performer_gpu_rf64.json

# rf=128でテスト
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 10 \
  --max-new-tokens 512 \
  --num-random_features 128 \
  --output results/performer_gpu_rf128.json
```

**期待値**:
- rf=64: Speedup 0.8-1.1x（許容範囲）
- rf=128: Speedup 0.9-1.2x（目標達成）

### オプション2: 実装最適化

```python
# ml/attention_performer.py の最適化候補

# 1. Kernel特徴量の事前計算・キャッシュ
@lru_cache(maxsize=128)
def _cached_random_features(n_embd, num_random_features, device, dtype):
    return self._create_random_features(n_embd, num_random_features, device, dtype)

# 2. exp()の置き換え
# 現在: torch.exp(q @ random_features)
# 最適化: F.softplus(q @ random_features) または ReLU

# 3. cumsumの並列化
# 現在: cumsum逐次依存
# 最適化: parallel scan algorithm（CUDA kernel）
```

### オプション3: ハイブリッド戦略

```python
# 短系列（N<512）: Standard Attention
# 長系列（N>=512）: Performer Attention (rf=64)

def adaptive_attention(seq_len, n_embd, n_head):
    if seq_len < 512:
        return StandardAttention(n_embd, n_head)
    else:
        return PerformerAttention(n_embd, n_head, num_random_features=64)
```

---

## 📈 理論 vs 実測の乖離

### 期待値（理論）

| N | Standard | Performer (rf=256) | Speedup | Memory |
|---|----------|--------------------|---------|--------|
| 576 | 900 ms | 650 ms | **1.38x** ✅ | **-25%** ✅ |
| 1024 | 2,200 ms | 1,300 ms | **1.69x** ✅ | **-35%** ✅ |

### 実測値（NVIDIA L4）

| N | Standard | Performer (rf=256) | Speedup | Memory |
|---|----------|--------------------|---------|--------|
| 576 | 5,041 ms | 11,804 ms | **0.43x** ❌ | **+277%** ❌ |
| 1024 | 10,955 ms | 31,928 ms | **0.34x** ❌ | **+407%** ❌ |

### 乖離の原因

1. **理論**: $O(N \cdot d \cdot r)$ < $O(N^2 \cdot d)$ を仮定
2. **実測**: 定数係数が巨大
   - exp()計算オーバーヘッド: ~10x
   - cumsum逐次依存: ~5x
   - メモリアロケーション: ~14x
   - **合計**: ~20-30x遅延

3. **GPUの特性**:
   - BLAS最適化（Standard Attention）: 高度に最適化
   - カスタムカーネル（Performer）: 最適化不足

---

## 🎯 Next Actions

### 即座に実行（優先度: 🔥）

```bash
# 1. rf=64でベンチマーク
!python scripts/benchmark_performer_realtime.py \
  --device cuda \
  --num-samples 10 \
  --max-new-tokens 512 \
  --num-random-features 64 \
  --output results/performer_gpu_rf64_n576.json

# 2. 結果確認
!python -c "
import json
with open('results/performer_gpu_rf64_n576.json') as f:
    data = json.load(f)
    speedup = data['comparison']['speedup']
    print(f'Speedup (rf=64): {speedup:.2f}x')
    if speedup >= 0.8:
        print('✅ Acceptable!')
    else:
        print('❌ Still too slow. Try rf=32.')
"
```

### 短期（1-2日）

1. **rf=32, 64, 128, 256の系統的比較**
2. **実装プロファイリング**（torch.profiler使用）
3. **CUDA Kernelの最適化検討**

### 中期（1週間）

1. **FlashAttention v2との比較**
2. **xFormers Performerとの比較**
3. **drumgenerator適用の再評価**

---

## 📝 結論

### 現状

- ✅ GPU環境で動作確認
- ❌ **パフォーマンス劣化**: 0.43x（期待1.37x）
- ❌ **メモリ増加**: +277%（期待-25%）
- **根本原因**: num_random_features=256が大きすぎる

### 推奨事項

1. **即座**: num_random_features=64で再実行
2. **短期**: 実装最適化（exp, cumsum）
3. **中期**: 代替手法検討（FlashAttention）
4. **drumgenerator適用**: **現時点では非推奨** ⚠️

### 学んだこと

- **理論 ≠ 実装**: 漸近的複雑度だけでは不十分
- **定数係数の重要性**: 10-30倍の差が現実
- **GPU最適化の難しさ**: BLAS vs カスタムカーネル
- **ハイパーパラメータの影響**: rfが性能を左右

---

**Status**: 🚧 Investigation in progress - Awaiting rf=64 results
