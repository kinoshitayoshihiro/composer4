# 📦 Results フォルダのPushコマンド

## 🎯 ワンライナー（推奨）

### すべてのJSONファイルを追加
```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3 && \
git add -f results/*.json && \
git commit -m 'feat: Add all benchmark results

Benchmark Results Summary:

GPU Results (NVIDIA L4):
- performer_gpu_n576.json (N=576, rf=256)
- performer_gpu_n1024.json (N=1024, rf=256)

CPU Results (M3):
- performer_realtime_cpu_n320.json (N=320, rf=256)

Legacy Results:
- performer_benchmark.json
- stage3_performer_benchmark_long.json
- stage3_performer_benchmark_medium.json

Key Findings:
- GPU: 0.43x speedup (rf=256 too large)
- CPU: 0.71x speedup (BLAS優位)
- Next: rf=64/128 testing

Status: Analysis complete, optimization in progress' && \
git push origin main
```

---

## 📋 個別ファイル指定（選択的）

### GPU結果のみ
```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3 && \
git add -f results/performer_gpu_n576.json \
           results/performer_gpu_n1024.json && \
git commit -m 'feat: Add GPU benchmark results (L4)' && \
git push origin main
```

### CPU結果のみ
```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3 && \
git add -f results/performer_realtime_cpu_n320.json && \
git commit -m 'feat: Add CPU benchmark results (M3)' && \
git push origin main
```

---

## 🔍 現在のファイル一覧

| ファイル | サイズ | 説明 |
|---------|--------|------|
| `benchmark_output.log` | 31B | ログファイル（空に近い） |
| `performer_benchmark.json` | 841B | 初期ベンチマーク（ダミー） |
| **`performer_gpu_n576.json`** | **1.3K** | **GPU N=576 (重要)** |
| **`performer_gpu_n1024.json`** | **1.3K** | **GPU N=1024 (重要)** |
| `performer_realtime_cpu_n320.json` | 1.2K | CPU N=320 (重要) |
| `stage3_performer_benchmark_long.json` | 859B | 旧版長系列 |
| `stage3_performer_benchmark_medium.json` | 851B | 旧版中系列 |

---

## ✅ 推奨コマンド（詳細版）

```bash
# 1. カレントディレクトリ移動
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

# 2. すべてのJSONを強制追加（.gitignore無視）
git add -f results/*.json

# 3. 状態確認
git status

# 4. コミット
git commit -m 'feat: Add comprehensive benchmark results

Benchmark Results Collection:

GPU Benchmarks (NVIDIA L4, Google Colab):
- performer_gpu_n576.json
  * N=576, rf=256
  * Speedup: 0.43x, Memory: +277%
  * Finding: rf=256 too large

- performer_gpu_n1024.json
  * N=1024, rf=256
  * Speedup: 0.34x, Memory: +407%
  * Finding: Degrades with longer sequences

CPU Benchmarks (M3 Max, macOS):
- performer_realtime_cpu_n320.json
  * N=320, rf=256
  * Speedup: 0.71x
  * Finding: BLAS optimization dominates

Legacy Results:
- performer_benchmark.json (initial dummy metrics)
- stage3_performer_benchmark_long.json (N=576 old)
- stage3_performer_benchmark_medium.json (N=96 old)
- benchmark_output.log (execution log)

Key Insights:
1. num_random_features=256 is too large for both CPU/GPU
2. exp() overhead ~10x, cumsum ~5x, memory ~14x
3. Theory (O(N·r)) ≠ Implementation (20-30x constants)
4. Next: Test rf=64/128 for practical speedup

Status: 🚧 Optimization in progress
References: docs/GPU_BENCHMARK_ANALYSIS.md, docs/GPU_BENCHMARK_VALUE.md'

# 5. Push
git push origin main
```

---

## 🚀 最速コマンド（コピペ用）

```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3 && git add -f results/*.json && git commit -m 'feat: Add all benchmark results (GPU+CPU)' && git push origin main
```

---

## ⚠️ 注意事項

### .gitignore設定確認
```bash
# 現在の設定
cat .gitignore | grep -A2 "認証情報"

# 出力:
# 認証情報
# *.json
# !package.json
# !results/*.json  # ← これで許可されているはず
```

### `-f`フラグの必要性
```bash
# .gitignoreで除外されているため-fが必要
git add -f results/*.json

# -fなしだと無視される
git add results/*.json
# > The following paths are ignored by one of your .gitignore files
```

---

## 📊 コミット後の確認

```bash
# 最新コミット確認
git log --oneline -1

# 追加されたファイル確認
git show --name-only

# リモート確認
git ls-remote origin main
```

---

## 🎯 実行例

```bash
$ cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

$ git add -f results/*.json

$ git status
On branch main
Changes to be committed:
  new file:   results/benchmark_output.log
  modified:   results/performer_gpu_n1024.json
  modified:   results/performer_gpu_n576.json
  new file:   results/performer_realtime_cpu_n320.json
  ...

$ git commit -m 'feat: Add all benchmark results'
[main abc1234] feat: Add all benchmark results
 7 files changed, 150 insertions(+)

$ git push origin main
Enumerating objects: 12, done.
Writing objects: 100% (12/12), done.
To https://github.com/kinoshitayoshihiro/composer4.git
   75b764a61..abc123456  main -> main
```

---

## 📝 コミットメッセージのバリエーション

### シンプル版
```bash
git commit -m 'feat: Add benchmark results (GPU N=576/1024, CPU N=320)'
```

### 詳細版（上記参照）

### 一行版
```bash
git commit -m 'feat: Add GPU/CPU benchmark results - rf=256 optimization needed'
```

---

**推奨**: 詳細版コミットメッセージで全体像を記録することをお勧めします！
