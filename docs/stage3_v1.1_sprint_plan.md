# Stage3 v1.1 Quality Enhancement Sprint

**期間**: 2週間 (10日間実働)  
**目標**: 生成品質の底上げ - インフラは維持、聴感とKPIで"わかる改善"を実現

## スプリント目標

Stage3 v1.0の堅牢なインフラ基盤を保ちつつ、以下4点を並行実装:

1. ✅ **GrooVAE/PerformanceRNN Humanizer統合** (優先度: 最高)
2. ✅ **REMI/MuMIDI部分導入** (優先度: 高)
3. ✅ **線形注意適用** (優先度: 中)
4. ✅ **外部ベンチ評価CI化** (優先度: 高)

## 実装計画 (優先度順)

### Day 1-3: GrooVAE Humanizer統合 (工数: 小)

**目的**: Velocity/Timing軸のKPI直上げ

**実装内容**:
```python
# scripts/humanize_with_groovae.py (新規)
- Magenta GrooVAE事前学習モデルのロード
- Stage3出力MIDI → GrooVAE → 表現豊かなMIDI
- retry_presets.pyの前段に差し込み

# ml/stage3_infer.py (拡張)
- --humanize フラグ追加
- generate() → humanize() → save_midi() のパイプライン

# tests/test_humanizer.py (新規)
- Velocity分布のK-S検定
- オンセットJitter分布の検証
- 元のピッチ/拍子構造が保持されることを確認
```

**成功基準**:
- LamdaスコアのVelocity軸: +5pt以上
- Timing軸: +3pt以上
- 構造破壊: 0件 (bar/beat違反率が増えない)

**デリバラブル**:
- `scripts/humanize_with_groovae.py` (150行)
- `tests/test_humanizer.py` (10テスト)
- `docs/humanizer_integration.md`
- CI追加: Velocity/Timing回帰テスト

---

### Day 4-6: REMI/MuMIDI部分導入 (工数: 中)

**目的**: 拍子・和声・楽器役割の整合性向上

**実装内容**:
```python
# ml/tokenizer_remi.py (新規)
class REMITokenizer:
    """REMI-style tokenizer with DURATION/CHORD/ROLE"""
    
    def __init__(self, vocab_config):
        # 既存トークン + REMI拡張
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<cond_end>"]
        self.structure_tokens = ["<BAR>", "<BEAT>", "<TSIG_X_Y>", "<TEMPO_X>"]
        
        # REMI拡張トークン
        self.duration_tokens = ["DUR_1/16", "DUR_1/8", "DUR_1/4", ..., "DUR_2"]
        self.chord_tokens = ["CHORD_C", "CHORD_Dm", "CHORD_G7", ...]
        self.role_tokens = ["ROLE_KICK", "ROLE_SNARE", "ROLE_HIHAT", ...]
    
    def encode_midi(self, midi_file):
        # バージョン互換フラグ
        if self.remi_enabled:
            return self._encode_remi(midi_file)
        else:
            return self._encode_legacy(midi_file)

# scripts/migrate_tokenizer.py (新規)
- 既存データの互換変換スクリプト
- v1.0 tokenizer → v1.1 REMI tokenizer
- --dry-run モードで影響範囲チェック

# tests/test_tokenizer_remi.py (新規)
- REMI拡張トークンのエンコード/デコード
- 後方互換性テスト (v1.0データが読める)
- DURATION/CHORDの抽出精度検証
```

**段階導入戦略**:
1. **Phase 1** (Day 4): DURATION トークンのみ追加 → 既存モデルでテスト
2. **Phase 2** (Day 5): CHORD トークン追加 → 和声整合性検証
3. **Phase 3** (Day 6): ROLE トークン追加 → ドラムパート精度向上

**成功基準**:
- 拍子違反率: <2% (現行3%から改善)
- 和声進行の妥当性: 手動評価10サンプルで8/10以上
- トークナイザ変換成功率: 100% (全データ移行可能)

**デリバラブル**:
- `ml/tokenizer_remi.py` (400行)
- `scripts/migrate_tokenizer.py` (200行)
- `tests/test_tokenizer_remi.py` (15テスト)
- `docs/remi_migration_guide.md`
- README更新: トークナイザバージョン管理セクション

---

### Day 7-8: 外部ベンチ評価CI化 (工数: 小〜中)

**目的**: 一般化性能の客観評価・回帰検知

**実装内容**:
```python
# scripts/eval_external_benchmarks.py (新規)
class ExternalBenchmarkEvaluator:
    """Evaluate on public datasets"""
    
    DATASETS = {
        "groove": {
            "path": "data/external/groove_midi",
            "subset": 100,  # サンプル数
            "metrics": ["velocity_diversity", "timing_jitter", "drum_coherence"]
        },
        "maestro": {
            "path": "data/external/maestro",
            "subset": 50,
            "metrics": ["harmonic_consistency", "phrase_structure"]
        },
        "lmd": {
            "path": "data/external/lmd_matched",
            "subset": 200,
            "metrics": ["genre_accuracy", "structure_validity"]
        }
    }
    
    def evaluate_dataset(self, dataset_name, model_output_dir):
        # 1. Load dataset subset
        # 2. Generate with Stage3
        # 3. Compute dataset-specific metrics
        # 4. Compare with internal KPI correlation
        pass

# .github/workflows/external_eval.yml (新規)
name: External Benchmark Evaluation
on:
  schedule:
    - cron: '0 0 * * 0'  # 週次実行
  workflow_dispatch:

jobs:
  external_eval:
    steps:
      - name: Download Groove MIDI Dataset
        run: wget https://storage.googleapis.com/magentadata/...
      
      - name: Run external evaluation
        run: python scripts/eval_external_benchmarks.py --all
      
      - name: Generate correlation report
        run: |
          python scripts/correlate_internal_external.py \
            --internal eval/stage3_report.json \
            --external eval/external_benchmarks.json \
            --output eval/correlation_report.md

# scripts/download_benchmarks.sh (新規)
#!/bin/bash
# Groove, MAESTRO, LMD-matched のサブセットダウンロード
# ライセンス確認済みデータのみ
```

**評価指標の相関分析**:
```yaml
internal_metrics:
  - score (Stage2)
  - pass_rate
  - bar_violation_rate
  - text_audio_cos

external_metrics:
  groove:
    - velocity_diversity (std of velocities)
    - timing_humanness (onset jitter)
    - drum_coherence (kick-snare pattern consistency)
  maestro:
    - harmonic_consistency (chord transition probability)
    - phrase_structure (repetition detection)
  lmd:
    - genre_classification_accuracy (vs ground truth)
    - structure_validity (intro-verse-chorus detection)

correlation_analysis:
  - Pearson相関: internal vs external
  - 外れ値検出: 内部Good/外部Badのサンプル抽出
```

**成功基準**:
- 週次CI実行成功率: 100%
- 内部KPIとGroove指標の相関: r > 0.6
- 外部評価レポート自動生成: PR/週次で可視化

**デリバラブル**:
- `scripts/eval_external_benchmarks.py` (300行)
- `scripts/correlate_internal_external.py` (150行)
- `scripts/download_benchmarks.sh` (50行)
- `.github/workflows/external_eval.yml`
- `docs/external_benchmark_guide.md`

---

### Day 9-10: 線形注意適用 (Performer) (工数: 中)

**目的**: max_length拡張 + バッチサイズ増大

**実装内容**:
```python
# ml/attention_linear.py (新規)
from performer_pytorch import SelfAttention as PerformerAttention

class LinearAttentionWrapper:
    """Performer-based linear attention for Stage3"""
    
    def __init__(self, dim, heads, causal=True):
        self.performer = PerformerAttention(
            dim=dim,
            heads=heads,
            causal=causal,
            nb_features=256,  # Random features
        )
    
    def forward(self, q, k, v, mask=None):
        return self.performer(q, k, v)

# ml/stage3_generator.py (拡張)
def create_model_with_linear_attention(config):
    model = GPT2LMHeadModel(config)
    
    # Replace self-attention layers
    if args.use_linear_attention:
        for layer in model.transformer.h:
            layer.attn = LinearAttentionWrapper(
                dim=config.n_embd,
                heads=config.n_head,
                causal=True
            )
    
    return model

# 段階適用戦略
# Phase 1 (Day 9): 推論のみ適用 → 速度・メモリ測定
# Phase 2 (Day 10): 学習適用 → LoRA併用で安定化
```

**ベンチマーク計画**:
```yaml
baseline (GPT-2 self-attention):
  max_length: 2048
  batch_size: 2
  memory: 4GB
  inference_time: 1.5s/sample

target (Performer):
  max_length: 4096  # 2x
  batch_size: 8     # 4x
  memory: <6GB      # 1.5x以下
  inference_time: <2.0s/sample  # 1.3x以下
```

**安定化対策**:
- LoRA併用: Performer導入時もLoRAでファインチューニング
- 段階的学習: まず短シーケンス(1024)で収束 → 徐々に伸ばす
- Gradient clipping: norm=1.0で学習安定化

**成功基準**:
- max_length: 2048 → 4096 (2倍)
- batch_size: 2 → 8 (4倍)
- メモリ増加: <50%
- 生成品質: 既存メトリクスで±5%以内

**デリバラブル**:
- `ml/attention_linear.py` (200行)
- `scripts/benchmark_attention.py` (100行)
- `tests/test_linear_attention.py` (8テスト)
- `docs/linear_attention_migration.md`
- Performance comparison report (速度/メモリ/品質)

---

## CI/テスト戦略拡張

### 新規CI Job追加

```yaml
# .github/workflows/quality_regression.yml (新規)
name: Quality Regression Tests

on:
  pull_request:
    paths:
      - 'ml/**'
      - 'scripts/humanize_*.py'
      - 'ml/tokenizer_*.py'

jobs:
  humanizer_regression:
    runs-on: ubuntu-latest
    steps:
      - name: Test Velocity distribution (K-S test)
        run: pytest tests/test_humanizer.py::test_velocity_ks
      
      - name: Test Timing jitter
        run: pytest tests/test_humanizer.py::test_timing_jitter
      
      - name: Verify structure preservation
        run: pytest tests/test_humanizer.py::test_structure_preserved
  
  tokenizer_compatibility:
    runs-on: ubuntu-latest
    steps:
      - name: Test backward compatibility
        run: pytest tests/test_tokenizer_remi.py::test_v1_0_compat
      
      - name: Test REMI encoding/decoding
        run: pytest tests/test_tokenizer_remi.py::test_remi_roundtrip
      
      - name: Verify migration script
        run: python scripts/migrate_tokenizer.py --dry-run --verify
```

### プロパティベーステスト (Hypothesis)

```python
# tests/test_caption_attrs_property.py (新規)
from hypothesis import given, strategies as st

@given(
    caption=st.text(min_size=1, max_size=200),
    noise=st.floats(min_value=0, max_value=0.3)
)
def test_caption_robustness_to_noise(caption, noise):
    """Caption normalization should be robust to noise"""
    normalizer = AttributeNormalizer()
    
    # Add character-level noise
    noisy_caption = add_char_noise(caption, noise_rate=noise)
    
    attrs_clean = normalizer.normalize(caption)
    attrs_noisy = normalizer.normalize(noisy_caption)
    
    # At least 3/5 attributes should match
    assert attribute_similarity(attrs_clean, attrs_noisy) >= 0.6

@given(caption=st.text(min_size=10, max_size=100))
def test_caption_word_order_invariance(caption):
    """Attribute extraction should be mostly order-invariant"""
    normalizer = AttributeNormalizer()
    
    words = caption.split()
    shuffled = ' '.join(random.sample(words, len(words)))
    
    attrs_orig = normalizer.normalize(caption)
    attrs_shuffled = normalizer.normalize(shuffled)
    
    # Genre/tempo should match (order-independent)
    assert attrs_orig['genre'] == attrs_shuffled['genre']
```

---

## ドキュメント体制強化

### 1. アーティファクト版管理

```python
# ml/model_metadata.py (新規)
class ModelMetadata:
    """Embed artifact hashes for reproducibility"""
    
    def __init__(self, model_dir):
        self.metadata = {
            "stage3_version": "v1.1",
            "tokenizer_version": "remi_v1",
            "attribute_vocab_hash": self._hash_file("configs/attribute_vocab.yaml"),
            "caption_normalizer_hash": self._hash_file("scripts/caption_to_attrs.py"),
            "training_seed": 42,
            "dependency_snapshot": self._capture_dependencies(),
        }
    
    def save(self, path):
        with open(path / "model_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def verify_reproducibility(self, other_metadata):
        """Check if two models are reproducible"""
        critical_fields = ["tokenizer_version", "attribute_vocab_hash", "training_seed"]
        return all(self.metadata[f] == other_metadata[f] for f in critical_fields)
```

### 2. リリースノート自動生成

```python
# scripts/generate_release_notes.py (新規)
def generate_release_notes(version, commits_since_last):
    """Generate stage3_release_notes.md"""
    
    template = f"""
# Stage3 {version} Release Notes

**Release Date**: {datetime.now().strftime("%Y-%m-%d")}
**Build ID**: {get_git_commit_hash()}

## Reproducibility Recipe

### Training Environment
- Python: {sys.version}
- PyTorch: {torch.__version__}
- Transformers: {transformers.__version__}
- CUDA: {torch.version.cuda}

### Data Snapshot
- Training samples: {count_training_samples()}
- Tokenizer version: {get_tokenizer_version()}
- Attribute vocab hash: {hash_file("configs/attribute_vocab.yaml")}

### Training Configuration
```yaml
{load_training_config()}
```

### Model Artifacts
- Model checkpoint: `output/stage3_model/checkpoint-{epoch}`
- Tokenizer: `output/stage3_model/tokenizer_stage3.json`
- Metadata: `output/stage3_model/model_metadata.json`

### Verification Steps
```bash
# Verify artifact hashes
python scripts/verify_model_metadata.py output/stage3_model

# Reproduce training
python ml/stage3_generator.py --config configs/stage3_train_v{version}.yaml
```

## Changes Since {previous_version}
{format_commits(commits_since_last)}

## Performance Metrics
{load_benchmark_results()}
"""
    
    return template
```

### 3. 性能回帰の可視化

```python
# scripts/generate_kpi_sparkline.py (新規)
def generate_kpi_trend(last_n_prs=10):
    """Generate KPI sparkline for last N PRs"""
    
    prs = fetch_last_n_prs(last_n_prs)
    metrics = []
    
    for pr in prs:
        report = load_eval_report(pr.number)
        metrics.append({
            "pr": pr.number,
            "pass_rate": report["overall"]["pass_rate"],
            "p50_score": report["overall"]["p50"],
            "violations": report["overall"]["bar_beat_violation_rate"],
        })
    
    # Generate ASCII sparkline
    sparkline = {
        "pass_rate": "▁▂▃▄▅▆▇█" * len(metrics),  # Visual trend
        "p50_score": create_sparkline([m["p50_score"] for m in metrics]),
        "violations": create_sparkline([m["violations"] for m in metrics], inverted=True),
    }
    
    # Add to PR comment
    comment = f"""
## 📊 KPI Trend (Last {last_n_prs} PRs)

| Metric | Trend | Current | Baseline |
|--------|-------|---------|----------|
| Pass Rate | {sparkline["pass_rate"]} | {metrics[-1]["pass_rate"]:.1%} | {metrics[0]["pass_rate"]:.1%} |
| P50 Score | {sparkline["p50_score"]} | {metrics[-1]["p50_score"]:.1f} | {metrics[0]["p50_score"]:.1f} |
| Violations | {sparkline["violations"]} | {metrics[-1]["violations"]:.1%} | {metrics[0]["violations"]:.1%} |
"""
    
    return comment
```

---

## スケジュール (10日間)

| Day | タスク | 担当 | 成果物 |
|-----|--------|------|--------|
| 1 | GrooVAE統合 - 設計・実装 | - | humanize_with_groovae.py |
| 2 | GrooVAE統合 - テスト・検証 | - | test_humanizer.py (10tests) |
| 3 | GrooVAE統合 - CI統合・ドキュメント | - | humanizer_integration.md |
| 4 | REMI導入 - DURATION追加 | - | tokenizer_remi.py (phase1) |
| 5 | REMI導入 - CHORD追加 | - | tokenizer_remi.py (phase2) |
| 6 | REMI導入 - ROLE追加・移行スクリプト | - | migrate_tokenizer.py |
| 7 | 外部ベンチ - データ準備・評価実装 | - | eval_external_benchmarks.py |
| 8 | 外部ベンチ - CI統合・相関分析 | - | external_eval.yml |
| 9 | Performer導入 - 推論適用・ベンチマーク | - | attention_linear.py |
| 10 | Performer導入 - 学習適用・検証 | - | Performance report |

---

## リスク管理

### 高リスク項目

1. **REMI移行の後方互換性**
   - リスク: 既存データが読めなくなる
   - 対策: migrate_tokenizer.py で100%変換保証 + v1.0 fallback機能

2. **Performer学習の不安定化**
   - リスク: 収束しない・品質低下
   - 対策: LoRA併用 + 段階的length増加 + gradient clipping

3. **外部ベンチのライセンス問題**
   - リスク: 再配布・商用利用制限
   - 対策: ライセンス確認済みデータのみ + READMEに明記

### 中リスク項目

1. **GrooVAEの過剰補正**
   - リスク: 元の構造が壊れる
   - 対策: structure_preservationテストで監視

2. **外部ベンチの相関が低い**
   - リスク: 内部KPIの妥当性に疑問
   - 対策: まず相関分析で確認 → 指標再設計は別スプリント

---

## 成功基準 (Definition of Done)

### 必須条件
- ✅ 全25+40=65テスト合格 (既存25 + 新規40)
- ✅ CI全ジョブグリーン (eval_gate + quality_regression + external_eval)
- ✅ LamdaスコアのVelocity軸: +5pt以上
- ✅ 拍子違反率: <2%
- ✅ max_length: 4096対応 (Performer)
- ✅ 外部ベンチ週次CI稼働

### 推奨条件
- 🟡 内部-外部KPI相関: r > 0.6
- 🟡 Performer学習速度: 1.5x以上
- 🟡 ドキュメント更新率: 100% (全新機能)

---

## 次々期 (Stage3 v1.2) への布石

v1.1完成後、以下を検討:

1. **Music Transformer統合** (相対位置注意)
2. **Constraint Repair** (OR-Tools CP-SAT)
3. **Symbolic Diffusion** (研究分岐)
4. **Stage4編曲器** (マルチトラック)

---

## まとめ

**このスプリントで達成すること**:
- ✅ 聴感の改善 (GrooVAE Humanizer)
- ✅ 構造整合性の向上 (REMI tokenizer)
- ✅ スケーラビリティ (Performer)
- ✅ 客観評価の確立 (外部ベンチ)

**v1.0の強みを保ちつつ、品質で"わかる改善"を実現!** 🚀
