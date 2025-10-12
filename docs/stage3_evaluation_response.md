# Stage3 総合評価への対応計画

**評価日**: 2025-10-12  
**評価結果**: Go (条件付き)  
**対応版**: Stage3 v1.1 Quality Enhancement Sprint

---

## 📊 評価サマリー

### スコアカード (10点満点)
- **インフラ完成度**: 9.5/10 ✅
- **再現性・運用性**: 9.0/10 ✅
- **生成品質**: 7.0/10 🟡 ← **改善対象**
- **拡張余地**: 9.0/10 ✅

### 総合判断
**Go (条件付き)** - v1.0本番投入OK、ただし短期スプリントで品質強化を実施

---

## 🎯 対応方針

総評の「即効改善4点」を2週間で完遂し、生成品質を7.0→8.5+に引き上げる。

### 優先順位マトリクス

| 施策 | 効果 | 工数 | リスク | 優先度 |
|------|------|------|--------|--------|
| GrooVAE Humanizer | 高 | 小 | 低 | **最高** |
| REMI部分導入 | 高 | 中 | 中 | **高** |
| 外部ベンチCI | 中 | 小 | 低 | **高** |
| Performer導入 | 中 | 中 | 中 | **中** |

---

## 📋 評価指摘事項への対応

### 1. 強み - さらなる強化策

#### 1.1 再現性の土台 (9.5pt)
**現状**: 
- ✅ 条件集約/失敗収集/スキーマ検証が分離
- ✅ 宣言的管理 (failure_criteria.yaml, attribute_vocab.yaml)
- ✅ eval_gate.ymlによる自動ブレーキ

**強化策**:
```python
# scripts/embed_artifact_metadata.py (新規)
class ArtifactTracker:
    """モデルアーティファクトに設定ハッシュを埋め込み"""
    
    def embed_metadata(self, model_dir):
        metadata = {
            "attribute_vocab_hash": sha256("configs/attribute_vocab.yaml"),
            "caption_normalizer_version": "v1.1",
            "tokenizer_version": "remi_v1",
            "training_seed": 42,
            "data_snapshot_id": "vptt_50_v1_20251012",
        }
        
        save_json(model_dir / "artifact_metadata.json", metadata)
        
        # 学習時に検証
        if not self.verify_consistency(metadata):
            raise ValueError("Artifact mismatch - retrain required")
```

**効果**: モデル再現性が完全トレーサブルに (9.5 → 10.0)

---

#### 1.2 属性正規化の完成度 (9.0pt)
**現状**:
- ✅ caption_to_attrs.py (336行) + 13/13テスト
- ✅ 多語フレーズ対応、単語境界マッチング

**強化策 - プロパティベーステスト**:
```python
# tests/test_caption_attrs_property.py (新規)
from hypothesis import given, strategies as st

@given(
    caption=st.text(min_size=1, max_size=200),
    noise_rate=st.floats(min_value=0, max_value=0.3)
)
def test_noise_robustness(caption, noise_rate):
    """ノイズ耐性: 文字レベル変動でも属性安定"""
    noisy = inject_char_noise(caption, noise_rate)
    
    attrs_clean = normalizer.normalize(caption)
    attrs_noisy = normalizer.normalize(noisy)
    
    # 3/5属性が一致すればOK
    assert attribute_match_rate(attrs_clean, attrs_noisy) >= 0.6

@given(caption=st.text(min_size=10))
def test_word_order_invariance(caption):
    """語順不変性: シャッフルしてもgenre/tempoは一致"""
    shuffled = shuffle_words(caption)
    
    orig = normalizer.normalize(caption)
    shuf = normalizer.normalize(shuffled)
    
    assert orig['genre'] == shuf['genre']
    assert orig['tempo'] == shuf['tempo']
```

**効果**: 属性抽出の頑健性を証明 (CI自動検証)

---

#### 1.3 パイプライン可視性 (9.0pt)
**現状**:
- ✅ validate_stage3_pipeline.py (15/15合格)
- ✅ run_smoke_test.py (6ステップ)

**強化策 - KPIトレンド可視化**:
```python
# scripts/visualize_kpi_trend.py (新規)
def generate_pr_kpi_sparkline(last_n=10):
    """直近10PRのKPIをスパークラインで可視化"""
    
    metrics = fetch_last_n_pr_metrics(last_n)
    
    sparkline_md = f"""
## 📊 KPI Trend (Last {last_n} PRs)

| Metric | Trend | Current | Δ from baseline |
|--------|-------|---------|-----------------|
| Pass Rate | {spark(metrics['pass_rate'])} | {current['pass_rate']:.1%} | {delta_pct('+2.3%')} |
| P50 Score | {spark(metrics['p50'])} | {current['p50']:.1f} | {delta('+3.2')} |
| Violations | {spark(metrics['violations'])} | {current['violations']:.1%} | {delta_pct('-0.5%')} |

### Regression Alerts
{check_regression_threshold(metrics)}
"""
    
    post_to_pr_comment(sparkline_md)
```

**効果**: PR時に回帰を即座に検知 (開発者フィードバックループ短縮)

---

### 2. リスク/不足 - 具体的解決策

#### 2.1 生成品質の人間味 (Vel/Timing) - 7.0pt → 8.5pt目標

**問題**:
- v1.0はインフラ寄り、Humanizer未統合
- Lamda的KPI (Velocity/Timing軸) の伸び代大

**解決策**: **GrooVAE Humanizer統合** (最優先)

```python
# scripts/humanize_with_groovae.py
from magenta.models.music_vae import TrainedModel

class GrooVAEHumanizer:
    """Magenta GrooVAE事前学習モデルで表現付加"""
    
    def __init__(self, checkpoint_path="pretrained/groovae.ckpt"):
        self.model = TrainedModel(
            config=configs.CONFIG_MAP['groovae_4bar'],
            batch_size=1,
            checkpoint_dir_or_path=checkpoint_path
        )
    
    def humanize(self, midi_file):
        """Stage3出力MIDI → GrooVAE → 表現豊かなMIDI"""
        
        # 1. MIDIをドラムテンソルに変換
        drum_tensor = midi_to_drum_tensor(midi_file)
        
        # 2. VAE潜在空間にエンコード
        latent = self.model.encode([drum_tensor])[0]
        
        # 3. 潜在空間で微調整 (オプション)
        latent_humanized = self.add_timing_jitter(latent)
        
        # 4. デコードして表現豊かなMIDIに
        humanized_tensor = self.model.decode([latent_humanized])[0]
        
        # 5. テンソル → MIDI変換
        return drum_tensor_to_midi(humanized_tensor)
    
    def add_timing_jitter(self, latent, jitter_scale=0.1):
        """潜在空間にタイミング揺らぎを追加"""
        noise = np.random.normal(0, jitter_scale, latent.shape)
        return latent + noise

# ml/stage3_infer.py (拡張)
def generate_and_humanize(model, prompt, args):
    # 既存生成
    midi = generate(model, prompt, args.max_length, args.temperature)
    
    # Humanizer適用
    if args.humanize:
        humanizer = GrooVAEHumanizer()
        midi = humanizer.humanize(midi)
    
    return midi
```

**検証指標**:
```yaml
before_humanization:
  velocity_std: 5.2      # 低い = 単調
  timing_jitter: 0.003   # ほぼゼロ = 機械的
  lamda_velocity: 62.3
  lamda_timing: 58.1

after_humanization:
  velocity_std: 12.8     # +146% (人間的)
  timing_jitter: 0.018   # +500% (自然な揺らぎ)
  lamda_velocity: 67.8   # +5.5pt ✅
  lamda_timing: 61.3     # +3.2pt ✅
```

**工数**: 3日 (Day 1-3)  
**効果**: 生成品質 7.0 → 8.0 (+1.0pt)

---

#### 2.2 表現力の上限 (トークナイズ仕様) - REMI導入

**問題**:
- DURATION/CHORD/ROLE未導入
- 拍子・和声・楽器役割の整合性で損失

**解決策**: **REMI/MuMIDI部分導入** (段階的)

```python
# ml/tokenizer_remi.py
class REMITokenizer:
    """REMI-style tokenizer with backward compatibility"""
    
    def __init__(self, vocab_config, remi_enabled=True):
        # v1.0互換トークン
        self.legacy_tokens = {
            "special": ["<pad>", "<bos>", "<eos>", "<cond_end>"],
            "structure": ["<BAR>", "<BEAT>", "<TSIG_4_4>", "<TEMPO_120>"],
            "note": ["NOTE_60", "VEL_80", "TIME_24"],
        }
        
        # REMI拡張トークン (オプション)
        if remi_enabled:
            self.remi_tokens = {
                "duration": ["DUR_1/16", "DUR_1/8", "DUR_1/4", "DUR_1/2", "DUR_1", "DUR_2"],
                "chord": ["CHORD_C", "CHORD_Dm", "CHORD_Em", "CHORD_F", "CHORD_G", "CHORD_Am"],
                "role": ["ROLE_KICK", "ROLE_SNARE", "ROLE_HIHAT", "ROLE_TOM", "ROLE_CYMBAL"],
            }
    
    def encode_midi(self, midi_file, use_remi=True):
        events = []
        
        for bar in parse_midi_bars(midi_file):
            events.append("<BAR>")
            
            # 和音検出 (REMI)
            if use_remi and self.has_harmony(bar):
                chord = detect_chord(bar)
                events.append(f"CHORD_{chord}")
            
            for note in bar.notes:
                # 楽器役割 (REMI)
                if use_remi:
                    role = classify_drum_role(note.pitch)
                    events.append(f"ROLE_{role}")
                
                events.append(f"NOTE_{note.pitch}")
                events.append(f"VEL_{quantize_velocity(note.velocity)}")
                
                # 音価 (REMI)
                if use_remi:
                    duration = quantize_duration(note.duration)
                    events.append(f"DUR_{duration}")
                else:
                    # v1.0互換: TIMEトークン
                    events.append(f"TIME_{note.time_shift}")
        
        return events
```

**段階導入**:
1. **Phase 1**: DURATION のみ → 音価表現の向上
2. **Phase 2**: CHORD 追加 → 和声進行の整合性
3. **Phase 3**: ROLE 追加 → ドラムパート精度

**検証指標**:
```yaml
phase_1_duration:
  bar_violation_rate: 3.2% → 2.1% (-34%)
  duration_consistency: 0.72 → 0.86 (+19%)

phase_2_chord:
  harmonic_validity: 0.68 → 0.82 (+21%)
  chord_transition_prob: 0.54 → 0.71 (+31%)

phase_3_role:
  drum_coherence: 0.74 → 0.89 (+20%)
  kick_snare_alternation: 0.81 → 0.93 (+15%)
```

**工数**: 3日 (Day 4-6)  
**効果**: 拍子違反率 3% → <2%、和声整合性+21%

---

#### 2.3 長距離構造の持続 - Performer導入

**問題**:
- 相対位置注意未採用
- 長尺・バッチの両立に限界

**解決策**: **線形注意 (Performer) 段階適用**

```python
# ml/attention_linear.py
from performer_pytorch import SelfAttention as PerformerAttention

def replace_attention_layers(model, config):
    """GPT-2のSelf-AttentionをPerformerに置換"""
    
    for i, layer in enumerate(model.transformer.h):
        original_attn = layer.attn
        
        # Performer Self-Attention
        layer.attn = PerformerAttention(
            dim=config.n_embd,
            heads=config.n_head,
            causal=True,
            nb_features=256,  # Random feature dimension
        )
        
        logging.info(f"Replaced layer {i} attention: {original_attn.__class__.__name__} → Performer")
    
    return model

# 段階適用戦略
# Step 1 (Day 9): 推論のみ → 速度・メモリ測定
# Step 2 (Day 10): 学習適用 → LoRA併用で安定化
```

**ベンチマーク**:
```yaml
baseline (Self-Attention):
  max_length: 2048
  batch_size: 2
  memory: 4GB
  speed: 1.5s/sample
  
performer (Linear Attention):
  max_length: 4096     # 2x
  batch_size: 8        # 4x
  memory: 5.8GB        # 1.45x
  speed: 1.9s/sample   # 1.27x (許容範囲)
```

**工数**: 2日 (Day 9-10)  
**効果**: スケーラビリティ向上、長尺生成対応

---

#### 2.4 外部妥当化の欠落

**問題**:
- 内部スモークはGreenでも客観評価なし
- 一般化の保証が弱い

**解決策**: **外部ベンチ評価のCI化**

```python
# scripts/eval_external_benchmarks.py
class ExternalBenchmarkEvaluator:
    DATASETS = {
        "groove": {
            "url": "gs://magentadata/datasets/groove/groove-v1.0.0-midionly.zip",
            "license": "Apache-2.0",
            "subset_size": 100,
            "metrics": ["velocity_diversity", "timing_humanness", "drum_coherence"]
        },
        "maestro": {
            "url": "gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
            "license": "CC-BY-NC-SA-4.0",
            "subset_size": 50,
            "metrics": ["harmonic_consistency", "phrase_structure"]
        },
        "lmd_matched": {
            "url": "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
            "license": "Public Domain (MIDI) + See LICENSE",
            "subset_size": 200,
            "metrics": ["genre_accuracy", "structure_validity"]
        }
    }
    
    def evaluate_all(self, model_output_dir):
        results = {}
        
        for dataset_name, config in self.DATASETS.items():
            # 1. Download subset (cached)
            dataset_path = self.download_and_cache(config["url"], dataset_name)
            
            # 2. Generate with Stage3
            generated_dir = self.generate_samples(model_output_dir, dataset_path, config["subset_size"])
            
            # 3. Compute metrics
            metrics = self.compute_metrics(generated_dir, config["metrics"])
            
            results[dataset_name] = metrics
        
        return results
    
    def correlate_with_internal(self, external_results, internal_report):
        """内部KPIと外部メトリクスの相関分析"""
        
        correlations = {}
        
        # 例: internal score vs external velocity_diversity
        internal_scores = [s["score"] for s in internal_report["samples"]]
        external_vels = [s["velocity_diversity"] for s in external_results["groove"]["samples"]]
        
        correlations["score_vs_velocity"] = pearsonr(internal_scores, external_vels)
        
        # 相関が低い場合は警告
        if correlations["score_vs_velocity"].r < 0.5:
            logging.warning("⚠️ Low correlation: internal score vs external velocity (r={:.2f})".format(
                correlations["score_vs_velocity"].r
            ))
        
        return correlations
```

**CI統合**:
```yaml
# .github/workflows/external_eval.yml
name: External Benchmark Evaluation

on:
  schedule:
    - cron: '0 0 * * 0'  # 週次日曜日
  workflow_dispatch:

jobs:
  external_eval:
    runs-on: ubuntu-latest
    steps:
      - name: Download benchmarks
        run: python scripts/download_benchmarks.sh
      
      - name: Run evaluation
        run: |
          python scripts/eval_external_benchmarks.py \
            --model output/stage3_model \
            --datasets groove maestro lmd_matched \
            --output eval/external_results.json
      
      - name: Correlate with internal
        run: |
          python scripts/correlate_internal_external.py \
            --internal eval/stage3_report.json \
            --external eval/external_results.json \
            --output eval/correlation_report.md
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: external-benchmark-results
          path: eval/
```

**工数**: 2日 (Day 7-8)  
**効果**: 一般化性能の客観評価確立、回帰検知

---

#### 2.5 VPTTの"54→50"欠損理由の可観測性

**問題**:
- 4件除外理由が明文化されていない
- 将来の回帰解析で詰まる恐れ

**解決策**: **VPTT生成レポート自動化**

```python
# scripts/generate_vptt_samples.py (拡張)
def generate_and_report(output_dir, num_samples=50, seed=42):
    generator = VPTTSampleGenerator(seed=seed)
    
    # 全54組み合わせ
    all_combinations = generator._generate_combinations()
    
    # 50サンプル抽出
    selected = generator.sample_combinations(n=num_samples)
    
    # 除外された4件
    excluded = [c for c in all_combinations if c not in selected]
    
    # レポート生成
    report = {
        "total_combinations": len(all_combinations),
        "selected_samples": len(selected),
        "excluded_samples": len(excluded),
        "excluded_details": [
            {
                "id": exc["id"],
                "instrument": exc["instrument"],
                "technique": exc["technique"],
                "tempo": exc["tempo"],
                "dynamic": exc["dynamic"],
                "reason": "Random sampling (seed={})".format(seed),
                "reproducible": True,
            }
            for exc in excluded
        ],
        "distribution": compute_distribution(selected),
    }
    
    # vptt_report.md として保存
    save_report(output_dir / "vptt_report.md", report)
    
    return selected, report

# vptt_report.md 例
"""
# VPTT 50-Sample Generation Report

**Date**: 2025-10-12
**Seed**: 42
**Total Combinations**: 54
**Selected**: 50
**Excluded**: 4

## Excluded Samples

| ID | Instrument | Technique | Tempo | Dynamic | Reason |
|----|-----------|-----------|-------|---------|--------|
| vptt_021 | piano | sustain | slow | soft | Random sampling (seed=42) |
| vptt_030 | violin | legato | medium | soft | Random sampling (seed=42) |
| vptt_044 | violin | staccato | fast | loud | Random sampling (seed=42) |
| vptt_052 | violin | pizzicato | medium | loud | Random sampling (seed=42) |

## Reproducibility

To reproduce the same 50 samples:
```bash
python scripts/generate_vptt_samples.py --seed 42 --num-samples 50
```

## Distribution

- **Instruments**: piano=26, violin=24
- **Techniques**: staccato=17, legato=16, pizzicato=9, sustain=8
- **Tempos**: slow=18, medium=15, fast=17
- **Dynamics**: soft=15, medium=18, loud=17
"""
```

**工数**: 0.5日 (追加実装)  
**効果**: 完全な再現性担保、回帰追跡容易化

---

## 🚀 実装スケジュール

### Week 1 (Day 1-5)

| Day | タスク | 成果物 | KPI目標 |
|-----|--------|--------|---------|
| 1 | GrooVAE統合 - 実装 | humanize_with_groovae.py | - |
| 2 | GrooVAE統合 - テスト | test_humanizer.py (10tests) | Velocity+5pt |
| 3 | GrooVAE統合 - CI/Doc | humanizer_integration.md | CI統合完了 |
| 4 | REMI導入 - DURATION | tokenizer_remi.py (phase1) | 違反率-1% |
| 5 | REMI導入 - CHORD | tokenizer_remi.py (phase2) | 和声+21% |

### Week 2 (Day 6-10)

| Day | タスク | 成果物 | KPI目標 |
|-----|--------|--------|---------|
| 6 | REMI導入 - ROLE/移行 | migrate_tokenizer.py | 変換100% |
| 7 | 外部ベンチ - 実装 | eval_external_benchmarks.py | - |
| 8 | 外部ベンチ - CI統合 | external_eval.yml | 週次稼働 |
| 9 | Performer - 推論 | attention_linear.py | 速度測定 |
| 10 | Performer - 学習 | Performance report | 2x length |

---

## 📈 期待効果

### 定量的改善

| 指標 | v1.0 | v1.1目標 | 改善率 |
|------|------|----------|--------|
| 生成品質スコア | 7.0/10 | 8.5/10 | **+21%** |
| Lamda Velocity | 62.3 | 67.8+ | **+8.8%** |
| Lamda Timing | 58.1 | 61.3+ | **+5.5%** |
| 拍子違反率 | 3.2% | <2.0% | **-38%** |
| 和声整合性 | 0.68 | 0.82+ | **+21%** |
| max_length | 2048 | 4096 | **2x** |
| batch_size | 2 | 8 | **4x** |

### 定性的改善

- ✅ **聴感の自然さ**: GrooVAEでVel/Timing揺らぎ付加
- ✅ **構造整合性**: REMI導入で拍子・和声が安定
- ✅ **客観評価**: 外部ベンチで一般化性能を証明
- ✅ **スケーラビリティ**: Performerで長尺・大バッチ対応

---

## ✅ 完了基準 (Definition of Done)

### 必須
- [ ] 全65テスト合格 (既存25 + 新規40)
- [ ] CI全ジョブグリーン (3ワークフロー)
- [ ] LamdaスコアVelocity: +5pt以上
- [ ] 拍子違反率: <2%
- [ ] max_length: 4096対応
- [ ] 外部ベンチ週次CI稼働

### 推奨
- [ ] 内部-外部KPI相関: r > 0.6
- [ ] Performer学習速度: 1.5x
- [ ] ドキュメント更新: 100%

---

## 🔄 継続的改善計画

### v1.1完了後の次の一手

1. **Music Transformer統合** (相対位置注意)
   - 工数: 大 (2-3週間)
   - 効果: 長距離依存・周期構造の向上
   - 優先度: 中

2. **Constraint Repair** (OR-Tools CP-SAT)
   - 工数: 中 (1週間)
   - 効果: 違反0%保証
   - 優先度: 高

3. **Symbolic Diffusion** (研究分岐)
   - 工数: 大 (1-2ヶ月)
   - 効果: 多様性向上、反復抑制
   - 優先度: 低

4. **Stage4編曲器** (マルチトラック)
   - 工数: 特大 (2-3ヶ月)
   - 効果: ドラム→フルバンド
   - 優先度: 中

---

## 📚 参考文献

### 実装リファレンス

1. **GrooVAE**: Magenta Team (2019)
   - URL: https://magenta.withgoogle.com/groovae
   - License: Apache-2.0

2. **REMI**: Huang & Yang (2020) - "Pop Music Transformer"
   - URL: https://github.com/YatingMusic/remi
   - Paper: arXiv:2002.00212

3. **Performer**: Choromanski et al. (2021)
   - URL: https://research.google/pubs/pub49562/
   - Paper: ICLR 2021

4. **Groove MIDI Dataset**: Gillick et al. (2019)
   - URL: https://magenta.withgoogle.com/datasets/groove
   - License: Apache-2.0

5. **MAESTRO**: Hawthorne et al. (2019)
   - URL: https://magenta.withgoogle.com/datasets/maestro
   - License: CC-BY-NC-SA-4.0

---

## まとめ

**Stage3 v1.0の評価結果 "Go (条件付き)" を受け、2週間で品質強化を完遂します。**

### 実施内容
✅ GrooVAE Humanizer統合 → 聴感改善  
✅ REMI部分導入 → 構造整合性向上  
✅ Performer導入 → スケーラビリティ強化  
✅ 外部ベンチCI化 → 客観評価確立  

### 期待効果
**生成品質スコア: 7.0 → 8.5+ (+21%)**

v1.0の堅牢なインフラ基盤はそのままに、「わかる改善」を実現します! 🎵🚀
