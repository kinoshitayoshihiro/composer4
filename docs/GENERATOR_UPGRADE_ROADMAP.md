# Generator Upgrade Roadmap (Stage3 統合計画)
**Date**: 2025-10-13  
**Status**: Flash Attention評価完了 → drumgenerator Stage3完成へ移行  
**Base**: composer4 (main@f80df845c)

---

## 🎯 Executive Summary

**Flash Attention評価結果** を踏まえ、以下の優先順位で実装を進めます:

1. ✅ **drumgenerator**: Standard Attention (FP32) で Stage3完成を最優先
2. 🔄 **他楽器**: LAMDA Stage2 クリーニング → 段階的に Stage3 へ移行
3. 📝 **Flash Attention**: N≥2048のユースケース向けにドキュメント化

**理由**:
- drumgenerator の主戦場は N≈512-768 (Flash は -12.6%遅い)
- N=2048 で Flash は 1.49x高速化を確認済み (将来の長シーケンス用)
- 混乱回避: Drum完成 → 他パートはLAMDAから段階的に

---

## 📊 Flash Attention 評価結果 (要約)

### 最終決定

| シーケンス長 | 推奨Attention | 理由 |
|-------------|--------------|------|
| **N < 1024** (drumgenerator) | **Standard (FP32)** ✅ | Flash は -12.6%遅い (N=576) |
| **N ≥ 2048** (将来用) | **Flash (BF16)** ⚡ | 1.49x高速化 |

### 実証データ

**N=576 (drumgenerator 想定)**:
- Standard: 5,828 ms (最速)
- Flash (FP16): 6,671 ms (**0.87x、-12.6%遅い**)

**N=2048 (長シーケンス)**:
- Standard: 37,814 ms
- Flash (BF16): 25,345 ms (**1.49x高速化**)

**結論**: drumgenerator は **Standard Attention のまま着手OK** 🎯

---

## 🏗️ Phase 1: drumgenerator Stage3 完成 (最優先)

### 目標
「生成 → 再評価 → A/Bレポート → CIゲート」を1ストロークで回す

### 実装タスク

#### 1.1 ワンボタン評価スクリプト (⏱️ 2時間)

**作成ファイル**: `scripts/run_stage3_drum_eval.sh`

```bash
#!/bin/bash
# Stage3 Drum: 生成→評価→レポート→判定の自動化

set -e
REPO=/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3
OUT=$REPO/output/drumgen_eval_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

echo "🎵 Stage3 Drum Evaluation Pipeline"
echo "Output: $OUT"

# 1) 生成 (10サンプル、pop_straight、BPM=120)
python -m adapters.run_drum_adapter \
  --n 10 --tempo 120 --length-bars 64 --style pop_straight \
  --density mid --swing 2 --seed 42 --out "$OUT/generated"

# 2) Stage2再評価
python scripts/quick_eval_stage2.py \
  --input-dir "$OUT/generated" --output-dir "$OUT/stage2"

# 3) A/Bレポート生成
python scripts/ab_summarize_v2.py \
  --input "$OUT/stage2" --output "$OUT/stage3_ab_report.md"

# 4) 受け入れ判定
python scripts/check_acceptance.py \
  --report "$OUT/stage3_ab_report.md" \
  --bar-violations 0.0 --hat-grid 0.85 --pass-rate 0.65

echo "✅ Evaluation complete: $OUT/stage3_ab_report.md"
```

**受け入れ基準**:
- Bar violation = 0% (マスト)
- Hat grid ≥ 0.85 (straight系)
- pass_rate ≥ 0.65
- text_audio_cos ≥ 0.60 (音声連携ON時)

#### 1.2 LegacyGeneratorAdapter 実装 (⏱️ 3時間)

**作成ファイル**: `adapters/legacy_generator_adapter.py`

提示された雛形を基に以下を実装:
- `AdapterMeta`: model_commit/tokenizer_hash/remi_version/vocab_sha256
- `GenerationLogger`: JSONL形式でログ記録
- `LegacyGeneratorAdapter`: 旧ジェネレーターのラッパー
  - `_normalize_conditions()`: 条件の標準化 (XMIDI/EMOPIA/VPTT/CLAP/MERT)
  - `generate()`: 生成 + sidecar (.meta.json) + ログ記録

**重要ポイント**:
```python
# Attention設定: drumgeneratorはStandardを明示的に指定
cfg = AttnAutoConfig(threshold=512)  # 既定でStandard
kind = apply_adaptive_attention(
    model,
    device="cuda",
    seq_len=576,  # drumgeneratorの典型的な長さ
    force="standard",  # Flash Attentionを無効化
    cfg=cfg
)
```

#### 1.3 drumgenerator本体の最小改修 (⏱️ 1時間)

**修正ファイル**: `drum_generator.py` (compose関数のみ)

```python
# 引数受け取りを追加 (デフォルトで従来動作)
def compose(self, *, tempo=None, length_bars=None, meter=None, 
            style=None, density=None, swing=None, rng=None):
    if tempo is not None:           self.tempo = tempo
    if length_bars is not None:     self.length_bars = length_bars
    if meter is not None:           self.meter = meter
    if style is not None:           self.style = style
    if density is not None:         self.density = density
    if swing is not None:           self.swing = swing
    if rng is not None:             self.rng = rng
    # 以降の内部処理は従来のまま
    return self._render_part(...)
```

**変更方針**: 関数1か所のみ、下位互換を維持

#### 1.4 Humanizer v1.1 連携 (⏱️ 2時間)

**修正ファイル**: `humanize_midi.py` または新規 `humanizer_v1_1.py`

```python
def humanize_midi_v1_1(midi_path: str, 
                       velocity_std: float = 12.0,
                       timing_jitter: float = 0.018,
                       swing: float = 0.0,
                       ar1: float = 0.6,
                       seed: int = 42) -> str:
    """
    Humanizer v1.1: AR(1) + BPM連動 + 拍LUT + スウィング
    完全再現性のため seed を受け取る
    """
    rng = random.Random(seed)
    # ... 既存の humanize 実装 ...
```

#### 1.5 ガード & 評価自動化 (⏱️ 2時間)

**作成ファイル**: `scripts/check_acceptance.py`

```python
def check_acceptance(report_path: str, 
                     bar_violations: float = 0.0,
                     hat_grid: float = 0.85,
                     pass_rate: float = 0.65,
                     text_audio_cos: float = 0.60) -> bool:
    """Stage3 AB レポートの受け入れ判定"""
    with open(report_path) as f:
        report = json.load(f)
    
    checks = {
        "Bar violation": report["metrics"]["bar_violation"] <= bar_violations,
        "Hat grid": report["metrics"]["hat_grid"] >= hat_grid,
        "Pass rate": report["metrics"]["pass_rate"] >= pass_rate,
    }
    
    if report.get("audio_enabled"):
        checks["Text-Audio cos"] = report["metrics"]["text_audio_cos"] >= text_audio_cos
    
    passed = all(checks.values())
    print(f"{'✅ PASS' if passed else '❌ FAIL'}: Stage3 Acceptance")
    for k, v in checks.items():
        print(f"  {k}: {'✅' if v else '❌'}")
    
    return passed
```

#### 1.6 最小テスト追加 (⏱️ 2時間)

**作成ファイル**:
- `tests/test_generators_smoke.py`: 1プロンプト→生成→MIDI存在→.meta.json存在→seed再現
- `tests/test_adapter_contract.py`: 返り値スキーマ・sidecar必須キー検証

### 📅 Phase 1 タイムライン (1日)

```
AM (午前):
- ✅ Flash Attention評価完了確認
- 🔨 LegacyGeneratorAdapter実装
- 🔨 drum_generator.py の compose() 改修

PM (午後):
- 🔨 run_stage3_drum_eval.sh 作成
- 🧪 3プロンプト×1サンプルでスモークテスト
- 📊 stage3_ab_report.md 生成確認

Evening (夕方):
- 🔨 check_acceptance.py 実装
- 🧪 tests/test_generators_smoke.py 追加
- ✅ CIゲート配線 (GitHub Actions)
```

---

## 🔧 Phase 2: 他楽器のLAMDA Stage2 (並行可能)

### 楽器別優先順位

1. **Piano** (2-3日)
2. **Bass** (2-3日)
3. **Strings** (3-4日)
4. **Guitar** (3-4日)

### 各楽器の標準手順

```bash
# 1. メタ構築
PYTHONPATH=. .venv/bin/python scripts/build_contract_records.py \
  --input-dir input/<instrument>_raw \
  --output-dir output/<instrument>_metadata

# 2. クリーニング
PYTHONPATH=. .venv/bin/python scripts/lamda_stage1_clean.py \
  --metadata-dir output/<instrument>_metadata \
  --input-dir input/<instrument>_raw \
  --output-dir output/<instrument>_cleaned

# 3. Stage2評価
PYTHONPATH=. .venv/bin/python scripts/lamda_stage2_extractor.py \
  --metadata-index output/<instrument>_metadata/<shard>.pickle \
  --metadata-dir output/<instrument>_metadata \
  --input-dir output/<instrument>_cleaned \
  --output-dir outputs/stage2_<instrument>_iter1 \
  --config configs/lamda/<instrument>_stage2.yaml \
  --print-summary
```

### 楽器別YAML差分 (configs/lamda/)

**共通ベース**: `drum_stage2.yaml` をコピー

**調整項目**:
- **Timing**: マイクロタイミング許容 (弦/鍵盤: ±30-40ms)
- **Velocity**: 楽器特性 (ピアノ: 広レンジ、弦: 中域中心)
- **Structure**: 周期候補 2/4/8/16 共通化
- **Articulation**: `technique_map.yaml` に基づく検出

### Stage2 → Stage3 移行条件

✅ **移行OK条件**:
- Stage2 pass_rate ≥ 90%
- velocity_coverage.json の空帯域 < 20%
- articulation.auto.yaml の calibrate 完了

---

## 📁 ディレクトリ構造 (提案)

```
composer2-3/
├── adapters/
│   ├── __init__.py
│   ├── legacy_generator_adapter.py  # 新規: 薄いアダプタ
│   └── run_drum_adapter.py          # 新規: Drum用エントリポイント
├── configs/
│   └── lamda/
│       ├── drum_stage2.yaml         # 既存
│       ├── piano_stage2.yaml        # 新規: Drum YAMLをコピー→調整
│       ├── bass_stage2.yaml         # 新規
│       └── strings_stage2.yaml      # 新規
├── scripts/
│   ├── run_stage3_drum_eval.sh      # 新規: ワンボタン評価
│   ├── check_acceptance.py          # 新規: 受け入れ判定
│   └── benchmark_performer_adaptive.py  # 既存: Flash評価完了
├── tests/
│   ├── test_generators_smoke.py     # 新規: E2Eスモーク
│   ├── test_adapter_contract.py     # 新規: アダプタ契約テスト
│   └── test_music_guards_time_sigs.py  # 既存: 拍子ガード
├── results/
│   ├── FLASH_ATTENTION_N2048_ANALYSIS.md  # 新規: Flash評価完了レポート
│   ├── standard_fp32_n2048.json
│   ├── sdpa_fp16_flash_n2048.json
│   └── sdpa_bf16_flash_n2048.json
└── docs/
    ├── GENERATOR_UPGRADE_ROADMAP.md  # 本ファイル
    └── ATTENTION_DECISION_RECORD.md  # 既存
```

---

## 🔬 Phase 1 詳細: drumgenerator Stage3完成

### Task 1.1: LegacyGeneratorAdapter 実装

**ファイル**: `adapters/legacy_generator_adapter.py`

**実装内容** (提示された雛形ベース):

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json, time, hashlib, random, os

@dataclass(frozen=True)
class AdapterMeta:
    model_commit: str
    tokenizer_hash: str
    remi_version: str = "1.1.0"
    vocab_sha256: str = ""

class GenerationLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "generations.jsonl"
    
    def log(self, row: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

class LegacyGeneratorAdapter:
    """旧ジェネレーターをStage3互換I/Fへ接続"""
    
    def __init__(self, legacy_impl, meta: AdapterMeta, out_dir: str):
        self.legacy_impl = legacy_impl
        self.meta = meta
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = GenerationLogger(self.out_dir / "gen_logs")
    
    def generate(self, n: int, conditions: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
        cond_norm = self._normalize_conditions(conditions)
        rng = random.Random(seed)
        
        # 旧実装の呼び出し
        results = self.legacy_impl.generate(n=n, conditions=cond_norm, seed=seed)
        
        finalized: List[Dict[str, Any]] = []
        for r in results:
            midi_path = Path(r["midi_path"]).resolve()
            if not midi_path.exists():
                raise FileNotFoundError(f"MIDI not found: {midi_path}")
            
            # gen_id生成 (MIDI bytes + seed + time)
            midi_bytes = midi_path.read_bytes()
            h = hashlib.sha1()
            h.update(midi_bytes)
            h.update(str(seed).encode())
            h.update(str(time.time()).encode())
            gen_id = h.hexdigest()[:16]
            
            # Sidecar メタ (versioned)
            sidecar = {
                "gen_id": gen_id,
                "seed": seed,
                "model_commit": self.meta.model_commit,
                "tokenizer_hash": self.meta.tokenizer_hash,
                "remi_version": self.meta.remi_version,
                "vocab_sha256": self.meta.vocab_sha256,
                "conditions": cond_norm,
                "legacy_meta": r.get("meta", {}),
                "runtime": {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
                },
                "artifacts": {"midi_path": str(midi_path)},
            }
            
            # Atomic write to .meta.json
            sidecar_path = midi_path.with_suffix(".meta.json")
            tmp = sidecar_path.with_suffix(".meta.json.tmp")
            tmp.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(sidecar_path)
            
            # Logger (JSONL)
            self.logger.log({
                "gen_id": gen_id,
                "midi_path": str(midi_path),
                "seed": seed,
                "model_commit": self.meta.model_commit,
                "tokenizer_hash": self.meta.tokenizer_hash,
                "conditions": cond_norm,
            })
            
            finalized.append({"midi_path": str(midi_path), "meta": sidecar})
        
        return finalized
    
    def _normalize_conditions(self, c: Dict[str, Any]) -> Dict[str, Any]:
        """条件の標準化 (下位互換を保ちつつキー整形)"""
        def pick(*keys, default=None):
            for k in keys:
                if c.get(k) is not None:
                    return c[k]
            return default
        
        return {
            "tempo": pick("tempo", default=120),
            "time_sig": pick("time_sig", "timesig", default="4/4"),
            "length_bars": pick("length_bars", "bars", default=64),
            "style": pick("style", "pattern", default="pop_straight"),
            "density": pick("density", default="mid"),
            "swing": pick("swing", default=0.0),
            "emotion": pick("emotion"),
            "genre": pick("genre"),
            "valence": pick("valence"),
            "arousal": pick("arousal"),
            "attrs": pick("attrs", default=[]),
            "technique": pick("technique"),
            "audio_clap_bucket": pick("audio_clap_bucket"),
            "audio_mert_bucket": pick("audio_mert_bucket"),
        }
```

### Task 1.2: テスト追加

**ファイル**: `tests/test_generators_smoke.py`

```python
import pytest
from adapters.legacy_generator_adapter import LegacyGeneratorAdapter, AdapterMeta
from pathlib import Path

def test_drum_adapter_smoke():
    """Drum: 生成→MIDI存在→.meta.json存在→seed再現"""
    # Setup
    adapter = LegacyGeneratorAdapter(
        legacy_impl=MockDrumGenerator(),
        meta=AdapterMeta(model_commit="test", tokenizer_hash="test"),
        out_dir="output/test_drum"
    )
    
    # Generate
    results = adapter.generate(n=1, conditions={"tempo": 120}, seed=42)
    
    # Assert
    assert len(results) == 1
    midi_path = Path(results[0]["midi_path"])
    assert midi_path.exists()
    assert midi_path.with_suffix(".meta.json").exists()
    
    # Seed reproducibility
    results2 = adapter.generate(n=1, conditions={"tempo": 120}, seed=42)
    assert results[0]["meta"]["gen_id"] == results2[0]["meta"]["gen_id"]
```

**ファイル**: `tests/test_adapter_contract.py`

```python
def test_sidecar_schema():
    """Sidecar .meta.json の必須キーを検証"""
    required_keys = [
        "gen_id", "seed", "model_commit", "tokenizer_hash",
        "remi_version", "conditions", "runtime", "artifacts"
    ]
    # ... 検証実装 ...
```

### Task 1.3: CI ゲート配線

**ファイル**: `.github/workflows/stage3_drum_ci.yml`

```yaml
name: Stage3 Drum CI

on: [push, pull_request]

jobs:
  drum-stage3-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run Stage3 Drum Evaluation
        run: bash scripts/run_stage3_drum_eval.sh
      
      - name: Check Acceptance
        run: |
          python scripts/check_acceptance.py \
            --report output/drumgen_eval_*/stage3_ab_report.md \
            --bar-violations 0.0 \
            --hat-grid 0.85 \
            --pass-rate 0.65
      
      - name: Upload Artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: stage3-drum-report
          path: output/drumgen_eval_*/stage3_ab_report.md
```

---

## 🎼 Phase 2 詳細: 他楽器のLAMDA Stage2

### Task 2.1: Piano Stage2 設定 (Day 3-4)

**作成ファイル**: `configs/lamda/piano_stage2.yaml`

**差分調整**:
```yaml
# Drum YAML からコピー → 以下を調整

timing:
  max_microtiming_jitter_ms: 35.0  # ピアノは広め (Drumは30.0)
  
velocity:
  targets_file: "configs/lamda/velocity_targets_piano.yaml"  # 専用
  # ピアノ: 広いダイナミックレンジ (ppp=20 → fff=115)
  
structure:
  note_density:
    min: 0.5   # ピアノはコード+メロディで密度高め (Drumは0.3)
    max: 4.0
    
articulation:
  technique_map_file: "configs/lamda/technique_map_piano.yaml"
  # ピアノ特有: sustain, staccato, legato
```

### Task 2.2: Bass/Strings/Guitar

同様の手順で `configs/lamda/<instrument>_stage2.yaml` を作成

**調整ポイント**:
- **Bass**: 低音域、sustain長め、note_density低め
- **Strings**: legato/vibrato、velocity中域中心
- **Guitar**: bend/slide、BPM連動のstrumming

### Stage2 成果物の検証

**期待される出力**:
```
outputs/stage2_<instrument>_iter1/
├── metrics_score.jsonl       # axes_raw含む
├── stage2_summary.json        # pass_rate/分布
├── velocity_coverage.json     # BPM帯×ビンの穴確認
└── audio_embeddings.parquet   # 音声連携ON時
```

**合格基準**:
- pass_rate ≥ 90%
- velocity_coverage 空帯域 < 20%
- Bar/Beat violation = 0%

---

## 🚀 Flash Attention 活用 (将来)

### N≥2048 のユースケース

**適用対象**:
- 長尺楽曲生成 (5分以上、N≥2048)
- マルチトラック同時生成 (combined sequence)
- リアルタイム拡張 (ライブコーディング)

### 実装ガイド

```python
# Flash Attention (BF16) を有効化
from ml.attn_selector import apply_adaptive_attention, AttnAutoConfig
from ml.attention_sdpa import replace_attention_layers_sdpa

# N≥2048 の場合のみ Flash を使用
if sequence_length >= 2048:
    model = model.to(torch.bfloat16)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.float()  # LayerNorm は FP32 で安定
    
    torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_mem_efficient=False,
        enable_math=False
    )
    
    apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=sequence_length,
        replace_sdpa_fn=replace_attention_layers_sdpa,
        force="sdpa",  # Flash強制
    )
else:
    # N < 2048: Standard Attention (既定)
    apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=sequence_length,
        force="standard",  # Standard強制
    )
```

**期待効果**:
- N=2048: 1.49x高速化 (37.8秒 → 25.3秒)
- N=4096: 2-3x高速化 (予想)
- メモリ: +28% (許容範囲)

---

## 📋 実装チェックリスト

### Phase 1: drumgenerator (1日)

- [ ] `adapters/legacy_generator_adapter.py` 実装
- [ ] `adapters/run_drum_adapter.py` 実装 (エントリポイント)
- [ ] `drum_generator.py` の `compose()` に引数受け取り追加
- [ ] `scripts/run_stage3_drum_eval.sh` 作成 (ワンボタン評価)
- [ ] `scripts/check_acceptance.py` 実装 (受け入れ判定)
- [ ] `tests/test_generators_smoke.py` 追加 (E2Eスモーク)
- [ ] `tests/test_adapter_contract.py` 追加 (スキーマ検証)
- [ ] `.github/workflows/stage3_drum_ci.yml` 作成 (CIゲート)
- [ ] スモークテスト実行 (3プロンプト×1サンプル)
- [ ] `stage3_ab_report.md` 生成確認
- [ ] CIゲート動作確認 (pass_rate≥0.65)

### Phase 2: Piano (2-3日)

- [ ] `configs/lamda/piano_stage2.yaml` 作成 (Drumから調整)
- [ ] `configs/lamda/velocity_targets_piano.yaml` 作成
- [ ] `configs/lamda/technique_map_piano.yaml` 作成
- [ ] メタ構築 (`build_contract_records.py`)
- [ ] クリーニング (`lamda_stage1_clean.py`)
- [ ] Stage2評価 (`lamda_stage2_extractor.py`)
- [ ] pass_rate ≥ 90% 確認
- [ ] velocity_coverage.json の空帯域 < 20% 確認

### Phase 2: Bass/Strings/Guitar (各2-3日)

- [ ] 各楽器の `configs/lamda/<instrument>_stage2.yaml` 作成
- [ ] Stage2評価ループ実行
- [ ] 合格基準達成確認

### Phase 3: 他楽器のStage3移行 (Week 2)

- [ ] `conditions/piano.parquet` 作成 (条件集約)
- [ ] `adapters/run_piano_adapter.py` 実装
- [ ] Piano用CIゲート追加
- [ ] Bass/Strings/Guitarも同様に展開

---

## 🎯 成功指標

### drumgenerator (Phase 1完了時)

- ✅ `run_stage3_drum_eval.sh` が1コマンドで完走
- ✅ Bar violation = 0%
- ✅ Hat grid ≥ 0.85
- ✅ pass_rate ≥ 0.65
- ✅ CI が赤止めする (失敗時)
- ✅ `generation_logger.py` の JSONL に全生成が記録される

### 他楽器 (Phase 2完了時)

- ✅ Stage2 pass_rate ≥ 90%
- ✅ velocity_coverage 空帯域 < 20%
- ✅ articulation.auto.yaml が calibrate 済み
- ✅ Stage3移行準備完了

---

## 🔍 よくある詰まりと回避策

### 1. 旧イベント表現 → REMI変換

**症状**: 旧ジェネレーターが独自のイベント表現を使用  
**対策**: アダプタ内に変換層を追加 (本体は触らない)

```python
def _convert_to_remi(self, legacy_events: List) -> List:
    """旧イベント→REMI v1.1 変換"""
    remi_events = []
    for evt in legacy_events:
        if evt.type == "note":
            remi_events.extend([
                ("BAR", evt.bar),
                ("BEAT", evt.beat),
                ("NOTE", evt.pitch),
                ("VELOCITY", evt.velocity),
                ("DURATION", evt.duration),
            ])
    return remi_events
```

### 2. CLAP/MERT ばらつき

**症状**: 同じ音声で異なる embedding が出る  
**対策**: `audio_embedding_cache.py` の共通量子化関数を使用

```python
from audio_embedding_cache import quantize_embedding

# 必ず共通の量子化を通す
clap_bucket = quantize_embedding(clap_emb, num_buckets=10)
mert_bucket = quantize_embedding(mert_emb, num_buckets=10)
```

### 3. メタ欠落

**症状**: `.meta.json` が生成されず A/B追跡が壊れる  
**対策**: アダプタで **必ず** sidecar を生成 (atomic write)

### 4. 拍子/小節逸脱

**症状**: 生成されたMIDIが小節境界を超える  
**対策**: `build_forbidden_mask()` を推論に挿入

```python
from music_guards import build_forbidden_mask

# 推論時
forbidden_mask = build_forbidden_mask(time_sig="4/4", max_bars=64)
logits[:, forbidden_mask] = float('-inf')
```

---

## 📊 タイムライン (Week 1)

```
Day 1 (Mon):
  AM: LegacyGeneratorAdapter実装
  PM: run_drum_adapter.py + run_stage3_drum_eval.sh
  Evening: スモークテスト (3プロンプト×1サンプル)

Day 2 (Tue):
  AM: check_acceptance.py + テスト追加
  PM: CIゲート配線
  Evening: CI動作確認 → drumgenerator Stage3完成 ✅

Day 3-4 (Wed-Thu):
  Piano Stage2: YAML作成 → クリーニング → 評価 → pass_rate≥90%

Day 5-7 (Fri-Sun):
  Bass/Strings/Guitar Stage2 (並行可能)
```

---

## 🏁 まとめ

### 現在の状態

1. ✅ **Flash Attention評価完了**: N=2048で1.49x高速化を実証
2. ✅ **drumgenerator決定**: Standard Attention (FP32) で着手OK
3. ✅ **評価フレームワーク**: 完全な検証環境が整備済み

### 次のアクション

1. **drumgenerator Stage3完成** (1日)
   - LegacyGeneratorAdapter実装
   - ワンボタン評価スクリプト
   - CIゲート配線

2. **他楽器のLAMDA Stage2** (Week 1-2)
   - Piano → Bass → Strings → Guitar
   - pass_rate≥90% 達成後にStage3移行

3. **Flash Attention活用** (将来)
   - N≥2048のユースケース向けにドキュメント化
   - 長尺楽曲生成での活用

### 優先順位

**最優先**: drumgenerator Stage3完成 (Phase 1)  
**並行作業**: 他楽器のLAMDA Stage2 (Phase 2)  
**将来作業**: Flash Attention活用ガイド作成

---

**References**:
- Flash Attention評価: `results/FLASH_ATTENTION_N2048_ANALYSIS.md`
- Attention決定記録: `docs/ATTENTION_DECISION_RECORD.md`
- 改革案原文: (このメッセージ)
