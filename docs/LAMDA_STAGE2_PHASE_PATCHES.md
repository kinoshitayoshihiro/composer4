# LAMDA Stage2 拡張パッチ適用完了（Phase 16/31/13/15/22/25/29/32）

**適用日**: 2025-10-23  
**スキーマバージョン**: `lamda_v2.3` → `lamda_v2.5`  
**対象ファイル**: `scripts/lamda_stage2_extractor.py`

---

## ✅ 適用完了パッチ一覧

### ① Key / Modulations（Phase16/31）

**目的**: ローカルキー列と転調点の精度向上

**変更内容**:
- `utilities.harmonic_utils.estimate_local_key_sequence` を優先使用
- フォールバック: 簡易実装（root同名長調）
- スキーマ: `lamda_v2.3`

**コード変更**:
```python
# Import追加
try:
    from utilities import harmonic_utils
except Exception:
    harmonic_utils = None

# estimate_local_keys_extended() 内で優先パス実装
if harmonic_utils and hasattr(harmonic_utils, "estimate_local_key_sequence"):
    seq = harmonic_utils.estimate_local_key_sequence(chordmap, win_beats=win_bars*4) or {}
    keys = seq.get("keys", [])
    out["key_hint"] = [[i, k] for i, k in enumerate(keys)]
    out["modulations"] = seq.get("modulations", [])
    return out
```

**合格基準**:
- `key_hint` ≥ 曲の小節数/4
- `modulations` が単調増加timeで整合

---

### ② Sections（RMS+novelty注入）（Phase13/15）

**目的**: 外部sections.json（RMS+novelty）の優先採用

**変更内容**:
- `--sections-json` 引数追加
- `auto_sections_from_energy_extended()` で外部JSON優先パス実装
- `Stage2Settings` に `sections_json: Optional[Path]` 追加

**コード変更**:
```python
# 引数追加
parser.add_argument(
    "--sections-json",
    type=Path,
    default=None,
    help="precomputed sections.json (RMS+novelty) を優先採用",
)

# auto_sections_from_energy_extended() 内
if sections_json and Path(sections_json).exists():
    try:
        with open(sections_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
```

**使用例**:
```bash
python scripts/lamda_stage2_extractor.py \
  --input-dir output/stage1/test/clean \
  --output-dir output/stage2/test_sections \
  --sections-json suno_ai/suno_themesong/song_001/analysis/sections.json
```

**合格基準**:
- `sections.unit == "bar"`
- `sections.sections[0].label == "intro"`
- `energy` が 0..1 正規化
- cut間隔 ≥ 8bars

---

### ③ Groove（Phase22/25/29）

**目的**: グルーヴ特徴の精度向上

**変更内容**:
- `utilities.groove_sampler_v2.analyze` を優先使用
- スキーマ: `lamda_v2.4`

**コード変更**:
```python
# Import追加
try:
    from utilities import groove_sampler_v2
except Exception:
    groove_sampler_v2 = None

# analyze_groove_extended() 内
if groove_sampler_v2 and hasattr(groove_sampler_v2, "analyze"):
    try:
        g = groove_sampler_v2.analyze(midi_data, downbeats_ql)
        return {
            "swing_pct": float(g.get("swing_pct", 0.0)),
            "backbeat_strength": float(g.get("backbeat_strength", 0.0)),
            "onset_deviation_hist": g.get("onset_deviation_hist", []),
            "rhythm_hash": g.get("rhythm_hash"),
        }
    except Exception:
        pass
```

**合格基準**:
- `swing_pct` ∈ [0,100]
- `backbeat_strength` ∈ [0,1]
- `onset_deviation_hist` は配列（空でも可）

---

### ④ Controls（PB/RPN/CC, Phase32）

**目的**: PB±8191規約とRPN順序検査の厳密化

**変更内容**:
- `utilities.pb_math` を優先使用（半音換算）
- `rpn_bend_range_semitone` フィールド追加
- スキーマ: `lamda_v2.5`（最終）

**コード変更**:
```python
# Import追加
try:
    from utilities import pb_math
except Exception:
    pb_math = None

# summarize_controls_extended() 内
if pb_math and (pb_min != 0 or pb_max != 0):
    try:
        semi = max(abs(pb_math.pb_to_semi(pb_min)), abs(pb_math.pb_to_semi(pb_max)))
        bend_range_semi = float(round(semi, 2))
    except Exception:
        bend_range_semi = None

out["pb_range"] = [int(pb_min), int(pb_max)]
out["cc_summary"] = cc_summary
out["rpn_seen"] = bool(rpn_seen)
if bend_range_semi is not None:
    out["rpn_bend_range_semitone"] = bend_range_semi
```

**合格基準**:
- `pb_range` の端点は整数
- `rpn_seen` がbool
- `rpn_bend_range_semitone` が存在すれば数値（≥0）

---

## 🔄 スキーマバージョン推移

| パッチ段階 | スキーマバージョン | 主な機能追加                  |
| ----- | --------- | ----------------------- |
| 初期状態  | v2.1      | 基本拡張メタ                  |
| ①適用   | v2.3      | Key/Modulations精度向上      |
| ②適用   | v2.3      | Sections外部注入対応          |
| ③適用   | v2.4      | Groove精度向上              |
| ④適用   | v2.5      | Controls厳密化（最終）        |

---

## 🎯 監査ゲート（各段共通）

### 一致率の回帰検出（影響監視）

```bash
python scripts/ab_chord_audit.py \
  --ext-dir data/lamda_chordmaps \
  --int-dir output/stage2/test_*/*json* \
  --out-csv analysis/ab_chords_audit.csv

# 平均一致率をリングへ
python scripts/ringbuffer_append.py \
  --csv  analysis/ab_chords_audit.csv \
  --ring analysis/consistency_ring.jsonl \
  --tag  stage2_patch
```

### 合格ライン（例）

- `match_rate_mean >= 0.85`
- `controls.rpn_bend_range_semitone` の出現率増加
- `pb_range` の外れ値（>±8191）が0件

---

## 🚀 最終コマンド（統合実行）

```bash
# 1) (任意) sectionsを先に作る
python ops/sections_from_audio.py \
  --audio <mix.wav> \
  --out analysis/sections.json \
  --min-bars 8 \
  --chordmap analysis/chordmap.json

# 2) Stage2 抽出（外部chordmap優先＋WL検証＋sections注入）
python scripts/lamda_stage2_extractor.py \
  --input-dir  output/stage1/test/clean \
  --output-dir output/stage2/prod \
  --lamda-chords-dir data/lamda_chordmaps \
  --whitelist-validate 1 \
  --sections-json analysis/sections.json

# 3) 監査実行
python scripts/ab_chord_audit.py \
  --ext-dir data/lamda_chordmaps \
  --int-dir output/stage2/prod/json \
  --out-csv analysis/ab_audit_v2.5.csv
```

---

## 📊 テストコマンド（段階別）

### ① Key/Modulations テスト

```bash
python scripts/lamda_stage2_extractor.py \
  --input-dir output/stage1/test/clean \
  --output-dir output/stage2/test_key \
  --lamda-chords-dir data/lamda_chordmaps

grep -R '"key_hint"' -n output/stage2/test_key | head
```

### ② Sections テスト

```bash
python scripts/lamda_stage2_extractor.py \
  --input-dir output/stage1/test/clean \
  --output-dir output/stage2/test_sections \
  --sections-json suno_ai/song_001/analysis/sections.json

jq '.["extended.sections_auto"]' output/stage2/test_sections/*.json | head
```

### ③ Groove テスト

```bash
python scripts/lamda_stage2_extractor.py \
  --input-dir output/stage1/test/clean \
  --output-dir output/stage2/test_groove

jq '.["extended.groove"]' output/stage2/test_groove/*.json | head
```

### ④ Controls テスト

```bash
python scripts/lamda_stage2_extractor.py \
  --input-dir output/stage1/test/clean \
  --output-dir output/stage2/test_controls

jq '.["extended.controls"]' output/stage2/test_controls/*.json | head
```

---

## 🔧 旧資産の確認コマンド

```bash
# 旧資産が import できるか即時チェック
python - <<'PY'
mods = [
    "utilities.harmonic_utils",
    "utilities.groove_sampler_v2",
    "utilities.pb_math"
]
import importlib
for m in mods:
    try:
        importlib.import_module(m)
        print("✅ OK", m)
    except Exception as e:
        print("❌ NG", m, "|", e)
PY
```

---

## ⚙️ NO-OP動作の安全性

すべてのパッチは**旧資産が存在しない場合でも安全に動作**します：

- **harmonic_utils なし**: 簡易実装（root同名長調）にフォールバック
- **sections_json なし**: 既存の規則ラベリング（最小8小節）で動作
- **groove_sampler_v2 なし**: 既定値（swing_pct=0.0, backbeat_strength=0.5）
- **pb_math なし**: ±8191規約のまま、半音換算フィールドは出力しない

---

## 📝 運用ヒント

1. **段階的導入**: ①→②→③→④の順に本番投入推奨
2. **監査継続**: 各段階でリングバッファに記録し、時系列で一致率監視
3. **旧資産整備**: `utilities/` 配下の実装が揃えば自動的に精度向上
4. **品質ゲート**: `match_rate >= 0.85` を維持できることを確認

---

## 🎉 実装完了ステータス

| フェーズ        | 要件                | 実装状況 | スキーマ  |
| ----------- | ----------------- | ---- | ----- |
| Phase 16/31 | Key/Modulations   | ✅    | v2.3  |
| Phase 13/15 | Sections (RMS+novelty) | ✅    | v2.3  |
| Phase 22/25/29 | Groove            | ✅    | v2.4  |
| Phase 32    | Controls (PB/RPN) | ✅    | v2.5  |

**総合完了率**: 100%（4パッチすべて適用完了）

---

**最終更新**: 2025-10-23  
**次のアクション**: テスト実行 → 品質ゲート → 本番投入
