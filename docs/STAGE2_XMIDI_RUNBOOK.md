# Stage2/XMIDI Integration Runbook

前回のコマンドを順次実行して、XMIDI感情ラベルをStage3に統合します。

## 1. XMIDIマニフェストビルド(drum labelあり)

```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

PYTHONPATH=. .venv311/bin/python scripts/build_xmidi_manifest.py \
  --xmidi-root data/XMIDI_Dataset \
  --output-manifest manifests/lamd_xmidi.jsonl \
  --output-labels outputs/stage3/xmidi_labels.csv \
  --drum-mapping config/drum_label_mapping.yaml
```

## 2. Stage2実行(XMIDI Dataset全体)

```bash
PYTHONPATH=. .venv311/bin/python scripts/lamda_stage2_extractor.py \
  --config configs/lamda/drums_stage2.yaml \
  --input-dir data/XMIDI_Dataset \
  --output-dir output/xmidi_stage2 \
  --emit-csv aggregate \
  --summary-out output/xmidi_stage2/stage2_summary.json \
  --streaming \
  --resume
```

## 3. loop_idカバレッジ確認

```bash
python - <<'PY'
import pandas as pd
s = pd.read_csv("output/xmidi_stage2/loop_summary.csv", usecols=["loop_id"])
l = pd.read_csv("outputs/stage3/xmidi_labels.csv", usecols=["loop_id"])
coverage = s.loop_id.isin(l.loop_id).mean()
print(f"loop_id coverage: {coverage:.1%}")
if coverage < 0.99:
    print(f"WARNING: Low coverage - check path alignment")
    missing = s[~s.loop_id.isin(l.loop_id)].head(10)
    print(f"\nMissing samples:\n{missing}")
else:
    print("✅ Coverage OK - proceed to Stage3")
PY
```

## 4. Stage3 conditions再構築

```bash
PYTHONPATH=. .venv311/bin/python scripts/collect_conditions.py \
  --stage2-summary output/xmidi_stage2/loop_summary.csv \
  --xmidi-labels outputs/stage3/xmidi_labels.csv \
  --captions outputs/stage3/music_captions.jsonl \
  --technique-meta outputs/stage3/technique_synth/technique_metadata.jsonl \
  --audio-cache outputs/stage3/embedding_cache \
  --output conditions/stage3_conditions.parquet \
  --stats-output conditions/stage3_conditions_stats.json
```

## 5. 検証

```bash
# Null率確認
cat conditions/stage3_conditions_stats.json

# Schema検証
PYTHONPATH=. .venv311/bin/python scripts/validate_conditions.py \
  conditions/stage3_conditions.parquet \
  --strict

# Drum labelカバレッジ確認
python - <<'PY'
import pandas as pd
df = pd.read_parquet("conditions/stage3_conditions.parquet")
print(f"Total rows: {len(df)}")
print(f"\nEmotion null rate: {df.emotion.isna().mean():.1%}")
print(f"Drum label null rate: {df.drum_label.isna().mean():.1%}")
print(f"\nDrum label distribution:")
print(df.drum_label.value_counts().head(10))
PY
```

## 6. ArrangerフィルタリングAPI使用例

```python
from pathlib import Path
from otobonAI.arranger import load_emotion_catalog, apply_qa_filters
import pandas as pd

# Emotion catalogロード
catalog = load_emotion_catalog(Path("outputs/stage3/xmidi_labels.csv"))

# 高エネルギーループのフィルタ
high_energy = catalog.filter(arousal_min=0.7)
print(f"High energy loops: {len(high_energy)}")

# Stage2 summaryにQAフィルタ適用
summary = pd.read_csv("output/xmidi_stage2/loop_summary.csv")
filtered = apply_qa_filters(
    summary,
    catalog,
    min_score=70.0,
    exclude_retry=True,
    allowed_drum_labels=["aggressive_halftime", "intense_energy"]
)
print(f"QA passed: {len(filtered)} / {len(summary)}")

# Arrangerコンテキスト取得
loop_id = filtered.iloc[0]["loop_id"]
context = catalog.get_context(loop_id)
print(f"Context: {context}")
```

## 7. EmotionAIフィードバック確認

EmotionAI v2は自動的に`velocity_scale`, `density_scale`を出力します。
Stage3 statsに記録されているか確認:

```bash
python - <<'PY'
import json
with open("conditions/stage3_conditions_stats.json") as f:
    stats = json.load(f)
print("Stage3 stats keys:")
print(json.dumps(list(stats.keys()), indent=2))

# EmotionAI delta統計があるか確認
if "emotion_ai_deltas" in stats:
    print("\n✅ EmotionAI feedback found")
    print(json.dumps(stats["emotion_ai_deltas"], indent=2))
else:
    print("\n⚠️  EmotionAI feedback not yet logged - extend collect_conditions.py")
PY
```

## 次のステップ

1. **QA gate拡張**: `config/quality_gates.yaml`の`drums.label_overrides`を使用
2. **Arrangerパイプライン統合**: `catalog.get_context(loop_id)`をアレンジャーに注入
3. **EmotionAIテレメトリ**: `scripts/collect_conditions.py`に`emotion_ai_deltas`集計を追加

詳細は`docs/XMIDI_MANIFEST.md`と`otobonAI/arranger/filters.py`を参照。
