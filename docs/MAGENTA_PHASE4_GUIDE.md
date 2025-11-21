# Magenta Phase 4 Implementation Guide — 装飾レイヤ

## Overview

Magenta fills/arpeggios are injected as **spice** (not backbone) into the V2 arrangement system via the `fill_slot` and `riff_slot` mechanism.

### Architecture

```
bars_with_slots.parquet (fill_slot, riff_slot)
         ↓
inject_magenta_fills.py (policy-driven sampling)
         ↓
magenta_fills.json (events with bar_start/bar_end)
         ↓
arrangement_orchestrator.py (merge with other plans)
         ↓
Final arrangement_plan.json → MIDI
```

---

## Components

### 1. Fill Generator (`otobonAI/magenta_fill_generator.py`)

**Purpose**: Generate fills/arpeggios using Magenta models.

**Current State**: Prototype with arpeggio patterns (no checkpoint).

**Usage**:
```python
from otobonAI.magenta_fill_generator import MagentaFillGenerator

gen = MagentaFillGenerator(model_path="models/groovae_2bar_humanize.ckpt")
fills = gen.generate_fills(
    section="chorus",
    bars=[16, 17, 18, 19],
    chordmap_locked=chordmap,
    guide_tone_hints=hints,
    policy={"temperature": 0.8, "max_events": 32}
)
```

**Future Work**: Load Magenta MusicVAE checkpoint for real generation.

---

### 2. Policy YAML (`config/magenta_policy.yaml`)

**Purpose**: Control Magenta usage rate, temperature, event limits.

**Key Settings**:
```yaml
enabled: true
magenta_use_prob: 0.3  # 30% of fill slots use Magenta

fill_policy:
  temperature: 0.8
  max_events: 32
  use_guide_tones: true

section_overrides:
  intro:
    magenta_use_prob: 0.5  # Higher for variety
  verse:
    magenta_use_prob: 0.2  # Lower for clarity
```

**Usage**: Adjust `magenta_use_prob` to control spice level.

---

### 3. QA Gates (`config/quality_gates.yaml`)

**Purpose**: Prevent Magenta from overwhelming the arrangement.

**Constraints**:
```yaml
magenta:
  defaults:
    max_event_ratio: 0.25  # Magenta <= 25% of total events
    max_consecutive_bars: 4
    min_rest_bars: 2
    velocity_range: [40, 110]
```

**Validation**:
```python
from otobonAI.qa.magenta_qa import MagentaQA

qa = MagentaQA.from_yaml("config/quality_gates.yaml")
result = qa.validate(fills, all_events, section="chorus")

if not result.passed:
    print(f"❌ QA FAIL: {result.violations}")
```

---

### 4. Injection Script (`scripts/inject_magenta_fills.py`)

**Purpose**: Inject fills into arrangement at fill_slot positions.

**Workflow**:
1. Load `bars_with_slots.parquet` (contains fill_slot, riff_slot)
2. Filter bars by `magenta_use_prob` (random sampling)
3. Generate fills via `MagentaFillGenerator`
4. Run QA validation (`MagentaQA`)
5. Save to `magenta_fills.json`

**Usage**:
```bash
python scripts/inject_magenta_fills.py \
  --bars-with-slots data/song_004/analysis/bars_with_slots.parquet \
  --chordmap data/song_004/locked_chordmap.json \
  --guide-tones data/song_004/guide_tones.json \
  --policy config/magenta_policy.yaml \
  --output data/song_004/magenta_fills.json \
  --validate
```

---

## Integration with Arrangement Orchestrator

### Step 1: Generate Magenta Fills

```bash
python scripts/inject_magenta_fills.py \
  --bars-with-slots data/song_004/analysis/bars_with_slots.parquet \
  --chordmap data/song_004/locked_chordmap.json \
  --output data/song_004/magenta_fills.json
```

### Step 2: Merge into Arrangement

```bash
python scripts/arrangement_orchestrator.py \
  --plan data/song_004/bass_plan.json \
  --plan data/song_004/guitar_plan.json \
  --plan data/song_004/piano_plan.json \
  --plan data/song_004/drums_plan.json \
  --plan data/song_004/magenta_fills.json \
  --tempo-bpm 120 \
  --ppq 480 \
  --output data/song_004/arrangement_plan.json
```

### Step 3: Render to MIDI

```bash
python scripts/json2midi.py \
  --input data/song_004/arrangement_plan.json \
  --output data/song_004/final.mid
```

---

## Quality Assurance Checklist

### Pre-Generation
- ✅ `magenta_use_prob` set appropriately (default: 0.3)
- ✅ `max_events` prevents overcrowding (default: 32)
- ✅ `temperature` balanced (default: 0.8)

### Post-Generation
- ✅ Run `--validate` flag to check QA gates
- ✅ Verify `max_event_ratio < 0.25` (Magenta <= 25%)
- ✅ Check `max_consecutive_bars <= 4`
- ✅ Listen test: fills enhance (not overpower) backbone

### Common Issues

**Issue**: Too many Magenta fills
- **Fix**: Lower `magenta_use_prob` in policy

**Issue**: QA fails on `max_event_ratio`
- **Fix**: Reduce `max_events` or `magenta_use_prob`

**Issue**: Pitch outliers
- **Fix**: Enable `use_guide_tones: true` in policy

---

## Roadmap Progress

### Phase 4 Checklist (装飾レイヤ)

- ✅ Create `MagentaFillGenerator` with prototype arpeggio
- ✅ Create `config/magenta_policy.yaml` with usage controls
- ✅ Extend `config/quality_gates.yaml` with Magenta constraints
- ✅ Create `MagentaQA` validator
- ✅ Create `inject_magenta_fills.py` integration script
- ✅ Document integration workflow
- ⏳ Load Magenta checkpoint (MusicVAE/GrooVAE) — **TODO**
- ⏳ Real Song test with Magenta fills enabled — **TODO**

### Next Phase (Phase 5: QA/CI ワンボタン化)

After Magenta Phase 4 completes, proceed to:
- One-button CI pipeline (build + QA + render)
- Automated quality reports
- Batch testing infrastructure

---

## Example Output

```json
{
  "fills": [
    {
      "bar_start": 16,
      "bar_end": 17,
      "events": [
        {"pitch": 60, "start_ql": 0.0, "duration_ql": 0.225, "velocity": 82},
        {"pitch": 64, "start_ql": 0.25, "duration_ql": 0.225, "velocity": 79},
        {"pitch": 67, "start_ql": 0.5, "duration_ql": 0.225, "velocity": 85}
      ],
      "source": "magenta_arpeggio",
      "confidence": 0.5
    }
  ],
  "total_fills": 1,
  "total_events": 3
}
```

---

## References

- Real Song Roadmap v2: `docs/real_song_roadmap_v2.md`
- Fill/Riff Slot System: `scripts/add_fill_riff_slots.py`
- Arrangement Orchestrator: `scripts/arrangement_orchestrator.py`
- Magenta GrooVAE: `ops/magenta_groove.py`
