# Stage3 Conditions Schema

## Overview

The `conditions/stage3_conditions.parquet` file aggregates all conditioning data for Stage3 generation, combining Stage2 metrics with emotion labels, captions, techniques, and audio embeddings.

## Required Columns

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `loop_id` | string | Unique loop identifier | Non-null, unique |
| `file_digest` | string | MD5 digest of source MIDI file | Non-null |

## Optional Columns (Conditioning Data)

| Column | Type | Description | Null Rate Limit |
|--------|------|-------------|-----------------|
| `emotion` | string | XMIDI emotion label (e.g., "happy", "sad") | ≤10% |
| `genre` | string | XMIDI genre label (e.g., "rock", "jazz") | ≤10% |
| `valence` | float | Emotion valence score [0, 1] | ≤20% |
| `arousal` | float | Emotion arousal score [0, 1] | ≤20% |
| `caption` | string | MetaScore-generated caption (Japanese/English) | ≤10% |
| `technique` | string | Comma-separated VPTT techniques (e.g., "ghost,flam") | ≤50% |
| `clap_embedding` | array(512) or string | CLAP audio embedding vector or path | ≤30% |
| `mert_embedding` | array(768) or string | MERT audio embedding vector or path | ≤30% |

## Stage2 Metrics (Inherited)

All columns from `loop_summary.csv` are preserved, including:

- `bpm`, `note_count`, `duration_ticks`
- `score.total`, `score.threshold_passed`
- `metrics.velocity_mean`, `metrics.ghost_rate`, etc.
- `articulation.*` fields

## Validation Rules

1. **Schema Compliance**: Required columns must exist with correct types
2. **Null Rate Limits**: Optional columns must not exceed specified null rates
3. **Value Ranges**: 
   - `valence` and `arousal` must be in [0, 1]
   - Negative scores or out-of-range values trigger errors
4. **Uniqueness**: `loop_id` must be unique across all rows
5. **Referential Integrity**: `file_digest` should correspond to existing MIDI files in `output/drumloops_cleaned/`

## Usage

### Validation

```bash
python scripts/validate_conditions.py conditions/stage3_conditions.parquet
```

### Generation

```bash
python scripts/collect_conditions.py \
  --stage2-summary output/drumloops_stage2/loop_summary.csv \
  --xmidi-labels outputs/stage3/xmidi_labels.csv \
  --captions outputs/stage3/music_captions.jsonl \
  --technique-meta outputs/stage3/technique_metadata.jsonl \
  --audio-cache outputs/stage3/embedding_cache \
  --output conditions/stage3_conditions.parquet \
  --stats-output conditions/stats.json
```

## CI Integration

The schema is validated automatically in CI via:

```yaml
- name: Validate Conditions Schema
  run: python scripts/validate_conditions.py conditions/stage3_conditions.parquet --strict
```

## Version History

- **v1.0** (2025-10-12): Initial schema with XMIDI, MetaScore, VPTT, CLAP/MERT support
