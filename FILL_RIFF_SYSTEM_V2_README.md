# Fill/Riff System V2: Slot-Based Collaborative Architecture

## Overview (2025-11-12)

**ボーカル中心のアゲアゲ装置 (Vocal-Centric Fill/Riff System)** を実装しました。

### Design Philosophy

```
「位置決め（スロット）は bars/sections。表現の造形は楽器別レンダラ。music21は和声支援のみ。」
```

- **Slot Planner**: `bars_with_slots.parquet` (fill_slot/riff_slot - where to fire)
- **Policy YAML**: Suno AI style prompt準拠 (how to fire - density, patterns, accent)
- **Instrument Renderers**: generate_*_plan_v2.py (what notes to play)
- **Harmony Resolver**: chordmap_locked_extended.json + music21 (voicing support)

### Architecture (三段ロケット協調型)

```
1. recommend_drums (optional): Pattern suggestions [FUTURE]
   ↓
2. generate_*_plan_v2 (core): Slot-based rendering ← PRIORITY
   ↓
3. adapt_drums_to_plan (optional): Kit/humanization [FUTURE]
   ↓
4. postprocess_plans_ignore_mute: Mute removal (always active)
```

## V2 Renderers (Completed)

### Melody Hint Manifest Workflow (NEW)

- `generate_piano_plan_v2.py`, `generate_strings_plan_v2.py`, and `generate_piano_strings_plans.py` now share a `--emit-melody-manifest` flag (paired with optional `--melody-manifest-path`).
- When `--vocal-f0` is supplied, each script exports `melody_hint_manifest.json`, capturing CREPE-derived per-bar stats (`voiced_ratio`, `slide_activity`, tags) plus metadata/inputs.
- Piano plans **annotate** every event overlapping a hint (for downstream guide-tone layering), while strings plans **drop** long-hint bars to keep the vocal lane clear; downstream pipelines only need to read the manifest to understand which bars were protected.
- The helper CLI `scripts/build_melody_hint_manifest.py` can produce the same manifest offline so Phase A / make_song_package workflows can persist the hints before rendering begins.
- Example end-to-end call:

```bash
python scripts/generate_piano_strings_plans.py \
  --song-dir song_packages/suno_project/song_001 \
  --config configs/arranger_weights.yaml \
  --vocal-f0 song_packages/suno_project/song_001/vocal_f0_crepe.parquet \
  --emit-piano --emit-strings --emit-melody-manifest
```

The resulting `song_packages/.../melody_hint_manifest.json` is referenced by `make_song_package_phase_a.sh` to keep the melody guard-rails intact through later orchestration.

### 1. Drums V2 (`scripts/generate_drums_plan_v2.py`)

**Status**: ✅ Implemented, tested (816 events)

**Features**:
- Boundary fill guarantee (always mode → 100% section end-1 coverage)
- fill_slot-driven fill generation (short/standard/long fills)
- Section density matrix (intro 0.3 → chorus 0.95)
- Accent patterns (buildup/uplifting/celebration curves)
- JSON serialization safe (all numpy→int/float converted)

**Hooks**:
- `--use-recommender`: Call recommend_drums.py [NOT IMPLEMENTED]
- `--post-adapt`: Call adapt_drums_to_plan.py [NOT IMPLEMENTED]

**Test**:
```bash
python3 scripts/generate_drums_plan_v2.py \
  --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
  --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
  --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
  --out /tmp/drums_test.json
```

#### Groove Vocabulary + RhythmAI Workflow (2025-11)

1. **Build groove vocab from Stage2** (outputs `data/groove_vocab.parquet` + stats JSON):

    ```bash
    .venv311/bin/python scripts/extract_groove_vocab.py \
      --stage2-dir outputs/stage2_drums_iter8_100PCT \
      --output-parquet data/groove_vocab.parquet \
      --output-stats data/groove_vocab_stats.json
    ```

   - Accepts `--labels-csv` when extra drum/emotion tags exist.
   - Filters out low-scoring loops (`score.total < 55` by default) before saving the parquet consumed by RhythmAI.

2. **Render drums with RhythmAI suggestions** (metadata includes `rhythm_ai.*` tags so downstream QA can track which bars used vocab patterns):

    ```bash
    .venv311/bin/python scripts/generate_drums_plan_v2.py \
      --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
      --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
      --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
      --groove-vocab data/groove_vocab.parquet \
      --tempo-bpm 90 \
      --out data/suno_ai/suno_themesong/song_004/plans/drums_plan_v2_rhythmai.json
    ```

   - Uses per-bar `drum_label`/`emotion` hints when available.
   - Plan metadata records whether the vocab was loaded plus the number of RhythmAI-tagged events (523 events for song_004 as of 2025-11-18).

3. **Baseline comparison (no RhythmAI)** for deterministic smoke tests or when the vocab is missing:

    ```bash
    .venv311/bin/python scripts/generate_drums_plan_v2.py \
      --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
      --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
      --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
      --groove-vocab data/groove_vocab.parquet \
      --tempo-bpm 90 \
      --disable-rhythmai \
      --out data/suno_ai/suno_themesong/song_004/plans/drums_plan_v2_no_ai.json
    ```

   - This run still records fill/slot metrics but generates the legacy deterministic patterns (1,043 events for song_004, useful for A/B diffs and regression gating).

> ✅ Recommendation: include both outputs in QA. The RhythmAI-enabled plan captures swing/density metadata, while the disabled run provides the historical baseline for diff tooling.

4. **CI / automation shortcut** — run both steps with one command so Stage2 completion immediately refreshes the vocab and plans:

    ```bash
    bash scripts/stage2_to_rhythmai.sh \
      --stage2-dir outputs/stage2_drums_iter8_100PCT \
      --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
      --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
      --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
      --song-root data/suno_ai/suno_themesong/song_004 \
      --rhythmai-out data/suno_ai/suno_themesong/song_004/plans/drums_plan_v2_rhythmai.json \
      --tempo-bpm 90
    ```

   This helper script rebuilds `data/groove_vocab.parquet` and emits both RhythmAI + baseline plans so CI smoke tests only need a single hook.

     **Hook ideas**

     - *Stage2 completion (bash)* – append the following to the end of your Stage2 driver (e.g., `RUN_SONG_004.sh`) so every successful batch refreshes the vocab + plans:

          ```bash
          if [[ -f "$STAGE2_DIR/loop_summary.csv" ]]; then
            bash scripts/stage2_to_rhythmai.sh \
              --stage2-dir "$STAGE2_DIR" \
              --bars "$SONG_ROOT/analysis/bars_with_slots.parquet" \
              --sections "$SONG_ROOT/analysis/sections.json" \
              --policy "$SONG_ROOT/policy/song_004.yaml" \
              --song-root "$SONG_ROOT" \
              --rhythmai-out "$SONG_ROOT/plans/drums_plan_v2_rhythmai.json" \
              --tempo-bpm "$DEFAULT_TEMPO"
          fi
          ```

     - *CI smoke test (GitHub Actions example)* – drop this job into `.github/workflows/ci.yml` to ensure every push regenerates the vocab and both plans for regression diffs:

          ```yaml
          jobs:
            rhythmai-smoke:
              runs-on: ubuntu-latest
              steps:
                - uses: actions/checkout@v4
                - uses: actions/setup-python@v5
                  with:
                    python-version: '3.11'
                - run: pip install -r requirements.txt
                - run: |
                    bash scripts/stage2_to_rhythmai.sh \
                      --stage2-dir outputs/stage2_drums_iter8_100PCT \
                      --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
                      --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
                      --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
                      --song-root data/suno_ai/suno_themesong/song_004 \
                      --rhythmai-out data/suno_ai/suno_themesong/song_004/plans/drums_plan_v2_rhythmai.json
          ```

     Adjust the paths/song IDs as needed per build.

### 2. Guitar V2 (`scripts/generate_guitar_plan_v2.py`)

**Status**: ✅ Implemented, tested (240 events)

**Features**:
- riff_slot-driven riff generation (chorus/pre_chorus/bridge)
- Riff types: strum (50%), broken_chord (30%), single_note (20%)
- Comping: offbeat chords (section density-based)
- Keeps tensions: sus2, sus4, add9 (Aimyon acoustic sweet spots)

**Test**:
```bash
python3 scripts/generate_guitar_plan_v2.py \
  --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
  --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
  --chordmap data/suno_ai/suno_themesong/song_004/analysis/chordmap_locked_extended.json \
  --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
  --out /tmp/guitar_test.json
```

### 3. Piano V2 (`scripts/generate_piano_plan_v2.py`)

**Status**: ✅ Implemented, tested (60 events)

**Features**:
- fill_slot-driven fill decorations (section boundary end-1)
- Comping styles (section-based): verse (offbeat 50%, arpeggio 30%), chorus (block 70%)
- Arpeggio: ascending (希望感), 15% probability
- Keeps tensions: 7, 9 (warm voicing)

**Test**:
```bash
python3 scripts/generate_piano_plan_v2.py \
  --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
  --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
  --chordmap data/suno_ai/suno_themesong/song_004/analysis/chordmap_locked_extended.json \
  --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
  --out /tmp/piano_test.json
```

### 4. Strings V2 (`scripts/generate_strings_plan_v2.py`)

**Status**: ✅ Implemented, tested (9 events)

**Features**:
- riff_slot-driven countermelody (chorus/bridge)
- Countermelody styles: call_response (40%), ascending_line (30%), sustain_pad (30%)
- Oriental flavor: Em harmonic minor scale hint
- Crescendo on: pre_chorus, chorus
- CREPE F0 reference (optional, call-response intervals [3, 6, 10])

**Test**:
```bash
python3 scripts/generate_strings_plan_v2.py \
  --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
  --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
  --chordmap data/suno_ai/suno_themesong/song_004/analysis/chordmap_locked_extended.json \
  --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
  --vocal-f0 data/suno_ai/suno_themesong/song_004/vocal_f0_crepe.parquet \
  --out /tmp/strings_test.json
```

### 5. Bass V2 (`scripts/generate_bass_plan_v2.py`)

**Status**: ✅ Implemented, tested (274 events)

**Features**:
- Always active (always_active: true)
- Root foundation (bass octave, chord root only)
- Pattern types: root_quarter (60%), root_eighth (25%), walking (15%)
- Walking bass: chorus/bridge uplifting (root, 3rd, root, 5th)

**Test**:
```bash
python3 scripts/generate_bass_plan_v2.py \
  --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
  --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
  --chordmap data/suno_ai/suno_themesong/song_004/analysis/chordmap_locked_extended.json \
  --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
  --out /tmp/bass_test.json
```

## Pipeline Integration

### STEP 6: Slot Planner (Completed)

```bash
python3 scripts/add_fill_riff_slots.py \
  --bars analysis/bars.parquet \
  --sections analysis/sections.json \
  --out analysis/bars_with_slots.parquet
```

**Output**: bars_with_slots.parquet (75% fill coverage, 60.3% riff coverage, 100% boundary coverage)

### STEP 19: V2 Renderers (Integrated for Drums, Pending for Others)

**Current Status**:
- ✅ Drums V2: Integrated with 3-tier fallback + postprocess_plans_ignore_mute
- ⚠️ Guitar/Piano/Strings/Bass V2: Standalone scripts ready, pipeline integration pending

**Integration Priority**:
1. V2 renderers (preferred) - use bars_with_slots.parquet + policy YAML
2. Legacy generators (fallback 1) - use chordmap_view_*.json
3. instrument_midi_to_plan_real.py (fallback 2) - use real MIDI sources

### STEP 20: MIDI Integration (Ready)

```bash
python3 scripts/json2midi.py \
  --tempo-map analysis/tempo_map.json \
  --split-tracks \
  plans/*.json -o midi/integrated.mid
```

**Features**:
- Variable tempo support (tempo_map.json)
- Track separation (drums: Ch.10, others: GM program)
- Plan-only workflow (no WAV dependency)

## Quality Gate (Pending Integration)

```bash
python3 scripts/quality_gate_fill_riff.py \
  --plans-dir plans/ \
  --bars bars_with_slots.parquet \
  --sections sections.json \
  --policy policy/song_004.yaml
```

**Checks**:
- Boundary fill rate ≥ 80%
- Chorus riff rate ≥ 30%
- Max 16th density < 75%

**Exit codes**: 0=pass, 1=warn+continue, 2=error (abort if STRICT=1)

## Policy YAML (song_004.yaml)

**Status**: ✅ Completed (474 lines, Suno AI style prompt準拠)

**Suno Style Prompt**:
```
J-Pop, Human Anthem, Storytelling Cypher, Ketsumeishi-style, GReeeeN-style, Yo Hitoto-style, Aimyon-style
```

**Structure**:
- Metadata: Suno prompt, emotional journey (hope → connection → celebration)
- Global: voicing_policy, voice_leading_bias (0.7), density_smoothing (0.4)
- Sections: 6 sections (intro/verse/pre_chorus/chorus/bridge/outro) with density matrix
- Instruments: Piano, Guitar, Strings, Bass, Drums (individual configs)
- Integration: vocal_priority=high, emotion_velocity_scale

**Density Matrix**:
```yaml
sections:
  intro: {piano: 0.25, guitar: 0.2, strings: 0.3, bass: 1.0, drums: 0.3}
  verse: {piano: 0.5, guitar: 0.4, strings: 0.4, bass: 1.0, drums: 0.5}
  pre_chorus: {piano: 0.7, guitar: 0.6, strings: 0.6, bass: 1.0, drums: 0.7}
  chorus: {piano: 0.9, guitar: 0.8, strings: 0.7, bass: 1.0, drums: 0.95}
  bridge: {piano: 0.6, guitar: 0.5, strings: 0.8, bass: 1.0, drums: 0.7}
  outro: {piano: 0.35, guitar: 0.3, strings: 0.5, bass: 1.0, drums: 0.4}
```

## Legacy Assets (Preserved)

### Keep (Fallback)
- `scripts/drums_midi_to_plan_real.py` - Real MIDI source fallback
- `scripts/bass/generate_bass_plan.py` - Legacy V1 (chordmap_view_bass.json)
- `scripts/guitar/generate_guitar_plan.py` - Legacy V1 (chordmap_view_guitar.json)
- `scripts/piano/generate_piano_plan.py` - Legacy V1 (chordmap_view_piano.json)
- `scripts/strings/generate_strings_plan.py` - Legacy V1 (chordmap_view_strings.json)

### Deprecated (Skipped in Pipeline)
- `scripts/recommend_drums.py` - Pattern recommendations (superseded by V2 slot system)
- `scripts/adapt_drums_to_plan.py` - Kit conversion (future integration as --post-adapt hook)
- `analysis/drum_accent_plan.json` - Auxiliary file (superseded by bars_with_slots.parquet)

## Migration Path

### Phase 1: V2 Standalone (CURRENT)
- ✅ All V2 renderers working independently
- ✅ Drums V2 integrated into pipeline (STEP 19)
- ⏭️ Guitar/Piano/Strings/Bass V2 callable manually

### Phase 2: Full Pipeline Integration (NEXT)
- Update STEP 19 for all instruments (V2 priority + fallback)
- Add quality gate (STEP 19.5 or STEP 21)
- End-to-end test (song_004 full pipeline)

### Phase 3: Collaborative Hooks (FUTURE)
- Implement `--use-recommender` (call recommend_drums from V2)
- Implement `--post-adapt` (call adapt_drums_to_plan from V2)
- Magenta integration (via recommender)

## File Tree (Relevant)

```
scripts/
├── generate_drums_plan_v2.py          # ✅ Drums V2 (collaborative architecture)
├── generate_guitar_plan_v2.py         # ✅ Guitar V2 (riff slots)
├── generate_piano_plan_v2.py          # ✅ Piano V2 (fill decorations)
├── generate_strings_plan_v2.py        # ✅ Strings V2 (countermelody)
├── generate_bass_plan_v2.py           # ✅ Bass V2 (always active)
├── add_fill_riff_slots.py             # ✅ Slot planner
├── quality_gate_fill_riff.py          # ✅ Quality gate
├── postprocess_plans_ignore_mute.py   # ✅ Mute removal
├── make_song_package_from_sources.sh  # ✅ Pipeline (Drums V2 integrated)
├── drums_midi_to_plan_real.py         # Fallback (real MIDI)
├── recommend_drums.py                 # [DEPRECATED] (superseded by V2)
├── adapt_drums_to_plan.py             # [FUTURE] (--post-adapt hook)
└── [bass|guitar|piano|strings]/generate_*_plan.py  # Legacy V1 (fallback)

data/suno_ai/suno_themesong/song_004/
├── analysis/
│   ├── bars_with_slots.parquet        # ✅ Slot source (75% fill, 60% riff)
│   ├── chordmap_locked_extended.json  # ✅ Chord source (G# dominance fixed)
│   ├── sections.json                  # ✅ Section boundaries
│   └── tempo_map.json                 # ✅ Variable tempo
├── policy/
│   └── song_004.yaml                  # ✅ Suno style preset (474 lines)
└── plans/
    ├── drums_plan.json                # V2 output (816 events)
    ├── guitar_plan.json               # V2 output (240 events)
    ├── piano_plan.json                # V2 output (60 events)
    ├── strings_plan.json              # V2 output (9 events)
    └── bass_plan.json                 # V2 output (274 events)
```

## Summary

**What We Built**:
- ✅ 5 slot-based renderers (Drums, Guitar, Piano, Strings, Bass)
- ✅ Collaborative architecture (hooks for recommend/adapt)
- ✅ Policy-driven (Suno AI style prompt準拠)
- ✅ Fallback-safe (legacy generators preserved)
- ✅ Plans-only workflow (no WAV dependency)

**What's Next**:
1. Complete STEP 19 integration for Guitar/Piano/Strings/Bass V2
2. Add quality gate to pipeline (STEP 19.5)
3. End-to-end test (song_004 full pipeline → MIDI → DAW)
4. Implement collaborative hooks (--use-recommender, --post-adapt)

**Design Wins**:
- "位置決めはbars/sections。造形は楽器別レンダラ。" → Clean separation of concerns
- 三段ロケット (recommend → V2 → adapt) → Extensible without breaking changes
- Slot system → Guaranteed fills/riffs at intended locations
- Policy YAML → Curator-friendly, AI-ready (future Magenta/演奏法AI integration)
