# DurationHumanizeAI Demo Plans

This folder hosts reproducible JSON plans that exercise the shared DurationHumanizeAI
annotation pipeline across piano, guitar, and strings.

## Inputs

All demos reuse the Phase 2 bass sandbox data:

- `sandbox/bass_demo/bars_with_slots.parquet`
- `sandbox/bass_demo/sections.json`
- `sandbox/bass_demo/chordmap_locked_extended.json`
- `policy/song_policy.template.yaml`

## Regenerating the plans

Run the generators from the repository root (torch is optional; when absent the
heuristic portion of DurationHumanizeAI still annotates every event). To capture
the EmotionAI → DurationHumanizeAI path, reuse the demo profile + rulebook that
live in this folder.

```bash
python3 scripts/generate_piano_plan_v2.py \
  --bars sandbox/bass_demo/bars_with_slots.parquet \
  --sections sandbox/bass_demo/sections.json \
  --chordmap sandbox/bass_demo/chordmap_locked_extended.json \
  --policy policy/song_policy.template.yaml \
  --out sandbox/humanize_demo/piano_plan_duration_ai.json \
  --seed 13 \
  --rhythm-manifest data/rhythm_vocab.yaml \
  --emotion-profile sandbox/humanize_demo/emotion_profile_demo.json \
  --rulebook sandbox/humanize_demo/rulebook_demo.yaml

python3 scripts/generate_guitar_plan_v2.py \
  --bars sandbox/bass_demo/bars_with_slots.parquet \
  --sections sandbox/bass_demo/sections.json \
  --chordmap sandbox/bass_demo/chordmap_locked_extended.json \
  --policy policy/song_policy.template.yaml \
  --out sandbox/humanize_demo/guitar_plan_duration_ai.json \
  --seed 7 \
  --rhythm-manifest data/rhythm_vocab.yaml

python3 scripts/generate_strings_plan_v2.py \
  --bars sandbox/bass_demo/bars_with_slots.parquet \
  --sections sandbox/bass_demo/sections.json \
  --chordmap sandbox/bass_demo/chordmap_locked_extended.json \
  --policy policy/song_policy.template.yaml \
  --out sandbox/humanize_demo/strings_plan_duration_ai.json \
  --seed 5 \
  --rhythm-manifest data/rhythm_vocab.yaml
```

Supplying a valid `humanize_duv.ckpt` per instrument plus `torch` will enable
full DUV inference; otherwise only the policy-driven heuristics are applied.
Each generated plan embeds RhythmAI vocabulary IDs inside the `humanize`
payload, demonstrating that metadata flows from RhythmAI → DurationHumanizeAI.
When the demo EmotionAI assets above are used, `metadata.emotion_tracking.*`
is populated and every `event.humanize.emotion` entry mirrors the per-bar
EmotionAI snapshot that DurationHumanizeAI consumed.
