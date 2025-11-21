# XMIDI Lamda Manifest Integration

## Objectives
- Canonical Lamda-style IDs for every `XMIDI_*.midi` clip
- Emotion + genre metadata normalized to valence/arousal space for EmotionAI / arranger flows
- Stage3 condition CSV that plugs directly into `scripts/collect_conditions.py`

## Deterministic ID & digest strategy
| Field | Rule |
| --- | --- |
| `loop_id` / manifest `id` | `sha1("xmidi:" + relative_path)`; stable across machines as long as project-relative layout is preserved |
| `signature_digest` | `sha1` of MIDI file bytes (configurable to `path` mode via CLI) |
| `relative_path` | Path relative to repo root when possible; falls back to absolute POSIX path |

## Emotion normalization (`config/xmidi_mapping.yaml`)
11 XMIDI emotions receive fixed valence/arousal anchors (range -1..1). Example:

| Emotion | Valence | Arousal |
| --- | --- | --- |
| angry | -0.70 | 0.85 |
| exciting | 0.65 | 0.90 |
| fear | -0.80 | 0.75 |
| funny | 0.60 | 0.50 |
| happy | 0.80 | 0.65 |
| lazy | 0.20 | -0.40 |
| magnificent | 0.55 | 0.55 |
| quiet | 0.10 | -0.60 |
| romantic | 0.45 | 0.10 |
| sad | -0.75 | -0.35 |
| warm | 0.50 | 0.00 |

Each manifest row keeps `emotion_meta` / `genre_meta` blocks so arranger tooling can recover hints (tempo ranges, swing expectations, palette colors).

## Usage
```
PYTHONPATH=. .venv311/bin/python scripts/build_xmidi_manifest.py \
    --xmidi-root data/XMIDI_Dataset \
    --output-manifest manifests/lamd_xmidi.jsonl \
    --output-labels outputs/stage3/xmidi_labels.csv
```
Optional flags:
- `--sample-limit N` for smoke tests (default processes all 108k files)
- `--instrument strings` to override manifest instrument tag
- `--signature-mode path` for faster but non-content digests if storage speed matters

## Outputs
1. `manifests/lamd_xmidi.jsonl` – Lamda manifest rows with emotion metadata embedded in `meta`
2. `outputs/stage3/xmidi_labels.csv` – columns (`loop_id`,`emotion`,`genre`,`valence`,`arousal`,`clip_id`,`relative_path`)

`collect_conditions.py` can now ingest the labels file and fuse XMIDI affective tags with Stage2 summaries, captions, or embedding caches.

## Downstream impact
- **Arranger/EmotionAI**: every Lamda loop now has deterministic emotion coordinates; arranger filters or prompt builders can query by valence/arousal without bespoke parsing.
- **Lamda scalability**: manifest builder is dataset-agnostic; drop-in support for future corpora (e.g., Magenta e-gmd wavs) only requires a new config + CLI invocation.

## Next steps
1. Run the builder on the full `data/XMIDI_Dataset` once disk IO permits; stash the resulting manifest under version control or artifact storage.
2. Wire `outputs/stage3/xmidi_labels.csv` into your Stage3 condition build command to verify emotion/genre stats.
3. When you are ready to integrate Magenta e-gmd (90 GB), mirror this workflow: craft a mapping YAML, then extend the builder or author a sibling script that describes the drum-specific metadata (Continue/Drumify/etc.).
