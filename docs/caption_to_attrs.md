# Caption to Attributes Normalizer

Converts natural language music captions to structured MuseCoco-style attribute tokens.

## Output Format

```
[genre][mood][tempo][intensity][texture]
```

## Usage

### Basic Usage

```bash
python scripts/caption_to_attrs.py \
    --input data/metascore_captions.jsonl \
    --output data/metascore_attributes.jsonl
```

### With Custom Vocabulary

```bash
python scripts/caption_to_attrs.py \
    --input data/metascore_captions.jsonl \
    --output data/metascore_attributes.jsonl \
    --vocab configs/attribute_vocab.yaml
```

### With Validation

```bash
python scripts/caption_to_attrs.py \
    --input data/metascore_captions.jsonl \
    --output data/metascore_attributes.jsonl \
    --validate \
    --verbose
```

## Input Format

JSONL file with entries:
```json
{"loop_id": "drum_001", "caption": "A cheerful jazz piano piece with fast tempo"}
{"loop_id": "drum_002", "caption": "A melancholic orchestral movement"}
```

## Output Format

JSONL file with entries:
```json
{
  "loop_id": "drum_001",
  "caption": "A cheerful jazz piano piece with fast tempo",
  "attributes": {
    "genre": "jazz",
    "mood": "cheerful",
    "tempo": "fast",
    "intensity": "unknown",
    "texture": "unknown"
  },
  "tokens": "[jazz][cheerful][fast][unknown][unknown]"
}
```

## Supported Attributes

### Genre
- `jazz` - jazz, swing, bebop, blues
- `classical` - classical, orchestral, symphony, baroque, romantic
- `rock` - rock, metal, punk, alternative
- `pop` - pop, dance, disco, electronic, edm
- `folk` - folk, country, bluegrass, acoustic
- `latin` - latin, salsa, bossa nova, tango
- `ambient` - ambient, atmospheric, soundscape, drone
- `other` - experimental, avant-garde, fusion

### Mood
- `cheerful` - cheerful, happy, joyful, upbeat, bright
- `calm` - calm, peaceful, serene, tranquil, relaxing
- `melancholic` - melancholic, sad, sorrowful, nostalgic
- `dramatic` - dramatic, intense, powerful, epic, grand
- `mysterious` - mysterious, enigmatic, dark, eerie
- `playful` - playful, whimsical, light, fun, bouncy
- `romantic` - romantic, tender, loving, sweet
- `neutral` - neutral, moderate, balanced

### Tempo
- `very_slow` - very slow, grave, largo, adagio
- `slow` - slow, lento, andante
- `moderate` - moderate, moderato, allegretto
- `fast` - fast, allegro, vivace, upbeat
- `very_fast` - very fast, presto, prestissimo, rapid

### Intensity
- `low` - soft, quiet, gentle, subtle
- `medium` - medium, moderate, balanced
- `high` - loud, strong, powerful, intense

### Texture
- `sparse` - sparse, minimal, simple, thin
- `moderate` - moderate, balanced, medium
- `dense` - dense, complex, rich, thick, layered

## Custom Vocabulary

Create a YAML file with custom synonym mappings:

```yaml
genre:
  jazz:
    - jazz
    - swing
    - bebop
  # ... more genres

mood:
  cheerful:
    - cheerful
    - happy
    - joyful
  # ... more moods

tempo:
  fast:
    - fast
    - allegro
    - vivace
  # ... more tempos

intensity:
  high:
    - loud
    - powerful
  # ... more intensities

texture:
  dense:
    - dense
    - rich
    - layered
  # ... more textures
```

## Examples

### Example 1: Jazz
```
Input:  "A cheerful jazz piano piece with fast tempo"
Output: [jazz][cheerful][fast][unknown][unknown]
```

### Example 2: Classical
```
Input:  "A melancholic orchestral movement with slow tempo and dense texture"
Output: [classical][melancholic][slow][unknown][dense]
```

### Example 3: Rock
```
Input:  "A powerful rock anthem with very fast tempo and high intensity"
Output: [rock][neutral][very_fast][high][unknown]
```

### Example 4: Ambient
```
Input:  "A calm atmospheric soundscape with very slow tempo and sparse minimal texture"
Output: [ambient][calm][very_slow][unknown][sparse]
```

## Testing

Run the test suite:
```bash
python -m pytest tests/test_caption_to_attrs.py -v
```

## Integration with Stage3

This script is part of the Stage3 condition aggregation pipeline:

1. **Collect raw captions** from MetaScore dataset
2. **Normalize to attributes** using this script
3. **Tokenize for GPT-2** in Stage3 generator
4. **Condition generation** on attribute tokens

See `scripts/collect_conditions.py` for the full pipeline.
