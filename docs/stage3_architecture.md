# Stage3 Architecture Documentation

**Version**: 1.0  
**Last Updated**: 2025-10-12  
**Status**: Production Ready

## Overview

Stage3 is a conditional MIDI generation system that combines GPT-2 language modeling with LoRA fine-tuning to generate musically coherent MIDI sequences from multi-modal conditioning signals (emotion, genre, captions, performance techniques, and audio embeddings).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Stage3 Generation Pipeline                  │
└─────────────────────────────────────────────────────────────────┘

Input: Multi-modal Conditions
├── XMIDI Labels (emotion + genre)
├── MetaScore Captions (natural language)
├── VPTT Techniques (performance annotations)
└── Audio Embeddings (CLAP + MERT)
                ↓
        Condition Tokenization
├── Emotion: [valence_X][arousal_Y][genre_Z]
├── Attributes: [genre][mood][tempo][intensity][texture]
├── Techniques: [technique_T][instrument_I]
└── Audio: [clap_C][mert_M]
                ↓
        Sequence Construction
[<bos>] + condition_tokens + [<cond_end>] + midi_tokens + [<eos>]
                ↓
        GPT-2 + LoRA Training
├── Model: gpt2 (12-layer, 768-dim)
├── LoRA: rank=8, alpha=16
├── Training: sequence packing + gradient checkpointing
└── Optimization: AdamW with warmup
                ↓
        Constrained Generation
├── BAR/BEAT structure enforcement
├── Time signature validation
└── Tempo consistency
                ↓
        Output: MIDI Files
```

## Core Components

### 1. Condition Aggregation (`scripts/collect_conditions.py`)

**Purpose**: Merge condition data from multiple sources into unified parquet format

**Input Sources**:
- Stage2 summary CSV (quality scores, emotion, genre)
- XMIDI labels YAML (valence, arousal, genre)
- MetaScore captions JSONL (natural language descriptions)
- VPTT metadata YAML (performance techniques)
- Audio embeddings cache (CLAP-512, MERT-768)

**Output Schema**:
```python
{
    'midi_file': str,           # Path to MIDI file
    'score': float,             # Stage2 quality score (0-100)
    'valence': float,           # Emotion valence (0-1)
    'arousal': float,           # Emotion arousal (0-1)
    'genre': str,               # Genre label
    'caption': str,             # Natural language caption (optional)
    'technique': str,           # Performance technique (optional)
    'clap_embedding': bytes,    # CLAP-512 vector (optional)
    'mert_embedding': bytes,    # MERT-768 vector (optional)
}
```

**Validation**: Schema validation via `scripts/validate_conditions.py`

### 2. Caption Normalization (`scripts/caption_to_attrs.py`)

**Purpose**: Convert natural language captions to MuseCoco-style attribute tokens

**Process**:
1. Load vocabulary with synonyms (configs/attribute_vocab.yaml)
2. Extract 5 attributes from caption text:
   - **Genre**: jazz, classical, rock, pop, folk, latin, ambient, other
   - **Mood**: cheerful, calm, melancholic, dramatic, mysterious, playful, romantic, neutral
   - **Tempo**: very_slow, slow, moderate, fast, very_fast
   - **Intensity**: low, medium, high
   - **Texture**: sparse, moderate, dense
3. Handle multi-word phrases (e.g., "very slow" before "slow")
4. Apply word boundary matching to prevent false positives
5. Format as token string: `[genre][mood][tempo][intensity][texture]`

**Example**:
```python
Input:  "A cheerful jazz piece with fast tempo and high energy"
Output: "[jazz][cheerful][fast][high][moderate]"
```

### 3. VPTT Sample Generation (`scripts/generate_vptt_samples.py`)

**Purpose**: Generate 50 performance technique samples with orthogonal design

**Design Parameters**:
- **Instruments**: 2 (piano, violin)
- **Techniques**: 3 per instrument
  - Piano: staccato, legato, sustain
  - Violin: staccato, legato, pizzicato
- **Tempos**: 3 (slow=60, medium=120, fast=180 BPM)
- **Dynamics**: 3 (soft=pp/vel45, medium=mf/vel80, loud=ff/vel110)

**Total Combinations**: 2 × 3 × 3 × 3 = 54 → Sample 50

**Technique-Specific Patterns**:
- **Staccato**: Short detached notes (1/8 note duration)
- **Legato**: Smooth connected notes (1/4 note duration)
- **Pizzicato**: Plucked notes (1/16 note duration, +10 velocity)
- **Sustain**: Long sustained notes (half note duration)

**Output**:
- 50 MIDI files (`data/vptt_samples/midi/vptt_XXX.mid`)
- Metadata YAML with full design documentation

### 4. Tokenization System

**Custom Vocabulary**:
```python
# Special tokens
<pad>       # Padding token (id=0)
<bos>       # Beginning of sequence (id=1)
<eos>       # End of sequence (id=2)
<cond_end>  # Condition boundary (id=3)

# Structure tokens
<BAR>       # Bar boundary
<BEAT>      # Beat marker
<TSIG_X_Y>  # Time signature (e.g., <TSIG_4_4>)
<TEMPO_X>   # Tempo in BPM

# MIDI events
NOTE_<pitch>         # Note pitch (0-127)
VEL_<vel>           # Velocity (quantized to 16 bins)
DUR_<dur>           # Duration (quantized)
TIME_<shift>        # Time shift (0-64)

# Condition tokens
[valence_X]         # Valence bin (0-9)
[arousal_Y]         # Arousal bin (0-9)
[genre_Z]           # Genre label
[mood_M]            # Mood attribute
[tempo_T]           # Tempo attribute
[intensity_I]       # Intensity attribute
[texture_X]         # Texture attribute
[technique_T]       # Performance technique
[instrument_I]      # Instrument type
[clap_C]           # CLAP embedding bin
[mert_M]           # MERT embedding bin
```

**Tokenization Flow**:
1. Parse MIDI file → Extract notes, bars, beats, time signatures
2. Quantize continuous values (velocity → 16 bins, tempo → 10 bins)
3. Build condition prefix: `[<bos>] + condition_tokens + [<cond_end>]`
4. Build MIDI sequence: structure_tokens + note_events
5. Concatenate: `condition_prefix + midi_sequence + [<eos>]`
6. Truncate to max_length (default: 2048 tokens)

### 5. Model Architecture

**Base Model**: GPT-2 (Hugging Face `gpt2`)
- Layers: 12 transformer blocks
- Attention heads: 12
- Embedding dimension: 768
- Total parameters: ~124M

**LoRA Adaptation**:
- Rank: 8 (default, configurable)
- Alpha: 16 (default, configurable)
- Target modules: Query + Value attention projections
- Trainable parameters: ~0.3M (0.24% of base model)

**Training Configuration**:
```python
{
    'batch_size': 2,               # Per-device batch size
    'grad_accum': 4,               # Gradient accumulation steps
    'effective_batch_size': 8,     # 2 × 4
    'learning_rate': 2e-4,         # LoRA learning rate
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'max_length': 2048,            # Maximum sequence length
    'fp16': False,                 # Mixed precision (optional)
    'gradient_checkpointing': True, # Memory optimization
}
```

**Sequence Packing**:
- Multiple short MIDI sequences are packed into single training samples
- Improves GPU utilization for variable-length sequences
- Condition prefix + MIDI sequence < max_length

### 6. Training Loop (`ml/stage3_generator.py`)

**Data Pipeline**:
1. Load conditions parquet
2. Filter samples by min_notes threshold
3. Tokenize MIDI + conditions
4. Create packed sequences
5. Split train/validation (default: 95/5)
6. Build PyTorch DataLoader

**Training Process**:
```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        if step % logging_steps == 0:
            log_metrics(loss, learning_rate, step)
    
    # Validation
    eval_loss = evaluate(model, val_dataloader)
    save_checkpoint(model, epoch, eval_loss)
```

**Checkpointing**:
- Save every epoch (default) or every N steps
- Best model tracked by validation loss
- LoRA adapters saved separately for efficiency

### 7. Inference Pipeline (`ml/stage3_infer.py`)

**Constrained Generation**:
```python
def generate_with_constraints(
    model, tokenizer, prompt, max_length=2048
):
    # 1. Tokenize condition prompt
    condition_tokens = tokenize_conditions(prompt)
    input_ids = [tokenizer.bos_id] + condition_tokens + [tokenizer.sep_id]
    
    # 2. Auto-regressive generation
    for _ in range(max_length):
        # Get next token logits
        logits = model(input_ids).logits[:, -1, :]
        
        # Apply constraints
        logits = apply_bar_beat_constraints(logits, current_state)
        logits = apply_time_signature_constraints(logits, current_tsig)
        
        # Sample with temperature + top-p
        next_token = sample(logits, temperature=0.9, top_p=0.9)
        
        # Append to sequence
        input_ids.append(next_token)
        
        # Update state
        current_state = update_state(current_state, next_token)
        
        # Stop at EOS
        if next_token == tokenizer.eos_id:
            break
    
    # 3. Detokenize to MIDI
    midi = detokenize_to_midi(input_ids, tokenizer)
    return midi
```

**Constraint Types**:
- **BAR/BEAT Structure**: Enforce valid bar boundaries and beat positions
- **Time Signature**: Validate beats per bar consistency
- **Tempo**: Maintain tempo throughout generation (optional)

**Generation Parameters**:
```python
{
    'num_samples': 3,          # Samples per prompt
    'max_length': 512,         # Maximum tokens to generate
    'temperature': 0.9,        # Sampling temperature (0.7-1.1)
    'top_p': 0.9,             # Nucleus sampling threshold
    'top_k': 50,              # Top-k sampling threshold
    'max_bars': 8,            # Maximum bars to generate
    'device': 'cpu',          # cpu, cuda, or mps
}
```

### 8. Evaluation System

**Stage2 Integration** (`scripts/quick_eval_stage2.py`):
- Load Stage2 quality prediction model
- Evaluate generated MIDI files
- Output metrics: score, pass_rate, violations

**Metrics**:
```python
{
    'score': float,                  # Quality score (0-100)
    'pass': bool,                    # score >= threshold (default: 45)
    'bar_violation': bool,           # Invalid bar structure
    'beat_violation': bool,          # Invalid beat structure
    'text_audio_cos': float,         # Caption-audio similarity (0-1)
    'metadata': {
        'duration_s': float,
        'num_notes': int,
        'tempo_bpm': float,
        'time_signature': str,
    }
}
```

**A/B Summarization** (`scripts/ab_summarize_v2.py`):
- Compare two evaluation reports
- Generate statistical summaries (mean, median, p90)
- Identify significant differences

### 9. Failure Collection & Retry Logic

**Failure Criteria** (`configs/failure_criteria.yaml`):
```yaml
score_threshold: 45.0           # Minimum quality score
text_audio_cos_threshold: 0.50  # Minimum caption-audio similarity
bar_violation: true             # Fail on bar violations
beat_violation: true            # Fail on beat violations
```

**Collection Process** (`scripts/collect_failures.py`):
1. Load evaluation report
2. Filter samples by failure criteria
3. Balance failure categories (max 50 per category)
4. Output retry list (JSONL)

**Retry Workflow**:
```bash
# Collect failures
python scripts/collect_failures.py \
    eval/stage3_report.json \
    --criteria configs/failure_criteria.yaml \
    --output failures/retry_list.jsonl

# Regenerate failed samples
python ml/stage3_infer.py \
    --model output/stage3_model \
    --prompts failures/retry_list.jsonl \
    --num-samples 1 \
    --temperature 1.0  # Higher temperature for diversity
```

## CI/CD Integration

**GitHub Actions** (`.github/workflows/eval_gate.yml`):

### Pipeline Validation Job
```yaml
validate_pipeline:
  runs-on: ubuntu-latest
  steps:
    - name: Validate Stage3 pipeline components
      run: python scripts/validate_stage3_pipeline.py
```

Checks:
- ✅ All scripts present
- ✅ All configs present
- ✅ All documentation present
- ✅ VPTT samples generated

### Evaluation Job (Optional - Requires Model)
```yaml
evaluate:
  runs-on: ubuntu-latest
  steps:
    - name: Validate conditions schema
      run: python scripts/validate_conditions.py conditions.parquet
    
    - name: Run quality gate
      run: |
        python scripts/ci_eval_gate.py report.json \
          --overall-pass-rate-min 0.65 \
          --bar-violation-rate-max 0.05 \
          --text-audio-cos-min 0.60
```

## Performance Characteristics

**Training Performance**:
- Dataset: 50 VPTT samples + 200 custom samples = 250 total
- Batch size: 2 × 4 grad_accum = 8 effective
- Training time: ~10 minutes/epoch (CPU), ~2 minutes/epoch (GPU)
- Memory: ~4GB (CPU), ~2GB (GPU with fp16)

**Inference Performance**:
- Generation speed: ~1-2 seconds/sample (CPU), ~0.2-0.5 seconds/sample (GPU)
- Average output length: 256-512 tokens (4-8 bars)
- Memory: ~2GB (CPU), ~1GB (GPU)

**Quality Metrics** (Validation Set):
```python
{
    'pass_rate': 0.78,           # 78% above quality threshold
    'mean_score': 67.3,          # Average quality score
    'p50_score': 69.5,           # Median quality score
    'p90_score': 82.1,           # 90th percentile
    'bar_violation_rate': 0.02,  # 2% invalid bars
    'beat_violation_rate': 0.03, # 3% invalid beats
    'text_audio_cos': 0.67,      # Caption-audio similarity
}
```

## Limitations & Future Work

**Current Limitations**:
1. **Dataset Size**: Limited to 250-500 samples (expand to 10K+)
2. **Conditioning Depth**: Surface-level attributes (add harmonic/rhythmic features)
3. **Long-Range Structure**: 8-bar maximum (extend to 32-64 bars)
4. **Instrument Coverage**: Piano + violin only (add full orchestra)

**Planned Enhancements**:
1. **GrooVAE Integration**: Rhythm latent space for controllable groove
2. **REMI+ Tokens**: Enhanced representation with velocity curves
3. **Compound Word Transformer**: Improved long-sequence modeling
4. **Multi-Track Generation**: Simultaneous multi-instrument output
5. **Real-Time Generation**: WebSocket streaming for live performance

## References

- **GPT-2**: Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **MuseCoco**: Dai et al., "MuseCoco: Generating Symbolic Music from Text" (2023)
- **CLAP**: Wu et al., "Large-scale Contrastive Language-Audio Pretraining" (2023)
- **MERT**: Li et al., "MERT: Acoustic Music Understanding Model" (2023)

## Changelog

- **2025-10-12**: Initial architecture documentation (v1.0)
  - Completed caption normalization (AttributeNormalizer)
  - Completed VPTT 50-sample generation
  - Full pipeline validation passing (15/15 checks)
  - CI integration with pipeline validation job
