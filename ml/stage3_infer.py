"""Stage3 inference script with constrained decoding.

Generate MIDI sequences from text/attribute prompts using trained Stage3 models.
Supports constraint-based decoding to enforce musical structure (bars, beats, time signatures).

Usage:

    PYTHONPATH=. python ml/stage3_infer.py \\
        --model outputs/stage3/models/stage3_gen_lora/model \\
        --tokenizer outputs/stage3/models/stage3_gen_lora/tokenizer_stage3.json \\
        --prompts configs/stage3/prompts_eval.yaml \\
        --out outputs/stage3/generated \\
        --num-samples 3 \\
        --temperature 0.9 \\
        --top-p 0.9
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import pretty_midi
import yaml

try:  # pragma: no cover - optional heavyweight dependencies
    import torch
except Exception:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore

try:  # pragma: no cover - optional heavyweight dependencies
    from transformers import GPT2LMHeadModel
except Exception:  # pragma: no cover - optional dependency guard
    GPT2LMHeadModel = None  # type: ignore


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class Stage3Tokenizer:
    """Minimal tokenizer loader for inference (must match training tokenizer)."""

    def __init__(self, vocab_path: Path) -> None:
        data = json.loads(vocab_path.read_text())
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["token_to_id"].items()}
        self.beat_division = data.get("beat_division", 24)
        self.max_time_shift = data.get("max_time_shift", 64)
        self.velocity_bins = data.get("velocity_bins", 16)
        self.max_duration = data.get("max_duration", 256)
        self.max_bars = data.get("max_bars", 16)

        self.pad_id = self.token_to_id.get("<pad>", 0)
        self.bos_id = self.token_to_id.get("<bos>", 1)
        self.eos_id = self.token_to_id.get("<eos>", 2)
        self.sep_id = self.token_to_id.get("<cond_end>", 3)
        self.bar_token_id = self.token_to_id.get("<BAR>")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @staticmethod
    def quantize(value: float, buckets: int = 10) -> int:
        """Quantize a value in [0, 1] to discrete bucket (floor-based, unbiased).
        
        Single source of truth for quantization - imported from stage3_generator.
        
        Args:
            value: Value in [0, 1] range
            buckets: Number of discrete buckets
        
        Returns:
            Bucket index in [0, buckets-1]
        """
        import math
        if math.isnan(value):
            return -1
        clamped = min(max(value, 0.0), 1.0)
        return min(int(clamped * buckets), buckets - 1)

    def encode_prompt(self, prompt: dict[str, object]) -> list[int]:
        """Encode a prompt dict into condition tokens."""
        tokens: list[int] = [self.bos_id]

        # Emotion
        emotion = prompt.get("emotion", "intense")
        tok = self.token_to_id.get(f"<emotion:{emotion}>")
        if tok is not None:
            tokens.append(tok)

        # Genre
        genre = prompt.get("genre", "pop")
        tok = self.token_to_id.get(f"<genre:{genre}>")
        if tok is not None:
            tokens.append(tok)

        # Valence/Arousal bins
        valence = prompt.get("valence")
        if valence is not None:
            bucket = int(min(10, max(0, round(float(valence) * 10))))
            tok = self.token_to_id.get(f"<valence:{bucket}>")
            if tok is not None:
                tokens.append(tok)

        arousal = prompt.get("arousal")
        if arousal is not None:
            bucket = int(min(10, max(0, round(float(arousal) * 10))))
            tok = self.token_to_id.get(f"<arousal:{bucket}>")
            if tok is not None:
                tokens.append(tok)

        # Score bucket
        score = prompt.get("score")
        if score is not None:
            bucket = int(min(9, max(0, round(float(score) / 10.0))))
            tok = self.token_to_id.get(f"<score:{bucket}>")
            if tok is not None:
                tokens.append(tok)

        # Caption tokens
        caption = str(prompt.get("caption", ""))
        for word in caption.split()[:8]:  # Limit caption words
            tok = self.token_to_id.get(f"<cap:{word}>")
            if tok is not None:
                tokens.append(tok)

        # Technique
        technique = prompt.get("technique")
        if technique:
            tok = self.token_to_id.get(f"<tech:{technique}>")
            if tok is not None:
                tokens.append(tok)

        # Audio similarity (CLAP/MERT) - use unified quantize method
        audio_clap = prompt.get("audio_clap")
        if audio_clap is not None:
            bucket = self.quantize(float(audio_clap), buckets=10)
            if bucket >= 0:
                tok = self.token_to_id.get(f"AUDIOCLAP_{bucket}")
                if tok is not None:
                    tokens.append(tok)

        audio_mert = prompt.get("audio_mert")
        if audio_mert is not None:
            bucket = self.quantize(float(audio_mert), buckets=10)
            if bucket >= 0:
                tok = self.token_to_id.get(f"AUDIOMERT_{bucket}")
                if tok is not None:
                    tokens.append(tok)

        # Separator
        tokens.append(self.sep_id)

        # Tempo
        tempo = prompt.get("tempo", 120)
        tempo_bucket = ((int(tempo) + 5) // 10) * 10
        tempo_bucket = max(40, min(240, tempo_bucket))
        tok = self.token_to_id.get(f"TEMPO_{tempo_bucket}")
        if tok is not None:
            tokens.append(tok)

        # Time signature
        tsig = prompt.get("time_signature", "4/4")
        tok = self.token_to_id.get(f"TSIG_{tsig}")
        if tok is not None:
            tokens.append(tok)

        return tokens


def build_forbidden_mask(
    tokenizer: Stage3Tokenizer,
    current_bar: int,
    max_bars: int,
    last_beat: int = 0,
    time_signature_beats: int = 4,
) -> set[int]:
    """Build set of forbidden token IDs based on music theory constraints.

    Args:
        tokenizer: Stage3 tokenizer instance
        current_bar: Current bar number (0-indexed)
        max_bars: Maximum allowed bars
        last_beat: Last generated BEAT number (1-4 for 4/4)
        time_signature_beats: Number of beats per bar (e.g., 4 for 4/4, 3 for 3/4)

    Returns:
        Set of token IDs to forbid
    """
    forbidden_ids = set()

    # Rule 1: BAR overflow prevention
    # If current_bar >= max_bars, forbid all BAR tokens >= max_bars
    if current_bar >= max_bars:
        for bar_num in range(max_bars, tokenizer.max_bars):
            bar_tok_name = f"BAR_{bar_num}"
            bar_tok_id = tokenizer.token_to_id.get(bar_tok_name)
            if bar_tok_id is not None:
                forbidden_ids.add(bar_tok_id)

    # Rule 2: BEAT order enforcement
    # If last_beat=2, forbid BEAT_1 (can't go backward)
    # If last_beat=4 (in 4/4), forbid BEAT_1-4, only allow new BAR
    if last_beat > 0:
        for beat_num in range(1, last_beat):
            beat_tok_name = f"BEAT_{beat_num}"
            beat_tok_id = tokenizer.token_to_id.get(beat_tok_name)
            if beat_tok_id is not None:
                forbidden_ids.add(beat_tok_id)

        # If last_beat is at max for time signature, forbid all BEAT tokens
        if last_beat >= time_signature_beats:
            for beat_num in range(1, time_signature_beats + 1):
                beat_tok_name = f"BEAT_{beat_num}"
                beat_tok_id = tokenizer.token_to_id.get(beat_tok_name)
                if beat_tok_id is not None:
                    forbidden_ids.add(beat_tok_id)

    return forbidden_ids


def generate_sequences(
    model: torch.nn.Module,
    tokenizer: Stage3Tokenizer,
    prompts: list[dict[str, object]],
    *,
    num_samples: int = 1,
    max_length: int = 2048,
    max_bars: int = 8,
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: int = 50,
    device: str = "cpu",
    enforce_bar_constraint: bool = True,
) -> list[list[int]]:
    """Generate token sequences from prompts with constraint sampling.

    Args:
        enforce_bar_constraint: Enforce max_bars limit
    """
    if torch is None:
        raise RuntimeError("torch is required for inference")

    model.eval()
    all_sequences: list[list[int]] = []

    with torch.no_grad():
        for prompt in prompts:
            logging.info("Generating for prompt: %s", prompt.get("name", "unnamed"))
            condition_ids = tokenizer.encode_prompt(prompt)

            for sample_idx in range(num_samples):
                input_ids = torch.tensor([condition_ids], dtype=torch.long, device=device)
                generated = input_ids.clone()

                current_bar = 0
                last_beat = 0
                time_sig_beats = 4  # Default 4/4, could parse from prompt
                bar_constraint_violated = False

                for step_idx in range(max_length - len(condition_ids)):
                    outputs = model(input_ids=generated)
                    logits = outputs.logits[:, -1, :] / temperature

                    # Music theory guards: build forbidden token mask
                    if enforce_bar_constraint:
                        forbidden_ids = build_forbidden_mask(
                            tokenizer=tokenizer,
                            current_bar=current_bar,
                            max_bars=max_bars,
                            last_beat=last_beat,
                            time_signature_beats=time_sig_beats,
                        )
                        for tok_id in forbidden_ids:
                            logits[:, tok_id] = float("-inf")

                    # Top-k filtering
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                        logits[indices_to_remove] = float("-inf")

                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[:, indices_to_remove] = float("-inf")

                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Check for EOS
                    if next_token.item() == tokenizer.eos_id:
                        break

                    # Track BAR/BEAT tokens for constraint state
                    next_token_str = tokenizer.id_to_token.get(next_token.item(), "")
                    if next_token_str.startswith("BAR_"):
                        try:
                            bar_num = int(next_token_str.split("_")[1])
                            current_bar = bar_num
                            last_beat = 0  # Reset beat on new bar
                            if current_bar >= max_bars:
                                logging.warning(
                                    "Bar constraint exceeded at step %d (bar %d)",
                                    step_idx,
                                    bar_num,
                                )
                                bar_constraint_violated = True
                                break
                        except (ValueError, IndexError):
                            pass
                    elif next_token_str.startswith("BEAT_"):
                        try:
                            beat_num = int(next_token_str.split("_")[1])
                            last_beat = beat_num
                        except (ValueError, IndexError):
                            pass

                    generated = torch.cat([generated, next_token], dim=-1)

                    # Early termination if constraint violated
                    if bar_constraint_violated:
                        # Append EOS
                        eos_tensor = torch.tensor(
                            [[tokenizer.eos_id]], dtype=torch.long, device=device
                        )
                        generated = torch.cat([generated, eos_tensor], dim=-1)
                        break

                all_sequences.append(generated[0].tolist())
                logging.info(
                    "  Sample %d: %d tokens (bars: %d)",
                    sample_idx + 1,
                    len(generated[0]),
                    current_bar + 1,
                )

    return all_sequences


def decode_to_midi(
    token_sequence: list[int],
    tokenizer: Stage3Tokenizer,
    output_path: Path,
    *,
    default_tempo: int = 120,
    default_tsig: tuple[int, int] = (4, 4),
) -> dict[str, object]:
    """Decode token sequence back to MIDI file with full structure parsing.

    Returns:
        Metadata dict with decode stats (notes, bars, warnings)
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)

    # Parse tokens and extract structure
    tempo_bpm = default_tempo
    time_sig_num, time_sig_denom = default_tsig
    current_bar = 0
    current_tick = 0
    current_inst_program = 0
    current_inst_is_drum = True

    # Track instruments by program
    instruments: dict[tuple[int, bool], pretty_midi.Instrument] = {}

    # Decode state
    ticks_per_beat = 480
    step_size = ticks_per_beat // tokenizer.beat_division
    beats_per_bar = time_sig_num
    ticks_per_bar = ticks_per_beat * beats_per_bar

    warnings_list: list[str] = []
    note_count = 0

    # Parse tokens
    i = 0
    while i < len(token_sequence):
        tok_id = token_sequence[i]
        tok_str = tokenizer.id_to_token.get(tok_id, "")

        # Skip control tokens
        if tok_str in ["<bos>", "<eos>", "<pad>", "<cond_end>"] or tok_str.startswith("<"):
            i += 1
            continue

        # Parse tempo
        if tok_str.startswith("TEMPO_"):
            try:
                tempo_bpm = int(tok_str.split("_")[1])
                pm.initial_tempo = tempo_bpm
            except (ValueError, IndexError):
                warnings_list.append(f"Invalid tempo token: {tok_str}")
            i += 1
            continue

        # Parse time signature
        if tok_str.startswith("TSIG_"):
            try:
                tsig_part = tok_str.split("_")[1]
                num, denom = tsig_part.split("/")
                time_sig_num = int(num)
                time_sig_denom = int(denom)
                beats_per_bar = time_sig_num
                ticks_per_bar = ticks_per_beat * beats_per_bar
            except (ValueError, IndexError):
                warnings_list.append(f"Invalid time sig token: {tok_str}")
            i += 1
            continue

        # Parse BAR boundary
        if tok_str.startswith("BAR_"):
            try:
                bar_num = int(tok_str.split("_")[1])
                current_bar = bar_num
                current_tick = current_bar * ticks_per_bar
            except (ValueError, IndexError):
                warnings_list.append(f"Invalid bar token: {tok_str}")
            i += 1
            continue

        # Parse TIME shift
        if tok_str.startswith("TIME_"):
            try:
                shift = int(tok_str.split("_")[1])
                current_tick += shift * step_size
            except (ValueError, IndexError):
                warnings_list.append(f"Invalid time token: {tok_str}")
            i += 1
            continue

        # Parse instrument change
        if tok_str.startswith("DRUM_") or tok_str.startswith("INST_"):
            try:
                is_drum = tok_str.startswith("DRUM_")
                program = int(tok_str.split("_")[1])
                current_inst_program = program
                current_inst_is_drum = is_drum

                # Create instrument if not exists
                inst_key = (program, is_drum)
                if inst_key not in instruments:
                    instruments[inst_key] = pretty_midi.Instrument(program=program, is_drum=is_drum)
            except (ValueError, IndexError):
                warnings_list.append(f"Invalid inst token: {tok_str}")
            i += 1
            continue

        # Parse NOTE event (NOTE + VEL + DUR sequence expected)
        if tok_str.startswith("NOTE_"):
            try:
                pitch = int(tok_str.split("_")[1])

                # Get velocity (next token)
                if i + 1 >= len(token_sequence):
                    warnings_list.append(f"NOTE without VEL at token {i}")
                    i += 1
                    continue
                vel_tok = tokenizer.id_to_token.get(token_sequence[i + 1], "")
                if not vel_tok.startswith("VEL_"):
                    warnings_list.append(f"NOTE not followed by VEL at token {i}")
                    i += 1
                    continue
                vel_bin = int(vel_tok.split("_")[1])
                velocity = int((vel_bin + 0.5) * (127 / tokenizer.velocity_bins))
                velocity = max(1, min(127, velocity))

                # Get duration (next next token)
                if i + 2 >= len(token_sequence):
                    warnings_list.append(f"NOTE without DUR at token {i}")
                    i += 2
                    continue
                dur_tok = tokenizer.id_to_token.get(token_sequence[i + 2], "")
                if not dur_tok.startswith("DUR_"):
                    warnings_list.append(f"NOTE not followed by DUR at token {i + 2}")
                    i += 2
                    continue
                dur_steps = int(dur_tok.split("_")[1])
                duration_ticks = dur_steps * step_size

                # Create note
                start_time = pm.tick_to_time(current_tick)
                end_time = pm.tick_to_time(current_tick + duration_ticks)
                note = pretty_midi.Note(
                    velocity=velocity, pitch=pitch, start=start_time, end=end_time
                )

                # Add to current instrument
                inst_key = (current_inst_program, current_inst_is_drum)
                if inst_key not in instruments:
                    instruments[inst_key] = pretty_midi.Instrument(
                        program=current_inst_program, is_drum=current_inst_is_drum
                    )
                instruments[inst_key].notes.append(note)
                note_count += 1

                i += 3  # Skip NOTE, VEL, DUR
                continue
            except (ValueError, IndexError) as e:
                warnings_list.append(f"Note parsing error at token {i}: {e}")
                i += 1
                continue

        # Unknown token
        i += 1

    # Add all instruments to MIDI
    for inst in instruments.values():
        if inst.notes:  # Only add if has notes
            pm.instruments.append(inst)

    # Write MIDI file
    pm.write(str(output_path))

    return {
        "note_count": note_count,
        "bar_count": current_bar + 1,
        "tempo": tempo_bpm,
        "time_signature": f"{time_sig_num}/{time_sig_denom}",
        "warnings": warnings_list,
        "instrument_count": len(instruments),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage3 inference with constrained decoding")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to tokenizer JSON")
    parser.add_argument("--prompts", type=Path, required=True, help="YAML file with prompts")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling threshold")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if torch is None or GPT2LMHeadModel is None:
        raise RuntimeError("Install torch and transformers for inference")

    # Load tokenizer
    logging.info("Loading tokenizer from %s", args.tokenizer)
    tokenizer = Stage3Tokenizer(args.tokenizer)

    # Load model
    logging.info("Loading model from %s", args.model)
    model = GPT2LMHeadModel.from_pretrained(str(args.model))
    model.to(args.device)

    # Load prompts
    logging.info("Loading prompts from %s", args.prompts)
    with args.prompts.open() as f:
        prompts_data = yaml.safe_load(f)
    prompts = prompts_data.get("prompts", [])

    if not prompts:
        logging.warning("No prompts found in %s", args.prompts)
        return

    # Generate
    logging.info("Generating %d samples per prompt for %d prompts", args.num_samples, len(prompts))
    sequences = generate_sequences(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples=args.num_samples,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
    )

    # Save outputs
    args.out.mkdir(parents=True, exist_ok=True)
    midi_dir = args.out / "midi"
    midi_dir.mkdir(exist_ok=True)

    for idx, seq in enumerate(sequences):
        midi_path = midi_dir / f"generated_{idx:04d}.mid"
        decode_to_midi(seq, tokenizer, midi_path)
        logging.info("Saved MIDI: %s", midi_path)

    # Save metadata
    metadata_path = args.out / "generation_metadata.json"
    metadata = {
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "prompts": str(args.prompts),
        "num_prompts": len(prompts),
        "num_samples": args.num_samples,
        "total_generated": len(sequences),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logging.info("Generation complete. Metadata saved to %s", metadata_path)


if __name__ == "__main__":  # pragma: no cover
    main()
