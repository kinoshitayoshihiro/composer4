"""Stage3 Generator training script.

This script prepares a condition-aware MIDI token dataset powered by Stage2 metrics
and Stage3 annotations (XMIDI emotion/genre, MetaScore captions, VioPTT techniques),
and trains a conditional autoregressive model based on GPT-2.

Usage example:

    PYTHONPATH=. ./.venv311/bin/python ml/stage3_generator.py \
        --metadata outputs/stage3/loop_summary_with_captions.csv \
        --midi-root output/drumloops_cleaned/2 \
        --technique-meta outputs/stage3/technique_synth/technique_metadata.jsonl \
        --out outputs/stage3/models/stage3_generator \
        --max-samples 128

For a quick schema validation without training, add ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
import pretty_midi

try:  # pragma: no cover - optional heavyweight dependencies
    import torch
    from torch.utils.data import Dataset, random_split
except Exception:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore
    Dataset = object  # type: ignore

try:  # pragma: no cover - optional heavyweight dependencies
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
except Exception:  # pragma: no cover - optional dependency guard
    GPT2Config = None  # type: ignore
    GPT2LMHeadModel = None  # type: ignore
    Trainer = None  # type: ignore
    TrainingArguments = None  # type: ignore

try:  # pragma: no cover - optional
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover - optional dependency guard
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore
    get_peft_model = None  # type: ignore

RE_CAPTION_TOKEN = re.compile(
    r"[一-龥々〆ヵヶぁ-んァ-ヴー]+|[A-Za-z]+|\d+|[\-+#&]+|[？！。、,.!?；;:〜~()（）『』・]+"
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    """Initialise logging format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def tokenize_caption(text: str, *, limit: int = 16) -> list[str]:
    """Tokenise Japanese/English captions into subwords.

    Falls back to character-level segmentation if no regex match is found.
    Duplicate tokens are removed while preserving order.
    """

    if not text:
        return []
    tokens = RE_CAPTION_TOKEN.findall(text)
    if not tokens:
        tokens = list(text.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for tok in tokens:
        if tok and tok not in seen:
            seen.add(tok)
            unique.append(tok)
        if len(unique) >= limit:
            break
    return unique


def quantize(value: float, *, buckets: int) -> int:
    """Quantize a floating point value in ``[0, 1]`` to discrete bucket index.

    Uses floor-based quantization to ensure monotonic mapping without banker's rounding bias.
    Examples (buckets=10):
        0.0 -> 0, 0.15 -> 1, 0.5 -> 5, 0.85 -> 8, 1.0 -> 9
    """
    if math.isnan(value):
        return -1
    clamped = min(max(value, 0.0), 1.0)
    return min(int(clamped * buckets), buckets - 1)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


class Stage3Tokenizer:
    """Dynamic tokenizer for Stage3 conditional MIDI sequences.

    Enhanced with musical structure tokens (BAR, BEAT, TSIG, TEMPO)
    for improved generation coherence and tempo-independent quantization.
    """

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    sep_token = "<cond_end>"
    bar_token = "<BAR>"
    beat_token = "<BEAT>"

    def __init__(
        self,
        *,
        beat_division: int = 24,
        max_time_shift: int = 64,
        velocity_bins: int = 16,
        max_duration: int = 256,
        max_bars: int = 16,
        audio_bins: int = 10,
    ) -> None:
        self.beat_division = beat_division
        self.max_time_shift = max_time_shift
        self.velocity_bins = velocity_bins
        self.max_duration = max_duration
        self.max_bars = max_bars
        self.audio_bins = audio_bins

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._init_base_vocab()

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    def _init_base_vocab(self) -> None:
        # Control tokens
        for tok in [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.bar_token,
            self.beat_token,
        ]:
            self._add_token(tok)

        # Time signature tokens (common meters)
        for num in [2, 3, 4, 5, 6, 7, 9, 12]:
            for denom in [4, 8, 16]:
                self._add_token(f"TSIG_{num}/{denom}")

        # Tempo tokens (BPM in 10s: 40-240)
        for bpm in range(40, 250, 10):
            self._add_token(f"TEMPO_{bpm}")

        # Bar position tokens (0..max_bars-1)
        for bar_num in range(self.max_bars):
            self._add_token(f"BAR_{bar_num}")

        # Audio embedding similarity tokens (CLAP/MERT)
        for i in range(self.audio_bins):
            self._add_token(f"AUDIOCLAP_{i}")
            self._add_token(f"AUDIOMERT_{i}")

        # Notes 0-127
        for note in range(128):
            self._add_token(f"NOTE_{note}")

        # Velocity bins
        for i in range(self.velocity_bins):
            self._add_token(f"VEL_{i}")

        # Duration bins (1..max_duration)
        for dur in range(1, self.max_duration + 1):
            self._add_token(f"DUR_{dur}")

        # Time shift bins (1..max_time_shift) - tempo-independent ticks
        for shift in range(1, self.max_time_shift + 1):
            self._add_token(f"TIME_{shift}")

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    # Exposed helpers -------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.sep_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def ensure_condition_token(self, token: str) -> int:
        return self._add_token(token)

    def ensure_instrument_token(self, program: int, is_drum: bool) -> int:
        prefix = "DRUM" if is_drum else "INST"
        return self._add_token(f"{prefix}_{program}")

    def time_shift_token(self, shift: int) -> int:
        shift = max(1, min(shift, self.max_time_shift))
        return self.token_to_id[f"TIME_{shift}"]

    def duration_token(self, duration: int) -> int:
        duration = max(1, min(duration, self.max_duration))
        return self.token_to_id[f"DUR_{duration}"]

    def velocity_token(self, velocity: int) -> int:
        velocity = max(1, min(velocity, 127))
        bin_size = math.ceil(128 / self.velocity_bins)
        bucket = min((velocity - 1) // bin_size, self.velocity_bins - 1)
        return self.token_to_id[f"VEL_{bucket}"]

    def note_token(self, pitch: int) -> int:
        pitch = max(0, min(127, pitch))
        return self.token_to_id[f"NOTE_{pitch}"]

    # MIDI encoding ---------------------------------------------------
    def encode_midi(self, midi: pretty_midi.PrettyMIDI) -> list[int]:
        """Encode a PrettyMIDI object into token ids."""

        tokens: list[int] = []
        events: list[tuple[int, int, int, int, int]] = []
        ticks_per_beat = midi.resolution
        step = max(1, ticks_per_beat // self.beat_division)

        for inst in midi.instruments:
            inst_token_id = self.ensure_instrument_token(inst.program, inst.is_drum)
            for note in inst.notes:
                start_tick = int(round(midi.time_to_tick(note.start)))
                end_tick = int(round(midi.time_to_tick(note.end)))
                if end_tick <= start_tick:
                    end_tick = start_tick + step
                events.append((start_tick, end_tick, note.pitch, note.velocity, inst_token_id))

        if not events:
            return tokens

        events.sort(key=lambda x: (x[0], x[2], x[3]))
        last_tick = 0
        current_inst = None

        for start_tick, end_tick, pitch, velocity, inst_tok in events:
            delta = start_tick - last_tick
            while delta > 0:
                shift = min(self.max_time_shift, max(1, delta // step))
                tokens.append(self.time_shift_token(shift))
                delta -= shift * step
            if inst_tok != current_inst:
                tokens.append(inst_tok)
                current_inst = inst_tok
            tokens.append(self.note_token(pitch))
            tokens.append(self.velocity_token(velocity))
            duration_steps = max(1, (end_tick - start_tick) // step)
            tokens.append(self.duration_token(duration_steps))
            last_tick = start_tick

        return tokens

    # Persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        data = {
            "token_to_id": self.token_to_id,
            "beat_division": self.beat_division,
            "max_time_shift": self.max_time_shift,
            "velocity_bins": self.velocity_bins,
            "max_duration": self.max_duration,
            "max_bars": self.max_bars,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class EncodedSample:
    tokens: list[int]
    source_path: Path
    conditions: list[str]


class Stage3Dataset(Dataset):  # type: ignore[misc]
    """Dataset producing Stage3 conditional token sequences."""

    def __init__(
        self,
        metadata_path: Path,
        midi_root: Path,
        tokenizer: Stage3Tokenizer,
        *,
        technique_metadata: Path | None = None,
        max_seq_len: int = 2048,
        max_caption_tokens: int = 16,
        min_notes: int = 4,
        max_samples: int | None = None,
        min_length: int = 20,
        max_length: int = 2048,
        genre_weights: dict[str, float] | None = None,
        tempo_bins: int = 5,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required to build Stage3Dataset")

        self.metadata_path = metadata_path
        self.midi_root = midi_root
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_caption_tokens = max_caption_tokens
        self.min_notes = min_notes
        self.max_samples = max_samples
        self.min_length = min_length
        self.max_length = max_length
        self.genre_weights = genre_weights or {}
        self.tempo_bins = tempo_bins

        self.samples: list[EncodedSample] = []
        self.condition_counter: Counter[str] = Counter()
        self.lengths: list[int] = []
        self.skipped_files: list[str] = []
        self.sample_weights: list[float] = []

        self.techniques_by_digest = self._load_technique_metadata(technique_metadata)
        self._load_metadata()
        self._compute_sample_weights()

    # Internal helpers ------------------------------------------------
    @staticmethod
    def _load_technique_metadata(path: Path | None) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = defaultdict(set)
        if path is None or not path.exists():
            return mapping
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skip invalid technique metadata line: %s", line[:120])
                continue
            digest = Path(obj.get("input_file", "")).stem
            tech = obj.get("technique")
            if digest and tech:
                mapping[digest].add(str(tech))
        return mapping

    def _load_metadata(self) -> None:
        logging.info("Loading metadata from %s", self.metadata_path)
        df = pd.read_csv(self.metadata_path)
        if self.max_samples is not None:
            df = df.head(self.max_samples)

        for idx, row in df.iterrows():
            midi_path = self.midi_root / str(row.get("filename", ""))
            if not midi_path.exists():
                # Try digest fallback
                digest = str(row.get("file_digest", "")) + ".mid"
                alt_path = self.midi_root / digest
                if alt_path.exists():
                    midi_path = alt_path
                else:
                    self.skipped_files.append(str(midi_path))
                    logging.debug("Skip missing MIDI: %s", midi_path)
                    continue

            try:
                midi = pretty_midi.PrettyMIDI(str(midi_path))
            except Exception as exc:  # pragma: no cover - depends on data
                logging.warning("Failed to read %s: %s", midi_path, exc)
                self.skipped_files.append(str(midi_path))
                continue

            if sum(len(inst.notes) for inst in midi.instruments) < self.min_notes:
                self.skipped_files.append(str(midi_path))
                continue

            condition_tokens = self._build_condition_tokens(row)
            condition_ids = [self.tokenizer.ensure_condition_token(tok) for tok in condition_tokens]
            for tok in condition_tokens:
                self.condition_counter[tok] += 1

            event_ids = self.tokenizer.encode_midi(midi)
            sequence = self._compose_sequence(condition_ids, event_ids)

            # Apply length filter
            if len(sequence) < self.min_length or len(sequence) > self.max_length:
                self.skipped_files.append(f"{midi_path} (length={len(sequence)})")
                continue

            self.samples.append(
                EncodedSample(tokens=sequence, source_path=midi_path, conditions=condition_tokens)
            )
            self.lengths.append(len(sequence))

            if (idx + 1) % 200 == 0:
                logging.info("Processed %d samples", idx + 1)

        logging.info("Loaded %d sequences (skipped %d)", len(self.samples), len(self.skipped_files))

    def _compose_sequence(self, condition_ids: list[int], event_ids: list[int]) -> list[int]:
        seq: list[int] = [self.tokenizer.bos_id]
        seq.extend(condition_ids)
        seq.append(self.tokenizer.sep_id)
        seq.extend(event_ids)
        seq.append(self.tokenizer.eos_id)
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
            seq[-1] = self.tokenizer.eos_id
        return seq

    def _build_condition_tokens(self, row: pd.Series) -> list[str]:
        tokens: list[str] = []

        emotion = str(row.get("label.emotion", "unknown")) or "unknown"
        tokens.append(f"<emotion:{emotion}>")

        genre = str(row.get("label.genre", "unknown")) or "unknown"
        tokens.append(f"<genre:{genre}>")

        valence = row.get("emotion.valence")
        if pd.notna(valence):
            bucket = quantize(float(valence), buckets=10)
            if bucket >= 0:
                tokens.append(f"<valence:{bucket}>")

        arousal = row.get("emotion.arousal")
        if pd.notna(arousal):
            bucket = quantize(float(arousal), buckets=10)
            if bucket >= 0:
                tokens.append(f"<arousal:{bucket}>")

        score_total = row.get("score.total")
        if pd.notna(score_total):
            bucket = int(min(9, max(0, round(float(score_total) / 10.0))))
            tokens.append(f"<score:{bucket}>")

        caption = str(row.get("label.caption", "") or "")
        for tok in tokenize_caption(caption, limit=self.max_caption_tokens):
            tokens.append(f"<cap:{tok}>")

        digest = str(row.get("file_digest", ""))
        if digest and digest in self.techniques_by_digest:
            for tech in sorted(self.techniques_by_digest[digest]):
                tokens.append(f"<tech:{tech}>")

        art_presence_raw = row.get("articulation.presence")
        if isinstance(art_presence_raw, str) and art_presence_raw.strip().startswith("{"):
            try:
                presence = json.loads(art_presence_raw)
                for name, value in presence.items():
                    if isinstance(value, (int, float)) and value >= 0.1:
                        tokens.append(f"<tech:{name}>")
            except json.JSONDecodeError:
                logging.debug("Invalid articulation.presence JSON: %s", art_presence_raw[:80])

        tokens = list(dict.fromkeys(tokens))  # preserve order, remove duplicates
        return tokens

    def _compute_sample_weights(self) -> None:
        """Compute sampling weights based on (genre, emotion) pair distribution.

        Two-stage balancing: rare (genre, emotion) combinations get higher weights
        to improve coverage of underrepresented scenarios (e.g., rock×sad, jazz×angry).
        """
        if not self.samples:
            return

        # Extract (genre, emotion) pairs
        pair_counts: Counter[tuple[str, str]] = Counter()

        for sample in self.samples:
            genre = "unknown"
            emotion = "unknown"

            # Extract genre and emotion from conditions
            for cond in sample.conditions:
                if cond.startswith("<genre:"):
                    genre = cond.replace("<genre:", "").replace(">", "")
                elif cond.startswith("<emotion:"):
                    emotion = cond.replace("<emotion:", "").replace(">", "")

            pair_counts[(genre, emotion)] += 1

        # Compute weights (inverse frequency for rare pairs)
        total_samples = len(self.samples)
        epsilon = 0.01  # Smoothing to avoid division by zero

        for sample in self.samples:
            genre = "unknown"
            emotion = "unknown"

            for cond in sample.conditions:
                if cond.startswith("<genre:"):
                    genre = cond.replace("<genre:", "").replace(">", "")
                elif cond.startswith("<emotion:"):
                    emotion = cond.replace("<emotion:", "").replace(">", "")

            # Compute pair frequency
            pair_freq = pair_counts[(genre, emotion)] / total_samples

            # Apply inverse frequency weighting
            # Rare pairs (low freq) get high weights
            base_weight = 1.0 / (pair_freq + epsilon)

            # Optional: apply user-specified genre weight multiplier
            genre_multiplier = self.genre_weights.get(genre, 1.0)
            weight = base_weight * genre_multiplier

            self.sample_weights.append(weight)

        # Normalize weights to sum to 1
        total_weight = sum(self.sample_weights)
        if total_weight > 0:
            self.sample_weights = [w / total_weight for w in self.sample_weights]

    # Dataset protocol ------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        sample = self.samples[index]
        tensor = torch.tensor(sample.tokens, dtype=torch.long)
        return {"input_ids": tensor}

    # Reporting -------------------------------------------------------
    def summary(self) -> dict[str, object]:
        if not self.samples:
            return {"total_samples": 0}
        return {
            "total_samples": len(self.samples),
            "min_length": int(min(self.lengths)),
            "max_length": int(max(self.lengths)),
            "avg_length": float(sum(self.lengths) / len(self.lengths)),
            "condition_counts": dict(self.condition_counter),
            "skipped_files": self.skipped_files,
        }

    def save_summary(self, path: Path) -> None:
        path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class Stage3Transformer(torch.nn.Module if torch is not None else object):
    """GPT-2 based autoregressive model with optional LoRA."""

    def __init__(
        self,
        vocab_size: int,
        *,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        lora_rank: int = 8,
        lora_alpha: int | None = None,
        use_lora: bool = True,
    ) -> None:
        if GPT2Config is None or GPT2LMHeadModel is None or torch is None:
            raise RuntimeError("Install torch and transformers to use Stage3Transformer")
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_positions=2048,
        )
        base_model = GPT2LMHeadModel(config)

        if (
            use_lora
            and LoraConfig is not None
            and get_peft_model is not None
            and TaskType is not None
        ):
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha or lora_rank * 2,
                target_modules=["c_attn"],
                inference_mode=False,
            )
            self.model = get_peft_model(base_model, lora_cfg)
            logging.info(
                "LoRA attached (rank=%d, alpha=%s)", lora_rank, lora_alpha or lora_rank * 2
            )
        else:
            self.model = base_model
            if use_lora:
                logging.warning("LoRA dependencies missing; continuing without LoRA")

    def forward(self, *args, **kwargs):  # type: ignore[override]
        return self.model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


def collate_batch(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """Standard collator with padding."""
    lengths = [item["input_ids"].shape[0] for item in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)

    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : seq.shape[0]] = seq
        attention_mask[i, : seq.shape[0]] = 1

    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate_batch_packed(
    batch: list[dict[str, torch.Tensor]], pad_id: int, eos_id: int, max_length: int = 2048
) -> dict[str, torch.Tensor]:
    """Sequence packing collator for improved GPU utilization.

    Concatenates multiple short sequences into a single packed sequence,
    separated by EOS tokens, up to max_length.
    """
    # Sort by length (ascending) for better packing
    sorted_batch = sorted(batch, key=lambda x: x["input_ids"].shape[0])

    packed_sequences: list[torch.Tensor] = []
    current_pack: list[torch.Tensor] = []
    current_length = 0

    for item in sorted_batch:
        seq = item["input_ids"]
        seq_len = seq.shape[0]

        # Try to add to current pack
        if current_length + seq_len + 1 <= max_length:  # +1 for EOS separator
            current_pack.append(seq)
            if current_length > 0:  # Add EOS separator between sequences
                current_length += 1
            current_length += seq_len
        else:
            # Finalize current pack
            if current_pack:
                packed = _pack_sequences(current_pack, eos_id)
                packed_sequences.append(packed)
            # Start new pack
            current_pack = [seq]
            current_length = seq_len

    # Finalize last pack
    if current_pack:
        packed = _pack_sequences(current_pack, eos_id)
        packed_sequences.append(packed)

    # Pad packed sequences to same length
    if not packed_sequences:
        # Fallback to standard collation
        return collate_batch(batch, pad_id)

    max_len = max(s.shape[0] for s in packed_sequences)
    input_ids = torch.full((len(packed_sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)

    for i, seq in enumerate(packed_sequences):
        input_ids[i, : seq.shape[0]] = seq
        attention_mask[i, : seq.shape[0]] = 1

    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _pack_sequences(sequences: list[torch.Tensor], eos_id: int) -> torch.Tensor:
    """Concatenate sequences with EOS separators."""
    parts: list[torch.Tensor] = []
    for i, seq in enumerate(sequences):
        parts.append(seq)
        if i < len(sequences) - 1:  # Add EOS between sequences
            parts.append(torch.tensor([eos_id], dtype=torch.long))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_stage3(args: argparse.Namespace) -> None:
    if torch is None or Trainer is None or TrainingArguments is None:
        raise RuntimeError("Install torch and transformers to run training")

    # Enhanced reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tokenizer = Stage3Tokenizer(
        beat_division=args.beat_division,
        max_time_shift=args.max_time_shift,
        velocity_bins=args.velocity_bins,
        max_duration=args.max_duration,
        max_bars=args.max_bars,
    )

    dataset = Stage3Dataset(
        metadata_path=args.metadata,
        midi_root=args.midi_root,
        tokenizer=tokenizer,
        technique_metadata=args.technique_meta,
        max_seq_len=args.max_length,
        max_caption_tokens=args.max_caption_tokens,
        min_notes=args.min_notes,
        max_samples=args.max_samples,
        min_length=getattr(args, "min_seq_length", 20),
        max_length=getattr(args, "max_seq_length", 2048),
    )

    summary = dataset.summary()
    logging.info("Dataset summary: %s", json.dumps(summary, ensure_ascii=False))

    args.out.mkdir(parents=True, exist_ok=True)
    dataset.save_summary(args.out / "dataset_summary.json")
    tokenizer.save(args.out / "tokenizer_stage3.json")

    if args.dry_run:
        logging.info("Dry run requested; skipping training step")
        return

    if len(dataset) == 0:
        raise RuntimeError("No samples available for training")

    eval_dataset = None
    if args.eval_split > 0.0 and len(dataset) > 1:
        eval_size = max(1, int(len(dataset) * args.eval_split))
        train_size = len(dataset) - eval_size
        train_dataset, eval_dataset = random_split(
            dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset = dataset

    model = Stage3Transformer(
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_lora=not args.disable_lora,
    )

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.steps if args.steps is not None else None,
        num_train_epochs=args.epochs if args.epochs is not None else None,
        logging_steps=max(1, args.logging_steps),
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 0,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=["none"],
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        max_grad_norm=args.clip_grad_norm,  # Gradient clipping
        lr_scheduler_type=args.lr_scheduler,  # Cosine/linear scheduler
        seed=args.seed,
    )

    # Choose collator based on packing option
    if getattr(args, "pack_sequences", False):
        data_collator = lambda batch: collate_batch_packed(
            batch, tokenizer.pad_id, tokenizer.eos_id, max_length=args.max_length
        )
    else:
        data_collator = lambda batch: collate_batch(batch, tokenizer.pad_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(args.out / "model"))
    logging.info("Training complete. Model saved to %s", args.out / "model")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage3 conditional MIDI generator")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with Stage3 labels")
    parser.add_argument(
        "--midi-root", type=Path, required=True, help="Directory containing MIDI files"
    )
    parser.add_argument(
        "--technique-meta", type=Path, help="JSONL metadata with synthesized techniques"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory")

    parser.add_argument(
        "--max-length", type=int, default=2048, help="Maximum token length per sample"
    )
    parser.add_argument(
        "--max-caption-tokens", type=int, default=16, help="Caption tokens per sample"
    )
    parser.add_argument(
        "--min-notes", type=int, default=4, help="Minimum notes required to keep a MIDI"
    )
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--beat-division", type=int, default=24, help="Quantisation steps per beat")
    parser.add_argument("--max-time-shift", type=int, default=64, help="Maximum TIME token value")
    parser.add_argument("--velocity-bins", type=int, default=16, help="Velocity bins")
    parser.add_argument(
        "--max-duration", type=int, default=256, help="Maximum duration token value"
    )

    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides steps)")
    parser.add_argument("--steps", type=int, help="Maximum training steps")
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging interval")
    parser.add_argument("--save-steps", type=int, help="Checkpoint interval")
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluation interval")
    parser.add_argument("--eval-split", type=float, default=0.05, help="Evaluation split ratio")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training if supported")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training if supported")
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing"
    )

    parser.add_argument("--n-layer", type=int, default=12, help="Transformer layers")
    parser.add_argument("--n-head", type=int, default=12, help="Attention heads")
    parser.add_argument("--n-embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha scaling")
    parser.add_argument(
        "--disable-lora", action="store_true", help="Disable LoRA even if available"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic", action="store_true", help="Enable deterministic mode for reproducibility"
    )
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant"],
        help="LR scheduler type",
    )
    parser.add_argument("--max-bars", type=int, default=16, help="Maximum bar position tokens")
    parser.add_argument(
        "--pack-sequences", action="store_true", help="Enable sequence packing for GPU efficiency"
    )
    parser.add_argument(
        "--min-seq-length", type=int, default=20, help="Minimum sequence length (filtering)"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length (filtering)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only build dataset/tokeniser and exit"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    train_stage3(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
