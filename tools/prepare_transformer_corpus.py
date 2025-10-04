from __future__ import annotations

__doc__ = """Prepare a small Transformer corpus from MIDI files.

This script builds a dataset suitable for training sequence models on personal
MIDI collections. It chops each MIDI file into fixed-size bar segments, applies
simple tokenisation and optionally merges metadata such as tags or lyric text.
Tag keys in YAML files should be paths relative to the input root and
preferably lowercase. The resulting dataset is written as JSONL files with
deterministic train/valid/test splits.

Example
-------
python -m tools.prepare_transformer_corpus \
  --in data/midi_personal \
  --out data/corpus/personal_v1 \
  --bars-per-sample 4 --quant 480 --min-notes 8 \
  --duv on --dur-bins 16 --vel-bins 8 --duv-max 1000 \
  --embed-offline embeds.json \
  --tags sections.yaml mood.yaml \
  --split 0.9 0.05 0.05 --seed 42
"""

import argparse
import json
import logging
import math
import os
import random
import warnings
import gzip
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Callable,
    Sequence,
    Iterator,
)
from collections import Counter
import concurrent.futures

import mido
import pretty_midi
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from utilities.pretty_midi_safe import pm_to_mido

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Tokenised sample with metadata."""

    tokens: List[str]
    meta: Dict[str, object]


def _duv_frequency(splits: Dict[str, List[Sample]]) -> Counter[str]:
    freq: Counter[str] = Counter()
    for samp_list in splits.values():
        for sample in samp_list:
            for tok in sample.tokens:
                if tok.startswith("DUV_"):
                    freq[tok] += 1
    return freq


def _collapse_duv_tokens_inplace(
    splits: Dict[str, List[Sample]], limit: Optional[int]
) -> Tuple[int, int]:
    """Collapse infrequent DUV tokens to ``DUV_OOV`` in-place.

    Returns a tuple ``(kept, collapsed)`` describing how many distinct DUV
    tokens remain and how many were merged into ``DUV_OOV``.
    """

    if limit is None:
        return 0, 0

    freq = _duv_frequency(splits)
    if not freq:
        return 0, 0

    keep = {tok for tok, _ in freq.most_common(max(limit, 0))}
    collapsed = len(freq) - len(keep)
    if collapsed:
        for samp_list in splits.values():
            for sample in samp_list:
                sample.tokens = [
                    tok if not tok.startswith("DUV_") or tok in keep else "DUV_OOV"
                    for tok in sample.tokens
                ]
    return len(freq) - collapsed, collapsed


def _total_notes(pm: pretty_midi.PrettyMIDI) -> int:
    """Return total note count in a PrettyMIDI object; safe on weird inputs."""
    try:
        return sum(len(inst.notes) for inst in getattr(pm, "instruments", []))
    except Exception:
        return 0


def _make_const_sec_to_beats(tempo_bpm: float) -> Callable[[float], float]:
    """Simple seconds->beats mapping under constant tempo assumption."""
    spb = 60.0 / max(float(tempo_bpm), 1e-6)
    return lambda t: t / spb


def is_fallback(mapper: Callable[[float], float]) -> bool:
    """Return whether *mapper* was produced via a fallback path."""

    return bool(getattr(mapper, "_fallback", False))


def build_beat_map(
    pm: "pretty_midi.PrettyMIDI", *, path: Path | None = None
) -> tuple[Callable[[float], float], float]:
    """Return a seconds→beats mapper and tempo estimate.

    The mapper preserves piecewise tempo changes exposed by
    :meth:`pretty_midi.PrettyMIDI.get_beats`.  When that information is
    unavailable we gracefully fall back to a constant tempo derived from
    :meth:`estimate_tempo` (or 120 BPM as a last resort).  The returned
    callable carries a ``_fallback`` attribute so callers can detect whether
    the simplified path was used without altering the public return arity.
    """

    beat_times: "np.ndarray[Any, np.dtype[np.float_]]"
    fallback = False
    try:
        beat_times = pm.get_beats()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("get_beats failed for %s: %s", path, exc)
        beat_times = np.array([], dtype=float)
        fallback = True

    tempo_est: Optional[float] = None
    if beat_times.size >= 2:
        diffs = np.diff(beat_times)
        valid = diffs > 0
        if np.any(valid):
            trimmed = diffs[valid]
            if trimmed.size:
                try:
                    q_low, q_high = np.quantile(trimmed, [0.01, 0.99])
                    mask = (trimmed >= q_low) & (trimmed <= q_high)
                    if np.any(mask):
                        trimmed = trimmed[mask]
                except Exception:  # pragma: no cover - defensive
                    pass
            if trimmed.size:
                avg = float(np.mean(trimmed))
                if avg > 0:
                    tempo_est = 60.0 / avg
        else:  # pragma: no cover - degenerate beat grid
            fallback = True
    else:
        fallback = True

    if tempo_est is None:
        try:
            t = float(pm.estimate_tempo())
            if math.isfinite(t) and t > 0:
                tempo_est = t
        except Exception as exc:  # pragma: no cover - logging only
            logger.debug("estimate_tempo failed for %s: %s", path, exc)
    if tempo_est is None:
        tempo_est = 120.0

    if beat_times.size >= 2 and not fallback:
        beat_idx = np.arange(len(beat_times), dtype=float)

        def sec_to_beats(t: float) -> float:
            if not math.isfinite(t):
                return 0.0
            if t <= beat_times[-1]:
                return float(np.interp(t, beat_times, beat_idx))
            extra = max(0.0, t - beat_times[-1]) * tempo_est / 60.0
            return float(beat_idx[-1] + extra)

    else:

        def sec_to_beats(t: float) -> float:
            if not math.isfinite(t):
                return 0.0
            return max(0.0, t) * tempo_est / 60.0

    setattr(sec_to_beats, "_fallback", fallback)
    return sec_to_beats, tempo_est


def get_time_signature(mid: mido.MidiFile) -> Tuple[float, str]:
    """Extract first time signature; default to 4/4."""

    for track in mid.tracks:
        for msg in track:
            if msg.type == "time_signature":
                beats_per_bar = msg.numerator * 4.0 / msg.denominator
                return beats_per_bar, f"{msg.numerator}/{msg.denominator}"
    return 4.0, "4/4"


def quantise_beats(value: float, quant: int) -> float:
    """Quantise ``value`` (in beats) to ``1/quant`` resolution."""

    return round(value * quant) / quant


def bin_duration(duration_beats: float, bins: int) -> int:
    """Map a duration in beats to a discrete bin index."""

    return int(np.clip(round(duration_beats * bins), 0, bins - 1))


def bin_velocity(velocity: int, bins: int) -> int:
    """Map velocity 0-127 to a discrete bin index."""

    return int(np.clip(velocity * bins // 128, 0, bins - 1))


def tokenize_notes(
    notes: Iterable["pretty_midi.Note"],
    *,
    duv: bool,
    dur_bins: int,
    vel_bins: int,
    quant: int,
) -> List[str]:
    """Tokenise notes into simple NOTE/DUR/VEL or NOTE/DUV tokens.

    Notes are sorted for determinism and start/end are expressed in beats.
    """

    tokens: List[str] = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch, n.end, n.velocity)):
        start = quantise_beats(n.start, quant)
        end = quantise_beats(n.end, quant)
        dur_beats = max(0.0, end - start)
        d_bin = bin_duration(dur_beats, dur_bins)
        v_bin = bin_velocity(n.velocity, vel_bins)
        tokens.append(f"NOTE_{n.pitch}")
        if duv:
            tokens.append(f"DUV_{d_bin}_{v_bin}")
        else:
            tokens.append(f"D_{d_bin}")
            tokens.append(f"V_{v_bin}")
    return tokens


def load_tag_maps(tag_files: Sequence[Path]) -> Dict[str, Dict[str, str]]:
    """Load per-file metadata from YAML files.

    Keys should be relative to the input root and lowercase to match
    :func:`normalize_key`.
    """
    if yaml is None:
        logger.warning("PyYAML not installed; skipping tags")
        return {}
    tag_map: Dict[str, Dict[str, str]] = {}
    for path in tag_files:
        if not path.is_file():
            logger.warning("tag file %s missing", path)
            continue
        data = yaml.safe_load(path.read_text()) or {}
        for fp, tags in data.items():
            tag_map.setdefault(fp, {}).update(tags or {})
    return tag_map


def gather_midi_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.mid") if p.is_file())


def normalize_key(path: Path, base: Path) -> str:
    """Return a base-relative, lowercased POSIX-style key for *path*."""

    try:
        rel = path.resolve().relative_to(base.resolve())
    except Exception:
        rel = Path(path.name)
    return rel.as_posix().lower()


def load_embed_map(path: Path) -> Dict[str, List[float]]:
    """Load pre-computed text embeddings from JSON or NPZ."""

    import numpy as np

    if path.suffix == ".npz":
        data = np.load(path)
        embed_map = {k: data[k].tolist() for k in data.files}
    else:
        embed_map = {k: list(v) for k, v in json.loads(path.read_text()).items()}
    dims = {len(v) for v in embed_map.values()}
    if len(dims) != 1:
        raise ValueError("inconsistent embedding vector lengths")
    return embed_map


def split_samples(
    pm: "pretty_midi.PrettyMIDI",
    *,
    bars_per_sample: int,
    min_notes: int,
    beats_per_bar: float,
    sec_to_beats: Callable[[float], float],
    include_programs: set[int] | None,
    drums_only: bool,
    exclude_drums: bool,
    max_segments: int | None = None,
    quant: int | None = None,
) -> Iterator[List[pretty_midi.Note]]:
    """Yield note groups for each slice of ``bars_per_sample`` bars."""

    import pretty_midi

    segment_beats = bars_per_sample * beats_per_bar
    total_beats = sec_to_beats(pm.get_end_time())
    if quant:
        total_beats = quantise_beats(total_beats, quant)
    n_segments = int(max(0, math.floor(total_beats / segment_beats + 1e-8)))
    if n_segments == 0 and total_beats > 0:
        n_segments = 1
    for i in range(n_segments):
        start = i * segment_beats
        end = start + segment_beats
        seg: List[pretty_midi.Note] = []
        for inst in pm.instruments:
            if drums_only and not inst.is_drum:
                continue
            if exclude_drums and inst.is_drum:
                continue
            if include_programs and inst.program not in include_programs:
                continue
            for n in inst.notes:
                n_start = sec_to_beats(n.start)
                if start <= n_start < end:
                    n_end = sec_to_beats(n.end)
                    if quant:
                        n_start = quantise_beats(n_start, quant)
                        n_end = quantise_beats(n_end, quant)
                    n_end = max(n_end, n_start)
                    local_start = max(n_start, start)
                    local_end = min(n_end, end)
                    if local_end > local_start:
                        seg.append(
                            pretty_midi.Note(
                                velocity=n.velocity,
                                pitch=n.pitch,
                                start=local_start - start,
                                end=local_end - start,
                            )
                        )
        if len(seg) >= min_notes:
            yield seg
        else:
            _inc_skip("too_few_notes")
        if max_segments is not None and i + 1 >= max_segments:
            break


def build_corpus(args: argparse.Namespace, files: Sequence[Path]) -> Dict[str, List[Sample]]:
    """Process *files* and return split samples."""

    tag_map = load_tag_maps([Path(p) for p in args.tags]) if args.tags else {}
    lyric_json = getattr(args, "lyric_json", None)
    if lyric_json:
        base = Path(args.in_dir).resolve()
        raw_map = json.loads(Path(lyric_json).read_text())
        lyric_map = {normalize_key(Path(k), base): v for k, v in raw_map.items()}
    else:
        lyric_map = {}
    embed_map: Dict[str, List[float]] = {}
    if getattr(args, "embed_offline", None):
        embed_map = load_embed_map(Path(args.embed_offline))
        dim = len(next(iter(embed_map.values()))) if embed_map else 0
        logger.info(
            "loaded %d offline embeddings (dim=%d)",
            len(embed_map),
            dim,
        )
    embed_model = None
    if lyric_map and not embed_map:
        try:  # pragma: no cover - optional dependency
            from sentence_transformers import SentenceTransformer

            embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:  # pragma: no cover - handled in tests
            logger.warning("sentence-transformers unavailable; storing raw text")

    global _ARGS, _BASE, _TAG_MAP, _LYRIC_MAP, _EMBED_MAP, _EMBED_MODEL
    _ARGS = args
    base = Path(args.in_dir)
    _BASE = base
    _TAG_MAP = tag_map
    _LYRIC_MAP = lyric_map
    _EMBED_MAP = embed_map
    _EMBED_MODEL = embed_model

    if args.max_files:
        files = files[: args.max_files]

    iterator: Iterable[Path] = files
    if args.progress:
        try:  # pragma: no cover - optional
            from tqdm import tqdm

            iterator = tqdm(iterator)
        except Exception:  # pragma: no cover
            pass

    lyric_matches = sum(1 for f in files if normalize_key(f, base) in lyric_map)
    logger.info("lyrics matched: %d/%d", lyric_matches, len(files))

    samples: List[Sample] = []
    use_mp = args.num_workers > 1 and embed_model is None
    paths = list(iterator)
    cfg = FastCfg()
    cfg.drums_only = getattr(args, "drums_only", False)
    cfg.min_file_notes = getattr(args, "min_file_notes", 0)
    cfg.min_file_seconds = getattr(args, "min_file_seconds", 0.0)
    cfg.max_file_seconds = getattr(args, "max_file_seconds", 0.0)
    cfg.silence_threshold_db = getattr(args, "silence_threshold_db", -60.0)
    cfg.silent_fraction = getattr(args, "silent_fraction", 0.95)
    cfg.skip_lyrics = getattr(args, "skip_lyrics", False)
    if use_mp:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                for res in ex.map(_worker, ((p, cfg) for p in paths)):
                    samples.extend(res)
        except Exception:
            logger.warning(
                "ProcessPoolExecutor failed; falling back to " "ThreadPoolExecutor(max_workers=%d)",
                args.num_workers,
            )
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                for res in ex.map(_worker, ((p, cfg) for p in paths)):
                    samples.extend(res)
    else:
        if args.num_workers > 1 and embed_model is not None:
            logger.warning("lyric embeddings disable multiprocessing; using " "single worker")
        for p in iterator:
            samples.extend(process_path(p, cfg))

    extra = {
        "lyric_matches": lyric_matches,
        "midi_file_count": len(files),
        "midi_files": len(files),
    }

    if not samples:
        logger.warning("no samples found under %s", args.in_dir)
        return {"train": [], "valid": [], "test": []}, {"extra": extra}

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_total = len(samples)
    n_train = int(args.split[0] * n_total)
    n_valid = int(args.split[1] * n_total)
    n_test = n_total - n_train - n_valid
    logger.info(
        "split counts train=%d valid=%d test=%d total=%d",
        n_train,
        n_valid,
        n_test,
        n_total,
    )
    splits = {
        "train": samples[:n_train],
        "valid": samples[n_train : n_train + n_valid],
        "test": samples[n_train + n_valid : n_train + n_valid + n_test],
    }

    keep_count, collapsed = _collapse_duv_tokens_inplace(splits, getattr(args, "duv_max", None))
    if keep_count or collapsed:
        logger.info("DUV kept: %d collapsed: %d", keep_count, collapsed)
    args._duv_stats = (keep_count, collapsed)
    args._duv_collapsed = True

    return splits, {"extra": extra}


def save_jsonl(path: Path, samples: Sequence[Sample], *, compress: str = "none") -> Path:
    """Write ``samples`` to ``path`` as JSONL, optionally gzip-compressed."""

    actual_path = path
    if compress == "gz":
        import gzip

        actual_path = path.with_suffix(path.suffix + ".gz")
        fh = gzip.open(actual_path, "wt", encoding="utf-8")
    else:
        fh = actual_path.open("w", encoding="utf-8")
    with fh:
        for s in samples:
            fh.write(json.dumps({"tokens": s.tokens, "meta": s.meta}) + "\n")
    return actual_path


def build_vocab(samples: Iterable[Sample]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for s in samples:
        for tok in s.tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_tag_vocab(samples: Iterable[Sample]) -> Dict[str, Dict[str, int]]:
    vocab: Dict[str, Dict[str, int]] = {}
    for s in samples:
        for k, v in s.meta.items():
            if not isinstance(v, str):
                continue
            vocab.setdefault(k, {})
            if v not in vocab[k]:
                vocab[k][v] = len(vocab[k])
    return vocab


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prepare_transformer_corpus")
    p.add_argument(
        "--in",
        dest="in_dir",
        type=str,
        required=True,
        help="input MIDI folder",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        type=str,
        required=True,
        help="output corpus root",
    )
    p.add_argument("--bars-per-sample", type=int, default=4)
    p.add_argument(
        "--quant",
        type=int,
        default=480,
        help="per-beat resolution (e.g., 96/240/480)",
    )
    p.add_argument("--min-notes", type=int, default=1)
    p.add_argument("--duv", type=str, choices=["on", "off"], default="off")
    p.add_argument("--dur-bins", type=int, default=16)
    p.add_argument("--vel-bins", type=int, default=8)
    p.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help=("YAML metadata files; keys relative to input root " "(prefer lowercase)"),
    )
    p.add_argument(
        "--lyric-json",
        type=str,
        default=None,
        help="JSON file mapping paths to lyrics",
    )
    p.add_argument(
        "--embed-offline",
        type=str,
        default=None,
        help="JSON or NPZ mapping paths to precomputed text embeddings",
    )
    p.add_argument("--split", nargs=3, type=float, default=(0.9, 0.05, 0.05))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--section-tokens", action="store_true")
    p.add_argument("--mood-tokens", action="store_true")
    p.add_argument("--include-programs", nargs="*", type=int, default=None)
    p.add_argument("--drums-only", action="store_true")
    p.add_argument("--exclude-drums", action="store_true")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-samples-per-file", type=int, default=None)
    p.add_argument("--progress", action="store_true")
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "worker processes (disabled when embedding text at runtime; "
            "use --embed-offline to re-enable)"
        ),
    )
    p.add_argument(
        "--duv-max",
        type=int,
        default=None,
        help="max distinct DUV tokens; rare pairs collapse to DUV_OOV",
    )
    p.add_argument(
        "--compress",
        choices=["none", "gz"],
        default="none",
        help="compress output JSONL",
    )
    # ---- Silence / envelope early exit
    p.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-60.0,
        help="RMS dB threshold to treat as silent (default: -60dB)",
    )
    p.add_argument(
        "--silent-fraction",
        type=float,
        default=0.95,
        help="fraction of silent frames to skip (default: 0.95)",
    )
    # ---- Dry run: collect stats only (no writes)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="collect stats only; skip writing outputs to --out",
    )

    # ---- Fast-path filters (file-level early skip) ----
    p.add_argument(
        "--min-file-notes",
        type=int,
        default=0,
        help=("Skip files whose total note count is below this " "threshold (0=disabled)"),
    )
    p.add_argument(
        "--min-file-seconds",
        type=float,
        default=0.0,
        help="Skip files shorter than this duration in seconds (0=disabled)",
    )
    p.add_argument(
        "--max-file-seconds",
        type=float,
        default=0.0,
        help="Skip files longer than this duration in seconds (0=disabled)",
    )
    # ---- Lyrics pipeline toggle ----
    p.add_argument(
        "--skip-lyrics",
        action="store_true",
        help="Disable lyrics matching (speeds up corpus build)",
    )
    return p


# silence pretty_midi/pkg_resources deprecation noise
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated.*",
    module="pretty_midi.*",
)


# ---- tiny, picklable config for process workers ----
class FastCfg:
    drums_only: bool = False
    min_file_notes: int = 0
    min_file_seconds: float = 0.0
    max_file_seconds: float = 0.0
    # envelope-based fast filter
    silence_threshold_db: float = -60.0
    silent_fraction: float = 0.95
    skip_lyrics: bool = False


# --- skip metrics (already added previously) --------------------------
_SKIP_COUNTS: Counter[str] = Counter()
_tempo_fallbacks = 0

_ARGS: argparse.Namespace | None = None
_BASE: Path | None = None
_TAG_MAP: Dict[str, Dict[str, str]] = {}
_LYRIC_MAP: Dict[str, str] = {}
_EMBED_MAP: Dict[str, List[float]] = {}
_EMBED_MODEL = None


def _inc_skip(reason: str) -> None:
    _SKIP_COUNTS[reason] += 1


def _safe_getattr(namespace: Any, key: str, default: Any | None = None) -> Any | None:
    """Return ``getattr(namespace, key, default)`` with a safe default."""

    return getattr(namespace, key, default)


def _normalise_label(label: str) -> str:
    """Return a lowercase label with extraneous whitespace collapsed."""

    if not label:
        return ""
    return " ".join(label.strip().lower().split())


def _instrument_hint(inst: pretty_midi.Instrument | None) -> str:
    """Return a canonical instrument hint for *inst* suitable for metadata."""

    if inst is None:
        return ""
    if getattr(inst, "is_drum", False):
        return "drums"
    label = inst.name or ""
    if not label:
        try:
            label = pretty_midi.program_to_instrument_name(inst.program)
        except Exception:  # pragma: no cover - defensive
            label = ""
    return _normalise_label(label)


def _worker(args: tuple[Path, FastCfg]) -> List[Sample]:
    """Process worker for multiprocessing; unwraps args."""
    return process_path(*args)


def process_path(midi_path: Path, ns: FastCfg) -> List[Sample]:
    global _tempo_fallbacks

    args = _ARGS
    base = _BASE
    if args is None or base is None:
        raise RuntimeError("build_corpus must set global state before use")

    try:
        pm = pretty_midi.PrettyMIDI(midi_path.as_posix())
    except Exception:
        _inc_skip("invalid_midi")
        return []
    try:
        _times, tempi = pm.get_tempo_changes()
        if not tempi or not float(tempi[0]) > 0:
            scale = 60.0 / (120.0 * pm.resolution)
            pm._tick_scales = [(0, scale)]  # noqa: SLF001
            if hasattr(pm, "_update_tick_to_time"):
                pm._update_tick_to_time(pm.resolution)  # noqa: SLF001
    except Exception:
        pass
    mid = pm_to_mido(pm)
    beats_per_bar, ts_str = get_time_signature(mid)

    instruments: List[pretty_midi.Instrument] = list(pm.instruments)
    notes_total = 0
    for inst in instruments:
        if ns.drums_only and not inst.is_drum:
            continue
        notes_total += len(inst.notes)
    dur_sec = float(pm.get_end_time() or 0.0)

    if ns.min_file_notes and notes_total < ns.min_file_notes:
        _inc_skip("too_few_notes")
        return []
    if ns.min_file_seconds and 0.0 < dur_sec < ns.min_file_seconds:
        _inc_skip("too_short")
        return []
    if ns.max_file_seconds and dur_sec > ns.max_file_seconds:
        _inc_skip("too_long")
        return []

    sec_to_beats, tempo_est = build_beat_map(pm, path=midi_path)
    if is_fallback(sec_to_beats):
        _tempo_fallbacks += 1
    rel = normalize_key(midi_path, base)
    segments: List[Sample] = []
    include_programs = set(args.include_programs) if args.include_programs else None
    first_inst: pretty_midi.Instrument | None = None
    for inst in instruments:
        if args.drums_only and not inst.is_drum:
            continue
        if args.exclude_drums and inst.is_drum:
            continue
        if include_programs and inst.program not in include_programs:
            continue
        if inst.notes:
            first_inst = inst
            break
    track_name = first_inst.name if first_inst else ""
    program = int(first_inst.program) if first_inst else -1
    channel = 9 if (first_inst and first_inst.is_drum) else 0
    instrument_name = _instrument_hint(first_inst)
    is_drum = bool(first_inst.is_drum) if first_inst else False
    for idx, seg in enumerate(
        split_samples(
            pm,
            bars_per_sample=args.bars_per_sample,
            min_notes=args.min_notes,
            beats_per_bar=beats_per_bar,
            sec_to_beats=sec_to_beats,
            include_programs=include_programs,
            drums_only=args.drums_only,
            exclude_drums=args.exclude_drums,
            max_segments=args.max_samples_per_file,
            quant=args.quant,
        )
    ):
        tokens = tokenize_notes(
            seg,
            duv=args.duv == "on",
            dur_bins=args.dur_bins,
            vel_bins=args.vel_bins,
            quant=args.quant,
        )
        tags = _TAG_MAP.get(rel, {})
        if args.section_tokens and tags.get("section"):
            tokens.insert(0, f"<SECTION={tags['section']}>")
        if args.mood_tokens and tags.get("mood"):
            tokens.insert(0, f"<MOOD={tags['mood']}>")
        meta: Dict[str, object] = {
            **tags,
            "source_path": rel,
            "path": rel,
            "segment_index": idx,
            "tempo_est": tempo_est,
            "beats_per_bar": beats_per_bar,
            "time_signature": ts_str,
            "track_name": track_name,
            "channel": channel,
            "program": program,
            "instrument": instrument_name,
            "is_drum": is_drum,
        }
        if rel in _EMBED_MAP:
            meta["text_emb"] = _EMBED_MAP[rel]
        elif (not ns.skip_lyrics) and rel in _LYRIC_MAP:
            text = _LYRIC_MAP[rel]
            if _EMBED_MODEL is not None:
                meta["text_emb"] = _EMBED_MODEL.encode(text).tolist()
            else:
                meta["text"] = text
        segments.append(Sample(tokens=tokens, meta=meta))
    return segments


# -------------- メイン処理 -------------------
def main(argv: list[str] | None = None) -> None:
    # 引数パース
    args = build_argparser().parse_args(argv)

    # ロガー設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("command: %s", " ".join(argv or []))
    logger.info("Python %s", sys.version.split()[0])
    logger.info("prepare_transformer_corpus starting up")

    # 入出力パス
    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    os.makedirs(out_dir, exist_ok=True)

    if args.duv_max is not None and args.duv != "on":
        raise SystemExit("--duv-max requires --duv on")
    if not math.isclose(sum(args.split), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise SystemExit("split must sum to 1.0")

    # 簡易ヘルプ
    if args.tags and not all(Path(p).is_file() for p in args.tags):
        missing = [p for p in args.tags if not Path(p).is_file()]
        logger.warning("一部のタグファイルが見つかりません: %s", missing)
        args.tags = [p for p in args.tags if Path(p).is_file()]
    if args.lyric_json and not Path(args.lyric_json).is_file():
        logger.error("歌詞 JSON ファイルが見つかりません: %s", args.lyric_json)
        return

    # MIDI ファイル収集
    files = gather_midi_files(in_dir)
    logger.info("files_scanned=%d", len(files))

    # I/O バッチング：num-workers > 1 のときシャッフル（ホットスポット回避）
    try:
        if getattr(args, "num_workers", 0) and int(args.num_workers) > 1:
            rnd = random.Random(getattr(args, "seed", None) or 0)
            rnd.shuffle(files)
    except Exception:
        pass

    # dry-run の明示ログ
    if _safe_getattr(args, "dry_run", False):
        logger.info(
            "dry_run is ON: no dataset files will be written (out=%s)",
            _safe_getattr(args, "out_dir", None),
        )
        # もし書き出し関数内で環境変数を見て分岐できるなら、フラグも立てておく
        os.environ["PREP_CORPUS_DRY_RUN"] = "1"

    max_files = getattr(args, "max_files", 0) or 0
    if max_files:
        files = files[:max_files]

    # コーパス構築
    splits, build_meta = build_corpus(args, files)
    meta_extra = (build_meta or {}).get("extra", {}) if isinstance(build_meta, dict) else {}
    lyric_matches = int(meta_extra.get("lyric_matches", 0))
    midi_file_count = int(
        meta_extra.get(
            "midi_file_count",
            meta_extra.get("midi_files", len(files)),
        )
    )
    logger.info("tempo fallback used: %d", _tempo_fallbacks)

    if getattr(args, "_duv_collapsed", False):
        keep_count, collapsed = getattr(args, "_duv_stats", (0, 0))
    else:
        keep_count, collapsed = _collapse_duv_tokens_inplace(splits, args.duv_max)
        if keep_count or collapsed:
            logger.info("DUV kept: %d collapsed: %d", keep_count, collapsed)

    # ボキャブラリ構築
    vocab = build_vocab(splits["train"])
    tag_vocab = build_tag_vocab(splits["train"])

    total_samples = sum(len(v) for v in splits.values())
    tag_hits = sum(1 for f in files if normalize_key(f, Path(args.in_dir)) in _TAG_MAP)
    duv_vocab_size = sum(1 for t in vocab if t.startswith("DUV_"))
    logger.info(
        "samples: %d tag hits: %d/%d duv vocab: %d",
        total_samples,
        tag_hits,
        midi_file_count,
        duv_vocab_size,
    )

    # メタデータ収集
    meta = {
        "midi_file_count": midi_file_count,
        "midi_files": midi_file_count,
        "lyric_matches": lyric_matches,
        "vocab_size": len(vocab),
        "tag_vocab_size": len(tag_vocab),
        "split": args.split,
        "duv_max": args.duv_max,
        "bars_per_sample": args.bars_per_sample,
        "quant": args.quant,
        "min_notes": args.min_notes,
        "dur_bins": args.dur_bins,
        "vel_bins": args.vel_bins,
        "embed_offline": bool(getattr(args, "embed_offline", None)),
        "tags": getattr(args, "tags", []),
        "min_file_notes": getattr(args, "min_file_notes", 0),
        "min_file_seconds": getattr(args, "min_file_seconds", 0.0),
        "max_file_seconds": getattr(args, "max_file_seconds", 0.0),
        "skipped_too_few_notes": _SKIP_COUNTS.get("too_few_notes", 0),
        "skipped_too_short": _SKIP_COUNTS.get("too_short", 0),
        "skipped_invalid_midi": _SKIP_COUNTS.get("invalid_midi", 0),
        "skipped_too_long": _SKIP_COUNTS.get("too_long", 0),
        "tempo_fallback_used": _tempo_fallbacks,
        "stats": {split: len(samples) for split, samples in splits.items()},
    }
    logger.info("meta: %s", json.dumps(meta, ensure_ascii=False, indent=2))
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # サンプル書き出し
    logger.info("writing samples...")
    for split, samples in splits.items():
        out_path = out_dir / f"{split}.jsonl"
        written_path = save_jsonl(out_path, samples, compress=args.compress)
        logger.info(
            "  %s: %d samples -> %s",
            split,
            len(samples),
            written_path,
        )

    # ボキャブラリ書き出し
    if args.compress == "gz":
        vocab_path = out_dir / "vocab.json.gz"
    else:
        vocab_path = out_dir / "vocab.json"
    logger.info("writing vocab: %s", vocab_path)
    with (
        gzip.open(vocab_path, "wt", encoding="utf-8")
        if args.compress == "gz"
        else open(vocab_path, "w", encoding="utf-8")
    ) as fh:
        json.dump(vocab, fh, ensure_ascii=False)

    # タグボキャブラリ書き出し
    if args.compress == "gz":
        tag_vocab_path = out_dir / "tag_vocab.json.gz"
    else:
        tag_vocab_path = out_dir / "tag_vocab.json"
    logger.info("writing tag vocab: %s", tag_vocab_path)
    with (
        gzip.open(tag_vocab_path, "wt", encoding="utf-8")
        if args.compress == "gz"
        else open(tag_vocab_path, "w", encoding="utf-8")
    ) as fh:
        json.dump(tag_vocab, fh, ensure_ascii=False)

    logger.info("done.")


if __name__ == "__main__":
    main()
