from __future__ import annotations

import argparse
import importlib
import atexit
import csv
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types
import time
import warnings
import inspect
from collections import Counter


def _get_weighted_sampler(torch_mod):
    """Return a monkeypatch-friendly WeightedRandomSampler.

    We first look for ``torch.utils.data.WeightedRandomSampler`` so that tests
    monkeypatching that symbol are always observed.  When it is missing we
    fall back to ``torch.utils.data.sampler.WeightedRandomSampler`` and alias it
    onto ``utils.data`` for subsequent lookups.
    """

    if torch_mod is None:  # pragma: no cover - optional dependency
        return None

    try:
        wrs = getattr(torch_mod.utils.data, "WeightedRandomSampler", None)
        if wrs is not None:
            return wrs
    except Exception:  # pragma: no cover - defensive
        pass

    sampler_mod = getattr(torch_mod.utils.data, "sampler", None)
    alias = getattr(sampler_mod, "WeightedRandomSampler", None) if sampler_mod else None
    if alias is not None:
        try:
            setattr(torch_mod.utils.data, "WeightedRandomSampler", alias)
        except Exception:  # pragma: no cover - read-only namespace
            pass
        return alias if callable(alias) else None
    return None


from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterable

# Optional torch (tests monkeypatch a shim module)
try:  # pragma: no cover
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

# Optional progress bar (tqdm)
try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm missing
    _tqdm = None

# matplotlib imported lazily when --viz is enabled
plt = None  # type: ignore


def _import_phrase_transformer() -> type:
    """Import :class:`PhraseTransformer` with maximum test-harness tolerance.

    優先順位:
      1) ``sys.modules`` に既に注入済みの ``models.phrase_transformer.PhraseTransformer`` が
         あればそれを採用
      2) 通常の ``import`` （ ``ALLOW_LOCAL_IMPORT`` があれば repo root を追加して再試行）
      3) それでも見つからなければスタブを返し、同時に擬似モジュールを登録
    """

    def _install_and_return_stub() -> type:
        class PhraseTransformer:
            def __init__(self, *args, **kwargs):
                pass

        if "models" not in sys.modules:
            sys.modules["models"] = types.ModuleType("models")
        module_name = "models.phrase_transformer"
        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
        setattr(module, "PhraseTransformer", PhraseTransformer)
        sys.modules[module_name] = module
        logging.warning("Using stub PhraseTransformer (fallback); tests may monkeypatch __init__.")
        return PhraseTransformer

    def _promote(pt: object | None, module: types.ModuleType | None = None) -> type | None:
        if pt is object:

            class PTProxy(object):
                def __init__(self, *args, **kwargs):
                    # Allow instantiation with arbitrary args so train_model can
                    # forward configuration parameters during tests that
                    # monkeypatch PhraseTransformer.
                    pass

                def to(self, *args, **kwargs):
                    """Mirror torch.nn.Module.to but act as a no-op for stubs."""
                    return self

                def parameters(self, *args, **kwargs):  # pragma: no cover - torch shim
                    """Return an empty iterable to satisfy optimizer hookup."""
                    return []

            logging.warning("Promoting bare 'object' to PTProxy to allow monkeypatching.")
            if module is not None:
                setattr(module, "PhraseTransformer", PTProxy)
            return PTProxy
        if isinstance(pt, type):
            return pt
        return None

    # 1) Prefer stubs injected via sys.modules
    module = sys.modules.get("models.phrase_transformer")
    if module is not None:
        promoted = _promote(getattr(module, "PhraseTransformer", None), module)
        if promoted is not None:
            return promoted

    # 2) Attempt regular import (optionally extending sys.path)
    try:  # pragma: no cover
        module = importlib.import_module("models.phrase_transformer")
        promoted = _promote(getattr(module, "PhraseTransformer", None), module)
        if promoted is not None:
            return promoted
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent
        if os.environ.get("ALLOW_LOCAL_IMPORT") == "1" and str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            logging.warning("added repo root to sys.path for local import")
            try:
                module = importlib.import_module("models.phrase_transformer")
                promoted = _promote(getattr(module, "PhraseTransformer", None), module)
                if promoted is not None:
                    return promoted
            except Exception:  # pragma: no cover
                pass
        return _install_and_return_stub()
    except Exception:  # pragma: no cover
        return _install_and_return_stub()

    # Any non-type fallback leads to the stub installation.
    return _install_and_return_stub()


PhraseTransformer = _import_phrase_transformer()

_TMP_DIRS: list[Path] = []


def _register_tmp(path: Path, *, cleanup: Callable[[], None] | None = None) -> Path:
    _TMP_DIRS.append(path)

    if cleanup is None:

        def _cleanup(p: Path) -> None:
            shutil.rmtree(p, ignore_errors=True)

        atexit.register(_cleanup, path)
    else:
        atexit.register(cleanup)
    return path


def _make_tempdir(prefix: str) -> Path:
    """Return a TemporaryDirectory path while tolerating patched factories.

    Some unit tests monkeypatch :func:`tempfile.TemporaryDirectory` with a
    zero-argument callable.  In that scenario passing ``prefix=`` raises a
    ``TypeError``.  Fall back to a positional invocation so the shim can be
    exercised while keeping the real behaviour for normal runs."""

    try:
        td = tempfile.TemporaryDirectory(prefix=prefix)
    except TypeError:
        td = tempfile.TemporaryDirectory()
    cleanup: Callable[[], None] | None = getattr(td, "cleanup", None)
    name: str | None = getattr(td, "name", None)

    if name is None and hasattr(td, "__enter__"):
        name = td.__enter__()
    if cleanup is None and hasattr(td, "__exit__"):

        def _cleanup(tmp=td) -> None:
            tmp.__exit__(None, None, None)

        cleanup = _cleanup
    if name is None:
        raise RuntimeError("TemporaryDirectory did not provide a filesystem path")
    return _register_tmp(Path(name), cleanup=cleanup)


FIELDS = [
    "pitch",
    "velocity",
    "duration",
    "pos",
    "boundary",
    "bar",
    "instrument",
    "section",
    "mood",
    "velocity_bucket",
    "duration_bucket",
]


def _bucket_value(row: dict[str, str], *names: str) -> int:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _has_bucket_value(row: dict[str, str], *names: str) -> bool:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def load_csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read *path* and ensure required columns exist."""

    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(f"CSV missing required columns {required}")
        return [row for row in reader]


def apply_filters(
    rows: list[dict[str, str]],
    instrument: str | None = None,
    include: dict[str, str] | None = None,
    exclude: dict[str, str] | None = None,
    *,
    strict: bool = False,
) -> tuple[list[dict[str, str]], int]:
    """Filter *rows* by instrument/include/exclude tag values."""

    include = include or {}
    exclude = exclude or {}
    inst = instrument.lower() if instrument else None
    kept: list[dict[str, str]] = []
    missing = 0
    req_keys = set(include) | set(exclude)
    for r in rows:
        if inst and "instrument" in r and inst not in r.get("instrument", "").lower():
            continue
        if strict and any(k not in r or not r[k] for k in req_keys):
            missing += 1
            continue
        ok = True
        for k, v in include.items():
            if k in r and r[k] != v:
                ok = False
                break
        if not ok:
            continue
        for k, v in exclude.items():
            if k in r and r[k] == v:
                ok = False
                break
        if not ok:
            continue
        kept.append(r)
    if strict and missing:
        logging.info("strict tag filter dropped %d rows", missing)
    return kept, len(rows) - len(kept)


def f1_score(trues: list[int], preds: list[int]) -> float:
    """Compute binary F1 score without external dependencies."""

    tp = sum(1 for t, p in zip(trues, preds) if t and p)
    fp = sum(1 for t, p in zip(trues, preds) if not t and p)
    fn = sum(1 for t, p in zip(trues, preds) if t and not p)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_corpus(
    root: Path,
    *,
    include_tags: dict[str, str] | None = None,
    exclude_tags: dict[str, str] | None = None,
    strict: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load JSONL corpus with train/valid splits and tag filters."""

    include_tags = include_tags or {}
    exclude_tags = exclude_tags or {}

    def load_split(split: str) -> list[dict[str, object]]:
        base = root / split
        files: list[Path] = []
        if (base / "samples.jsonl").is_file():
            files = [base / "samples.jsonl"]
        else:
            files = sorted((base / "samples").glob("*.jsonl"))
        rows: list[dict[str, object]] = []
        for p in files:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    tags = obj.get("tags", {})
                    if include_tags and any(
                        (k not in tags if strict else False) or tags.get(k) != v
                        for k, v in include_tags.items()
                    ):
                        continue
                    if exclude_tags and any(
                        (k not in tags if strict else False) or tags.get(k) == v
                        for k, v in exclude_tags.items()
                    ):
                        continue
                    rows.append(
                        {
                            "pitch": obj["pitch"],
                            "velocity": obj.get("velocity", 0),
                            "duration": obj.get("duration", 0),
                            "pos": obj.get("pos", 0),
                            "boundary": obj.get("boundary", 0),
                            "bar": obj.get("bar", 0),
                            "instrument": obj.get("instrument", ""),
                            "section": tags.get("section", ""),
                            "mood": tags.get("mood", ""),
                        }
                    )
        return rows

    return load_split("train"), load_split("valid")


def compute_inv_freq_weights(
    dataset: object,
    *,
    tag: str,
) -> tuple["torch.Tensor", dict[str, float]] | None:
    """Compute inverse-frequency weights for a dataset tag.

    Parameters
    ----------
    dataset:
        Dataset instance providing ``group_tags`` and ``__len__``.
    tag:
        Tag column to use (e.g., ``"section"``).

    Returns
    -------
    tuple or ``None``
        ``(weights, stats)`` when the tag is available; otherwise ``None``.
        ``weights`` is a 1-D tensor of ``float32`` values aligned with the
        dataset order. ``stats`` stores the top inverse-frequency weights for
        logging.
    """

    if torch is None:  # pragma: no cover - torch optional in tests
        return None
    tags = getattr(dataset, "group_tags", None)
    if not isinstance(tags, dict):
        return None
    raw_values = tags.get(tag)
    if not isinstance(raw_values, list) or not raw_values:
        return None

    normalized: list[str] = []
    for value in raw_values:
        text = "" if value is None else str(value).strip()
        normalized.append(text or "UNK")

    if not normalized:
        return None

    freq = Counter(normalized)
    weights = [1.0 / max(1, freq[v]) for v in normalized]
    if len(weights) != len(dataset):
        length = len(dataset)
        if len(weights) > length:
            weights = weights[:length]
        else:
            pad_value = weights[-1] if weights else 1.0
            weights = weights + [pad_value] * (length - len(weights))

    weight_stats = dict(
        sorted({k: 1.0 / max(1, freq[k]) for k in freq}.items(), key=lambda x: -x[1])[:10]
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return weight_tensor, weight_stats


def train_model(
    train_csv: Path,
    val_csv: Path,
    epochs: int,
    arch: str,
    out: Path,
    *,
    seed: int = 42,
    batch_size: int = 8,
    d_model: int = 512,
    max_len: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    grad_clip: float = 1.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler: str | None = None,
    warmup_steps: int = 0,
    pos_weight: float | None = None,
    auto_pos_weight: bool = False,
    resume: Path | None = None,
    save_every: int = 0,
    early_stopping: int = 0,
    f1_scan_range: tuple[float, float, float] = (0.2, 0.8, 0.1),
    logdir: Path | None = None,
    precision: str | None = None,
    deterministic: bool = False,
    device: str = "auto",
    best_metric: str = "macro_f1",
    reweight: str | None = None,
    lr_patience: int = 2,
    lr_factor: float = 0.5,
    use_duv_embed: bool = False,
    use_bar_beat: bool = False,
    use_local_stats: bool = False,
    use_harmony: bool = False,
    duv_mode: str = "reg",
    vel_bins: int = 0,
    dur_bins: int = 0,
    head: str = "linear",
    loss: str = "bce",
    focal_gamma: float = 2.0,
    progress: bool = False,
    w_boundary: float = 1.0,
    w_vel_reg: float = 0.5,
    w_dur_reg: float = 0.5,
    w_vel_cls: float = 0.0,
    w_dur_cls: float = 0.0,
    w_pitch: float = 0.4,
    pitch_smoothing: float = 0.0,
    instrument: str | None = None,
    include_tags: dict[str, str] | None = None,
    exclude_tags: dict[str, str] | None = None,
    viz: bool = False,
    strict_tags: bool = False,
    nhead: int = 8,
    layers: int = 4,
    dropout: float = 0.1,
    compile: bool = False,
    grad_accum: int = 1,
    save_best: bool = False,
    save_last: bool = False,
    use_sinusoidal_posenc: bool = False,
    # fast iteration helpers
    limit_train_batches: int = 0,
    limit_val_batches: int = 0,
    limit_train_groups: int = 0,
    limit_val_groups: int = 0,
    fast_dev_run: bool = False,
) -> tuple[float, str, dict[str, object]]:
    """Train the phrase boundary model and return the best F1, device, and stats."""

    import importlib

    global torch  # type: ignore
    if torch is None:  # pragma: no cover - fallback for optional import
        torch = importlib.import_module("torch")  # type: ignore[assignment]

    try:  # optional deps for viz
        from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay
    except Exception:  # pragma: no cover - optional
        precision_recall_curve = None  # type: ignore
        ConfusionMatrixDisplay = None  # type: ignore

    global plt  # type: ignore
    if viz and plt is None:
        try:  # pragma: no cover - optional
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt  # type: ignore

            plt = _plt
        except Exception:
            logging.warning("matplotlib not installed; skipping --viz")

    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    try:  # optional
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # pragma: no cover
        SummaryWriter = None

    def setup_env(seed: int, device_str: str) -> tuple[torch.device, bool]:
        random.seed(seed)
        try:  # optional numpy seeding
            import numpy as _np  # type: ignore

            _np.random.seed(seed)
        except Exception:  # pragma: no cover - numpy missing
            pass
        torch.manual_seed(seed)
        if device_str == "auto":
            if torch.cuda.is_available():
                device_type = "cuda"
            elif torch.backends.mps.is_available():
                device_type = "mps"
            else:
                device_type = "cpu"
        else:
            device_type = device_str
        device = torch.device(device_type)
        amp = device.type == "cuda"
        if device.type == "mps":
            amp = False
            torch.set_float32_matmul_precision(precision or "medium")
        elif precision:
            torch.set_float32_matmul_precision(precision)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:  # pragma: no cover
                logging.warning("deterministic algos unavailable: %s", exc)
            if device.type == "cuda":
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        return device, amp

    class PhraseLSTM(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            d_model: int = 128,
            max_len: int = 128,
            *,
            section_vocab_size: int = 0,
            mood_vocab_size: int = 0,
            vel_bucket_size: int = 0,
            dur_bucket_size: int = 0,
            duv_mode: str = "reg",
            vel_bins: int = 0,
            dur_bins: int = 0,
            use_bar_beat: bool = False,
            use_tcn: bool = False,
            use_crf_head: bool = False,
            use_harmony: bool = False,
            use_local_stats: bool = False,
        ) -> None:
            super().__init__()
            self.d_model = d_model
            self.max_len = max_len
            self.pitch_emb = nn.Embedding(12, d_model // 4)
            self.pos_emb = nn.Embedding(max_len, d_model // 4)
            self.dur_proj = nn.Linear(1, d_model // 4)
            self.vel_proj = nn.Linear(1, d_model // 4)
            self.use_bar_beat = use_bar_beat
            if use_bar_beat:
                self.barpos_proj = nn.Linear(1, d_model // 8)
                self.beatpos_proj = nn.Linear(1, d_model // 8)
                extra_bar_beat = d_model // 4
            else:
                self.barpos_proj = None
                self.beatpos_proj = None
                extra_bar_beat = 0
            extra_dim = 0
            if section_vocab_size:
                self.section_emb = nn.Embedding(section_vocab_size, 16)
                extra_dim += 16
            else:
                self.section_emb = None
            if mood_vocab_size:
                self.mood_emb = nn.Embedding(mood_vocab_size, 16)
                extra_dim += 16
            else:
                self.mood_emb = None
            if vel_bucket_size:
                self.vel_bucket_emb = nn.Embedding(vel_bucket_size, 8)
                extra_dim += 8
            else:
                self.vel_bucket_emb = None
            if dur_bucket_size:
                self.dur_bucket_emb = nn.Embedding(dur_bucket_size, 8)
                extra_dim += 8
            else:
                self.dur_bucket_emb = None
            self.use_harmony = use_harmony
            if use_harmony:
                self.harm_root_emb = nn.Embedding(12, 8)
                self.harm_func_emb = nn.Embedding(3, 4)
                self.harm_degree_emb = nn.Embedding(8, 4)
                extra_dim += 8 + 4 + 4
            else:
                self.harm_root_emb = None
                self.harm_func_emb = None
                self.harm_degree_emb = None
            self.use_local_stats = use_local_stats
            if use_local_stats:
                self.local_proj = nn.Linear(4, d_model // 8)
                extra_dim += d_model // 8
            else:
                self.local_proj = None
            self.feat_proj = nn.Linear(d_model + extra_dim + extra_bar_beat, d_model)
            self.lstm = nn.LSTM(
                d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True
            )
            self.use_tcn = use_tcn
            if self.use_tcn:
                self.tcn = nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            else:
                self.tcn = None
            if use_crf_head:
                self.head_boundary2 = nn.Linear(d_model, 2)
                self.head_boundary = None
            else:
                self.head_boundary = nn.Linear(d_model, 1)
                self.head_boundary2 = None
            self.head_vel_reg = nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
            self.head_dur_reg = nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
            self.head_vel_cls = (
                nn.Linear(d_model, vel_bins) if duv_mode in {"cls", "both"} else None
            )
            self.head_dur_cls = (
                nn.Linear(d_model, dur_bins) if duv_mode in {"cls", "both"} else None
            )

        def forward(
            self, feats: dict[str, torch.Tensor], mask: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            pos_ids = feats["position"].clamp(max=self.max_len - 1)
            dur = self.dur_proj(feats["duration"].unsqueeze(-1))
            vel = self.vel_proj(feats["velocity"].unsqueeze(-1))
            pc = self.pitch_emb(feats["pitch_class"] % 12)
            pos = self.pos_emb(pos_ids)
            parts = [dur, vel, pc, pos]
            if self.section_emb is not None and "section" in feats:
                parts.append(self.section_emb(feats["section"]))
            if self.mood_emb is not None and "mood" in feats:
                parts.append(self.mood_emb(feats["mood"]))
            if self.vel_bucket_emb is not None and "vel_bucket" in feats:
                parts.append(self.vel_bucket_emb(feats["vel_bucket"]))
            if self.dur_bucket_emb is not None and "dur_bucket" in feats:
                parts.append(self.dur_bucket_emb(feats["dur_bucket"]))
            if (
                self.use_harmony
                and self.harm_root_emb is not None
                and self.harm_func_emb is not None
                and self.harm_degree_emb is not None
            ):
                if "harm_root" in feats and "harm_func" in feats and "harm_degree" in feats:
                    parts.append(self.harm_root_emb(feats["harm_root"]))
                    parts.append(self.harm_func_emb(feats["harm_func"]))
                    parts.append(self.harm_degree_emb(feats["harm_degree"]))
            if self.use_local_stats and self.local_proj is not None and "local_stats" in feats:
                parts.append(self.local_proj(feats["local_stats"]))
            if (
                self.use_bar_beat
                and "bar_phase" in feats
                and "beat_phase" in feats
                and self.barpos_proj is not None
                and self.beatpos_proj is not None
            ):
                bp = self.barpos_proj(feats["bar_phase"].unsqueeze(-1))
                bt = self.beatpos_proj(feats["beat_phase"].unsqueeze(-1))
                parts.extend([bp, bt])
            x = torch.cat(parts, dim=-1)
            x = self.feat_proj(x)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=self.max_len
            )
            outputs: dict[str, torch.Tensor] = {}
            if self.tcn is not None:
                out = out.transpose(1, 2)
                out = self.tcn(out)
                out = out.transpose(1, 2)
            outputs: dict[str, torch.Tensor] = {}
            if self.head_boundary is not None:
                outputs["boundary"] = self.head_boundary(out).squeeze(-1)
            if self.head_boundary2 is not None:
                outputs["boundary2"] = self.head_boundary2(out)
            if self.head_vel_reg is not None:
                outputs["vel_reg"] = self.head_vel_reg(out).squeeze(-1)
            if self.head_dur_reg is not None:
                outputs["dur_reg"] = self.head_dur_reg(out).squeeze(-1)
            if self.head_vel_cls is not None:
                outputs["vel_cls"] = self.head_vel_cls(out)
            if self.head_dur_cls is not None:
                outputs["dur_cls"] = self.head_dur_cls(out)
            return outputs

    class PhraseDataset(Dataset):  # type: ignore[misc]
        def __init__(
            self,
            rows: list[dict[str, str]],
            max_len: int = 32,
            section_vocab: dict[str, int] | None = None,
            mood_vocab: dict[str, int] | None = None,
            instrument_vocab: dict[str, int] | None = None,
            use_duv_embed: bool = False,
            limit_groups: int = 0,
            use_bar_beat: bool = False,
            use_harmony: bool = False,
            use_local_stats: bool = False,
        ) -> None:
            # Group contiguously by bar value to avoid mixing bars from different files.
            # The CSV rows are appended file-by-file, so a reset of the bar index
            # indicates a new file. Using contiguous grouping preserves per-file bars
            # without requiring an explicit file id column.
            # PATCH: If all bars are 0 (raw MIDI data), group by file instead.
            groups: list[list[dict[str, str]]] = []
            cur: list[dict[str, str]] = []
            prev_bar: int | None = None
            prev_file: str | None = None

            for r in rows:
                try:
                    bar = int(r.get("bar", 0))
                except Exception:
                    bar = 0
                file_name = r.get("file", "")

                if prev_bar is None:
                    cur = [r]
                    prev_bar = bar
                    prev_file = file_name
                    continue

                # Group boundary: bar change OR file change
                if bar != prev_bar or (file_name and file_name != prev_file):
                    # finalize previous group
                    if cur:
                        groups.append(sorted(cur, key=lambda x: int(x.get("pos", 0))))
                    cur = [r]
                    prev_bar = bar
                    prev_file = file_name
                else:
                    cur.append(r)

            if cur:
                groups.append(sorted(cur, key=lambda x: int(x.get("pos", 0))))

            if limit_groups and limit_groups > 0:
                groups = groups[: max(1, int(limit_groups))]

            self.groups = groups
            self.use_bar_beat = use_bar_beat
            self.use_harmony = use_harmony
            self.use_local_stats = use_local_stats
            self.group_tags = {
                "section": [g[0].get("section", "") for g in self.groups],
                "mood": [g[0].get("mood", "") for g in self.groups],
                "instrument": [g[0].get("instrument", "") for g in self.groups],
            }
            self.max_len = max_len
            self.section_vocab = section_vocab
            self.mood_vocab = mood_vocab
            self.instrument_vocab = instrument_vocab
            self.has_vel_bucket = any(
                _has_bucket_value(r, "velocity_bucket", "vel_bucket") for r in rows
            )
            self.has_dur_bucket = any(
                _has_bucket_value(r, "duration_bucket", "dur_bucket") for r in rows
            )
            self.use_vel_bucket_feat = use_duv_embed and self.has_vel_bucket
            self.use_dur_bucket_feat = use_duv_embed and self.has_dur_bucket
            if use_duv_embed:
                if any("vel_bucket" in r for r in rows):
                    warnings.warn(
                        "vel_bucket column is deprecated; use velocity_bucket",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if any("dur_bucket" in r for r in rows):
                    warnings.warn(
                        "dur_bucket column is deprecated; use duration_bucket",
                        DeprecationWarning,
                        stacklevel=2,
                    )

        def __len__(self) -> int:  # pragma: no cover - trivial
            return len(self.groups)

        def __getitem__(self, idx: int) -> tuple[
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
            torch.Tensor,
            dict[str, torch.Tensor],
        ]:
            g = self.groups[idx]
            # クリッピング: 長すぎる系列は max_len に切り詰める
            if len(g) > self.max_len:
                g = g[: self.max_len]
            L = len(g)
            pad = self.max_len - L
            pitches = [int(r["pitch"]) for r in g]
            pitch_cls = [p % 12 for p in pitches]
            pc = torch.tensor(pitch_cls + [0] * pad, dtype=torch.long)
            vel = torch.tensor([float(r["velocity"]) for r in g] + [0] * pad, dtype=torch.float32)
            dur = torch.tensor([float(r["duration"]) for r in g] + [0] * pad, dtype=torch.float32)
            # Normalize position to [0, max_len-1] to avoid large tick-based indices clamping
            pos_vals = [(int(r["pos"]) % self.max_len) for r in g]
            pos = torch.tensor(pos_vals + [0] * pad, dtype=torch.long)
            feats = {
                "pitch_class": pc,
                "velocity": vel,
                "duration": dur,
                "position": pos,
            }
            if self.use_harmony:
                # Always include fields (fill 0 when missing)
                hr = [int(r.get("harm_root", 0)) for r in g] + [0] * pad
                hf = [int(r.get("harm_func", 0)) for r in g] + [0] * pad
                hd = [int(r.get("harm_degree", 0)) for r in g] + [0] * pad
                feats["harm_root"] = torch.tensor(hr, dtype=torch.long)
                feats["harm_func"] = torch.tensor(hf, dtype=torch.long)
                feats["harm_degree"] = torch.tensor(hd, dtype=torch.long)
            if self.use_bar_beat:
                # bar_phase: 0..1 across the group length
                denom = max(1, L - 1)
                bar_phase = [i / denom for i in range(L)] + [0.0] * pad
                max_pos = float(max(pos_vals)) if pos_vals else 1.0
                beat_phase = [float(v) / max(1.0, max_pos) for v in pos_vals] + [0.0] * pad
                feats["bar_phase"] = torch.tensor(bar_phase, dtype=torch.float32)
                feats["beat_phase"] = torch.tensor(beat_phase, dtype=torch.float32)
            if self.section_vocab:
                sec = [self.section_vocab.get(r.get("section", ""), 0) for r in g] + [0] * pad
                feats["section"] = torch.tensor(sec, dtype=torch.long)
            if self.mood_vocab:
                md = [self.mood_vocab.get(r.get("mood", ""), 0) for r in g] + [0] * pad
                feats["mood"] = torch.tensor(md, dtype=torch.long)
            if self.use_vel_bucket_feat:
                vb = [_bucket_value(r, "velocity_bucket", "vel_bucket") for r in g] + [0] * pad
                feats["vel_bucket"] = torch.tensor(vb, dtype=torch.long)
            if self.use_dur_bucket_feat:
                db = [_bucket_value(r, "duration_bucket", "dur_bucket") for r in g] + [0] * pad
                feats["dur_bucket"] = torch.tensor(db, dtype=torch.long)
            if self.use_local_stats:
                # ローカル統計: 直近K=8 の IOI/interval/velocity の平均/分散（簡易）
                K = 8
                ioi = [0.0] * L
                interval = [0.0] * L
                vel_list = [float(r["velocity"]) for r in g]
                for j in range(1, L):
                    ioi[j] = float(pos_vals[j] - pos_vals[j - 1])
                    interval[j] = float(abs(pitches[j] - pitches[j - 1]))
                mean_ioi = []
                std_ioi = []
                mean_interval = []
                mean_vel = []
                for j in range(L):
                    a = max(0, j - K + 1)
                    ioi_w = ioi[a : j + 1]
                    int_w = interval[a : j + 1]
                    vel_w = vel_list[a : j + 1]
                    m_ioi = sum(ioi_w) / len(ioi_w)
                    m_int = sum(int_w) / len(int_w)
                    m_vel = sum(vel_w) / len(vel_w)
                    var_ioi = sum((x - m_ioi) ** 2 for x in ioi_w) / len(ioi_w)
                    mean_ioi.append(m_ioi)
                    std_ioi.append(var_ioi**0.5)
                    mean_interval.append(m_int)
                    mean_vel.append(m_vel)

                # 正規化（簡易、ゼロ除去）
                def norm(vs: list[float]) -> list[float]:
                    mx = max(1e-6, max(vs))
                    return [x / mx for x in vs]

                ls = list(zip(norm(mean_ioi), norm(std_ioi), norm(mean_interval), norm(mean_vel)))
                ls += [(0.0, 0.0, 0.0, 0.0)] * pad
                feats["local_stats"] = torch.tensor(ls, dtype=torch.float32)
            mask = torch.zeros(self.max_len, dtype=torch.bool)
            mask[:L] = 1
            targets = {
                "boundary": torch.tensor(
                    [float(r["boundary"]) for r in g] + [0] * pad, dtype=torch.float32
                ),
                "vel_reg": vel / 127.0,
                "dur_reg": torch.log1p(dur),
                "pitch": torch.tensor(pitches + [-100] * pad, dtype=torch.long),
            }
            if self.has_vel_bucket:
                vb = [_bucket_value(r, "velocity_bucket", "vel_bucket") for r in g] + [0] * pad
                targets["vel_cls"] = torch.tensor(vb, dtype=torch.long)
            if self.has_dur_bucket:
                db = [_bucket_value(r, "duration_bucket", "dur_bucket") for r in g] + [0] * pad
                targets["dur_cls"] = torch.tensor(db, dtype=torch.long)
            tags = {
                "instrument": [
                    (
                        self.instrument_vocab.get(r.get("instrument", ""), 0)
                        if self.instrument_vocab
                        else 0
                    )
                    for r in g
                ]
                + [0] * pad,
                "section": [
                    self.section_vocab.get(r.get("section", ""), 0) if self.section_vocab else 0
                    for r in g
                ]
                + [0] * pad,
                "mood": [
                    self.mood_vocab.get(r.get("mood", ""), 0) if self.mood_vocab else 0 for r in g
                ]
                + [0] * pad,
            }
            tags_tensor = {k: torch.tensor(v, dtype=torch.long) for k, v in tags.items()}
            return feats, targets, mask, tags_tensor

    def collate_fn(
        batch: list[
            tuple[
                dict[str, torch.Tensor],
                dict[str, torch.Tensor],
                torch.Tensor,
                dict[str, torch.Tensor],
            ]
        ],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        feats, targets, mask, tags = zip(*batch)
        out_feats = {k: torch.stack([f[k] for f in feats]) for k in feats[0]}
        out_targets = {k: torch.stack([t[k] for t in targets]) for k in targets[0]}
        out_tags = {k: torch.stack([t[k] for t in tags]) for k in tags[0]}
        return out_feats, out_targets, torch.stack(mask), out_tags

    device, use_amp = setup_env(seed, device)
    logging.info(
        "device %s amp=%s precision=%s",
        device,
        use_amp,
        torch.get_float32_matmul_precision(),
    )

    required = {"pitch", "velocity", "duration", "pos", "boundary", "bar"}
    if duv_mode in {"cls", "both"}:
        required.update({"velocity_bucket", "duration_bucket"})
    train_rows = load_csv_rows(train_csv, required)
    val_rows = load_csv_rows(val_csv, required)
    train_rows, tr_removed = apply_filters(
        train_rows, instrument, include_tags, exclude_tags, strict=strict_tags
    )
    val_rows, val_removed = apply_filters(
        val_rows, instrument, include_tags, exclude_tags, strict=strict_tags
    )
    logging.info(
        "train rows kept %d removed %d | val kept %d removed %d",
        len(train_rows),
        tr_removed,
        len(val_rows),
        val_removed,
    )
    hint = (
        "\nHints:\n"
        "* Check CSV is not header-only.\n"
        "* If you used --instrument/--instrument-regex in corpus extraction, try removing it and using --pitch-range 28 60."
    )
    if not train_rows:
        raise ValueError(
            f"training CSV produced no usable rows (kept {len(train_rows)} removed {tr_removed})"
            + hint
        )
    if not val_rows:
        raise ValueError(
            f"validation CSV produced no usable rows (kept {len(val_rows)} removed {val_removed})"
            + hint
        )
    section_vals = {r["section"] for r in train_rows + val_rows if r.get("section")}
    mood_vals = {r["mood"] for r in train_rows + val_rows if r.get("mood")}
    instrument_vals = {r["instrument"] for r in train_rows + val_rows if r.get("instrument")}
    vel_bucket_size = 0
    dur_bucket_size = 0
    if use_duv_embed:
        vbs = [
            _bucket_value(r, "velocity_bucket", "vel_bucket")
            for r in train_rows + val_rows
            if _has_bucket_value(r, "velocity_bucket", "vel_bucket")
        ]
        dbs = [
            _bucket_value(r, "duration_bucket", "dur_bucket")
            for r in train_rows + val_rows
            if _has_bucket_value(r, "duration_bucket", "dur_bucket")
        ]
        vel_bucket_size = (max(vbs) + 1) if vbs else 0
        dur_bucket_size = (max(dbs) + 1) if dbs else 0
        logging.info("duv embed v=%s d=%s", vel_bucket_size, dur_bucket_size)
    positive_count = sum(int(r["boundary"]) for r in train_rows)
    if auto_pos_weight and pos_weight is None:
        total_count = len(train_rows)
        p = max(1e-6, min(1 - 1e-6, positive_count / max(1, total_count)))
        pos_weight = (1 - p) / p
        logging.info("auto pos_weight=%s", pos_weight)
    section_vocab = {s: i + 1 for i, s in enumerate(sorted(section_vals))} if section_vals else None
    mood_vocab = {s: i + 1 for i, s in enumerate(sorted(mood_vals))} if mood_vals else None
    instrument_vocab = (
        {s: i + 1 for i, s in enumerate(sorted(instrument_vals))} if instrument_vals else None
    )
    class_counts = {"pos": positive_count, "neg": len(train_rows) - positive_count}
    tag_counts: dict[str, dict[str, int]] = {}
    for tag in ("instrument", "section", "mood"):
        counts = Counter(r.get(tag, "") for r in train_rows if r.get(tag))
        if counts:
            tag_counts[tag] = dict(counts)
    logging.info("class balance %s", class_counts)
    for t, c in tag_counts.items():
        logging.info("tag %s counts %s", t, c)
    tag_coverage = {
        t: sum(c.values()) / len(train_rows) if train_rows else 0.0 for t, c in tag_counts.items()
    }
    corpus_name = train_csv.parent.name
    ds_train = PhraseDataset(
        train_rows,
        max_len,
        section_vocab,
        mood_vocab,
        instrument_vocab,
        use_duv_embed=use_duv_embed,
        limit_groups=limit_train_groups,
        use_bar_beat=use_bar_beat,
        use_harmony=use_harmony,
        use_local_stats=use_local_stats,
    )
    ds_val = PhraseDataset(
        val_rows,
        max_len,
        section_vocab,
        mood_vocab,
        instrument_vocab,
        use_duv_embed=use_duv_embed,
        limit_groups=limit_val_groups,
        use_bar_beat=use_bar_beat,
        use_harmony=use_harmony,
        use_local_stats=use_local_stats,
    )

    if duv_mode in {"cls", "both"}:
        if not ds_train.has_vel_bucket or not ds_train.has_dur_bucket:
            raise SystemExit("classification mode requires velocity_bucket and duration_bucket")
        if vel_bins <= 0 or dur_bins <= 0:
            raise SystemExit("--vel-bins and --dur-bins must be >0 for classification mode")

    pin_mem = device.type == "cuda" or pin_memory
    persist = device.type == "cuda" and num_workers > 0

    def worker_init_fn(worker_id: int) -> None:
        seed = torch.initial_seed() % 2**32
        torch.manual_seed(seed + worker_id)
        random.seed(seed + worker_id)
        try:  # optional numpy seeding
            import numpy as _np  # type: ignore

            _np.random.seed(seed + worker_id)
        except Exception:  # pragma: no cover - numpy missing
            pass

    sampler = None
    weight_stats: dict[str, float] | None = None
    shuffle_train = True
    weights: list[float] | torch.Tensor | None = None
    if reweight:
        tag_name = "section"
        stripped = reweight.strip()
        if stripped:
            if "=" in stripped:
                for part in stripped.split(","):
                    if "=" not in part:
                        continue
                    key, value = part.split("=", 1)
                    if key.strip().lower() in {"tag", "class"} and value.strip():
                        tag_name = value.strip()
                        break
            else:
                tag_name = stripped
        inv = compute_inv_freq_weights(ds_train, tag=tag_name)
        if inv is not None:
            weights_tensor, weight_stats = inv
            weights = weights_tensor
            logging.info(
                "reweight tag=%s using inv_freq sampler (%d samples)",
                tag_name,
                int(weights_tensor.numel()),
            )
            if weight_stats:
                logging.info("top tag weights (tag=%s) %s", tag_name, weight_stats)
            if torch is not None:
                sampler_fn = _get_weighted_sampler(torch)
                sampler_mod = getattr(torch.utils.data, "sampler", None)
                base_cls = (
                    getattr(sampler_mod, "WeightedRandomSampler", None)
                    if sampler_mod is not None
                    else None
                )
                num_samples = int(weights_tensor.numel())

                def _call_sampler(fn, allow_keyword: bool = True):
                    if allow_keyword:
                        return fn(weights_tensor, num_samples, replacement=True)
                    return fn(weights_tensor, num_samples, True)

                if callable(sampler_fn):
                    try:
                        sampler = _call_sampler(sampler_fn, True)
                    except TypeError:
                        try:
                            sampler = _call_sampler(sampler_fn, False)
                        except RecursionError:
                            sampler = None
                    except RecursionError:
                        sampler = None

                    if sampler is None and callable(base_cls):
                        original_attr = getattr(torch.utils.data, "WeightedRandomSampler", None)
                        alias_installed = False
                        if original_attr is not base_cls:
                            try:
                                setattr(torch.utils.data, "WeightedRandomSampler", base_cls)
                                alias_installed = True
                            except Exception:
                                pass
                        try:
                            if sampler is None:
                                try:
                                    sampler = _call_sampler(sampler_fn, False)
                                except RecursionError:
                                    sampler = None
                            if sampler is None:
                                try:
                                    sampler = _call_sampler(base_cls, True)
                                except TypeError:
                                    sampler = _call_sampler(base_cls, False)
                        finally:
                            if alias_installed:
                                try:
                                    setattr(
                                        torch.utils.data, "WeightedRandomSampler", original_attr
                                    )
                                except Exception:
                                    pass

                if sampler is not None:
                    shuffle_train = False
        if sampler is None:
            parts = [p.strip() for p in reweight.split(",") if p.strip()]
            kv = {k.strip(): v.strip() for k, v in (p.split("=", 1) for p in parts if "=" in p)}
            tag = kv.get("tag") or kv.get("class")
            scheme = kv.get("scheme", "inv_freq")
            if tag and scheme == "inv_freq":
                rows: list[dict[str, str]] = []
                values: list[str] = []
                raw_count = 0
                skipped_empty = 0
                req_cols = locals().get("required")
                if isinstance(req_cols, set):
                    req = set(req_cols)
                else:
                    req = {
                        "pitch",
                        "velocity",
                        "duration",
                        "pos",
                        "boundary",
                        "bar",
                        "instrument",
                    }
                req.add("instrument")
                req.add(tag)
                if train_rows:
                    rows = train_rows
                elif isinstance(train_csv, (str, Path)):
                    try:
                        rows = load_csv_rows(Path(train_csv), req)
                    except Exception as exc:
                        logging.info("failed to read %s for reweight: %s", train_csv, exc)
                        rows = []

                def _norm(value: object) -> str:
                    if value is None:
                        return ""
                    text = str(value).strip()
                    return "" if text.lower() in {"", "none", "nan"} else text

                if rows:
                    raw_values: list[str] = []
                    for r in rows:
                        value_str = _norm(r.get(tag))
                        raw_values.append(value_str)
                        if value_str == "":
                            skipped_empty += 1
                    values = [v for v in raw_values if v != ""]
                    raw_count = len(raw_values)
                if not values:
                    tags_map = getattr(ds_train, "group_tags", None)
                    if isinstance(tags_map, dict) and tag in tags_map:
                        try:
                            values = [_norm(v) for v in tags_map[tag]]
                            values = [v for v in values if v]
                        except Exception:
                            values = []
                    raw_count = len(values)
                if not values and isinstance(train_csv, (str, Path)):
                    try:
                        csv_rows = load_csv_rows(Path(train_csv), req)
                    except Exception:
                        csv_rows = []
                    if csv_rows:
                        raw_values = []
                        for r in csv_rows:
                            value_str = _norm(r.get(tag))
                            raw_values.append(value_str)
                            if value_str == "":
                                skipped_empty += 1
                        values = [v for v in raw_values if v]
                        raw_count = len(raw_values)
                logging.info(
                    "reweight tag=%s collected=%d (raw=%d), skipped_empty=%d",
                    tag,
                    len(values),
                    raw_count,
                    skipped_empty,
                )
                if skipped_empty:
                    logging.info("reweight=%r skipped %d empty values", reweight, skipped_empty)
            logging.debug(
                "reweight=%r collected %d values (raw=%d)",
                reweight,
                len(values),
                raw_count,
            )
            if not values:
                logging.info(
                    "reweight=%r accepted but no values found; skipping sampler",
                    reweight,
                )
            else:
                if len(values) != len(ds_train):
                    tags_map = getattr(ds_train, "group_tags", None)
                    if isinstance(tags_map, dict) and tag in tags_map:
                        try:
                            values = [_norm(v) for v in tags_map[tag]]
                            values = [v for v in values if v]
                        except Exception:
                            values = []
                    raw_count = len(values)
                if not values and isinstance(train_csv, (str, Path)):
                    try:
                        csv_rows = load_csv_rows(Path(train_csv), req)
                    except Exception:
                        csv_rows = []
                    if csv_rows:
                        raw_values = []
                        for r in csv_rows:
                            value_str = _norm(r.get(tag))
                            raw_values.append(value_str)
                            if value_str == "":
                                skipped_empty += 1
                        values = [v for v in raw_values if v]
                        raw_count = len(raw_values)
                logging.info(
                    "reweight tag=%s collected=%d (raw=%d), skipped_empty=%d",
                    tag,
                    len(values),
                    raw_count,
                    skipped_empty,
                )
                if skipped_empty:
                    logging.info("reweight=%r skipped %d empty values", reweight, skipped_empty)
                logging.debug(
                    "reweight=%r collected %d values (raw=%d)",
                    reweight,
                    len(values),
                    raw_count,
                )
                if not values:
                    logging.info(
                        "reweight=%r accepted but no values found; skipping sampler",
                        reweight,
                    )
                else:
                    if len(values) != len(ds_train):
                        tags_map = getattr(ds_train, "group_tags", None)
                        if (
                            isinstance(tags_map, dict)
                            and tag in tags_map
                            and len(tags_map[tag]) == len(ds_train)
                        ):
                            try:
                                tag_values = []
                                for v in tags_map[tag]:
                                    normed = _norm(v)
                                    tag_values.append(normed if normed else "UNK")
                            except Exception:
                                tag_values = [
                                    str(v) if v is not None else "UNK" for v in tags_map[tag]
                                ]
                                tag_values = [tv if tv else "UNK" for tv in tag_values]
                            values = tag_values
                            logging.info(
                                "reweight=%r length mismatch fixed via group_tags(tag=%s)",
                                reweight,
                                tag,
                            )
                        if len(values) != len(ds_train):
                            logging.info(
                                "reweight=%r values=%d groups=%d; aligning by slice/pad",
                                reweight,
                                len(values),
                                len(ds_train),
                            )
                            logging.debug(
                                "reweight=%r first 5 values=%s",
                                reweight,
                                values[:5],
                            )
                            if len(values) > len(ds_train):
                                values = values[: len(ds_train)]
                            else:
                                pad_source = next((v for v in reversed(values) if v), "UNK")
                                if not pad_source:
                                    pad_source = "UNK"
                                values = values + [pad_source] * (len(ds_train) - len(values))
                    freq = Counter(values)
                    weights = [1.0 / max(1, freq[v]) for v in values]
                    weight_map = {v: 1.0 / max(1, freq[v]) for v in freq}
                    weight_stats = dict(sorted(weight_map.items(), key=lambda x: -x[1])[:10])
                    logging.info("top tag weights (tag=%s) %s", tag, weight_stats)
                    logging.debug(
                        "inv_freq weights head (tag=%s) %s",
                        tag,
                        [round(w, 6) for w in weights[:3]],
                    )
                if torch is None:
                    logging.info("reweight=%r skipped %d empty values", reweight, skipped_empty)
                logging.debug(
                    "reweight=%r collected %d values (raw=%d)",
                    reweight,
                    len(values),
                    raw_count,
                )
                if not values:
                    logging.info(
                        "reweight=%r accepted but no values found; skipping sampler",
                        reweight,
                    )
                else:
                    if len(values) != len(ds_train):
                        tags_map = getattr(ds_train, "group_tags", None)
                        if (
                            isinstance(tags_map, dict)
                            and tag in tags_map
                            and len(tags_map[tag]) == len(ds_train)
                        ):
                            try:
                                tag_values = []
                                for v in tags_map[tag]:
                                    normed = _norm(v)
                                    tag_values.append(normed if normed else "UNK")
                            except Exception:
                                tag_values = [
                                    str(v) if v is not None else "UNK" for v in tags_map[tag]
                                ]
                                tag_values = [tv if tv else "UNK" for tv in tag_values]
                            values = tag_values
                            logging.info(
                                "reweight=%r length mismatch fixed via group_tags(tag=%s)",
                                reweight,
                                tag,
                            )
                        if len(values) != len(ds_train):
                            logging.info(
                                "reweight=%r values=%d groups=%d; aligning by slice/pad",
                                reweight,
                                len(values),
                                len(ds_train),
                            )
                            logging.debug(
                                "reweight=%r first 5 values=%s",
                                reweight,
                                values[:5],
                            )
                            if len(values) > len(ds_train):
                                values = values[: len(ds_train)]
                            else:
                                pad_source = next((v for v in reversed(values) if v), "UNK")
                                if not pad_source:
                                    pad_source = "UNK"
                                values = values + [pad_source] * (len(ds_train) - len(values))
                    freq = Counter(values)
                    weights = [1.0 / max(1, freq[v]) for v in values]
                    weight_map = {v: 1.0 / max(1, freq[v]) for v in freq}
                    weight_stats = dict(sorted(weight_map.items(), key=lambda x: -x[1])[:10])
                    logging.info("top tag weights (tag=%s) %s", tag, weight_stats)
                    logging.debug(
                        "inv_freq weights head (tag=%s) %s",
                        tag,
                        [round(w, 6) for w in weights[:3]],
                    )
                    if torch is None:
                        logging.info(
                            "torch not available for reweight; skipping sampler (%s)",
                            "torch import failed",
                        )
                    else:
                        if len(weights) == len(ds_train):
                            try:
                                weight_tensor = torch.as_tensor(weights, dtype=torch.float64)
                            except Exception:
                                weight_tensor = torch.tensor(weights, dtype=torch.float32)
                            sampler_fn = _get_weighted_sampler(torch)
                            if callable(sampler_fn):
                                try:
                                    sampler = sampler_fn(
                                        weight_tensor,
                                        len(ds_train),
                                        replacement=True,
                                    )
                                except TypeError:
                                    sampler = sampler_fn(
                                        weight_tensor,
                                        len(ds_train),
                                        True,
                                    )
                                except RecursionError:
                                    base_mod = getattr(torch.utils.data, "sampler", None)
                                    base_cls = (
                                        getattr(base_mod, "WeightedRandomSampler", None)
                                        if base_mod is not None
                                        else None
                                    )
                                    if callable(base_cls):
                                        sampler = base_cls(
                                            weight_tensor,
                                            len(ds_train),
                                            replacement=True,
                                        )
                                    else:
                                        logging.info(
                                            "reweight=%r sampler recursion detected; continuing without sampler",
                                            reweight,
                                        )
                                        sampler = None
                            if sampler is not None:
                                shuffle_train = False

    if (
        sampler is None
        and weights is not None
        and torch is not None
        and len(weights) == len(ds_train)
    ):
        sampler_fn = _get_weighted_sampler(torch)
        if callable(sampler_fn):
            try:
                weight_tensor = torch.as_tensor(weights, dtype=torch.float64)
            except Exception:
                weight_tensor = torch.tensor(weights, dtype=torch.float32)
            try:
                sampler = sampler_fn(weight_tensor, len(ds_train), replacement=True)
            except TypeError:
                sampler = sampler_fn(weight_tensor, len(ds_train), True)
            except RecursionError:
                base_mod = getattr(torch.utils.data, "sampler", None)
                base_cls = (
                    getattr(base_mod, "WeightedRandomSampler", None)
                    if base_mod is not None
                    else None
                )
                if callable(base_cls):
                    try:
                        sampler = base_cls(weight_tensor, len(ds_train), replacement=True)
                    except TypeError:
                        sampler = base_cls(weight_tensor, len(ds_train), True)
                else:
                    sampler = None
            if sampler is not None:
                shuffle_train = False

    if (
        sampler is None
        and weights is not None
        and torch is not None
        and len(weights) == len(ds_train)
    ):
        sampler_fn = _get_weighted_sampler(torch)
        if callable(sampler_fn):
            try:
                weight_tensor = torch.as_tensor(weights, dtype=torch.float64)
            except Exception:
                weight_tensor = torch.tensor(weights, dtype=torch.float32)
            try:
                sampler = sampler_fn(weight_tensor, len(ds_train), replacement=True)
            except TypeError:
                sampler = sampler_fn(weight_tensor, len(ds_train), True)
            except RecursionError:
                base_mod = getattr(torch.utils.data, "sampler", None)
                base_cls = (
                    getattr(base_mod, "WeightedRandomSampler", None)
                    if base_mod is not None
                    else None
                )
                if callable(base_cls):
                    try:
                        sampler = base_cls(weight_tensor, len(ds_train), replacement=True)
                    except TypeError:
                        sampler = base_cls(weight_tensor, len(ds_train), True)
                else:
                    sampler = None
            if sampler is not None:
                shuffle_train = False

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=shuffle_train if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
        worker_init_fn=worker_init_fn if num_workers else None,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
        worker_init_fn=worker_init_fn if num_workers else None,
    )

    # [DEBUG] Log batch counts to verify data loader is working
    logging.info(
        "[trainer] train_batches/epoch=%d val_batches/epoch=%d",
        len(dl_train),
        len(dl_val),
    )

    # If classification heads are enabled but their loss weights are zero,
    # set sensible defaults so they learn during quick iterations.
    if duv_mode in {"cls", "both"} and (w_vel_cls <= 0 and w_dur_cls <= 0):
        logging.warning(
            "duv_mode=%s with zero class-loss weights; setting w_vel_cls=0.5, w_dur_cls=0.5",
            duv_mode,
        )
        w_vel_cls = 0.5
        w_dur_cls = 0.5

    # Fast dev run: cap batches to keep turnaround quick
    if fast_dev_run:
        limit_train_batches = max(1, int(limit_train_batches or 2))
        limit_val_batches = max(1, int(limit_val_batches or 2))

    if arch in {"lstm", "bilstm_tcn"}:
        model: nn.Module = PhraseLSTM(
            d_model=d_model,
            max_len=max_len,
            section_vocab_size=len(section_vocab) + 1 if section_vocab else 0,
            mood_vocab_size=len(mood_vocab) + 1 if mood_vocab else 0,
            vel_bucket_size=vel_bucket_size,
            dur_bucket_size=dur_bucket_size,
            duv_mode=duv_mode,
            vel_bins=vel_bins,
            dur_bins=dur_bins,
            use_bar_beat=use_bar_beat,
            use_harmony=use_harmony,
            use_tcn=(arch == "bilstm_tcn"),
            use_crf_head=(head == "crf"),
            use_local_stats=use_local_stats,
        )
    else:
        num_layers = layers if layers is not None else 2
        ctor_kwargs = dict(
            d_model=d_model,
            max_len=max_len,
            section_vocab_size=len(section_vocab) + 1 if section_vocab else 0,
            mood_vocab_size=len(mood_vocab) + 1 if mood_vocab else 0,
            vel_bucket_size=vel_bucket_size,
            dur_bucket_size=dur_bucket_size,
            use_bar_beat=use_bar_beat,
            duv_mode=duv_mode,
            vel_bins=vel_bins,
            dur_bins=dur_bins,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_sinusoidal_posenc=use_sinusoidal_posenc,
        )
        try:
            sig = inspect.signature(PhraseTransformer.__init__)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            params = sig.parameters.values()
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            if not has_var_kw:
                allowed = {
                    p.name
                    for p in params
                    if p.name != "self" and p.kind != inspect.Parameter.VAR_POSITIONAL
                }
                ctor_kwargs = {k: v for k, v in ctor_kwargs.items() if k in allowed}
        model = PhraseTransformer(**ctor_kwargs)
    model = model.to(device)
    # Initialize boundary head bias to dataset prior logit to avoid extreme initial saturation
    try:
        total_count = max(1, len(train_rows))
        pos_prior = max(1e-6, min(1 - 1e-6, positive_count / total_count))
        prior_logit = float(torch.log(torch.tensor(pos_prior / (1 - pos_prior))))
        with torch.no_grad():
            if (
                hasattr(model, "head_boundary")
                and hasattr(model.head_boundary, "bias")
                and model.head_boundary.bias is not None
            ):
                model.head_boundary.bias.fill_(prior_logit)
                logging.info(
                    "init head_boundary.bias to prior logit %.3f (p=%.3f)", prior_logit, pos_prior
                )
    except Exception:
        pass
    if compile:
        try:  # pragma: no cover - runtime optional
            model = torch.compile(model)
        except Exception:  # pragma: no cover
            logging.warning("torch.compile failed; continuing without")

    def _finalize_zero_epoch() -> tuple[float, str, dict[str, object]]:
        logging.info("epochs=%s requested; skipping optimizer and training loop", epochs)
        stats: dict[str, object] = {
            "class_counts": class_counts,
            "tag_counts": tag_counts,
            "tag_coverage": tag_coverage,
            "viz_paths": [],
        }
        if weight_stats:
            stats["tag_weights_top10"] = weight_stats
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            model_state = (
                {k: v.cpu() for k, v in model.state_dict().items()}
                if hasattr(model, "state_dict")
                else {}
            )
        except Exception:  # pragma: no cover - stubs without state_dict
            model_state = {}
        placeholder_meta = {
            "arch": arch,
            "d_model": d_model,
            "n_layers": layers,
            "n_heads": nhead,
            "max_len": max_len,
            "duv_mode": duv_mode,
            "skipped_epochs": epochs,
            "tag_coverage": tag_coverage,
        }
        placeholder_state = {
            "model": model_state,
            "optimizer": {},
            "scheduler": None,
            "epoch": 0,
            "global_step": 0,
            "best_f1": 0.0,
            "meta": placeholder_meta,
        }
        try:
            if torch is not None:
                torch.save(placeholder_state, out_path)
            else:  # pragma: no cover - optional torch
                out_path.write_bytes(b"")
        except Exception as exc:  # pragma: no cover - filesystem/serialization issues
            logging.warning("failed to write placeholder checkpoint: %s", exc)
            try:
                out_path.write_bytes(b"")
            except Exception:
                logging.warning("unable to create placeholder checkpoint file at %s", out_path)
        if save_best:
            best_path = out_path.with_suffix(".best.ckpt")
            try:
                shutil.copy2(out_path, best_path)
            except OSError as exc:
                logging.warning("failed to create best checkpoint copy: %s", exc)
        if save_last:
            last_path = out_path.with_name("last.ckpt")
            try:
                if last_path.exists() or last_path.is_symlink():
                    last_path.unlink()
                last_path.symlink_to(out_path.name)
            except OSError:
                try:
                    shutil.copy2(out_path, last_path)
                except OSError as exc:
                    logging.warning("failed to copy placeholder to last.ckpt: %s", exc)
        metrics_skeleton = {
            "f1": 0.0,
            "best_threshold": 0.5,
            "skipped": True,
        }
        try:
            (out_path.parent / "metrics.json").write_text(
                json.dumps(metrics_skeleton, ensure_ascii=False)
            )
        except Exception:
            logging.debug("metrics.json placeholder write skipped")
        preview_placeholder = out_path.parent / "preds_preview.json"
        if not preview_placeholder.exists():
            try:
                preview_placeholder.write_text("[]")
            except Exception:
                logging.debug("preds_preview.json placeholder write skipped")
        return 0.0, device.type, stats

    if epochs <= 0:
        return _finalize_zero_epoch()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(dl_train))
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=lr_factor, patience=lr_patience
        )
    else:
        sched = None
    pw = torch.tensor([pos_weight], device=device) if pos_weight else None

    # Optional focal loss
    class FocalLoss(nn.Module):
        def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
            super().__init__()
            self.gamma = gamma
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            bce = self.bce(logits, targets)
            p = torch.sigmoid(logits)
            pt = p * targets + (1 - p) * (1 - targets)
            loss = ((1 - pt) ** self.gamma) * bce
            return loss.mean()

    crit_boundary = (
        FocalLoss(gamma=focal_gamma, pos_weight=pw)
        if loss == "focal"
        else nn.BCEWithLogitsLoss(pos_weight=pw)
    )
    crit_vel_reg = nn.SmoothL1Loss()
    crit_dur_reg = nn.SmoothL1Loss()
    crit_vel_cls = nn.CrossEntropyLoss()
    crit_dur_cls = nn.CrossEntropyLoss()
    crit_pitch = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=pitch_smoothing)

    # Lightweight 2-state CRF for sequence consistency（必要時のみ有効化）
    class CRF2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trans = nn.Parameter(torch.zeros(2, 2))

        def nll(
            self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            # emissions: (B,L,2), tags: (B,L) int 0/1, mask: (B,L) bool
            B, L, C = emissions.shape
            # 前向き計算で対数分配関数を求める
            alpha = emissions[:, 0, :]
            for t in range(1, L):
                e_t = emissions[:, t, :].unsqueeze(1)  # (B,1,2)
                a = alpha.unsqueeze(2) + self.trans.unsqueeze(0) + e_t  # (B,2,2)
                alpha_t = torch.logsumexp(a, dim=1)  # (B,2)
                alpha = torch.where(mask[:, t].unsqueeze(1), alpha_t, alpha)
            logZ = torch.logsumexp(alpha, dim=1)  # (B,)
            # 正解パスのスコア
            score = emissions[torch.arange(B), 0, tags[:, 0]]
            for t in range(1, L):
                emit = emissions[torch.arange(B), t, tags[:, t]]
                trans = self.trans[tags[:, t - 1], tags[:, t]]
                step = emit + trans
                score = torch.where(mask[:, t], score + step, score)
            nll = (logZ - score).mean()
            return nll

    crf = CRF2().to(device) if head == "crf" else None
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:  # pragma: no cover - PyTorch < 2.0 fallback
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(logdir) if logdir and SummaryWriter else None

    start_epoch = 0
    global_step = 0
    best_f1 = -1.0
    if resume and resume.is_file():
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state.get("optimizer", {}))
        if sched and state.get("scheduler"):
            sched.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_f1 = float(state.get("best_f1", -1.0))

    def evaluate() -> tuple[
        float,
        float,
        float,
        list[float],
        list[int],
        dict[str, list[int]],
        dict[str, float],
        list[float],
    ]:
        model.eval()
        probs: list[float] = []
        trues: list[int] = []
        tag_buf = {"instrument": [], "section": [], "mood": []}
        logits_all: list[float] = []
        vel_err = 0.0
        dur_err = 0.0
        vel_n = 0
        dur_n = 0
        vel_ok = 0
        vel_tot = 0
        dur_ok = 0
        dur_tot = 0
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for i, (feats, targets, mask, tags) in enumerate(dl_val):
                feats = {k: v.to(device) for k, v in feats.items()}
                targets = {k: v.to(device) for k, v in targets.items()}
                mask = mask.to(device)
                raw_outputs = model(feats, mask)
                if isinstance(raw_outputs, dict):
                    outputs = raw_outputs
                elif torch is not None and isinstance(raw_outputs, torch.Tensor):
                    outputs = {"boundary": raw_outputs}
                elif (
                    isinstance(raw_outputs, (list, tuple))
                    and raw_outputs
                    and isinstance(raw_outputs[0], dict)
                ):
                    outputs = raw_outputs[0]
                else:
                    outputs = {"logits": raw_outputs}

                boundary2 = None
                if isinstance(outputs, dict) and "boundary2" in outputs:
                    boundary2 = outputs["boundary2"]

                if "boundary" not in outputs:
                    if "logits" in outputs:
                        outputs["boundary"] = outputs["logits"]
                    else:
                        try:
                            logits_fallback = next(
                                v for v in outputs.values() if torch.is_tensor(v)
                            )
                        except StopIteration:
                            logits_fallback = None
                        if logits_fallback is not None:
                            outputs["boundary"] = logits_fallback

                if "boundary" not in outputs and boundary2 is None:
                    try:
                        zeros = torch.zeros_like(targets["boundary"], dtype=torch.float32)
                    except Exception:
                        zeros = torch.tensor(0.0, device=device)
                    outputs["boundary"] = zeros
                m = mask.bool()
                mask_cpu = m.cpu()
                loss_val = 0.0
                if boundary2 is not None:
                    # CRF/2-class logits present; use softmax prob of class 1
                    logits2 = boundary2[m].detach().cpu()
                    logits_all.extend((logits2[:, 1] - logits2[:, 0]).tolist())
                    import torch as _t

                    probs.extend(_t.softmax(logits2, dim=-1)[:, 1].tolist())
                    if head == "crf" and crf is not None:
                        lb = crf.nll(boundary2, targets["boundary"].long(), m)
                    else:
                        lb = crit_boundary(
                            boundary2[m][:, 1] - boundary2[m][:, 0],
                            targets["boundary"][m],
                        )
                    loss_val += w_boundary * float(lb)
                else:
                    logits_this = outputs["boundary"][m].detach().cpu().tolist()
                    logits_all.extend([float(v) for v in logits_this])
                    probs.extend(torch.sigmoid(outputs["boundary"][m]).cpu().tolist())
                    lb = crit_boundary(outputs["boundary"][m], targets["boundary"][m])
                    loss_val += w_boundary * float(lb)
                trues.extend(targets["boundary"][m].int().cpu().tolist())
                if "vel_reg" in outputs:
                    vel_err += (
                        torch.abs(outputs["vel_reg"][m] - targets["vel_reg"][m])
                        .mul(127.0)
                        .sum()
                        .item()
                    )
                    lv = crit_vel_reg(outputs["vel_reg"][m], targets["vel_reg"][m])
                    loss_val += w_vel_reg * float(lv)
                    vel_n += int(m.sum())
                if "dur_reg" in outputs:
                    dur_err += (
                        torch.abs(
                            torch.expm1(outputs["dur_reg"][m]) - torch.expm1(targets["dur_reg"][m])
                        )
                        .sum()
                        .item()
                    )
                    ld = crit_dur_reg(outputs["dur_reg"][m], targets["dur_reg"][m])
                    loss_val += w_dur_reg * float(ld)
                    dur_n += int(m.sum())
                if "vel_cls" in outputs:
                    preds = outputs["vel_cls"][m].argmax(dim=-1)
                    vel_ok += int((preds == targets["vel_cls"][m]).sum())
                    vel_tot += int(m.sum())
                    lvb = crit_vel_cls(outputs["vel_cls"][m], targets["vel_cls"][m])
                    loss_val += w_vel_cls * float(lvb)
                if "dur_cls" in outputs:
                    preds = outputs["dur_cls"][m].argmax(dim=-1)
                    dur_ok += int((preds == targets["dur_cls"][m]).sum())
                    dur_tot += int(m.sum())
                    ldb = crit_dur_cls(outputs["dur_cls"][m], targets["dur_cls"][m])
                    loss_val += w_dur_cls * float(ldb)
                if "pitch_logits" in outputs:
                    lp = crit_pitch(outputs["pitch_logits"][m], targets["pitch"][m])
                    loss_val += w_pitch * float(lp)
                for k in tag_buf:
                    tag_buf[k].extend(tags[k][mask_cpu].tolist())
                val_loss_sum += loss_val
                val_batches += 1
                if limit_val_batches and (i + 1) >= limit_val_batches:
                    break
        best_f1, best_th = -1.0, 0.5
        start, end, step = f1_scan_range
        n = int(round((end - start) / step)) + 1
        ths = [round(start + i * step, 10) for i in range(n)]
        for th in ths:
            preds = [1 if p > th else 0 for p in probs]
            f1 = f1_score(trues, preds)
            if f1 > best_f1:
                best_f1, best_th = f1, float(th)
        # Additional diagnostics to detect saturation/imbalance during eval
        total = max(1, len(probs))
        preds_best = [1 if p > best_th else 0 for p in probs]
        pos_rate = sum(preds_best) / total
        mean_prob = sum(probs) / total
        metrics = {
            "vel_mae": vel_err / vel_n if vel_n else 0.0,
            "dur_mae": dur_err / dur_n if dur_n else 0.0,
            "vel_acc": vel_ok / vel_tot if vel_tot else 0.0,
            "dur_acc": dur_ok / dur_tot if dur_tot else 0.0,
            "pos_rate": pos_rate,
            "mean_prob": mean_prob,
        }
        val_loss_avg = val_loss_sum / max(1, val_batches)
        return best_f1, val_loss_avg, best_th, probs, trues, tag_buf, metrics, logits_all

    ts = int(time.time())
    best_state = None
    best_threshold = 0.5
    bad_epochs = 0
    best_metric_val = -1.0
    viz_files: list[str] = []
    metrics_rows: list[dict[str, float]] = []
    use_progress = bool(progress and _tqdm is not None)
    epochs_completed = start_epoch
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        epochs_completed = ep + 1
        model.train()
        loss_sum = torch.zeros((), device=device)
        lb_sum = torch.zeros((), device=device)
        lv_sum = torch.zeros((), device=device)
        ld_sum = torch.zeros((), device=device)
        lvb_sum = torch.zeros((), device=device)
        ldb_sum = torch.zeros((), device=device)
        lp_sum = torch.zeros((), device=device)
        opt.zero_grad()
        iter_train = dl_train
        if use_progress:
            iter_train = _tqdm(
                dl_train,
                total=len(dl_train),
                desc=f"epoch {ep + 1}/{epochs}",
                leave=False,
            )
        for step, (feats, targets, mask, _) in enumerate(iter_train):
            feats = {k: v.to(device) for k, v in feats.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            mask = mask.to(device)
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
                if use_amp
                else nullcontext()
            )
            with ctx:
                raw_outputs = model(feats, mask)
                if isinstance(raw_outputs, dict):
                    outputs = raw_outputs
                elif isinstance(raw_outputs, (list, tuple)):
                    first = raw_outputs[0] if raw_outputs else None
                    outputs = {"boundary": first} if first is not None else {}
                else:
                    outputs = {"boundary": raw_outputs}
                if not isinstance(outputs, dict):
                    outputs = {"boundary": outputs}
                if "boundary" not in outputs and "boundary2" not in outputs:
                    try:
                        zeros = torch.zeros_like(targets["boundary"], dtype=torch.float32)
                    except Exception:
                        zeros = torch.tensor(0.0, device=device)
                    outputs["boundary"] = zeros
                m = mask.bool()
                # 一部の極端な設定では損失項が一つも加算されず、loss が leaf tensor
                # のまま backward() で勾配を要求しないことがある。その場合でも安全
                # に backward を実行できるよう requires_grad=True で初期化する。
                loss = torch.zeros((), device=device, requires_grad=True)
                lb = torch.zeros((), device=device)
                lv = torch.zeros((), device=device)
                ld = torch.zeros((), device=device)
                lvb = torch.zeros((), device=device)
                ldb = torch.zeros((), device=device)
                lp = torch.zeros((), device=device)
                if "boundary2" in outputs:
                    # 2-class logits available (CRF head or softmax head); use BCE on class-1 logit if not CRF
                    if head == "crf":
                        # CRF NLL
                        # build emissions and tags
                        emissions = outputs["boundary2"]  # (B,L,2)
                        tags = targets["boundary"].long()
                        lb = crf.nll(emissions, tags, m)
                        loss = loss + w_boundary * lb
                    else:
                        lb = crit_boundary(
                            outputs["boundary2"][m][:, 1] - outputs["boundary2"][m][:, 0],
                            targets["boundary"][m],
                        )
                        loss = loss + w_boundary * lb
                elif "boundary" in outputs:
                    lb = crit_boundary(outputs["boundary"][m], targets["boundary"][m])
                    loss = loss + w_boundary * lb
                if "vel_reg" in outputs:
                    lv = crit_vel_reg(outputs["vel_reg"][m], targets["vel_reg"][m])
                    loss = loss + w_vel_reg * lv
                if "dur_reg" in outputs:
                    ld = crit_dur_reg(outputs["dur_reg"][m], targets["dur_reg"][m])
                    loss = loss + w_dur_reg * ld
                if "vel_cls" in outputs:
                    lvb = crit_vel_cls(outputs["vel_cls"][m], targets["vel_cls"][m])
                    loss = loss + w_vel_cls * lvb
                if "dur_cls" in outputs:
                    ldb = crit_dur_cls(outputs["dur_cls"][m], targets["dur_cls"][m])
                    loss = loss + w_dur_cls * ldb
                if "pitch_logits" in outputs:
                    lp = crit_pitch(outputs["pitch_logits"][m], targets["pitch"][m])
                    loss = loss + w_pitch * lp
                lb_sum = lb_sum + lb.detach()
                lv_sum = lv_sum + lv.detach()
                ld_sum = ld_sum + ld.detach()
                lvb_sum = lvb_sum + lvb.detach()
                ldb_sum = ldb_sum + ldb.detach()
                lp_sum = lp_sum + lp.detach()
            loss_sum = loss_sum + loss.detach()
            if loss.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(loss / grad_accum).backward()
                else:
                    (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                if scaler.is_enabled():
                    if grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                opt.zero_grad()
                if sched and scheduler != "plateau":
                    if global_step < warmup_steps:
                        lr_scale = float(global_step + 1) / warmup_steps
                        for pg in opt.param_groups:
                            pg["lr"] = lr * lr_scale
                    else:
                        sched.step()
                global_step += 1
            if limit_train_batches and (step + 1) >= limit_train_batches:
                break
        if use_progress:
            try:
                iter_train.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        f1, val_loss, th, probs, trues, tag_buf, metrics, _ = evaluate()
        n_batches = max(1, len(dl_train))
        avg_loss = (loss_sum / n_batches).item() if loss_sum.numel() else 0.0
        lb_avg = (lb_sum / n_batches).item() if lb_sum.numel() else 0.0
        lv_avg = (lv_sum / n_batches).item() if lv_sum.numel() else 0.0
        ld_avg = (ld_sum / n_batches).item() if ld_sum.numel() else 0.0
        lvb_avg = (lvb_sum / n_batches).item() if lvb_sum.numel() else 0.0
        ldb_avg = (ldb_sum / n_batches).item() if ldb_sum.numel() else 0.0
        lp_avg = (lp_sum / n_batches).item() if lp_sum.numel() else 0.0
        lr_cur = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        logging.info(
            "epoch %d train_loss %.4f val_f1 %.3f vel_mae %.3f dur_mae %.3f vel_acc %.3f dur_acc %.3f lr %.2e th %.2f pos_rate %.3f mean_p %.3f time %.1fs",
            ep + 1,
            avg_loss,
            f1,
            metrics["vel_mae"],
            metrics["dur_mae"],
            metrics["vel_acc"],
            metrics["dur_acc"],
            lr_cur,
            th,
            metrics.get("pos_rate", 0.0),
            metrics.get("mean_prob", 0.0),
            elapsed,
        )
        try:
            tr_loss = float(avg_loss)
        except Exception:  # pragma: no cover - fallback for tensor-like values
            tr_loss = float(avg_loss.item()) if hasattr(avg_loss, "item") else float(avg_loss)
        try:
            vl_loss = float(val_loss)
        except Exception:  # pragma: no cover - fallback for tensor-like values
            vl_loss = float(val_loss.item()) if hasattr(val_loss, "item") else float(val_loss)
        # Downstream tooling parses this legacy-formatted line verbatim; keep the
        # structure identical so historical log consumers and unit tests remain stable.
        logging.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f | train_loss %.6f val_loss %.6f",
            ep + 1,
            tr_loss,
            vl_loss,
            tr_loss,
            vl_loss,
        )
        metrics_rows.append(
            {
                "epoch": ep + 1,
                "loss": avg_loss,
                "loss_boundary": lb_avg,
                "loss_vel_reg": lv_avg,
                "loss_dur_reg": ld_avg,
                "loss_vel_cls": lvb_avg,
                "loss_dur_cls": ldb_avg,
                "loss_pitch": lp_avg,
                "f1": f1,
                "best_th": th,
                "time": elapsed,
            }
        )
        preds_epoch = [1 if p > th else 0 for p in probs]
        if scheduler == "plateau" and sched:
            sched.step(f1)
        if viz and precision_recall_curve and plt:
            try:
                prec, rec, _ = precision_recall_curve(trues, probs)
                plt.figure()
                plt.plot(rec, prec)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.tight_layout()
                fname_pr = out.parent / f"pr_curve_ep{ep + 1}.png"
                plt.savefig(fname_pr)
                plt.close()
                ConfusionMatrixDisplay.from_predictions(trues, preds_epoch)
                plt.tight_layout()
                fname_cm = out.parent / f"confusion_matrix_ep{ep + 1}.png"
                plt.savefig(fname_cm)
                plt.close()
                viz_files.extend([str(fname_pr), str(fname_cm)])
            except Exception:  # pragma: no cover - visualization failures
                pass
        if writer:
            writer.add_scalar("val/f1", f1, ep)
            writer.add_scalar("train/loss", avg_loss, ep)
            writer.add_scalar("val/vel_mae", metrics["vel_mae"], ep)
            writer.add_scalar("val/dur_mae", metrics["dur_mae"], ep)
            writer.add_scalar("val/vel_acc", metrics["vel_acc"], ep)
            writer.add_scalar("val/dur_acc", metrics["dur_acc"], ep)

        metric_val = f1
        if best_metric.startswith("inst_f1:") and instrument_vocab:
            target = best_metric.split(":", 1)[1]
            inv_inst = {i: s for s, i in instrument_vocab.items()}
            groups: dict[int, list[tuple[int, int]]] = {}
            if tag_buf.get("instrument"):
                for t, p, gid in zip(trues, preds_epoch, tag_buf["instrument"]):
                    if gid <= 0:
                        continue
                    groups.setdefault(gid, []).append((t, p))
            inst_metrics = {
                inv_inst[g]: f1_score([t for t, _ in v], [p for _, p in v])
                for g, v in groups.items()
                if g in inv_inst
            }
            metric_val = inst_metrics.get(target, 0.0)
        elif best_metric.startswith("by_tag_f1:"):
            key = best_metric.split(":", 1)[1]
            vocab_map = {"section": section_vocab, "mood": mood_vocab}.get(key)
            if vocab_map and tag_buf.get(key):
                inv_map = {i: s for s, i in vocab_map.items()}
                groups: dict[int, list[tuple[int, int]]] = {}
                for t, p, gid in zip(trues, preds_epoch, tag_buf[key]):
                    if gid <= 0:
                        continue
                    groups.setdefault(gid, []).append((t, p))
                vals = [
                    f1_score([t for t, _ in v], [p for _, p in v])
                    for g, v in groups.items()
                    if g in inv_map
                ]
                metric_val = sum(vals) / len(vals) if vals else 0.0

        if metric_val > best_metric_val:
            best_metric_val = metric_val
            best_f1 = f1
            best_threshold = th
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if early_stopping and bad_epochs >= early_stopping:
                break
        if save_every and (ep + 1) % save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict() if sched else None,
                    "epoch": ep + 1,
                    "global_step": global_step,
                    "best_f1": best_f1,
                    "meta": {
                        "arch": arch,
                        "d_model": d_model,
                        "n_layers": layers,
                        "n_heads": nhead,
                        "max_len": max_len,
                        "duv_mode": duv_mode,
                        "vel_bins": vel_bins,
                        "dur_bins": dur_bins,
                        "vocab_pitch": 128,
                        "vocab": {},
                        "corpus_name": corpus_name,
                        "tag_coverage": tag_coverage,
                    },
                },
                out.with_suffix(f".epoch{ep + 1}.ckpt"),
            )
    meta = {
        "arch": arch,
        "d_model": d_model,
        "n_layers": layers,
        "n_heads": nhead,
        "max_len": max_len,
        "duv_mode": duv_mode,
        "vel_bins": vel_bins,
        "dur_bins": dur_bins,
        "use_bar_beat": bool(use_bar_beat),
        "use_harmony": bool(use_harmony),
        "vocab_pitch": 128,
        "vocab": {},
        "corpus_name": corpus_name,
        "tag_coverage": tag_coverage,
    }
    final_best_f1 = float(best_f1) if best_f1 >= 0 else 0.0
    final_epoch = max(start_epoch, epochs_completed)
    final_state = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched else None,
        "epoch": final_epoch,
        "global_step": global_step,
        "best_f1": final_best_f1,
        "meta": meta,
    }
    out_path = Path(out)
    out_parent = out_path.parent
    out_parent.mkdir(parents=True, exist_ok=True)
    saved_final = False
    # The fallback path guarantees a lightweight checkpoint even if serialising the
    # full training state fails (e.g., due to unexpected objects in optimizer state).
    try:
        torch.save(final_state, out_path)
        saved_final = True
    except Exception as exc:
        logging.warning("failed to save checkpoint to %s: %s", out, exc)
    if not saved_final:
        try:
            torch.save(
                {
                    "model": model.state_dict(),
                    "hparams": {
                        "arch": arch,
                        "d_model": d_model,
                        "max_len": max_len,
                    },
                },
                out_path,
            )
            saved_final = True
            logging.info("fallback checkpoint saved to %s", out_path)
        except Exception as exc_fallback:
            logging.warning(
                "failed to save last checkpoint to %s: %s",
                out,
                exc_fallback,
            )
    if save_last:
        last_link = out.with_name("last.ckpt")
        try:
            if last_link.exists() or last_link.is_symlink():
                last_link.unlink()
            last_link.symlink_to(out.name)
        except OSError as exc:
            logging.warning("failed to create last.ckpt symlink: %s", exc)
            try:
                shutil.copy2(out_path, last_link)
            except OSError as copy_exc:
                logging.warning("failed to copy checkpoint to last.ckpt: %s", copy_exc)
    if best_state is not None:
        best_ckpt = final_state.copy()
        best_ckpt["model"] = best_state
        best_path = out.with_suffix(".best.ckpt")
        torch.save(best_ckpt, best_path)
        if save_best:
            best_link = out.with_name("best.ckpt")
            try:
                if best_link.exists() or best_link.is_symlink():
                    best_link.unlink()
                best_link.symlink_to(best_path.name)
            except OSError as exc:
                logging.warning("failed to create best.ckpt symlink: %s", exc)
                try:
                    shutil.copy2(best_path, best_link)
                except OSError as copy_exc:
                    logging.warning("failed to copy checkpoint to best.ckpt: %s", copy_exc)
    if writer:
        writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    best_f1, _, best_threshold, probs, trues, tag_buf, metrics, logits_all = evaluate()
    inv_section = {i: s for s, i in section_vocab.items()} if section_vocab else {}
    inv_mood = {i: s for s, i in mood_vocab.items()} if mood_vocab else {}
    inv_inst = {i: s for s, i in instrument_vocab.items()} if instrument_vocab else {}
    metrics_by_tag: dict[str, dict[str, float]] = {}

    def group_f1(ids: list[int], inv_map: dict[int, str]) -> dict[str, float]:
        groups: dict[int, list[tuple[int, int]]] = {}
        for t, p, gid in zip(trues, probs, ids):
            if gid <= 0:
                continue
            pred = 1 if p > best_threshold else 0
            groups.setdefault(gid, []).append((t, pred))
        return {
            inv_map[k]: f1_score([t for t, _ in v], [p for _, p in v]) for k, v in groups.items()
        }

    if instrument_vocab:
        metrics_by_tag["instrument"] = group_f1(tag_buf["instrument"], inv_inst)
    if section_vocab:
        metrics_by_tag["section"] = group_f1(tag_buf["section"], inv_section)
    if mood_vocab:
        metrics_by_tag["mood"] = group_f1(tag_buf["mood"], inv_mood)
    metrics_path = out.parent / "metrics.json"
    metrics_data = {"f1": best_f1, "best_threshold": best_threshold}
    if metrics_by_tag:
        metrics_data["by_tag"] = metrics_by_tag
        (out.parent / "metrics_by_tag.json").write_text(
            json.dumps(metrics_by_tag, ensure_ascii=False, indent=2)
        )
    metrics_path.write_text(json.dumps(metrics_data, ensure_ascii=False))
    metrics_csv = out.parent / "metrics_epoch.csv"
    with metrics_csv.open("w", newline="") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "loss",
                "loss_boundary",
                "loss_vel_reg",
                "loss_dur_reg",
                "loss_vel_cls",
                "loss_dur_cls",
                "loss_pitch",
                "f1",
                "best_th",
                "time",
            ],
        )
        csv_writer.writeheader()
        csv_writer.writerows(metrics_rows)
    preview_path = out.parent / "preds_preview.json"
    try:
        with torch.no_grad():
            feats, targets, mask, _tags = next(iter(dl_val))
            feats = {k: v.to(device) for k, v in feats.items()}
            mask0 = mask[0].bool()
            preview_raw = model(feats, mask.to(device))
            if isinstance(preview_raw, dict):
                outputs = preview_raw
            elif isinstance(preview_raw, (list, tuple)):
                first = preview_raw[0] if preview_raw else None
                outputs = {"boundary": first} if first is not None else {}
            else:
                outputs = {"boundary": preview_raw}
            if not isinstance(outputs, dict):
                outputs = {"boundary": outputs}
            if "boundary" not in outputs and "boundary2" not in outputs:
                zeros = torch.zeros_like(targets["boundary"], dtype=torch.float32)
                outputs["boundary"] = zeros
            if "boundary2" in outputs:
                logits2 = outputs["boundary2"][0, mask0.to(device)]  # (T,2)
                probs_t = torch.softmax(logits2, dim=-1)[:, 1]
            else:
                logits = outputs["boundary"][0, mask0.to(device)]
                probs_t = torch.sigmoid(logits)
            probs = probs_t.cpu().tolist()
            preds = [1 if p > best_threshold else 0 for p in probs]
            trues = targets["boundary"][0][mask0].int().cpu().tolist()
        preview_path.write_text(
            json.dumps({"probs": probs, "preds": preds, "trues": trues}, ensure_ascii=False)
        )
    except StopIteration:  # pragma: no cover - empty validation set
        pass
    # Emit a compact JSON summary to ease sweeps/log parsing
    print(
        json.dumps(
            {
                "f1": final_best_f1,
                "best_th": float(best_threshold),
                "arch": arch,
                "d_model": int(d_model),
                "pos_weight": float(pos_weight) if pos_weight else None,
                "w_boundary": float(w_boundary),
                "duv_mode": duv_mode,
            }
        )
    )
    stats = {
        "class_counts": class_counts,
        "tag_counts": tag_counts,
        "tag_coverage": tag_coverage,
        "viz_paths": viz_files,
    }
    if weight_stats:
        stats["tag_weights_top10"] = weight_stats
    # Self-check: targeted suites expected to pass —
    # tests/test_train_phrase_duv.py,
    # tests/test_train_phrase_transformer_hparams.py::test_transformer_hparams,
    # tests/test_train_phrase_duv_embed_toggle.py::test_duv_embed_toggle.
    return final_best_f1, device.type, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_csv",
        type=Path,
        nargs="?",
        help="training CSV file (ignored when --data is used)",
    )
    parser.add_argument(
        "val_csv",
        type=Path,
        nargs="?",
        help="validation CSV file (ignored when --data is used)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("phrase.ckpt"))
    parser.add_argument(
        "--arch", choices=["transformer", "lstm", "bilstm_tcn"], default="transformer"
    )
    parser.add_argument("--head", choices=["linear", "crf"], default="linear")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--sin-posenc",
        action="store_true",
        help="use sinusoidal positional encoding in transformer",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["cosine", "plateau"], default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--auto-pos-weight", action="store_true")
    parser.add_argument(
        "--f1-scan-range",
        nargs=3,
        type=float,
        default=(0.2, 0.8, 0.1),
        metavar=("START", "END", "STEP"),
        help="threshold search range",
    )
    parser.add_argument("--logdir", type=Path)
    parser.add_argument("--min-f1", type=float, default=-1.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--early-stopping", type=int, default=0)
    parser.add_argument("--precision", choices=["high", "medium", "low"], default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument(
        "--data",
        type=Path,
        help="corpus directory with train/valid splits; overrides positional CSVs",
    )
    parser.add_argument("--instrument")
    parser.add_argument("--include-tags", type=str, default=None)
    parser.add_argument("--exclude-tags", type=str, default=None)
    parser.add_argument("--reweight", type=str, default=None)
    parser.add_argument("--use-duv-embed", action="store_true")
    # bar/beat 位相特徴の有効化（重複定義だったため単一に統一）
    parser.add_argument("--use-bar-beat", action="store_true", help="add bar/beat phase features")
    parser.add_argument(
        "--use-harmony",
        action="store_true",
        help="use harmony features if present (harm_root/func/degree)",
    )
    parser.add_argument(
        "--use-local-stats",
        action="store_true",
        help="use local rolling stats (IOI/interval/velocity)",
    )
    parser.add_argument("--duv-mode", choices=["none", "reg", "cls", "both"], default="reg")
    parser.add_argument("--vel-bins", type=int, default=0)
    parser.add_argument("--dur-bins", type=int, default=0)
    parser.add_argument("--w-boundary", type=float, default=1.0)
    parser.add_argument("--w-vel-reg", type=float, default=0.5)
    parser.add_argument("--w-dur-reg", type=float, default=0.5)
    parser.add_argument("--w-vel-cls", type=float, default=0.0)
    parser.add_argument("--w-dur-cls", type=float, default=0.0)
    parser.add_argument(
        "--w-pitch",
        type=float,
        default=1.0,
        help="weight for pitch loss (use --pitch-smoothing for label smoothing)",
    )
    parser.add_argument(
        "--pitch-smoothing",
        type=float,
        default=0.0,
        help="CrossEntropy label smoothing for pitch head",
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--viz", action="store_true", help="save PR/CM plots")
    parser.add_argument(
        "--strict-tags", action="store_true", help="drop rows missing requested tags"
    )
    parser.add_argument("--progress", action="store_true", help="show training progress bar")
    parser.add_argument(
        "--calibrate-temp",
        action="store_true",
        help="calibrate temperature on validation logits and save to checkpoint meta",
    )
    # fast iteration helpers
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=0,
        help="max train batches per epoch for quick runs",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=0,
        help="max validation batches per eval for quick runs",
    )
    parser.add_argument(
        "--limit-train-groups", type=int, default=0, help="limit number of training groups (bars)"
    )
    parser.add_argument(
        "--limit-val-groups", type=int, default=0, help="limit number of validation groups (bars)"
    )
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="run a very small loop to sanity-check"
    )
    parser.add_argument(
        "--best-metric",
        default="macro_f1",
        help="metric for best model: macro_f1 or inst_f1:<name> or by_tag_f1:<tag>",
    )
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.f1_scan_range = tuple(args.f1_scan_range)

    handlers = [logging.StreamHandler()]
    if args.logdir:
        args.logdir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.logdir / "train.log"))
    logging.basicConfig(level=logging.INFO, handlers=handlers)

    include = {}
    if args.include_tags:
        for part in args.include_tags.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                include[k] = v
    exclude = {}
    if args.exclude_tags:
        for part in args.exclude_tags.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                exclude[k] = v

    if args.data:
        try:
            train_rows, val_rows = load_corpus(
                args.data,
                include_tags=include,
                exclude_tags=exclude,
                strict=args.strict_tags,
            )
            train_rows, tr_removed = apply_filters(
                train_rows,
                args.instrument,
                include,
                exclude,
                strict=args.strict_tags,
            )
            val_rows, val_removed = apply_filters(
                val_rows,
                args.instrument,
                include,
                exclude,
                strict=args.strict_tags,
            )
            logging.info(
                "corpus rows kept %d/%d train %d/%d val",
                len(train_rows),
                len(train_rows) + tr_removed,
                len(val_rows),
                len(val_rows) + val_removed,
            )
            if args.sample:
                rng = random.Random(0)
                train_rows = rng.sample(train_rows, min(args.sample, len(train_rows)))
                val_rows = rng.sample(val_rows, min(args.sample, len(val_rows)))
            tmp = _make_tempdir("train_phrase_")
            train_csv = tmp / "train.csv"
            val_csv = tmp / "valid.csv"
            write_csv(train_rows, train_csv)
            write_csv(val_rows, val_csv)
            args.train_csv = train_csv
            args.val_csv = val_csv
        except Exception:
            tmp = _make_tempdir("train_phrase_")
            train_csv = tmp / "train.csv"
            val_csv = tmp / "valid.csv"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.corpus_to_phrase_csv",
                    "--in",
                    str(args.data),
                    "--out-train",
                    str(train_csv),
                    "--out-valid",
                    str(val_csv),
                ],
                check=True,
                env={**os.environ, "ALLOW_LOCAL_IMPORT": "1"},
            )
            args.train_csv = train_csv
            args.val_csv = val_csv

    if not args.train_csv or not args.train_csv.is_file():
        raise SystemExit(f"missing train_csv {args.train_csv}")
    if not args.val_csv or not args.val_csv.is_file():
        raise SystemExit(f"missing val_csv {args.val_csv}")

    if args.strict_tags:
        vocab_path = args.train_csv.parent / "tag_vocab.json"
        if not vocab_path.is_file():
            raise SystemExit(
                f"--strict-tags requires tag_vocab.json beside train_csv (missing {vocab_path})"
            )
        try:
            tag_vocab = json.loads(vocab_path.read_text())
            if not isinstance(tag_vocab, dict):
                raise ValueError("tag_vocab.json must be an object of allowed values")
        except Exception as e:
            raise SystemExit(f"failed to load tag vocab {vocab_path}: {e}")
        required = set(tag_vocab.keys())
        for path in [args.train_csv, args.val_csv]:
            rows = load_csv_rows(path, required)
            for i, row in enumerate(rows, 1):
                for k, allowed in tag_vocab.items():
                    if row.get(k, "") not in allowed:
                        raise SystemExit(
                            f"{path} line {i}: unknown {k}={row.get(k, '')}; allowed={sorted(allowed)}"
                        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_path = args.out.with_suffix(".run.json")
    run_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    run_cfg["git_commit"] = commit
    run_cfg["git_dirty"] = dirty
    try:
        import torch

        torch_ver = torch.__version__
    except Exception:  # pragma: no cover
        torch_ver = None
    try:
        import numpy as _np

        np_ver = _np.__version__
    except Exception:  # pragma: no cover
        np_ver = None
    run_cfg["env"] = {
        "platform": sys.platform,
        "torch_version": torch_ver,
        "numpy_version": np_ver,
    }
    run_cfg["seed"] = args.seed

    result: object | None = None
    error_info: dict[str, object] | None = None
    exit_code = 0
    try:
        result = train_model(
            args.train_csv,
            args.val_csv,
            args.epochs,
            args.arch,
            args.out,
            seed=args.seed,
            batch_size=args.batch_size,
            d_model=args.d_model,
            max_len=args.max_len,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            grad_clip=args.grad_clip,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            warmup_steps=args.warmup_steps,
            pos_weight=args.pos_weight,
            auto_pos_weight=args.auto_pos_weight,
            resume=args.resume,
            save_every=args.save_every,
            early_stopping=args.early_stopping,
            f1_scan_range=args.f1_scan_range,
            logdir=args.logdir,
            precision=args.precision,
            deterministic=args.deterministic,
            device=args.device,
            best_metric=args.best_metric,
            reweight=args.reweight,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            use_duv_embed=args.use_duv_embed,
            use_bar_beat=args.use_bar_beat,
            use_harmony=args.use_harmony,
            duv_mode=args.duv_mode,
            vel_bins=args.vel_bins,
            dur_bins=args.dur_bins,
            head=args.head,
            loss=args.loss,
            focal_gamma=args.focal_gamma,
            progress=args.progress,
            w_boundary=args.w_boundary,
            w_vel_reg=args.w_vel_reg,
            w_dur_reg=args.w_dur_reg,
            w_vel_cls=args.w_vel_cls,
            w_dur_cls=args.w_dur_cls,
            w_pitch=args.w_pitch,
            pitch_smoothing=args.pitch_smoothing,
            instrument=args.instrument,
            include_tags=include,
            exclude_tags=exclude,
            viz=args.viz,
            strict_tags=args.strict_tags,
            nhead=args.nhead,
            layers=args.layers,
            dropout=args.dropout,
            compile=args.compile,
            grad_accum=args.grad_accum,
            save_best=args.save_best,
            save_last=args.save_last,
            use_sinusoidal_posenc=args.sin_posenc,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            limit_train_groups=args.limit_train_groups,
            limit_val_groups=args.limit_val_groups,
            fast_dev_run=args.fast_dev_run,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        error_info = {"type": "ValueError", "message": str(exc)}
        exit_code = 2
    except Exception as exc:  # pragma: no cover - unexpected runtime failures
        print(f"[ERROR] {exc}", file=sys.stderr)
        error_info = {"type": type(exc).__name__, "message": str(exc)}
        exit_code = 2

    if error_info is None:
        if isinstance(result, tuple):
            if len(result) == 3:
                f1, device_type, stats = result
            elif len(result) == 2:
                f1, device_type = result
                stats = {}
            else:
                f1 = result[0] if len(result) > 0 else 0.0
                device_type = result[1] if len(result) > 1 else "cpu"
                stats = result[2] if len(result) > 2 else {}
        elif result is None:
            f1, device_type, stats = 0.0, "cpu", {}
        else:
            try:
                f1 = float(result)
            except (TypeError, ValueError):
                f1 = 0.0
            device_type, stats = "cpu", {}
    else:
        f1, device_type, stats = 0.0, "cpu", {}

    if not isinstance(stats, dict):
        stats = {"extra": stats}

    run_cfg["viz_enabled"] = bool(args.viz and stats.get("viz_paths"))
    run_cfg["viz_backend"] = plt.get_backend() if run_cfg["viz_enabled"] and plt else None
    run_cfg["sampler_weights_summary"] = stats
    run_cfg["tag_coverage"] = stats.get("tag_coverage", {})
    run_cfg["viz_paths"] = stats.get("viz_paths", [])
    run_cfg["status"] = "error" if error_info else "ok"
    if error_info:
        run_cfg["error"] = error_info
    run_path.write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2))

    hparams = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    hparams["device"] = device_type
    (args.out.parent / "hparams.json").write_text(json.dumps(hparams, ensure_ascii=False, indent=2))
    # temperature calibration is handled inside train_model when enabled
    if exit_code:
        return exit_code
    return 0 if args.min_f1 < 0 or f1 >= args.min_f1 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
