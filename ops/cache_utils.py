# ops/cache_utils.py
"""Minimal cache utilities for chord recognition pipeline"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple
import numpy as np

__all__ = [
    "hash_params", "ensure_cache_dir", "save_npz", "load_npz", "compute_and_cache",
    "digest_files",
]

def hash_params(**kwargs) -> str:
    """Generate MD5 hash from parameters"""
    s = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def ensure_cache_dir(cache_dir: Path) -> Path:
    """Create cache directory if not exists"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def save_npz(path: Path, **arrays) -> None:
    """Save numpy arrays to compressed npz"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load numpy arrays from npz"""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def compute_and_cache(
    func: Callable[[], Tuple], 
    cache_path: Path, 
    use_cache: bool, 
    keys: Tuple[str, ...]
):
    """Generic cache wrapper. `func` must return tuple aligned with `keys`."""
    if use_cache and cache_path.exists():
        data = load_npz(cache_path)
        return tuple(data[k] for k in keys)
    out = func()  # tuple
    save_npz(cache_path, **{k: v for k, v in zip(keys, out)})
    return out

def digest_files(files: Iterable[Path]) -> str:
    """Generate digest from file metadata (mtime + size)"""
    parts = []
    for p in files:
        try:
            st = p.stat()
            parts.append(f"{p.name}:{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            parts.append(f"{p.name}:NA:NA")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:16]
