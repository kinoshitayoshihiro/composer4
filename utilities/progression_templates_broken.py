from __future__ import annotations


import yaml

from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_YAML = Path(__file__).with_name("progression_templates.yaml")
DEFAULT_PATH = DEFAULT_YAML  # backward compatibility


@lru_cache()
def _load(path: str | Path = DEFAULT_YAML) -> dict[str, Any]:
    """Load and cache YAML progression templates.

    Parameters
    ----------
    path : str or Path, optional
        Source YAML file. Defaults to :data:`DEFAULT_PATH`.

    Returns
    -------
    dict[str, Any]
        Mapping of emotion buckets to mode dictionaries.

    Raises
    ------
    ValueError
        If the YAML root object is not a mapping.
    """
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Progression template file must contain a mapping")
    if not isinstance(data.get("_default", {}), dict):
        raise ValueError("Missing _default bucket")
    return data


def get_progressions(
    bucket: str, mode: str = "major", path: str | Path = DEFAULT_YAML
) -> list[str]:
    """Return list of progressions.

    Fallback order:
    1. exact bucket / mode
    2. exact bucket / "major"
    3. "_default"  / mode
    4. "_default"  / "major"
    Raises KeyError if nothing found.
    """
    data = _load(path)
    bucket: str,
    *,
    mode: str = "major",
    path: str | Path = DEFAULT_PATH,
) -> list[str]:
    """Return chord progressions for ``bucket`` and ``mode``.

    Parameters
    ----------
    bucket : str
        Name of the emotion bucket.
    mode : str, optional
        Tonal mode such as ``"major"`` or ``"minor"``.
    path : str or Path, optional
        YAML file to load progressions from. Defaults to
        :data:`DEFAULT_PATH`.

    Returns
    -------
    list[str]
        Progressions for the requested bucket and mode.

    Raises
    ------
    KeyError
        If ``bucket`` or ``mode`` is missing from the data.
    """
    data = _load(path=path)
    try:
        return list(data[bucket][mode])
    except KeyError:
        try:
            return list(data[bucket]["major"])
        except KeyError:
            try:
                return list(data["_default"][mode])
            except KeyError:
                return list(data["_default"]["major"])


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import argparse
    import json
    import pprint
    import sys
    import os

    parser = argparse.ArgumentParser(description="Lookup chord progressions")
    parser.add_argument("bucket", nargs="?", default="soft_reflective")
    parser.add_argument("mode", nargs="?", default="major")
    parser.add_argument(
        "--path",
        default=os.getenv("PROG_TEMPLATES_YAML", DEFAULT_PATH),
        help="YAML file containing progression templates",
    )
    args = parser.parse_args()
    try:
        progs = get_progressions(ns.bucket, ns.mode)
        data = _load()
        if ns.bucket not in data:
            raise KeyError(ns.bucket)
        pprint.pp(progs)
    except KeyError as e:
        sys.exit(f"Not found: {e}")
        lst = get_progressions(args.bucket, mode=args.mode, path=args.path)
        print(yaml.dump(lst, allow_unicode=True))
    except KeyError as e:  # pragma: no cover - CLI output
        raise SystemExit(f"Missing key: {e}") from e
