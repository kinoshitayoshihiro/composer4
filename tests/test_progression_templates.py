import pytest

from utilities.progression_templates import get_progressions
import importlib
from pathlib import Path

import pytest

from utilities.progression_templates import _load, get_progressions


@pytest.mark.parametrize(
    "bucket,mode",
    [
        ("soft_reflective", "major"),
        ("soft_reflective", "minor"),
        ("_default", "major"),
        ("_default", "minor"),
        ("unknown", "minor"),
    ],
)
def test_progressions(bucket: str, mode: str) -> None:
    progs = get_progressions(bucket, mode=mode)
    assert isinstance(progs, list)
    assert len(progs) >= 3


@pytest.mark.parametrize(
    "bucket,mode",
    [
        ("soft_reflective", "major"),
        ("soft_reflective", "minor"),
        ("_default", "major"),
        ("_default", "minor"),
        ("unknown", "minor"),
    ],
)
def test_lookup(bucket: str, mode: str) -> None:
    lst = get_progressions(bucket, mode=mode)
    assert isinstance(lst, list) and lst


def test_cache() -> None:
    id1 = id(_load())
    id2 = id(_load())
    assert id1 == id2


@pytest.mark.parametrize(
    "bucket,mode",
    [
        ("missing", "major"),
        ("soft_reflective", "dorian"),
    ],
)
def test_keyerror(bucket: str, mode: str) -> None:
    with pytest.raises(KeyError):
        get_progressions(bucket, mode=mode)


def test_basic_lookup(tmp_path: Path) -> None:
    yaml_file = tmp_path / "progs.yaml"
    yaml_file.write_text(
        "soft_reflective:\n"
        "  major:\n"
        "    - 'I V vi IV'\n"
        "  minor:\n"
        "    - 'i VII VI VII'\n"
    )
    mod = importlib.import_module("utilities.progression_templates")
    lst = mod.get_progressions("soft_reflective", mode="major", path=yaml_file)
    assert lst == ["I V vi IV"]


def test_cache_identity(tmp_path: Path) -> None:
    yaml_file = tmp_path / "progs.yaml"
    yaml_file.write_text("dummy: {}\n")
    mod = importlib.import_module("utilities.progression_templates")
    id1 = id(mod._load(path=yaml_file))
    id2 = id(mod._load(path=yaml_file))
    assert id1 == id2, "lru_cache should return same dict instance"


@pytest.mark.parametrize(
    "bucket, mode", [("missing", "major"), ("soft_reflective", "dorian")]
)
def test_key_error(tmp_path: Path, bucket: str, mode: str) -> None:
    yaml_file = tmp_path / "progs.yaml"
    yaml_file.write_text("soft_reflective:\n  major: ['I IV V']\n")
    import utilities.progression_templates as pt

    with pytest.raises(KeyError):
        pt.get_progressions(bucket, mode=mode, path=yaml_file)
