import pandas as pd
import pytest

from scripts import lamda_make_pairs as pairs


def _make_sample_frame() -> pd.DataFrame:
    data = {
        "md5": ["src", "tgt1", "tgt2"],
        "genre": ["Funk", "Funk", "Rock"],
        "output_path": ["src.mid", "tgt1.mid", "tgt2.mid"],
        "input_path": ["src_in.mid", "tgt1_in.mid", "tgt2_in.mid"],
        "shard_index": [0, 0, 1],
        "metrics.energy": [0.9, 0.85, 0.2],
        "metrics.sync": [0.1, 0.15, 0.95],
    }
    return pd.DataFrame(data)


def test_split_query_variants() -> None:
    assert pairs._split_query(None) == (None, None)
    assert pairs._split_query("funk -> soul") == ("funk", "soul")
    assert pairs._split_query("latin→house") == ("latin", "house")
    assert pairs._split_query("only_source ") == ("only_source", None)


def test_filter_required_columns_drops_blank() -> None:
    frame = pd.DataFrame(
        {
            "col": ["value", " ", None],
            "other": [1, 2, 3],
        }
    )
    filtered = pairs._filter_required_columns(frame, ["col"])
    assert list(filtered["col"]) == ["value"]


def test_filter_required_columns_missing_column() -> None:
    frame = pd.DataFrame({"col": ["value"]})
    with pytest.raises(KeyError):
        pairs._filter_required_columns(frame, ["missing"])


def test_build_pairs_basic() -> None:
    frame = _make_sample_frame()
    metric_cols = ["metrics.energy", "metrics.sync"]

    pairs_list = pairs._build_pairs(
        frame,
        metric_cols,
        source_filter="funk",
        target_filter=None,
        filter_column="genre",
        top_k=2,
        max_pairs=None,
        min_similarity=0.0,
        allow_self=False,
    )

    assert pairs_list, "Expected at least one generated pair"
    first_pair = pairs_list[0]
    assert first_pair["input_md5"] == "src"
    assert first_pair["target_md5"] == "tgt1"
    assert first_pair["input_output_path"] == "src.mid"
    assert first_pair["target_output_path"] == "tgt1.mid"


def test_build_pairs_respects_min_similarity() -> None:
    frame = _make_sample_frame()
    metric_cols = ["metrics.energy", "metrics.sync"]

    pairs_list = pairs._build_pairs(
        frame,
        metric_cols,
        source_filter="funk",
        target_filter=None,
        filter_column="genre",
        top_k=2,
        max_pairs=None,
        min_similarity=1.01,
        allow_self=False,
    )

    assert pairs_list == []


def test_build_pairs_allow_self_flag() -> None:
    frame = _make_sample_frame()
    metric_cols = ["metrics.energy", "metrics.sync"]

    duplicate = frame.iloc[[0]].copy()
    frame_with_duplicate = pd.concat([frame, duplicate], ignore_index=True)

    pairs_no_self = pairs._build_pairs(
        frame_with_duplicate,
        metric_cols,
        source_filter="funk",
        target_filter=None,
        filter_column="genre",
        top_k=3,
        max_pairs=None,
        min_similarity=0.0,
        allow_self=False,
    )
    assert all(pair["input_md5"] != pair["target_md5"] for pair in pairs_no_self)

    pairs_with_self = pairs._build_pairs(
        frame_with_duplicate,
        metric_cols,
        source_filter="funk",
        target_filter=None,
        filter_column="genre",
        top_k=3,
        max_pairs=None,
        min_similarity=0.0,
        allow_self=True,
    )
    assert any(pair["input_md5"] == pair["target_md5"] for pair in pairs_with_self)
