#!/usr/bin/env python3
"""PR Comment Aggregator for Composer4 dashboards."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.request
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

MARKER = "<!-- COMPOSER4_DASHBOARD -->"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pluck(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return cast(Any, current)


def _float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    if not filtered:
        return None
    return float(statistics.median(filtered))


def _fmt(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.3f}{suffix}"


def _delta(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    return after - before


def _fmt_delta(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}{suffix}"


def parse_simple_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            if ":" not in line:
                continue
            key, _, rest = line.strip().partition(":")
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1] if stack else root
            content = rest.strip()
            if not content:
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
                continue
            if content == "null":
                parent[key] = None
            else:
                try:
                    parent[key] = float(content)
                except ValueError:
                    parent[key] = content
    return root


def _tempo_bin(bpm: Any, edges: Sequence[float]) -> str:
    numeric = _safe_float(bpm)
    if numeric is None:
        return "NA"
    for edge in edges:
        if numeric < edge:
            return f"<{edge}"
    return f">={edges[-1]}"


def _bucket_key(
    row: Mapping[str, Any],
    edges: Sequence[float],
) -> Tuple[str, str]:
    inst = str(pluck(row, "instrument") or "NA")
    tempo = pluck(row, "tempo.bpm")
    return inst, _tempo_bin(tempo, edges)


def _choose_tau(
    tau_map: Mapping[str, Any],
    inst: str,
    tempo_bin: str,
    fallback: Optional[float],
) -> Optional[float]:
    bit_raw = tau_map.get("by_instrument_tempo")
    if isinstance(bit_raw, Mapping):
        inst_map = bit_raw.get(inst)
        if isinstance(inst_map, Mapping):
            value = inst_map.get(tempo_bin)
            if isinstance(value, (int, float)):
                return float(value)
    bii_raw = tau_map.get("by_instrument")
    if isinstance(bii_raw, Mapping):
        value = bii_raw.get(inst)
        if isinstance(value, (int, float)):
            return float(value)
    byt_raw = tau_map.get("by_tempo")
    if isinstance(byt_raw, Mapping):
        value = byt_raw.get(tempo_bin)
        if isinstance(value, (int, float)):
            return float(value)
    global_tau = tau_map.get("global")
    if isinstance(global_tau, (int, float)):
        return float(global_tau)
    return fallback


def collect_ab(
    rows: Sequence[Mapping[str, Any]],
    axes: Sequence[str],
    tau: float,
) -> Dict[str, Any]:
    axes_block: Dict[str, Optional[float]] = {}
    for axis in axes:
        axes_block[axis] = _median([_float(pluck(row, f"axes_raw.{axis}")) for row in rows])

    cos_values = [_float(pluck(row, "metrics.text_audio_cos")) for row in rows]
    cos_filtered = [v for v in cos_values if v is not None]
    mismatch = None
    if cos_filtered:
        mismatch = 100.0 * (sum(1 for value in cos_filtered if value < tau) / len(cos_filtered))

    return {
        "axes": axes_block,
        "text_audio_cos": {"p50": _median(cos_filtered)},
        "mismatch_rate": {"rate": mismatch},
    }


def _mismatch_at_tau(
    values: Sequence[Optional[float]],
    tau: float,
) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return 100.0 * (sum(1 for value in filtered if value < tau) / len(filtered))


def recalc_mismatch(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    tau: float,
) -> Optional[float]:
    values = [_float(pluck(row, key)) for row in rows]
    return _mismatch_at_tau(values, tau)


def build_ab_section(
    before_rows: Sequence[Mapping[str, Any]],
    after_rows: Sequence[Mapping[str, Any]],
    axes: Sequence[str],
    tau: float,
    tau_file: Optional[str],
) -> Optional[str]:
    if not before_rows or not after_rows:
        return None

    before = collect_ab(before_rows, axes, tau)
    after = collect_ab(after_rows, axes, tau)

    tau_map = parse_simple_yaml(tau_file) if tau_file else {}
    tau_prop = tau_map.get("global")
    tau_proposed = float(tau_prop) if isinstance(tau_prop, (int, float)) else None

    lines: List[str] = []
    lines.append("### A/B Summary")
    header = f"_tau (current): **{tau:.3f}**_"
    if tau_proposed is not None:
        header += f"  |  _proposed: **{tau_proposed:.3f}**_"
    lines.append(header)
    lines.append("")

    lines.append("**Axes (p50)**")
    for axis in axes:
        before_axis = before["axes"].get(axis)
        after_axis = after["axes"].get(axis)
        lines.append(
            "- `{axis}` | before: {before_val} → after: {after_val}"
            " | Δ {delta_val}".format(
                axis=axis,
                before_val=_fmt(before_axis),
                after_val=_fmt(after_axis),
                delta_val=_fmt_delta(_delta(before_axis, after_axis)),
            )
        )
    lines.append("")

    before_cos = before["text_audio_cos"]["p50"]
    after_cos = after["text_audio_cos"]["p50"]
    before_mismatch = before["mismatch_rate"]["rate"]
    after_mismatch = after["mismatch_rate"]["rate"]

    lines.append("**CLAP**")
    lines.append(
        "- `text_audio_cos.p50` | before: {before_val} → after: {after_val}"
        " | Δ {delta_val}".format(
            before_val=_fmt(before_cos),
            after_val=_fmt(after_cos),
            delta_val=_fmt_delta(_delta(before_cos, after_cos)),
        )
    )
    lines.append(
        "- `caption_mismatch_rate` | before: {before_val} → after: {after_val}"
        " | Δ {delta_val}".format(
            before_val=_fmt(before_mismatch, "%"),
            after_val=_fmt(after_mismatch, "%"),
            delta_val=_fmt_delta(
                _delta(before_mismatch, after_mismatch),
                "pp",
            ),
        )
    )

    if tau_proposed is not None:
        lines.append("")
        lines.append(f"_@tau_proposed = {tau_proposed:.3f}_")
        before_preview = recalc_mismatch(
            before_rows,
            "metrics.text_audio_cos",
            tau_proposed,
        )
        after_preview = recalc_mismatch(
            after_rows,
            "metrics.text_audio_cos",
            tau_proposed,
        )
        lines.append(
            "- `caption_mismatch_rate@tau_proposed` | before: {before_val}"
            " → after: {after_val} | Δ {delta_val}".format(
                before_val=_fmt(before_preview, "%"),
                after_val=_fmt(after_preview, "%"),
                delta_val=_fmt_delta(
                    _delta(before_preview, after_preview),
                    "pp",
                ),
            )
        )
        lines.append("")

    return "\n".join(lines)


def build_crops_section(rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    if not rows:
        return None
    num_rows = len(rows)
    stats: Dict[str, Optional[float]] = {
        "cos_mean_p50": _median([row.get("cos_mean") for row in rows]),
        "cos_std_p50": _median([row.get("cos_std") for row in rows]),
        "cos_iqr_p50": _median([row.get("cos_iqr") for row in rows]),
    }
    lines: List[str] = []
    lines.append("### CLAP Multi-crops (10s)")
    lines.append(f"- files: {num_rows}")
    lines.append(f"- `cos_mean.p50`: {_fmt(stats['cos_mean_p50'])}")
    lines.append(f"- `cos_std.p50`: {_fmt(stats['cos_std_p50'])}")
    lines.append(f"- `cos_iqr.p50`: {_fmt(stats['cos_iqr_p50'])}")
    lines.append("")
    lines.append("_狙い: 長尺・変動素材で一致スコアの安定化（分散・IQRの減少を確認）_")
    lines.append("")
    return "\n".join(lines)


def build_mert_crops_section(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    if not rows:
        return None

    def med(key: str) -> Optional[float]:
        return _median([row.get(key) for row in rows])

    num_rows = len(rows)
    stats: Dict[str, Optional[float]] = {
        "cos_mean_p50": med("cos_mean"),
        "cos_std_p50": med("cos_std"),
        "cos_iqr_p50": med("cos_iqr"),
    }
    lines: List[str] = []
    lines.append("### MERT Multi-crops (8–10s @24k)")
    lines.append(f"- files: {num_rows}")
    lines.append(f"- `cos_mean.p50`: {_fmt(stats['cos_mean_p50'])}")
    lines.append(f"- `cos_std.p50`: {_fmt(stats['cos_std_p50'])}")
    lines.append(f"- `cos_iqr.p50`: {_fmt(stats['cos_iqr_p50'])}")
    lines.append("")
    lines.append("_狙い: MERTでの一致スコア安定化（分散/IQRの減少）_")
    lines.append("")
    return "\n".join(lines)


def build_va_section(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return None
    lines: List[str] = []
    lines.append("### EMOPIA Valence/Arousal")
    lines.append(
        "- `valence.p50`: {vp}  |  `valence.IQR`: {vi}  |  n={nv}".format(
            vp=_fmt(data.get("valence_p50")),
            vi=_fmt(data.get("valence_iqr")),
            nv=data.get("n_valence", 0),
        )
    )
    lines.append(
        "- `arousal.p50`: {ap}  |  `arousal.IQR`: {ai}  |  n={na}".format(
            ap=_fmt(data.get("arousal_p50")),
            ai=_fmt(data.get("arousal_iqr")),
            na=data.get("n_arousal", 0),
        )
    )
    lines.append("")
    lines.append(
        "_狙い: 感情曲面（V/A）の中心と散らばりを定点観測 → " "Humanizerや奏法選択の係数キャリブへ_"
    )
    lines.append("")
    return "\n".join(lines)


def build_va_buckets_section(
    path: Optional[str],
    topk: int = 3,
) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return None
    buckets = payload.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        return None
    lines: List[str] = []
    lines.append("### EMOPIA VA Buckets (Top-3 by n)")
    for bucket in buckets[:topk]:
        lines.append(
            "- `{instrument} × {tempo}` | n={n} | V.p50 {vp} / IQR {vi}"
            " | A.p50 {ap} / IQR {ai}".format(
                instrument=bucket.get("instrument", "NA"),
                tempo=bucket.get("tempo_bin", "NA"),
                n=bucket.get("n", 0),
                vp=_fmt(bucket.get("valence_p50")),
                vi=_fmt(bucket.get("valence_iqr")),
                ap=_fmt(bucket.get("arousal_p50")),
                ai=_fmt(bucket.get("arousal_iqr")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_tau_bucket_preview(
    title: str,
    before_rows: Sequence[Mapping[str, Any]],
    after_rows: Sequence[Mapping[str, Any]],
    tau_yaml_path: Optional[str],
    cos_key: str,
    tempo_edges: Sequence[float],
    topk: int = 3,
) -> Optional[str]:
    if not tau_yaml_path or not os.path.exists(tau_yaml_path):
        return None
    tau_map = parse_simple_yaml(tau_yaml_path)
    if not before_rows or not after_rows:
        return None

    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for label, rows in ("before", before_rows), ("after", after_rows):
        for row in rows:
            inst, tempo_bin = _bucket_key(row, tempo_edges)
            value = _float(pluck(row, cos_key))
            if value is None:
                continue
            buckets.setdefault((inst, tempo_bin), {"before": [], "after": []})
            buckets[(inst, tempo_bin)][label].append(value)

    preview: List[Dict[str, Any]] = []
    global_tau = tau_map.get("global")
    fallback = float(global_tau) if isinstance(global_tau, (int, float)) else None

    for (inst, tempo_bin), values in buckets.items():
        tau = _choose_tau(tau_map, inst, tempo_bin, fallback)
        if tau is None:
            continue
        before_vals = values["before"]
        after_vals = values["after"]
        if not before_vals or not after_vals:
            continue
        before_rate = _mismatch_at_tau(before_vals, tau)
        after_rate = _mismatch_at_tau(after_vals, tau)
        if before_rate is None or after_rate is None:
            continue
        preview.append(
            {
                "inst": inst,
                "tempo_bin": tempo_bin,
                "tau": tau,
                "before": before_rate,
                "after": after_rate,
                "delta": after_rate - before_rate,
                "n": max(len(before_vals), len(after_vals)),
            }
        )

    if not preview:
        return None

    preview.sort(
        key=lambda item: (
            -abs(item["delta"]),
            -item["n"],
            item["inst"],
            item["tempo_bin"],
        )
    )

    lines: List[str] = []
    lines.append(f"### {title} (Buckets @ proposed τ, Top-{topk} by |Δ|)")
    for entry in preview[:topk]:
        lines.append(
            "- `{inst} × {tempo}` | τ={tau:.3f} | mismatch: {before:.1f}%"
            " → {after:.1f}% (Δ {delta:+.1f}pp) | n={n}".format(
                inst=entry["inst"],
                tempo=entry["tempo_bin"],
                tau=entry["tau"],
                before=entry["before"],
                after=entry["after"],
                delta=entry["delta"],
                n=entry["n"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_guard_section(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return None
    ok = payload.get("ok")
    if ok is None:
        return None
    raw_results = payload.get("results")
    results: List[Dict[str, Any]] = []
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict):
                results.append(cast(Dict[str, Any], item))

    def _flag(status: Any) -> str:
        return "PASS ✅" if bool(status) else "FAIL ❌"

    lines: List[str] = []
    lines.append("### Guard Summary")
    lines.append(f"- result: **{_flag(ok)}**")
    for result in results:
        kind = result.get("kind", "?")
        report_obj = result.get("report")
        if isinstance(report_obj, dict):
            report = cast(Dict[str, Any], report_obj)
        else:
            report = {}
        global_delta = report.get("global_delta")
        if isinstance(global_delta, (int, float)):
            delta_text = f"{float(global_delta):+.1f}pp"
        else:
            delta_text = "—"
        bad_buckets_obj = report.get("bad_buckets")
        if isinstance(bad_buckets_obj, list):
            bad_buckets: List[Dict[str, Any]] = []
            for entry in bad_buckets_obj:
                if isinstance(entry, dict):
                    bad_buckets.append(cast(Dict[str, Any], entry))
        else:
            bad_buckets = []
        lines.append(
            "- {kind}: global Δ {delta} | bad buckets: {count}".format(
                kind=kind,
                delta=delta_text,
                count=len(bad_buckets),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _github_request(
    url: str,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
) -> Any:
    request = urllib.request.Request(url, method=method)
    request.add_header("Accept", "application/vnd.github+json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(  # type: ignore[arg-type]
        request,
        data=data,
    ) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _get_pr_number_from_event() -> Optional[int]:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not os.path.exists(event_path):
        return None
    with open(event_path, "r", encoding="utf-8") as handle:
        event = json.load(handle)
    number = event.get("pull_request", {}).get("number")
    return int(number) if isinstance(number, int) else None


def _find_existing_comment(
    repo: str,
    pr_number: int,
    marker: str,
    token: str,
) -> Optional[int]:
    page = 1
    while True:
        url = (
            "https://api.github.com/repos/"
            f"{repo}/issues/{pr_number}/comments?per_page=100&page={page}"
        )
        comments = _github_request(url, token=token)
        if not comments:
            return None
        for comment in comments:
            body = comment.get("body") or ""
            if marker in body:
                identifier = comment.get("id")
                return int(identifier) if isinstance(identifier, int) else None
        page += 1


def _upsert_comment(
    repo: str,
    pr_number: int,
    body: str,
    token: str,
) -> None:
    existing = _find_existing_comment(repo, pr_number, MARKER, token)
    if existing is not None:
        url = f"https://api.github.com/repos/{repo}/issues/comments/{existing}"
        _github_request(url, method="PATCH", body={"body": body}, token=token)
    else:
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        _github_request(url, method="POST", body={"body": body}, token=token)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", default="output/ab/before.jsonl")
    parser.add_argument("--after", default="output/ab/after.jsonl")
    parser.add_argument("--tau", type=float, default=0.50)
    parser.add_argument("--tau-file")
    parser.add_argument("--tau-file-mert")
    parser.add_argument(
        "--axes",
        action="append",
        default=["velocity", "structure"],
    )
    parser.add_argument(
        "--crops-jsonl",
        default="artifacts/clap_multicrops.jsonl",
    )
    parser.add_argument(
        "--mert-crops-jsonl",
        default="artifacts/mert_multicrops.jsonl",
    )
    parser.add_argument("--va-json", default="artifacts/va_summary.json")
    parser.add_argument(
        "--va-buckets-json",
        default="artifacts/va_buckets.json",
    )
    parser.add_argument(
        "--tempo-edges",
        default="95,110,130,150",
        help="comma separated tempo edges",
    )
    parser.add_argument("--title", default="Composer4 Dashboard")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--guard-json",
        default="artifacts/tau_guard.summary.json",
    )
    return parser.parse_args(argv)


def _normalize_edges(raw: str) -> List[float]:
    edges: List[float] = []
    if not raw:
        return edges
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            edges.append(float(piece))
        except ValueError:
            continue
    return edges


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    tau_file = args.tau_file
    if not tau_file:
        if os.path.exists("configs/metrics/tau.yaml"):
            tau_file = "configs/metrics/tau.yaml"
        elif os.path.exists("artifacts/auto_tau.yaml"):
            tau_file = "artifacts/auto_tau.yaml"

    tau_file_mert = args.tau_file_mert
    if not tau_file_mert:
        if os.path.exists("configs/metrics/tau_mert.yaml"):
            tau_file_mert = "configs/metrics/tau_mert.yaml"
        elif os.path.exists("artifacts/auto_tau_mert.yaml"):
            tau_file_mert = "artifacts/auto_tau_mert.yaml"

    before_rows = read_jsonl(args.before)
    after_rows = read_jsonl(args.after)
    clap_rows = read_jsonl(args.crops_jsonl)
    mert_rows = read_jsonl(args.mert_crops_jsonl)

    sections: List[str] = []
    ab_section = build_ab_section(
        before_rows,
        after_rows,
        args.axes,
        args.tau,
        tau_file,
    )
    if ab_section:
        sections.append(ab_section)

    clap_section = build_crops_section(clap_rows)
    if clap_section:
        sections.append(clap_section)

    mert_section = build_mert_crops_section(mert_rows)
    if mert_section:
        sections.append(mert_section)

    va_section = build_va_section(args.va_json)
    if va_section:
        sections.append(va_section)

    va_bucket_section = build_va_buckets_section(args.va_buckets_json)
    if va_bucket_section:
        sections.append(va_bucket_section)

    edges = _normalize_edges(args.tempo_edges)
    if not edges:
        edges = [95.0, 110.0, 130.0, 150.0]

    clap_tau_section = build_tau_bucket_preview(
        "CLAP Tau Preview",
        before_rows,
        after_rows,
        tau_file,
        "metrics.text_audio_cos",
        edges,
    )
    if clap_tau_section:
        sections.append(clap_tau_section)

    mert_tau_section = build_tau_bucket_preview(
        "MERT Tau Preview",
        before_rows,
        after_rows,
        tau_file_mert,
        "metrics.text_audio_cos_mert",
        edges,
    )
    if mert_tau_section:
        sections.append(mert_tau_section)

    guard_section = build_guard_section(args.guard_json)
    if guard_section:
        sections.append(guard_section)

    if not sections:
        print(
            "[pr_comment_aggregator] No sections to post; missing artifacts?",
            file=sys.stderr,
        )
        return 0

    body = [MARKER, f"## {args.title}", ""]
    body.extend(sections)
    body.append("_auto-posted by Composer4 PR dashboard_")
    content = "\n".join(body)

    if args.dry_run:
        print(content)
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = _get_pr_number_from_event()
    if not all([token, repo, pr_number]):
        print(
            "[pr_comment_aggregator] Missing envs. Use --dry-run locally.",
            file=sys.stderr,
        )
        return 1

    assert token is not None
    assert repo is not None
    assert pr_number is not None
    _upsert_comment(str(repo), int(pr_number), content, str(token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
