#!/usr/bin/env python3
"""Guard: decide whether a retry result should be accepted.

Compares before/after metrics and evaluates min_delta conditions
from control block to determine if retry was effective.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Load JSONL file line by line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _index_by_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index rows by loop_id/id/hash."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = r.get("loop_id") or r.get("id") or r.get("hash")
        if k:
            out[str(k)] = r
    return out


def _coerce_float(val: Any) -> float | None:
    """Convert value to float, return None if not possible."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def should_accept(
    prev: Dict[str, Any],
    post: Dict[str, Any],
    control: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate if retry should be accepted based on min_delta criteria.

    Args:
        prev: Metrics before retry
        post: Metrics after retry
        control: Control block from preset containing min_delta

    Returns:
        (accepted: bool, metadata: dict)
    """
    min_delta_raw = control.get("min_delta")
    if min_delta_raw is None:
        # No criteria, accept by default
        return True, {"reason": "no_min_delta_criteria"}

    prev_axes = prev.get("axes_raw", {})
    post_axes = post.get("axes_raw", {})
    prev_score = _coerce_float(prev.get("score_total"))
    post_score = _coerce_float(post.get("score_total"))

    # Handle both old float format and new dict format
    if isinstance(min_delta_raw, dict):
        md_dict = min_delta_raw
        total_req = _coerce_float(md_dict.get("score_total"))
        axes_req = md_dict.get("axes_raw", {})

        delta_total = None
        ok_total = True
        if total_req is not None:
            if prev_score is not None and post_score is not None:
                delta_total = post_score - prev_score
                ok_total = delta_total >= total_req
            else:
                ok_total = False

        deltas_axes: Dict[str, float] = {}
        ok_axes = True
        if isinstance(axes_req, dict):
            for k, threshold in axes_req.items():
                threshold_f = _coerce_float(threshold)
                if threshold_f is None:
                    continue
                prev_val = _coerce_float(prev_axes.get(k))
                post_val = _coerce_float(post_axes.get(k))
                if prev_val is not None and post_val is not None:
                    delta = post_val - prev_val
                    deltas_axes[k] = delta
                    if delta < threshold_f:
                        ok_axes = False
                else:
                    ok_axes = False

        # Audio boost: small text_audio_cos improvement can help
        prev_audio = prev.get("audio", {})
        post_audio = post.get("audio", {})
        prev_cos = _coerce_float(prev_audio.get("text_audio_cos"))
        post_cos = _coerce_float(post_audio.get("text_audio_cos"))
        audio_boost = False
        delta_audio_cos = None

        if prev_cos is not None and post_cos is not None:
            delta_audio_cos = post_cos - prev_cos
            # Small boost: +0.02 improvement helps borderline cases
            if delta_audio_cos >= 0.02:
                audio_boost = True

        # Final decision: (ok_total AND ok_axes) OR audio_boost
        final_ok = (ok_total and ok_axes) or (ok_total and audio_boost)

        # Determine acceptance reason
        accept_reason = "rejected"
        if final_ok:
            if ok_total and ok_axes:
                accept_reason = "score+axes"
            elif ok_total and audio_boost:
                accept_reason = "score+audio_boost"
            else:
                accept_reason = "accepted"
        else:
            # Rejection reason
            if not ok_total:
                accept_reason = "insufficient_score_gain"
            elif not ok_axes:
                accept_reason = "axes_degraded"
            else:
                accept_reason = "rejected_unknown"

        meta = {
            "delta_total": delta_total,
            "deltas_axes": deltas_axes,
            "delta_audio_cos": delta_audio_cos,
            "ok_total": ok_total,
            "ok_axes": ok_axes,
            "audio_boost": audio_boost,
            "accept_reason": accept_reason,
            "criteria": {"score_total": total_req, "axes_raw": axes_req},
        }
        return final_ok, meta
    else:
        # Legacy float format: treat as simple score threshold
        threshold_f = _coerce_float(min_delta_raw)
        if threshold_f is None:
            return True, {"reason": "invalid_min_delta_format"}

        if prev_score is not None and post_score is not None:
            delta = post_score - prev_score
            ok = delta >= threshold_f
            return ok, {
                "delta_total": delta,
                "threshold": threshold_f,
                "ok": ok,
            }
        else:
            return False, {"reason": "missing_score_values"}


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description=("Guard: decide whether a retry result should be accepted.")
    )
    ap.add_argument(
        "--before",
        required=True,
        help="metrics_score.jsonl (before retry)",
    )
    ap.add_argument(
        "--after",
        required=True,
        help="metrics_score.jsonl (after retry)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSONL of guard decisions",
    )
    args = ap.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    out_path = Path(args.out)

    if not before_path.exists():
        print(f"ERROR: Before file not found: {before_path}", file=sys.stderr)
        return 1
    if not after_path.exists():
        print(f"ERROR: After file not found: {after_path}", file=sys.stderr)
        return 1

    prev_idx = _index_by_id(_load_jsonl(before_path))
    out_rows = []

    for post in _load_jsonl(after_path):
        loop_id = str(post.get("loop_id") or post.get("id") or post.get("hash"))
        prev = prev_idx.get(loop_id)
        ctrl = post.get("_retry_control") or {}

        if not prev or not ctrl:
            # No comparison possible
            continue

        ok, meta = should_accept(prev, post, ctrl)
        decision = {
            "loop_id": loop_id,
            "accepted_by_guard": bool(ok),
            "guard_meta": meta,
            "_retry_state": post.get("_retry_state"),
            "_retry_control": ctrl,
        }
        out_rows.append(decision)

    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    accepted_count = sum(1 for r in out_rows if r["accepted_by_guard"])
    total_count = len(out_rows)
    print(
        f"Wrote {total_count} guard decisions to {out_path} "
        f"(accepted: {accepted_count}, rejected: {total_count - accepted_count})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
