"""
adapters.meta_key_normalizer
----------------------------
Normalize heterogeneous META_DATA keys into a canonical schema used by Stage2 overlays.
- Handles snake/camel/kebab case variants and common synonyms.
- NO-OP safe: unknown keys are preserved under "extras".
"""
from __future__ import annotations
from typing import Dict, Any

_CANON = {
    "total_number_of_tracks": "tracks",
    "total_tracks": "tracks",
    "numTracks": "tracks",
    "total_number_of_opus_midi_events": "opus_events",
    "opusEvents": "opus_events",
    "average_median_mode_time_ms": "avg_time_ms",
    "avgMedianModeTimeMs": "avg_time_ms",
    "average_median_mode_dur_ms": "avg_dur_ms",
    "avgMedianModeDurMs": "avg_dur_ms",
    "average_median_mode_vel": "avg_vel",
    "avgMedianModeVel": "avg_vel",
    "total_number_of_chords": "total_chords",
    "totalChords": "total_chords",
    "ms_chords_counts": "ms_chords_counts",
    "pitches_times_sum_ms": "pitches_times_sum_ms",
    "total_pitches_counts": "total_pitches_counts",
    "totalPitchesCounts": "total_pitches_counts",
    "midi_patches": "midi_patches",
    "patches": "midi_patches",
    "total_patches_counts": "total_patches_counts",
    "tempo_change_count": "tempo_changes",
    "tempoChangeCount": "tempo_changes",
    "text_events_count": "text_events",
    "lyric_events_count": "lyric_events",
}

def normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "counts": {},
        "timing_stats_ms": {},
        "velocity_stats": {},
        "patches": None,
        "patch_counts": None,
        "pitch_hist": None,
        "extras": {}
    }
    for k, v in (meta or {}).items():
        canon = _CANON.get(k, None)
        if canon is None:
            out["extras"][k] = v
            continue
        if canon == "tracks":
            out["counts"]["tracks"] = int(v)
        elif canon == "opus_events":
            out["counts"]["opus_events"] = int(v)
        elif canon == "tempo_changes":
            out["counts"]["tempo_changes"] = int(v)
        elif canon == "text_events":
            out["counts"]["text_events"] = int(v)
        elif canon == "lyric_events":
            out["counts"]["lyric_events"] = int(v)
        elif canon == "total_chords":
            out["counts"]["total_chords"] = int(v)
        elif canon == "avg_time_ms":
            out["timing_stats_ms"]["avg_time"] = float(v)
        elif canon == "avg_dur_ms":
            out["timing_stats_ms"]["avg_dur"] = float(v)
        elif canon == "avg_vel":
            out["velocity_stats"]["avg_vel"] = float(v)
        elif canon == "midi_patches":
            out["patches"] = v
        elif canon == "total_patches_counts":
            out["patch_counts"] = v
        elif canon == "total_pitches_counts":
            out["pitch_hist"] = v
        elif canon == "ms_chords_counts":
            out["ms_chords_counts"] = v
        elif canon == "pitches_times_sum_ms":
            out["pitches_times_sum_ms"] = v
        else:
            out["extras"][k] = v
    # thin empty branches
    if not out["extras"]: out.pop("extras", None)
    if not out["patch_counts"]: out.pop("patch_counts", None)
    if not out.get("pitch_hist"): pass
    return out
