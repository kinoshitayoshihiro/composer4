#!/usr/bin/env python3
"""
LAMDA v2.6 Stage2 Extractor - Clean implementation using lamda_v2 modules.

Extract extended metadata from MIDI files:
- Tempo grid & downbeats (tempo_timing)
- Chord progressions (chord_analyzer)
- Key hints & modulations (key_analyzer)
- Section segmentation (section_analyzer - Phase2-4)
- Groove profiles (groove_analyzer - Phase3)
- Control summaries (controls_analyzer - Phase3)

Design principles:
- Minimal dependencies (pretty_midi only)
- Clear separation of concerns
- TDD-validated components
- NO-OP fallbacks for robustness
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pretty_midi
except ImportError:
    pretty_midi = None  # type: ignore

# Import lamda_v2 modules
from scripts.lamda_v2.tempo_timing import build_beat_grid
from scripts.lamda_v2.chord_analyzer import extract_bar_chords
from scripts.lamda_v2.key_analyzer import estimate_local_key_sequence, to_key_hints_payload
from scripts.lamda_v2.section_analyzer import auto_segment_sections
from scripts.lamda_v2.groove_analyzer import analyze_groove
from scripts.lamda_v2.controls_analyzer import analyze_controls

# LAMDA integration (optional, NO-OP safe)
from scripts.lamda_v2.lamda_sources import LamdaSources
from scripts.lamda_v2.outlier_stats import summarize_outliers
from scripts.lamda_v2.lamda_fusion_utils import (
    decode_kilo_to_events,
    timesig_rescue,
    patch_summary_from_meta,
    local_hist_from_pm,
    stats_from_meta,
)
import yaml


SCHEMA_VERSION = "lamda_v2.6"


def extract_stage2_metadata(
    midi_path: Path,
    lamda_sources: Optional[LamdaSources] = None,
    signature_map_yaml: Optional[Path] = None,
) -> Dict[str, Any]:
    """Extract Stage2 metadata from MIDI file.

    Parameters
    ----------
    midi_path : Path
        Path to MIDI file.
    lamda_sources : LamdaSources, optional
        LAMDA data sources (KILO/META/SIGNATURES/TOTALS).
        If None, LAMDA integration is skipped (NO-OP).
    signature_map_yaml : Path, optional
        SIGNATURES ID→拍子マッピングYAML.
        Default: configs/lamda/signature_id_map.yaml

    Returns
    -------
    Dict[str, Any]
        Metadata payload with schema_version="lamda_v2.6".

    Examples
    --------
    >>> meta = extract_stage2_metadata(Path("test.mid"))
    >>> meta["schema_version"]
    'lamda_v2.6'
    >>> meta["tempo_map"]
    [[0.0, 120.0], [4.0, 140.0]]
    """
    if pretty_midi is None:
        return _error_payload("pretty_midi not available")

    # Pathオブジェクトに変換
    from pathlib import Path

    midi_path = Path(midi_path) if isinstance(midi_path, str) else midi_path

    if not midi_path.exists():
        return _error_payload(f"File not found: {midi_path}")

    try:
        # Load MIDI
        pm = pretty_midi.PrettyMIDI(str(midi_path))

        # Phase1: Tempo & timing
        grid = build_beat_grid(pm)
        tempo_map = grid.get("tempo_map", [[0.0, 120.0]])
        timesig_map = grid.get("timesig_map", [[0, "4/4"]])
        timesig_map_time = grid.get("timesig_map_time", [(0.0, "4/4")])
        downbeats_sec = grid.get("downbeats_sec", [])
        downbeats_ql = grid.get("downbeats_ql", [])

        # Phase2-2: Chord extraction
        chordmap = extract_bar_chords(
            pm,
            downbeats_ql,
            min_dwell_ql=2.0,
            extended_vocab=True,
        )

        # Phase2-3: Key hints & modulations
        key_seq = estimate_local_key_sequence(
            chordmap,
            win_bars=4,
            min_hold=4,
            ks_weight=0.7,
        )
        key_payload = to_key_hints_payload(key_seq)

        # Phase2-4: Sections
        sections = auto_segment_sections(
            pm,
            downbeats_ql,
            min_bars=8,
        )

        # Phase3: Groove & controls (NO-OP safe)
        groove = analyze_groove(pm, downbeats_sec)
        controls = analyze_controls(pm)

        # Build payload
        payload = {
            "schema_version": SCHEMA_VERSION,
            "tempo_map": tempo_map,
            "timesig_map": timesig_map,
            "timesig_map_time": timesig_map_time,
            "downbeats_sec": downbeats_sec,
            "downbeats_ql": downbeats_ql,
            "chordmap": chordmap,
            "key_hint": key_payload.get("key_hint", []),
            "modulations": key_payload.get("modulations", []),
            "sections_auto": sections,
            "groove": groove,
            "controls": controls,
        }

        # ========================================
        # LAMDA Fusion (optional, NO-OP safe)
        # ========================================
        if lamda_sources:
            file_id = midi_path.stem

            # (a) KILO → chordmap_external（優先進行）
            kilo_seq = lamda_sources.get_kilo_chords(file_id)
            if kilo_seq:
                payload["chordmap_external"] = decode_kilo_to_events(kilo_seq, unit="ql")

            # (b) SIGNATURES → timesig rescue（1/4→4/4補正の裏取り）
            sigs = lamda_sources.get_signatures(file_id)
            if sigs:
                # Load signature map
                if signature_map_yaml is None:
                    signature_map_yaml = Path("configs/lamda/signature_id_map.yaml")

                if signature_map_yaml.exists():
                    sig_map = yaml.safe_load(signature_map_yaml.open())
                else:
                    sig_map = {}

                # Convert sig_id → time_signature
                labels = [sig_map.get(int(sid), f"unknown:{sid}") for sid, _ in sigs]
                payload["signatures"] = labels

                # Timesig rescue (if all "4/4" + all "1/4" + avg_bar≈4.0QL)
                timesig_rescue(grid, labels)

                # Update payload with rescued timesig
                payload["timesig_map"] = grid.get("timesig_map", timesig_map)
                payload["timesig_map_time"] = grid.get("timesig_map_time", timesig_map_time)

            # (c) META → patches/statistics
            meta_entry = lamda_sources.get_meta(file_id)
            if meta_entry:
                payload["lamda_meta_present"] = True
                payload["patch_summary"] = patch_summary_from_meta(meta_entry)
                payload["note_stats_meta"] = stats_from_meta(meta_entry)

            # (d) TOTALS → outlier scores（pitch/dur/vel）
            totals = lamda_sources.get_totals()
            if totals:
                local_hist = local_hist_from_pm(pm)
                payload["outliers"] = summarize_outliers(local_hist, totals, method="chi2")

        return payload

    except Exception as e:
        return _error_payload(str(e))


def _error_payload(error_msg: str) -> Dict[str, Any]:
    """Create error payload with minimal schema."""
    return {
        "schema_version": f"{SCHEMA_VERSION}_error",
        "error": error_msg,
        "tempo_map": [[0.0, 120.0]],
        "timesig_map": [[0, "4/4"]],
        "timesig_map_time": [(0.0, "4/4")],
        "downbeats_sec": [],
        "downbeats_ql": [],
        "chordmap": {"unit": "ql", "events": []},
        "key_hint": [],
        "modulations": [],
        "sections_auto": {"unit": "bar", "sections": [], "energy": []},
        "groove": {"swing_pct": 0.0, "backbeat_strength": 0.5, "onset_deviation_hist": []},
        "controls": {"pb_range": [0, 0], "cc_summary": {}, "rpn_seen": False, "integrity": 1.0},
    }


def extract_to_json(
    midi_path: Path,
    output_path: Optional[Path] = None,
    lamda_sources: Optional[LamdaSources] = None,
    signature_map_yaml: Optional[Path] = None,
) -> Path:
    """Extract Stage2 metadata and save to JSON file.

    Parameters
    ----------
    midi_path : Path
        Path to MIDI file.
    output_path : Path, optional
        Output JSON path. If None, uses midi_path.stem + ".stage2.json".
    lamda_sources : LamdaSources, optional
        LAMDA data sources (KILO/META/SIGNATURES/TOTALS).
    signature_map_yaml : Path, optional
        SIGNATURES ID→拍子マッピングYAML.

    Returns
    -------
    Path
        Path to written JSON file.

    Examples
    --------
    >>> json_path = extract_to_json(Path("test.mid"))
    >>> json_path.name
    'test.stage2.json'
    """
    meta = extract_stage2_metadata(midi_path, lamda_sources, signature_map_yaml)

    if output_path is None:
        output_path = midi_path.parent / f"{midi_path.stem}.stage2.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return output_path


def batch_extract(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.mid",
    lamda_sources: Optional[LamdaSources] = None,
    signature_map_yaml: Optional[Path] = None,
) -> List[Path]:
    """Batch extract Stage2 metadata from MIDI files.

    Parameters
    ----------
    input_dir : Path
        Directory containing MIDI files.
    output_dir : Path
        Directory for output JSON files.
    pattern : str, optional
        File pattern (default: "*.mid").
    lamda_sources : LamdaSources, optional
        LAMDA data sources (KILO/META/SIGNATURES/TOTALS).
    signature_map_yaml : Path, optional
        SIGNATURES ID→拍子マッピングYAML.

    Returns
    -------
    List[Path]
        List of written JSON paths.

    Examples
    --------
    >>> paths = batch_extract(Path("midis"), Path("output"))
    >>> len(paths)
    42
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(input_dir.glob(pattern))
    json_paths = []

    for midi_path in midi_files:
        output_path = output_dir / f"{midi_path.stem}.stage2.json"
        try:
            extract_to_json(midi_path, output_path, lamda_sources, signature_map_yaml)
            json_paths.append(output_path)
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            continue

    return json_paths


def main():
    """CLI entry point for Stage2 extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="LAMDA v2.6 Stage2 Metadata Extractor")
    parser.add_argument("input", type=Path, help="MIDI file or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file or directory")
    parser.add_argument("-p", "--pattern", default="*.mid", help="File pattern for batch mode")

    # LAMDA integration options (all optional, NO-OP safe)
    parser.add_argument("--lamda-kilo", type=Path, help="LAMDa_KILO_CHORDS_DATA.pickle")
    parser.add_argument("--lamda-meta-dir", type=Path, help="META_DATA directory")
    parser.add_argument("--lamda-signatures", type=Path, help="LAMDa_SIGNATURES_DATA.pickle")
    parser.add_argument("--lamda-totals", type=Path, help="LAMDa_TOTALS.pickle")
    parser.add_argument("--lamda-id-map", type=Path, help="auto_file_id_map.csv")
    parser.add_argument(
        "--signature-map-yaml",
        type=Path,
        default=Path("configs/lamda/signature_id_map.yaml"),
        help="SIGNATURES ID→拍子マッピングYAML",
    )

    args = parser.parse_args()

    # Initialize LAMDA sources (if any option is provided)
    lamda_sources = None
    if any([args.lamda_kilo, args.lamda_meta_dir, args.lamda_signatures, args.lamda_totals]):
        lamda_sources = LamdaSources(
            kilo=str(args.lamda_kilo) if args.lamda_kilo else None,
            meta_dir=str(args.lamda_meta_dir) if args.lamda_meta_dir else None,
            signatures=str(args.lamda_signatures) if args.lamda_signatures else None,
            totals=str(args.lamda_totals) if args.lamda_totals else None,
            id_map_csv=str(args.lamda_id_map) if args.lamda_id_map else None,
        )

        # Show LAMDA summary
        print("📊 LAMDA Sources:")
        for key, available in lamda_sources.summary().items():
            status = "✅" if available else "❌"
            print(f"  {status} {key}")

    if args.input.is_file():
        # Single file mode
        json_path = extract_to_json(args.input, args.output, lamda_sources, args.signature_map_yaml)
        print(f"✅ Extracted: {json_path}")
    elif args.input.is_dir():
        # Batch mode
        output_dir = args.output or args.input / "stage2_output"
        json_paths = batch_extract(
            args.input, output_dir, args.pattern, lamda_sources, args.signature_map_yaml
        )
        print(f"✅ Extracted {len(json_paths)} files to {output_dir}")
    else:
        print(f"❌ Error: {args.input} not found")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
