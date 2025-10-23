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


SCHEMA_VERSION = "lamda_v2.6"


def extract_stage2_metadata(midi_path: Path) -> Dict[str, Any]:
    """Extract Stage2 metadata from MIDI file.
    
    Parameters
    ----------
    midi_path : Path
        Path to MIDI file.
    
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
        "controls": {"pb_range": [0, 0], "cc_summary": {}, "rpn_seen": False},
    }


def extract_to_json(midi_path: Path, output_path: Optional[Path] = None) -> Path:
    """Extract Stage2 metadata and save to JSON file.
    
    Parameters
    ----------
    midi_path : Path
        Path to MIDI file.
    output_path : Path, optional
        Output JSON path. If None, uses midi_path.stem + ".stage2.json".
    
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
    meta = extract_stage2_metadata(midi_path)
    
    if output_path is None:
        output_path = midi_path.parent / f"{midi_path.stem}.stage2.json"
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    return output_path


def batch_extract(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.mid",
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
            extract_to_json(midi_path, output_path)
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
    
    args = parser.parse_args()
    
    if args.input.is_file():
        # Single file mode
        json_path = extract_to_json(args.input, args.output)
        print(f"✅ Extracted: {json_path}")
    elif args.input.is_dir():
        # Batch mode
        output_dir = args.output or args.input / "stage2_output"
        json_paths = batch_extract(args.input, output_dir, args.pattern)
        print(f"✅ Extracted {len(json_paths)} files to {output_dir}")
    else:
        print(f"❌ Error: {args.input} not found")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
