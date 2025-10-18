#!/usr/bin/env python3
"""Build metadata index pickle from .meta.json files."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm


def build_index_from_json_metadata(
    input_dir: Path,
    output_pickle: Path,
) -> None:
    """
    Aggregate .meta.json files into a single pickle index.
    
    Args:
        input_dir: Directory containing .midi and .meta.json files
        output_pickle: Output path for aggregated pickle file
    """
    input_path = Path(input_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    # Find all .meta.json files
    json_files = sorted(input_path.rglob("*.meta.json"))
    print(f"Found {len(json_files):,} .meta.json files in {input_path}")
    
    if not json_files:
        raise ValueError(f"No .meta.json files found in {input_path}")
    
    # Build loop records
    loop_records: List[Dict[str, Any]] = []
    failed_count = 0
    
    for json_path in tqdm(json_files, desc="Processing"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Find corresponding .midi file
            midi_path = json_path.with_suffix(".midi")
            if not midi_path.exists():
                # Try .mid extension
                midi_path = json_path.with_suffix(".mid")
            
            if not midi_path.exists():
                print(f"Warning: No MIDI file for {json_path.name}")
                failed_count += 1
                continue
            
            # Build record matching LAMDa format
            record = {
                "path": str(midi_path.relative_to(input_path)),
                "name": midi_path.stem,
                "tempo": metadata.get("tempo", 120.0),
                "time_signature": metadata.get("time_signature", "4/4"),
                "duration_sec": metadata.get("duration_sec", 0.0),
                "notes": metadata.get("notes", 0),
                "density": metadata.get("density", 0.0),
                "bars": metadata.get("bars", 0.0),
                "grid_off_std_ms": metadata.get("grid_off_std_ms", 0.0),
                "grid_off_mean_ms": metadata.get("grid_off_mean_ms", 0.0),
                "kick_on_beat_rate": metadata.get("kick_on_beat_rate", 0.0),
                "velocity_std": metadata.get("velocity_std", 0.0),
                "velocity_mean": metadata.get("velocity_mean", 0.0),
                "clean_actions": metadata.get("clean_actions", []),
                "reason_codes": metadata.get("reason_codes", []),
            }
            
            loop_records.append(record)
            
        except Exception as e:
            print(f"Error processing {json_path.name}: {e}")
            failed_count += 1
            continue
    
    print(f"\nProcessed: {len(loop_records):,}")
    print(f"Failed: {failed_count:,}")
    
    if not loop_records:
        raise ValueError("No valid loop records generated")
    
    # Build index structure matching LAMDa format
    index_data = {
        "version": "2.0",  # New version for JSON-based metadata
        "source": "clean_midi.py",
        "total_loops": len(loop_records),
        "loops": loop_records,
        "config": {
            "input_dir": str(input_path),
            "tmidix_path": "data/Los-Angeles-MIDI/CODE",  # Default, may not be used
        },
    }
    
    # Write pickle
    output_path = Path(output_pickle).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nWrote index to: {output_path}")
    print(f"Total loops: {len(loop_records):,}")


if __name__ == "__main__":
    input_dir = Path("output/drumloops_v3_test")
    output_pickle = Path("output/drumloops_v3_metadata/drumloops_metadata_v3.pickle")
    
    build_index_from_json_metadata(
        input_dir=input_dir,
        output_pickle=output_pickle,
    )
