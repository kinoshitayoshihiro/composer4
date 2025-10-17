#!/usr/bin/env python3
"""
GuitarSet Dataset Importer

GuitarSetの生MIDIデータを統一フォーマットに変換し、
Stage1パイプラインに統合可能な形式で出力。

Input:  data/external/guitarset/annotation/*.jams
Output: data/external/guitarset/raw/*.mid

Target techniques:
- guitar_strum: ~150 files
- guitar_arpeggio: ~80 files
- guitar_fingerpicking: ~130 files
"""

import argparse
import pathlib
import json
from typing import Dict, List, Any
import pretty_midi as pm


def parse_guitarset_jams(jams_file: pathlib.Path) -> Dict[str, Any]:
    """
    JAMS annotation fileを解析
    
    Returns:
        {
            "notes": [(time, duration, pitch, velocity), ...],
            "techniques": {time: technique_name, ...},
            "tempo": bpm,
        }
    """
    with open(jams_file, 'r') as f:
        jams_data = json.load(f)
    
    # Extract note events
    notes = []
    annotations = jams_data.get("annotations", [])
    
    for annot in annotations:
        if annot.get("namespace") == "note_midi":
            data = annot.get("data", [])
            for event in data:
                time = event.get("time", 0.0)
                duration = event.get("duration", 0.0)
                value = event.get("value", {})
                pitch = value.get("pitch", 60)
                velocity = value.get("velocity", 64)
                
                notes.append((time, duration, pitch, velocity))
    
    # Extract techniques (if available)
    techniques = {}
    for annot in annotations:
        if annot.get("namespace") == "technique":
            data = annot.get("data", [])
            for event in data:
                time = event.get("time", 0.0)
                technique = event.get("value", "unknown")
                techniques[time] = technique
    
    # Extract tempo
    tempo = 120.0  # Default
    file_metadata = jams_data.get("file_metadata", {})
    if "tempo" in file_metadata:
        tempo = float(file_metadata["tempo"])
    
    return {
        "notes": sorted(notes, key=lambda x: x[0]),
        "techniques": techniques,
        "tempo": tempo,
    }


def jams_to_midi(jams_data: Dict[str, Any], output_path: pathlib.Path):
    """JAMS data → MIDI変換"""
    midi = pm.PrettyMIDI(initial_tempo=jams_data["tempo"])
    
    # Create guitar instrument
    guitar = pm.Instrument(program=24, name="Acoustic Guitar")  # Nylon Guitar
    
    # Add notes
    for time, duration, pitch, velocity in jams_data["notes"]:
        note = pm.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=float(time),
            end=float(time + duration),
        )
        guitar.notes.append(note)
    
    midi.instruments.append(guitar)
    
    # Write
    midi.write(str(output_path))


def classify_technique(jams_file: pathlib.Path, jams_data: Dict[str, Any]) -> str:
    """
    ファイル名とアノテーションから奏法分類
    
    Returns: "strum" | "arpeggio" | "fingerpicking" | "mixed"
    """
    filename = jams_file.stem.lower()
    techniques = list(jams_data["techniques"].values())
    
    # Filename-based heuristics
    if "strum" in filename or "comp" in filename:
        return "strum"
    elif "arpeggio" in filename or "arp" in filename:
        return "arpeggio"
    elif "fingerpick" in filename or "travis" in filename:
        return "fingerpicking"
    
    # Technique annotation-based
    strum_count = sum(1 for t in techniques if "strum" in t.lower())
    arpeggio_count = sum(1 for t in techniques if "arpeggio" in t.lower())
    pick_count = sum(1 for t in techniques if "pick" in t.lower())
    
    if strum_count > arpeggio_count and strum_count > pick_count:
        return "strum"
    elif arpeggio_count > strum_count and arpeggio_count > pick_count:
        return "arpeggio"
    elif pick_count > 0:
        return "fingerpicking"
    
    # Default: mixed
    return "mixed"


def main():
    parser = argparse.ArgumentParser(description="GuitarSet JAMS → MIDI converter")
    parser.add_argument("--guitarset-dir", type=str,
                        default="data/external/guitarset",
                        help="GuitarSet root directory")
    parser.add_argument("--output-dir", type=str,
                        default="data/external/guitarset/raw",
                        help="Output directory for converted MIDI files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run (no file output)")
    
    args = parser.parse_args()
    
    guitarset_dir = pathlib.Path(args.guitarset_dir)
    annotation_dir = guitarset_dir / "annotation"
    output_dir = pathlib.Path(args.output_dir)
    
    if not annotation_dir.exists():
        print(f"[ERROR] Annotation directory not found: {annotation_dir}")
        print(f"[ERROR] Please download GuitarSet annotations first")
        print(f"[ERROR] URL: https://zenodo.org/record/3371780")
        return 1
    
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process JAMS files
    jams_files = list(annotation_dir.glob("*.jams"))
    print(f"[INFO] Found {len(jams_files)} JAMS files")
    
    # Statistics
    stats = {
        "strum": 0,
        "arpeggio": 0,
        "fingerpicking": 0,
        "mixed": 0,
    }
    
    for jams_file in jams_files:
        try:
            # Parse JAMS
            jams_data = parse_guitarset_jams(jams_file)
            
            # Classify technique
            technique = classify_technique(jams_file, jams_data)
            stats[technique] += 1
            
            # Generate output filename
            output_file = output_dir / f"{jams_file.stem}_{technique}.mid"
            
            if args.dry_run:
                print(f"[DRY-RUN] {jams_file.name} → {output_file.name} ({technique})")
            else:
                # Convert to MIDI
                jams_to_midi(jams_data, output_file)
                print(f"[OK] {output_file.name} ({technique})")
        
        except Exception as e:
            print(f"[ERROR] {jams_file.name}: {e}")
            continue
    
    # Summary
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Total:          {len(jams_files)}")
    print(f"  Strum:          {stats['strum']}")
    print(f"  Arpeggio:       {stats['arpeggio']}")
    print(f"  Fingerpicking:  {stats['fingerpicking']}")
    print(f"  Mixed:          {stats['mixed']}")
    print("="*50)
    
    if not args.dry_run:
        print(f"\n[COMPLETE] Output: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
