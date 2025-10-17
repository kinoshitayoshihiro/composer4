#!/usr/bin/env python3
"""
URMP Dataset Importer

URMP (University of Rochester Multi-Modal Performance)の
strings MIDIデータをStage1統合用に前処理。

Input:  data/external/urmp/Dataset/*/*_vn.mid (violin)
                                  /*/*_va.mid (viola)
                                  /*/*_vc.mid (cello)
                                  /*/*_db.mid (double bass)
Output: data/external/urmp/raw/strings/*.mid

Target techniques:
- strings_legato: ~200 files (最優先)
- strings_spiccato: ~150 files
- strings_staccato: ~100 files
"""

import argparse
import pathlib
import shutil
from typing import Dict
import mido


def classify_strings_technique(midi_file: pathlib.Path) -> str:
    """
    MIDI内容から弦楽器奏法を推定
    
    Heuristics:
    - Legato: 高いnote overlap (>80%)
    - Staccato: 短いnote duration (<30% of inter-onset)
    - Spiccato: 中程度のduration + 高velocity variation
    - Mixed: その他
    
    Returns: "legato" | "staccato" | "spiccato" | "mixed"
    """
    try:
        mid = mido.MidiFile(midi_file)
    except Exception as e:
        print(f"[ERROR] Failed to parse {midi_file.name}: {e}")
        return "mixed"
    
    # Extract note events
    notes = []
    current_time = 0
    
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append({
                    'time': current_time,
                    'pitch': msg.pitch,
                    'velocity': msg.velocity,
                    'duration': 0,  # Will be filled by note_off
                })
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Find corresponding note_on
                for note in reversed(notes):
                    if note['pitch'] == msg.pitch and note['duration'] == 0:
                        note['duration'] = current_time - note['time']
                        break
    
    if not notes:
        return "mixed"
    
    # Remove notes without duration
    notes = [n for n in notes if n['duration'] > 0]
    
    if len(notes) < 10:
        return "mixed"
    
    # Calculate metrics
    durations = [n['duration'] for n in notes]
    velocities = [n['velocity'] for n in notes]
    
    avg_duration = sum(durations) / len(durations)
    avg_velocity = sum(velocities) / len(velocities)
    velocity_std = (sum((v - avg_velocity)**2 for v in velocities) / len(velocities)) ** 0.5
    
    # Calculate inter-onset intervals (IOI)
    notes_sorted = sorted(notes, key=lambda x: x['time'])
    iois = [notes_sorted[i+1]['time'] - notes_sorted[i]['time'] 
            for i in range(len(notes_sorted) - 1)]
    
    if not iois:
        return "mixed"
    
    avg_ioi = sum(iois) / len(iois)
    
    # Calculate overlap ratio
    overlaps = [n['duration'] / avg_ioi for n in notes if avg_ioi > 0]
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    
    # Classification
    duration_ratio = avg_duration / avg_ioi if avg_ioi > 0 else 0
    
    if avg_overlap > 0.8 or duration_ratio > 0.9:
        return "legato"
    elif duration_ratio < 0.3:
        return "staccato"
    elif velocity_std > 15 and 0.3 <= duration_ratio <= 0.6:
        return "spiccato"
    else:
        return "mixed"


def main():
    parser = argparse.ArgumentParser(description="URMP strings MIDI importer")
    parser.add_argument("--urmp-dir", type=str,
                        default="data/external/urmp",
                        help="URMP root directory")
    parser.add_argument("--output-dir", type=str,
                        default="data/external/urmp/raw/strings",
                        help="Output directory for processed MIDI files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run (no file output)")
    
    args = parser.parse_args()
    
    urmp_dir = pathlib.Path(args.urmp_dir)
    dataset_dir = urmp_dir / "Dataset"
    output_dir = pathlib.Path(args.output_dir)
    
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        print(f"[ERROR] Please download URMP dataset first:")
        print(f"[ERROR] bash scripts/download_external_datasets.sh urmp")
        return 1
    
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all strings MIDI files
    # URMP naming: {piece_id}_{instrument}.mid
    # Instruments: vn (violin), va (viola), vc (cello), db (double bass)
    
    strings_patterns = ["*_vn.mid", "*_va.mid", "*_vc.mid", "*_db.mid"]
    midi_files = []
    
    for pattern in strings_patterns:
        midi_files.extend(dataset_dir.glob(f"*/{pattern}"))
    
    print(f"[INFO] Found {len(midi_files)} strings MIDI files")
    
    # Statistics
    stats: Dict[str, int] = {
        "legato": 0,
        "staccato": 0,
        "spiccato": 0,
        "mixed": 0,
    }
    
    instrument_counts = {
        "vn": 0,  # violin
        "va": 0,  # viola
        "vc": 0,  # cello
        "db": 0,  # double bass
    }
    
    for midi_file in midi_files:
        try:
            # Extract instrument from filename
            # Example: 01_Spring_vn.mid → vn
            parts = midi_file.stem.split('_')
            instrument = parts[-1] if parts else "unknown"
            
            if instrument in instrument_counts:
                instrument_counts[instrument] += 1
            
            # Classify technique
            technique = classify_strings_technique(midi_file)
            stats[technique] += 1
            
            # Generate output filename
            # Format: {piece_id}_{instrument}_{technique}.mid
            piece_id = '_'.join(parts[:-1]) if len(parts) > 1 else midi_file.stem
            output_file = output_dir / f"{piece_id}_{instrument}_{technique}.mid"
            
            if args.dry_run:
                print(f"[DRY-RUN] {midi_file.name} → {output_file.name} ({technique})")
            else:
                # Copy with new name
                shutil.copy2(midi_file, output_file)
                print(f"[OK] {output_file.name} ({technique})")
        
        except Exception as e:
            print(f"[ERROR] {midi_file.name}: {e}")
            continue
    
    # Summary
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Total:     {len(midi_files)}")
    print(f"\n  By Technique:")
    print(f"    Legato:    {stats['legato']} 🔴 (Priority)")
    print(f"    Staccato:  {stats['staccato']}")
    print(f"    Spiccato:  {stats['spiccato']}")
    print(f"    Mixed:     {stats['mixed']}")
    print(f"\n  By Instrument:")
    print(f"    Violin:    {instrument_counts['vn']}")
    print(f"    Viola:     {instrument_counts['va']}")
    print(f"    Cello:     {instrument_counts['vc']}")
    print(f"    D.Bass:    {instrument_counts['db']}")
    print("="*50)
    
    if not args.dry_run:
        print(f"\n[COMPLETE] Output: {output_dir}")
        print(f"\n[NEXT STEP] Integrate into Stage1 pipeline:")
        print(f"  Edit: scripts/run_stage1_clean_multi.sh")
        print(f"  Add: URMP strings {output_dir} ...")
    
    return 0


if __name__ == "__main__":
    exit(main())
