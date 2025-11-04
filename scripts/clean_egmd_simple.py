#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-GMD Simple Cleaner
Simplified MIDI cleaning for E-GMD v1.0.0 dataset
"""

import os
import sys
import argparse
import shutil
import pickle
import hashlib
from pathlib import Path

# Add TMIDIX to path
sys.path.append("data/Los-Angeles-MIDI/CODE")
try:
    import TMIDIX
except ImportError:
    print("WARNING: TMIDIX not found, using basic MIDI loading")
    TMIDIX = None

def scan_midi_files(input_dir, extension=".midi"):
    """Scan directory for MIDI files"""
    print(f"Scanning {input_dir} for {extension} files...")
    files = list(Path(input_dir).rglob(f"*{extension}"))
    print(f"Found {len(files)} files")
    return files

def load_midi_with_tmidix(filepath):
    """Load MIDI file using TMIDIX"""
    try:
        if TMIDIX:
            score = TMIDIX.midi2single_track_ms_score(str(filepath))
            return score
        else:
            # Fallback: just check if file is readable
            with open(filepath, 'rb') as f:
                data = f.read()
            return len(data) > 0
    except Exception as e:
        return None

def compute_md5(filepath):
    """Compute MD5 hash of file"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def clean_egmd_dataset(args):
    """Main cleaning function"""
    
    # Setup
    input_dir = Path(args.input_dir)
    output_cleaned = Path(args.output_cleaned)
    output_metadata = Path(args.output_metadata)
    
    output_cleaned.mkdir(parents=True, exist_ok=True)
    output_metadata.mkdir(parents=True, exist_ok=True)
    
    # Scan files
    midi_files = scan_midi_files(input_dir, args.file_extension)
    
    if len(midi_files) == 0:
        print("ERROR: No MIDI files found!")
        return
    
    # Process files
    print(f"\nProcessing {len(midi_files)} files...")
    print(f"Max workers: {args.max_workers}")
    print(f"Shard size: {args.shard_size}")
    
    cleaned_files = []
    metadata_list = []
    md5_hashes = set()
    duplicates = 0
    errors = 0
    
    print(f"Processing {len(midi_files)} files...")
    for i, midi_file in enumerate(midi_files):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(midi_files)} ({100*i/len(midi_files):.1f}%)")
        
        try:
            # Compute MD5 for deduplication
            md5 = compute_md5(midi_file)
            if md5 is None:
                errors += 1
                continue
            
            if md5 in md5_hashes:
                duplicates += 1
                continue
            
            md5_hashes.add(md5)
            
            # Try to load MIDI
            score = load_midi_with_tmidix(midi_file)
            if score is None:
                errors += 1
                continue
            
            # Copy to output (preserve directory structure for identification)
            # Use hex-based directory structure like LAMDa
            hex_char = md5[0]  # First char of MD5 hash
            output_subdir = output_cleaned / hex_char
            output_subdir.mkdir(exist_ok=True)
            
            # Generate clean filename
            clean_name = f"{args.dataset_name}_{len(cleaned_files):06d}.mid"
            output_path = output_subdir / clean_name
            
            shutil.copy2(midi_file, output_path)
            
            # Store metadata
            # Use relative path for cleaned_path (relative to output_cleaned dir)
            cleaned_path_rel = str(output_path.relative_to(output_cleaned))
            
            metadata = {
                'original_path': str(midi_file),
                'cleaned_path': cleaned_path_rel,  # Relative to output_cleaned
                'md5': md5,
                'filename': midi_file.name,
                'file_index': len(cleaned_files)
            }
            
            metadata_list.append(metadata)
            cleaned_files.append(output_path)
            
        except Exception as e:
            if args.verbose:
                print(f"Error processing {midi_file}: {e}")
            errors += 1
    
    print(f"\nCleaning complete!")
    print(f"  Total files: {len(midi_files)}")
    print(f"  Cleaned: {len(cleaned_files)}")
    print(f"  Duplicates: {duplicates}")
    print(f"  Errors: {errors}")
    
    # Save metadata in shards
    print(f"\nSaving metadata in shards (size: {args.shard_size})...")
    shard_size = args.shard_size
    num_shards = (len(metadata_list) + shard_size - 1) // shard_size
    
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(metadata_list))
        shard_data = metadata_list[start_idx:end_idx]
        
        shard_path = output_metadata / f"drums_{i:04d}.pkl"
        with open(shard_path, 'wb') as f:
            pickle.dump(shard_data, f)
    
    # Save index
    index_data = {
        'total_files': len(cleaned_files),
        'num_shards': num_shards,
        'shard_size': shard_size,
        'dataset_name': args.dataset_name,
        'md5_hashes': list(md5_hashes)
    }
    
    index_path = output_metadata / "drums_index.pkl"
    with open(index_path, 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"✅ Metadata saved: {num_shards} shards + index")
    print(f"   Index: {index_path}")

def main():
    parser = argparse.ArgumentParser(description="E-GMD Simple Cleaner")
    parser.add_argument("--input-dir", required=True, help="Input directory containing MIDI files")
    parser.add_argument("--output-cleaned", required=True, help="Output directory for cleaned MIDI files")
    parser.add_argument("--output-metadata", required=True, help="Output directory for metadata pickles")
    parser.add_argument("--dataset-name", default="egmd", help="Dataset name prefix")
    parser.add_argument("--file-extension", default=".midi", help="MIDI file extension")
    parser.add_argument("--max-workers", type=int, default=8, help="Max parallel workers")
    parser.add_argument("--shard-size", type=int, default=500, help="Metadata shard size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("E-GMD Simple Cleaner")
    print("=" * 70)
    print(f"Input: {args.input_dir}")
    print(f"Output (cleaned): {args.output_cleaned}")
    print(f"Output (metadata): {args.output_metadata}")
    print(f"Extension: {args.file_extension}")
    print("=" * 70)
    
    clean_egmd_dataset(args)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
