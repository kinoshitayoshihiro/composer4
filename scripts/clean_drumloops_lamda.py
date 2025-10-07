#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drum Loops LAMDa-Style Cleaner and Deduplicator

Based on Los Angeles MIDI Dataset Maker methodology:
1. MD5 hash deduplication (identical files)
2. Pitch sum deduplication (musical similarity)
3. Pitch distribution histogram (final fingerprint)

Project Los Angeles / Tegridy Code 2023
Adapted for Drum Loops 2025
"""

import os
import sys
import hashlib
import shutil
from collections import Counter
from tqdm import tqdm
import random

# Add TMIDIX to path
sys.path.append("data/Los-Angeles-MIDI/CODE")
import TMIDIX

# ===================================================================================
# Configuration
# ===================================================================================

DATASET_ADDR = "data/loops"
OUTPUT_DIR = "data/cleaned_drumloops"
PICKLE_OUTPUT = "data/drumloops_metadata.pickle"

# LAMDa filtering criteria
MIN_NOTES = 8  # Minimum notes for a valid drum loop (lower than LAMDa's 256)
MAX_FILE_SIZE = 1000000  # 1MB max file size
START_FILE_NUMBER = 0
SAVE_INTERVAL = 5000  # Save progress every N files

# ===================================================================================
# Setup
# ===================================================================================

print("=" * 70)
print("Drum Loops LAMDa-Style Cleaner")
print("=" * 70)

# Create output directories
print("Creating output directories...")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create hexadecimal subdirs (0-f) like LAMDa
output_dirs_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]

for o in output_dirs_list:
    subdir = os.path.join(OUTPUT_DIR, o)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

print("Done!")

# ===================================================================================
# Collect files
# ===================================================================================

print("=" * 70)
print("Scanning drum loops directory...")
print("This may take a while...")
print("=" * 70)

filez = []
for dirpath, dirnames, filenames in os.walk(DATASET_ADDR):
    for file in filenames:
        if file.endswith(".mid") or file.endswith(".midi"):
            # Skip symbolic links if needed
            full_path = os.path.join(dirpath, file)
            if not os.path.islink(full_path):
                filez.append(full_path)

print("Found", len(filez), "MIDI files")
print("=" * 70)

# Randomize to get diverse sample
print("Randomizing file list...")
random.shuffle(filez)

# ===================================================================================
# Process files with LAMDa methodology
# ===================================================================================

print("=" * 70)
print("Processing drum loops with LAMDa deduplication...")
print("=" * 70)

input_files_count = START_FILE_NUMBER
files_count = 0

# LAMDa deduplication data structures
all_md5_names = []
all_pitches_sums = []
all_pitches_counts = []
all_pitches_and_counts = []

# Additional metadata for drum loops
drumloops_metadata = []

for f in tqdm(filez[START_FILE_NUMBER:], desc="Processing"):
    try:
        input_files_count += 1

        fn = os.path.basename(f)

        # LAMDa Step 1: File size filter
        file_size = os.path.getsize(f)

        if file_size <= MAX_FILE_SIZE:

            # Read file data
            fdata = open(f, "rb").read()

            # LAMDa Step 2: MD5 hash for exact duplicates
            md5sum = hashlib.md5(fdata).hexdigest()
            md5name = str(md5sum) + ".mid"

            if str(md5sum) not in all_md5_names:

                # Parse MIDI with TMIDIX
                score = TMIDIX.midi2score(fdata)

                # Extract all events
                events_matrix = []
                itrack = 1

                while itrack < len(score):
                    for event in score[itrack]:
                        events_matrix.append(event)
                    itrack += 1

                events_matrix.sort(key=lambda x: x[1])

                # Extract notes only
                notes = [y for y in events_matrix if y[0] == "note"]

                # LAMDa Step 3: Minimum notes check
                if len(notes) >= MIN_NOTES:

                    times = [n[1] for n in notes]
                    durs = [n[2] for n in notes]

                    # Validate timing
                    if min(times) >= 0 and min(durs) >= 0:
                        # Check for chord presence (polyphonic)
                        if len(times) > len(set(times)):

                            all_md5_names.append(str(md5sum))

                            # LAMDa Step 4: Pitch sum deduplication
                            pitches = [n[4] for n in notes]
                            pitches_sum = sum(pitches)

                            if pitches_sum not in all_pitches_sums:

                                all_pitches_sums.append(pitches_sum)

                                # LAMDa Step 5: Pitch distribution histogram
                                pitches_and_counts = sorted(
                                    [[key, val] for key, val in Counter(pitches).most_common()],
                                    reverse=True,
                                    key=lambda x: x[1],
                                )
                                pitches_counts = [p[1] for p in pitches_and_counts]

                                # Final deduplication check
                                if pitches_counts not in all_pitches_counts:

                                    # Save file to hexadecimal subdir
                                    shutil.copy2(
                                        f, os.path.join(OUTPUT_DIR, str(md5name[0]), md5name)
                                    )

                                    all_pitches_counts.append(pitches_counts)
                                    all_pitches_and_counts.append(pitches_and_counts)

                                    # Extract drum loop metadata
                                    patches = [n[3] for n in notes]
                                    patches_counts = Counter(patches).most_common()
                                    vels = [n[5] for n in notes]

                                    # Parse filename for genre/tempo info
                                    basename = os.path.basename(f)
                                    parts = (
                                        basename.replace(".mid", "").replace(".midi", "").split("_")
                                    )

                                    metadata = {
                                        "md5": md5sum,
                                        "original_file": f,
                                        "cleaned_file": os.path.join(
                                            OUTPUT_DIR, str(md5name[0]), md5name
                                        ),
                                        "num_notes": len(notes),
                                        "pitches_sum": pitches_sum,
                                        "pitches_counts": pitches_counts,
                                        "pitches_and_counts": pitches_and_counts,
                                        "patches_counts": patches_counts,
                                        "avg_velocity": sum(vels) / len(vels),
                                        "duration_ms": max(times),
                                        "filename_parts": parts,
                                    }

                                    drumloops_metadata.append(metadata)

                                    files_count += 1

                                    # Auto-save progress
                                    if files_count % SAVE_INTERVAL == 0:
                                        print()
                                        print("=" * 70)
                                        print(
                                            f"CHECKPOINT: Saving progress at {files_count} files..."
                                        )
                                        TMIDIX.Tegridy_Any_Pickle_File_Writer(
                                            drumloops_metadata, PICKLE_OUTPUT.replace(".pickle", "")
                                        )
                                        print(f"Saved {files_count} unique drum loops")
                                        print(
                                            f"Good files ratio: {files_count / input_files_count:.2%}"
                                        )
                                        print("=" * 70)

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Interrupted! Saving current progress...")
        break

    except Exception as ex:
        # Silently skip bad files
        continue

# ===================================================================================
# Final save
# ===================================================================================

print()
print("=" * 70)
print("Saving final results...")
print("=" * 70)

TMIDIX.Tegridy_Any_Pickle_File_Writer(drumloops_metadata, PICKLE_OUTPUT.replace(".pickle", ""))

print("=" * 70)
print("RESULTS:")
print("=" * 70)
print(f"Total files scanned: {input_files_count}")
print(f"Unique drum loops: {files_count}")
print(f"Good files ratio: {files_count / input_files_count:.2%}")
print(f"Reduction: {100 - (files_count / input_files_count * 100):.1f}%")
print("=" * 70)
print(f"Cleaned files saved to: {OUTPUT_DIR}")
print(f"Metadata saved to: {PICKLE_OUTPUT}")
print("=" * 70)
print("Done! 🥁")
print("=" * 70)
