#!/usr/bin/env python
"""Create mini training dataset matching existing MIDI files."""

from pathlib import Path
import pandas as pd

# Load full metadata
csv_path = Path("outputs/stage3/loop_summary_with_captions.csv")
midi_dir = Path("output/drumloops_cleaned/2")

df = pd.read_csv(csv_path)

# Get existing MIDI files (digest-based)
existing_digests = set()
for midi_file in midi_dir.glob("*.mid"):
    digest = midi_file.stem  # Remove .mid extension
    existing_digests.add(digest)

print(f"Found {len(existing_digests)} MIDI files")
print(f"CSV has {len(df)} rows")

# Filter CSV to only rows with existing MIDI files
df_filtered = df[df["file_digest"].isin(existing_digests)].copy()

print(f"Matched {len(df_filtered)} rows")

# Take first 50 for mini training
df_mini = df_filtered.head(50)

output_path = Path("outputs/stage3/mini_train_matched.csv")
df_mini.to_csv(output_path, index=False)

print(f"Saved {len(df_mini)} samples to {output_path}")
print(f"Sample file_digests: {df_mini['file_digest'].head(3).tolist()}")
