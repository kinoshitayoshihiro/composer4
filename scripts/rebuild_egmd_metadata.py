#!/usr/bin/env python3
"""
Rebuild E-GMD metadata with relative paths
"""
import pickle
from pathlib import Path

# Paths
output_cleaned = Path("output/rhythm_ai/egmd_cleaned")
output_metadata = Path("output/rhythm_ai/egmd_metadata")

print("🔧 Rebuilding E-GMD metadata with relative paths...")
print(f"  Cleaned dir: {output_cleaned}")
print(f"  Metadata dir: {output_metadata}")

# Find all cleaned files
cleaned_files = sorted(output_cleaned.glob("*/egmd_*.mid"))
print(f"\n✅ Found {len(cleaned_files)} cleaned files")

# Build metadata list
metadata_list = []
for i, f in enumerate(cleaned_files):
    if i % 500 == 0:
        print(f"  Processing: {i}/{len(cleaned_files)}")
    
    # Relative path from output_cleaned
    rel_path = str(f.relative_to(output_cleaned))
    
    # Extract hex char from directory
    hex_char = f.parent.name
    
    metadata_list.append({
        'original_path': '',  # Unknown (lost during Stage1)
        'cleaned_path': rel_path,  # e.g., "4/egmd_000000.mid"
        'md5': f'{hex_char}{"0"*31}',  # Placeholder MD5
        'filename': f.name,
        'file_index': i
    })

# Save in shards
shard_size = 500
num_shards = (len(metadata_list) + shard_size - 1) // shard_size

print(f"\n💾 Saving {num_shards} shards...")
for i in range(num_shards):
    start_idx = i * shard_size
    end_idx = min(start_idx + shard_size, len(metadata_list))
    shard_data = metadata_list[start_idx:end_idx]
    
    shard_path = output_metadata / f"drums_{i:04d}.pkl"
    with open(shard_path, 'wb') as f:
        pickle.dump(shard_data, f)
    
    print(f"  Shard {i:04d}: {len(shard_data)} records → {shard_path}")

# Save index
index_data = {
    'total_files': len(cleaned_files),
    'num_shards': num_shards,
    'shard_size': shard_size,
    'dataset_name': 'egmd',
    'md5_hashes': {}  # Empty (lost during Stage1)
}

index_path = output_metadata / "drums_index.pkl"
with open(index_path, 'wb') as f:
    pickle.dump(index_data, f)

print(f"\n✅ Metadata rebuilt:")
print(f"  Index: {index_path}")
print(f"  Shards: {num_shards}")
print(f"  Total files: {len(cleaned_files)}")

# Verify first shard
print(f"\n🔍 Verifying first shard...")
with open(output_metadata / "drums_0000.pkl", 'rb') as f:
    first_shard = pickle.load(f)

print(f"  Shard 0 entries: {len(first_shard)}")
print(f"  Sample entry:")
print(f"    cleaned_path: {first_shard[0]['cleaned_path']}")
print(f"    filename: {first_shard[0]['filename']}")
print(f"    file_index: {first_shard[0]['file_index']}")

print(f"\n✅ Done!")
