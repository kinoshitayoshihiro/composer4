#!/usr/bin/env python3
import pickle
from pathlib import Path

# 以前のresumeファイルが残っているか確認
resume_pkl = Path("output/drums_metadata/resume.pkl")
if resume_pkl.exists():
    with open(resume_pkl, "rb") as f:
        resume_data = pickle.load(f)
    
    print("⚠️  Resume file found:")
    print(f"   Path: {resume_pkl}")
    print(f"   Files processed: {len(resume_data.get('processed', []))}")
    print(f"   Last run: Already processed all files")
    print()
    print("🔧 Solution:")
    print("   Option 1: Delete resume file to force reprocessing")
    print("   Option 2: Run without --resume flag")
else:
    print("✅ No resume file found")

# 既存shardを確認
shard_dir = Path("output/drums_metadata")
if shard_dir.exists():
    shards = sorted(shard_dir.glob("drums_*.pkl"))
    if shards:
        print(f"\n⚠️  Found {len(shards)} existing shard files")
        print("   These will be kept in --resume mode")
    else:
        print("\n✅ No existing shards")
