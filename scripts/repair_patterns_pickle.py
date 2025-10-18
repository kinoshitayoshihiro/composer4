#!/usr/bin/env python3
"""
既存 Stage2 pattern pickles を canonical module名で再保存

目的:
- 古い pickle が __main__ などlegacy module名で保存されている場合、
  現在の extract_stage2_patterns module名で再保存し、
  将来は通常の pickle.load で読めるようにする。

使い方:
    python scripts/repair_patterns_pickle.py
    
処理:
1. data/patterns/stage2_*.pickle をバックアップ（.backup）
2. 互換ローダで読み込み
3. 通常の pickle.dump で再保存（canonical名）
"""
from pathlib import Path
import shutil
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.pickle_compat import resave_pickle

def main():
    patterns_dir = Path("data/patterns")
    if not patterns_dir.exists():
        print(f"❌ Patterns directory not found: {patterns_dir}")
        return
    
    pattern_files = sorted(patterns_dir.glob("stage2_*.pickle"))
    if not pattern_files:
        print(f"❌ No stage2 pattern files found in {patterns_dir}")
        return
    
    print(f"Found {len(pattern_files)} pattern files to repair:")
    for pf in pattern_files:
        print(f"  - {pf.name}")
    
    print("\n" + "="*60)
    
    for pf in pattern_files:
        backup_path = pf.with_suffix(".pickle.backup")
        
        # Backup
        print(f"\n📦 {pf.name}")
        print(f"  → Backing up to {backup_path.name}...")
        shutil.copy2(pf, backup_path)
        
        # Resave
        print(f"  → Resaving with canonical module names...")
        try:
            resave_pickle(
                str(pf),
                str(pf),
                rename_map={"__main__": "extract_stage2_patterns"}
            )
            print(f"  ✅ Success")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            # Restore backup
            print(f"  → Restoring from backup...")
            shutil.copy2(backup_path, pf)
    
    print("\n" + "="*60)
    print("✅ All pattern pickles repaired!")
    print(f"📁 Backups saved in {patterns_dir}/*.backup")
    print("\nYou can now safely use standard pickle.load() for these files.")

if __name__ == "__main__":
    main()
