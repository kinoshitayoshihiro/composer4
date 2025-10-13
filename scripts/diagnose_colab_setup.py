"""
Colab Diagnostic Script for Stage2 Setup
Google Driveからダウンロードしたデータの構造を診断
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

def find_directory(root: Path, name: str, max_depth: int = 3) -> List[Path]:
    """指定されたディレクトリ名を再帰的に検索"""
    results = []
    
    def search(current: Path, depth: int):
        if depth > max_depth:
            return
        try:
            for item in current.iterdir():
                if item.is_dir():
                    if item.name == name:
                        results.append(item)
                    search(item, depth + 1)
        except PermissionError:
            pass
    
    search(root, 0)
    return results

def get_directory_stats(path: Path) -> Dict[str, Any]:
    """ディレクトリの統計情報を取得"""
    file_count = 0
    dir_count = 0
    total_size = 0
    
    try:
        for item in path.rglob('*'):
            if item.is_file():
                file_count += 1
                total_size += item.stat().st_size
            elif item.is_dir():
                dir_count += 1
    except PermissionError:
        pass
    
    return {
        "files": file_count,
        "directories": dir_count,
        "total_size_mb": total_size / (1024 * 1024)
    }

def main():
    print("🔍 Colab Stage2 Data Diagnostic")
    print("=" * 60)
    
    # 基本パス
    repo_root = Path("/content/composer4")
    download_root = Path("/content/composer4/_drive_download")
    
    # 1) リポジトリの確認
    print("\n📂 Repository Check:")
    if repo_root.exists():
        print(f"  ✅ Repository found: {repo_root}")
        print(f"     Files: {len(list(repo_root.glob('*')))}")
    else:
        print(f"  ❌ Repository not found: {repo_root}")
        return
    
    # 2) ダウンロードディレクトリの確認
    print("\n📥 Download Directory Check:")
    if download_root.exists():
        print(f"  ✅ Download directory found: {download_root}")
        print(f"\n  Directory structure (top 2 levels):")
        for item in sorted(download_root.rglob('*')):
            if item.relative_to(download_root).parts.__len__() <= 2:
                indent = "    " * (len(item.relative_to(download_root).parts) - 1)
                icon = "📁" if item.is_dir() else "📄"
                print(f"    {indent}{icon} {item.name}")
    else:
        print(f"  ❌ Download directory not found: {download_root}")
        print(f"     Please run the download script first")
        return
    
    # 3) 'output' ディレクトリの検索
    print("\n🔎 Searching for 'output' directory:")
    output_dirs = find_directory(download_root, "output", max_depth=3)
    
    if not output_dirs:
        print("  ❌ No 'output' directory found")
        print("\n  💡 Possible issues:")
        print("     1. Google Drive folder doesn't contain 'output'")
        print("     2. Folder structure is different than expected")
        print("     3. Download incomplete or failed")
        print("\n  🔍 All directories in download root:")
        for item in sorted(download_root.rglob('*')):
            if item.is_dir():
                print(f"     📁 {item.relative_to(download_root)}")
    else:
        print(f"  ✅ Found {len(output_dirs)} 'output' director{'y' if len(output_dirs) == 1 else 'ies'}:")
        for i, output_dir in enumerate(output_dirs, 1):
            print(f"\n  [{i}] {output_dir}")
            stats = get_directory_stats(output_dir)
            print(f"      Files: {stats['files']}")
            print(f"      Subdirectories: {stats['directories']}")
            print(f"      Total size: {stats['total_size_mb']:.2f} MB")
            
            # サブディレクトリの確認
            print(f"      Subdirectories:")
            for subdir in sorted(output_dir.iterdir()):
                if subdir.is_dir():
                    sub_stats = get_directory_stats(subdir)
                    print(f"        📁 {subdir.name}: {sub_stats['files']} files, {sub_stats['total_size_mb']:.2f} MB")
    
    # 4) シンボリックリンクの確認
    print("\n🔗 Symbolic Link Check:")
    output_link = repo_root / "output"
    if output_link.exists():
        if output_link.is_symlink():
            target = output_link.resolve()
            print(f"  ✅ Symbolic link exists")
            print(f"     Link: {output_link}")
            print(f"     Target: {target}")
            if target.exists():
                print(f"     ✅ Target is valid")
            else:
                print(f"     ❌ Target is broken")
        else:
            print(f"  ⚠️ 'output' exists but is not a symbolic link")
            print(f"     Type: {'directory' if output_link.is_dir() else 'file'}")
    else:
        print(f"  ❌ No symbolic link at {output_link}")
    
    # 5) Stage2に必要なディレクトリの確認
    print("\n📋 Stage2 Required Directories:")
    required_dirs = [
        "drum_metadata",
        "drum_cleaned",
        "stage2_drum_iter1"
    ]
    
    output_path = output_link if output_link.exists() else (output_dirs[0] if output_dirs else None)
    if output_path:
        for req_dir in required_dirs:
            dir_path = output_path / req_dir
            if dir_path.exists():
                stats = get_directory_stats(dir_path)
                print(f"  ✅ {req_dir}: {stats['files']} files, {stats['total_size_mb']:.2f} MB")
            else:
                print(f"  ⚠️ {req_dir}: NOT FOUND (will be created during Stage2)")
    else:
        print("  ❌ Cannot check (no output directory found)")
    
    # 6) 推奨アクション
    print("\n💡 Recommended Actions:")
    if not output_dirs:
        print("  1. Verify Google Drive folder structure")
        print("  2. Check if 'output' directory exists in shared folder")
        print("  3. Re-run download script with correct folder URL")
        print("  4. Alternative: Manually upload 'output' to /content/composer4/")
    elif not output_link.exists() or not output_link.is_symlink():
        print("  1. Create symbolic link manually:")
        print(f"     ln -sfn {output_dirs[0]} {repo_root / 'output'}")
    else:
        print("  ✅ Setup looks good! Ready for Stage2")
        print("  Next step: cd /content/composer4 && bash scripts/run_stage2_drum.sh")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
