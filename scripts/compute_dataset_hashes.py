#!/usr/bin/env python3
"""
datasets.lock のSHA1ハッシュを計算・更新するスクリプト

使用方法:
    python scripts/compute_dataset_hashes.py
    python scripts/compute_dataset_hashes.py --verify  # 検証のみ
"""

import hashlib
import os
from pathlib import Path
from typing import List, Tuple
import argparse


def compute_sha1(filepath: Path) -> str:
    """ファイルのSHA1ハッシュを計算"""
    sha1 = hashlib.sha1()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()


def get_file_size(filepath: Path) -> int:
    """ファイルサイズをバイト単位で取得"""
    return filepath.stat().st_size


def format_size(size_bytes: int) -> str:
    """バイトサイズを人間が読みやすい形式に変換"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f}MB"
    else:
        return f"{size_bytes / 1024 ** 3:.1f}GB"


def parse_lock_file(lock_path: Path) -> List[Tuple[str, str, str, str]]:
    """datasets.lock を解析して (相対パス, SHA1, サイズ, 説明) のリストを返す"""
    entries = []
    with open(lock_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # コメント行とヘッダー行をスキップ
            if not line or line.startswith('#') or line.startswith('==='):
                continue
            
            # パース: <path> <sha1> <size> <description>
            parts = line.split(None, 3)
            if len(parts) >= 4:
                rel_path, sha1, size, desc = parts
                # 説明からクォートを除去
                desc = desc.strip('"')
                entries.append((rel_path, sha1, size, desc))
    
    return entries


def update_lock_file(lock_path: Path, root_dir: Path, verify_only: bool = False) -> bool:
    """datasets.lock を更新または検証"""
    print(f"📂 Root directory: {root_dir}")
    print(f"🔒 Lock file: {lock_path}\n")
    
    entries = parse_lock_file(lock_path)
    updated_lines = []
    all_valid = True
    
    # ヘッダー読み込み
    with open(lock_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('==='):
                break
            updated_lines.append(line.rstrip())
    
    # 各エントリを処理
    for rel_path, old_sha1, old_size, desc in entries:
        filepath = root_dir / rel_path
        
        if not filepath.exists():
            print(f"⚠️  MISSING: {rel_path}")
            updated_lines.append(f"{rel_path}  <MISSING>  <MISSING>  \"{desc}\"")
            all_valid = False
            continue
        
        # ハッシュ計算
        print(f"🔍 Computing hash: {rel_path}...", end=' ')
        new_sha1 = compute_sha1(filepath)
        new_size = get_file_size(filepath)
        new_size_str = format_size(new_size)
        
        # 検証モード
        if verify_only:
            if old_sha1 == '<未計算>' or old_sha1 == '<MISSING>':
                print(f"❓ NOT LOCKED (hash: {new_sha1[:8]}...)")
            elif old_sha1 == new_sha1:
                print(f"✅ VERIFIED")
            else:
                print(f"❌ MISMATCH (expected: {old_sha1[:8]}..., got: {new_sha1[:8]}...)")
                all_valid = False
        else:
            # 更新モード
            if old_sha1 == '<未計算>' or old_sha1 == '<MISSING>':
                print(f"🆕 COMPUTED: {new_sha1[:8]}... ({new_size_str})")
            elif old_sha1 == new_sha1:
                print(f"✅ UNCHANGED ({new_size_str})")
            else:
                print(f"🔄 UPDATED: {old_sha1[:8]}... → {new_sha1[:8]}... ({new_size_str})")
        
        # 新しいエントリを追加
        updated_lines.append(f"{rel_path}  {new_sha1}  {new_size_str}  \"{desc}\"")
    
    # ファイル更新
    if not verify_only:
        print(f"\n💾 Writing updated lock file...")
        with open(lock_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(updated_lines))
            f.write('\n')
        print(f"✅ Lock file updated: {lock_path}")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description='Compute or verify dataset hashes')
    parser.add_argument('--verify', action='store_true', help='Verify mode (no updates)')
    parser.add_argument('--lock-file', type=Path, default='data/datasets.lock',
                       help='Path to datasets.lock file')
    args = parser.parse_args()
    
    root_dir = Path(__file__).parent.parent
    lock_path = root_dir / args.lock_file
    
    if not lock_path.exists():
        print(f"❌ Lock file not found: {lock_path}")
        return 1
    
    mode_str = "VERIFICATION" if args.verify else "UPDATE"
    print(f"\n{'='*60}")
    print(f"  datasets.lock {mode_str}")
    print(f"{'='*60}\n")
    
    all_valid = update_lock_file(lock_path, root_dir, verify_only=args.verify)
    
    print(f"\n{'='*60}")
    if args.verify:
        if all_valid:
            print("✅ All datasets verified successfully!")
            return 0
        else:
            print("❌ Verification failed - datasets do not match lock file")
            return 1
    else:
        print("✅ Lock file update complete!")
        return 0


if __name__ == '__main__':
    exit(main())
