#!/usr/bin/env python3
"""
SoundFont Manager - SF2ファイルのハッシュ管理・検証

機能:
- SF2ファイルのSHA256計算・記録
- soundfonts.lock管理（バージョン固定）
- ハッシュ検証（不一致時に警告）
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


class SoundFontManager:
    """SoundFontファイルのバージョン管理・検証"""
    
    def __init__(self, lock_file: Path = Path("data/soundfonts.lock")):
        """
        Args:
            lock_file: ハッシュ記録ファイルのパス
        """
        self.lock_file = lock_file
        self.hashes: Dict[str, str] = {}
        
        if self.lock_file.exists():
            self._load_lock()
    
    def _load_lock(self) -> None:
        """soundfonts.lockをロード"""
        with open(self.lock_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.hashes = data.get('soundfonts', {})
    
    def _save_lock(self) -> None:
        """soundfonts.lockを保存"""
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "description": "SoundFont file hashes for version control",
            "soundfonts": self.hashes
        }
        
        with open(self.lock_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(self.hashes)} soundfont hashes to {self.lock_file}")
    
    @staticmethod
    def calculate_hash(sf2_path: Path, chunk_size: int = 8192) -> str:
        """
        SF2ファイルのSHA256ハッシュを計算
        
        Args:
            sf2_path: SF2ファイルのパス
            chunk_size: 読み込みチャンクサイズ（bytes）
        
        Returns:
            SHA256ハッシュ（hex文字列）
        """
        if not sf2_path.exists():
            raise FileNotFoundError(f"SoundFont not found: {sf2_path}")
        
        sha256 = hashlib.sha256()
        
        with open(sf2_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def register(self, sf2_path: Path, name: Optional[str] = None) -> str:
        """
        SF2ファイルをハッシュ計算して登録
        
        Args:
            sf2_path: SF2ファイルのパス
            name: 登録名（省略時はファイル名）
        
        Returns:
            計算されたハッシュ
        """
        if not sf2_path.exists():
            raise FileNotFoundError(f"SoundFont not found: {sf2_path}")
        
        key = name or sf2_path.name
        file_hash = self.calculate_hash(sf2_path)
        
        self.hashes[key] = {
            "path": str(sf2_path),
            "sha256": file_hash,
            "size_bytes": sf2_path.stat().st_size
        }
        
        print(f"✅ Registered: {key}")
        print(f"   Path: {sf2_path}")
        print(f"   SHA256: {file_hash[:16]}...")
        print(f"   Size: {sf2_path.stat().st_size:,} bytes")
        
        return file_hash
    
    def verify(self, sf2_path: Path, name: Optional[str] = None) -> Tuple[bool, str]:
        """
        SF2ファイルのハッシュを検証
        
        Args:
            sf2_path: 検証するSF2ファイルのパス
            name: 登録名（省略時はファイル名）
        
        Returns:
            (検証成功, メッセージ)
        """
        if not sf2_path.exists():
            return False, f"❌ SoundFont not found: {sf2_path}"
        
        key = name or sf2_path.name
        
        if key not in self.hashes:
            return False, f"⚠️  {key} not registered in {self.lock_file}"
        
        expected_hash = self.hashes[key]['sha256']
        actual_hash = self.calculate_hash(sf2_path)
        
        if actual_hash == expected_hash:
            return True, f"✅ {key} hash verified: {actual_hash[:16]}..."
        else:
            return False, (
                f"❌ {key} hash mismatch!\n"
                f"   Expected: {expected_hash[:16]}...\n"
                f"   Actual:   {actual_hash[:16]}...\n"
                f"   → SoundFont file may have been modified or corrupted"
            )
    
    def register_all(self, sf2_dir: Path) -> int:
        """
        ディレクトリ内の全SF2ファイルを登録
        
        Args:
            sf2_dir: SF2ファイルが格納されたディレクトリ
        
        Returns:
            登録されたファイル数
        """
        if not sf2_dir.exists():
            print(f"⚠️  Directory not found: {sf2_dir}")
            return 0
        
        sf2_files = list(sf2_dir.glob("*.sf2")) + list(sf2_dir.glob("*.SF2"))
        
        if not sf2_files:
            print(f"⚠️  No .sf2 files found in {sf2_dir}")
            return 0
        
        print(f"\n🔍 Found {len(sf2_files)} SoundFont file(s) in {sf2_dir}")
        
        for sf2_path in sf2_files:
            self.register(sf2_path)
        
        self._save_lock()
        return len(sf2_files)
    
    def verify_all(self, sf2_dir: Path) -> Tuple[int, int]:
        """
        ディレクトリ内の全SF2ファイルを検証
        
        Args:
            sf2_dir: SF2ファイルが格納されたディレクトリ
        
        Returns:
            (成功数, 失敗数)
        """
        if not sf2_dir.exists():
            print(f"⚠️  Directory not found: {sf2_dir}")
            return 0, 0
        
        sf2_files = list(sf2_dir.glob("*.sf2")) + list(sf2_dir.glob("*.SF2"))
        
        if not sf2_files:
            print(f"⚠️  No .sf2 files found in {sf2_dir}")
            return 0, 0
        
        print(f"\n🔍 Verifying {len(sf2_files)} SoundFont file(s)...")
        
        success_count = 0
        fail_count = 0
        
        for sf2_path in sf2_files:
            is_valid, message = self.verify(sf2_path)
            print(f"  {message}")
            
            if is_valid:
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n📊 Verification Summary:")
        print(f"   ✅ Success: {success_count}")
        print(f"   ❌ Failed:  {fail_count}")
        
        return success_count, fail_count


def main():
    """CLI エントリポイント"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SoundFont Manager - SF2ファイルのハッシュ管理・検証"
    )
    parser.add_argument(
        'action',
        choices=['register', 'verify', 'register-all', 'verify-all'],
        help='実行するアクション'
    )
    parser.add_argument(
        '--sf2',
        type=Path,
        help='SF2ファイルまたはディレクトリのパス'
    )
    parser.add_argument(
        '--name',
        help='登録名（省略時はファイル名）'
    )
    parser.add_argument(
        '--lock-file',
        type=Path,
        default=Path('data/soundfonts.lock'),
        help='ロックファイルのパス'
    )
    
    args = parser.parse_args()
    
    manager = SoundFontManager(lock_file=args.lock_file)
    
    if args.action == 'register':
        if not args.sf2:
            parser.error("--sf2 が必要です")
        manager.register(args.sf2, args.name)
        manager._save_lock()
    
    elif args.action == 'verify':
        if not args.sf2:
            parser.error("--sf2 が必要です")
        is_valid, message = manager.verify(args.sf2, args.name)
        print(message)
        exit(0 if is_valid else 1)
    
    elif args.action == 'register-all':
        if not args.sf2:
            parser.error("--sf2 （ディレクトリ）が必要です")
        count = manager.register_all(args.sf2)
        print(f"\n✅ Registered {count} SoundFont(s)")
    
    elif args.action == 'verify-all':
        if not args.sf2:
            parser.error("--sf2 （ディレクトリ）が必要です")
        success, fail = manager.verify_all(args.sf2)
        exit(0 if fail == 0 else 1)


if __name__ == '__main__':
    main()
