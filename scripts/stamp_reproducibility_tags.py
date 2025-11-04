#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stamp_reproducibility_tags.py
--------------------------------------------------
SongPackageに再現性タグを自動焼き込み

追加される情報:
  - arranger_version: Gitコミットハッシュ or タグ
  - backend_profile: features_backend設定のスナップショット
  - kpi_gate_version: gate_prod.yamlのハッシュ
  - generation_timestamp: 生成日時（ISO 8601形式）
  - python_version: Python実行バージョン

Usage:
  python3 scripts/stamp_reproducibility_tags.py \
      --song-dir song_packages/suno_project/song_001 \
      --arranger-config configs/arranger_weights.yaml \
      --gate-config configs/gate_prod.yaml

Output:
  - song_package.yaml: 再現性タグが追加される
"""

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    print("❌ pyyaml が見つかりません。`pip install pyyaml` を実行してください。")
    sys.exit(1)


def get_git_commit_hash(repo_path: Path = Path(".")) -> Optional[str]:
    """Gitコミットハッシュを取得"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]  # 短縮版
    except Exception:
        return None


def get_git_tag(repo_path: Path = Path(".")) -> Optional[str]:
    """Git最新タグを取得"""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def compute_file_hash(file_path: Path) -> str:
    """ファイルのSHA256ハッシュを計算"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # 短縮版


def extract_backend_profile(arranger_config_path: Path) -> Dict[str, Any]:
    """features_backend設定のスナップショットを抽出"""
    with open(arranger_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    backend = config.get("features_backend", {})
    
    # 重要な設定項目のみ抽出（詳細は省略）
    profile = {
        "chords": backend.get("chords", "unknown"),
        "chroma": backend.get("chroma", "unknown"),
        "hat_density": backend.get("hat_density", "unknown"),
        "beats": backend.get("beats", "unknown"),
        "downbeats": backend.get("downbeats", "unknown"),
        "loudness": backend.get("loudness", "unknown"),
    }
    
    # Essentia設定がある場合は追加
    if "essentia" in backend:
        profile["essentia_hpcp_harmonics"] = backend["essentia"].get("hpcp", {}).get("harmonics", "unknown")
    
    # Chordino設定がある場合は追加
    if "chordino" in backend:
        profile["chordino_min_confidence"] = backend["chordino"].get("min_confidence", "unknown")
    
    return profile


def stamp_reproducibility_tags(
    song_dir: Path,
    arranger_config: Path,
    gate_config: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    再現性タグを生成して song_package.yaml に焼き込み
    """
    song_pkg_path = song_dir / "song_package.yaml"
    
    # 既存のsong_package.yaml読み込み
    if song_pkg_path.exists():
        with open(song_pkg_path, 'r', encoding='utf-8') as f:
            song_pkg = yaml.safe_load(f) or {}
    else:
        song_pkg = {}
    
    # 再現性タグ生成
    tags: Dict[str, Any] = {
        "arranger_version": get_git_tag() or get_git_commit_hash() or "unknown",
        "backend_profile": extract_backend_profile(arranger_config),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    
    # KPI Gate設定ハッシュ
    if gate_config and gate_config.exists():
        tags["kpi_gate_config_hash"] = compute_file_hash(gate_config)
    
    # Arranger設定ハッシュ
    tags["arranger_config_hash"] = compute_file_hash(arranger_config)
    
    # 既存タグと統合
    song_pkg["reproducibility"] = tags
    
    # 書き戻し
    with open(song_pkg_path, 'w', encoding='utf-8') as f:
        yaml.dump(song_pkg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    return tags


def main():
    ap = argparse.ArgumentParser(description="SongPackageに再現性タグを焼き込み")
    ap.add_argument("--song-dir", type=Path, required=True, help="SongPackageディレクトリ")
    ap.add_argument("--arranger-config", type=Path, default=Path("configs/arranger_weights.yaml"), help="Arranger設定YAML")
    ap.add_argument("--gate-config", type=Path, default=Path("configs/gate_prod.yaml"), help="KPI Gate設定YAML（任意）")
    args = ap.parse_args()
    
    # 検証
    if not args.song_dir.exists():
        print(f"❌ Song directory not found: {args.song_dir}")
        sys.exit(1)
    
    if not args.arranger_config.exists():
        print(f"❌ Arranger config not found: {args.arranger_config}")
        sys.exit(1)
    
    # タグ焼き込み
    print(f"🔖 Stamping reproducibility tags to: {args.song_dir / 'song_package.yaml'}")
    tags = stamp_reproducibility_tags(args.song_dir, args.arranger_config, args.gate_config)
    
    # 結果表示
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Reproducibility Tags:")
    print(f"  Arranger Version: {tags['arranger_version']}")
    print(f"  Backend Profile:")
    for k, v in tags["backend_profile"].items():
        print(f"    {k:25s}: {v}")
    print(f"  Generation Timestamp: {tags['generation_timestamp']}")
    print(f"  Python Version: {tags['python_version']}")
    if "kpi_gate_config_hash" in tags:
        print(f"  KPI Gate Config Hash: {tags['kpi_gate_config_hash']}")
    print(f"  Arranger Config Hash: {tags['arranger_config_hash']}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ Reproducibility tags saved to: {args.song_dir / 'song_package.yaml'}")


if __name__ == "__main__":
    main()
