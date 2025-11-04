#!/usr/bin/env python3
"""
Rhythm AI Pickle Builder

Stage2特徴量 + Song Package → Pickle統合
  - lite/fat モード（既定: lite = featuresは外部Parquet参照）
  - family列のゆらぎに強い検出
  - IDマップ/重複/欠損/型情報などのメタ充実
  - song_package.yaml を再帰探索

Usage:
    python scripts/build_rhythm_pickle.py \
        --stage2-features output/rhythm_ai/rhythm_features_merged.parquet \
        --song-packages output/rhythm_ai/song_packages \
        --output output/rhythm_ai/rhythm_patterns.pickle \
        --metadata-out output/rhythm_ai/rhythm_metadata.json \
        --mode lite \
        --id-column loop_id
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib

import pandas as pd
import yaml
from tqdm import tqdm


def sha256_file(p: Path) -> Optional[str]:
    """ファイルのSHA256計算"""
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def pick_family_column(df: pd.DataFrame) -> Optional[str]:
    """Family列の自動検出（ゆらぎ対応）"""
    for cand in ("family_label", "family", "Family"):
        if cand in df.columns:
            return cand
    return None


def load_song_packages(packages_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Song Package読み込み（再帰探索）"""
    packages: Dict[str, Dict[str, Any]] = {}
    yaml_files = sorted(packages_dir.rglob("song_package.yaml"))
    
    for package_yaml in tqdm(yaml_files, desc="Loading packages"):
        try:
            with open(package_yaml, 'r', encoding='utf-8') as f:
                package_data = yaml.safe_load(f) or {}
            
            song_id = package_data.get('song_id') or str(package_yaml.parent.relative_to(packages_dir))
            packages[song_id] = package_data
        except Exception as e:
            logging.warning(f"Failed to read {package_yaml}: {e}")
    
    return packages


def build_pickle(
    stage2_features: Path,
    song_packages: Path,
    output: Path,
    metadata_out: Path,
    mode: str = "lite",
    id_column: str = "loop_id",
):
    """Pickle構築"""
    
    # Stage2特徴量読み込み
    print(f"📊 Loading Stage2 features: {stage2_features}")
    features_df = pd.read_parquet(stage2_features)
    features_df = features_df.reset_index(drop=True)
    
    print(f"  Total records: {len(features_df)}")
    print(f"  Columns: {list(features_df.columns)[:10]}...")
    
    # Song Package読み込み
    print(f"\n📦 Loading Song Packages: {song_packages}")
    packages = load_song_packages(song_packages)
    
    print(f"  Total packages: {len(packages)}")
    
    # 統合データ構築
    print(f"\n🔧 Building integrated pickle...")
    
    # family列の検出とユニーク
    fam_col = pick_family_column(features_df)
    if fam_col is not None and pd.api.types.is_string_dtype(features_df[fam_col]):
        features_df[fam_col] = features_df[fam_col].astype("category")
        family_labels = list(features_df[fam_col].cat.categories)
    elif fam_col is not None:
        family_labels = sorted(map(str, features_df[fam_col].dropna().unique().tolist()))
    else:
        family_labels = []
    
    # IDマップ（可能なら）
    id_index = {}
    id_nulls = 0
    id_dups = 0
    if id_column in features_df.columns:
        ids = features_df[id_column]
        id_nulls = int(ids.isna().sum())
        
        # 最初出現を優先（重複があっても推論時は先勝ちでよい運用）
        seen = set()
        for i, v in enumerate(ids):
            if pd.isna(v):
                continue
            if v not in seen:
                id_index[str(v)] = int(i)
                seen.add(v)
        id_dups = int(len(ids) - len(seen) - id_nulls)
    
    # 型/欠損の簡易統計
    dtypes = {c: str(t) for c, t in features_df.dtypes.items()}
    nulls = {c: int(features_df[c].isna().sum()) for c in features_df.columns}
    non_nulls = {c: int(len(features_df) - nulls[c]) for c in features_df.columns}
    
    # Preview rows（JSON化のため全ての値を安全に変換）
    preview_df = features_df.head(3).copy()
    preview_rows = []
    for _, row in preview_df.iterrows():
        row_dict = {}
        for col, val in row.items():
            # 配列チェックを最初に
            if hasattr(val, 'tolist'):  # numpy array
                row_dict[col] = val.tolist()
            elif hasattr(val, 'item'):  # numpy scalar
                row_dict[col] = val.item()
            elif pd.isna(val):
                row_dict[col] = None
            elif isinstance(val, (int, float, str, bool)):
                row_dict[col] = val
            else:
                row_dict[col] = str(val)
        preview_rows.append(row_dict)
    
    features_sha256 = sha256_file(stage2_features)
    created_utc = datetime.utcnow().isoformat() + "Z"
    
    # features の保存モード
    if mode not in ("lite", "fat"):
        raise ValueError("--mode must be 'lite' or 'fat'")
    
    if mode == "fat":
        features_payload = features_df.to_dict(orient='records')
        features_path = str(stage2_features)
    else:
        features_payload = None
        features_path = str(stage2_features)
    
    pickle_data = {
        "version": "1.1.0",
        "type": "rhythm_patterns",
        "created_utc": created_utc,
        "mode": mode,
        "features_path": features_path,
        "features": features_payload,  # liteでは None
        "song_packages": packages,
        "metadata": {
            "num_features": int(len(features_df)),
            "num_packages": int(len(packages)),
            "feature_columns": list(features_df.columns),
            "family_column": fam_col,
            "family_labels": family_labels,
            "id_column": id_column if id_column in features_df.columns else None,
            "id_nulls": id_nulls,
            "id_duplicates": id_dups,
            "features_sha256": features_sha256,
            "dtypes": dtypes,
            "nulls": nulls,
            "non_nulls": non_nulls,
            "preview_rows": preview_rows,
        },
        "indices": {
            "id_index": id_index  # 可能なときのみ中身が入る
        }
    }
    
    # Pickle保存
    print(f"\n💾 Saving pickle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'wb') as f:
        pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # メタデータ保存
    print(f"📝 Saving metadata: {metadata_out}")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_out, 'w', encoding='utf-8') as f:
        json.dump(pickle_data['metadata'], f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"Features:      {pickle_data['metadata']['num_features']}")
    print(f"Song Packages: {pickle_data['metadata']['num_packages']}")
    print(f"Family Labels: {pickle_data['metadata']['family_labels']}")
    print(f"Mode:          {pickle_data['mode']}")
    if pickle_data['mode'] == 'lite':
        print(f"Features path: {pickle_data['features_path']}")
    print(f"ID column:     {pickle_data['metadata']['id_column']}")
    print(f"ID nulls:      {pickle_data['metadata']['id_nulls']}")
    print(f"ID duplicates: {pickle_data['metadata']['id_duplicates']}")
    print(f"Pickle size:   {output.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rhythm AI Pickle Builder"
    )
    parser.add_argument(
        '--stage2-features',
        type=Path,
        required=True,
        help='Stage2 features parquet file'
    )
    parser.add_argument(
        '--song-packages',
        type=Path,
        required=True,
        help='Song packages directory'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output pickle file'
    )
    parser.add_argument(
        '--metadata-out',
        type=Path,
        required=True,
        help='Metadata JSON output'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='lite',
        choices=['lite', 'fat'],
        help="lite: featuresは外部Parquet参照 / fat: featuresをピクル内に内包"
    )
    parser.add_argument(
        '--id-column',
        type=str,
        default='loop_id',
        help="IDカラム名（例: loop_id/pattern_id 等）"
    )
    
    args = parser.parse_args()
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 処理実行
    build_pickle(
        stage2_features=args.stage2_features,
        song_packages=args.song_packages,
        output=args.output,
        metadata_out=args.metadata_out,
        mode=args.mode,
        id_column=args.id_column
    )


if __name__ == '__main__':
    main()
