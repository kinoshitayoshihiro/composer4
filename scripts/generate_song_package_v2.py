#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_song_package_v2.py (統合レイアウト準拠版)
- Multi-dataset support (--dataset can be given multiple times or comma-separated)
- Optional dataset-level diagnostics injection (vocal_features/mix_diagnostics)
- Optional audio_chordmap path injection
- Optional index CSV output summarizing generated packages

「正本＝JSON/YAML/Parquet、DB＝索引、pickleは使わない」方式準拠。
bars.parquet を必須のハブとして、sections/chordmap/anchors を仕様の真として束ねます。

Example:
  python scripts/generate_song_package_v2.py \
    --base "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/Los-Angeles-MIDI/LOCAL_LAMDA" \
    --dataset moisesdb --dataset musdb18 \
    --include-dataset-level --add-audio-chordmap \
    --code-version "local_lamda_moises_integration.py@<git-hash>" \
    --index-out "/tmp/song_packages_index.csv"
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path


def read_json_safe(p: Path):
    """JSON読み込み（失敗時None）"""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_md5_head(p: Path, n=16):
    """ファイルのMD5ハッシュ先頭n文字"""
    try:
        b = p.read_bytes()
        return hashlib.md5(b).hexdigest()[:n]
    except Exception:
        return None


def write_yaml(data: dict, out_path: Path):
    """YAML出力（PyYAMLあれば使用、なければ簡易書式）"""
    try:
        import yaml
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return
    except Exception:
        # Minimal YAML writer（PyYAML未導入時のフォールバック）
        def dump(d, indent=0):
            lines = []
            prefix = "  " * indent
            for k, v in d.items():
                if v is None:
                    lines.append(f"{prefix}{k}:")
                elif isinstance(v, dict):
                    lines.append(f"{prefix}{k}:")
                    lines.extend(dump(v, indent + 1))
                elif isinstance(v, list):
                    lines.append(f"{prefix}{k}:")
                    for item in v:
                        if isinstance(item, dict):
                            lines.append(f"{prefix}  -")
                            for k2, v2 in item.items():
                                lines.append(f"{prefix}    {k2}: {repr(v2)}")
                        else:
                            lines.append(f"{prefix}  - {repr(item)}")
                else:
                    lines.append(f"{prefix}{k}: {repr(v)}")
            return lines
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(dump(data)) + "\n", encoding="utf-8")


def make_rel(from_dir: Path, target: Path):
    """相対パス生成（失敗時は絶対パス）"""
    try:
        return os.path.relpath(target, start=from_dir)
    except Exception:
        return str(target)


def normalize_datasets(ds_args):
    """dataset引数を正規化（カンマ区切り対応＋重複排除）"""
    out = []
    for item in ds_args:
        out.extend([s.strip() for s in item.split(",") if s.strip()])
    # de-duplicate preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def build_for_dataset(
    base: Path,
    dataset: str,
    code_version: str,
    include_dataset_level: bool = False,
    add_audio_chordmap: bool = False,
    dry=False,
    verbose=True
):
    """
    単一datasetのsong_package.yaml生成
    
    Args:
        base: LOCAL_LAMDA ベースディレクトリ
        dataset: データセット名（moisesdb, musdb18, etc）
        code_version: コードバージョン（例: "local_lamda_moises_integration.py@abc123"）
        include_dataset_level: dataset-level diagnostics（vocal_features/mix_diagnostics）を含めるか
        add_audio_chordmap: per-song audio_chordmap.yaml を含めるか
        dry: Dry-run（書き込みなし）
        verbose: 進捗表示
    
    Returns:
        (count, index_rows): 生成数とインデックス行リスト
    """
    # 統合レイアウト準拠パス
    wav_ds_root = base / "Local_Lamda_wav" / "wav_guide" / dataset
    midi_root = base / "Local_Lamda_midi" / "midi_guide"
    specs_root = base / "Local_Lamda_specs"

    if not wav_ds_root.exists():
        print(f"[WARN] WAV dataset root not found: {wav_ds_root}", file=sys.stderr)
        return 0, []

    # dataset-level diagnostics (optional)
    ds_vocal_feat = wav_ds_root / "vocal_features.parquet"
    ds_mix_diag = wav_ds_root / "mix_diagnostics.parquet"

    song_dirs = sorted([p for p in wav_ds_root.iterdir() if p.is_dir()])
    count = 0
    index_rows = []
    
    for sdir in song_dirs:
        song_id = sdir.name
        
        # WAV系成果物（Stage2）
        beat_grid = sdir / "beat_grid.json"
        accent_grid = sdir / "accent_grid.json"
        audio_chordmap = sdir / "audio_chordmap.yaml"
        bars_parquet = sdir / f"{song_id}.bars.parquet"
        
        # bars.parquet は必須（ハブ）
        if not bars_parquet.exists():
            if verbose:
                print(f"[SKIP] {song_id}: bars.parquet missing", file=sys.stderr)
            continue

        # manifest*.json (optional - file_id取得用)
        manifest_json = None
        for cand in sdir.glob("manifest*.json"):
            manifest_json = cand
            break

        # MIDI系成果物（Stage1）
        midi_song_dir = midi_root / song_id
        stage1_json = midi_song_dir / "stage1_clean.json"
        stage1_mid = midi_song_dir / "stage1_clean.mid"
        
        # MIDIガイドパート（任意）
        guides = {
            "piano": midi_song_dir / "piano.mid",
            "guitar": midi_song_dir / "guitar.mid",
            "bass": midi_song_dir / "bass.mid",
            "drums": midi_song_dir / "drums.mid",
            "vocal": midi_song_dir / "vocal.mid",
        }

        # 楽曲仕様三点（Stage3 - 任意）
        specs_dir = specs_root / song_id
        sections = specs_dir / "sections.json"
        chordmap = specs_dir / "chordmap.json"
        anchors = specs_dir / "lyric_anchors.json"

        # ID生成
        run_id = "local-" + datetime.datetime.now().isoformat(timespec="seconds")
        
        # MIDI content_id（Stage1から取得、なければMD5）
        midi_content_id = None
        if stage1_json.exists():
            s1 = read_json_safe(stage1_json)
            if s1:
                midi_content_id = s1.get("content_id")
        if midi_content_id is None and stage1_mid.exists():
            midi_content_id = compute_md5_head(stage1_mid, n=16)

        # WAV file_id（manifestから取得）
        wav_file_id = None
        if manifest_json and manifest_json.exists():
            mf = read_json_safe(manifest_json)
            if mf:
                wav_file_id = mf.get("file_id")

        # 出力先（MIDIガイド側）
        out_pkg_dir = midi_song_dir
        
        # song_package.yaml構築
        pkg = {
            "version": "1.0",
            "ids": {
                "song_id": song_id,
                "run_id": run_id,
                "code_version": code_version,
                "midi_content_id": midi_content_id,
                "wav_file_id": wav_file_id,
                "dataset": dataset
            },
            "spec": {},
            "hub": {"bars_parquet": make_rel(out_pkg_dir, bars_parquet)},
            "guides": {"midi": {}},
            "diagnostics": {}
        }
        
        # 楽曲仕様三点（任意）
        if sections.exists():
            pkg["spec"]["sections"] = make_rel(out_pkg_dir, sections)
        if chordmap.exists():
            pkg["spec"]["chordmap"] = make_rel(out_pkg_dir, chordmap)
        if anchors.exists():
            pkg["spec"]["anchors"] = make_rel(out_pkg_dir, anchors)

        # MIDIガイド（Stage1 clean + パート別）
        if stage1_mid.exists():
            pkg["guides"]["midi"]["stage1_clean"] = make_rel(out_pkg_dir, stage1_mid)
        for name, pth in guides.items():
            if pth.exists():
                pkg["guides"]["midi"][name] = make_rel(out_pkg_dir, pth)

        # WAV系diagnostics（Stage2成果物）
        if beat_grid.exists():
            pkg["diagnostics"]["wav_beat_grid"] = make_rel(out_pkg_dir, beat_grid)
        if accent_grid.exists():
            pkg["diagnostics"]["wav_accent_grid"] = make_rel(out_pkg_dir, accent_grid)
        if add_audio_chordmap and audio_chordmap.exists():
            pkg["diagnostics"]["wav_audio_chordmap"] = make_rel(out_pkg_dir, audio_chordmap)

        # dataset-level diagnostics（任意）
        if include_dataset_level:
            pkg["diagnostics"]["dataset_level"] = {}
            if ds_vocal_feat.exists():
                pkg["diagnostics"]["dataset_level"]["vocal_features"] = make_rel(
                    out_pkg_dir, ds_vocal_feat
                )
            if ds_mix_diag.exists():
                pkg["diagnostics"]["dataset_level"]["mix_diagnostics"] = make_rel(
                    out_pkg_dir, ds_mix_diag
                )

        # 書き込み
        out_path = out_pkg_dir / "song_package.yaml"
        if not dry:
            write_yaml(pkg, out_path)
            if verbose:
                print(f"[OK] {dataset}/{song_id} -> {out_path}")
        else:
            if verbose:
                print(f"[DRY] {dataset}/{song_id} -> {out_path}")

        count += 1
        index_rows.append({
            "dataset": dataset,
            "song_id": song_id,
            "package_path": str(out_path),
            "bars_parquet": str(bars_parquet),
            "midi_content_id": midi_content_id or "",
            "wav_file_id": wav_file_id or ""
        })

    return count, index_rows


def main():
    ap = argparse.ArgumentParser(description="Song Package自動生成（統合レイアウト準拠版）")
    ap.add_argument("--base", required=True, help="Path to LOCAL_LAMDA base")
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="wav_guide dataset name(s). You can repeat this flag or pass comma-separated names (e.g., 'moisesdb,musdb18')."
    )
    ap.add_argument("--code-version", default="local_lamda_moises_integration.py@unknown")
    ap.add_argument(
        "--include-dataset-level",
        action="store_true",
        help="Inject dataset-level vocal_features/mix_diagnostics into each package's diagnostics"
    )
    ap.add_argument(
        "--add-audio-chordmap",
        action="store_true",
        help="Include per-song audio_chordmap.yaml path in diagnostics"
    )
    ap.add_argument("--index-out", default=None, help="Write a CSV index summarizing generated packages")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    datasets = normalize_datasets(args.dataset)

    total = 0
    all_rows = []
    
    for ds in datasets:
        print(f"\n=== Processing dataset: {ds} ===")
        count, rows = build_for_dataset(
            base=base,
            dataset=ds,
            code_version=args.code_version,
            include_dataset_level=args.include_dataset_level,
            add_audio_chordmap=args.add_audio_chordmap,
            dry=args.dry_run,
            verbose=True
        )
        total += count
        all_rows.extend(rows)

    print(f"\n[DONE] {total} song_package.yaml {'would be ' if args.dry_run else ''}written.")

    # CSV index出力
    if args.index_out and all_rows and not args.dry_run:
        import csv
        out_csv = Path(args.index_out)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as w:
            writer = csv.DictWriter(w, fieldnames=["dataset", "song_id", "package_path", "bars_parquet", "midi_content_id", "wav_file_id"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[INDEX] wrote {out_csv}")


if __name__ == "__main__":
    main()
