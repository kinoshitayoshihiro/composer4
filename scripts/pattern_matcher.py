#!/usr/bin/env python3
"""
pattern_matcher.py
------------------
Suno SongPackage（sections.json/chordmap.json/bars.parquet）から
Rhythm/Harmonyの既存Pickleで「近いパターン」をTop-Kで引き当てます。

出力:
  - matches_rhythm.json    … Top-K rhythm candidates（loop_id, family, tempo, score等）
  - matches_harmony.json   … (任意) harmony candidates（song_id, key, score等）

Usage:
  python3 scripts/pattern_matcher.py \
    --song-dir song_packages/suno_project/song_001 \
    --rhythm-pickle output/rhythm_ai/rhythm_patterns.pickle \
    --harmony-pickle output/harmony_wav/harmony_patterns.pickle \
    --topk 5 --per-section
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


def _safe_read_parquet(p: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(p)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet {p}: {e}")


def _load_song_package(song_dir: Path) -> Dict[str, Any]:
    song_dir = Path(song_dir)
    pkg_yaml = song_dir / "song_package.yaml"

    # bars.parquet: bars.parquet or {song_id}.bars.parquet
    bars_parquet = song_dir / "bars.parquet"
    if not bars_parquet.exists():
        # Fallback: {song_id}.bars.parquet
        bars_candidates = list(song_dir.glob("*.bars.parquet"))
        if bars_candidates:
            bars_parquet = bars_candidates[0]

    chordmap_json = song_dir / "chordmap.json"
    sections_json = song_dir / "sections.json"

    if not pkg_yaml.exists():
        raise FileNotFoundError(f"song_package.yaml not found in {song_dir}")
    if not bars_parquet.exists():
        raise FileNotFoundError(
            f"bars.parquet not found in {song_dir} (also checked *.bars.parquet)"
        )

    # chordmap/sections: オプション（Rhythm専用時は不要）
    chordmap_exists = chordmap_json.exists()
    sections_exists = sections_json.exists()

    with open(pkg_yaml, "r", encoding="utf-8") as f:
        pkg = yaml.safe_load(f)

    bars = _safe_read_parquet(bars_parquet)

    chordmap = {}
    if chordmap_exists:
        with open(chordmap_json, "r", encoding="utf-8") as f:
            chordmap = json.load(f)

    sections = {}
    if sections_exists:
        with open(sections_json, "r", encoding="utf-8") as f:
            sections = json.load(f)

    return {"meta": pkg, "bars": bars, "chordmap": chordmap, "sections": sections}


def _tempo_from_meta_or_bars(pkg: Dict[str, Any]) -> float:
    meta = pkg["meta"]
    # Try top-level tempo first (rhythm packages)
    tempo = meta.get("tempo")
    if tempo:
        return float(tempo)

    # Fallback: nested meta.tempo_bpm (suno packages)
    tempo = meta.get("meta", {}).get("tempo_bpm")
    if tempo:
        return float(tempo)

    # sections.jsonのtempo_map平均
    sections = pkg.get("sections", {})
    tempo_map = sections.get("tempo_map", [])
    if len(tempo_map) > 0:
        tempo_vals = [t[1] if isinstance(t, list) else t for t in tempo_map]
        return float(np.mean(tempo_vals))

    return 120.0


def _family_hint_from_bars(bars: pd.DataFrame) -> str:
    # swing_target の中央値で判定（>0.05→SWING_8 それ以外 STRAIGHT_8）
    swing_med = float(bars["swing_target"].median()) if "swing_target" in bars.columns else 0.0
    return "SWING_8" if swing_med > 0.05 else "STRAIGHT_8"


def _density_hint_from_bars(bars: pd.DataFrame) -> float:
    if "density_target" in bars.columns:
        return float(bars["density_target"].median())
    return 4.0


def _load_rhythm_pickle(rhythm_pickle: Path) -> pd.DataFrame:
    with open(rhythm_pickle, "rb") as f:
        blob = pickle.load(f)

    # v1.1.0 liteモード想定: features_path を参照して外部Parquetを読む
    if isinstance(blob, dict) and blob.get("mode") == "lite":
        features_path = blob.get("features_path")
        if features_path and Path(features_path).exists():
            return pd.read_parquet(features_path)
        else:
            raise RuntimeError(f"lite mode pickle but features_path not found: {features_path}")

    # fatモード: そのままfeatures配列をDataFrameに
    if isinstance(blob, dict) and "features" in blob:
        return pd.DataFrame(blob["features"])

    raise RuntimeError("rhythm pickle 形式を解釈できません")


def _load_harmony_pickle(harmony_pickle: Optional[Path]) -> Optional[pd.DataFrame]:
    if harmony_pickle is None:
        return None

    with open(harmony_pickle, "rb") as f:
        blob = pickle.load(f)

    if isinstance(blob, dict) and blob.get("mode") == "lite":
        features_path = blob.get("features_path")
        if features_path and Path(features_path).exists():
            return pd.read_parquet(features_path)

    if isinstance(blob, dict) and "features" in blob:
        return pd.DataFrame(blob["features"])

    return None


def _score_rhythm_candidates(
    df: pd.DataFrame,
    song_tempo: float,
    family_hint: str,
    density_hint: float,
    topk: int = 5,
    tempo_tau: float = 15.0,
    w_tempo: float = 0.45,
    w_family: float = 0.35,
    w_density: float = 0.20,
) -> List[Dict[str, Any]]:
    # 必須カラムを事前に安全取得
    tempo_col = "tempo_bpm" if "tempo_bpm" in df.columns else None
    family_col = None
    for cand in ("family_label", "family", "Family"):
        if cand in df.columns:
            family_col = cand
            break
    density_col = (
        "hat_density"
        if "hat_density" in df.columns
        else ("density" if "density" in df.columns else None)
    )
    id_col = "loop_id" if "loop_id" in df.columns else ("id" if "id" in df.columns else None)

    if tempo_col is None or family_col is None or density_col is None or id_col is None:
        raise RuntimeError(
            f"Required columns not found: tempo={tempo_col}, family={family_col}, density={density_col}, id={id_col}"
        )

    cand = df[[id_col, tempo_col, family_col, density_col]].copy()
    cand = cand.dropna()

    # スコア成分
    tempo_score = np.exp(-np.abs(cand[tempo_col].astype(float) - song_tempo) / tempo_tau)
    family_score = (cand[family_col].astype(str) == family_hint).astype(float)
    density_score = np.exp(-np.abs(cand[density_col].astype(float) - density_hint) / 3.0)

    score = w_tempo * tempo_score + w_family * family_score + w_density * density_score

    cand["score"] = score
    top = cand.sort_values("score", ascending=False).head(topk)

    results: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        results.append(
            {
                "loop_id": str(row[id_col]),
                "tempo_bpm": float(row[tempo_col]),
                "family": str(row[family_col]),
                "density": float(row[density_col]),
                "score": float(row["score"]),
            }
        )

    return results


def _extract_chord_sequence(chordmap: Dict[str, Any]) -> List[str]:
    # chordmap.json (events[{time,root,quality,...}])
    events = chordmap.get("events", [])
    seq = []
    for ev in events:
        root = ev.get("root", "C")
        quality = ev.get("quality", "maj")
        # 簡易化: root+quality
        seq.append(f"{root}{quality}")
    return seq


def _score_harmony_candidates(
    df_h: Optional[pd.DataFrame],
    chord_seq: List[str],
    topk: int = 5,
) -> List[Dict[str, Any]]:
    if df_h is None or len(df_h) == 0 or len(chord_seq) == 0:
        return []

    # v1: 先頭コード一致 + シーケンス長の近さ で簡易スコア
    id_col = (
        "song_id"
        if "song_id" in df_h.columns
        else ("id" if "id" in df_h.columns else df_h.columns[0])
    )
    first_col = "first_chord" if "first_chord" in df_h.columns else None
    length_col = "length" if "length" in df_h.columns else None

    # 必須がなければあきらめる（将来: ローマン化DTW）
    if first_col is None or length_col is None:
        return []

    first = chord_seq[0]
    length = len(chord_seq)

    df = df_h[[id_col, first_col, length_col]].copy()
    df["s_first"] = (df[first_col].astype(str) == first).astype(float)
    df["s_len"] = np.exp(-np.abs(df[length_col].astype(float) - length) / 8.0)
    df["score"] = 0.6 * df["s_first"] + 0.4 * df["s_len"]

    top = df.sort_values("score", ascending=False).head(topk)
    res: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        res.append(
            {
                "song_id": str(row[id_col]),
                "first_chord": str(row[first_col]),
                "length": int(row[length_col]),
                "score": float(row["score"]),
            }
        )

    return res


def main():
    ap = argparse.ArgumentParser(
        description="Pattern Matcher (Rhythm/Harmony) for Suno SongPackage"
    )
    ap.add_argument("--song-dir", type=Path, required=True, help="SongPackage directory")
    ap.add_argument(
        "--rhythm-pickle",
        type=Path,
        required=True,
        help="rhythm_patterns.pickle (lite/fatどちらでも)",
    )
    ap.add_argument(
        "--harmony-pickle", type=Path, default=None, help="(optional) harmony_patterns.pickle"
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="出力先（既定: song-dir）")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--per-section", action="store_true", help="セクション別Top-K（将来拡張）")
    args = ap.parse_args()

    pkg = _load_song_package(args.song_dir)
    bars = pkg["bars"]
    chordmap = pkg["chordmap"]

    tempo = _tempo_from_meta_or_bars(pkg)
    family_hint = _family_hint_from_bars(bars)
    density_hint = _density_hint_from_bars(bars)

    print(f"📊 Song Analysis:")
    print(f"  Tempo: {tempo:.1f} BPM")
    print(f"  Family Hint: {family_hint}")
    print(f"  Density Hint: {density_hint:.1f}")
    print()

    df_r = _load_rhythm_pickle(args.rhythm_pickle)
    print(f"✅ Rhythm Pickle: {len(df_r)} patterns loaded")

    df_h = _load_harmony_pickle(args.harmony_pickle)
    if df_h is not None:
        print(f"✅ Harmony Pickle: {len(df_h)} patterns loaded")

    rhythm_topk = _score_rhythm_candidates(df_r, tempo, family_hint, density_hint, topk=args.topk)
    print(f"\n🎯 Top-{args.topk} Rhythm Matches:")
    for i, match in enumerate(rhythm_topk, 1):
        print(
            f"  {i}. {match['loop_id']} (score={match['score']:.3f}, tempo={match['tempo_bpm']:.1f}, family={match['family']})"
        )

    chord_seq = _extract_chord_sequence(chordmap)
    harmony_topk = _score_harmony_candidates(df_h, chord_seq, topk=args.topk)

    if len(harmony_topk) > 0:
        print(f"\n🎯 Top-{args.topk} Harmony Matches:")
        for i, match in enumerate(harmony_topk, 1):
            print(f"  {i}. {match['song_id']} (score={match['score']:.3f})")

    out_dir = args.out_dir or args.song_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "matches_rhythm.json", "w", encoding="utf-8") as f:
        json.dump({"topk": args.topk, "matches": rhythm_topk}, f, indent=2, ensure_ascii=False)

    if len(harmony_topk) > 0:
        with open(out_dir / "matches_harmony.json", "w", encoding="utf-8") as f:
            json.dump({"topk": args.topk, "matches": harmony_topk}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Output:")
    print(f"  - {out_dir / 'matches_rhythm.json'}")
    if len(harmony_topk) > 0:
        print(f"  - {out_dir / 'matches_harmony.json'}")


if __name__ == "__main__":
    main()
