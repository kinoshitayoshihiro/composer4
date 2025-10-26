#!/usr/bin/env python3
"""
WAV → LOCAL LAMDA ビルダー

WAVステム（MUSDB18/MoisesDB）からLOCAL LAMDA互換データを生成。
- KILO: chroma → bar-level chord sequence
- SIGNATURES: beat tracking → time signature（1/4救済込み）
- TOTALS: chroma/RMS → 256-bin histogram
- META: 簡易メタデータ（audio_proxy）

Usage:
    python scripts/local_lamda/build_local_from_wav.py \\
        --wav-root data/musdb18_wavs \\
        --out-dir data/LOCAL_LAMDA/wav_version/musdb18 \\
        --stem accompaniment

Dependencies:
    pip install librosa soundfile
"""
import argparse
import csv
import hashlib
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install librosa soundfile")
    exit(1)

# 除外ディレクトリ
EXCLUDE_DIRS = {"temp", "quarantine", ".cache", ".trash", ".git", "__pycache__"}

# 簡易コードテンプレート（maj/min）
# 実運用では既存のchord_analyzerに接続推奨
CHORD_TEMPLATES = {
    "C:maj": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
    "C#:maj": np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
    "D:maj": np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
    "D#:maj": np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),
    "E:maj": np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
    "F:maj": np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
    "F#:maj": np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]),
    "G:maj": np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]),
    "G#:maj": np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]),
    "A:maj": np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
    "A#:maj": np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
    "B:maj": np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]),
    "C:min": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
    "C#:min": np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
    "D:min": np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
    "D#:min": np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
    "E:min": np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
    "F:min": np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
    "F#:min": np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
    "G:min": np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]),
    "G#:min": np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]),
    "A:min": np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
    "A#:min": np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
    "B:min": np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
}


def stable_id(rel_path: str) -> str:
    """相対パスからLOCAL IDを生成"""
    return "LOCAL:" + hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:20]


def iter_wavs(root: Path, prefer_stems=("accompaniment", "other", "mixture")):
    """
    WAVファイルを収集（優先度順）

    MUSDB18: track/mix.wav, drums.wav, bass.wav, other.wav, vocals.wav
    MoisesDB: track/guitar.wav, bass.wav, drums.wav, vocals.wav, ...

    Args:
        root: WAVルートディレクトリ
        prefer_stems: 優先するステム名（前方一致）

    Yields:
        (相対パス, WAVファイルパス)
    """
    # ハーモニック系のステム（MoisesDB用）
    harmonic_stems = {
        "guitar",
        "guitar_1",
        "guitar_2",
        "piano",
        "keys",
        "pad",
        "strings",
        "synth",
        "organ",
        "rhodes",
        "harp",
        "brass",
        "woodwinds",
        "accompaniment",
        "other",
        "mixture",
    }

    for track_dir in root.rglob("*"):
        if not track_dir.is_dir():
            continue
        if any(excl in track_dir.parts for excl in EXCLUDE_DIRS):
            continue

        # WAVファイルがあるディレクトリのみ処理
        wav_files = list(track_dir.glob("*.wav"))
        if not wav_files:
            continue

        # 優先度順にステムを探す
        best_stem = None
        for stem_name in prefer_stems:
            candidates = [w for w in wav_files if stem_name in w.stem.lower()]
            if candidates:
                best_stem = candidates[0]
                break

        # 優先ステムが見つからない場合、ハーモニック系を探す
        if not best_stem:
            for wav in wav_files:
                stem_lower = wav.stem.lower()
                if any(h in stem_lower for h in harmonic_stems):
                    # drums/vocals/percussionは除外
                    if not any(x in stem_lower for x in ("drum", "vocal", "perc")):
                        best_stem = wav
                        break

        if best_stem:
            rel_path = str(best_stem.relative_to(root))
            yield rel_path, best_stem


def extract_chroma_bars(wav_path, sr=22050, hop=512):
    """
    WAVからbar-level chromaを抽出

    Returns:
        (bar_chroma, tempo, beats): (n_bars, 12), float, (n_beats,)
    """
    try:
        y, sr = librosa.load(wav_path, sr=sr, mono=True)
    except Exception as e:
        print(f"[skip] {wav_path.name}: {e}")
        return None, 0, []

    # ビート検出
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="time")

    if len(beats) < 8:
        return None, tempo, beats

    # 4拍=1小節と仮定
    beats = np.asarray(beats)
    n_bars = len(beats) // 4
    bars = [beats[i * 4 : (i + 1) * 4] for i in range(n_bars)]

    # Chroma特徴量
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    times = librosa.times_like(C, sr=sr, hop_length=hop)

    bar_chroma = []
    for bar_beats in bars:
        # 小節内のフレームを抽出
        idx = np.where((times >= bar_beats[0]) & (times < bar_beats[-1]))[0]
        if len(idx) == 0:
            bar_chroma.append(np.zeros(12))
            continue
        # 中央値で頑健化
        bar_chroma.append(np.median(C[:, idx], axis=1))

    return np.stack(bar_chroma), tempo, beats


def chroma_to_chord(chroma_vec):
    """
    Chromaベクトルから最尤コードを推定

    Args:
        chroma_vec: (12,) numpy array

    Returns:
        chord_name: "C:maj", "A:min", "N" (no chord)
    """
    norm = np.linalg.norm(chroma_vec)
    if norm < 1e-6:
        return "N"

    chroma_norm = chroma_vec / norm
    best_score, best_name = -1e9, "N"

    for name, template in CHORD_TEMPLATES.items():
        score = float(np.dot(chroma_norm, template))
        if score > best_score:
            best_score, best_name = score, name

    return best_name


def rescue_1_4_signature(sig_list):
    """
    1/4救済: 1/4と4/4が混在する場合、1/4を4/4に統合

    Args:
        sig_list: ["4/4", "1/4", "4/4", ...]

    Returns:
        rescued_list: ["4/4", "4/4", "4/4", ...]
    """
    c = Counter(sig_list)
    if "1/4" in c and "4/4" in c:
        # 1/4を4/4に統合
        return ["4/4" if s == "1/4" else s for s in sig_list]
    return sig_list


def main():
    ap = argparse.ArgumentParser(
        description="WAV → LOCAL LAMDA builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--wav-root",
        required=True,
        help="WAV root directory (contains track folders)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for LOCAL LAMDA files",
    )
    ap.add_argument(
        "--stem",
        default="accompaniment",
        help="Preferred stem name (accompaniment, other, mixture, ...)",
    )
    args = ap.parse_args()

    root = Path(args.wav_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"❌ WAV root not found: {root}")
        exit(1)

    prefer_stems = (args.stem, "other", "mixture")

    # データ収集
    kilo_data = []
    signatures_data = []
    totals = {
        "pitch_hist_256": np.zeros(256, dtype=np.int64),
        "dur_hist_256": np.zeros(256, dtype=np.int64),
        "vel_hist_256": np.zeros(256, dtype=np.int64),
    }
    id_rows = []

    print(f"📂 Scanning WAV files in {root}...")
    print(f"🎵 Preferred stems: {prefer_stems}")
    print()

    wav_files = list(iter_wavs(root, prefer_stems))
    print(f"✓ Found {len(wav_files)} tracks")
    print()

    for i, (rel_path, wav_path) in enumerate(wav_files, 1):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(wav_files)}...")

        # Chroma抽出
        bar_chroma, tempo, beats = extract_chroma_bars(wav_path)
        if bar_chroma is None or len(bar_chroma) == 0:
            continue

        # LOCAL ID生成
        local_id = stable_id(rel_path)

        # KILO: bar-level chord sequence
        chord_seq = [chroma_to_chord(c) for c in bar_chroma]
        kilo_data.append([local_id, {"tokens": chord_seq, "bars": len(chord_seq)}])

        # SIGNATURES: 拍子（4/4仮定 + 1/4救済）
        sig_list = ["4/4" if len(beats) // 4 >= 4 else "1/4"] * len(chord_seq)
        sig_list = rescue_1_4_signature(sig_list)
        sig_counter = Counter(sig_list)
        sig_rows = [[sig, count] for sig, count in sig_counter.items()]
        signatures_data.append([local_id, sig_rows])

        # TOTALS: chroma → 256-bin pitch histogram
        chroma_sum = np.sum(bar_chroma, axis=0)  # (12,)
        for pitch_class, val in enumerate(chroma_sum):
            # 12 pitch classes → 256 bins (linear mapping)
            bin_idx = int((pitch_class / 12.0) * 256)
            totals["pitch_hist_256"][bin_idx] += int(val * 1000)

        # ID_MAP（絶対パスも記録してコピー不要）
        id_rows.append(
            {
                "local_id": local_id,
                "relative_path": rel_path,
                "absolute_path": str(wav_path.resolve()),  # WAVの実際の場所
                "source": f"wav:{args.stem}",
            }
        )

    # 保存
    print(f"\n💾 Saving LOCAL LAMDA files...")

    with open(out_dir / "LOCAL_KILO_CHORDS_DATA.pickle", "wb") as f:
        pickle.dump(kilo_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ LOCAL_KILO_CHORDS_DATA.pickle ({len(kilo_data)} entries)")

    with open(out_dir / "LOCAL_SIGNATURES_DATA.pickle", "wb") as f:
        pickle.dump(signatures_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ LOCAL_SIGNATURES_DATA.pickle ({len(signatures_data)} entries)")

    with open(out_dir / "LOCAL_TOTALS.pickle", "wb") as f:
        pickle.dump(totals, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ LOCAL_TOTALS.pickle")

    with open(out_dir / "LOCAL_ID_MAP.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["local_id", "relative_path", "absolute_path", "source"]
        )
        writer.writeheader()
        writer.writerows(id_rows)
    print(f"  ✓ LOCAL_ID_MAP.csv ({len(id_rows)} entries)")

    # META（簡易版、空で出力）
    meta_data = []
    for row in id_rows:
        meta_data.append(
            [
                row["local_id"],
                {
                    "total_number_of_tracks": 1,
                    "total_number_of_notes": 0,  # WAVからは不明
                    "source": "audio_proxy",
                },
            ]
        )
    with open(out_dir / "LOCAL_META_DATA_000001.pickle", "wb") as f:
        pickle.dump(meta_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ LOCAL_META_DATA_000001.pickle (audio_proxy)")

    print(f"\n✅ WAV→LOCAL LAMDA complete!")
    print(f"📁 Output: {out_dir}")
    print(f"📊 Tracks: {len(kilo_data)}")


if __name__ == "__main__":
    main()
