#!/usr/bin/env python3
"""
Enhanced Bar-Level Chord Extraction with Prior Fusion (Standalone Edition)

単独動作版: stem_harmony_bar_level.py への依存を削除し、完全に自己完結。

Features:
- KILO/CHORDS事前chordmapとの重み付き融合
- Stage2 downbeats/tempoとの統合
- 信頼度ベースの競合解決
- 音響推定（Triad-onlyテンプレートマッチング内蔵）

Usage:
    # 音響推定のみ
    python ops/stem_harmony_bar_level_fusion.py \\
        --stems-dir suno_themesong/song_001/stemswav_001 \\
        --downbeats-sec-json work/tempo_downbeats.json \\
        --out-chordmap analysis/chordmap.json
    
    # 事前融合モード
    python ops/stem_harmony_bar_level_fusion.py \\
        --stems-dir suno_themesong/song_001/stemswav_001 \\
        --downbeats-sec-json work/tempo_downbeats.json \\
        --prior-chordmap analysis/kilo_chordmaps/song.chordmap.json \\
        --out-chordmap analysis/chordmap_fused.json \\
        --prior-weight 0.6
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

import numpy as np


# =====================================
# 型定義（独立化）
# =====================================


@dataclass
class ChordEvent:
    """1小節のコード情報"""

    root: str
    quality: str
    confidence: float = 0.5


# =====================================
# コード認識テンプレート（stem_harmony_bar_levelから複製）
# =====================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def maj_template() -> np.ndarray:
    """Major triad template [1,0,0,0,1,0,0,1,0,0,0,0]"""
    t = np.zeros(12, dtype=float)
    t[[0, 4, 7]] = 1.0
    return t


def min_template() -> np.ndarray:
    """Minor triad template [1,0,0,1,0,0,0,1,0,0,0,0]"""
    t = np.zeros(12, dtype=float)
    t[[0, 3, 7]] = 1.0
    return t


def rotate12(v: np.ndarray, k: int) -> np.ndarray:
    """12要素ベクトルをk個シフト"""
    return np.roll(v, k)


def cos_sim_columns(A: np.ndarray, B: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    列ごとのコサイン類似度

    Parameters
    ----------
    A : np.ndarray
        [12, T]
    B : np.ndarray
        [12, S]

    Returns
    -------
    np.ndarray
        [T, S] コサイン類似度行列
    """
    norm_A = np.linalg.norm(A, axis=0, keepdims=True) + eps
    norm_B = np.linalg.norm(B, axis=0, keepdims=True) + eps
    A_norm = A / norm_A
    B_norm = B / norm_B
    return (A_norm.T @ B_norm).clip(0.0, 1.0)


def build_chord_templates() -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Triad-onlyテンプレート生成（Major/Minor × 12音階 = 24コード）

    Returns
    -------
    templates : np.ndarray
        [12, 24] (12音階 × 24コード)
    labels : List[Tuple[str, str]]
        [(root, quality), ...] 24要素
    """
    types = [
        ("", maj_template()),  # major
        ("m", min_template()),  # minor
    ]

    templates = []
    labels = []

    for root_idx in range(12):
        for quality, tmpl in types:
            rotated = rotate12(tmpl, root_idx)
            templates.append(rotated)
            labels.append((NOTE_NAMES[root_idx], quality))

    return np.array(templates).T, labels  # [12, 24]


def extract_bar_level_chords(
    wav_path: str,
    downbeats_sec: List[float],
    sr: int = 22050,
    bins_per_octave: int = 36,
) -> List[ChordEvent]:
    """
    WAV → Bar-level Chord Recognition（音響推定）

    Parameters
    ----------
    wav_path : str
        Stem WAVファイルパス
    downbeats_sec : List[float]
        小節頭の時刻（秒）リスト
    sr : int
        サンプリングレート
    bins_per_octave : int
        CQT bins per octave

    Returns
    -------
    List[ChordEvent]
        各小節のコード推定結果
    """
    import librosa

    # 音声読み込み
    try:
        y, sr_loaded = librosa.load(wav_path, sr=sr, mono=True)
    except Exception as e:
        print(f"❌ Failed to load {wav_path}: {e}")
        return []

    # HPSS（Harmonic成分抽出）
    try:
        y_harmonic, _ = librosa.effects.hpss(y)
    except Exception:
        y_harmonic = y

    # CQT Chroma抽出
    try:
        C = librosa.feature.chroma_cqt(
            y=y_harmonic,
            sr=sr_loaded,
            bins_per_octave=bins_per_octave,
        )
    except Exception as e:
        print(f"❌ Chroma extraction failed: {e}")
        return []

    # Downbeats → Frame indices
    hop_length = 512
    frame_rate = sr_loaded / hop_length
    downbeat_frames = [int(t * frame_rate) for t in downbeats_sec]

    # 小節ごとにChroma集約
    n_bars = len(downbeat_frames) - 1
    C_bars = []

    for i in range(n_bars):
        start_frame = downbeat_frames[i]
        end_frame = downbeat_frames[i + 1] if i + 1 < len(downbeat_frames) else C.shape[1]

        if start_frame >= C.shape[1]:
            # フレーム範囲外
            C_bars.append(np.zeros(12))
        else:
            bar_chroma = np.mean(C[:, start_frame:end_frame], axis=1)
            C_bars.append(bar_chroma)

    C_bars = np.array(C_bars).T  # [12, n_bars]

    # テンプレートマッチング
    templates, labels = build_chord_templates()
    sim = cos_sim_columns(C_bars, templates)  # [n_bars, 24]

    # 各小節で最尤コード選択
    events = []
    for bar_idx in range(n_bars):
        chord_idx = int(np.argmax(sim[bar_idx]))
        confidence = float(sim[bar_idx, chord_idx])

        root, quality = labels[chord_idx]
        events.append(
            ChordEvent(
                root=root,
                quality=quality,
                confidence=confidence,
            )
        )

    return events


def load_chord_prior(prior_json: Path) -> Dict[int, Dict[str, Any]]:
    """
    KILO/CHORDS由来のchordmap.jsonを読み込み

    Parameters
    ----------
    prior_json : Path
        chordmap.jsonのパス

    Returns
    -------
    Dict[int, Dict[str, Any]]
        バーインデックス → {root, quality, confidence}
    """
    if not prior_json.exists():
        return {}

    j = json.loads(prior_json.read_text(encoding="utf-8"))
    events = j.get("events", [])

    bar_map = {}
    for e in events:
        # timeはQL想定（1bar=4QL）
        bar = int(round(float(e.get("time", 0)) / 4.0))
        bar_map[bar] = {
            "root": e.get("root", "C"),
            "quality": e.get("quality", ""),
            "confidence": e.get("confidence", 0.5),
        }

    return bar_map


def fuse_chord_maps(
    prior: Dict[int, Dict[str, Any]],
    audio: List[ChordEvent],
    w_prior: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    事前chordmapと音響推定を融合

    Parameters
    ----------
    prior : Dict[int, Dict[str, Any]]
        事前chordmap（バーインデックス→コード）
    audio : List[ChordEvent]
        音響推定結果
    w_prior : float
        事前の重み（0.0-1.0）

    Returns
    -------
    List[Dict[str, Any]]
        融合後のイベントリスト
    """
    # 音響推定をバーマップに変換
    audio_map = {}
    for i, evt in enumerate(audio):
        audio_map[i] = {
            "root": evt.root,
            "quality": evt.quality,
            "confidence": getattr(evt, "confidence", 0.5),
        }

    # 全バー集合
    all_bars = sorted(set(list(prior.keys()) + list(audio_map.keys())))

    fused = []
    for bar in all_bars:
        p = prior.get(bar)
        a = audio_map.get(bar)

        if p and a:
            # 一致判定
            same = p["root"] == a["root"] and p["quality"] == a["quality"]
            if same:
                # 信頼度統合
                conf = min(1.0, p["confidence"] * w_prior + a["confidence"] * (1 - w_prior))
                result = {
                    "time": float(bar * 4.0),
                    "root": p["root"],
                    "quality": p["quality"],
                    "confidence": conf,
                    "source": "both",
                }
            else:
                # 競合: 重み付き信頼度で勝者決定
                cp = p["confidence"] * w_prior
                ca = a["confidence"] * (1 - w_prior)
                winner = p if cp >= ca else a
                result = {
                    "time": float(bar * 4.0),
                    "root": winner["root"],
                    "quality": winner["quality"],
                    "confidence": winner["confidence"],
                    "source": "prior" if winner == p else "audio",
                }
            fused.append(result)

        elif p:
            fused.append(
                {
                    "time": float(bar * 4.0),
                    "root": p["root"],
                    "quality": p["quality"],
                    "confidence": p["confidence"],
                    "source": "prior",
                }
            )

        elif a:
            fused.append(
                {
                    "time": float(bar * 4.0),
                    "root": a["root"],
                    "quality": a["quality"],
                    "confidence": a["confidence"],
                    "source": "audio",
                }
            )

    return fused


def main():
    ap = argparse.ArgumentParser(
        description="Enhanced bar-level chord extraction with prior fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--stems-dir",
        type=Path,
        required=True,
        help="Directory containing stem WAV files",
    )
    ap.add_argument(
        "--downbeats-sec-json",
        type=Path,
        required=True,
        help="JSON with {downbeats_sec:[...], tempo_map:[[t,bpm],...]}",
    )
    ap.add_argument(
        "--prior-chordmap",
        type=Path,
        default=None,
        help="KILO/CHORDS chordmap.json (optional)",
    )
    ap.add_argument(
        "--out-chordmap",
        type=Path,
        required=True,
        help="Output chordmap.json path",
    )
    ap.add_argument(
        "--prior-weight",
        type=float,
        default=0.6,
        help="Prior weight (0.0-1.0)",
    )

    args = ap.parse_args()

    # Downbeats読み込み
    print(f"📂 Loading downbeats from {args.downbeats_sec_json}")
    meta = json.loads(args.downbeats_sec_json.read_text(encoding="utf-8"))
    downbeats_sec = meta.get("downbeats_sec", [])

    # 音響推定
    print("🎵 Extracting chords from stems...")
    stems = sorted(args.stems_dir.glob("*.wav"))
    if not stems:
        print(f"❌ No WAV files found in {args.stems_dir}")
        return 1

    print(f"   Found {len(stems)} stems")

    # 音響推定実行
    audio_events = extract_bar_level_chords(
        str(stems[0]),  # TODO: 複数ステム対応
        downbeats_sec=downbeats_sec,
    )

    print(f"   Audio: {len(audio_events)} bars")

    # 事前chordmap読み込み（任意）
    prior = {}
    if args.prior_chordmap:
        print(f"📚 Loading prior chordmap from {args.prior_chordmap}")
        prior = load_chord_prior(args.prior_chordmap)
        print(f"   Prior: {len(prior)} bars")

    # 融合
    if prior:
        print(f"🔀 Fusing prior and audio (w_prior={args.prior_weight})...")
        fused = fuse_chord_maps(prior, audio_events, args.prior_weight)
        print(f"   Fused: {len(fused)} bars")
    else:
        # 事前なし: 音響のみ
        fused = [
            {
                "time": float(i * 4.0),
                "root": evt.root,
                "quality": evt.quality,
                "confidence": getattr(evt, "confidence", 0.5),
                "source": "audio",
            }
            for i, evt in enumerate(audio_events)
        ]

    # 出力
    out = {
        "unit": "ql",
        "events": fused,
        "meta": {
            "prior_weight": args.prior_weight,
            "n_bars": len(fused),
            "sources": {
                "prior": sum(1 for e in fused if e["source"] in ["prior", "both"]),
                "audio": sum(1 for e in fused if e["source"] in ["audio", "both"]),
                "both": sum(1 for e in fused if e["source"] == "both"),
            },
        },
    }

    args.out_chordmap.parent.mkdir(parents=True, exist_ok=True)
    args.out_chordmap.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Wrote: {args.out_chordmap}")
    print(f"\n📊 Source distribution:")
    print(f"   Prior:  {out['meta']['sources']['prior']}")
    print(f"   Audio:  {out['meta']['sources']['audio']}")
    print(f"   Both:   {out['meta']['sources']['both']}")

    # サンプル表示
    if fused:
        print(f"\n📊 Sample chords (first 8 bars):")
        for e in fused[:8]:
            bar = int(e["time"] / 4.0)
            print(
                f"   Bar {bar:3d}: {e['root']:3s} {e['quality']:4s} "
                f"(conf={e['confidence']:.2f}, src={e['source']})"
            )

    return 0


if __name__ == "__main__":
    exit(main())
