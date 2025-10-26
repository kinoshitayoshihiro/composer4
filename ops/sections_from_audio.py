#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/sections_from_audio.py (統合強化版)

高精度セクション自動推定（物理解析 + ML境界検出 + テンポカーブ + キー推定）

改良点：
1. テンポカーブ推定（tempo_from_mix.py統合）→ tempo_map, timesig付与
2. RMSピーク検出（peak_extractor.py統合）→ エネルギー変化点の精密検出
3. キー推定（Krumhansl-Schmuckler法）→ key_hint付与
4. セクション細分化（5-7区間目標）→ pre_chorus/bridge検出
5. バリデーション（section_validator.py統合）→ 健全性チェック

出力フォーマット:
{
  "unit": "bar",
  "sections": [{"bar": 0, "label": "intro"}, ...],
  "energy": [[bar, 0.0-1.0], ...],
  "tempo_map": [[bar, bpm], ...],  # 新規
  "timesig": {"num": 4, "denom": 4},  # 新規
  "key_hint": [[bar, "D"], [32, "G"], ...]  # 新規（転調対応）
}
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import librosa.display  # noqa: F401

# 統合ユーティリティ
try:
    from utilities.peak_extractor import extract_peaks, PeakExtractorConfig

    HAVE_PEAK_EXTRACTOR = True
except ImportError:
    HAVE_PEAK_EXTRACTOR = False
    print("[WARN] utilities.peak_extractor not available, using basic peak detection")

# 統合ユーティリティ
try:
    from utilities.peak_extractor import extract_peaks, PeakExtractorConfig

    HAVE_PEAK_EXTRACTOR = True
except ImportError:
    HAVE_PEAK_EXTRACTOR = False
    print("[WARN] utilities.peak_extractor not available, using basic peak detection")

# ----------------------------
# テンポ推定（tempo_from_mix.py簡易版）
# ----------------------------


def estimate_tempo_curve(y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
    """簡易テンポ推定（固定BPM仮定）"""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    if tempo is None or len(beats) == 0:
        tempo = 120.0
        dur = librosa.get_duration(y=y, sr=sr)
        beats = np.linspace(0, dur, max(4, int(dur * tempo / 60)))
    return float(tempo), beats


def extract_tempo_map(
    y: np.ndarray, sr: int, bars: List[Tuple[int, float]]
) -> List[Tuple[int, float]]:
    """バー毎のBPM（簡易版：固定BPM）"""
    tempo, _ = estimate_tempo_curve(y, sr)
    # 将来: tempo_from_mix.py の可変テンポ推定に置き換え可能
    return [(0, tempo)]


# ----------------------------
# キー推定（Krumhansl-Schmuckler法）
# ----------------------------

# Major/Minor profile (Krumhansl-Kessler 1982)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key_from_chroma(C: np.ndarray) -> str:
    """クロマベクトルからキー推定（Krumhansl-Schmuckler）"""
    if C.ndim > 1:
        C = C.mean(axis=1)  # 時間平均
    C = C / (np.sum(C) + 1e-9)  # 正規化

    best_corr = -1.0
    best_key = "C"

    for shift in range(12):
        C_shifted = np.roll(C, shift)
        # Major
        corr_maj = np.corrcoef(C_shifted, MAJOR_PROFILE / np.sum(MAJOR_PROFILE))[0, 1]
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = PITCH_CLASSES[shift]
        # Minor
        corr_min = np.corrcoef(C_shifted, MINOR_PROFILE / np.sum(MINOR_PROFILE))[0, 1]
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = PITCH_CLASSES[shift] + "m"

    return best_key


def extract_key_hints(C_bars: np.ndarray, section_boundaries: List[int]) -> List[Tuple[int, str]]:
    """セクション毎にキー推定"""
    key_hints = []
    for i in range(len(section_boundaries)):
        bar_start = section_boundaries[i]
        bar_end = section_boundaries[i + 1] if i + 1 < len(section_boundaries) else C_bars.shape[1]

        if bar_end > bar_start:
            C_section = C_bars[:, bar_start:bar_end]
            key = estimate_key_from_chroma(C_section)
            # 前のキーと異なる場合のみ追加（転調検出）
            if not key_hints or key_hints[-1][1] != key:
                key_hints.append((int(bar_start), key))

    return key_hints


# ----------------------------
# ユーティリティ
# ----------------------------


def _load_mix_from_stems(stems_dir: Path, exclude: List[str]) -> tuple[np.ndarray, int]:
    """stems_dir の wav をミックスダウン（除外語を名前に含むものは除く）。"""
    ys = []
    sr_ref = None
    for p in sorted(stems_dir.glob("*.wav")):
        name = p.stem.lower()
        if any(ex.lower() in name for ex in exclude):
            continue
        try:
            y, sr = sf.read(str(p), always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr_ref is None:
                sr_ref = sr
            elif sr != sr_ref:
                y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=sr_ref)
            ys.append(y.astype(float))
        except Exception:
            continue
    if not ys:
        raise RuntimeError("No stems loaded (check --stems / --exclude)")
    L = max(len(y) for y in ys)
    mix = np.zeros(L, dtype=float)
    for y in ys:
        if len(y) < L:
            y = np.pad(y, (0, L - len(y)))
        mix += y
    mix /= max(1.0, np.max(np.abs(mix)) + 1e-9)
    return mix, int(sr_ref or 44100)


def _bars_from_beats(beats_sec: np.ndarray, ts_num: int) -> List[Tuple[int, float]]:
    """拍列からバー先頭の index→time(sec) を返す。"""
    bars = []
    if beats_sec is None or len(beats_sec) == 0:
        return bars
    for i in range(0, len(beats_sec), ts_num):
        bars.append((i // ts_num, float(beats_sec[i])))
    return bars


def _per_bar_aggregate(
    times: np.ndarray, values: np.ndarray, bar_times: List[Tuple[int, float]], sr: int, hop: int
) -> List[Tuple[int, float]]:
    """フレーム系列を小節毎に集約（平均）。"""
    out: List[Tuple[int, float]] = []
    if not len(times):
        return out
    idx = 0
    for b, t0 in bar_times:
        t1 = bar_times[b + 1][1] if (b + 1) < len(bar_times) else times[-1]
        while idx < len(times) and times[idx] < t0:
            idx += 1
        j = idx
        while j < len(times) and times[j] < t1:
            j += 1
        if j > idx:
            out.append((b, float(np.mean(values[idx:j]))))
        else:
            out.append((b, 0.0))
        idx = j
    return out


def _novelty_curve(y: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """簡易 novelty: onset_strength を DoG 平滑して強調。"""
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    # Difference-of-Gaussians (近似)
    k_small = int(round(0.5 * sr / hop / 2)) + 1
    k_large = int(round(2.0 * sr / hop / 2)) + 1
    small = librosa.decompose.nn_filter(onset, aggregate=np.median, metric="cosine", width=k_small)
    large = librosa.decompose.nn_filter(onset, aggregate=np.median, metric="cosine", width=k_large)
    nov = np.clip(small - large, 0, None)
    t = librosa.frames_to_time(np.arange(len(nov)), sr=sr, hop_length=hop)
    return t, nov


def _chroma_contrast(y: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """クロマのフレーム差分ノルム（和声変化度）。"""
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=12)
    d = np.linalg.norm(np.diff(C, axis=1), axis=0, ord=1)
    d = np.concatenate([[0.0], d])
    t = librosa.frames_to_time(np.arange(len(d)), sr=sr, hop_length=hop)
    return t, d


def detect_sections(
    y: np.ndarray,
    sr: int,
    *,
    ts_num: int = 4,
    min_bars: int = 4,
    max_sections: int = 12,
    peak_prom: float = 0.15,
    use_peak_extractor: bool = True,
) -> tuple[
    List[Tuple[int, str]],
    List[Tuple[int, float]],
    List[Tuple[int, float]],
    List[Tuple[int, str]],
    np.ndarray,
]:
    """小節境界の推定と改良ラベリング（物理解析統合版）

    Returns:
        sections: [(bar_index, label)]
        energy_bar: [(bar, e in [0..1])]
        tempo_map: [(bar, bpm)]
        key_hints: [(bar, key)]
        C_bars: クロマ行列 (12, n_bars)
    """
    hop = 512

    # 1) ビート/バー推定
    tempo, beats = estimate_tempo_curve(y, sr)
    if beats is None or len(beats) == 0:
        dur = librosa.get_duration(y=y, sr=sr)
        beats = np.linspace(0, dur, max(4, int(dur * tempo / 60)))
    bars = _bars_from_beats(beats, ts_num=ts_num)
    if len(bars) < 2:
        # フォールバック
        tempo_map = [(0, tempo)]
        key_hints = [(0, "C")]
        C_bars = np.zeros((12, 2))
        return [(0, "intro"), (1, "outro")], [(0, 0.3), (1, 0.3)], tempo_map, key_hints, C_bars

    # 2) RMSピーク検出（peak_extractor統合）
    if use_peak_extractor and HAVE_PEAK_EXTRACTOR:
        try:
            # WAVを一時ファイルに保存（peak_extractorがファイルパスを要求）
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, y, sr)
                cfg = PeakExtractorConfig(
                    sr=sr, threshold_db=-20.0, min_distance_ms=30.0, rms_smooth_ms=20.0
                )
                peaks_sec = extract_peaks(tmp.name, cfg)
                os.unlink(tmp.name)
            print(f"[INFO] peak_extractor: {len(peaks_sec)} energy peaks detected")
        except Exception as e:
            print(f"[WARN] peak_extractor failed: {e}, using fallback")
            peaks_sec = []
    else:
        peaks_sec = []

    # 3) 新規性カーブ（onset + chroma差分）
    t1, nov = _novelty_curve(y, sr, hop)
    t2, har = _chroma_contrast(y, sr, hop)
    t = t1
    nov = nov / (np.max(nov) + 1e-6)
    har = har / (np.max(har) + 1e-6)
    combo = 0.6 * nov + 0.4 * har
    bar_times = bars
    combo_bar = _per_bar_aggregate(t, combo, bar_times, sr, hop)

    # 4) エネルギー（RMS）をバー集約
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    t_r = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    ebar = _per_bar_aggregate(t_r, rms, bar_times, sr, hop)
    if ebar:
        mx = max(e for _, e in ebar) + 1e-6
        ebar = [(b, float(e / mx)) for b, e in ebar]

    # 5) クロマ計算（キー推定用）
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=12)
    t_c = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop)
    C_bar_list = []
    for b, t0 in bar_times:
        t1 = bar_times[b + 1][1] if (b + 1) < len(bar_times) else t_c[-1]
        mask = (t_c >= t0) & (t_c < t1)
        if np.any(mask):
            C_bar_list.append(C[:, mask].mean(axis=1))
        else:
            C_bar_list.append(np.zeros(12))
    C_bars = np.column_stack(C_bar_list) if C_bar_list else np.zeros((12, len(bars)))

    # 6) ピーク抽出（境界候補）- ChatGPT診断: energy曲線を直接使用
    import scipy.signal as sig

    # エネルギー曲線から直接ピーク検出（novelty併用を廃止）
    e_vals = np.array([e for _, e in ebar])

    # ChatGPT診断: スムージングを軽減（32→16で適度な平滑化）
    k = max(1, len(e_vals) // 16)
    if k > 1:
        e_vals_smooth = sig.convolve(e_vals, np.ones(k) / k, mode="same")
    else:
        e_vals_smooth = e_vals

    # peaks_secをバーインデックスに変換
    peaks_from_rms = []
    for p_sec in peaks_sec:
        # 最も近いバーを探す
        dists = [abs(p_sec - t) for _, t in bar_times]
        if dists:
            bar_idx = int(np.argmin(dists))
            peaks_from_rms.append(bar_idx)

    # energyベースのピーク検出（ChatGPT推奨: prominence=0.06, distance=min_bars-1）
    distance = max(3, min_bars - 1)
    prom = 0.06  # ChatGPT推奨値（固定）
    peaks_energy, props = sig.find_peaks(e_vals_smooth, distance=distance, prominence=prom)

    print(f"[DEBUG] energy curve: min={np.min(e_vals):.3f}, max={np.max(e_vals):.3f}")
    print(f"[DEBUG] prominence threshold: {prom:.3f}, distance: {distance}")
    print(f"[DEBUG] peaks_energy indices: {peaks_energy.tolist()}")
    print(
        f"[DEBUG] prominence values: {props['prominences'][:10] if len(props['prominences']) > 0 else []}"
    )

    # 統合（エネルギーピークを優先）
    peaks = sorted(set(peaks_from_rms + peaks_energy.tolist()))

    print(
        f"[INFO] Combined peaks: {len(peaks)} (rms={len(peaks_from_rms)}, energy={len(peaks_energy)})"
    )

    # 7) バー境界列を構築（ChatGPT診断: min_barsを4→3に緩和、max_sections上限チェックを後段へ）
    cuts = sorted(set([0] + peaks + [len(bars) - 1]))
    pruned = [cuts[0]]
    min_gap = max(3, min_bars - 1)  # 4→3に緩和
    for i in range(1, len(cuts)):
        if (cuts[i] - pruned[-1]) < min_gap:
            continue
        pruned.append(cuts[i])
    if pruned[-1] != cuts[-1]:
        pruned.append(cuts[-1])

    # max_sections制限（ChatGPT診断: 重要度ベースで削減、均等削減を廃止）
    if len(pruned) > max_sections:
        # エネルギー変化量でソート（急変点を優先保持）
        delta_scores = []
        for i in range(1, len(pruned) - 1):  # intro/outroは固定
            b = pruned[i]
            e_before = np.mean([e for bb, e in ebar if pruned[i - 1] <= bb < b])
            e_after = np.mean([e for bb, e in ebar if b <= bb < pruned[i + 1]])
            delta_scores.append((abs(e_after - e_before), i))
        delta_scores.sort(reverse=True)
        keep_indices = {0, len(pruned) - 1}  # intro/outro固定
        keep_indices.update([idx for _, idx in delta_scores[: max_sections - 2]])
        pruned = [pruned[i] for i in sorted(keep_indices)]

    # 8) 改良ラベリング（ChatGPT診断: chorus検出の prominence を 0.15→0.10 に緩和）
    sections: List[Tuple[int, str]] = []
    n = len(pruned)

    # エネルギープロファイル分析
    energy_profile = []
    for i in range(n - 1):
        b_start, b_end = pruned[i], pruned[i + 1]
        e_local = np.mean([e for bb, e in ebar if b_start <= bb < b_end])
        energy_profile.append(e_local)
    if n > 0:
        energy_profile.append(energy_profile[-1] if energy_profile else 0.5)

    # エネルギーピーク検出（chorus候補）
    if len(energy_profile) >= 3:
        e_arr = np.array(energy_profile)
        e_peaks, _ = sig.find_peaks(e_arr, prominence=0.10)  # 0.15→0.10に緩和
        chorus_indices = set(e_peaks.tolist())
    else:
        chorus_indices = set()

    for i, b in enumerate(pruned):
        if i == 0:
            label = "intro"
        elif i == n - 1:
            label = "outro"
        elif i in chorus_indices:
            label = "chorus"
        elif i > 0 and (i - 1) in chorus_indices:
            # chorusの直前 → pre_chorus
            label = "pre_chorus"
        elif energy_profile[i] < 0.4:
            # 低エネルギー → bridge
            label = "bridge"
        else:
            # デフォルト → verse
            label = "verse"

        sections.append((int(b), label))

    # 9) テンポマップ生成
    tempo_map = extract_tempo_map(y, sr, bars)

    # 10) キーヒント生成
    section_bars = [s[0] for s in sections]
    key_hints = extract_key_hints(C_bars, section_bars)

    print(f"[INFO] Sections: {len(sections)}, Energy peaks (chorus): {len(chorus_indices)}")
    print(f"[INFO] Key hints: {key_hints}")

    return sections, ebar, tempo_map, key_hints, C_bars


# ----------------------------
# バリデーション
# ----------------------------


def validate_sections_data(
    sections: List[Tuple[int, str]], energy: List[Tuple[int, float]]
) -> bool:
    """sections.json の健全性チェック"""
    # 1) bar は単調増加
    bars = [s[0] for s in sections]
    if bars != sorted(bars):
        print("[ERROR] Validation failed: bars not monotonically increasing")
        return False

    # 2) 最小長チェック（8小節推奨）
    for i in range(len(bars) - 1):
        length = bars[i + 1] - bars[i]
        if length < 4:  # 4小節未満は警告
            print(f"[WARN] Section at bar {bars[i]} is only {length} bars (recommended >= 8)")

    # 3) energy ∈ [0, 1]
    for b, e in energy:
        if not (0.0 <= e <= 1.0):
            print(f"[ERROR] Validation failed: energy at bar {b} = {e} (must be in [0, 1])")
            return False

    print("[OK] Validation passed")
    return True


# ----------------------------
# CLI
# ----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Infer sections.json from stems (enhanced with tempo/key/peak detection)"
    )
    ap.add_argument("--stems", required=True, help="Path to stems directory or single WAV")
    ap.add_argument("--out", required=True, help="Output sections.json path")
    ap.add_argument(
        "--exclude",
        action="append",
        default=["vocals", "backing"],
        help="Exclude stems with these keywords",
    )
    ap.add_argument("--ts-num", type=int, default=4, help="Time signature numerator (default 4/4)")
    ap.add_argument(
        "--min-bars", type=int, default=4, help="Minimum section length in bars (ChatGPT推奨: 4)"
    )
    ap.add_argument(
        "--max-sections", type=int, default=12, help="Maximum number of sections (ChatGPT推奨: 12)"
    )
    ap.add_argument(
        "--peak-prom",
        type=float,
        default=0.15,
        help="Peak prominence for novelty detection (ChatGPT推奨: 0.15)",
    )
    ap.add_argument(
        "--no-peak-extractor", action="store_true", help="Disable peak_extractor (use novelty only)"
    )
    args = ap.parse_args()

    # ステムロードまたは単一WAV
    stems_path = Path(args.stems)
    if stems_path.is_file():
        # 単一WAV
        y, sr = sf.read(str(stems_path), always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(float)
        y /= max(1.0, np.max(np.abs(y)) + 1e-9)
    else:
        # ステムディレクトリ
        y, sr = _load_mix_from_stems(stems_path, exclude=args.exclude)

    # セクション検出（統合版）
    sections, energy, tempo_map, key_hints, C_bars = detect_sections(
        y,
        sr,
        ts_num=args.ts_num,
        min_bars=args.min_bars,
        max_sections=args.max_sections,
        peak_prom=args.peak_prom,
        use_peak_extractor=(not args.no_peak_extractor),
    )

    # バリデーション
    validate_sections_data(sections, energy)

    # JSON出力（拡張フォーマット）
    obj: Dict[str, object] = {
        "unit": "bar",
        "sections": [{"bar": b, "label": lab} for b, lab in sections],
        "energy": [[b, float(e)] for b, e in energy],
        "tempo_map": [[b, float(bpm)] for b, bpm in tempo_map],
        "timesig": {"num": args.ts_num, "denom": 4},
        "key_hint": [[b, key] for b, key in key_hints],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[SUCCESS] sections.json generated: {args.out}")
    print(f"{'='*60}")
    print(f"Sections: {len(sections)}")
    for b, lab in sections:
        print(f"  Bar {b:3d}: {lab}")
    print(f"\nTempo: {tempo_map[0][1]:.1f} BPM (bar 0)")
    print(f"Keys: {', '.join([f'Bar {b}: {k}' for b, k in key_hints])}")
    print(f"Energy range: [{min(e for _, e in energy):.2f}, {max(e for _, e in energy):.2f}]")


if __name__ == "__main__":
    main()
