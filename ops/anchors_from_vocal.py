#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/anchors_from_vocal.py — Vocal STEMから lyric_anchors.json を作る最小実用版

目的:
- ボーカルWAVから発話アンカー（音節/子音起点）を抽出
- （任意）歌詞テキストを簡易アライン → 各アンカーに token を割当
- 子音クラス推定（stress / sibilant / plosive を中心）とクラス別ウィンドウ付与
- （任意）sections.json を読んで section / time_ql を付加

窓方式（window-mode）:
- class: クラス別ウィンドウ（stress/sibilant/plosive）
- fixed: 全アンカー一律の固定ウィンドウ
- beat: 拍長に対する比率
- proportional: 前後ギャップに比例
- energy: 局所RMS強度連動

依存: numpy, librosa（PyYAMLは任意）
注意: 追加の学習器は使わず、DSP+ヒューリスティクスで軽量実装
"""
from __future__ import annotations
import argparse, json, sys, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import librosa

try:
    import yaml  # 任意: クラス別ウィンドウ設定のYAML読込
except Exception:
    yaml = None


# ---------------------------- utils: sections ----------------------------
def _safe_load_json(p: Optional[Path]) -> dict:
    if not p or not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sections_labeler_and_ql(
    sections_path: Optional[Path],
    beat_times: Optional[np.ndarray],
    default_ql_per_beat: float = 1.0,
):
    data = _safe_load_json(sections_path)
    secs = sorted(
        data.get("sections") or [], key=lambda s: float(s.get("time_sec", s.get("time", 0)))
    )
    ts = sorted(data.get("time_sigs") or [], key=lambda s: int(s.get("bar", 0)))
    if ts:
        den = int(ts[-1].get("den", 4))
        ql_per_beat = 4.0 / den if den > 0 else default_ql_per_beat
    else:
        ql_per_beat = default_ql_per_beat
    markers = []
    for s in secs:
        lab = str(s.get("label", "")).strip().lower() or None
        if lab is None:
            continue
        if "time_sec" in s:
            t = float(s["time_sec"])
        elif "time" in s:
            t = float(s["time"])
        else:
            t = None
        if t is None:
            continue
        markers.append((t, lab))
    markers.sort(key=lambda x: x[0])

    def label_at_sec(t: float) -> Optional[str]:
        lab = None
        for ts, L in markers:
            if ts <= t:
                lab = L
            else:
                break
        return lab

    def sec_to_ql(t: float) -> float:
        if beat_times is None or len(beat_times) == 0:
            # ざっくり: 秒→拍→QL
            # 楽曲内で一定テンポ前提の簡易換算（厳密でなくてOK）
            bpm = float(data.get("tempo", 120.0))
            beats = t * (bpm / 60.0)
            return beats * ql_per_beat
        # 最も近い拍インデックス
        idx = int(np.argmin(np.abs(beat_times - t)))
        return idx * ql_per_beat

    return label_at_sec, sec_to_ql


# ---------------------------- DSP: anchors ----------------------------
@dataclass
class Anchor:
    time: float
    token: Optional[str]
    classes: List[str]
    section: Optional[str]
    time_ql: Optional[float]
    windows_ms: Dict[str, float]


# sibilant/plosive推定の閾値（経験則、必要に応じて変更）
_DEFAULT_THRESH = {
    "zcr_sibilant": 0.12,  # 高ゼロ交差率
    "centroid_sibilant": 3000.0,  # Hz
    "onset_plosive": 0.8,  # 正規化オンセット強度
}

_DEF_WINDOWS = {  # ms
    "stress": {"pre": 0.0, "post": 80.0},
    "sibilant": {"pre": 30.0, "post": 20.0},
    "plosive": {"pre": 10.0, "post": 60.0},
}

# 簡易ローマ字/英語トークンの頭子音→クラス
_SIBILANT_HEADS = ("s", "sh", "sj", "z", "j", "ch", "ts")
_PLOSIVE_HEADS = ("p", "t", "k", "b", "d", "g")


def _normalize_audio(y: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(y))
    if m > 0:
        y = y / m
    return y.astype(np.float32)


def _extract_candidates(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """return times_sec, onset_strength (0..1), zcr_win (0..1)
    - onset: librosa.onset.onset_strength + peak-pick
    - zcr/centroid: 25 ms 窓、10 ms hop
    """
    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    # adaptive thresholding
    onset_frames = librosa.util.peak_pick(
        oenv, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=np.median(oenv) * 0.5, wait=3
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    # normalize onset strength 0..1
    if len(oenv):
        oenv_n = (oenv - oenv.min()) / max(1e-9, (oenv.max() - oenv.min()))
        ost_n = oenv_n[onset_frames] if len(onset_frames) else np.array([])
    else:
        ost_n = np.array([])
    # zcr
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=int(0.025 * sr), hop_length=int(0.010 * sr)
    )[0]
    zcr_t = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=int(0.010 * sr))
    # resample zcr to onset times (nearest)
    z_idx = np.clip(np.searchsorted(zcr_t, onset_times), 0, len(zcr) - 1)
    z_at = zcr[z_idx]
    return onset_times, ost_n, z_at


def _spectral_centroid_at(y: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    hop = int(0.010 * sr)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    ct = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop)
    idx = np.clip(np.searchsorted(ct, times), 0, len(cent) - 1)
    return cent[idx]


# ---------------------------- token alignment ----------------------------
def _simple_tokenize_ja_en(text: str) -> List[str]:
    # 句読点/記号を空白扱い → 分割
    t = re.sub(r"[\,\.\!\?\(\)\[\]\{\}、。！？・：:；;\-＿—〜~]", " ", text)
    toks = [w for w in t.strip().split() if w.strip()]
    return toks


def _class_from_token_head(token: str) -> Optional[str]:
    s = token.strip().lower()
    if not s:
        return None
    # 2文字頭を優先
    for h in _SIBILANT_HEADS:
        if s.startswith(h):
            return "sibilant"
    for h in _PLOSIVE_HEADS:
        if s.startswith(h):
            return "plosive"
    # デフォルトは stress（伸ばし/母音多め想定）
    return "stress"


def _align_tokens_to_onsets(
    tokens: List[str], onset_times: np.ndarray
) -> List[Tuple[float, Optional[str]]]:
    if not tokens:
        return [(float(t), None) for t in onset_times]
    T, N = len(tokens), len(onset_times)
    if N == 0:
        return []
    # 長さ差は素直に間引き/重複（均等）
    if T == N:
        return list(zip(onset_times.tolist(), tokens))
    if T > N:
        # トークンが多い → 均等スキップ
        keep_idx = np.round(np.linspace(0, T - 1, N)).astype(int)
        toks2 = [tokens[i] for i in keep_idx]
        return list(zip(onset_times.tolist(), toks2))
    else:
        # トークンが少ない → 均等に複製
        rep_idx = np.floor(np.linspace(0, T - 1, N)).astype(int)
        toks2 = [tokens[i] for i in rep_idx]
        return list(zip(onset_times.tolist(), toks2))


# ---------------------------- class windows ----------------------------
def _load_windows(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if not path or not path.exists():
        return _DEF_WINDOWS
    if path.suffix.lower() in (".yaml", ".yml") and yaml:
        try:
            d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return {
                k: {"pre": float(v.get("pre", 0)), "post": float(v.get("post", 0))}
                for k, v in (d.get("windows", {}) or {}).items()
            } or _DEF_WINDOWS
        except Exception:
            return _DEF_WINDOWS
    else:
        try:
            d = json.loads(path.read_text(encoding="utf-8")) or {}
            return {
                k: {"pre": float(v.get("pre", 0)), "post": float(v.get("post", 0))}
                for k, v in (d.get("windows", {}) or {}).items()
            } or _DEF_WINDOWS
        except Exception:
            return _DEF_WINDOWS


# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Extract lyric anchors from vocal stem (onset-based, optional lyric alignment)"
    )
    ap.add_argument("--vocal", required=True, help="Vocal WAV path")
    ap.add_argument("--lyrics", help="Lyrics text file (optional)")
    ap.add_argument("--sections", help="sections.json path (optional)")
    ap.add_argument("--bars", help="bars.parquet path (for tempo/beat alignment)")
    ap.add_argument("--tempo-map", help="tempo_map.json path (optional, for BPM validation)")
    ap.add_argument("--force-bpm", type=float, help="Force BPM (overrides bars/tempo-map)")
    ap.add_argument("--windows", help="YAML/JSON for class windows (stress/sibilant/plosive)")

    # 窓方式の拡張: class(既定)/fixed/beat/proportional/energy
    ap.add_argument(
        "--window-mode",
        choices=["class", "fixed", "beat", "proportional", "energy"],
        default="class",
        help="Window calculation mode (default: class)",
    )

    # fixed
    ap.add_argument("--fixed-pre", type=float, default=None, help="fixed mode: pre window ms")
    ap.add_argument("--fixed-post", type=float, default=None, help="fixed mode: post window ms")

    # beat基準（拍長に対する比率）
    ap.add_argument(
        "--beat-pre-frac", type=float, default=0.25, help="beat mode: pre as fraction of beat dur"
    )
    ap.add_argument(
        "--beat-post-frac", type=float, default=0.35, help="beat mode: post as fraction of beat dur"
    )

    # proportional（前後ギャップに比例）
    ap.add_argument("--prop-k-pre", type=float, default=0.5)
    ap.add_argument("--prop-k-post", type=float, default=0.7)
    ap.add_argument("--prop-min-ms", type=float, default=20.0)
    ap.add_argument("--prop-max-ms", type=float, default=140.0)

    # energy（局所RMSに応じて拡縮）
    ap.add_argument("--energy-base-pre", type=float, default=40.0)
    ap.add_argument("--energy-base-post", type=float, default=60.0)
    ap.add_argument(
        "--energy-alpha", type=float, default=0.6, help="scale = 1 + alpha*(E-baseline)"
    )
    ap.add_argument("--energy-baseline", type=float, default=0.5)
    ap.add_argument("--energy-win-ms", type=float, default=50.0)

    # sibilant強調/限定
    ap.add_argument(
        "--sibilant-scale", type=float, default=1.0, help=">1.0 to enlarge sibilant windows"
    )
    ap.add_argument(
        "--sibilant-only", action="store_true", help="emit only anchors that include sibilant class"
    )

    ap.add_argument("--out", required=True, help="Output lyric_anchors.json path")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--zcr-sibilant", type=float, default=_DEFAULT_THRESH["zcr_sibilant"])
    ap.add_argument("--centroid-sibilant", type=float, default=_DEFAULT_THRESH["centroid_sibilant"])
    ap.add_argument("--onset-plosive", type=float, default=_DEFAULT_THRESH["onset_plosive"])
    args = ap.parse_args()

    print(f"[INFO] Loading vocal: {args.vocal}")
    y, sr = librosa.load(args.vocal, sr=args.sr, mono=True)
    y = _normalize_audio(y)

    # Onset/特徴量
    print(f"[INFO] Extracting onset candidates...")
    onset_times, onset_str_norm, z_at = _extract_candidates(y, sr)
    if len(onset_times) == 0:
        print("[WARN] no onsets detected; anchors will be empty", file=sys.stderr)

    cent_at = _spectral_centroid_at(y, sr, onset_times)

    # 歌詞の読み込み＆簡易アライン
    tokens: List[str] = []
    if args.lyrics and Path(args.lyrics).exists():
        print(f"[INFO] Loading lyrics: {args.lyrics}")
        text = Path(args.lyrics).read_text(encoding="utf-8")
        tokens = _simple_tokenize_ja_en(text)
        print(f"[INFO] Tokenized {len(tokens)} words from lyrics")
    else:
        print(f"[INFO] No lyrics provided, anchors will have no tokens")

    onset_pairs = _align_tokens_to_onsets(tokens, onset_times)

    # class windows
    win_map = _load_windows(Path(args.windows) if args.windows else None)

    # sections + QL
    # bars.parquetがあれば、そこから拍時刻配列を生成（ビート検出は完全スキップ）
    if args.bars and Path(args.bars).exists():
        print(f"[INFO] Loading bars from: {args.bars}")
        try:
            import pandas as pd

            bars_df = pd.read_parquet(args.bars)

            # time_signatureから拍数を取得
            def beats_in_bar(ts):
                try:
                    # "4/4" → 4, "3/4" → 3
                    num = int(str(ts).split("/")[0])
                except Exception:
                    num = 4
                return max(num, 1)

            # 各小節から拍グリッドを生成（start_sec, end_secの線形分割）
            if "start_sec" in bars_df.columns and "end_sec" in bars_df.columns:
                beat_times_list = []
                total_bars = len(bars_df)

                # 平均拍数を事前算出（BPM妥当性チェックで使用）
                avg_bpb = (
                    (bars_df["end_beat"] - bars_df["start_beat"]).mean()
                    if "start_beat" in bars_df.columns
                    else 4.0
                )

                for _, row in bars_df.sort_values("bar_index").iterrows():
                    ts = row.get("time_signature", "4/4")
                    bpb = beats_in_bar(ts)
                    t_start = float(row["start_sec"])
                    t_end = float(row["end_sec"])
                    dur = t_end - t_start

                    # 小節内を均等分割（endpoint=False: 次小節の頭拍は次の小節で生成）
                    for k in range(bpb):
                        t = t_start + dur * (k / bpb)
                        beat_times_list.append(t)

                beats = np.array(beat_times_list)

                # BPM決定ロジック（優先順位）
                # 1. --force-bpm（最優先）
                # 2. bars.parquetのtempo_bpm中央値
                # 3. tempo_map.jsonの中央値
                # 4. フォールバック（120.0、警告）

                bpm_bars = None
                if "tempo_bpm" in bars_df.columns:
                    bpm_bars = float(bars_df["tempo_bpm"].median())

                bpm_tempo_map = None
                if args.tempo_map and Path(args.tempo_map).exists():
                    try:
                        import json

                        tempo_map_data = json.loads(Path(args.tempo_map).read_text())
                        if isinstance(tempo_map_data, dict) and "tempo_points" in tempo_map_data:
                            tempo_points = tempo_map_data["tempo_points"]
                            bpms = [
                                p[1] for p in tempo_points if isinstance(p, list) and len(p) >= 2
                            ]
                            if bpms:
                                import statistics

                                bpm_tempo_map = float(statistics.median(bpms))
                    except Exception as e:
                        print(f"[WARN] Failed to load tempo_map.json: {e}")

                # BPM決定
                if args.force_bpm is not None:
                    tempo_val = float(args.force_bpm)
                    print(f"[INFO] BPM source: --force-bpm (forced={tempo_val:.2f})")
                elif bpm_bars is not None:
                    tempo_val = bpm_bars
                    print(f"[INFO] BPM source: bars.parquet (median={tempo_val:.2f})")
                elif bpm_tempo_map is not None:
                    tempo_val = bpm_tempo_map
                    print(f"[INFO] BPM source: tempo_map.json (median={tempo_val:.2f})")
                else:
                    tempo_val = 120.0
                    print(f"[WARN] BPM source: fallback (DEFAULT=120.0, no tempo_bpm/tempo_map)")

                # BPM妥当性チェック（bars.parquetとの差が1.5以上なら警告）
                if bpm_bars is not None and abs(tempo_val - bpm_bars) > 1.5:
                    print(
                        f"[WARN] BPM mismatch: bars={bpm_bars:.2f}, used={tempo_val:.2f} (diff={abs(tempo_val - bpm_bars):.2f})"
                    )

                # 拍数妥当性チェック（68小節×4拍=272拍を期待）
                expected_beats = total_bars * avg_bpb
                if abs(len(beats) - expected_beats) > 1:
                    print(
                        f"[WARN] Beat count mismatch: expected={expected_beats:.0f}, got={len(beats)} (diff={abs(len(beats) - expected_beats):.0f})"
                    )

                # ログ表示: bars由来であることを明示
                print(
                    f"[INFO] Total bars: {total_bars}, avg beats/bar: {avg_bpb:.1f}, total beats: {len(beats)}"
                )
            else:
                # fallback: librosa.beat.beat_track
                print(
                    f"[WARN] bars.parquet missing start_sec/end_sec columns, falling back to librosa"
                )
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", tightness=100)
                tempo_val = (
                    float(tempo[0])
                    if isinstance(tempo, np.ndarray) and len(tempo) > 0
                    else float(tempo)
                )
                print(f"[INFO] Detected tempo: {tempo_val:.1f} BPM, {len(beats)} beats")
        except Exception as e:
            print(f"[WARN] Failed to load bars.parquet: {e}, falling back to librosa")
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", tightness=100)
            tempo_val = (
                float(tempo[0])
                if isinstance(tempo, np.ndarray) and len(tempo) > 0
                else float(tempo)
            )
            print(f"[INFO] Detected tempo: {tempo_val:.1f} BPM, {len(beats)} beats")
    else:
        # bars.parquet未指定 → librosa.beat.beat_track
        print(f"[INFO] Detecting beats (no bars.parquet provided)...")
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", tightness=100)
        tempo_val = (
            float(tempo[0]) if isinstance(tempo, np.ndarray) and len(tempo) > 0 else float(tempo)
        )
        print(f"[INFO] Detected tempo: {tempo_val:.1f} BPM, {len(beats)} beats")

    label_at_sec, sec_to_ql = _sections_labeler_and_ql(
        Path(args.sections) if args.sections else None, np.array(beats) if len(beats) else None
    )

    # energy用のRMSトラック
    def _local_energy(t: float) -> float:
        win = max(10.0, float(args.energy_win_ms)) / 1000.0
        i0 = max(0, int((t - 0.5 * win) * sr))
        i1 = min(len(y), int((t + 0.5 * win) * sr))
        if i1 <= i0:
            return 0.0
        seg = y[i0:i1]
        e = float(np.sqrt(np.mean(seg**2)) if len(seg) else 0.0)
        return float(min(1.0, max(0.0, e)))

    # beat長の取得（近傍拍の間隔）
    def _beat_duration_at(t: float) -> Optional[float]:
        if beats is None or len(beats) < 2:
            return None
        idx = int(np.clip(np.searchsorted(beats, t), 1, len(beats) - 1))
        return float(beats[idx] - beats[idx - 1])

    print(f"[INFO] Analyzing anchors with window-mode={args.window_mode}...")
    anchors: List[Anchor] = []
    # 事前に時系列配列を用意（比例モードのため）
    times_seq = [float(t) for (t, _tok) in onset_pairs]

    for idx, ((t, tok), ost, z, c) in enumerate(zip(onset_pairs, onset_str_norm, z_at, cent_at)):
        # ヒューリスティクス分類
        klasses: List[str] = []
        # sibilant: 高ZCR or 高centroid
        if (z >= args.zcr_sibilant) or (c >= args.centroid_sibilant):
            klasses.append("sibilant")
        # plosive: 強い立ち上がり
        if ost >= args.onset_plosive:
            klasses.append("plosive")
        # token 先頭子音で上書き/補完
        if tok:
            c2 = _class_from_token_head(tok)
            if c2 and c2 not in klasses:
                klasses.append(c2)
        if not klasses:
            klasses = ["stress"]

        # windows（モード別に決定）
        if args.window_mode == "fixed":
            pre = float(
                args.fixed_pre if args.fixed_pre is not None else _DEF_WINDOWS["stress"]["pre"]
            )
            post = float(
                args.fixed_post if args.fixed_post is not None else _DEF_WINDOWS["stress"]["post"]
            )
        elif args.window_mode == "beat":
            bd = _beat_duration_at(float(t))
            if bd is None:
                # フォールバック: class
                base = win_map.get(klasses[0], win_map.get("stress", _DEF_WINDOWS))
                pre, post = float(base["pre"]), float(base["post"])
            else:
                pre = max(0.0, float(args.beat_pre_frac) * 1000.0 * bd)
                post = max(0.0, float(args.beat_post_frac) * 1000.0 * bd)
        elif args.window_mode == "proportional":
            # 前後ギャップ比例（クランプ）
            t_prev = times_seq[idx - 1] if idx > 0 else None
            t_next = times_seq[idx + 1] if idx + 1 < len(times_seq) else None
            gap_pre = (t - t_prev) if t_prev is not None else 0.08
            gap_post = (t_next - t) if t_next is not None else 0.12
            pre = float(
                np.clip(
                    1000.0 * gap_pre * float(args.prop_k_pre),
                    float(args.prop_min_ms),
                    float(args.prop_max_ms),
                )
            )
            post = float(
                np.clip(
                    1000.0 * gap_post * float(args.prop_k_post),
                    float(args.prop_min_ms),
                    float(args.prop_max_ms),
                )
            )
        elif args.window_mode == "energy":
            e = _local_energy(float(t))
            scale = 1.0 + float(args.energy_alpha) * (e - float(args.energy_baseline))
            base_pre = float(args.energy_base_pre)
            base_post = float(args.energy_base_post)
            pre = max(0.0, base_pre * scale)
            post = max(0.0, base_post * scale)
        else:
            # class（既定）
            base = win_map.get(klasses[0], win_map.get("stress", _DEF_WINDOWS))
            pre, post = float(base["pre"]), float(base["post"])

        # sibilant強調倍率
        if args.sibilant_scale and ("sibilant" in klasses):
            s = float(args.sibilant_scale)
            if s != 1.0:
                pre, post = pre * s, post * s

        # sibilant-only の場合：非sibilantはスキップ
        if args.sibilant_only and ("sibilant" not in klasses):
            continue

        w = {"pre": float(pre), "post": float(post)}

        # section/ql
        sec = label_at_sec(t) if label_at_sec else None
        ql = sec_to_ql(t) if sec_to_ql else None
        anchors.append(
            Anchor(
                time=float(t),
                token=tok,
                classes=klasses,
                section=sec,
                time_ql=float(ql) if ql is not None else None,
                windows_ms=w,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "unit": "sec",
        "anchors": [
            {
                "time": a.time,
                "token": a.token,
                "classes": a.classes,  # 先頭要素が主クラス
                "section": a.section,
                "time_ql": a.time_ql,
                "windows_ms": {
                    "pre": a.windows_ms.get("pre", 0.0),
                    "post": a.windows_ms.get("post", 0.0),
                },
            }
            for a in anchors
        ],
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] anchors={len(anchors)} -> {out_path}")

    # Summary statistics
    if anchors:
        class_counts = {}
        for a in anchors:
            for c in a.classes:
                class_counts[c] = class_counts.get(c, 0) + 1
        print(f"[INFO] Class distribution:")
        for c, cnt in sorted(class_counts.items()):
            print(f"  {c}: {cnt}")

        if args.window_mode != "class":
            avg_pre = np.mean([a.windows_ms["pre"] for a in anchors])
            avg_post = np.mean([a.windows_ms["post"] for a in anchors])
            print(f"[INFO] Average window: pre={avg_pre:.1f}ms, post={avg_post:.1f}ms")


if __name__ == "__main__":
    main()
