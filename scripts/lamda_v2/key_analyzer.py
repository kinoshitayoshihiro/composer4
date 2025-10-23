#!/usr/bin/env python3
from __future__ import annotations
"""
Lamda v2 — Phase2: Local key hints and modulation detection (production-safe minimal).

入力: chordmap dict (QL基準) = {"unit":"ql", "events":[{"time": float, "root": str, "quality": str, ...}, ...]}
出力: {"keys": ["C", "C", ...], "modulations": [{"time": ql, "from": "C", "to": "G"}, ...]}

設計方針:
- まずは "窓内の root 最多数決" + スムージング(min_hold) の素朴法で安定化。
- 将来は K-S プロファイルや n-gram 事前分布に差し替え可能(APIは維持)。
- enharmonic は # 優先（C, C#, D, ...）。
"""
from typing import Dict, Any, List, Tuple
import numpy as np

ROOTS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
NAME2PC = {n: i for i, n in enumerate(ROOTS)}

# Krumhansl-Schmuckler key profiles (normalized weights)
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _events_to_bar_roots(chordmap: Dict[str, Any]) -> List[str]:
    ev = (chordmap or {}).get("events") or []
    if not ev:
        return []
    # 最終バー index を推定（最後のイベント時刻を 4QL=1bar で割る）
    last_ql = max(float(e.get("time", 0.0)) for e in ev)
    bars = int(last_ql // 4.0) + 1
    labels = ["N"] * max(0, bars)
    j = 0
    for b in range(bars):
        t_ql = float(b * 4.0)
        while j + 1 < len(ev) and float(ev[j + 1].get("time", 0.0)) <= t_ql:
            j += 1
        r = (ev[j].get("root") or "N").upper()
        labels[b] = r if r in NAME2PC else "N"
    return labels


def _majority(seq: List[str]) -> str:
    cnt: Dict[str, int] = {}
    for x in seq:
        if not x or x == "N":
            continue
        cnt[x] = cnt.get(x, 0) + 1
    if not cnt:
        return "C"
    return max(cnt.items(), key=lambda kv: kv[1])[0]


def estimate_local_key_sequence(
    chordmap: Dict[str, Any],
    win_bars: int = 4,
    min_hold: int = 4,
    ks_weight: float = 0.6,
) -> Dict[str, Any]:
    """
    バー列からローカルキー（=多数決 root を key とみなす簡易版）を生成し、
    min_hold でデバウンスして転調点を抽出。
    
    ks_weight: Krumhansl-Schmuckler profile との加重平均比率 (0.0~1.0)
               0.5以上でK-Sプロファイル優先、0.0で従来の多数決のみ
    
    Returns {"keys": [key per bar], "modulations": [{"time": ql, "from": k0, "to": k1}, ...]}
    """
    roots = _events_to_bar_roots(chordmap)
    if not roots:
        return {"keys": [], "modulations": []}

    # スライディング多数決 + K-Sプロファイル
    keys_raw: List[str] = []
    n = len(roots)
    for i in range(n):
        lo = max(0, i - win_bars + 1)
        hi = i + 1
        win = roots[lo:hi]
        key_vote = _majority(win)
        
        # K-S: 窓内PCヒスト（root triad近似）
        pc = np.zeros(12)
        for r in win:
            if r == "N":
                continue
            idx = NAME2PC[r]
            pc[idx] += 1.0           # root
            pc[(idx + 4) % 12] += 0.6  # major third
            pc[(idx + 7) % 12] += 0.8  # perfect fifth
        
        if pc.sum() > 0:
            best_key = key_vote
            best_score = -1e9
            for rot in range(12):
                maj = float(np.dot(np.roll(pc, -rot), _KS_MAJOR))
                minr = float(np.dot(np.roll(pc, -rot), _KS_MINOR))
                if maj >= minr:
                    score, cand = maj, ROOTS[rot]
                else:
                    score, cand = minr, ROOTS[rot]
                if score > best_score:
                    best_score, best_key = score, cand
            # blend: KS優先比率 ks_weight（>0.5 を推奨）
            key_blend = best_key if ks_weight >= 0.5 else key_vote
            keys_raw.append(key_blend)
        else:
            keys_raw.append(key_vote)

    # デバウンスして安定列へ
    keys: List[str] = []
    last = None
    span = 0
    for k in keys_raw:
        if k == last:
            span += 1
        else:
            # 直前の短スパンがあれば巻き戻して埋め直す
            if last is not None and span < min_hold and len(keys) >= span:
                for j in range(span):
                    keys[len(keys) - 1 - j] = k  # 新しいキーで塗り替え
            last = k
            span = 1
        keys.append(k)

    # 最後のスパン処理はそのまま

    # 転調点抽出（min_hold 後の最終列から）
    mods: List[Dict[str, Any]] = []
    prev = keys[0]
    for i, k in enumerate(keys[1:], start=1):
        if k != prev:
            mods.append({"time": float(i * 4.0), "from": prev, "to": k})
            prev = k

    return {"keys": keys, "modulations": mods}


def to_key_hints_payload(seq: Dict[str, Any]) -> Dict[str, Any]:
    """Convert estimate_local_key_sequence() output to Stage2 payload fields.
    Returns {"key_hint": [[bar, key], ...], "modulations": [{"time": ql, "to": key}, ...]}
    """
    keys = seq.get("keys", [])
    key_hint = [[i, k] for i, k in enumerate(keys)]
    mods = seq.get("modulations", [])
    # payload としては "to" のみがあれば良い（from は監査用）
    mods_out = [{"time": m.get("time", 0.0), "to": m.get("to", "C")} for m in mods]
    return {"key_hint": key_hint, "modulations": mods_out}


# Backward compatibility alias
def estimate_local_keys(
    chordmap: Dict[str, Any],
    win_bars: int = 8,
) -> Dict[str, Any]:
    """Estimate local key for each bar using sliding window.
    
    Legacy API for backward compatibility. Internally calls estimate_local_key_sequence().
    """
    seq = estimate_local_key_sequence(chordmap, win_bars=win_bars, min_hold=4)
    return to_key_hints_payload(seq)