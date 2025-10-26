#!/usr/bin/env python3
"""
chordmap_unify.py - Chordmap Schema Unifier

バラバラな chordmap（秒/QL・配列/辞書・"Am7"表記 等）を統一スキーマ
{unit:"ql", events:[{time, root, quality}]} に正規化。

機能:
- 秒 → QL 変換（tempo_map対応）
- シンボル表記 → root/quality 分解
- 短いN（休符）の除去
- X-N-X パターンの吸収
- QL snap（グリッド丸め）
"""
from __future__ import annotations
import json
import math
import re
from typing import Dict, List, Any, Tuple, Optional

ROOTS = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]
ALIASES = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

_CHORD_RE = re.compile(r"^(?P<root>[A-G](?:#|b)?)(?P<qual>.*)$")


def _norm_root(s: str) -> Optional[str]:
    """Root正規化（#に統一、大文字化）"""
    if not s:
        return None
    s = s.strip().title()
    return ALIASES.get(s, s) if s in ALIASES or s in ROOTS else None


def _qual_from_suffix(suf: str) -> str:
    """サフィックスからquality推定"""
    s = (suf or "").lower().strip()
    if s in ("", "maj"):
        return "maj"
    if s in ("m", "min"):
        return "min"
    if s in ("maj7", "ma7", "Δ7", "△7"):
        return "maj7"
    if s in ("m7", "min7"):
        return "min7"
    if s in ("7", "dom7"):
        return "dom7"
    if s in ("m7b5", "min7b5", "ø7", "ø"):
        return "min7b5"
    if s in ("dim", "dim7", "o7", "°7"):
        return "dim7"
    if s in ("sus2",):
        return "sus2"
    if s in ("sus4", "sus"):
        return "sus4"
    if s in ("add9",):
        return "add9"
    if s in ("6", "maj6"):
        return "maj6"
    if s in ("m6", "min6"):
        return "min6"
    if s in ("aug", "+"):
        return "aug"
    # フォールバック：未対応はそのまま
    return s if s else "maj"


def _parse_chord_symbol(sym: str) -> Tuple[str, str]:
    """コード記号を (root, quality) に分解"""
    if not isinstance(sym, str):
        return ("N", "")
    sym = sym.strip()
    if sym.upper() in ("N", "NC", "X", ""):
        return ("N", "")
    m = _CHORD_RE.match(sym)
    if not m:
        return ("N", "")
    rt = _norm_root(m.group("root"))
    if not rt:
        return ("N", "")
    return (rt, _qual_from_suffix(m.group("qual")))


def _to_ql(sec: float, tempo_map: Optional[List[Tuple[float, float]]] = None) -> float:
    """秒 → QL変換
    
    tempo_map: List[(t_sec, bpm)] 昇順
    なければ 120bpm を仮定
    """
    if not tempo_map:
        bpm = 120.0
        return float(sec) * bpm / 60.0 * 4.0
    
    # 近傍bpm
    idx = 0
    for i in range(len(tempo_map)):
        if tempo_map[i][0] <= sec:
            idx = i
        else:
            break
    bpm = max(1e-6, float(tempo_map[idx][1]))
    return float(sec) * bpm / 60.0 * 4.0


def _snap(x: float, q: Optional[float]) -> float:
    """QLグリッドに丸める"""
    if not q or q <= 0:
        return float(x)
    return round(x / q) * q


def unify_chordmap_dict(
    data: Any,
    *,
    to_unit: str = "ql",
    snap_ql: Optional[float] = None,
    merge_N: bool = False,
    min_N_ql: float = 0.0,
    glue_same_root: bool = False,
    tempo_map: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """入力形式の揺れ（配列/辞書/秒表記/シンボル表記）を正規化
    
    出力: {"unit":"ql", "events":[{"time":..., "root":"C", "quality":"maj"}, ...]}
    
    Args:
        data: 入力chordmap（配列/辞書/既存スキーマ）
        to_unit: "ql" or "sec" (既定: "ql")
        snap_ql: QLグリッドに丸める（Noneで無効）
        merge_N: 短いNを削除
        min_N_ql: Nの最短長（これ未満は除去）
        glue_same_root: X -> N -> X を X に吸収
        tempo_map: 秒→QL変換用のテンポマップ
    """
    
    def _emit(evts: List[Dict[str, Any]]):
        """イベントリストを統一スキーマに変換"""
        out = []
        for e in evts:
            # time取得（time/ql/sec等の表記揺れ対応）
            t = float(e.get("time", e.get("ql", e.get("sec", 0.0))))
            
            # unit変換
            u = (data.get("unit") if isinstance(data, dict) else None) or e.get("unit")
            if u and u != to_unit:
                if u == "sec" and to_unit == "ql":
                    t = _to_ql(t, tempo_map)
                elif u == "ql" and to_unit == "sec":
                    # QL→秒は逆算（簡易実装）
                    if tempo_map:
                        bpm = tempo_map[0][1] if tempo_map else 120.0
                    else:
                        bpm = 120.0
                    t = t * 60.0 / (4.0 * bpm)
            
            # QL snap
            if to_unit == "ql" and snap_ql:
                t = _snap(t, snap_ql)
            
            # root/quality抽出
            root = e.get("root")
            qual = e.get("quality")
            chord = e.get("chord")
            
            if chord and (not root or not qual):
                root, qual = _parse_chord_symbol(chord)
            
            if not root:
                root = "N" if (qual == "" or qual is None) else "C"
            
            if root == "N":
                qual = ""
            
            out.append({
                "time": float(t),
                "root": str(root),
                "quality": str(qual or "")
            })
        
        # 時系列ソート
        out.sort(key=lambda x: x["time"])
        
        # N（休符）のマージ処理
        if merge_N:
            tmp = []
            for i, e in enumerate(out):
                if e["root"] == "N":
                    # 持続長を次イベントまでとみなす
                    t0 = e["time"]
                    t1 = out[i + 1]["time"] if i + 1 < len(out) else t0 + 1000.0
                    
                    # 短すぎるN除去
                    if (t1 - t0) < min_N_ql:
                        continue
                    
                    # X-N-X吸収
                    if glue_same_root and i > 0 and i + 1 < len(out):
                        if out[i - 1]["root"] == out[i + 1]["root"]:
                            continue
                
                tmp.append(e)
            out = tmp
        
        return {"unit": to_unit, "events": out}
    
    # 入力形式の判定と変換
    if isinstance(data, dict):
        # 既存スキーマ: {"unit":"ql", "events":[...]}
        if "events" in data:
            return _emit(list(data["events"]))
        
        # 辞書形式: {"0.0": "Am", "4.0": "C7", ...}
        try:
            ev = []
            for k, v in data.items():
                if k in ("unit", "tempo_map", "key_changes"):
                    continue  # メタデータはスキップ
                try:
                    t = float(k)
                except Exception:
                    continue
                
                if isinstance(v, dict):
                    ev.append({"time": t, **v})
                else:
                    rt, ql = _parse_chord_symbol(str(v))
                    ev.append({"time": t, "root": rt, "quality": ql})
            
            if ev:
                return _emit(ev)
        except Exception:
            pass
    
    if isinstance(data, list):
        # 配列形式: [{"ql": 0.0, "chord": "Bm7"}, ...]
        return _emit(list(data))
    
    raise ValueError(f"Unsupported chordmap format: {type(data)}")


def unify_file(in_path: str, out_path: str, **kw):
    """ファイル単位での統一処理"""
    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    uni = unify_chordmap_dict(raw, **kw)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(uni, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Chordmap Schema Unifier")
    ap.add_argument("--input", required=True, help="Input chordmap JSON")
    ap.add_argument("--output", required=True, help="Output unified JSON")
    ap.add_argument("--to-unit", default="ql", choices=["ql", "sec"])
    ap.add_argument("--snap-ql", type=float, default=None, help="Snap to QL grid")
    ap.add_argument("--merge-N", action="store_true", help="Merge short N chords")
    ap.add_argument("--min-N-ql", type=float, default=0.0, help="Min N duration")
    ap.add_argument("--glue-same-root", action="store_true", help="Glue X-N-X patterns")
    
    args = ap.parse_args()
    
    unify_file(
        args.input,
        args.output,
        to_unit=args.to_unit,
        snap_ql=args.snap_ql,
        merge_N=args.merge_N,
        min_N_ql=args.min_N_ql,
        glue_same_root=args.glue_same_root,
    )
    
    print(f"[OK] Unified chordmap: {args.output}")
