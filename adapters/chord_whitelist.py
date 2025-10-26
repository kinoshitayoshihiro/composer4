from __future__ import annotations
from typing import Dict, Any, List, Tuple

def _m21_ok(sym: str) -> Tuple[bool, str]:
    try:
        from music21 import harmony
        cs = harmony.ChordSymbol(sym)
        return True, cs.figure  # music21の正規化表記
    except Exception:
        return False, sym

def validate_events(events: List[Dict[str,Any]]) -> Tuple[List[Dict[str,Any]], Dict[str,int]]:
    """
    events: [{"time":QL,"root":...,"quality":...,"confidence":...}, ...]
    - music21で解釈できるものだけ残す
    - figure（root+quality）を正規化し、必要なら修正（fixed）カウント
    """
    cleaned: List[Dict[str,Any]] = []
    stats = {"total":0, "valid":0, "fixed":0, "dropped":0}
    for e in events:
        stats["total"] += 1
        root = (e.get("root") or "").strip()
        qual = (e.get("quality") or "").strip()
        if root == "N":
            cleaned.append(e); stats["valid"] += 1; continue
        fig = f"{root}{qual}"
        ok, canon = _m21_ok(fig)
        if ok:
            if canon != fig:
                e = dict(e)
                i = 1
                if len(canon) >= 2 and canon[1] in ("#","-"):
                    i = 2
                e["root"], e["quality"] = canon[:i], canon[i:]
                stats["fixed"] += 1
            cleaned.append(e); stats["valid"] += 1
        else:
            stats["dropped"] += 1
    return cleaned, stats
