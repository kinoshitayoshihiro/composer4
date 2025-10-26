#!/usr/bin/env python3
"""
和声再推定カスケード - quality:"" を創作せず再推定 or N/5で明示
"""
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ==================== PC辞書定義 ====================
# Pitch Class Set → (quality, tension_hints)
PCSET_TO_QUALITY = {
    # 3和音
    frozenset([0, 4, 7]): ("maj", []),
    frozenset([0, 3, 7]): ("m", []),
    frozenset([0, 3, 6]): ("dim", []),
    frozenset([0, 4, 8]): ("aug", []),
    
    # sus
    frozenset([0, 5, 7]): ("sus4", []),
    frozenset([0, 2, 7]): ("sus2", []),
    
    # 7th系
    frozenset([0, 4, 7, 11]): ("maj7", []),
    frozenset([0, 4, 7, 10]): ("7", []),
    frozenset([0, 3, 7, 10]): ("m7", []),
    frozenset([0, 3, 7, 11]): ("mM7", []),
    frozenset([0, 3, 6, 10]): ("m7b5", []),  # half-diminished
    frozenset([0, 3, 6, 9]): ("dim7", []),
    
    # 6th
    frozenset([0, 4, 7, 9]): ("6", []),
    frozenset([0, 3, 7, 9]): ("m6", []),
    
    # 9th系（4音でも9thヒント）
    frozenset([0, 4, 7, 10, 2]): ("9", [9]),
    frozenset([0, 3, 7, 10, 2]): ("m9", [9]),
    frozenset([0, 4, 7, 11, 2]): ("maj9", [9]),
    
    # add系
    frozenset([0, 4, 7, 2]): ("add9", [9]),
    frozenset([0, 3, 7, 2]): ("madd9", [9]),
    frozenset([0, 4, 7, 5]): ("add11", [11]),
    
    # パワーコード（完全5度のみ）
    frozenset([0, 7]): ("5", []),
    
    # 13th系（5音以上）
    frozenset([0, 4, 7, 10, 2, 9]): ("13", [9, 13]),
}

# ==================== ヘルパー関数 ====================
def normalize_pcset(pc_list: List[int]) -> frozenset:
    """PCリストを0基準に正規化"""
    if not pc_list:
        return frozenset()
    pc_set = set(x % 12 for x in pc_list)
    if not pc_set:
        return frozenset()
    root = min(pc_set)
    return frozenset((x - root) % 12 for x in pc_set)

def pcset_lookup(pc_list: List[int]) -> Optional[Tuple[str, List[int]]]:
    """PC辞書で直判定"""
    pcset = normalize_pcset(pc_list)
    return PCSET_TO_QUALITY.get(pcset)

def is_powerchord(pc_list: List[int]) -> bool:
    """2音で完全5度を含むか"""
    pcset = normalize_pcset(pc_list)
    return pcset == frozenset([0, 7])

def extract_bass(event: Dict) -> Optional[int]:
    """イベントからベース音（最低音 or bass role）を抽出"""
    # 仮実装: pcsetの最低音
    pcset = event.get("pcset", [])
    if not pcset:
        return None
    return min(pcset) % 12

def key_probability(pc_list: List[int], key_profile: Dict[int, float]) -> float:
    """キー確率（Krumhansl風）での整合スコア"""
    # 簡易版: PCの重み付き和
    if not pc_list or not key_profile:
        return 0.5
    score = sum(key_profile.get(pc % 12, 0.1) for pc in pc_list)
    return min(1.0, score / len(pc_list))

def bass_aided_candidate(
    event: Dict,
    prev_event: Optional[Dict],
    next_event: Optional[Dict],
    key_profile: Dict[int, float]
) -> Optional[Tuple[str, float]]:
    """ベース優先でroot固定→品質再評価"""
    bass = extract_bass(event)
    if bass is None:
        return None
    
    pcset = event.get("pcset", [])
    if not pcset:
        return None
    
    # ベースをrootとしてPC再配置
    normalized = frozenset((x - bass) % 12 for x in pcset)
    result = PCSET_TO_QUALITY.get(normalized)
    if result:
        quality, tensions = result
        conf = key_probability(pcset, key_profile) * 0.7  # ベース推定は信頼度やや低
        return (quality, conf)
    return None

@dataclass
class ChordCandidate:
    root: str
    quality: str
    tensions: List[int]
    confidence: float
    fix_flags: List[str]

# ==================== カスケード本体 ====================
def reestimate_chord(
    event: Dict,
    prev_event: Optional[Dict],
    next_event: Optional[Dict],
    key_profile: Dict[int, float]
) -> ChordCandidate:
    """
    quality:""のイベントを再推定
    
    カスケード順:
    1. PC辞書直判定
    2. ベース優先
    3. 短ギャップ補完（前後が同じ→補間）
    4. パワーコード判定
    5. fallback: N (No-Chord)
    """
    pcset = event.get("pcset", [])
    root_str = event.get("root", "C")
    
    # 1) PC辞書
    lookup = pcset_lookup(pcset)
    if lookup:
        quality, tensions = lookup
        conf = key_probability(pcset, key_profile) * 0.9
        return ChordCandidate(
            root=root_str,
            quality=quality,
            tensions=tensions,
            confidence=max(0.6, conf),
            fix_flags=["pc_lookup"]
        )
    
    # 2) ベース優先
    bass_cand = bass_aided_candidate(event, prev_event, next_event, key_profile)
    if bass_cand:
        quality, conf = bass_cand
        return ChordCandidate(
            root=root_str,
            quality=quality,
            tensions=[],
            confidence=conf,
            fix_flags=["bass_aided"]
        )
    
    # 3) 短ギャップ補完（前後が同じ品質なら補間）
    if prev_event and next_event:
        prev_q = prev_event.get("quality", "")
        next_q = next_event.get("quality", "")
        if prev_q and prev_q == next_q and prev_q not in ["", "N"]:
            return ChordCandidate(
                root=prev_event.get("root", "C"),
                quality=prev_q,
                tensions=prev_event.get("tensions", []),
                confidence=0.5,
                fix_flags=["short_gap_fill"]
            )
    
    # 4) パワーコード判定
    if is_powerchord(pcset):
        return ChordCandidate(
            root=root_str,
            quality="5",
            tensions=[],
            confidence=0.45,
            fix_flags=["fallback_5"]
        )
    
    # 5) No-Chord fallback
    return ChordCandidate(
        root=root_str,
        quality="N",
        tensions=[],
        confidence=0.2,
        fix_flags=["fallback_N"]
    )

# ==================== ラベル強度判定 ====================
def label_strength(confidence: float, fix_flags: List[str]) -> str:
    """gold / silver / bronze"""
    if "fallback_N" in fix_flags or "fallback_5" in fix_flags:
        return "bronze"
    if confidence >= 0.6 and not fix_flags:
        return "gold"
    if confidence >= 0.5:
        return "silver"
    return "bronze"

# ==================== chordmap全体の再推定 ====================
def reestimate_chordmap(
    chordmap: Dict,
    key_profile: Optional[Dict[int, float]] = None
) -> Tuple[Dict, Dict]:
    """
    chordmap.json全体を再推定
    
    Returns:
        (fixed_chordmap, qa_metrics)
    """
    if key_profile is None:
        # デフォルトC major Krumhansl profile（簡易版）
        key_profile = {
            0: 1.0, 2: 0.6, 4: 0.7, 5: 0.5, 7: 0.8, 9: 0.6, 11: 0.5,  # C major
            1: 0.2, 3: 0.2, 6: 0.3, 8: 0.2, 10: 0.3  # 非スケール音
        }
    
    events = chordmap.get("events", [])
    if not events:
        return chordmap, {"empty": True}
    
    fixed_events = []
    stats = {
        "total": len(events),
        "reestimated": 0,
        "gold": 0,
        "silver": 0,
        "bronze": 0,
        "N_count": 0,
        "powerchord_count": 0,
        "fix_flags_histogram": {}
    }
    
    for i, ev in enumerate(events):
        prev_ev = events[i-1] if i > 0 else None
        next_ev = events[i+1] if i < len(events)-1 else None
        
        if ev.get("quality") == "":
            # 再推定
            cand = reestimate_chord(ev, prev_ev, next_ev, key_profile)
            ev_fixed = {
                "time": ev.get("time", 0.0),
                "root": cand.root,
                "quality": cand.quality,
                "tensions": cand.tensions,
                "confidence": round(cand.confidence, 3),
                "fix_flags": cand.fix_flags,
                "label_strength": label_strength(cand.confidence, cand.fix_flags)
            }
            stats["reestimated"] += 1
            
            # 統計
            if cand.quality == "N":
                stats["N_count"] += 1
            elif cand.quality == "5":
                stats["powerchord_count"] += 1
            
            for flag in cand.fix_flags:
                stats["fix_flags_histogram"][flag] = stats["fix_flags_histogram"].get(flag, 0) + 1
        else:
            # 既存のまま（label_strength付与）
            conf = ev.get("confidence", 0.7)
            ev_fixed = ev.copy()
            ev_fixed["label_strength"] = label_strength(conf, ev.get("fix_flags", []))
        
        # 強度集計
        strength = ev_fixed.get("label_strength", "bronze")
        stats[strength] = stats.get(strength, 0) + 1
        
        fixed_events.append(ev_fixed)
    
    fixed_chordmap = chordmap.copy()
    fixed_chordmap["events"] = fixed_events
    fixed_chordmap["provenance"] = fixed_chordmap.get("provenance", {})
    fixed_chordmap["provenance"]["note"] = "reestimated with cascade (no synthetic 'maj'补完)"
    
    # QAメトリクス計算
    total = stats["total"]
    qa = {
        "total_events": total,
        "reestimated_count": stats["reestimated"],
        "reestimated_rate": stats["reestimated"] / total if total > 0 else 0.0,
        "gold_count": stats["gold"],
        "silver_count": stats["silver"],
        "bronze_count": stats["bronze"],
        "bronze_rate": stats["bronze"] / total if total > 0 else 0.0,
        "N_count": stats["N_count"],
        "N_rate": stats["N_count"] / total if total > 0 else 0.0,
        "powerchord_count": stats["powerchord_count"],
        "powerchord_rate": stats["powerchord_count"] / total if total > 0 else 0.0,
        "avg_confidence": sum(ev.get("confidence", 0.5) for ev in fixed_events) / total if total > 0 else 0.0,
        "fix_flags_histogram": stats["fix_flags_histogram"]
    }
    
    return fixed_chordmap, qa

# ==================== CLI用（単一ファイル再推定）====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="和声再推定（単一ファイル）")
    parser.add_argument("input_json", help="入力chordmap.json")
    parser.add_argument("--output", help="出力先（指定なしはinput.fixed.json）")
    parser.add_argument("--qa-output", help="QAメトリクス出力先")
    args = parser.parse_args()
    
    with open(args.input_json, "r") as f:
        original = json.load(f)
    
    fixed, qa = reestimate_chordmap(original)
    
    output_path = args.output or args.input_json.replace(".json", ".fixed.json")
    with open(output_path, "w") as f:
        json.dump(fixed, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Fixed: {output_path}")
    print(f"  Reestimated: {qa['reestimated_count']}/{qa['total_events']} ({qa['reestimated_rate']*100:.1f}%)")
    print(f"  Bronze rate: {qa['bronze_rate']*100:.1f}%")
    print(f"  N-Chord: {qa['N_count']} ({qa['N_rate']*100:.1f}%)")
    print(f"  PowerChord: {qa['powerchord_count']} ({qa['powerchord_rate']*100:.1f}%)")
    
    if args.qa_output:
        with open(args.qa_output, "w") as f:
            json.dump(qa, f, indent=2)
        print(f"✓ QA: {args.qa_output}")
