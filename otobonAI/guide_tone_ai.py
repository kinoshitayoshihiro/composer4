"""
GuideToneAI: Guide Tone Hints 管理と Bar毎ガイドトーン推奨提供

v1:   harmony + section + tempo
v1.5: + lyric_anchors
v2:   + CREPE/OaF
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np

from .rulebook_engine import Rulebook


class GuideToneAI:
    """
    Guide Tone Hints を管理し、Bar毎のガイドトーン推奨を提供
    
    使用例:
        guidetone_ai = GuideToneAI(
            hints_path=Path("analysis/guide_tone_hints.json"),
            rulebook_path=Path("configs/otobonAI/rulebook.yaml"),
            lyric_anchors_path=Path("analysis/lyric_anchors.json")
        )
        
        guide = guidetone_ai.suggest_for_bar(
            bar_index=23,
            role="strings",
            chord_symbol="C#m7(9)",
            context={"section": "chorus", "slots": {"fill": True}}
        )
        
        # → {"preferred_degrees": [3,7,9], "register": "mid_high", ...}
    """
    
    def __init__(
        self,
        hints_path: Path,
        rulebook_path: Optional[Path] = None,
        lyric_anchors_path: Optional[Path] = None
    ):
        """
        Args:
            hints_path: guide_tone_hints.json のパス
            rulebook_path: rulebook.yaml のパス（optional）
            lyric_anchors_path: lyric_anchors.json のパス（v1.5, optional）
        """
        self.hints = self._load_hints(hints_path)
        self.engine = Rulebook.load(rulebook_path) if rulebook_path and rulebook_path.exists() else None
        self.anchors = self._load_anchors(lyric_anchors_path) if lyric_anchors_path and lyric_anchors_path.exists() else {}
        
        # Bar index → event mapping
        self.bar_events = {ev["bar"]: ev for ev in self.hints["events"]}
    
    def suggest_for_bar(
        self,
        bar_index: int,
        role: str = "strings",
        chord_symbol: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        指定Barのガイドトーンヒントを取得
        
        Args:
            bar_index: Bar番号
            role: 楽器ロール（strings/bass/piano/drums）
            chord_symbol: コードシンボル（"C#m7(9)"等）
            context: 追加コンテキスト（section, slots等）
        
        Returns:
            {
                "preferred_degrees": [3, 7, 9],
                "register": "mid_high",
                "approx_pitch": 72,
                "motion": "step",
                "notes_per_bar": 1.6,
                "lyric_anchor_weight": 0.85,  # v1.5
                "phrase_role": "climax",      # v1.5
                "stress_alignment": True,     # v1.5
                "vowel_rich": False           # v1.5
            }
        """
        # v1 base
        if bar_index not in self.bar_events:
            # デフォルト値
            base = {
                "scale_degree": 3,
                "register": "mid",
                "approx_pitch": 60,
                "motion": "step",
                "notes_per_bar": 1.0
            }
        else:
            base = dict(self.bar_events[bar_index])
        
        # v1 の scale_degree を preferred_degrees に変換
        if "preferred_degrees" not in base and "scale_degree" in base:
            base["preferred_degrees"] = [base["scale_degree"]]
        
        # v1.5: lyric_anchor 補正
        anchor_info = self._apply_lyric_anchor_correction(bar_index, base)
        base.update(anchor_info)
        
        # Rulebook query (optional refinement)
        if self.engine and context:
            full_context = self._build_context(bar_index, role, chord_symbol, context)
            actions = self.engine.find_matching(full_context, "guide_tone")
            base = self._apply_guidetone_actions(base, actions)
        
        return base
    
    def _load_hints(self, path: Path) -> Dict:
        """guide_tone_hints.json 読み込み"""
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_anchors(self, path: Path) -> Dict[int, Dict]:
        """
        lyric_anchors.json 読み込み、Bar単位に集約
        （EmotionAI と同じロジック）
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        bar_duration = 2.0  # 秒（仮定）
        
        bar_anchors = {}
        for anchor in data.get("anchors", []):
            time = anchor.get("time", 0.0)
            bar = int(time / bar_duration)
            
            if bar not in bar_anchors:
                bar_anchors[bar] = {
                    "stress_count": 0,
                    "has_stress": False,
                    "phrase_boundaries": [],
                    "classes": [],
                    "vowel_rich": False
                }
            
            classes = anchor.get("classes", [])
            bar_anchors[bar]["classes"].extend(classes)
            
            if "stress" in classes:
                bar_anchors[bar]["stress_count"] += 1
                bar_anchors[bar]["has_stress"] = True
            
            if "phrase_end" in classes or "boundary" in classes:
                bar_anchors[bar]["phrase_boundaries"].append("end")
            
            # 母音豊か判定（vowel, sustained等のクラス）
            if any(c in classes for c in ["vowel", "sustained", "long"]):
                bar_anchors[bar]["vowel_rich"] = True
        
        # stress_level 計算
        for bar, info in bar_anchors.items():
            info["stress_level"] = min(1.0, info["stress_count"] * 0.3)
        
        return bar_anchors
    
    def _apply_lyric_anchor_correction(self, bar_index: int, base: Dict) -> Dict:
        """
        lyric_anchors による補正を適用
        
        v1.5 追加フィールド:
            - lyric_anchor_weight: 0-1
            - phrase_role: "begin"/"build"/"climax"/"release"
            - stress_alignment: bool
            - vowel_rich: bool
        """
        result = {
            "lyric_anchor_weight": 0.0,
            "phrase_role": "mid",
            "stress_alignment": False,
            "vowel_rich": False
        }
        
        if bar_index not in self.anchors:
            return result
        
        anchor = self.anchors[bar_index]
        
        # anchor_weight 設定
        result["lyric_anchor_weight"] = anchor.get("stress_level", 0.0)
        result["vowel_rich"] = anchor.get("vowel_rich", False)
        
        # 強勢音節 → テンションノート追加
        if anchor.get("has_stress", False):
            result["stress_alignment"] = True
            
            # preferred_degrees にテンション追加
            current_degrees = base.get("preferred_degrees", [base.get("scale_degree", 3)])
            tension_degrees = [9, 11]
            
            # 重複排除
            extended_degrees = list(set(current_degrees + tension_degrees))
            base["preferred_degrees"] = sorted(extended_degrees)
        
        # phrase_boundary 判定
        boundaries = anchor.get("phrase_boundaries", [])
        if "end" in boundaries:
            result["phrase_role"] = "release"
            base["motion"] = "leap_to_resolution"
        elif anchor.get("stress_count", 0) > 2:
            result["phrase_role"] = "climax"
        elif anchor.get("stress_count", 0) > 0:
            result["phrase_role"] = "build"
        
        # 母音豊か → 音数減（伸ばす）
        if result["vowel_rich"]:
            current_notes = base.get("notes_per_bar", 1.0)
            base["notes_per_bar"] = current_notes * 0.8
        
        return result
    
    def _build_context(
        self,
        bar_index: int,
        role: str,
        chord_symbol: str,
        extra: Dict
    ) -> Dict:
        """BarContext 構築"""
        base = self.bar_events.get(bar_index, {})
        anchor = self.anchors.get(bar_index, {})
        
        return {
            "bar_index": bar_index,
            "section": extra.get("section", base.get("section", "unknown")),
            "role": role,
            "chord_symbol": chord_symbol,
            "scale_degree": base.get("scale_degree"),
            "tempo_bpm": extra.get("tempo_bpm", 120.0),
            "lyric_anchor": {
                "has_anchor": anchor.get("has_stress", False),
                "stress_level": anchor.get("stress_level", 0.0),
                "phrase_pos": "end" if "end" in anchor.get("phrase_boundaries", []) else "mid",
                "vowel_rich": anchor.get("vowel_rich", False)
            },
            "slots": extra.get("slots", {}),
            **extra
        }
    
    def _apply_guidetone_actions(self, base: Dict, actions: List) -> Dict:
        """Rulebook actions を base に適用"""
        for action in actions:
            gt_action = action.get_guidetone_action()
            if gt_action:
                # priority_tones 更新
                if gt_action.priority_tones:
                    # 文字列（"3rd", "7th"）を数値に変換
                    degrees = []
                    for tone in gt_action.priority_tones:
                        if tone == "root":
                            degrees.append(1)
                        elif tone == "3rd":
                            degrees.append(3)
                        elif tone == "5th":
                            degrees.append(5)
                        elif tone == "7th":
                            degrees.append(7)
                        elif tone == "9th":
                            degrees.append(9)
                        elif tone == "11th":
                            degrees.append(11)
                        elif tone == "13th":
                            degrees.append(13)
                        elif isinstance(tone, int):
                            degrees.append(tone)
                    
                    if degrees:
                        base["preferred_degrees"] = degrees
                
                # register 更新
                if gt_action.default_register:
                    base["register"] = gt_action.default_register
                
                # motion 更新
                if gt_action.motion:
                    base["motion"] = gt_action.motion
                
                # notes_per_bar 更新
                if gt_action.notes_per_bar is not None:
                    base["notes_per_bar"] = gt_action.notes_per_bar
        
        return base
