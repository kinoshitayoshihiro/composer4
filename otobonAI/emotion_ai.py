"""
EmotionAI: Emotion Profile 管理と Bar毎感情パラメータ提供

v1:   harmony + section + tempo
v1.5: + lyric_anchors
v2:   + CREPE/OaF
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np

from .rulebook_engine import Rulebook


class EmotionAI:
    """
    Emotion Profile を管理し、Bar毎の感情パラメータを提供
    
    使用例:
        emotion_ai = EmotionAI(
            profile_path=Path("analysis/emotion_profile.json"),
            rulebook_path=Path("configs/otobonAI/rulebook.yaml"),
            lyric_anchors_path=Path("analysis/lyric_anchors.json")
        )
        
        emotion = emotion_ai.get_bar_emotion(
            bar_index=23,
            role="strings",
            context={"chord_symbol": "C#m7", "section": "chorus"}
        )
        
        # → {"energy": 0.68, "tension": 0.72, "anchor_weight": 0.85, ...}
    """
    
    def __init__(
        self,
        profile_path: Path,
        rulebook_path: Optional[Path] = None,
        lyric_anchors_path: Optional[Path] = None,
        crepe_path: Optional[Path] = None
    ):
        """
        Args:
            profile_path: emotion_profile.json のパス
            rulebook_path: rulebook.yaml のパス（optional）
            lyric_anchors_path: lyric_anchors.json のパス（v1.5, optional）
            crepe_path: crepe_pitch.json のパス（v2.0, optional）
        """
        self.profile = self._load_profile(profile_path)
        self.engine = Rulebook.load(rulebook_path) if rulebook_path and rulebook_path.exists() else None
        self.anchors = self._load_anchors(lyric_anchors_path) if lyric_anchors_path and lyric_anchors_path.exists() else {}
        self.crepe = self._load_crepe(crepe_path) if crepe_path and crepe_path.exists() else {}
        
        # Bar index → event mapping
        self.bar_events = {ev["bar"]: ev for ev in self.profile["events"]}
    
    def get_bar_emotion(
        self,
        bar_index: int,
        role: str = "strings",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        指定Barの感情パラメータを取得
        
        Args:
            bar_index: Bar番号
            role: 楽器ロール（strings/bass/piano/drums）
            context: 追加コンテキスト（chord_symbol, section等）
        
        Returns:
            {
                "energy": 0.68,
                "tension": 0.72,
                "brightness": 0.65,
                "valence": 0.70,
                "density": 0.80,
                "anchor_weight": 0.85,        # v1.5
                "has_lyric_stress": True,     # v1.5
                "phrase_position": "end",     # v1.5
                "vocal_focus": True,          # v1.5
                "tags": ["peak", "vocal_climax"]
            }
        """
        # v1 base
        if bar_index not in self.bar_events:
            # デフォルト値
            base = {
                "energy": 0.5,
                "tension": 0.5,
                "brightness": 0.5,
                "valence": 0.5,
                "density": 0.5,
                "tags": []
            }
        else:
            base = dict(self.bar_events[bar_index])
        
        # v1.5: lyric_anchor 補正
        anchor_info = self._apply_lyric_anchor_correction(bar_index, base)
        base.update(anchor_info)
        
        # v2.0: CREPE 補正（未実装）
        if bar_index in self.crepe:
            crepe_info = self._apply_crepe_correction(bar_index, base)
            base.update(crepe_info)
        
        # Rulebook query (optional refinement)
        if self.engine and context:
            full_context = self._build_context(bar_index, role, context)
            actions = self.engine.find_matching(full_context, "emotion")
            base = self._apply_emotion_actions(base, actions)
        
        return base
    
    def _load_profile(self, path: Path) -> Dict:
        """emotion_profile.json 読み込み"""
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_anchors(self, path: Path) -> Dict[int, Dict]:
        """
        lyric_anchors.json 読み込み、Bar単位に集約
        
        Returns:
            {
                bar_idx: {
                    "stress_count": 3,
                    "stress_level": 0.85,
                    "has_stress": True,
                    "phrase_boundaries": ["end"],
                    "classes": ["stress", "sibilant", ...]
                }
            }
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # tempo_map から Bar境界を推定（簡易版）
        # 本来は bars_with_slots.parquet の時間情報を使うべき
        # ここでは 1 bar = 2秒 と仮定（BPM 120 = 2 beats/sec, 4 beats/bar）
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
                    "classes": []
                }
            
            classes = anchor.get("classes", [])
            bar_anchors[bar]["classes"].extend(classes)
            
            if "stress" in classes:
                bar_anchors[bar]["stress_count"] += 1
                bar_anchors[bar]["has_stress"] = True
            
            if "phrase_end" in classes or "boundary" in classes:
                bar_anchors[bar]["phrase_boundaries"].append("end")
        
        # stress_level 計算（0-1）
        for bar, info in bar_anchors.items():
            # stress_count が多いほど重要度高い
            info["stress_level"] = min(1.0, info["stress_count"] * 0.3)
        
        return bar_anchors
    
    def _load_crepe(self, path: Path) -> Dict[int, Dict]:
        """
        CREPE pitch 情報読み込み（v2.0）
        
        Returns:
            {
                bar_idx: {
                    "pitch_avg": 72.3,
                    "energy_avg": 0.78,
                    "confidence_avg": 0.92
                }
            }
        """
        # v2.0 で実装
        return {}
    
    def _apply_lyric_anchor_correction(self, bar_index: int, base: Dict) -> Dict:
        """
        lyric_anchors による補正を適用
        
        v1.5 追加フィールド:
            - anchor_weight: 0-1（歌の主役度）
            - has_lyric_stress: bool
            - phrase_position: "begin"/"mid"/"end"
            - vocal_focus: bool
        """
        result = {
            "anchor_weight": 0.0,
            "has_lyric_stress": False,
            "phrase_position": "mid",
            "vocal_focus": False
        }
        
        if bar_index not in self.anchors:
            return result
        
        anchor = self.anchors[bar_index]
        
        # anchor_weight 設定
        result["anchor_weight"] = anchor.get("stress_level", 0.0)
        result["has_lyric_stress"] = anchor.get("has_stress", False)
        
        # phrase_position 推定
        boundaries = anchor.get("phrase_boundaries", [])
        if "end" in boundaries:
            result["phrase_position"] = "end"
        elif anchor.get("stress_count", 0) > 0:
            result["phrase_position"] = "begin"  # 強勢が多い = フレーズ開始の可能性
        
        # vocal_focus: anchor_weight > 0.5 なら True
        result["vocal_focus"] = result["anchor_weight"] > 0.5
        
        # Energy/Tension 補正
        if result["has_lyric_stress"]:
            base["energy"] = min(1.0, base.get("energy", 0.5) + 0.1)
            base["tension"] = min(1.0, base.get("tension", 0.5) + 0.1)
        
        # phrase_end → tension ピーク
        if result["phrase_position"] == "end":
            base["tension"] = min(1.0, base.get("tension", 0.5) + 0.15)
            if "phrase_end" not in base.get("tags", []):
                base.setdefault("tags", []).append("phrase_end")
        
        # vocal_focus → tags 追加
        if result["vocal_focus"]:
            if "vocal_focus" not in base.get("tags", []):
                base.setdefault("tags", []).append("vocal_focus")
        
        return result
    
    def _apply_crepe_correction(self, bar_index: int, base: Dict) -> Dict:
        """
        CREPE pitch/energy による補正（v2.0）
        """
        # v2.0 で実装
        crepe = self.crepe.get(bar_index, {})
        return {
            "vocal_pitch_avg": crepe.get("pitch_avg", 0.0),
            "vocal_energy_avg": crepe.get("energy_avg", 0.0)
        }
    
    def _build_context(self, bar_index: int, role: str, extra: Dict) -> Dict:
        """BarContext 構築"""
        base = self.bar_events.get(bar_index, {})
        anchor = self.anchors.get(bar_index, {})
        
        return {
            "bar_index": bar_index,
            "section": extra.get("section", base.get("section", "unknown")),
            "role": role,
            "chord_symbol": extra.get("chord_symbol", ""),
            "tempo_bpm": extra.get("tempo_bpm", 120.0),
            "emotion": {
                "local_energy": base.get("energy", 0.5),
                "local_tension": base.get("tension", 0.5)
            },
            "lyric_anchor": {
                "has_anchor": anchor.get("has_stress", False),
                "stress_level": anchor.get("stress_level", 0.0),
                "phrase_pos": "end" if "end" in anchor.get("phrase_boundaries", []) else "mid"
            },
            **extra
        }
    
    def _apply_emotion_actions(self, base: Dict, actions: List) -> Dict:
        """Rulebook actions を base に適用"""
        for action in actions:
            emo_action = action.get_emotion_action()
            if emo_action:
                base["energy"] = np.clip(base["energy"] + emo_action.energy_delta, 0.0, 1.0)
                base["tension"] = np.clip(base["tension"] + emo_action.tension_delta, 0.0, 1.0)
                base["brightness"] = np.clip(base["brightness"] + emo_action.brightness_delta, 0.0, 1.0)
                base["valence"] = np.clip(base["valence"] + emo_action.valence_delta, 0.0, 1.0)
                base["density"] = np.clip(base["density"] + emo_action.density_delta, 0.0, 1.0)
                
                # tags 追加
                existing_tags = set(base.get("tags", []))
                new_tags = existing_tags.union(emo_action.tags_add)
                base["tags"] = list(new_tags)
        
        return base
