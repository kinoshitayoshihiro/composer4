#!/usr/bin/env python3
"""
dynamics_router.py

BarContext から適切なダイナミクスプロファイルを選択するルーター。
dynamics_profiles.yaml と dynamics_routing_bass.yaml を読み込み、
section / chord_function / emotion_tag / vocal_density / kick_pattern_tag / phrase_role
からマッチングして、profile_id と profile データを返す。
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class DynamicsProfile:
    """ダイナミクスプロファイル"""

    profile_id: str
    description: str
    energy_level: str  # low / medium / high
    note_density_scale: Dict[str, float]
    base_velocity: Dict[str, int]
    velocity_spread: Dict[str, int]
    accent_beats: Dict[str, List[float]]
    fill_frequency_scale: Dict[str, float]
    pedal_point_bias: Dict[str, float]
    ghost_note_density: Dict[str, str]


@dataclass
class RoutingRule:
    """ルーティングルール"""

    id: str
    description: str
    target_instrument: str
    section_in: Optional[List[str]]
    chord_function_in: Optional[List[str]]
    emotion_tag_in: Optional[List[str]]
    vocal_density_in: Optional[List[str]]
    kick_pattern_tag_in: Optional[List[str]]
    phrase_role_in: Optional[List[str]]
    profile: str


class DynamicsRouter:
    """ダイナミクスプロファイルルーター"""

    def __init__(
        self,
        profiles_path: str = "config/dynamics_profiles.yaml",
        routing_path: str = "config/dynamics_routing_bass.yaml",
        instrument: str = "bass",
    ):
        """
        Args:
            profiles_path: dynamics_profiles.yaml のパス
            routing_path: dynamics_routing_bass.yaml のパス
            instrument: 対象楽器（bass / drums / guitar）
        """
        self.instrument = instrument
        self.profiles: Dict[str, DynamicsProfile] = {}
        self.routing_rules: List[RoutingRule] = []
        self.default_profile_id: str = "NEUTRAL_BALANCED"

        self._load_profiles(profiles_path)
        self._load_routing_rules(routing_path)

    def _load_profiles(self, path: str):
        """dynamics_profiles.yaml を読み込む"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"dynamics_profiles.yaml not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for profile_id, profile_data in data.get("profiles", {}).items():
            self.profiles[profile_id] = DynamicsProfile(
                profile_id=profile_id,
                description=profile_data.get("description", ""),
                energy_level=profile_data.get("energy_level", "medium"),
                note_density_scale=profile_data.get("note_density_scale", {}),
                base_velocity=profile_data.get("base_velocity", {}),
                velocity_spread=profile_data.get("velocity_spread", {}),
                accent_beats=profile_data.get("accent_beats", {}),
                fill_frequency_scale=profile_data.get("fill_frequency_scale", {}),
                pedal_point_bias=profile_data.get("pedal_point_bias", {}),
                ghost_note_density=profile_data.get("ghost_note_density", {}),
            )

        print(f"✅ Loaded {len(self.profiles)} dynamics profiles from {path}")

    def _load_routing_rules(self, path: str):
        """dynamics_routing_bass.yaml を読み込む"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"dynamics_routing.yaml not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.default_profile_id = data.get("default_profile", "NEUTRAL_BALANCED")

        for rule_data in data.get("routing_rules", []):
            self.routing_rules.append(
                RoutingRule(
                    id=rule_data.get("id", ""),
                    description=rule_data.get("description", ""),
                    target_instrument=rule_data.get("target_instrument", "bass"),
                    section_in=rule_data.get("section_in"),
                    chord_function_in=rule_data.get("chord_function_in"),
                    emotion_tag_in=rule_data.get("emotion_tag_in"),
                    vocal_density_in=rule_data.get("vocal_density_in"),
                    kick_pattern_tag_in=rule_data.get("kick_pattern_tag_in"),
                    phrase_role_in=rule_data.get("phrase_role_in"),
                    profile=rule_data.get("profile", "NEUTRAL_BALANCED"),
                )
            )

        print(f"✅ Loaded {len(self.routing_rules)} routing rules from {path}")
        print(f"   Default profile: {self.default_profile_id}")

    def _matches_condition(self, value: Any, condition: Optional[List[str]]) -> bool:
        """条件マッチング判定

        Args:
            value: BarContext の値（例: "chorus"）
            condition: ルールの条件リスト（例: ["chorus", "climax"]）

        Returns:
            True: マッチ（条件が None または value が条件リストに含まれる）
            False: 不一致
        """
        if condition is None:
            return True  # 条件なし = 常にマッチ
        if value is None:
            return False  # 値が None で条件がある = 不一致
        return value in condition

    def match_profile(
        self,
        section: Optional[str] = None,
        chord_function: Optional[str] = None,
        emotion_tag: Optional[str] = None,
        vocal_density: Optional[str] = None,
        kick_pattern_tag: Optional[str] = None,
        phrase_role: Optional[str] = None,
    ) -> DynamicsProfile:
        """BarContext からダイナミクスプロファイルを選択

        Args:
            section: セクション（verse, chorus等）
            chord_function: コード機能（TONIC, DOMINANT等）
            emotion_tag: 感情タグ（BRIGHT_ENERGETIC等）
            vocal_density: ボーカル密度（sparse, medium, high_density等）
            kick_pattern_tag: キックパターンタグ（ROCK_8BEAT_STANDARD等）
            phrase_role: フレーズロール（DRIVE_FORWARD等）

        Returns:
            マッチしたダイナミクスプロファイル
        """
        # 上から順にルールを評価
        for rule in self.routing_rules:
            # 楽器チェック
            if rule.target_instrument != self.instrument:
                continue

            # 各条件をチェック
            if not self._matches_condition(section, rule.section_in):
                continue
            if not self._matches_condition(chord_function, rule.chord_function_in):
                continue
            if not self._matches_condition(emotion_tag, rule.emotion_tag_in):
                continue
            if not self._matches_condition(vocal_density, rule.vocal_density_in):
                continue
            if not self._matches_condition(kick_pattern_tag, rule.kick_pattern_tag_in):
                continue
            if not self._matches_condition(phrase_role, rule.phrase_role_in):
                continue

            # すべての条件がマッチ → このルールのプロファイルを返す
            profile_id = rule.profile
            if profile_id in self.profiles:
                return self.profiles[profile_id]
            else:
                print(f"⚠️  Rule {rule.id} references unknown profile: {profile_id}")
                continue

        # どのルールにもマッチしない → デフォルトプロファイル
        return self.profiles.get(self.default_profile_id, self.profiles["NEUTRAL_BALANCED"])

    def get_profile_summary(
        self, profile: DynamicsProfile, instrument: str = "bass"
    ) -> Dict[str, Any]:
        """プロファイルのサマリーを取得（楽器固有の値のみ）

        Args:
            profile: ダイナミクスプロファイル
            instrument: 楽器名（bass / drums / guitar）

        Returns:
            楽器固有のサマリー辞書
        """
        return {
            "profile_id": profile.profile_id,
            "description": profile.description,
            "energy_level": profile.energy_level,
            "note_density_scale": profile.note_density_scale.get(instrument, 1.0),
            "base_velocity": profile.base_velocity.get(instrument, 80),
            "velocity_spread": profile.velocity_spread.get(instrument, 8),
            "accent_beats": profile.accent_beats.get(instrument, [0.0]),
            "fill_frequency_scale": profile.fill_frequency_scale.get(instrument, 1.0),
            "pedal_point_bias": profile.pedal_point_bias.get(instrument, 0.3),
            "ghost_note_density": profile.ghost_note_density.get("drums", "medium"),
        }


# ===== サンプル実行 =====
if __name__ == "__main__":
    import sys

    # DynamicsRouter 初期化
    router = DynamicsRouter(
        profiles_path="config/dynamics_profiles.yaml",
        routing_path="config/dynamics_routing_bass.yaml",
        instrument="bass",
    )

    print("\n" + "=" * 60)
    print("DynamicsRouter サンプル実行")
    print("=" * 60)

    # テストケース1: SAD系・静かなAメロ
    print("\n【テストケース1】SAD系・静かなAメロ")
    print("  section='verse', emotion_tag='SAD_RESIGNED', phrase_role='FOUNDATION_CALM'")
    profile1 = router.match_profile(
        section="verse", emotion_tag="SAD_RESIGNED", phrase_role="FOUNDATION_CALM"
    )
    summary1 = router.get_profile_summary(profile1, "bass")
    print(f"  → Profile: {summary1['profile_id']}")
    print(f"     {summary1['description']}")
    print(
        f"     base_velocity={summary1['base_velocity']}, pedal_point_bias={summary1['pedal_point_bias']}"
    )

    # テストケース2: サビ＋明るめ
    print("\n【テストケース2】サビ＋明るめ感情")
    print("  section='chorus', emotion_tag='BRIGHT_ENERGETIC', phrase_role='DRIVE_FORWARD'")
    profile2 = router.match_profile(
        section="chorus", emotion_tag="BRIGHT_ENERGETIC", phrase_role="DRIVE_FORWARD"
    )
    summary2 = router.get_profile_summary(profile2, "bass")
    print(f"  → Profile: {summary2['profile_id']}")
    print(f"     {summary2['description']}")
    print(
        f"     base_velocity={summary2['base_velocity']}, fill_frequency_scale={summary2['fill_frequency_scale']}"
    )

    # テストケース3: 浮遊感のあるブリッジ
    print("\n【テストケース3】浮遊感のあるブリッジ")
    print("  section='bridge', emotion_tag='MYSTERIOUS_FLOATING', phrase_role='FLOAT_AMBIENT'")
    profile3 = router.match_profile(
        section="bridge", emotion_tag="MYSTERIOUS_FLOATING", phrase_role="FLOAT_AMBIENT"
    )
    summary3 = router.get_profile_summary(profile3, "bass")
    print(f"  → Profile: {summary3['profile_id']}")
    print(f"     {summary3['description']}")
    print(
        f"     note_density_scale={summary3['note_density_scale']}, pedal_point_bias={summary3['pedal_point_bias']}"
    )

    # テストケース4: 条件なし（デフォルト）
    print("\n【テストケース4】条件なし（デフォルト）")
    profile4 = router.match_profile()
    summary4 = router.get_profile_summary(profile4, "bass")
    print(f"  → Profile: {summary4['profile_id']}")
    print(f"     {summary4['description']}")

    print("\n✅ サンプル実行完了")
