#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ghost_hh_config.py
--------------------------------------------------
Ghost HH補完ルール（configs/ghost_hh_rules.yaml）の読み込みヘルパー

Usage:
    from scripts.ghost_hh_config import load_ghost_hh_config
    
    config = load_ghost_hh_config()
    max_ghost = config.get_max_ghost_per_bar()
    vel_range = config.get_velocity_range()
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional


class GhostHHConfig:
    """Ghost HH補完ルール設定クラス"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._active_phase = config_dict.get("active_phase", "phase_c")
        self._apply_phase_override()
    
    def _apply_phase_override(self):
        """Phase別設定を適用"""
        phase_configs = self._config.get("phase_config", {})
        phase_override = phase_configs.get(self._active_phase, {})
        
        for key_path, value in phase_override.items():
            # ネストしたキーをドット記法で処理
            keys = key_path.split('.')
            target = self._config
            for k in keys[:-1]:
                target = target.get(k, {})
            if isinstance(target, dict):
                target[keys[-1]] = value
    
    def is_enabled(self) -> bool:
        """Ghost HH補完が有効か"""
        return self._config.get("ghost_hh", {}).get("enabled", True)
    
    def get_min_rel(self) -> float:
        """相対密度下限"""
        return self._config.get("ghost_hh", {}).get("min_rel", 0.40)
    
    def get_max_ghost_per_bar(self) -> int:
        """小節あたり上限"""
        return self._config.get("ghost_hh", {}).get("max_ghost_per_bar", 4)
    
    def get_velocity_range(self) -> Tuple[int, int]:
        """ベロシティ範囲 (min, max)"""
        vel_config = self._config.get("ghost_hh", {}).get("velocity_range", {})
        return (vel_config.get("min", 22), vel_config.get("max", 28))
    
    def get_duration_beats(self) -> float:
        """デュレーション（拍単位）"""
        return self._config.get("ghost_hh", {}).get("duration_beats", 0.20)
    
    def get_ghost_pitch(self) -> int:
        """Ghost HH Pitch"""
        return self._config.get("ghost_hh", {}).get("pitch", 42)
    
    def get_hh_pitches(self) -> List[int]:
        """HH系Pitchセット（カウント対象）"""
        return self._config.get("ghost_hh", {}).get("hh_pitches", [42, 44, 46, 51, 53, 59])
    
    def get_placement_strategy(self) -> str:
        """配置パターン"""
        return self._config.get("ghost_hh", {}).get("placement", {}).get("strategy", "uniform_eighth")
    
    def get_placement_subdivision(self) -> float:
        """配置細分化"""
        return self._config.get("ghost_hh", {}).get("placement", {}).get("subdivision", 0.5)
    
    def is_break_boost_enabled(self) -> bool:
        """Break優遇が有効か"""
        return self._config.get("break_boost", {}).get("enabled", True)
    
    def get_break_activity_threshold(self) -> float:
        """Break判定しきい値"""
        return self._config.get("break_boost", {}).get("activity_threshold", 0.30)
    
    def get_break_boost_factor(self) -> float:
        """Break小節での補完優先度"""
        return self._config.get("break_boost", {}).get("boost_factor", 1.5)
    
    def is_break_verbose(self) -> bool:
        """Break検出時のログ出力"""
        return self._config.get("break_boost", {}).get("verbose", True)
    
    def get_max_ghost_ratio(self) -> float:
        """Ghost HH注入率上限（警告しきい値）"""
        return self._config.get("monitoring", {}).get("max_ghost_ratio", 0.30)
    
    def get_max_consecutive_bars(self) -> int:
        """連続Ghost HH小節数上限（警告しきい値）"""
        return self._config.get("monitoring", {}).get("max_consecutive_bars", 10)
    
    def get_report_path_template(self) -> str:
        """レポート出力先テンプレート"""
        return self._config.get("monitoring", {}).get("report_path", "{song_dir}/ghost_hh_report.json")


def load_ghost_hh_config(config_path: Optional[Path] = None) -> GhostHHConfig:
    """
    Ghost HH補完ルールをロード
    
    Args:
        config_path: YAMLファイルパス（省略時は configs/ghost_hh_rules.yaml）
    
    Returns:
        GhostHHConfig インスタンス
    """
    if config_path is None:
        # デフォルトパス（スクリプト実行位置から推定）
        config_path = Path(__file__).parent.parent / "configs" / "ghost_hh_rules.yaml"
    
    if not config_path.exists():
        # フォールバック（ハードコード値）
        print(f"⚠️  Ghost HH config not found: {config_path}, using defaults")
        return GhostHHConfig({
            "ghost_hh": {
                "enabled": True,
                "min_rel": 0.40,
                "max_ghost_per_bar": 4,
                "velocity_range": {"min": 22, "max": 28},
                "duration_beats": 0.20,
                "pitch": 42,
                "hh_pitches": [42, 44, 46, 51, 53, 59],
                "placement": {"strategy": "uniform_eighth", "subdivision": 0.5},
            },
            "break_boost": {
                "enabled": True,
                "activity_threshold": 0.30,
                "boost_factor": 1.5,
                "verbose": True,
            },
            "monitoring": {
                "max_ghost_ratio": 0.30,
                "max_consecutive_bars": 10,
                "report_path": "{song_dir}/ghost_hh_report.json",
            },
            "active_phase": "phase_c",
        })
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return GhostHHConfig(config_dict)


# CLI テスト用
if __name__ == "__main__":
    import sys
    
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    config = load_ghost_hh_config(config_path)
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Ghost HH Configuration:")
    print(f"  Enabled: {config.is_enabled()}")
    print(f"  Min Rel: {config.get_min_rel()}")
    print(f"  Max Ghost/Bar: {config.get_max_ghost_per_bar()}")
    print(f"  Velocity Range: {config.get_velocity_range()}")
    print(f"  Duration Beats: {config.get_duration_beats()}")
    print(f"  Ghost Pitch: {config.get_ghost_pitch()}")
    print(f"  HH Pitches: {config.get_hh_pitches()}")
    print(f"  Placement Strategy: {config.get_placement_strategy()}")
    print(f"  Placement Subdivision: {config.get_placement_subdivision()}")
    print()
    print("Break Boost:")
    print(f"  Enabled: {config.is_break_boost_enabled()}")
    print(f"  Activity Threshold: {config.get_break_activity_threshold()}")
    print(f"  Boost Factor: {config.get_break_boost_factor()}")
    print(f"  Verbose: {config.is_break_verbose()}")
    print()
    print("Monitoring:")
    print(f"  Max Ghost Ratio: {config.get_max_ghost_ratio()}")
    print(f"  Max Consecutive Bars: {config.get_max_consecutive_bars()}")
    print(f"  Report Path Template: {config.get_report_path_template()}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
