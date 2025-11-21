#!/usr/bin/env python3
"""
EmotionAI Profile Loader - Phase 125
階層プリセット（CLI > song > project > style > base）読み込み
Deep merge with override priority
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dicts (override優先)
    
    Args:
        base: ベース辞書
        override: オーバーライド辞書
    
    Returns:
        マージ結果（override優先）
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_emotion_profile(
    base_path: Path = Path("configs/emotion_profiles/base.yaml"),
    style: Optional[str] = None,
    project_path: Optional[Path] = None,
    song_path: Optional[Path] = None,
    cli_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    階層プリセット読み込み（優先順位: CLI > song > project > style > base）
    
    Args:
        base_path: base.yamlパス（デフォルト: configs/emotion_profiles/base.yaml）
        style: スタイルプリセット名（例: "ballad" → styles/ballad.yaml）
        project_path: プロジェクトレベルのYAMLパス
        song_path: ソングレベルのYAMLパス
        cli_override: CLI指定のオーバーライド辞書
    
    Returns:
        マージ済みemotionプロファイル辞書
    
    Example:
        >>> profile = load_emotion_profile(
        ...     style="ballad",
        ...     song_path=Path("configs/emotion_profiles/songs/song001.yaml"),
        ...     cli_override={"sections": {"chorus": {"energy": 0.95}}}
        ... )
        >>> profile["sections"]["chorus"]["energy"]
        0.95  # CLI指定が最優先
    """
    # 1. Base読み込み
    if not base_path.exists():
        raise FileNotFoundError(f"Base profile not found: {base_path}")
    
    with open(base_path, "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f)
    
    # 2. Style層（存在すれば）
    if style:
        style_path = base_path.parent / "styles" / f"{style}.yaml"
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                style_data = yaml.safe_load(f)
            profile = deep_merge(profile, style_data)
    
    # 3. Project層（存在すれば）
    if project_path and project_path.exists():
        with open(project_path, "r", encoding="utf-8") as f:
            project_data = yaml.safe_load(f)
        profile = deep_merge(profile, project_data)
    
    # 4. Song層（存在すれば）
    if song_path and song_path.exists():
        with open(song_path, "r", encoding="utf-8") as f:
            song_data = yaml.safe_load(f)
        profile = deep_merge(profile, song_data)
    
    # 5. CLI層（存在すれば）
    if cli_override:
        profile = deep_merge(profile, cli_override)
    
    return profile


def validate_emotion_profile(profile: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    EmotionプロファイルYAMLのバリデーション
    
    Args:
        profile: Emotionプロファイル辞書
    
    Returns:
        (valid: bool, errors: list[str])
    
    Checks:
        - scale.energy存在（min=0.0, max=1.0）
        - scale.valence存在（min=-1.0, max=1.0）
        - sections各値がscale範囲内
        - instrument_map各キーの存在性
    """
    errors = []
    
    # scale存在チェック
    if "scale" not in profile:
        errors.append("Missing 'scale' key")
        return False, errors
    
    scale = profile["scale"]
    if "energy" not in scale:
        errors.append("Missing 'scale.energy'")
    else:
        e = scale["energy"]
        if e.get("min") != 0.0 or e.get("max") != 1.0:
            errors.append(f"Invalid scale.energy range: min={e.get('min')}, max={e.get('max')} (expected 0.0..1.0)")
    
    if "valence" not in scale:
        errors.append("Missing 'scale.valence'")
    else:
        v = scale["valence"]
        if v.get("min") != -1.0 or v.get("max") != 1.0:
            errors.append(f"Invalid scale.valence range: min={v.get('min')}, max={v.get('max')} (expected -1.0..1.0)")
    
    # sections範囲チェック
    if "sections" in profile:
        for sec_name, sec_data in profile["sections"].items():
            if "energy" in sec_data:
                e = sec_data["energy"]
                if not (0.0 <= e <= 1.0):
                    errors.append(f"Section '{sec_name}' energy out of range: {e} (expected 0.0..1.0)")
            if "valence" in sec_data:
                v = sec_data["valence"]
                if not (-1.0 <= v <= 1.0):
                    errors.append(f"Section '{sec_name}' valence out of range: {v} (expected -1.0..1.0)")
    
    # instrument_map存在チェック
    if "instrument_map" not in profile:
        errors.append("Missing 'instrument_map'")
    
    return len(errors) == 0, errors


def get_section_emotion(
    profile: Dict[str, Any],
    section_label: str,
    fallback_defaults: bool = True
) -> Dict[str, float]:
    """
    セクションのenergy/valence取得
    
    Args:
        profile: Emotionプロファイル辞書
        section_label: セクションラベル（例: "chorus"）
        fallback_defaults: セクション未定義時にdefaults使用するか
    
    Returns:
        {"energy": float, "valence": float}
    
    Example:
        >>> profile = load_emotion_profile(style="ballad")
        >>> get_section_emotion(profile, "chorus")
        {'energy': 0.85, 'valence': 0.15}
    """
    sections = profile.get("sections", {})
    
    if section_label in sections:
        sec = sections[section_label]
        return {
            "energy": sec.get("energy", profile.get("defaults", {}).get("energy", 0.5)),
            "valence": sec.get("valence", profile.get("defaults", {}).get("valence", 0.0))
        }
    
    if fallback_defaults and "defaults" in profile:
        return {
            "energy": profile["defaults"].get("energy", 0.5),
            "valence": profile["defaults"].get("valence", 0.0)
        }
    
    # 最終フォールバック
    return {"energy": 0.5, "valence": 0.0}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EmotionAI Profile Loader")
    parser.add_argument("--base", type=Path, default=Path("configs/emotion_profiles/base.yaml"),
                        help="Base YAML path")
    parser.add_argument("--style", type=str, help="Style preset name (e.g. 'ballad')")
    parser.add_argument("--project", type=Path, help="Project-level YAML path")
    parser.add_argument("--song", type=Path, help="Song-level YAML path")
    parser.add_argument("--validate", action="store_true", help="Validate profile")
    parser.add_argument("--section", type=str, help="Get section emotion (e.g. 'chorus')")
    
    args = parser.parse_args()
    
    try:
        profile = load_emotion_profile(
            base_path=args.base,
            style=args.style,
            project_path=args.project,
            song_path=args.song
        )
        
        if args.validate:
            valid, errors = validate_emotion_profile(profile)
            if valid:
                print("✅ Profile validation PASSED")
            else:
                print("❌ Profile validation FAILED:")
                for err in errors:
                    print(f"  - {err}")
                exit(1)
        
        if args.section:
            emotion = get_section_emotion(profile, args.section)
            print(f"Section '{args.section}':")
            print(f"  energy: {emotion['energy']:.2f}")
            print(f"  valence: {emotion['valence']:.2f}")
        else:
            import json
            print(json.dumps(profile, indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
