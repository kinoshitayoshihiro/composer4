"""
Chord Fit Configuration Loader

セクション別のChord Fit重み係数をYAMLから読み込み。
v3.1のコードトーン命中率計算で使用。

Usage:
    from ml.chord_fit_config import get_chord_fit_weights
    
    weights = get_chord_fit_weights(section='Chorus')
    # → {'beat_strong': 0.40, 'beat_weak': 0.20, 'bass_root_bonus': 0.10, 'clash_penalty': -0.30}
"""

import yaml
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# デフォルト重み係数（gate_prod.yamlがない場合のフォールバック）
DEFAULT_WEIGHTS = {
    'beat_strong': 0.35,
    'beat_weak': 0.20,
    'bass_root_bonus': 0.10,
    'clash_penalty': -0.30
}

# グローバルキャッシュ（起動時1回読み込み）
_CHORD_FIT_CONFIG_CACHE = None


def load_chord_fit_config() -> Dict:
    """
    gate_prod.yamlからchord_fit設定を読み込み
    
    Returns:
        chord_fit設定辞書
    """
    global _CHORD_FIT_CONFIG_CACHE
    
    if _CHORD_FIT_CONFIG_CACHE is not None:
        return _CHORD_FIT_CONFIG_CACHE
    
    config_path = Path(__file__).parent.parent / "config" / "gate_prod.yaml"
    
    if not config_path.exists():
        logger.warning(f"gate_prod.yaml not found: {config_path}, using defaults")
        _CHORD_FIT_CONFIG_CACHE = {'weights_by_section': {}}
        return _CHORD_FIT_CONFIG_CACHE
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            gate_config = yaml.safe_load(f)
        
        chord_fit_config = gate_config.get('chord_fit', {})
        _CHORD_FIT_CONFIG_CACHE = chord_fit_config
        
        logger.info(f"Chord fit config loaded: {len(chord_fit_config.get('weights_by_section', {}))} sections")
        return chord_fit_config
    
    except Exception as e:
        logger.error(f"Failed to load chord_fit config: {e}")
        _CHORD_FIT_CONFIG_CACHE = {'weights_by_section': {}}
        return _CHORD_FIT_CONFIG_CACHE


def get_chord_fit_weights(section: str = None) -> Dict[str, float]:
    """
    指定セクションのChord Fit重み係数を取得
    
    Args:
        section: セクション名（Chorus, Verse, Bridge等）。Noneの場合はデフォルト。
    
    Returns:
        重み係数辞書
        {
            'beat_strong': float,      # 強拍のコードトーン一致度
            'beat_weak': float,        # 弱拍のコードトーン一致度
            'bass_root_bonus': float,  # ベース音がルート音に一致
            'clash_penalty': float     # 半音衝突ペナルティ（負の値）
        }
    """
    config = load_chord_fit_config()
    weights_by_section = config.get('weights_by_section', {})
    
    if section and section in weights_by_section:
        section_weights = weights_by_section[section]
        logger.debug(f"Using chord_fit weights for section={section}: {section_weights}")
        return section_weights
    
    # セクション指定なし、または未定義セクション → デフォルト
    logger.debug(f"Using default chord_fit weights (section={section} not found)")
    return DEFAULT_WEIGHTS


def reload_chord_fit_config():
    """
    Chord Fit設定を再読み込み（開発/テスト用）
    """
    global _CHORD_FIT_CONFIG_CACHE
    _CHORD_FIT_CONFIG_CACHE = None
    logger.info("Chord fit config cache cleared, will reload on next access")


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Chord Fit Config Test ===\n")
    
    for section in ['Chorus', 'Verse', 'Bridge', 'Intro', 'Outro', 'Pre-Chorus', 'Unknown']:
        weights = get_chord_fit_weights(section)
        print(f"{section:12s}: beat_strong={weights['beat_strong']:.2f}, "
              f"beat_weak={weights['beat_weak']:.2f}, "
              f"bass_root={weights['bass_root_bonus']:+.2f}, "
              f"clash={weights['clash_penalty']:+.2f}")
