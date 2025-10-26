"""
Re-ranking Configuration Loader

グリッドサーチで特定されたベストパラメータを自動ロードし、
ジェネレーター実行時に再ランクの既定値として使用します。

Usage:
    from utils.rerank_config import load_best
    
    rerank_params = load_best()  # ab_v3_best.yaml から読み込み
    # または
    rerank_params = load_best("path/to/custom.yaml")
"""

from pathlib import Path
import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# デフォルト設定（ab_v3_best.yaml が存在しない場合のフォールバック）
_DEFAULT = {
    "threshold": 0.25,
    "w_proba": 0.55,
    "w_accent": 0.30,
    "w_density": 0.10,
    "w_section": 0.05,
}


def load_best(path: str = "data/ab_v3_best.yaml") -> Dict[str, Any]:
    """
    ベスト再ランクパラメータを読み込み
    
    Args:
        path: YAMLファイルのパス（デフォルト: data/ab_v3_best.yaml）
    
    Returns:
        再ランクパラメータの辞書:
        {
            "threshold": float,   # ML confidence threshold
            "w_proba": float,     # 確率重み
            "w_accent": float,    # アクセント重み
            "w_density": float,   # 密度重み
            "w_section": float,   # セクション重み
        }
    """
    p = Path(path)
    
    # ファイルが存在しない場合はデフォルトを返す
    if not p.exists():
        logger.debug(f"Re-rank config not found at {path}, using defaults")
        return _DEFAULT.copy()
    
    try:
        with open(p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        if not config:
            logger.warning(f"Empty config at {path}, using defaults")
            return _DEFAULT.copy()
        
        # "selected" セクションから読み込み
        selected = config.get("selected", {})
        
        result = {
            "threshold": float(selected.get("threshold", _DEFAULT["threshold"])),
            "w_proba": float(selected.get("w_proba", _DEFAULT["w_proba"])),
            "w_accent": float(selected.get("w_accent", _DEFAULT["w_accent"])),
            "w_density": float(selected.get("w_density", _DEFAULT["w_density"])),
            "w_section": float(selected.get("w_section", _DEFAULT["w_section"])),
        }
        
        logger.info(f"Loaded re-rank config from {path}: threshold={result['threshold']:.2f}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to load re-rank config from {path}: {e}")
        logger.info("Falling back to default re-rank parameters")
        return _DEFAULT.copy()


def get_default() -> Dict[str, Any]:
    """
    デフォルト再ランクパラメータを取得
    
    Returns:
        デフォルトパラメータの辞書
    """
    return _DEFAULT.copy()
