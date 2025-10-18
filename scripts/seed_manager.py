#!/usr/bin/env python3
"""
Seed管理ユーティリティ - 全処理で乱数seedを一本化

使用方法:
    from scripts.seed_manager import SeedManager
    
    # 方法1: 環境変数から取得（優先度最高）
    sm = SeedManager()
    seed = sm.get_seed()  # COMPOSER_SEED=42 なら 42
    
    # 方法2: CLIから明示的に指定
    sm = SeedManager(cli_seed=12345)
    seed = sm.get_seed()  # 12345
    
    # 方法3: YAMLから読み込み
    sm = SeedManager()
    seed = sm.get_seed_from_yaml('project/song.yaml')
    
    # 適用
    sm.apply_global_seed(seed)  # numpy, random, PyTorchすべてに適用
"""

import os
import random
from typing import Optional
import yaml
from pathlib import Path


class SeedManager:
    """乱数seedの統一管理"""
    
    DEFAULT_SEED = 42
    ENV_VAR_NAME = "COMPOSER_SEED"
    
    def __init__(self, cli_seed: Optional[int] = None):
        """
        Args:
            cli_seed: CLI引数から渡されたseed（優先度: CLI > 環境変数 > YAML > デフォルト）
        """
        self.cli_seed = cli_seed
    
    def get_seed(self, yaml_path: Optional[Path] = None) -> int:
        """
        優先度に従ってseedを取得
        
        優先順位:
            1. CLI引数 (--seed)
            2. 環境変数 (COMPOSER_SEED)
            3. YAML設定 (meta.seed)
            4. デフォルト値 (42)
        
        Args:
            yaml_path: 構造YAMLファイルパス（オプション）
        
        Returns:
            決定されたseed値
        """
        # 1. CLI引数（最優先）
        if self.cli_seed is not None:
            return self.cli_seed
        
        # 2. 環境変数
        env_seed = os.environ.get(self.ENV_VAR_NAME)
        if env_seed is not None:
            try:
                return int(env_seed)
            except ValueError:
                print(f"⚠️  Invalid {self.ENV_VAR_NAME}={env_seed}, using default")
        
        # 3. YAMLファイル
        if yaml_path:
            yaml_seed = self.get_seed_from_yaml(yaml_path)
            if yaml_seed is not None:
                return yaml_seed
        
        # 4. デフォルト
        return self.DEFAULT_SEED
    
    def get_seed_from_yaml(self, yaml_path: Path) -> Optional[int]:
        """YAMLファイルからseedを読み込み"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # meta.seed を探す
            if isinstance(data, dict):
                meta = data.get('meta', {})
                if isinstance(meta, dict) and 'seed' in meta:
                    return int(meta['seed'])
        except Exception as e:
            print(f"⚠️  Failed to read seed from {yaml_path}: {e}")
        
        return None
    
    def apply_global_seed(self, seed: int) -> None:
        """
        全ライブラリに乱数seedを適用
        
        Args:
            seed: 適用するseed値
        """
        # Python標準
        random.seed(seed)
        
        # NumPy
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        
        # PyTorch（もし使う場合）
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        print(f"🌱 Global seed set: {seed}")
    
    def get_source_info(self, yaml_path: Optional[Path] = None) -> str:
        """seedの取得元を説明文で返す"""
        if self.cli_seed is not None:
            return f"CLI argument (--seed {self.cli_seed})"
        
        env_seed = os.environ.get(self.ENV_VAR_NAME)
        if env_seed is not None:
            return f"Environment variable ({self.ENV_VAR_NAME}={env_seed})"
        
        if yaml_path:
            yaml_seed = self.get_seed_from_yaml(yaml_path)
            if yaml_seed is not None:
                return f"YAML file ({yaml_path}: meta.seed={yaml_seed})"
        
        return f"Default value ({self.DEFAULT_SEED})"


def add_seed_argument(parser):
    """argparseにseed引数を追加するヘルパー"""
    parser.add_argument(
        '--seed', type=int, default=None,
        help=(
            f'Random seed for reproducibility. '
            f'Priority: CLI > ${SeedManager.ENV_VAR_NAME} > YAML > default({SeedManager.DEFAULT_SEED})'
        )
    )


# CLI使用例
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Seed Manager Demo')
    add_seed_argument(parser)
    parser.add_argument('--yaml', type=Path, help='Structure YAML file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Seed Manager Demo")
    print("=" * 60)
    
    sm = SeedManager(cli_seed=args.seed)
    seed = sm.get_seed(yaml_path=args.yaml)
    
    print(f"\n📍 Seed source: {sm.get_source_info(args.yaml)}")
    print(f"🎲 Selected seed: {seed}")
    
    sm.apply_global_seed(seed)
    
    # デモ: 乱数生成
    print(f"\n🧪 Random test:")
    print(f"   Python random: {random.random():.6f}")
    
    try:
        import numpy as np
        print(f"   NumPy random:  {np.random.random():.6f}")
    except ImportError:
        print("   NumPy: not installed")
    
    print("\n✅ Seed applied successfully!")
    print("\nUsage:")
    print(f"  1. CLI:        python script.py --seed 12345")
    print(f"  2. Env var:    export {SeedManager.ENV_VAR_NAME}=12345 && python script.py")
    print(f"  3. YAML:       (set meta.seed in structure.yaml)")
    print(f"  4. Default:    (uses {SeedManager.DEFAULT_SEED})")
