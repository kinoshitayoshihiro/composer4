#!/usr/bin/env python3
"""
旧名エントリーポイント（薄いラッパ）

既存のスクリプトが `python scripts/lamda_stage2_extractor.py ...` を
呼び出している場合でも、このラッパ経由で新実装へ透過的に流せます。

旧実装（5974行）は scripts/lamda_stage2_extractor_legacy.py として保存されています。
"""
import sys
from scripts.lamda_v2.compat.lamda_stage2_extractor_shim import main

if __name__ == "__main__":
    sys.exit(main())
