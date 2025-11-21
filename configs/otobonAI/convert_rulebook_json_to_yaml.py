#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rulebook.json → rulebook.yaml 変換スクリプト

- 入力:  configs/otobonAI/rulebook.json
- 出力:  configs/otobonAI/rulebook.yaml
"""

import json
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML が未インストールです: pip install pyyaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = REPO_ROOT / "configs" / "otobonAI" / "rulebook.json"
YAML_PATH = REPO_ROOT / "configs" / "otobonAI" / "rulebook.yaml"


def main() -> None:
    if not JSON_PATH.exists():
        raise SystemExit(f"❌ JSON が見つかりません: {JSON_PATH}")

    print(f"📥 Loading JSON: {JSON_PATH}")
    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # そのまま YAML 化（キー順は JSON の順序を維持）
    print(f"📤 Writing YAML: {YAML_PATH}")
    with YAML_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,   # 日本語をそのまま
            sort_keys=False,      # キー順を変えない
            width=120,            # 折り返し幅
            default_flow_style=False,
        )

    print("✅ Done. Created:", YAML_PATH)


if __name__ == "__main__":
    main()