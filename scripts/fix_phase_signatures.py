#!/usr/bin/env python3
"""
Phase 14-18のシグネチャにseed引数を追加

現状:
    def _phase_14_harmonic_awareness(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any]
    ) -> None:

修正後:
    def _phase_14_harmonic_awareness(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
"""

import re
from pathlib import Path

project_root = Path(__file__).parent.parent
files = [
    "generator/bass_params_stage2.py",
    "generator/piano_params_stage2.py",
    "generator/guitar_params_stage2.py",
    "generator/strings_params_stage2.py"
]

pattern = re.compile(
    r'(def _phase_1[4-8]_\w+\(\s*'
    r'self,\s*'
    r'part: Any,\s*'
    r'section_meta: Dict\[str, Any\],\s*'
    r'mix_context: Dict\[str, Any\],\s*'
    r'params: Dict\[str, Any\])\s*'
    r'(\) -> None:)',
    re.MULTILINE
)

for file_path in files:
    full_path = project_root / file_path
    if not full_path.exists():
        print(f"⚠️  Skipped: {file_path} (not found)")
        continue
    
    content = full_path.read_text()
    original = content
    
    # seed引数を追加
    content = pattern.sub(r'\1,\n        seed: Optional[int]\2', content)
    
    if content != original:
        full_path.write_text(content)
        matches = pattern.findall(original)
        print(f"✅ Fixed {file_path}: {len(matches)} methods updated")
    else:
        print(f"ℹ️  No changes: {file_path}")

print("\n🎉 All signature fixes complete!")
