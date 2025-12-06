#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
section_normalizer.py

5楽器統合用セクション名正規化ユーティリティ

使い方:
    from section_normalizer import SectionNormalizer

    normalizer = SectionNormalizer(policy_yaml_path)
    section_id = normalizer.normalize("verse_a")  # → "VERSE"
    section_id = normalizer.normalize_with_priority(
        bar_index, section_candidates
    )  # → priority順で最適なセクションを選択
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml


class SectionNormalizer:
    """
    セクション名正規化クラス

    YAML policy の section_normalization.mapping/priority に基づき、
    生のセクション名を正規化済みセクションIDに変換します。

    priority順位制御により、複数のセクション候補がある場合に
    最優先セクションを選択します。
    """

    def __init__(self, policy_yaml_path: Optional[str | Path] = None):
        """
        Args:
            policy_yaml_path: Dynamics Policy YAML path
                              None の場合はデフォルトマッピングを使用
        """
        self.mapping: Dict[str, str] = {}
        self.priority: List[str] = []

        if policy_yaml_path:
            policy = self._load_yaml(policy_yaml_path)
            section_norm = policy.get("section_normalization", {}) or {}
            self.mapping = section_norm.get("mapping", {})
            self.priority = section_norm.get("priority", [])

        # デフォルトマッピング（YAML未指定時）
        if not self.mapping:
            self.mapping = {
                "intro": "INTRO",
                "verse": "VERSE",
                "verse_a": "VERSE",
                "verse_b": "VERSE",
                "prechorus": "PRE_CHORUS",
                "pre_chorus": "PRE_CHORUS",
                "chorus": "CHORUS",
                "chorus_a": "CHORUS",
                "chorus_b": "CHORUS",
                "bridge": "BRIDGE",
                "solo": "SOLO",
                "interlude": "INTERLUDE",
                "breakdown": "BREAKDOWN",
                "climax": "CLIMAX",
                "outro": "OUTRO",
                "unknown": "UNKNOWN",
            }

        # デフォルト優先順位（YAML未指定時）
        if not self.priority:
            self.priority = [
                "CLIMAX",
                "CHORUS",
                "PRE_CHORUS",
                "SOLO",
                "BRIDGE",
                "VERSE",
                "BREAKDOWN",
                "INTERLUDE",
                "INTRO",
                "OUTRO",
                "UNKNOWN",
            ]

    def _load_yaml(self, path: str | Path) -> Dict:
        """YAML読込"""
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def normalize(self, raw_section: Optional[str]) -> str:
        """
        生のセクション名を正規化

        Args:
            raw_section: 生のセクション名（例: "verse_a", "PreChorus"）

        Returns:
            正規化済みセクションID（例: "VERSE", "PRE_CHORUS"）
        """
        if not raw_section:
            return "UNKNOWN"

        # 小文字変換してマッピング検索
        key = str(raw_section).strip().lower()
        normalized = self.mapping.get(key, None)

        if normalized:
            return normalized.upper()

        # マッピングにない場合は大文字化して返す
        return str(raw_section).strip().upper()

    def normalize_with_priority(
        self,
        bar_index: int,
        section_candidates: List[str],
    ) -> str:
        """
        優先順位に基づいて最適なセクションを選択

        複数のセクション候補がある場合、priority順で最優先のものを返します。

        Args:
            bar_index: 小節インデックス（デバッグ用）
            section_candidates: セクション候補リスト（生の名前）

        Returns:
            最優先の正規化済みセクションID

        Examples:
            >>> normalizer.normalize_with_priority(43, ["verse", "climax"])
            "CLIMAX"  # priority: CLIMAX > VERSE
        """
        if not section_candidates:
            return "UNKNOWN"

        # 全候補を正規化
        normalized_candidates = [self.normalize(sec) for sec in section_candidates]

        # priority順で最初に見つかったものを返す
        for priority_sec in self.priority:
            if priority_sec in normalized_candidates:
                return priority_sec

        # priorityにない場合は最初の候補を返す
        return normalized_candidates[0]

    def get_priority_index(self, section_id: str) -> int:
        """
        セクションの優先順位インデックスを取得

        Args:
            section_id: 正規化済みセクションID

        Returns:
            優先順位インデックス（0が最優先、見つからない場合は999）
        """
        try:
            return self.priority.index(section_id.upper())
        except (ValueError, AttributeError):
            return 999

    def is_higher_priority(self, section_a: str, section_b: str) -> bool:
        """
        section_a が section_b より優先度が高いか判定

        Args:
            section_a: セクションA（正規化済み）
            section_b: セクションB（正規化済み）

        Returns:
            True if section_a > section_b in priority
        """
        return self.get_priority_index(section_a) < self.get_priority_index(section_b)


# グローバルインスタンス（共通利用）
_global_normalizer: Optional[SectionNormalizer] = None


def get_global_normalizer(policy_yaml_path: Optional[str | Path] = None) -> SectionNormalizer:
    """
    グローバル SectionNormalizer インスタンスを取得

    Args:
        policy_yaml_path: 初回呼び出し時に指定するYAMLパス

    Returns:
        共通のSectionNormalizerインスタンス
    """
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = SectionNormalizer(policy_yaml_path)
    return _global_normalizer


def normalize_section(
    raw_section: Optional[str], policy_yaml_path: Optional[str | Path] = None
) -> str:
    """
    セクション名を正規化（グローバルインスタンス利用）

    Args:
        raw_section: 生のセクション名
        policy_yaml_path: 初回呼び出し時に指定するYAMLパス

    Returns:
        正規化済みセクションID
    """
    normalizer = get_global_normalizer(policy_yaml_path)
    return normalizer.normalize(raw_section)
