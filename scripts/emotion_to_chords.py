#!/usr/bin/env python3
"""
Emotion to Chord Progression Mapper
====================================

物語の感情からコード進行を生成するシステム

統合要素:
- utilities/progression_templates.yaml (既存の感情→コード進行)
- LAMDa統合データベース (180,000曲のコード理論)
- emotion_profile (valence/arousal)
"""

import sqlite3
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class EmotionContext:
    """感情コンテキスト"""

    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    intensity: float  # 0.0 to 1.0
    section: str  # verse, chorus, bridge
    genre: Optional[str] = None

    @property
    def mood_category(self) -> str:
        """Valence/Arousalから感情カテゴリを判定"""
        if self.valence >= 0.3 and self.arousal >= 0.5:
            return "energetic"
        elif self.valence >= 0.3 and self.arousal < 0.5:
            return "peaceful"
        elif self.valence < 0.3 and self.arousal >= 0.5:
            return "tense"
        else:
            return "soft_reflective"

    @property
    def key_mode(self) -> str:
        """Valenceから調性（major/minor）を判定"""
        return "major" if self.valence >= 0 else "minor"


@dataclass
class ChordProgression:
    """コード進行"""

    chords: List[str]
    source: str  # "template" or "lamda"
    quality_score: float  # 0.0 to 1.0
    emotional_match: float  # 0.0 to 1.0

    def to_absolute_chords(self, root_key: str = "C") -> List[str]:
        """ディグリー記法を絶対コードに変換"""
        # TODO: 実装
        return self.chords


class EmotionChordMapper:
    """
    感情からコード進行を生成

    動作フロー:
    1. EmotionContextから感情カテゴリ判定
    2. progression_templates.yamlから基本進行取得
    3. LAMDaデータベースで類似進行検索
    4. 最適な進行を選択（または混合）
    """

    def __init__(self, templates_path: Path, lamda_db_path: Optional[Path] = None):
        self.templates_path = templates_path
        self.lamda_db_path = lamda_db_path

        # テンプレートロード
        with open(templates_path) as f:
            self.templates = yaml.safe_load(f)

        # LAMDa DB接続
        self.lamda_conn = None
        if lamda_db_path and lamda_db_path.exists():
            self.lamda_conn = sqlite3.connect(str(lamda_db_path))
            print(f"✅ LAMDa database connected: {lamda_db_path}")
        else:
            print("⚠️  LAMDa database not found, using templates only")

    def generate_progression(
        self, emotion: EmotionContext, num_alternatives: int = 5
    ) -> List[ChordProgression]:
        """
        感情からコード進行生成

        Args:
            emotion: 感情コンテキスト
            num_alternatives: 候補数

        Returns:
            コード進行候補リスト（quality_scoreでソート済み）
        """
        candidates = []

        # 1. テンプレートから取得
        template_progs = self._get_template_progressions(emotion)
        candidates.extend(template_progs)

        # 2. LAMDaから類似進行検索
        if self.lamda_conn:
            lamda_progs = self._search_lamda_progressions(emotion, base_progressions=template_progs)
            candidates.extend(lamda_progs)

        # 3. 品質スコアでソート
        candidates.sort(key=lambda x: x.quality_score, reverse=True)

        return candidates[:num_alternatives]

    def _get_template_progressions(self, emotion: EmotionContext) -> List[ChordProgression]:
        """progression_templates.yamlから進行取得"""
        mood = emotion.mood_category
        mode = emotion.key_mode

        # テンプレート取得
        templates = self.templates.get(mood, {}).get(mode, [])
        if not templates:
            # フォールバック
            templates = self.templates.get("_default", {}).get(mode, [])

        progressions = []
        for template in templates:
            # ディグリー記法のまま保持
            chords = template.split()

            # 感情マッチスコア計算
            emotional_match = self._calculate_emotional_match(chords, emotion)

            progressions.append(
                ChordProgression(
                    chords=chords,
                    source="template",
                    quality_score=emotional_match * 0.7,  # テンプレートは70%基準
                    emotional_match=emotional_match,
                )
            )

        return progressions

    def _search_lamda_progressions(
        self,
        emotion: EmotionContext,
        base_progressions: List[ChordProgression],
        max_results: int = 10,
    ) -> List[ChordProgression]:
        """LAMDaデータベースから類似進行検索"""
        if not self.lamda_conn:
            return []

        cursor = self.lamda_conn.cursor()
        progressions = []

        # ベース進行から検索クエリ構築
        for base_prog in base_progressions[:3]:  # 上位3つ
            # TODO: コード進行の類似度検索
            # 現在は単純にイベント数でフィルタリング

            min_events = int(500 * emotion.intensity)
            max_events = int(2000 * emotion.intensity)

            cursor.execute(
                """
                SELECT hash_id, progression, total_events, chord_events
                FROM progressions
                WHERE total_events BETWEEN ? AND ?
                ORDER BY RANDOM()
                LIMIT ?
            """,
                (min_events, max_events, max_results),
            )

            for row in cursor.fetchall():
                hash_id, prog_json, total_events, chord_events = row

                # JSON解析
                import json

                try:
                    prog_data = json.loads(prog_json)
                    if not prog_data:
                        continue

                    # 最初の数コードを抽出
                    chords = []
                    for chord_obj in prog_data[:8]:
                        # LAMDa構造: {'chord': 'E-minor seventh chord', 'root': 'E', ...}
                        chord_name = chord_obj.get("chord", "Unknown")
                        root = chord_obj.get("root", "")
                        # シンプルな表記に変換
                        if "major" in chord_name.lower() and "minor" not in chord_name.lower():
                            simple = f"{root}maj"
                        elif "minor" in chord_name.lower():
                            simple = f"{root}m"
                        else:
                            simple = f"{root}"
                        
                        if "seventh" in chord_name.lower():
                            simple += "7"
                        
                        chords.append(simple if simple else chord_name)

                    # 品質スコア計算
                    quality_score = self._calculate_quality_score(
                        total_events, chord_events, emotion
                    )

                    emotional_match = self._calculate_emotional_match(chords, emotion)

                    progressions.append(
                        ChordProgression(
                            chords=chords,
                            source=f"lamda:{hash_id[:8]}",
                            quality_score=quality_score,
                            emotional_match=emotional_match,
                        )
                    )

                except json.JSONDecodeError:
                    continue

        return progressions

    def _calculate_emotional_match(self, chords: List[str], emotion: EmotionContext) -> float:
        """感情マッチスコア計算"""
        score = 0.5  # ベーススコア

        # Valenceとコード性質のマッチ
        major_count = sum(1 for c in chords if any(m in c.lower() for m in ["maj", "I", "IV", "V"]))
        minor_count = sum(1 for c in chords if any(m in c.lower() for m in ["min", "i", "vi"]))

        if emotion.valence >= 0 and major_count > minor_count:
            score += 0.3
        elif emotion.valence < 0 and minor_count > major_count:
            score += 0.3

        # Arousalとコード複雑度のマッチ
        complex_count = sum(
            1 for c in chords if any(m in c.lower() for m in ["7", "9", "sus", "dim", "aug"])
        )

        if emotion.arousal >= 0.5 and complex_count >= 2:
            score += 0.2
        elif emotion.arousal < 0.5 and complex_count < 2:
            score += 0.2

        return min(score, 1.0)

    def _calculate_quality_score(
        self, total_events: int, chord_events: int, emotion: EmotionContext
    ) -> float:
        """LAMDa進行の品質スコア計算"""
        base_score = 0.8  # LAMDaは80%基準

        # イベント数が適切か
        ideal_events = 1000 * emotion.intensity
        event_diff = abs(total_events - ideal_events)
        event_penalty = min(event_diff / ideal_events, 0.3)

        # コードイベントの割合
        chord_ratio = chord_events / total_events if total_events > 0 else 0
        if 0.3 <= chord_ratio <= 0.7:
            chord_bonus = 0.1
        else:
            chord_bonus = 0.0

        return base_score - event_penalty + chord_bonus

    def close(self):
        """リソース解放"""
        if self.lamda_conn:
            self.lamda_conn.close()


# ============================================================================
# コマンドラインインターフェース
# ============================================================================


def main():
    """デモ実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate chord progressions from emotions")
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("utilities/progression_templates.yaml"),
        help="Progression templates YAML",
    )
    parser.add_argument(
        "--lamda-db", type=Path, default=Path("data/test_lamda.db"), help="LAMDa unified database"
    )
    parser.add_argument(
        "--valence", type=float, default=0.5, help="Valence: -1.0 (negative) to 1.0 (positive)"
    )
    parser.add_argument(
        "--arousal", type=float, default=0.7, help="Arousal: 0.0 (calm) to 1.0 (excited)"
    )
    parser.add_argument("--intensity", type=float, default=0.6, help="Intensity: 0.0 to 1.0")
    parser.add_argument(
        "--section",
        type=str,
        default="chorus",
        choices=["verse", "chorus", "bridge"],
        help="Song section",
    )

    args = parser.parse_args()

    # Mapper初期化
    mapper = EmotionChordMapper(
        templates_path=args.templates,
        lamda_db_path=args.lamda_db if args.lamda_db.exists() else None,
    )

    # 感情コンテキスト作成
    emotion = EmotionContext(
        valence=args.valence, arousal=args.arousal, intensity=args.intensity, section=args.section
    )

    print("=" * 70)
    print("🎼 Emotion to Chord Progression Generator")
    print("=" * 70)
    print(f"\n📊 Emotion Context:")
    print(f"  Valence: {emotion.valence:+.2f} ({emotion.key_mode})")
    print(f"  Arousal: {emotion.arousal:.2f}")
    print(f"  Intensity: {emotion.intensity:.2f}")
    print(f"  Mood Category: {emotion.mood_category}")
    print(f"  Section: {emotion.section}")

    # コード進行生成
    print(f"\n🎵 Generating chord progressions...\n")
    progressions = mapper.generate_progression(emotion, num_alternatives=5)

    print(f"Found {len(progressions)} candidates:\n")
    for i, prog in enumerate(progressions, 1):
        print(f"{i}. {'→'.join(prog.chords)}")
        print(f"   Source: {prog.source}")
        print(f"   Quality: {prog.quality_score:.2f}")
        print(f"   Emotional Match: {prog.emotional_match:.2f}")
        print()

    mapper.close()


if __name__ == "__main__":
    main()
