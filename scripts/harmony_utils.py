#!/usr/bin/env python3
"""
harmony_utils.py - harmony_beat.json専用ユーティリティ

既存のchordmap_utilsと同じインターフェースを提供しつつ、
感情メタデータ・XMusic統合を活かす設計。

Usage:
    from harmony_utils import load_harmony_beat, get_harmony_chord_at_bar

    harmony = load_harmony_beat(Path("harmony_beat.json"))
    chord_event = get_harmony_chord_at_bar(harmony, bar_idx=5)
    if chord_event:
        print(f"Bar 5: {chord_event.symbol} ({chord_event.emotion})")
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class HarmonyChordEvent:
    """
    harmony_beat.jsonのコードイベント (感情統合版)

    Attributes:
        bar: 小節番号 (0-indexed)
        beat: 小節内の拍位置 (1=頭拍, 3=3拍目など)
        symbol: コードシンボル ("Em7", "Cmaj9", "F#m7b5" etc.)
        duration_beats: コードの持続時間 (beat単位)

        # Function分析 (音楽理論)
        function: コード機能 ("tonic", "subdominant", "dominant" etc.)
        degree: ディグリー ("i", "V", "IVmaj7" etc.)

        # 感情レイヤー (3系統統合: emotion_chord_map + XMusic + EmotionAI)
        emotion: emotion_chord_mapの感情 ("melancholic", "energetic" etc.)
        xmusic_emotion: XMusic 8感情カテゴリ ("joy", "pride", "despair" etc.)
        valence: 感情価 (0.0-1.0, 0=ネガティブ, 1=ポジティブ)
        arousal: 覚醒度 (0.0-1.0, 0=穏やか, 1=興奮)
        tension: 緊張度 (0.0-1.0, 0=安定, 1=緊張)
        energy: エネルギー (0.0-1.0)
        brightness: 明るさ (0.0-1.0)

        # フレーズ構造 (lyric_harmony連携)
        phrase_id: フレーズID ("A1", "B1", "C1" etc.)
        phrase_role: フレーズ役割 ("opening", "climax", "resolution" etc.)
        section: セクション ("intro", "verse", "chorus" etc.)
    """

    bar: int
    beat: float
    symbol: str
    duration_beats: float

    # Function分析
    function: Optional[str] = None
    degree: Optional[str] = None

    # 感情レイヤー (3系統統合)
    emotion: Optional[str] = None
    xmusic_emotion: Optional[str] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    tension: Optional[float] = None
    energy: Optional[float] = None
    brightness: Optional[float] = None

    # フレーズ構造
    phrase_id: Optional[str] = None
    phrase_role: Optional[str] = None
    section: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換 (既存のchord辞書と互換性)"""
        return {
            "bar": self.bar,
            "beat": self.beat,
            "symbol": self.symbol,
            "duration_beats": self.duration_beats,
            "function": self.function,
            "degree": self.degree,
            "emotion": self.emotion,
            "xmusic_emotion": self.xmusic_emotion,
            "valence": self.valence,
            "arousal": self.arousal,
            "tension": self.tension,
            "energy": self.energy,
            "brightness": self.brightness,
            "phrase_id": self.phrase_id,
            "phrase_role": self.phrase_role,
            "section": self.section,
        }


def load_harmony_beat(path: Path) -> Dict[str, Any]:
    """
    harmony_beat.json読み込み + 検証

    Args:
        path: harmony_beat.jsonのパス

    Returns:
        Dict with keys: "version", "unit", "meta", "chords"

    Raises:
        ValueError: unitが"beat"でない場合
        FileNotFoundError: ファイルが存在しない場合
    """
    if not path.exists():
        raise FileNotFoundError(f"harmony_beat.json not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    # unit検証
    if data.get("unit") != "beat":
        raise ValueError(
            f"harmony_beat.json expected unit='beat', got '{data.get('unit')}'\n" f"File: {path}"
        )

    # chords配列をbar/beat順にソート
    chords = data.get("chords") or []
    chords = sorted(chords, key=lambda ev: (ev.get("bar", 0), ev.get("beat", 1)))
    data["chords"] = chords

    return data


def get_harmony_chord_at_bar(harmony: Dict[str, Any], bar_idx: int) -> Optional[HarmonyChordEvent]:
    """
    bar_idxに有効なコードイベントを返す。

    duration_beatsから小節スパンを計算:
    - 4.0 beats = 1 bar
    - 8.0 beats = 2 bars (サビの長いコード等)
    - 32.0 beats = 8 bars (イントロの長い一発コード等)

    Args:
        harmony: load_harmony_beat()で読み込んだデータ
        bar_idx: 小節番号 (0-indexed)

    Returns:
        HarmonyChordEvent or None (該当コードがない場合)

    Example:
        >>> harmony = load_harmony_beat(Path("harmony_beat.json"))
        >>> chord = get_harmony_chord_at_bar(harmony, bar_idx=10)
        >>> if chord:
        ...     print(f"Bar 10: {chord.symbol} - {chord.emotion}")
    """
    events: List[Dict[str, Any]] = harmony.get("chords", [])

    for ev in events:
        start_bar = int(ev.get("bar", 0))
        dur_beats = float(ev.get("duration_beats", 4.0))

        # duration_beats → 小節数に変換 (4 beats = 1 bar)
        # 最低1小節は保証
        dur_bars = max(1, int(round(dur_beats / 4.0)))

        # bar_idxがこのコードのスパン内にあるか
        if start_bar <= bar_idx < start_bar + dur_bars:
            return HarmonyChordEvent(
                bar=start_bar,
                beat=float(ev.get("beat", 1)),
                symbol=ev["symbol"],
                duration_beats=dur_beats,
                function=ev.get("function"),
                degree=ev.get("degree"),
                emotion=ev.get("emotion"),
                xmusic_emotion=ev.get("xmusic_emotion"),
                valence=ev.get("valence"),
                arousal=ev.get("arousal"),
                tension=ev.get("tension"),
                energy=ev.get("energy"),
                brightness=ev.get("brightness"),
                phrase_id=ev.get("phrase_id"),
                phrase_role=ev.get("phrase_role"),
                section=ev.get("section"),
            )

    return None


def get_harmony_chord_at_position(
    harmony: Dict[str, Any], bar_idx: int, beat: float
) -> Optional[HarmonyChordEvent]:
    """
    (bar, beat)位置に有効なコードイベントを返す。

    将来的に小節内コードチェンジ (beat=3での転換等) に対応。

    Args:
        harmony: load_harmony_beat()で読み込んだデータ
        bar_idx: 小節番号 (0-indexed)
        beat: 小節内の拍位置 (1.0-5.0)

    Returns:
        HarmonyChordEvent or None

    Example:
        >>> # Bar 10の3拍目のコード
        >>> chord = get_harmony_chord_at_position(harmony, bar_idx=10, beat=3.0)
    """
    events: List[Dict[str, Any]] = harmony.get("chords", [])

    # 同じbar内で beat <= target_beat の最も近いイベントを探す
    candidates = []
    for ev in events:
        ev_bar = int(ev.get("bar", 0))
        ev_beat = float(ev.get("beat", 1))

        # 同じ小節で、指定beatより前のコード
        if ev_bar == bar_idx and ev_beat <= beat:
            candidates.append((ev_beat, ev))

    if candidates:
        # 最も近い（最大の）beatを選択
        candidates.sort(reverse=True)
        _, best_ev = candidates[0]

        return HarmonyChordEvent(
            bar=int(best_ev.get("bar", 0)),
            beat=float(best_ev.get("beat", 1)),
            symbol=best_ev["symbol"],
            duration_beats=float(best_ev.get("duration_beats", 4.0)),
            function=best_ev.get("function"),
            degree=best_ev.get("degree"),
            emotion=best_ev.get("emotion"),
            xmusic_emotion=best_ev.get("xmusic_emotion"),
            valence=best_ev.get("valence"),
            arousal=best_ev.get("arousal"),
            tension=best_ev.get("tension"),
            energy=best_ev.get("energy"),
            brightness=best_ev.get("brightness"),
            phrase_id=best_ev.get("phrase_id"),
            phrase_role=best_ev.get("phrase_role"),
            section=best_ev.get("section"),
        )

    # 同じ小節にない → 前の小節から継続しているコードを探す
    return get_harmony_chord_at_bar(harmony, bar_idx)


def parse_harmony_symbol(symbol: str) -> Dict[str, Any]:
    """
    harmony_beat.jsonのコードシンボルをパース。

    既存のchordmap_utils.parse_symbol()と互換性を保つラッパー。

    Args:
        symbol: コードシンボル ("Em7", "Cmaj9", "F#m7b5" etc.)

    Returns:
        Dict with keys: "root", "quality", "bass" (optional)

    Example:
        >>> parsed = parse_harmony_symbol("Em7")
        >>> print(parsed["root"])  # "E"
        >>> print(parsed["quality"])  # "m7"
    """
    from chordmap_utils import parse_symbol

    return parse_symbol(symbol)


def get_harmony_chord_tones(harmony_event: HarmonyChordEvent, octave: int = 4) -> List[int]:
    """
    HarmonyChordEventから構成音のMIDIノート番号を取得。

    既存のchordmap_utils.get_chord_tones()のラッパー。

    Args:
        harmony_event: コードイベント
        octave: 基準オクターブ (4 = C4)

    Returns:
        MIDIノート番号のリスト (例: [60, 64, 67] = C-E-G)

    Example:
        >>> chord = get_harmony_chord_at_bar(harmony, 10)
        >>> notes = get_harmony_chord_tones(chord, octave=3)
        >>> print(notes)  # [48, 52, 55] = C3-E3-G3
    """
    from chordmap_utils import parse_symbol, get_chord_tones

    parsed = parse_symbol(harmony_event.symbol)
    return get_chord_tones(parsed, bass_octave=octave)


# ============================================================================
# 統計・分析ユーティリティ
# ============================================================================


def get_harmony_stats(harmony: Dict[str, Any]) -> Dict[str, Any]:
    """
    harmony_beat.jsonの統計情報を取得。

    Returns:
        Dict with keys:
            - num_events: コードイベント数
            - avg_duration_beats: 平均持続時間
            - emotions: 感情別のイベント数
            - functions: 機能別のイベント数
    """
    chords = harmony.get("chords", [])

    if not chords:
        return {
            "num_events": 0,
            "avg_duration_beats": 0.0,
            "emotions": {},
            "functions": {},
        }

    durations = [ev.get("duration_beats", 4.0) for ev in chords]
    emotions = {}
    functions = {}

    for ev in chords:
        emotion = ev.get("xmusic_emotion") or ev.get("emotion") or "unknown"
        emotions[emotion] = emotions.get(emotion, 0) + 1

        function = ev.get("function") or "unknown"
        functions[function] = functions.get(function, 0) + 1

    return {
        "num_events": len(chords),
        "avg_duration_beats": sum(durations) / len(durations),
        "min_duration_beats": min(durations),
        "max_duration_beats": max(durations),
        "emotions": emotions,
        "functions": functions,
    }


if __name__ == "__main__":
    # 簡易テスト
    import sys

    if len(sys.argv) < 2:
        print("Usage: python harmony_utils.py <harmony_beat.json>")
        sys.exit(1)

    harmony_path = Path(sys.argv[1])

    try:
        harmony = load_harmony_beat(harmony_path)
        stats = get_harmony_stats(harmony)

        print(f"✅ Loaded: {harmony_path}")
        print(f"   Events: {stats['num_events']}")
        print(f"   Avg duration: {stats['avg_duration_beats']:.1f} beats")
        print(f"   Emotions: {stats['emotions']}")
        print(f"   Functions: {stats['functions']}")

        # 最初の3小節のコードを表示
        print("\n📊 First 3 bars:")
        for bar_idx in range(3):
            chord = get_harmony_chord_at_bar(harmony, bar_idx)
            if chord:
                print(f"   Bar {bar_idx}: {chord.symbol} ({chord.emotion} / {chord.function})")

    except Exception as exc:
        print(f"❌ Error: {exc}")
        sys.exit(1)
