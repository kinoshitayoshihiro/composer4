#!/usr/bin/env python3
"""
Vocal Analyzer - Suno vocal stems分析

機能:
- テンポ検出 (BPM extraction)
- セクション検出 (Verse/Chorus/Bridge)
- エネルギーカーブ抽出 (Intensity mapping)
- Onset/Rest検出 (既存vocal_sync_fixed統合)

Usage:
    python scripts/vocal_analyzer.py <vocal_wav_path>
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import librosa
    import librosa.display
except ImportError:
    print("⚠️ librosa not installed. Run: pip install librosa")
    librosa = None

try:
    from utilities.vocal_sync_fixed import extract_onsets, find_long_rests
except ImportError:
    print("⚠️ vocal_sync_fixed not found")
    extract_onsets = None
    find_long_rests = None


@dataclass
class VocalSection:
    """ボーカルセクション情報"""

    start_time: float  # 開始時刻 (seconds)
    end_time: float  # 終了時刻 (seconds)
    section_type: str  # verse, chorus, bridge, intro, outro
    energy_level: float  # 0.0-1.0
    peak_frequency: Optional[float] = None  # ピーク周波数 (Hz)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VocalAnalysis:
    """ボーカル分析結果"""

    wav_path: Path
    duration: float  # 総時間 (seconds)
    tempo: float  # BPM
    tempo_confidence: float  # 0.0-1.0

    # セクション情報
    sections: List[VocalSection] = field(default_factory=list)

    # エネルギー情報
    energy_curve: List[Tuple[float, float]] = field(default_factory=list)  # (time, energy)
    avg_energy: float = 0.0
    max_energy: float = 0.0

    # Onset/Rest情報 (vocal_sync_fixed統合)
    onsets: List[float] = field(default_factory=list)  # onset時刻 (seconds)
    long_rests: List[Tuple[float, float]] = field(default_factory=list)  # (start, end)

    def to_dict(self) -> dict:
        """JSON出力用"""
        return {
            "wav_path": str(self.wav_path),
            "duration": self.duration,
            "tempo": self.tempo,
            "tempo_confidence": self.tempo_confidence,
            "sections": [
                {
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration": s.duration,
                    "section_type": s.section_type,
                    "energy_level": s.energy_level,
                    "peak_frequency": s.peak_frequency,
                }
                for s in self.sections
            ],
            "energy_curve_samples": len(self.energy_curve),
            "avg_energy": self.avg_energy,
            "max_energy": self.max_energy,
            "onsets_count": len(self.onsets),
            "long_rests_count": len(self.long_rests),
        }


class VocalAnalyzer:
    """ボーカル分析エンジン"""

    def __init__(self):
        if not librosa:
            raise RuntimeError("librosa required. Install: pip install librosa")

    def analyze(self, wav_path: Path, verbose: bool = True) -> VocalAnalysis:
        """
        ボーカルWAV分析

        Args:
            wav_path: ボーカルWAVファイル
            verbose: 詳細出力

        Returns:
            VocalAnalysis
        """
        if not wav_path.exists():
            raise FileNotFoundError(f"Vocal WAV not found: {wav_path}")

        if verbose:
            print(f"🎤 Analyzing vocal: {wav_path.name}")

        # オーディオ読み込み
        y, sr = librosa.load(wav_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        if verbose:
            print(f"  ⏱️  Duration: {duration:.2f}s")
            print(f"  🔊 Sample rate: {sr} Hz")

        # テンポ検出
        tempo, tempo_confidence = self._extract_tempo(y, sr, verbose)

        # エネルギーカーブ
        energy_curve, avg_energy, max_energy = self._extract_energy_curve(y, sr, verbose)

        # セクション検出
        sections = self._detect_sections(y, sr, energy_curve, verbose)

        # Onset検出 (vocal_sync_fixed統合)
        onsets = self._extract_onsets(wav_path, verbose)

        # Rest検出
        long_rests = self._extract_long_rests(wav_path, verbose)

        analysis = VocalAnalysis(
            wav_path=wav_path,
            duration=duration,
            tempo=tempo,
            tempo_confidence=tempo_confidence,
            sections=sections,
            energy_curve=energy_curve,
            avg_energy=avg_energy,
            max_energy=max_energy,
            onsets=onsets,
            long_rests=long_rests,
        )

        if verbose:
            print(f"\n✅ Analysis complete!")
            print(f"   Tempo: {tempo:.1f} BPM (confidence: {tempo_confidence:.2f})")
            print(f"   Sections: {len(sections)}")
            print(f"   Avg Energy: {avg_energy:.3f}")
            print(f"   Onsets: {len(onsets)}")
            print(f"   Long Rests: {len(long_rests)}")

        return analysis

    def _extract_tempo(self, y: np.ndarray, sr: int, verbose: bool) -> Tuple[float, float]:
        """テンポ検出"""
        if verbose:
            print("  🎵 Extracting tempo...")

        # librosaのbeat tracking
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # Confidence計算 (beat強度の標準偏差)
        beat_strength = librosa.util.sync(onset_env, beats, aggregate=np.median)
        confidence = float(np.std(beat_strength) / (np.mean(beat_strength) + 1e-6))
        confidence = min(confidence, 1.0)

        return float(tempo), confidence

    def _extract_energy_curve(
        self, y: np.ndarray, sr: int, verbose: bool
    ) -> Tuple[List[Tuple[float, float]], float, float]:
        """エネルギーカーブ抽出"""
        if verbose:
            print("  📊 Extracting energy curve...")

        # RMS energy計算 (フレームサイズ: 0.1秒)
        frame_length = int(sr * 0.1)
        hop_length = frame_length // 2
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # 時間軸
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # (time, energy) ペア
        energy_curve = [(float(t), float(e)) for t, e in zip(times, rms)]

        avg_energy = float(np.mean(rms))
        max_energy = float(np.max(rms))

        return energy_curve, avg_energy, max_energy

    def _detect_sections(
        self,
        y: np.ndarray,
        sr: int,
        energy_curve: List[Tuple[float, float]],
        verbose: bool,
    ) -> List[VocalSection]:
        """セクション検出 (エネルギー変化ベース)"""
        if verbose:
            print("  🎯 Detecting sections...")

        # 簡易実装: エネルギー閾値でセクション分割
        sections = []
        energies = np.array([e for _, e in energy_curve])
        times = np.array([t for t, _ in energy_curve])

        # 高エネルギー = Chorus, 低エネルギー = Verse
        threshold = np.median(energies)

        in_section = False
        start_idx = 0
        current_type = "verse"

        for i, energy in enumerate(energies):
            is_high = energy > threshold

            if not in_section:
                if is_high or energy > 0.01:  # セクション開始
                    in_section = True
                    start_idx = i
                    current_type = "chorus" if is_high else "verse"
            else:
                # セクション変化検出 (エネルギーが大きく変化)
                if i > start_idx + 10:  # 最低1秒の長さ
                    if (is_high and current_type == "verse") or (
                        not is_high and current_type == "chorus"
                    ):
                        # 前のセクションを保存
                        sections.append(
                            VocalSection(
                                start_time=float(times[start_idx]),
                                end_time=float(times[i - 1]),
                                section_type=current_type,
                                energy_level=float(np.mean(energies[start_idx:i])),
                            )
                        )
                        # 新しいセクション開始
                        start_idx = i
                        current_type = "chorus" if is_high else "verse"

        # 最後のセクション
        if in_section:
            sections.append(
                VocalSection(
                    start_time=float(times[start_idx]),
                    end_time=float(times[-1]),
                    section_type=current_type,
                    energy_level=float(np.mean(energies[start_idx:])),
                )
            )

        return sections

    def _extract_onsets(self, wav_path: Path, verbose: bool) -> List[float]:
        """Onset抽出 (vocal_sync_fixed統合)"""
        if not extract_onsets:
            return []

        if verbose:
            print("  🎶 Extracting onsets (vocal_sync_fixed)...")

        try:
            onsets = extract_onsets(str(wav_path))
            return onsets
        except Exception as e:
            if verbose:
                print(f"    ⚠️ Onset extraction failed: {e}")
            return []

    def _extract_long_rests(self, wav_path: Path, verbose: bool) -> List[Tuple[float, float]]:
        """Long Rest抽出 (vocal_sync_fixed統合)"""
        if not find_long_rests:
            return []

        if verbose:
            print("  🔇 Detecting long rests (vocal_sync_fixed)...")

        try:
            rests = find_long_rests(str(wav_path), min_duration=0.5)
            return rests
        except Exception as e:
            if verbose:
                print(f"    ⚠️ Rest detection failed: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(description="Analyze Suno vocal stem")
    parser.add_argument("wav_path", type=Path, help="Path to vocal WAV file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/vocal_analysis.json"),
        help="Output JSON path",
    )

    args = parser.parse_args()

    # 分析実行
    analyzer = VocalAnalyzer()
    analysis = analyzer.analyze(args.wav_path, verbose=True)

    # JSON出力
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(analysis.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n📄 Analysis exported: {args.output}")
    print("=" * 70)
    print("🎉 Vocal analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
