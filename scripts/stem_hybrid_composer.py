#!/usr/bin/env python3
"""
Stem WAV + ChordMap ハイブリッド作曲システム

【設計方針】
1. Stem WAV から音響特徴抽出（Onset/Rhythm/Velocity/Pitch）
2. ChordMap/Sections から構造情報取得（Chord Progression/Tempo Map）
3. ハイブリッド統合 → Plan JSON生成
4. midi_writer.py 経由で高品質MIDI生成

【優位性】
- ✅ 実音響特徴の活用（815バイト問題解決）
- ✅ ChordMap併用で構造的一貫性
- ✅ 楽器別Stem分離済み（高精度）
- ✅ midi_writer統一設計（Tempo Track 0限定）

Usage:
    python scripts/stem_hybrid_composer.py \
        --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
        --analysis data/suno_ai/suno_themesong/song_001/analysis \
        --output output/hybrid_midi \
        --bars 32 \
        --emotion energetic
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

# プロジェクトルート追加
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロガー設定
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# ========================================
# 1. Stem WAV特徴抽出
# ========================================


class StemFeatureExtractor:
    """Stem WAVから音響特徴を抽出"""

    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length

    def extract_onsets(self, wav_path: Path) -> Dict[str, Any]:
        """Onset検出（時刻 + 強度）"""
        logger.info(f"📊 Onset抽出: {wav_path.name}")

        try:
            y, sr = librosa.load(wav_path, sr=self.sr)

            # Onset strength envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

            # Onset detection
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, units="frames"
            )

            # 時刻変換
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)

            # 強度抽出
            onset_strengths = onset_env[onset_frames]

            return {
                "times": onset_times.tolist(),
                "strengths": onset_strengths.tolist(),
                "count": len(onset_times),
            }

        except Exception as e:
            logger.warning(f"Onset抽出失敗: {e}")
            return {"times": [], "strengths": [], "count": 0}

    def extract_rhythm(self, wav_path: Path) -> Dict[str, Any]:
        """Rhythm特徴抽出（Tempo/Beat Grid）"""
        logger.info(f"🥁 Rhythm抽出: {wav_path.name}")

        try:
            y, sr = librosa.load(wav_path, sr=self.sr)

            # Tempo推定
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)

            # Beat時刻
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)

            return {
                "tempo": float(tempo),
                "beat_times": beat_times.tolist(),
                "beat_count": len(beat_times),
            }

        except Exception as e:
            logger.warning(f"Rhythm抽出失敗: {e}")
            return {"tempo": 120.0, "beat_times": [], "beat_count": 0}

    def extract_pitch_contour(self, wav_path: Path) -> Dict[str, Any]:
        """Pitch輪郭抽出（Melody/Bass用）"""
        logger.info(f"🎵 Pitch抽出: {wav_path.name}")

        try:
            y, sr = librosa.load(wav_path, sr=self.sr)

            # F0推定（pYIN）
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                hop_length=self.hop_length,
            )

            # 時刻変換
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=self.hop_length)

            # 有効なPitch抽出
            valid_indices = ~np.isnan(f0)
            valid_times = times[valid_indices]
            valid_f0 = f0[valid_indices]
            valid_probs = voiced_probs[valid_indices]

            return {
                "times": valid_times.tolist(),
                "f0": valid_f0.tolist(),
                "confidence": valid_probs.tolist(),
                "count": len(valid_f0),
            }

        except Exception as e:
            logger.warning(f"Pitch抽出失敗: {e}")
            return {"times": [], "f0": [], "confidence": [], "count": 0}

    def extract_dynamics(self, wav_path: Path) -> Dict[str, Any]:
        """Velocity/Dynamics抽出"""
        logger.info(f"🔊 Dynamics抽出: {wav_path.name}")

        try:
            y, sr = librosa.load(wav_path, sr=self.sr)

            # RMS energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

            # 時刻変換
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)

            # 正規化（0-127範囲）
            rms_norm = rms / (rms.max() + 1e-9)
            velocity = (rms_norm * 127).astype(int)

            return {"times": times.tolist(), "velocity": velocity.tolist(), "rms": rms.tolist()}

        except Exception as e:
            logger.warning(f"Dynamics抽出失敗: {e}")
            return {"times": [], "velocity": [], "rms": []}


# ========================================
# 2. ChordMap/Sections読み込み
# ========================================


class StructureLoader:
    """ChordMap/Sectionsから構造情報読み込み"""

    @staticmethod
    def load_chordmap(chordmap_path: Path) -> List[Dict[str, Any]]:
        """chordmap.json読み込み"""
        if not chordmap_path.exists():
            logger.warning(f"ChordMap未発見: {chordmap_path}")
            return []

        try:
            data = json.loads(chordmap_path.read_text())
            events = data.get("events", [])
            logger.info(f"✅ ChordMap読み込み: {len(events)} events")
            return events
        except Exception as e:
            logger.error(f"ChordMap読み込みエラー: {e}")
            return []

    @staticmethod
    def load_sections(sections_path: Path) -> Dict[str, Any]:
        """sections.json読み込み"""
        if not sections_path.exists():
            logger.warning(f"Sections未発見: {sections_path}")
            return {}

        try:
            data = json.loads(sections_path.read_text())
            logger.info(f"✅ Sections読み込み: {len(data.get('sections', []))} sections")
            return data
        except Exception as e:
            logger.error(f"Sections読み込みエラー: {e}")
            return {}

    @staticmethod
    def load_tempo_map(sections_data: Dict[str, Any]) -> List[List[float]]:
        """Tempo Map抽出"""
        tempo_map = sections_data.get("tempo_map", [])
        if tempo_map:
            logger.info(f"✅ Tempo Map: {len(tempo_map)} changes")
        else:
            logger.warning("Tempo Map未発見（固定120 BPM）")
        return tempo_map


# ========================================
# 3. ハイブリッド統合（Stem + ChordMap）
# ========================================


class HybridPlanGenerator:
    """Stem特徴 + ChordMap → Plan JSON生成"""

    def __init__(
        self,
        stem_features: Dict[str, Dict[str, Any]],
        chordmap: List[Dict[str, Any]],
        sections: Dict[str, Any],
        tempo_map: List[List[float]],
        bars: int = 32,
        emotion: str = "energetic",
    ):
        self.stem_features = stem_features
        self.chordmap = chordmap
        self.sections = sections
        self.tempo_map = tempo_map
        self.bars = bars
        self.emotion = emotion

    def generate_drums_plan(self) -> Dict[str, Any]:
        """Drums Plan生成（Stem Onset + ChordMap構造）"""
        logger.info("🥁 Drums Plan生成")

        drums_stem = self.stem_features.get("drums", {})
        onset_data = drums_stem.get("onsets", {})
        rhythm_data = drums_stem.get("rhythm", {})

        # Onset → Kick/Snare配置
        onset_times = onset_data.get("times", [])
        onset_strengths = onset_data.get("strengths", [])

        # 小節グリッド
        beats_per_bar = 4
        total_beats = self.bars * beats_per_bar

        # ChordMap → 小節境界
        bar_times = []
        for i, event in enumerate(self.chordmap[: self.bars]):
            time_ql = event.get("time", i * 4.0)
            bar_times.append(time_ql)

        return {
            "part": "drums",
            "bars": self.bars,
            "tempo_map": self.tempo_map,
            "onset_times": onset_times,
            "onset_strengths": onset_strengths,
            "bar_times": bar_times,
            "emotion": self.emotion,
            "source": "stem_hybrid",
        }

    def generate_bass_plan(self) -> Dict[str, Any]:
        """Bass Plan生成（Pitch Contour + ChordMap）"""
        logger.info("🎸 Bass Plan生成")

        bass_stem = self.stem_features.get("bass", {})
        pitch_data = bass_stem.get("pitch", {})
        onset_data = bass_stem.get("onsets", {})

        # Pitch → Note配置
        pitch_times = pitch_data.get("times", [])
        f0_values = pitch_data.get("f0", [])

        # ChordMap → Root Note
        chord_roots = []
        for event in self.chordmap[: self.bars]:
            root = event.get("root", "C")
            chord_roots.append(root)

        return {
            "part": "bass",
            "bars": self.bars,
            "tempo_map": self.tempo_map,
            "pitch_times": pitch_times,
            "f0": f0_values,
            "chord_roots": chord_roots,
            "onset_times": onset_data.get("times", []),
            "emotion": self.emotion,
            "source": "stem_hybrid",
        }

    def generate_piano_plan(self) -> Dict[str, Any]:
        """Piano Plan生成（ChordMap + Dynamics）"""
        logger.info("🎹 Piano Plan生成")

        piano_stem = self.stem_features.get("keyboard", {})
        dynamics_data = piano_stem.get("dynamics", {})

        # ChordMap → Voicing
        chords = []
        for event in self.chordmap[: self.bars]:
            chord_symbol = f"{event.get('root', 'C')}{event.get('quality', '')}"
            chords.append(chord_symbol)

        return {
            "part": "piano",
            "bars": self.bars,
            "tempo_map": self.tempo_map,
            "chords": chords,
            "velocity_curve": dynamics_data.get("velocity", []),
            "emotion": self.emotion,
            "source": "stem_hybrid",
        }

    def generate_guitar_plan(self) -> Dict[str, Any]:
        """Guitar Plan生成"""
        logger.info("🎸 Guitar Plan生成")

        guitar_stem = self.stem_features.get("guitar", {})

        chords = []
        for event in self.chordmap[: self.bars]:
            chord_symbol = f"{event.get('root', 'C')}{event.get('quality', '')}"
            chords.append(chord_symbol)

        return {
            "part": "guitar",
            "bars": self.bars,
            "tempo_map": self.tempo_map,
            "chords": chords,
            "emotion": self.emotion,
            "source": "stem_hybrid",
        }

    def generate_strings_plan(self) -> Dict[str, Any]:
        """Strings Plan生成"""
        logger.info("🎻 Strings Plan生成")

        strings_stem = self.stem_features.get("strings", {})

        chords = []
        for event in self.chordmap[: self.bars]:
            chord_symbol = f"{event.get('root', 'C')}{event.get('quality', '')}"
            chords.append(chord_symbol)

        return {
            "part": "strings",
            "bars": self.bars,
            "tempo_map": self.tempo_map,
            "chords": chords,
            "emotion": self.emotion,
            "source": "stem_hybrid",
        }

    def generate_all_plans(self) -> Dict[str, Dict[str, Any]]:
        """全パートPlan生成"""
        return {
            "drums": self.generate_drums_plan(),
            "bass": self.generate_bass_plan(),
            "piano": self.generate_piano_plan(),
            "guitar": self.generate_guitar_plan(),
            "strings": self.generate_strings_plan(),
        }


# ========================================
# 4. メイン処理
# ========================================


class StemHybridComposer:
    """Stem WAV + ChordMap ハイブリッド作曲システム"""

    def __init__(
        self,
        stems_dir: Path,
        analysis_dir: Path,
        output_dir: Path,
        bars: int = 32,
        emotion: str = "energetic",
    ):
        self.stems_dir = stems_dir
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.bars = bars
        self.emotion = emotion

        self.extractor = StemFeatureExtractor()
        self.structure_loader = StructureLoader()

    def map_stem_files(self) -> Dict[str, Path]:
        """Stem WAVファイルマッピング"""
        logger.info(f"📁 Stemディレクトリ: {self.stems_dir}")

        stem_map = {}

        # パターンマッチ
        patterns = {
            "drums": ["drums", "drum", "percussion"],
            "bass": ["bass"],
            "guitar": ["guitar"],
            "keyboard": ["keyboard", "piano", "keys"],
            "strings": ["strings", "string"],
            "synth": ["synth"],
            "vocals": ["vocals", "vocal"],
        }

        for wav_file in sorted(self.stems_dir.glob("*.wav")):
            name_lower = wav_file.name.lower()

            for key, keywords in patterns.items():
                if any(kw in name_lower for kw in keywords):
                    stem_map[key] = wav_file
                    logger.info(f"  {key}: {wav_file.name}")
                    break

        return stem_map

    def extract_stem_features(self, stem_map: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
        """全Stem特徴抽出"""
        logger.info("🎵 Stem特徴抽出開始")

        features = {}

        for part, wav_path in stem_map.items():
            if part == "vocals":
                continue  # Vocal除外

            logger.info(f"\n--- {part.upper()} ---")

            features[part] = {
                "onsets": self.extractor.extract_onsets(wav_path),
                "rhythm": self.extractor.extract_rhythm(wav_path),
                "pitch": self.extractor.extract_pitch_contour(wav_path),
                "dynamics": self.extractor.extract_dynamics(wav_path),
            }

        return features

    def run(self) -> Path:
        """メイン実行"""
        logger.info("🚀 Stem Hybridシステム起動")

        # 1. Stem WAVマッピング
        stem_map = self.map_stem_files()

        if not stem_map:
            raise ValueError(f"Stem WAV未発見: {self.stems_dir}")

        # 2. Stem特徴抽出
        stem_features = self.extract_stem_features(stem_map)

        # 3. ChordMap/Sections読み込み
        chordmap_path = self.analysis_dir / "chordmap.json"
        sections_path = self.analysis_dir / "sections.json"

        chordmap = self.structure_loader.load_chordmap(chordmap_path)
        sections = self.structure_loader.load_sections(sections_path)
        tempo_map = self.structure_loader.load_tempo_map(sections)

        # 4. Plan生成
        logger.info("\n📋 Plan JSON生成")
        plan_generator = HybridPlanGenerator(
            stem_features=stem_features,
            chordmap=chordmap,
            sections=sections,
            tempo_map=tempo_map,
            bars=self.bars,
            emotion=self.emotion,
        )

        plans = plan_generator.generate_all_plans()

        # 5. Plan保存
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = self.output_dir / "hybrid_plan.json"

        plan_data = {
            "meta": {
                "source": "stem_hybrid",
                "bars": self.bars,
                "emotion": self.emotion,
                "stems_dir": str(self.stems_dir),
                "analysis_dir": str(self.analysis_dir),
            },
            "tempo_map": tempo_map,
            "plans": plans,
        }

        plan_path.write_text(json.dumps(plan_data, indent=2, ensure_ascii=False))

        logger.info(f"✅ Plan保存: {plan_path}")

        # 6. midi_writer呼び出し（TODO: 実装）
        logger.info("\n🎹 midi_writer.py 呼び出し（未実装）")
        logger.info("  → scripts/midi_writer.py --plan hybrid_plan.json")

        return plan_path


def main():
    parser = argparse.ArgumentParser(description="Stem WAV + ChordMap ハイブリッド作曲システム")
    parser.add_argument("--stems", type=Path, required=True, help="Stem WAVディレクトリ")
    parser.add_argument(
        "--analysis", type=Path, required=True, help="分析JSONディレクトリ（chordmap/sections）"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/hybrid_midi"), help="出力ディレクトリ"
    )
    parser.add_argument("--bars", type=int, default=32, help="生成小節数")
    parser.add_argument(
        "--emotion",
        choices=["energetic", "calm", "melancholic", "hopeful", "intense"],
        default="energetic",
        help="感情プロファイル",
    )

    args = parser.parse_args()

    composer = StemHybridComposer(
        stems_dir=args.stems,
        analysis_dir=args.analysis,
        output_dir=args.output,
        bars=args.bars,
        emotion=args.emotion,
    )

    plan_path = composer.run()

    print(f"\n{'='*60}")
    print(f"✅ Stem Hybrid作曲完了")
    print(f"📋 Plan JSON: {plan_path}")
    print(f"{'='*60}\n")

    print("次のステップ:")
    print(f"  python scripts/midi_writer.py --plan {plan_path}")


if __name__ == "__main__":
    main()
