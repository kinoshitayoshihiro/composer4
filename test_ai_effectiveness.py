#!/usr/bin/env python3
"""
AI機能効果判定テストスクリプト
song_001の実データを使用して各AI機能の効果を判定する
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).parent))


class NumpyEncoder(json.JSONEncoder):
    """numpy型をJSON化するためのカスタムエンコーダ"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class AIEffectivenessTest:
    """AI機能の効果判定テストクラス"""

    def __init__(self, song_dir: str):
        self.song_dir = Path(song_dir)
        self.results = {"harmony_ai": {}, "magenta": {}, "onsets_and_frames": {}, "rhythm_ai": {}}

    def test_harmony_ai(self) -> dict[str, Any]:
        """和声AI効果判定テスト"""
        print("\n=== 和声AI効果判定テスト ===")

        chordmap_path = self.song_dir / "chordmap.json"
        if not chordmap_path.exists():
            return {"status": "failed", "reason": "chordmap.json not found"}

        with open(chordmap_path) as f:
            chordmap = json.load(f)

        events = chordmap.get("events", [])

        # 和声進行の分析
        chord_transitions = []
        chord_quality_diversity = set()
        root_diversity = set()

        for i, event in enumerate(events):
            root = event.get("root")
            quality = event.get("quality")

            if root:
                root_diversity.add(root)
            if quality:
                chord_quality_diversity.add(quality)

            if i > 0:
                prev_root = events[i - 1].get("root")
                chord_transitions.append((prev_root, root))

        # 和声的特徴の評価
        total_chords = len(events)
        unique_roots = len(root_diversity)
        unique_qualities = len(chord_quality_diversity)

        # スコア計算
        diversity_score = (unique_roots / 12.0) * 0.5 + (unique_qualities / 10.0) * 0.5
        complexity_score = min(1.0, total_chords / 100.0)

        # 音楽理論的妥当性チェック
        common_progressions = [
            ("D", "A"),
            ("A", "D"),
            ("G", "D"),
            ("D", "G"),
            ("A", "E"),
            ("E", "A"),
        ]

        valid_transitions = sum(
            1 for trans in chord_transitions if trans in common_progressions or trans[0] == trans[1]
        )
        theory_score = valid_transitions / max(1, len(chord_transitions))

        overall_score = diversity_score * 0.3 + complexity_score * 0.3 + theory_score * 0.4

        result = {
            "status": "success",
            "metrics": {
                "total_chords": total_chords,
                "unique_roots": unique_roots,
                "unique_qualities": unique_qualities,
                "chord_transitions": len(chord_transitions),
                "valid_transitions": valid_transitions,
                "diversity_score": round(diversity_score, 3),
                "complexity_score": round(complexity_score, 3),
                "theory_score": round(theory_score, 3),
                "overall_score": round(overall_score, 3),
            },
            "evaluation": (
                "excellent"
                if overall_score > 0.7
                else "good" if overall_score > 0.5 else "acceptable"
            ),
            "details": {
                "roots_used": sorted(list(root_diversity)),
                "qualities_used": sorted(list(chord_quality_diversity)),
            },
        }

        print(f"Total chords: {total_chords}")
        print(f"Unique roots: {unique_roots}")
        print(f"Unique qualities: {unique_qualities}")
        print(f"Overall score: {overall_score:.3f} ({result['evaluation']})")

        self.results["harmony_ai"] = result
        return result

    def test_magenta(self) -> dict[str, Any]:
        """Magenta効果判定テスト"""
        print("\n=== Magenta効果判定テスト ===")

        # Magentaが生成したMIDIファイルを探す
        full_arrangement_path = self.song_dir / "full_arrangement.mid"

        if not full_arrangement_path.exists():
            return {"status": "failed", "reason": "full_arrangement.mid not found"}

        try:
            # MIDIファイルの基本分析
            import pretty_midi

            midi_data = pretty_midi.PrettyMIDI(str(full_arrangement_path))

            # 各楽器の分析
            instruments_info = []
            total_notes = 0
            velocity_values = []
            note_durations = []

            for instrument in midi_data.instruments:
                notes = instrument.notes
                total_notes += len(notes)

                for note in notes:
                    velocity_values.append(note.velocity)
                    note_durations.append(note.end - note.start)

                instruments_info.append(
                    {
                        "program": instrument.program,
                        "name": (
                            instrument.name if instrument.name else f"Program {instrument.program}"
                        ),
                        "notes_count": len(notes),
                        "is_drum": instrument.is_drum,
                    }
                )

            # ベロシティの変動（人間らしさの指標）
            velocity_variance = np.var(velocity_values) if velocity_values else 0
            velocity_mean = np.mean(velocity_values) if velocity_values else 0

            # デュレーションの変動
            duration_variance = np.var(note_durations) if note_durations else 0
            duration_mean = np.mean(note_durations) if note_durations else 0

            # スコア計算
            humanization_score = min(1.0, velocity_variance / 500.0)  # 高い変動 = 人間らしい
            complexity_score = min(1.0, total_notes / 1000.0)
            arrangement_score = min(1.0, len(instruments_info) / 5.0)

            overall_score = (
                humanization_score * 0.4 + complexity_score * 0.3 + arrangement_score * 0.3
            )

            result = {
                "status": "success",
                "metrics": {
                    "total_instruments": len(instruments_info),
                    "total_notes": total_notes,
                    "velocity_mean": round(velocity_mean, 2),
                    "velocity_variance": round(velocity_variance, 2),
                    "duration_mean": round(duration_mean, 3),
                    "duration_variance": round(duration_variance, 3),
                    "humanization_score": round(humanization_score, 3),
                    "complexity_score": round(complexity_score, 3),
                    "arrangement_score": round(arrangement_score, 3),
                    "overall_score": round(overall_score, 3),
                },
                "evaluation": (
                    "excellent"
                    if overall_score > 0.7
                    else "good" if overall_score > 0.5 else "acceptable"
                ),
                "instruments": instruments_info,
            }

            print(f"Total instruments: {len(instruments_info)}")
            print(f"Total notes: {total_notes}")
            print(f"Velocity variance: {velocity_variance:.2f}")
            print(f"Overall score: {overall_score:.3f} ({result['evaluation']})")

            self.results["magenta"] = result
            return result

        except Exception as e:
            return {"status": "failed", "reason": f"Error analyzing MIDI: {str(e)}"}

    def test_onsets_and_frames(self) -> dict[str, Any]:
        """Onsets-and-Frames実データ抽出テスト"""
        print("\n=== Onsets-and-Frames実データ抽出テスト ===")

        vocal_path = self.song_dir / "vocal.wav"
        piano_path = self.song_dir / "piano.wav"

        if not vocal_path.exists() and not piano_path.exists():
            return {"status": "failed", "reason": "No audio files found"}

        results = {"status": "success", "files_analyzed": [], "metrics": {}}

        try:
            import librosa

            # Vocal analysis
            if vocal_path.exists():
                print(f"Analyzing {vocal_path.name}...")
                y_vocal, sr_vocal = librosa.load(str(vocal_path), sr=None)

                # オンセット検出
                onset_frames = librosa.onset.onset_detect(y=y_vocal, sr=sr_vocal)
                onset_times = librosa.frames_to_time(onset_frames, sr=sr_vocal)

                # ピッチ検出（基本周波数）
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y_vocal,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=sr_vocal,
                )

                vocal_info = {
                    "duration_sec": len(y_vocal) / sr_vocal,
                    "sample_rate": sr_vocal,
                    "onset_count": len(onset_frames),
                    "avg_onset_interval": (
                        float(np.mean(np.diff(onset_times))) if len(onset_times) > 1 else 0
                    ),
                    "f0_median": float(np.nanmedian(f0)) if not np.all(np.isnan(f0)) else 0,
                    "f0_std": float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0,
                    "voiced_ratio": float(np.mean(voiced_flag)),
                }

                results["files_analyzed"].append("vocal.wav")
                results["metrics"]["vocal"] = vocal_info

                print(f"  Onsets detected: {len(onset_frames)}")
                print(f"  F0 median: {vocal_info['f0_median']:.2f} Hz")

            # Piano analysis
            if piano_path.exists():
                print(f"Analyzing {piano_path.name}...")
                y_piano, sr_piano = librosa.load(str(piano_path), sr=None)

                # オンセット検出
                onset_frames = librosa.onset.onset_detect(y=y_piano, sr=sr_piano)
                onset_times = librosa.frames_to_time(onset_frames, sr=sr_piano)

                # スペクトル特徴
                spectral_centroids = librosa.feature.spectral_centroid(y=y_piano, sr=sr_piano)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y_piano, sr=sr_piano)[0]

                piano_info = {
                    "duration_sec": len(y_piano) / sr_piano,
                    "sample_rate": sr_piano,
                    "onset_count": len(onset_frames),
                    "avg_onset_interval": (
                        float(np.mean(np.diff(onset_times))) if len(onset_times) > 1 else 0
                    ),
                    "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                    "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                }

                results["files_analyzed"].append("piano.wav")
                results["metrics"]["piano"] = piano_info

                print(f"  Onsets detected: {len(onset_frames)}")
                print(f"  Spectral centroid: {piano_info['spectral_centroid_mean']:.2f} Hz")

            # スコア計算
            total_onsets = sum(
                results["metrics"][k].get("onset_count", 0) for k in results["metrics"]
            )

            onset_density_score = min(1.0, total_onsets / 500.0)
            file_coverage_score = len(results["files_analyzed"]) / 2.0

            overall_score = onset_density_score * 0.6 + file_coverage_score * 0.4

            results["summary"] = {
                "total_onsets": total_onsets,
                "onset_density_score": round(onset_density_score, 3),
                "file_coverage_score": round(file_coverage_score, 3),
                "overall_score": round(overall_score, 3),
                "evaluation": (
                    "excellent"
                    if overall_score > 0.7
                    else "good" if overall_score > 0.5 else "acceptable"
                ),
            }

            print(f"\nTotal onsets: {total_onsets}")
            print(f"Overall score: {overall_score:.3f} ({results['summary']['evaluation']})")

            self.results["onsets_and_frames"] = results
            return results

        except Exception as e:
            return {"status": "failed", "reason": f"Error analyzing audio: {str(e)}"}

    def test_rhythm_ai(self) -> dict[str, Any]:
        """rhythmAI効果判定テスト"""
        print("\n=== rhythmAI効果判定テスト ===")

        # リズム関連データの探索
        tempo_map_path = self.song_dir / "tempo_map.json"
        bars_path = self.song_dir / "bars.parquet"

        if not tempo_map_path.exists() and not bars_path.exists():
            return {"status": "failed", "reason": "No rhythm data found"}

        result = {"status": "success", "metrics": {}}

        # Tempo map analysis
        if tempo_map_path.exists():
            with open(tempo_map_path) as f:
                tempo_map = json.load(f)

            tempo_changes = tempo_map.get("changes", [])
            tempos = [change.get("bpm", 0) for change in tempo_changes if "bpm" in change]

            if tempos:
                result["metrics"]["tempo"] = {
                    "count": len(tempo_changes),
                    "mean_bpm": round(float(np.mean(tempos)), 2),
                    "std_bpm": round(float(np.std(tempos)), 2),
                    "min_bpm": round(float(np.min(tempos)), 2),
                    "max_bpm": round(float(np.max(tempos)), 2),
                }
                print(f"Tempo changes: {len(tempo_changes)}")
                print(f"Mean BPM: {result['metrics']['tempo']['mean_bpm']}")

        # Bars analysis
        if bars_path.exists():
            try:
                bars_df = pd.read_parquet(bars_path)

                bar_durations = []
                if "duration" in bars_df.columns:
                    bar_durations = bars_df["duration"].tolist()
                elif "end" in bars_df.columns and "start" in bars_df.columns:
                    bar_durations = (bars_df["end"] - bars_df["start"]).tolist()

                result["metrics"]["bars"] = {
                    "total_bars": len(bars_df),
                    "columns": list(bars_df.columns),
                }

                if bar_durations:
                    result["metrics"]["bars"]["mean_duration"] = round(
                        float(np.mean(bar_durations)), 3
                    )
                    result["metrics"]["bars"]["std_duration"] = round(
                        float(np.std(bar_durations)), 3
                    )

                print(f"Total bars: {len(bars_df)}")

            except Exception as e:
                print(f"Warning: Could not read bars.parquet: {e}")

        # スコア計算
        tempo_consistency_score = 0
        if "tempo" in result["metrics"]:
            tempo_std = result["metrics"]["tempo"].get("std_bpm", 0)
            tempo_consistency_score = max(0, 1.0 - (tempo_std / 20.0))  # 低い標準偏差 = 高い一貫性

        bars_score = 0
        if "bars" in result["metrics"]:
            total_bars = result["metrics"]["bars"].get("total_bars", 0)
            bars_score = min(1.0, total_bars / 100.0)

        overall_score = tempo_consistency_score * 0.5 + bars_score * 0.5

        result["summary"] = {
            "tempo_consistency_score": round(tempo_consistency_score, 3),
            "bars_score": round(bars_score, 3),
            "overall_score": round(overall_score, 3),
            "evaluation": (
                "excellent"
                if overall_score > 0.7
                else "good" if overall_score > 0.5 else "acceptable"
            ),
        }

        print(f"Overall score: {overall_score:.3f} ({result['summary']['evaluation']})")

        self.results["rhythm_ai"] = result
        return result

    def run_all_tests(self) -> dict[str, Any]:
        """全テストを実行"""
        print(f"\n{'='*60}")
        print("AI機能効果判定テスト")
        print(f"対象: {self.song_dir}")
        print(f"{'='*60}")

        # 各テストを実行
        self.test_harmony_ai()
        self.test_magenta()
        self.test_onsets_and_frames()
        self.test_rhythm_ai()

        # 総合評価
        scores = []
        for test_name, test_result in self.results.items():
            if isinstance(test_result, dict):
                if "metrics" in test_result and "overall_score" in test_result["metrics"]:
                    scores.append(test_result["metrics"]["overall_score"])
                elif "summary" in test_result and "overall_score" in test_result["summary"]:
                    scores.append(test_result["summary"]["overall_score"])

        overall_score = np.mean(scores) if scores else 0

        print(f"\n{'='*60}")
        print("総合評価")
        print(f"{'='*60}")
        print(f"Overall score: {overall_score:.3f}")
        print(
            f"Evaluation: {'excellent' if overall_score > 0.7 else 'good' if overall_score > 0.5 else 'acceptable'}"
        )

        # 結果を保存
        report_path = self.song_dir / "ai_effectiveness_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tests": self.results,
                    "overall_score": round(float(overall_score), 3),
                    "evaluation": (
                        "excellent"
                        if overall_score > 0.7
                        else "good" if overall_score > 0.5 else "acceptable"
                    ),
                },
                f,
                indent=2,
                ensure_ascii=False,
                cls=NumpyEncoder,
            )

        print(f"\nレポート保存: {report_path}")

        return self.results


def main():
    """メイン関数"""
    song_dir = (
        "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/suno_ai/suno_themesong/song_001"
    )

    tester = AIEffectivenessTest(song_dir)
    tester.run_all_tests()

    return 0


if __name__ == "__main__":
    sys.exit(main())
