# Stage2データ × 既存Generator統合計画

既存Generator（Pattern-based）にStage2高品質データ（Learning-based）を統合し、研究論文ベースの品質向上を実現

---

## 🎯 統合の目的

### 現状の課題
| 既存Generator | 問題点 | Stage2データによる解決 |
|--------------|--------|---------------------|
| **Pattern-based** | 固定パターンライブラリ（手動定義） | Stage2データから学習した動的パターン |
| **ルールベース選択** | if-else分岐による単純選択 | LAMDAメトリクスベースの最適選択 |
| **品質保証なし** | 生成結果の品質が未評価 | Stage2閾値による品質保証 |
| **データ活用なし** | POP909/SLAKH/LAMDAを未活用 | 3,559ファイルの高品質データ活用 |

### 統合後の利点
1. ✅ **品質保証**: Stage2メトリクス（Real+5%閾値）で品質担保
2. ✅ **Learning-based**: 3,559ファイルから学習した最適パターン選択
3. ✅ **研究論文ベース**: LAMDA論文のメトリクスを直接活用
4. ✅ **既存機能維持**: Controls/Humanization/Emotion mappingはそのまま活用

---

## 🏗️ アーキテクチャ

### 統合レイヤー構造

```
┌─────────────────────────────────────────────────┐
│  既存Generator (Pattern-based)                   │
│  - PianoGenerator (986行)                       │
│  - BassGenerator (2,487行)                      │
│  - GuitarGenerator/StringsGenerator/DrumGenerator│
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│  Stage2統合レイヤー（新規実装）                    │
│  1. Pattern Extractor: Stage2データ→Pattern DB   │
│  2. Pattern Recommender: ML-based選択            │
│  3. Quality Validator: Stage2メトリクス検証      │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│  Stage2データ（3,559 files）                      │
│  - Piano: 554 files (100% pass, 64.0%)          │
│  - Bass: 584 files (100% pass, 76.9%)           │
│  - Guitar: 963 files (67.7% pass, 42.9%)        │
│  - Strings: 696 files (69.7% pass, 51.1%)       │
│  - Drums: 873 loops (9-100% pass, 55.7%)        │
└─────────────────────────────────────────────────┘
```

---

## 📦 Phase 1: Pattern Extraction（パターン抽出）

### 1.1 Stage2データからPattern抽出

```python
# scripts/extract_stage2_patterns.py

import pickle
from pathlib import Path
import pretty_midi as pm
import json

class Stage2PatternExtractor:
    """Stage2データからパターンを抽出・保存"""
    
    def __init__(self, instrument: str):
        self.instrument = instrument
        self.stage2_dir = Path(f"output/{self._get_dataset()}/clean/{instrument}")
        self.stage2_results = self._load_stage2_results()
    
    def _get_dataset(self) -> str:
        """楽器→データセットマッピング"""
        mapping = {
            "melody": "pop909",
            "chords": "pop909",
            "bass": "slakh",
            "guitar": "slakh",
            "strings": "slakh",
            "drums": "slakh",
        }
        return mapping.get(self.instrument, "slakh")
    
    def _load_stage2_results(self) -> dict:
        """Stage2スコアリング結果を読み込み"""
        results_file = Path(f"output/test_results/{self.instrument}_full.json")
        
        if results_file.exists():
            with open(results_file, "r") as f:
                data = json.load(f)
                # File名 → Metricsのマッピング
                return {entry["file"]: entry for entry in data}
        
        return {}
    
    def extract_patterns(self) -> list[dict]:
        """全MIDIファイルからパターン抽出"""
        midi_files = list(self.stage2_dir.glob("*.mid"))
        patterns = []
        
        print(f"Extracting patterns from {len(midi_files)} files...")
        
        for midi_file in midi_files:
            try:
                # MIDI読み込み
                midi = pm.PrettyMIDI(str(midi_file))
                
                # Stage2メトリクス取得
                metrics = self.stage2_results.get(midi_file.name, {})
                
                # パターン特徴抽出
                pattern = {
                    "file": midi_file.name,
                    "notes": self._extract_notes(midi),
                    "tempo": midi.estimate_tempo(),
                    "duration": midi.get_end_time(),
                    "num_notes": sum(len(inst.notes) for inst in midi.instruments),
                    
                    # Stage2メトリクス
                    "metrics": metrics.get("metrics", {}),
                    "score": metrics.get("score", 0.0),
                    "passed": metrics.get("passed", False),
                    
                    # Technique推定（楽器別）
                    "technique": self._estimate_technique(midi),
                    
                    # Chord progression（Pianoの場合）
                    "chord_progression": self._extract_chords(midi) if self.instrument in ["melody", "chords"] else None,
                }
                
                patterns.append(pattern)
                
            except Exception as e:
                print(f"Error processing {midi_file.name}: {e}")
                continue
        
        print(f"Extracted {len(patterns)} patterns")
        return patterns
    
    def _extract_notes(self, midi: pm.PrettyMIDI) -> list[dict]:
        """MIDI → Note sequence"""
        notes = []
        for inst in midi.instruments:
            for note in inst.notes:
                notes.append({
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "start": note.start,
                    "end": note.end,
                    "duration": note.end - note.start,
                })
        
        # Sort by start time
        notes.sort(key=lambda n: n["start"])
        return notes
    
    def _estimate_technique(self, midi: pm.PrettyMIDI) -> str:
        """楽器別Technique推定"""
        notes = self._extract_notes(midi)
        
        if not notes:
            return "unknown"
        
        if self.instrument == "bass":
            return self._estimate_bass_technique(notes)
        elif self.instrument == "guitar":
            return self._estimate_guitar_technique(notes)
        elif self.instrument == "strings":
            return self._estimate_strings_technique(notes)
        elif self.instrument in ["melody", "chords"]:
            return self._estimate_piano_technique(notes)
        
        return "default"
    
    def _estimate_bass_technique(self, notes: list[dict]) -> str:
        """Bass technique推定"""
        # Inter-onset interval
        iois = [notes[i+1]["start"] - notes[i]["start"] 
                for i in range(len(notes)-1)]
        avg_ioi = sum(iois) / len(iois) if iois else 1.0
        
        # Note duration ratio
        durations = [n["duration"] for n in notes]
        avg_duration = sum(durations) / len(durations)
        duration_ratio = avg_duration / avg_ioi if avg_ioi > 0 else 0.5
        
        # Velocity variation
        velocities = [n["velocity"] for n in notes]
        vel_std = np.std(velocities) if len(velocities) > 1 else 0
        
        # Classification
        if avg_ioi < 0.3 and vel_std > 15:
            return "slap"  # Fast + high velocity variation
        elif avg_ioi > 0.8:
            return "walking"  # Slow, steady
        elif duration_ratio < 0.3:
            return "pick"  # Short, staccato
        else:
            return "fingerstyle"
    
    def _estimate_guitar_technique(self, notes: list[dict]) -> str:
        """Guitar technique推定"""
        # Chord detection (3+ simultaneous notes)
        simultaneous_notes = self._count_simultaneous_notes(notes)
        
        if simultaneous_notes > 3:
            # Strumming: Many simultaneous notes
            return "strum"
        elif len(notes) > 20 and simultaneous_notes < 2:
            # Arpeggio: Many notes, mostly single
            return "arpeggio"
        elif len(notes) < 10:
            # Power chord: Few notes, some simultaneous
            return "power_chord"
        else:
            return "fingerpicking"
    
    def _estimate_strings_technique(self, notes: list[dict]) -> str:
        """Strings technique推定"""
        # Note overlap ratio
        overlaps = []
        for i in range(len(notes)-1):
            if notes[i]["end"] > notes[i+1]["start"]:
                overlap = notes[i]["end"] - notes[i+1]["start"]
                overlaps.append(overlap)
        
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        
        # Duration ratio
        durations = [n["duration"] for n in notes]
        avg_duration = sum(durations) / len(durations)
        
        # Velocity variation
        velocities = [n["velocity"] for n in notes]
        vel_std = np.std(velocities) if len(velocities) > 1 else 0
        
        # Classification
        if avg_overlap > 0.1 or avg_duration > 1.0:
            return "legato"  # High overlap, long notes
        elif avg_duration < 0.2:
            return "staccato"  # Short notes
        elif vel_std > 15 and 0.2 < avg_duration < 0.6:
            return "spiccato"  # High velocity variation, medium duration
        elif avg_duration > 2.0:
            return "sustained"  # Very long notes
        else:
            return "mixed"
    
    def _estimate_piano_technique(self, notes: list[dict]) -> str:
        """Piano technique推定"""
        simultaneous_notes = self._count_simultaneous_notes(notes)
        
        if self.instrument == "melody":
            # Melody: Mostly single notes
            return "melody"
        else:
            # Chords: Multiple simultaneous notes
            if simultaneous_notes > 3:
                return "block_chords"
            elif len(notes) > 30:
                return "arpeggio"
            else:
                return "syncopated_chords"
    
    def _count_simultaneous_notes(self, notes: list[dict]) -> int:
        """同時発音数をカウント"""
        if not notes:
            return 0
        
        max_simultaneous = 0
        for note in notes:
            # Count notes that overlap with this note
            overlapping = sum(1 for n in notes 
                            if n["start"] < note["end"] and n["end"] > note["start"])
            max_simultaneous = max(max_simultaneous, overlapping)
        
        return max_simultaneous
    
    def _extract_chords(self, midi: pm.PrettyMIDI) -> list[str]:
        """Chord progression抽出（簡易版）"""
        # TODO: 完全実装（Chord recognition）
        return []
    
    def save_patterns(self, patterns: list[dict], output_path: str):
        """パターンをPickle保存"""
        with open(output_path, "wb") as f:
            pickle.dump(patterns, f)
        
        print(f"Saved {len(patterns)} patterns to {output_path}")


def main():
    """全楽器のパターン抽出"""
    import numpy as np
    
    instruments = ["melody", "chords", "bass", "guitar", "strings"]
    
    for instrument in instruments:
        print(f"\n{'='*50}")
        print(f"Extracting patterns: {instrument}")
        print('='*50)
        
        extractor = Stage2PatternExtractor(instrument)
        patterns = extractor.extract_patterns()
        
        # Filter: Stage2合格のみ
        passed_patterns = [p for p in patterns if p["passed"]]
        
        print(f"\nTotal: {len(patterns)} patterns")
        print(f"Passed: {len(passed_patterns)} patterns ({len(passed_patterns)/len(patterns)*100:.1f}%)")
        
        # Save
        output_path = f"data/patterns_stage2_{instrument}.pkl"
        extractor.save_patterns(passed_patterns, output_path)


if __name__ == "__main__":
    main()
```

### 1.2 実行

```bash
# Pattern抽出実行
python scripts/extract_stage2_patterns.py

# 出力:
# data/patterns_stage2_melody.pkl     (277 patterns)
# data/patterns_stage2_chords.pkl     (277 patterns)
# data/patterns_stage2_bass.pkl       (584 patterns)
# data/patterns_stage2_guitar.pkl     (963 → 651 passed)
# data/patterns_stage2_strings.pkl    (999 → 696 passed)
```

---

## 🤖 Phase 2: Pattern Recommender（ML推薦システム）

### 2.1 Pattern類似度検索

```python
# ml/pattern_recommender.py

import pickle
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PatternQuery:
    """パターン検索クエリ"""
    tempo: float
    emotion: str
    technique: Optional[str] = None
    chord_progression: Optional[List[str]] = None
    duration: Optional[float] = None


class PatternRecommender:
    """Stage2パターン推薦システム"""
    
    def __init__(self, instrument: str, patterns_path: str):
        self.instrument = instrument
        
        with open(patterns_path, "rb") as f:
            self.patterns = pickle.load(f)
        
        print(f"Loaded {len(self.patterns)} patterns for {instrument}")
    
    def recommend(
        self,
        query: PatternQuery,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> List[Dict]:
        """
        類似パターンを推薦
        
        Args:
            query: 検索クエリ
            top_k: 推薦数
            min_score: 最小スコア
        
        Returns:
            推薦パターン（スコア降順）
        """
        scores = []
        
        for pattern in self.patterns:
            # 類似度スコア計算
            similarity = self._calculate_similarity(query, pattern)
            
            # Stage2品質スコアも考慮
            quality = pattern["score"]
            
            # 総合スコア（類似度70% + 品質30%）
            total_score = similarity * 0.7 + quality * 0.3
            
            if total_score >= min_score:
                scores.append((pattern, total_score))
        
        # Sort by score (降順)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Top-K
        recommended = [
            {
                "pattern": p,
                "score": s,
                "file": p["file"],
                "technique": p["technique"],
                "metrics": p["metrics"],
            }
            for p, s in scores[:top_k]
        ]
        
        return recommended
    
    def _calculate_similarity(
        self,
        query: PatternQuery,
        pattern: Dict,
    ) -> float:
        """類似度スコア計算（0.0-1.0）"""
        scores = []
        
        # 1. Tempo similarity
        tempo_diff = abs(query.tempo - pattern["tempo"])
        tempo_score = max(0, 1 - tempo_diff / 50)  # ±50 BPM以内
        scores.append(tempo_score)
        
        # 2. Technique match
        if query.technique:
            technique_score = 1.0 if query.technique == pattern["technique"] else 0.3
            scores.append(technique_score)
        
        # 3. Duration similarity (if specified)
        if query.duration:
            duration_diff = abs(query.duration - pattern["duration"])
            duration_score = max(0, 1 - duration_diff / 10)  # ±10秒以内
            scores.append(duration_score)
        
        # 4. Chord progression similarity (Piano only)
        if query.chord_progression and pattern["chord_progression"]:
            chord_score = self._chord_similarity(
                query.chord_progression,
                pattern["chord_progression"]
            )
            scores.append(chord_score)
        
        # Average
        return sum(scores) / len(scores) if scores else 0.0
    
    def _chord_similarity(
        self,
        chords1: List[str],
        chords2: List[str],
    ) -> float:
        """Chord progression類似度"""
        if not chords1 or not chords2:
            return 0.5
        
        # Jaccard similarity
        set1 = set(chords1)
        set2 = set(chords2)
        
        intersection = set1 & set2
        union = set1 | set2
        
        return len(intersection) / len(union) if union else 0.0


# 楽器別Recommenderファクトリー
class RecommenderFactory:
    """楽器別Recommender生成"""
    
    _instances = {}
    
    @classmethod
    def get_recommender(cls, instrument: str) -> PatternRecommender:
        """Recommenderインスタンス取得（シングルトン）"""
        if instrument not in cls._instances:
            patterns_path = f"data/patterns_stage2_{instrument}.pkl"
            cls._instances[instrument] = PatternRecommender(instrument, patterns_path)
        
        return cls._instances[instrument]
```

---

## 🔧 Phase 3: Generator統合（既存Generator拡張）

### 3.1 PianoGenerator拡張

```python
# generator/piano_generator_stage2.py
# 既存PianoGeneratorを拡張（ファイル名は既存と分けて開発）

from generator.piano_generator import PianoGenerator as BasePianoGenerator
from ml.pattern_recommender import RecommenderFactory, PatternQuery
import pretty_midi as pm
from music21 import stream, note

class PianoGeneratorStage2(BasePianoGenerator):
    """Stage2統合版PianoGenerator"""
    
    def __init__(self, *args, use_stage2=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_stage2 = use_stage2
        
        if use_stage2:
            # Melody + Chords recommender
            self.melody_recommender = RecommenderFactory.get_recommender("melody")
            self.chords_recommender = RecommenderFactory.get_recommender("chords")
    
    def compose(self, section_data: dict) -> stream.Part:
        """既存composeをオーバーライド"""
        
        if not self.use_stage2:
            # Fallback: 既存のPattern-based生成
            return super().compose(section_data)
        
        # Stage2パターン推薦
        query = PatternQuery(
            tempo=section_data.get("tempo", 120),
            emotion=section_data.get("emotion", "default"),
            technique=section_data.get("part_params", {}).get("piano", {}).get("technique"),
            chord_progression=section_data.get("chord_progression", []),
            duration=section_data.get("length_in_measures", 4) * 4.0,  # measures → beats
        )
        
        # Melody推薦
        melody_patterns = self.melody_recommender.recommend(query, top_k=3)
        
        # Chords推薦
        chords_patterns = self.chords_recommender.recommend(query, top_k=3)
        
        if not melody_patterns or not chords_patterns:
            # Fallback: 既存生成
            print("[PianoGen] No Stage2 patterns found, using default generation")
            return super().compose(section_data)
        
        # Best patternを選択
        best_melody = melody_patterns[0]["pattern"]
        best_chords = chords_patterns[0]["pattern"]
        
        print(f"[PianoGen] Using Stage2 patterns:")
        print(f"  Melody: {best_melody['file']} (score: {melody_patterns[0]['score']:.2f})")
        print(f"  Chords: {best_chords['file']} (score: {chords_patterns[0]['score']:.2f})")
        
        # Stage2パターン→music21 Part変換
        part = self._apply_stage2_patterns(best_melody, best_chords, section_data)
        
        # 既存のHumanization/Controls適用
        part = self._apply_humanization(part, section_data)
        part = self._apply_controls(part, section_data)
        
        return part
    
    def _apply_stage2_patterns(
        self,
        melody_pattern: dict,
        chords_pattern: dict,
        section_data: dict,
    ) -> stream.Part:
        """Stage2パターン→music21 Part"""
        part = stream.Part()
        part.id = "Piano"
        
        # Melody notes追加
        for note_dict in melody_pattern["notes"]:
            n = note.Note(
                pitch=note_dict["pitch"],
                quarterLength=note_dict["duration"] * 2,  # seconds → QL (approx)
            )
            n.volume.velocity = note_dict["velocity"]
            n.offset = note_dict["start"] * 2
            part.append(n)
        
        # Chords notes追加
        for note_dict in chords_pattern["notes"]:
            n = note.Note(
                pitch=note_dict["pitch"],
                quarterLength=note_dict["duration"] * 2,
            )
            n.volume.velocity = note_dict["velocity"]
            n.offset = note_dict["start"] * 2
            part.append(n)
        
        return part
    
    def _apply_humanization(self, part: stream.Part, section_data: dict) -> stream.Part:
        """既存Humanization適用"""
        # BasePartGeneratorのHumanization機能を活用
        from utilities import humanizer
        
        # TODO: 既存humanizer統合
        return part
    
    def _apply_controls(self, part: stream.Part, section_data: dict) -> stream.Part:
        """既存Controls適用"""
        # CC11/CC64等を適用
        # TODO: 既存controls_bundle統合
        return part
```

### 3.2 BassGenerator拡張

```python
# generator/bass_generator_stage2.py

from generator.bass_generator import BassGenerator as BaseBassGenerator
from ml.pattern_recommender import RecommenderFactory, PatternQuery

class BassGeneratorStage2(BaseBassGenerator):
    """Stage2統合版BassGenerator"""
    
    def __init__(self, *args, use_stage2=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_stage2 = use_stage2
        
        if use_stage2:
            self.recommender = RecommenderFactory.get_recommender("bass")
    
    def compose(self, section_data: dict) -> stream.Part:
        """Stage2パターン推薦版"""
        
        if not self.use_stage2:
            return super().compose(section_data)
        
        # Query作成
        query = PatternQuery(
            tempo=section_data.get("tempo", 120),
            emotion=section_data.get("emotion", "default"),
            technique=self._select_technique(section_data),  # 既存technique選択ロジック活用
            duration=section_data.get("length_in_measures", 4) * 4.0,
        )
        
        # Recommendation
        recommended = self.recommender.recommend(query, top_k=3)
        
        if not recommended:
            return super().compose(section_data)
        
        best_pattern = recommended[0]["pattern"]
        
        print(f"[BassGen] Using Stage2 pattern: {best_pattern['file']}")
        print(f"  Technique: {best_pattern['technique']}")
        print(f"  Score: {recommended[0]['score']:.2f}")
        
        # Pattern適用
        part = self._apply_stage2_pattern(best_pattern, section_data)
        
        # Kick同期（既存機能）
        if section_data.get("drums_pattern"):
            part = self._sync_with_kick(part, section_data["drums_pattern"])
        
        # Humanization/Controls
        part = self._apply_humanization(part, section_data)
        
        return part
```

---

## ✅ Phase 4: 品質検証（Stage2メトリクス）

### 4.1 生成結果のStage2検証

```python
# ml/stage2_validator.py

from scripts.stage2_instrument_metrics import calculate_metrics

class Stage2Validator:
    """生成結果をStage2メトリクスで検証"""
    
    def __init__(self, instrument: str, threshold: float):
        self.instrument = instrument
        self.threshold = threshold
    
    def validate(self, midi_path: str) -> dict:
        """
        MIDIファイルをStage2メトリクスで検証
        
        Returns:
            {
                "passed": bool,
                "score": float,
                "metrics": dict,
            }
        """
        # Stage2メトリクス計算
        metrics = calculate_metrics(midi_path, self.instrument)
        
        # スコア計算
        score = self._calculate_score(metrics)
        
        # 閾値判定
        passed = score >= self.threshold
        
        return {
            "passed": passed,
            "score": score,
            "metrics": metrics,
        }
    
    def _calculate_score(self, metrics: dict) -> float:
        """メトリクス→総合スコア"""
        # 楽器別の重み付き平均
        if self.instrument == "bass":
            weights = {
                "root_accuracy": 0.3,
                "groove_quality": 0.3,
                "pitch_range_fit": 0.2,
                "velocity_consistency": 0.2,
            }
        elif self.instrument == "guitar":
            weights = {
                "arpeggio_quality": 0.3,
                "chord_consonance": 0.3,
                "strum_pattern_quality": 0.2,
                "pitch_range_fit": 0.2,
            }
        # ... 他の楽器
        
        score = sum(metrics.get(k, 0) * w for k, w in weights.items())
        return score
```

---

## 🎵 Phase 5: 作曲統合（ModularComposer連携）

### 5.1 ModularComposer拡張

```python
# modular_composer_stage2.py

from generator.piano_generator_stage2 import PianoGeneratorStage2
from generator.bass_generator_stage2 import BassGeneratorStage2
from ml.stage2_validator import Stage2Validator

class ModularComposerStage2:
    """Stage2統合版作曲システム"""
    
    def __init__(self, use_stage2=True):
        self.use_stage2 = use_stage2
        
        # Generators（Stage2版）
        self.piano_gen = PianoGeneratorStage2(use_stage2=use_stage2)
        self.bass_gen = BassGeneratorStage2(use_stage2=use_stage2)
        # ... 他の楽器
        
        # Validators
        self.validators = {
            "piano": Stage2Validator("melody", threshold=0.45),
            "bass": Stage2Validator("bass", threshold=0.40),
            # ... 他の楽器
        }
    
    def compose(self, song_structure: dict) -> dict:
        """楽曲生成（全セクション）"""
        parts = {}
        
        for section in song_structure["sections"]:
            # Piano生成
            piano_part = self.piano_gen.compose(section)
            
            # Stage2検証
            if self.use_stage2:
                validation = self._validate_part(piano_part, "piano")
                
                if not validation["passed"]:
                    print(f"[Warning] Piano part failed Stage2 validation")
                    print(f"  Score: {validation['score']:.2f} (threshold: 0.45)")
                    # Retry or fallback
            
            parts["piano"] = piano_part
            
            # Bass生成（Piano情報を渡す）
            section["piano_part"] = piano_part
            bass_part = self.bass_gen.compose(section)
            parts["bass"] = bass_part
            
            # ... 他の楽器
        
        return parts
    
    def _validate_part(self, part, instrument: str) -> dict:
        """Part→MIDI→Stage2検証"""
        # music21 Part → MIDI
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            part.write("midi", fp=tmp.name)
            
            # Stage2検証
            validation = self.validators[instrument].validate(tmp.name)
            
            return validation
```

---

## 📅 実装スケジュール

### Week 1: Pattern Extraction
- [ ] Day 1-2: `extract_stage2_patterns.py`実装
- [ ] Day 3-4: 全楽器パターン抽出実行
- [ ] Day 5: Pattern DB検証（pickle確認）

### Week 2: Pattern Recommender
- [ ] Day 1-3: `PatternRecommender`実装
- [ ] Day 4-5: 類似度計算アルゴリズム調整
- [ ] Day 5: Top-K推薦テスト

### Week 3: Generator統合
- [ ] Day 1-2: `PianoGeneratorStage2`実装
- [ ] Day 3-4: `BassGeneratorStage2`実装
- [ ] Day 5: Guitar/Strings/Drums Generator拡張

### Week 4: 品質検証 + 統合テスト
- [ ] Day 1-2: `Stage2Validator`実装
- [ ] Day 3-4: `ModularComposerStage2`統合
- [ ] Day 5: 全楽器統合テスト

---

## 🎯 成功指標

### 定量評価
| 指標 | 既存Generator | Stage2統合後 | 目標 |
|-----|--------------|-------------|------|
| **Piano品質** | 不明（検証なし） | Stage2メトリクス | ≥64% |
| **Bass品質** | 不明 | Stage2メトリクス | ≥77% |
| **Pattern多様性** | 固定（~20パターン） | 動的（277-584パターン） | 10倍増 |
| **合格率** | 不明 | Stage2閾値検証 | ≥70% |

### 定性評価
- ✅ 既存機能（Controls/Humanization）維持
- ✅ Emotion mapping維持
- ✅ 研究論文ベース（LAMDA）の品質保証
- ✅ Learning-based動的パターン選択

---

## 📝 次のアクション

**Immediate (今日):**
1. ⏸️ `extract_stage2_patterns.py`実装開始
2. ⏸️ Piano/Bass pattern抽出テスト

**Short-term (今週):**
1. Pattern抽出完了（全楽器）
2. PatternRecommender実装
3. PianoGeneratorStage2実装・テスト

**作曲デモ計画:**
完成後、**既存Generator + Stage2統合版**で実際の楽曲生成デモを実行します。

- ✅ 3-5曲の完全楽曲生成
- ✅ Stage2品質保証
- ✅ 既存機能（Humanization/Controls）活用
- ✅ 研究論文ベースの高品質作曲

---

統合計画は以上です。既存Generatorの成熟した機能を維持しながら、Stage2データで品質向上を実現します！
