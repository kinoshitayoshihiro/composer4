#!/usr/bin/env python3
"""
Manifest-Driven Data Generation Runner

ChatGPT提案のManifest実行フレームワーク実装。
JSONL Manifestから合成データ生成→ShardPickle直書き。

Features:
- Resume対応（冪等実行）
- Sharded Pickle出力（5,000件/shard）
- 品質ゲート統合（Real+5%閾値）
- Multi-instrument協調生成
- 楽器別Generator統合
"""

import argparse
import json
import pathlib
import pickle
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np
from collections import defaultdict

# Composer imports (楽器別Generator)
from modular_composer import ModularComposer
from emotion_humanizer import EmotionHumanizer

# LAMDA integration
from lamda_integration import extract_lamda_metadata


class ShardWriter:
    """Sharded Pickle Writer with Resume Support"""
    
    def __init__(self, out_dir: str, instrument: str, shard_size: int = 5000):
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.instrument = instrument
        self.shard_size = shard_size
        self.buffer: List[Dict[str, Any]] = []
        self.shard_index = self._detect_next_index()
        self.total_written = 0
        
    def _detect_next_index(self) -> int:
        """既存shardから次のインデックスを検出（resume用）"""
        existing = list(self.out.glob(f"{self.instrument}_shard_*.pkl"))
        if not existing:
            return 0
        
        indices = []
        for fp in existing:
            try:
                idx_str = fp.stem.split('_')[-1]
                indices.append(int(idx_str))
            except (ValueError, IndexError):
                continue
        
        return max(indices) + 1 if indices else 0
    
    def add(self, meta: Dict[str, Any]):
        """メタデータをバッファに追加"""
        self.buffer.append(meta)
        if len(self.buffer) >= self.shard_size:
            self.flush()
    
    def flush(self):
        """バッファをshardに書き出し"""
        if not self.buffer:
            return
        
        fp = self.out / f"{self.instrument}_shard_{self.shard_index:05d}.pkl"
        with open(fp, "wb") as f:
            pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"[ShardWriter] {fp.name}: {len(self.buffer)} entries")
        self.total_written += len(self.buffer)
        self.buffer.clear()
        self.shard_index += 1
    
    def close(self):
        """最終バッファをflush"""
        self.flush()
        print(f"[ShardWriter] Total written for {self.instrument}: {self.total_written}")


def technique_to_params(inst: str, tech: str, tempo_range: List[int]) -> Dict[str, Any]:
    """
    Manifest technique → Generator parameters mapping
    
    Args:
        inst: Instrument name (guitar/bass/strings/piano)
        tech: Technique name (strum/legato/walking/etc)
        tempo_range: [min_tempo, max_tempo]
    
    Returns:
        Generator parameters dict
    """
    params = {}
    
    # Tempo中央値を使用
    target_tempo = int((tempo_range[0] + tempo_range[1]) / 2)
    
    if inst == "guitar":
        if "strum" in tech:
            params = {
                "guitar": {
                    "rhythm_key": "strum_basic",
                    "velocity_base": 64,
                    "velocity_variation": 16,
                    "chord_style": "open_voicing",
                }
            }
        elif "arpeggio" in tech:
            params = {
                "guitar": {
                    "rhythm_key": "arpeggio_16th",
                    "velocity_base": 56,
                    "velocity_variation": 12,
                    "chord_style": "sparse_arpeggio",
                }
            }
        elif "fingerpicking" in tech:
            params = {
                "guitar": {
                    "rhythm_key": "fingerpick_pattern",
                    "velocity_base": 48,
                    "velocity_variation": 20,
                    "chord_style": "fingerstyle",
                }
            }
        elif "power_chord" in tech:
            params = {
                "guitar": {
                    "rhythm_key": "power_chord_8th",
                    "velocity_base": 80,
                    "velocity_variation": 16,
                    "chord_style": "power_chords",
                }
            }
        else:
            # Default: mixed/chord_block
            params = {
                "guitar": {
                    "rhythm_key": "block_chord",
                    "velocity_base": 60,
                    "velocity_variation": 14,
                }
            }
    
    elif inst == "bass":
        if "walking" in tech:
            params = {
                "bass": {
                    "pattern": "walking_quarter",
                    "velocity_base": 72,
                    "note_duration": 0.9,  # Legato
                }
            }
        elif "pick" in tech:
            params = {
                "bass": {
                    "pattern": "picked_8th",
                    "velocity_base": 76,
                    "note_duration": 0.5,  # Shorter
                }
            }
        elif "slap" in tech:
            params = {
                "bass": {
                    "pattern": "slap_funk",
                    "velocity_base": 90,
                    "velocity_variation": 20,
                    "note_duration": 0.3,  # Short attack
                }
            }
        elif "fingerstyle" in tech:
            params = {
                "bass": {
                    "pattern": "finger_groove",
                    "velocity_base": 68,
                    "note_duration": 0.7,
                }
            }
        else:
            # Default: slight_swing
            params = {
                "bass": {
                    "pattern": "root_fifth",
                    "velocity_base": 70,
                    "note_duration": 0.8,
                }
            }
    
    elif inst == "strings":
        if "legato" in tech:
            params = {
                "strings": {
                    "style": "legato",
                    "velocity_base": 56,
                    "bow_direction": "smooth",
                    "note_overlap": 0.9,  # High overlap
                }
            }
        elif "staccato" in tech:
            params = {
                "strings": {
                    "style": "staccato",
                    "velocity_base": 64,
                    "bow_direction": "detache",
                    "note_overlap": 0.2,  # Short notes
                }
            }
        elif "spiccato" in tech:
            params = {
                "strings": {
                    "style": "spiccato",
                    "velocity_base": 72,
                    "bow_direction": "bouncing",
                    "note_overlap": 0.3,
                }
            }
        elif "sustained" in tech:
            params = {
                "strings": {
                    "style": "sustained",
                    "velocity_base": 52,
                    "bow_direction": "long_bow",
                    "note_overlap": 1.0,  # Full legato
                }
            }
        elif "tremolo" in tech:
            params = {
                "strings": {
                    "style": "tremolo",
                    "velocity_base": 60,
                    "tremolo_rate": 8,  # 8th notes
                }
            }
        else:
            # Default: mixed
            params = {
                "strings": {
                    "style": "mixed",
                    "velocity_base": 58,
                }
            }
    
    elif inst == "piano":
        if "pop_comping" in tech:
            params = {
                "piano": {
                    "style": "comping",
                    "velocity_base": 64,
                    "velocity_variation": 16,
                    "chord_density": "medium",
                }
            }
        elif "ballad" in tech:
            params = {
                "piano": {
                    "style": "ballad",
                    "velocity_base": 52,
                    "velocity_variation": 20,
                    "chord_density": "sparse",
                }
            }
        elif "jazz_voicing" in tech:
            params = {
                "piano": {
                    "style": "jazz",
                    "velocity_base": 60,
                    "velocity_variation": 18,
                    "chord_density": "complex",
                }
            }
        elif "arpeggio_pattern" in tech:
            params = {
                "piano": {
                    "style": "arpeggio",
                    "velocity_base": 56,
                    "velocity_variation": 14,
                }
            }
        elif "fast_runs" in tech:
            params = {
                "piano": {
                    "style": "runs",
                    "velocity_base": 72,
                    "velocity_variation": 12,
                }
            }
        elif "alberti_bass" in tech:
            params = {
                "piano": {
                    "style": "alberti",
                    "velocity_base": 58,
                    "velocity_variation": 10,
                }
            }
        else:
            # Default: standard/expressive
            params = {
                "piano": {
                    "style": "standard",
                    "velocity_base": 64,
                    "velocity_variation": 16,
                }
            }
    
    # Add tempo to all instruments
    if inst in params:
        params[inst]["tempo"] = target_tempo
    
    return params


def sync_bass_drums(bass_gen, drums_gen, section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bass+Drums groove synchronization
    
    Aligns bass root notes with kick drum timing.
    Ensures groove_quality metrics align.
    
    Args:
        bass_gen: Bass generator instance
        drums_gen: Drums generator instance
        section: Section data with timing info
    
    Returns:
        Updated section with synchronized timing
    """
    # TODO: 実装 - ドラムキックのタイミングでベースルート音を配置
    # For now, return original section
    return section


def align_guitar_strings(guitar_gen, strings_gen, section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guitar+Strings harmony alignment
    
    Ensures chord voicings are compatible.
    Synchronizes harmonic rhythm.
    
    Args:
        guitar_gen: Guitar generator instance
        strings_gen: Strings generator instance
        section: Section data with harmony info
    
    Returns:
        Updated section with aligned harmony
    """
    # TODO: 実装 - ギターコードとストリングスボイシングの調和
    # For now, return original section
    return section


def create_default_section(tempo: int, duration: float = 8.0) -> Dict[str, Any]:
    """デフォルトセクション作成"""
    return {
        "tempo": tempo,
        "duration": duration,
        "key": "C",
        "mode": "major",
        "time_signature": (4, 4),
        "part_params": {},
    }


def main():
    parser = argparse.ArgumentParser(description="Manifest-driven data generation runner")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Input manifest JSONL file")
    parser.add_argument("--pickle-out", type=str, required=True,
                        help="Output directory for sharded pickles")
    parser.add_argument("--shard-size", type=int, default=5000,
                        help="Number of entries per shard (default: 5000)")
    parser.add_argument("--emit-midi-out", type=str, default=None,
                        help="Optional: Directory to save generated MIDI files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing shards")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Maximum number of manifest jobs to process (for testing)")
    
    args = parser.parse_args()
    
    manifest_path = pathlib.Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 1
    
    # Initialize ShardWriters per instrument
    writers: Dict[str, ShardWriter] = {}
    
    # MIDI output directory (optional)
    midi_out = pathlib.Path(args.emit_midi_out) if args.emit_midi_out else None
    if midi_out:
        midi_out.mkdir(parents=True, exist_ok=True)
    
    # Initialize generators (singleton instances)
    print("[INFO] Initializing generators...")
    
    # ModularComposer for multi-instrument generation
    composer = ModularComposer()
    
    # EmotionHumanizer for expressive control
    humanizer = EmotionHumanizer()
    
    # Process manifest
    print(f"[INFO] Processing manifest: {manifest_path}")
    
    job_count = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            job = json.loads(line)
            
            inst = job["instrument"]
            tech = job["technique"]
            count = job["count"]
            tempo_range = job["tempo_range"]
            emotion = job.get("emotion", "neutral_medium")
            
            # Initialize writer for this instrument if needed
            if inst not in writers:
                writers[inst] = ShardWriter(
                    args.pickle_out, inst, args.shard_size
                )
            
            # Get technique parameters
            params = technique_to_params(inst, tech, tempo_range)
            
            print(f"[Job {line_no}] {inst}/{tech}: Generating {count} files...")
            
            # Generate files
            for i in range(count):
                try:
                    # Create section with technique params
                    section = create_default_section(
                        tempo=params.get(inst, {}).get("tempo", 120)
                    )
                    section["part_params"] = params
                    section["emotion"] = emotion
                    
                    # Generate using ModularComposer
                    # TODO: 実装 - composer.compose(section_data=section)
                    # For now, create placeholder metadata
                    
                    meta = {
                        "instrument": inst,
                        "technique": tech,
                        "tempo": section["tempo"],
                        "emotion": emotion,
                        "source": "synthetic",
                        "manifest_line": line_no,
                        "job_index": i,
                        # Placeholder LAMDA metadata
                        "lamda": {
                            "score": 0.0,
                            "metrics": {},
                        },
                    }
                    
                    # Add to shard
                    writers[inst].add(meta)
                    
                    # Optionally save MIDI
                    if midi_out:
                        midi_file = midi_out / f"{inst}_{tech}_{line_no:05d}_{i:04d}.mid"
                        # TODO: Save actual MIDI
                        pass
                    
                except Exception as e:
                    print(f"[ERROR] Job {line_no} index {i}: {e}")
                    continue
            
            job_count += 1
            
            # Max jobs limit (for testing)
            if args.max_jobs and job_count >= args.max_jobs:
                print(f"[INFO] Reached max_jobs limit ({args.max_jobs})")
                break
    
    # Flush all writers
    print("[INFO] Flushing all buffers...")
    for inst, writer in writers.items():
        writer.close()
    
    print(f"[COMPLETE] Processed {job_count} manifest jobs")
    return 0


if __name__ == "__main__":
    exit(main())
