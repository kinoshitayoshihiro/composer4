#!/usr/bin/env python3
"""
Append to Pickle Shard

既存shardに新規データを段階的に追加。
Resume対応で重複回避、自動shard分割。

Usage:
    python scripts/append_to_pickle_shard.py \\
        --input-dir data/suno_clean/guitar_strum_mid \\
        --pickle-dir data/shards/hybrid \\
        --instrument guitar \\
        --technique strum \\
        --source suno \\
        --resume
"""

import argparse
import pathlib
import pickle
import json
from typing import Dict, List, Any, Optional
import hashlib


class PickleShardAppender:
    """Pickle Shard段階的追加ツール（Resume対応）"""
    
    def __init__(
        self,
        pickle_dir: str,
        instrument: str,
        shard_size: int = 5000,
        resume: bool = True,
    ):
        self.pickle_dir = pathlib.Path(pickle_dir)
        self.pickle_dir.mkdir(parents=True, exist_ok=True)
        self.instrument = instrument
        self.shard_size = shard_size
        self.resume = resume
        
        # Load existing shards
        self.current_shard_index = self._detect_latest_shard()
        self.buffer: List[Dict[str, Any]] = []
        self.processed_files = set()
        
        # Load current shard if exists
        if self.current_shard_index >= 0:
            self._load_current_shard()
        
        # Resume state
        if self.resume:
            self._load_resume_state()
    
    def _detect_latest_shard(self) -> int:
        """最新shardのインデックスを検出"""
        existing = list(self.pickle_dir.glob(f"{self.instrument}_shard_*.pkl"))
        if not existing:
            return -1
        
        indices = []
        for fp in existing:
            try:
                idx_str = fp.stem.split('_')[-1]
                indices.append(int(idx_str))
            except (ValueError, IndexError):
                continue
        
        return max(indices) if indices else -1
    
    def _load_current_shard(self):
        """現在のshardを読み込み（追加用）"""
        shard_path = self.pickle_dir / f"{self.instrument}_shard_{self.current_shard_index:05d}.pkl"
        
        if shard_path.exists():
            try:
                with open(shard_path, "rb") as f:
                    self.buffer = pickle.load(f)
                print(f"[INFO] Loaded existing shard: {shard_path.name} ({len(self.buffer)} entries)")
            except Exception as e:
                print(f"[WARNING] Failed to load shard {shard_path}: {e}")
                self.buffer = []
        else:
            self.buffer = []
    
    def _load_resume_state(self):
        """Resume状態を読み込み（重複回避用）"""
        resume_file = self.pickle_dir / f"{self.instrument}_resume.json"
        
        if resume_file.exists():
            try:
                with open(resume_file, "r") as f:
                    resume_data = json.load(f)
                    self.processed_files = set(resume_data.get("processed_files", []))
                print(f"[INFO] Resume state loaded: {len(self.processed_files)} files already processed")
            except Exception as e:
                print(f"[WARNING] Failed to load resume state: {e}")
                self.processed_files = set()
        else:
            self.processed_files = set()
    
    def _save_resume_state(self):
        """Resume状態を保存"""
        resume_file = self.pickle_dir / f"{self.instrument}_resume.json"
        
        resume_data = {
            "instrument": self.instrument,
            "current_shard_index": self.current_shard_index,
            "buffer_size": len(self.buffer),
            "processed_files": list(self.processed_files),
        }
        
        with open(resume_file, "w") as f:
            json.dump(resume_data, f, indent=2)
    
    def add_entry(self, metadata: Dict[str, Any]):
        """エントリを追加（自動flush）"""
        # Check if already processed (resume)
        file_id = metadata.get("file_path", "")
        if self.resume and file_id in self.processed_files:
            print(f"[SKIP] Already processed: {file_id}")
            return
        
        # Add to buffer
        self.buffer.append(metadata)
        self.processed_files.add(file_id)
        
        # Auto flush if buffer full
        if len(self.buffer) >= self.shard_size:
            self.flush()
    
    def flush(self):
        """バッファをshardに書き出し"""
        if not self.buffer:
            return
        
        # Determine shard path
        if self.current_shard_index < 0:
            # First shard
            self.current_shard_index = 0
        
        shard_path = self.pickle_dir / f"{self.instrument}_shard_{self.current_shard_index:05d}.pkl"
        
        # Write shard
        with open(shard_path, "wb") as f:
            pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"[FLUSH] {shard_path.name}: {len(self.buffer)} entries")
        
        # Reset buffer for next shard
        if len(self.buffer) >= self.shard_size:
            self.buffer = []
            self.current_shard_index += 1
        
        # Save resume state
        self._save_resume_state()
    
    def close(self):
        """最終バッファをflush"""
        self.flush()
        print(f"[COMPLETE] Total processed: {len(self.processed_files)} files")


def extract_metadata_from_midi(
    midi_file: pathlib.Path,
    instrument: str,
    technique: str,
    source: str,
    stage2_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    MIDIファイルからメタデータ抽出
    
    Args:
        midi_file: MIDIファイルパス
        instrument: 楽器名
        technique: 奏法名
        source: データソース（real/suno/external）
        stage2_result: Stage2スコアリング結果（オプション）
    
    Returns:
        Metadata dict
    """
    import pretty_midi as pm
    
    # Load MIDI
    try:
        midi = pm.PrettyMIDI(str(midi_file))
    except Exception as e:
        print(f"[ERROR] Failed to load MIDI {midi_file.name}: {e}")
        return None
    
    # Extract basic info
    metadata = {
        "file_path": str(midi_file),
        "file_name": midi_file.name,
        "instrument": instrument,
        "technique": technique,
        "source": source,
        "duration": midi.get_end_time(),
        "num_notes": sum(len(inst.notes) for inst in midi.instruments),
        "tempo": midi.estimate_tempo() if hasattr(midi, 'estimate_tempo') else None,
    }
    
    # Add Stage2 results if available
    if stage2_result:
        metadata["stage2_score"] = stage2_result.get("score", 0.0)
        metadata["stage2_passed"] = stage2_result.get("passed", False)
        metadata["lamda"] = stage2_result.get("metrics", {})
    
    # Add Suno-specific metadata
    if source == "suno":
        # Try to load conversion metadata
        meta_file = midi_file.with_suffix(".json")
        if meta_file.exists():
            with open(meta_file, "r") as f:
                conversion_meta = json.load(f)
                metadata["conversion_method"] = conversion_meta.get("conversion_method", "unknown")
                metadata["source_wav"] = conversion_meta.get("source_wav", "")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Append to pickle shard")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory with MIDI files")
    parser.add_argument("--pickle-dir", type=str, required=True,
                        help="Pickle shard directory")
    parser.add_argument("--instrument", type=str, required=True,
                        help="Instrument name (guitar/bass/strings/piano)")
    parser.add_argument("--technique", type=str, required=True,
                        help="Technique name (strum/legato/pick/etc)")
    parser.add_argument("--source", type=str, required=True,
                        choices=["real", "suno", "external"],
                        help="Data source")
    parser.add_argument("--shard-size", type=int, default=5000,
                        help="Shard size (default: 5000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous state (skip processed files)")
    parser.add_argument("--stage2-results", type=str, default=None,
                        help="Optional: Stage2 results JSON file")
    
    args = parser.parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1
    
    # Load Stage2 results if provided
    stage2_results = {}
    if args.stage2_results:
        stage2_file = pathlib.Path(args.stage2_results)
        if stage2_file.exists():
            with open(stage2_file, "r") as f:
                stage2_data = json.load(f)
                # Convert to dict keyed by filename
                for entry in stage2_data:
                    filename = pathlib.Path(entry.get("file", "")).name
                    stage2_results[filename] = entry
            print(f"[INFO] Loaded Stage2 results: {len(stage2_results)} entries")
    
    # Initialize appender
    appender = PickleShardAppender(
        pickle_dir=args.pickle_dir,
        instrument=args.instrument,
        shard_size=args.shard_size,
        resume=args.resume,
    )
    
    # Find MIDI files
    midi_files = list(input_dir.glob("*.mid"))
    if not midi_files:
        print(f"[ERROR] No MIDI files found in: {input_dir}")
        return 1
    
    print(f"[INFO] Found {len(midi_files)} MIDI files")
    
    # Process each MIDI
    added_count = 0
    skipped_count = 0
    
    for midi_file in midi_files:
        try:
            # Get Stage2 result if available
            stage2_result = stage2_results.get(midi_file.name)
            
            # Extract metadata
            metadata = extract_metadata_from_midi(
                midi_file,
                instrument=args.instrument,
                technique=args.technique,
                source=args.source,
                stage2_result=stage2_result,
            )
            
            if metadata is None:
                print(f"[SKIP] Failed to extract metadata: {midi_file.name}")
                skipped_count += 1
                continue
            
            # Add to shard
            appender.add_entry(metadata)
            added_count += 1
            
            if added_count % 100 == 0:
                print(f"[PROGRESS] Added {added_count}/{len(midi_files)} files...")
        
        except Exception as e:
            print(f"[ERROR] {midi_file.name}: {e}")
            skipped_count += 1
    
    # Final flush
    appender.close()
    
    # Summary
    print("\n" + "="*50)
    print(f"[SUMMARY]")
    print(f"  Added:   {added_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total:   {len(midi_files)}")
    print("="*50)
    
    print(f"\n[OUTPUT] Pickle shards: {args.pickle_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
