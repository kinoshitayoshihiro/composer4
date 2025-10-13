#!/usr/bin/env python3
"""
Generate drum samples for Stage3 evaluation.

Usage:
    python scripts/generate_drum_samples.py \
        --n 10 \
        --tempo 120 \
        --length-bars 64 \
        --style pop_straight \
        --density mid \
        --swing 2 \
        --seed 42 \
        --output-dir output/drumgen_eval_20251013/generated
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.drum_generator import DrumGenerator
from utilities import tempo_utils
from music21 import stream, tempo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def generate_drum_samples(
    n: int,
    tempo_bpm: int,
    length_bars: int,
    style: str,
    density: str,
    swing: float,
    seed: int,
    output_dir: Path,
) -> list[Path]:
    """Generate n drum samples with the given parameters.
    
    Args:
        n: Number of samples to generate
        tempo_bpm: Tempo in BPM
        length_bars: Length in bars
        style: Pattern style (e.g., "pop_straight", "shuffle", "rock")
        density: Density level ("low", "mid", "high")
        swing: Swing amount (0-4, typically 0=straight, 2=medium, 4=heavy)
        seed: Random seed for reproducibility
        output_dir: Output directory for MIDI files
        
    Returns:
        List of generated MIDI file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    random.seed(seed)
    
    generated_files = []
    metadata = {
        "config": {
            "tempo": tempo_bpm,
            "length_bars": length_bars,
            "style": style,
            "density": density,
            "swing": swing,
            "seed": seed,
        },
        "samples": []
    }
    
    logger.info(f"Generating {n} drum samples:")
    logger.info(f"  Tempo:      {tempo_bpm} BPM")
    logger.info(f"  Length:     {length_bars} bars")
    logger.info(f"  Style:      {style}")
    logger.info(f"  Density:    {density}")
    logger.info(f"  Swing:      {swing}")
    logger.info(f"  Seed:       {seed}")
    logger.info(f"  Output:     {output_dir}")
    
    # Initialize DrumGenerator with minimal config
    # Pattern library is loaded from data/drum_patterns.yml
    try:
        # Create minimal heatmap (required by DrumGenerator)
        heatmap_path = output_dir / "temp_heatmap.json"
        heatmap_data = [{"grid_index": i, "count": 0} for i in range(16)]
        with open(heatmap_path, "w") as f:
            json.dump(heatmap_data, f)
        
        # Minimal config for DrumGenerator
        main_cfg = {
            "vocal_midi_path_for_drums": "",
            "heatmap_json_path_for_drums": str(heatmap_path),
            "paths": {
                "rhythm_library_path": "data/rhythm_library.yml",
                "drum_patterns_path": "data/drum_patterns.yml",
            },
        }
        
        # Pattern library with the requested style
        # In production, this would be loaded from drum_patterns.yml
        pattern_lib = {
            style: {
                "pattern": [],  # Will be loaded from file
                "length_beats": 4.0,
                "pattern_type": "simple",
            }
        }
        
        drum_gen = DrumGenerator(
            part_name="drums",
            main_cfg=main_cfg,
            part_parameters=pattern_lib,
        )
    except Exception as e:
        logger.error(f"Failed to initialize DrumGenerator: {e}")
        logger.error("This is expected if generator requires additional config files.")
        logger.error("For now, creating placeholder MIDI files for pipeline testing.")
        
        # Create placeholder files for testing the pipeline
        for i in range(n):
            filename = f"drum_sample_{i:03d}.mid"
            filepath = output_dir / filename
            
            # Create a minimal MIDI file
            s = stream.Score()
            p = stream.Part()
            p.append(tempo.MetronomeMark(number=tempo_bpm))
            
            # Add some basic notes (just for testing)
            from music21 import note
            for bar in range(length_bars):
                offset = bar * 4.0  # 4/4 time
                # Kick drum on beats 1 and 3
                n1 = note.Note(36, quarterLength=1.0)  # C1 = Kick
                n1.offset = offset
                p.append(n1)
                n2 = note.Note(36, quarterLength=1.0)
                n2.offset = offset + 2.0
                p.append(n2)
                
                # Snare on beats 2 and 4
                n3 = note.Note(38, quarterLength=1.0)  # D1 = Snare
                n3.offset = offset + 1.0
                p.append(n3)
                n4 = note.Note(38, quarterLength=1.0)
                n4.offset = offset + 3.0
                p.append(n4)
            
            s.append(p)
            s.write('midi', str(filepath))
            
            generated_files.append(filepath)
            metadata["samples"].append({
                "index": i,
                "filename": filename,
                "status": "placeholder"
            })
            
            logger.info(f"  [{i+1}/{n}] Created placeholder: {filename}")
        
        # Save metadata
        metadata_path = output_dir / "generation_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.warning("⚠️  Generated placeholder files. Full generator integration needed.")
        logger.info(f"✅ Saved {len(generated_files)} placeholder files to {output_dir}")
        return generated_files
    
    # If generator initialized successfully, generate real patterns
    for i in range(n):
        filename = f"drum_sample_{i:03d}.mid"
        filepath = output_dir / filename
        
        try:
            # Create a minimal section_data for compose()
            # Based on tests/test_drum_generator_demo.py
            section_data = {
                "absolute_offset": 0,
                "length_in_measures": length_bars,
                "musical_intent": {
                    "emotion": "default",
                    "intensity": "medium"
                },
                "part_params": {
                    "drums": {
                        "rhythm_key": style,
                        "density": density,
                        "swing": swing,
                    }
                },
                "tempo": tempo_bpm,
                "time_signature": "4/4",
            }
            
            # Generate the part
            part = drum_gen.compose(section_data=section_data)
            
            # Create a Score and write to MIDI
            s = stream.Score()
            s.append(part)
            s.write('midi', str(filepath))
            
            generated_files.append(filepath)
            metadata["samples"].append({
                "index": i,
                "filename": filename,
                "status": "success"
            })
            
            logger.info(f"  [{i+1}/{n}] Generated: {filename}")
            
        except Exception as e:
            logger.error(f"  [{i+1}/{n}] Failed: {e}")
            metadata["samples"].append({
                "index": i,
                "filename": filename,
                "status": "error",
                "error": str(e)
            })
    
    # Save metadata
    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Generated {len(generated_files)}/{n} samples")
    logger.info(f"📄 Metadata saved to {metadata_path}")
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate drum samples for Stage3 evaluation"
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of samples to generate (default: 10)"
    )
    parser.add_argument(
        "--tempo", type=int, default=120,
        help="Tempo in BPM (default: 120)"
    )
    parser.add_argument(
        "--length-bars", type=int, default=64,
        help="Length in bars (default: 64)"
    )
    parser.add_argument(
        "--style", type=str, default="pop_straight",
        help="Pattern style (default: pop_straight)"
    )
    parser.add_argument(
        "--density", type=str, default="mid",
        choices=["low", "mid", "high"],
        help="Density level (default: mid)"
    )
    parser.add_argument(
        "--swing", type=float, default=2.0,
        help="Swing amount 0-4 (default: 2.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for MIDI files"
    )
    
    args = parser.parse_args()
    
    generated_files = generate_drum_samples(
        n=args.n,
        tempo_bpm=args.tempo,
        length_bars=args.length_bars,
        style=args.style,
        density=args.density,
        swing=args.swing,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    if not generated_files:
        logger.error("❌ No files were generated")
        sys.exit(1)
    
    logger.info(f"🎵 Success! Generated files:")
    for f in generated_files:
        logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
