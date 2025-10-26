#!/usr/bin/env python3
"""
Generate song_package.yaml for each Gold/Silver quality song.

出口一本化：
- 入口：WAV（Moises/MUSDB）+ MIDI（LAMDA）二刀流
- 出口：song_package.yaml で論理統合（bars.parquet + 楽曲仕様3点）
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def determine_label_strength(bronze_rate: float, avg_confidence: float) -> str:
    """
    Quality gate判定
    - Gold: bronze_rate ≤ 0.20 かつ avg_confidence ≥ 0.5
    - Silver: bronze_rate ≤ 0.40 かつ avg_confidence ≥ 0.4
    - Bronze: それ以外
    """
    if bronze_rate <= 0.20 and avg_confidence >= 0.5:
        return 'gold'
    elif bronze_rate <= 0.40 and avg_confidence >= 0.4:
        return 'silver'
    else:
        return 'bronze'


def load_qa_report(qa_path: Path) -> Dict:
    """Load QA report and extract Gold/Silver songs"""
    with open(qa_path) as f:
        qa = json.load(f)
    
    qualified = []
    for r in qa['results']:
        if r['status'] != 'success' or r.get('empty'):
            continue
        
        bronze_rate = r.get('bronze_rate', 1.0)
        avg_confidence = r.get('avg_confidence', 0.0)
        label_strength = determine_label_strength(bronze_rate, avg_confidence)
        
        if label_strength in ('gold', 'silver'):
            qualified.append({
                'song_id': r['song_id'],
                'label_strength': label_strength,
                'avg_confidence': avg_confidence,
                'bronze_rate': bronze_rate,
                'total_events': r.get('total_events', 0)
            })
    
    logger.info(f"Loaded {len(qualified)} Gold/Silver songs from QA report")
    gold_count = sum(1 for s in qualified if s['label_strength'] == 'gold')
    silver_count = sum(1 for s in qualified if s['label_strength'] == 'silver')
    logger.info(f"  Gold: {gold_count}, Silver: {silver_count}")
    
    return qualified


def find_midi_content_id(clean_json_path: Path) -> Optional[str]:
    """Extract content_id from stage1_clean.json"""
    if not clean_json_path.exists():
        return None
    
    try:
        with open(clean_json_path) as f:
            data = json.load(f)
        return data.get('content_id')
    except Exception as e:
        logger.warning(f"Could not read {clean_json_path}: {e}")
        return None


def build_package(
    song_id: str,
    midi_guide_root: Path,
    wav_guide_root: Optional[Path],
    specs_root: Optional[Path],
    quality_info: Dict,
    run_id: str,
    code_version: str,
    dataset: str = 'midi_guide'
) -> Dict:
    """
    Build song_package.yaml structure
    
    Args:
        song_id: Song identifier
        midi_guide_root: Path to midi_guide root
        wav_guide_root: Path to wav_guide root (optional)
        specs_root: Path to specs root (optional)
        quality_info: Quality metrics from QA
        run_id: Run identifier
        code_version: Code version string
        dataset: Dataset name (moisesdb, musdb18, midi_guide)
    """
    midi_dir = midi_guide_root / song_id
    
    # IDs
    ids = {
        'song_id': song_id,
        'run_id': run_id,
        'code_version': code_version,
    }
    
    # MIDI content_id
    clean_json = midi_dir / 'stage1_clean.json'
    midi_content_id = find_midi_content_id(clean_json)
    if midi_content_id:
        ids['midi_content_id'] = midi_content_id
    
    # WAV file_id (from manifest if exists)
    if wav_guide_root:
        wav_dir = wav_guide_root / dataset / song_id
        manifest_files = list(wav_dir.glob('manifest*.json'))
        if manifest_files:
            try:
                with open(manifest_files[0]) as f:
                    manifest = json.load(f)
                if 'file_id' in manifest:
                    ids['wav_file_id'] = manifest['file_id']
            except:
                pass
    
    # Dataset
    ids['dataset'] = dataset
    
    # Paths (相対パス)
    paths = {'midi': {}, 'diagnostics': {}}
    
    # MIDI files
    if (midi_dir / 'stage1_clean.mid').exists():
        paths['midi']['stage1_clean'] = 'stage1_clean.mid'
    if (midi_dir / 'stage1_clean.json').exists():
        paths['midi']['stage1_clean_meta'] = 'stage1_clean.json'
    
    # MIDI parts
    for part in ['piano', 'guitar', 'bass', 'drums', 'vocal']:
        part_file = midi_dir / f'{part}.mid'
        if part_file.exists():
            paths['midi'][part] = f'{part}.mid'
    
    # Beat grid
    beat_grid = midi_dir / 'beat_grid.json'
    if beat_grid.exists():
        paths['beat_grid'] = 'beat_grid.json'
    
    # Bars (MIDI側を優先、WAV側があればそちらを参照)
    bars_midi = midi_dir / f'{song_id}.bars.parquet'
    bars_wav = None
    if wav_guide_root:
        wav_bars = wav_guide_root / dataset / song_id / f'{song_id}.bars.parquet'
        if wav_bars.exists():
            bars_wav = wav_bars
    
    if bars_wav:
        # WAV側のbarsを使う場合（相対パス）
        rel_path = os.path.relpath(bars_wav, midi_dir)
        paths['bars'] = rel_path
        paths['bars_ref'] = 'wav'
    elif bars_midi.exists():
        paths['bars'] = f'{song_id}.bars.parquet'
        paths['bars_ref'] = 'midi'
    
    # Chordmap
    chordmap = midi_dir / 'chordmap.json'
    if chordmap.exists():
        paths['chordmap'] = 'chordmap.json'
    
    chordmap_raw = midi_dir / 'chordmap.raw.json'
    if chordmap_raw.exists():
        paths['chordmap_raw'] = 'chordmap.raw.json'
    
    # Sections
    sections = midi_dir / 'sections.json'
    if sections.exists():
        paths['sections'] = 'sections.json'
    
    # Specs (if separate directory exists)
    if specs_root:
        spec_dir = specs_root / song_id
        spec_paths = {}
        
        if (spec_dir / 'sections.json').exists():
            rel_path = os.path.relpath(spec_dir / 'sections.json', midi_dir)
            spec_paths['sections'] = rel_path
        
        if (spec_dir / 'chordmap.json').exists():
            rel_path = os.path.relpath(spec_dir / 'chordmap.json', midi_dir)
            spec_paths['chordmap'] = rel_path
        
        if (spec_dir / 'lyric_anchors.json').exists():
            rel_path = os.path.relpath(spec_dir / 'lyric_anchors.json', midi_dir)
            spec_paths['anchors'] = rel_path
        
        if spec_paths:
            paths['spec'] = spec_paths
    
    # WAV diagnostics
    if wav_guide_root:
        wav_dir = wav_guide_root / dataset / song_id
        
        if (wav_dir / 'beat_grid.json').exists():
            rel_path = os.path.relpath(wav_dir / 'beat_grid.json', midi_dir)
            paths['diagnostics']['wav_beat_grid'] = rel_path
        
        if (wav_dir / 'accent_grid.json').exists():
            rel_path = os.path.relpath(wav_dir / 'accent_grid.json', midi_dir)
            paths['diagnostics']['wav_accent_grid'] = rel_path
        
        if (wav_dir / 'audio_chordmap.yaml').exists():
            rel_path = os.path.relpath(wav_dir / 'audio_chordmap.yaml', midi_dir)
            paths['diagnostics']['wav_audio_chordmap'] = rel_path
    
    # MIDI diagnostics
    midi_features = midi_dir / 'midi_features.parquet'
    if midi_features.exists():
        paths['diagnostics']['midi_features'] = 'midi_features.parquet'
    
    # Provenance
    provenance = {
        'source': 'lamda:midi_integration',
        'code_version': code_version,
        'created_utc': datetime.now(timezone.utc).isoformat()
    }
    
    # Quality summary
    quality_summary = {
        'avg_confidence': round(quality_info['avg_confidence'], 3),
        'bronze_rate': round(quality_info['bronze_rate'], 3),
        'label_strength': quality_info['label_strength'],
        'total_events': quality_info['total_events']
    }
    
    # Build package
    package = {
        'version': '1.0',
        'ids': ids,
        'paths': paths,
        'provenance': provenance,
        'quality_summary': quality_summary,
        'notes': 'このパッケージだけで編曲・レンダー・QAが再現できる'
    }
    
    return package


def main():
    parser = argparse.ArgumentParser(
        description='Generate song_package.yaml for Gold/Silver quality songs'
    )
    parser.add_argument(
        '--qa-report',
        type=Path,
        default=Path('qa_chordmap_full_reestimation.json'),
        help='QA report JSON path'
    )
    parser.add_argument(
        '--midi-guide-root',
        type=Path,
        required=True,
        help='Path to midi_guide root directory'
    )
    parser.add_argument(
        '--wav-guide-root',
        type=Path,
        help='Path to wav_guide root directory (optional)'
    )
    parser.add_argument(
        '--specs-root',
        type=Path,
        help='Path to specs root directory (optional)'
    )
    parser.add_argument(
        '--dataset',
        default='midi_guide',
        help='Dataset name (moisesdb, musdb18, midi_guide)'
    )
    parser.add_argument(
        '--run-id',
        default=f"local-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}",
        help='Run identifier'
    )
    parser.add_argument(
        '--code-version',
        default='local_lamda_midi_integration.py@unknown',
        help='Code version string'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without writing'
    )
    parser.add_argument(
        '--index-out',
        type=Path,
        help='Output CSV index of generated packages'
    )
    
    args = parser.parse_args()
    
    # Load QA report
    qualified_songs = load_qa_report(args.qa_report)
    
    if not qualified_songs:
        logger.error("No Gold/Silver songs found in QA report")
        return 1
    
    # Generate packages
    packages_created = []
    
    for song_info in qualified_songs:
        song_id = song_info['song_id']
        midi_dir = args.midi_guide_root / song_id
        
        if not midi_dir.exists():
            logger.warning(f"MIDI directory not found for {song_id}, skipping")
            continue
        
        # Build package
        package = build_package(
            song_id=song_id,
            midi_guide_root=args.midi_guide_root,
            wav_guide_root=args.wav_guide_root,
            specs_root=args.specs_root,
            quality_info=song_info,
            run_id=args.run_id,
            code_version=args.code_version,
            dataset=args.dataset
        )
        
        # Write package
        package_path = midi_dir / 'song_package.yaml'
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would create: {package_path}")
            if len(packages_created) < 3:  # Show first 3 samples
                print(yaml.dump(package, default_flow_style=False, allow_unicode=True))
        else:
            # Backup existing
            if package_path.exists():
                backup_path = package_path.with_suffix('.yaml.bak')
                package_path.rename(backup_path)
                logger.info(f"Backed up existing package to {backup_path}")
            
            with open(package_path, 'w', encoding='utf-8') as f:
                yaml.dump(package, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✓ Created: {package_path}")
        
        packages_created.append({
            'song_id': song_id,
            'package_path': str(package_path),
            'label_strength': song_info['label_strength'],
            'avg_confidence': song_info['avg_confidence'],
            'bronze_rate': song_info['bronze_rate']
        })
    
    # Write index
    if args.index_out and not args.dry_run:
        import csv
        
        with open(args.index_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'song_id', 'package_path', 'label_strength', 
                'avg_confidence', 'bronze_rate'
            ])
            writer.writeheader()
            writer.writerows(packages_created)
        
        logger.info(f"✓ Wrote index to {args.index_out}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary: {len(packages_created)} packages {'would be ' if args.dry_run else ''}created")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
