#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/generate_stage1_jsons.py

Stage1パイプライン統合（v4.1対応）
- chordmap.json（コード進行） + スキーマ統一
- sections.json（セクション構造）
- lyric_anchors.json（歌詞アンカー）
- mix_context.json（ミックスコンテキスト）

全てのStage1 JSONを一括生成します。
"""
from __future__ import annotations
import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, List

# v4.1: スキーマ統一コンバータ
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ops.chordmap_unify import unify_chordmap_dict
    _HAS_UNIFY = True
except ImportError as e:
    _HAS_UNIFY = False
    print(f"[WARN] chordmap_unify not available: {e}", file=sys.stderr)

def run_command(cmd: List[str], description: str) -> bool:
    """Run subprocess command with error handling"""
    print(f"[RUN] {description}")
    print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"[ERROR] Command not found: {cmd[0]}")
        return False

def generate_chordmap(
    stems_dir: Path,
    output_path: Path,
    sections_path: Optional[Path] = None,
    exclude: List[str] = None,
    force_key: Optional[str] = None,
    use_7th: bool = False,
    use_enhanced: bool = False,
    use_extended: bool = False
) -> bool:
    """Generate chordmap.json using stem_harmony"""
    
    # Select appropriate version
    if use_extended:
        script = "ops/stem_harmony_extended.py"
    elif use_enhanced:
        script = "ops/stem_harmony_7th_v2.py"
    elif use_7th:
        script = "ops/stem_harmony_7th.py"
    else:
        script = "ops/stem_harmony.py"
    
    cmd = [
        sys.executable,
        script,
        "--stems", str(stems_dir),
        "--out", str(output_path)
    ]
    
    if sections_path and sections_path.exists():
        cmd.extend(["--sections", str(sections_path)])
    
    if exclude:
        for ex in exclude:
            cmd.extend(["--exclude", ex])
    
    if force_key:
        cmd.extend(["--force-key", force_key])
    
    return run_command(cmd, f"Generate chordmap ({script})")

def generate_sections(
    audio_path: Path,
    output_path: Path,
    method: str = "laplacian"
) -> bool:
    """Generate sections.json using section detection"""
    
    # Check if ops/section_detector.py exists
    script = Path("ops/section_detector.py")
    if not script.exists():
        print(f"[WARN] {script} not found, skipping sections.json generation")
        # Create minimal sections.json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sections_data = {
            "unit": "sec",
            "sections": [
                {"label": "full", "start": 0.0, "end": 180.0}
            ]
        }
        output_path.write_text(json.dumps(sections_data, indent=2))
        print(f"[OK] Created minimal sections.json -> {output_path}")
        return True
    
    cmd = [
        sys.executable,
        str(script),
        "--audio", str(audio_path),
        "--out", str(output_path),
        "--method", method
    ]
    
    return run_command(cmd, "Generate sections.json")

def generate_lyric_anchors(
    vocal_path: Path,
    lyric_path: Optional[Path],
    output_path: Path,
    sections_path: Optional[Path] = None,
    window_mode: str = "class",
    sibilant_scale: float = 1.0,
    sibilant_only: bool = False
) -> bool:
    """Generate lyric_anchors.json using anchors_from_vocal"""
    
    # Check if ops/anchors_from_vocal.py exists
    script = Path("ops/anchors_from_vocal.py")
    if not script.exists():
        print(f"[WARN] {script} not found, skipping lyric_anchors.json generation")
        # Create empty anchors
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anchors_data = {
            "unit": "sec",
            "anchors": []
        }
        output_path.write_text(json.dumps(anchors_data, indent=2))
        print(f"[OK] Created empty lyric_anchors.json -> {output_path}")
        return True
    
    if not vocal_path.exists():
        print(f"[WARN] Vocal file not found: {vocal_path}")
        return False
    
    cmd = [
        sys.executable,
        str(script),
        "--vocal", str(vocal_path),
        "--out", str(output_path)
    ]
    
    if lyric_path and lyric_path.exists():
        cmd.extend(["--lyrics", str(lyric_path)])
    
    if sections_path and sections_path.exists():
        cmd.extend(["--sections", str(sections_path)])
    
    if window_mode != "class":
        cmd.extend(["--window-mode", window_mode])
    
    if sibilant_scale != 1.0:
        cmd.extend(["--sibilant-scale", str(sibilant_scale)])
    
    if sibilant_only:
        cmd.append("--sibilant-only")
    
    return run_command(cmd, "Generate lyric_anchors.json")

def generate_mix_context(
    stems_dir: Path,
    output_path: Path
) -> bool:
    """Generate mix_context.json (stem levels and characteristics)"""
    
    # Analyze stems and create mix context
    stems = list(stems_dir.glob("*.wav"))
    if not stems:
        print(f"[WARN] No WAV files found in {stems_dir}")
        return False
    
    mix_data = {
        "stems": []
    }
    
    for stem_path in sorted(stems):
        stem_name = stem_path.stem
        
        # Detect stem type
        name_lower = stem_name.lower()
        if "vocal" in name_lower or "voice" in name_lower:
            stem_type = "vocals"
        elif "drum" in name_lower or "beat" in name_lower:
            stem_type = "drums"
        elif "bass" in name_lower:
            stem_type = "bass"
        elif "guitar" in name_lower:
            stem_type = "guitar"
        elif "piano" in name_lower or "key" in name_lower:
            stem_type = "piano"
        else:
            stem_type = "other"
        
        mix_data["stems"].append({
            "name": stem_name,
            "path": str(stem_path.relative_to(stems_dir.parent)),
            "type": stem_type,
            "level": 1.0,  # Default level
            "pan": 0.0      # Center
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(mix_data, indent=2, ensure_ascii=False))
    print(f"[OK] Created mix_context.json -> {output_path}")
    
    return True

def main():
    ap = argparse.ArgumentParser(description="Generate all Stage1 JSONs (chordmap, sections, lyric_anchors, mix_context)")
    ap.add_argument("--song-dir", required=True, help="Song directory (contains stems/, analysis/, etc.)")
    ap.add_argument("--stems-subdir", default="stemswav_001", help="Stems subdirectory name")
    ap.add_argument("--output-dir", help="Output directory (default: song_dir/analysis)")
    ap.add_argument("--exclude", action="append", default=[], help="Stems to exclude (e.g., Vocals)")
    ap.add_argument("--force-key", help="Force key (e.g., C, Dm)")
    
    # Chord recognition options
    ap.add_argument("--use-7th", action="store_true", help="Use 7th chords version")
    ap.add_argument("--use-enhanced", action="store_true", help="Use 7th Enhanced version (local key prior)")
    ap.add_argument("--use-extended", action="store_true", help="Use extended chords version (sus4/add9/6th)")
    
    # Section detection options
    ap.add_argument("--section-method", default="laplacian", choices=["laplacian", "novelty"], help="Section detection method")
    
    # Lyric anchors options
    ap.add_argument("--window-mode", choices=["class","fixed","beat","proportional","energy"], default="class",
                    help="Anchor window mode (default: class)")
    ap.add_argument("--sibilant-scale", type=float, default=1.0, help="Sibilant window scale factor")
    ap.add_argument("--sibilant-only", action="store_true", help="Generate only sibilant anchors")
    
    # Skip options
    ap.add_argument("--skip-chordmap", action="store_true", help="Skip chordmap generation")
    ap.add_argument("--skip-sections", action="store_true", help="Skip sections generation")
    ap.add_argument("--skip-lyrics", action="store_true", help="Skip lyric anchors generation")
    ap.add_argument("--skip-mix", action="store_true", help="Skip mix context generation")
    
    args = ap.parse_args()
    
    song_dir = Path(args.song_dir)
    if not song_dir.exists():
        print(f"[ERROR] Song directory not found: {song_dir}")
        return 1
    
    # Setup paths
    stems_dir = song_dir / args.stems_subdir
    if not stems_dir.exists():
        # Try "stems" directory
        stems_dir = song_dir / "stems"
        if not stems_dir.exists():
            print(f"[ERROR] Stems directory not found: {stems_dir}")
            return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else (song_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find vocal and lyric files
    vocal_candidates = list(stems_dir.glob("*vocal*.wav")) + list(stems_dir.glob("*Vocal*.wav"))
    vocal_path = vocal_candidates[0] if vocal_candidates else None
    
    lyric_candidates = list(song_dir.glob("lyric*.txt")) + list(song_dir.glob("lyrics*.txt"))
    lyric_path = lyric_candidates[0] if lyric_candidates else None
    
    # Find mix audio (for section detection)
    mix_candidates = list(stems_dir.glob("*mix*.wav")) + list(stems_dir.glob("*Mix*.wav"))
    if not mix_candidates:
        # Use first stem
        mix_candidates = list(stems_dir.glob("*.wav"))
    mix_audio = mix_candidates[0] if mix_candidates else None
    
    print("=" * 60)
    print("Stage1 Pipeline - JSON Generation")
    print("=" * 60)
    print(f"Song dir:    {song_dir}")
    print(f"Stems dir:   {stems_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Vocal:       {vocal_path if vocal_path else 'Not found'}")
    print(f"Lyrics:      {lyric_path if lyric_path else 'Not found'}")
    print(f"Mix audio:   {mix_audio if mix_audio else 'Not found'}")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # 1. Generate sections.json (do this first, as chordmap may use it)
    if not args.skip_sections:
        total_count += 1
        sections_path = output_dir / "sections.json"
        if mix_audio:
            if generate_sections(mix_audio, sections_path, args.section_method):
                success_count += 1
                print(f"✅ sections.json -> {sections_path}")
            else:
                print(f"❌ sections.json generation failed")
        else:
            print(f"⚠️  Skipping sections.json (no audio found)")
    else:
        print(f"⏭️  Skipping sections.json (--skip-sections)")
    
    # 2. Generate chordmap.json
    if not args.skip_chordmap:
        total_count += 1
        chordmap_path = output_dir / "chordmap.json"
        sections_path = output_dir / "sections.json"
        
        if generate_chordmap(
            stems_dir,
            chordmap_path,
            sections_path=sections_path if sections_path.exists() else None,
            exclude=args.exclude,
            force_key=args.force_key,
            use_7th=args.use_7th,
            use_enhanced=args.use_enhanced,
            use_extended=args.use_extended
        ):
            # v4.1: スキーマ統一（秒/QL・配列/辞書ゆれを吸収）
            if _HAS_UNIFY and chordmap_path.exists():
                try:
                    with open(chordmap_path, "r", encoding="utf-8") as f:
                        raw_chordmap = json.load(f)
                    
                    unified = unify_chordmap_dict(
                        raw_chordmap,
                        to_unit="ql",
                        snap_ql=0.25,  # 16分音符グリッド
                        merge_N=True,
                        min_N_ql=2.0,  # 最小2QL（8分音符）
                        glue_same_root=True,
                    )
                    
                    with open(chordmap_path, "w", encoding="utf-8") as f:
                        json.dump(unified, f, ensure_ascii=False, indent=2)
                    
                    print(f"[INFO] Unified chordmap schema (events: {len(unified.get('events', []))})")
                except Exception as e:
                    print(f"[WARN] Chordmap unification failed: {e}", file=sys.stderr)
            
            success_count += 1
            print(f"✅ chordmap.json -> {chordmap_path}")
        else:
            print(f"❌ chordmap.json generation failed")
    else:
        print(f"⏭️  Skipping chordmap.json (--skip-chordmap)")
    
    # 3. Generate lyric_anchors.json
    if not args.skip_lyrics:
        total_count += 1
        anchors_path = output_dir / "lyric_anchors.json"
        sections_path = output_dir / "sections.json"
        
        if vocal_path:
            if generate_lyric_anchors(
                vocal_path, 
                lyric_path, 
                anchors_path,
                sections_path=sections_path if sections_path.exists() else None,
                window_mode=args.window_mode,
                sibilant_scale=args.sibilant_scale,
                sibilant_only=args.sibilant_only
            ):
                success_count += 1
                print(f"✅ lyric_anchors.json -> {anchors_path}")
            else:
                print(f"❌ lyric_anchors.json generation failed")
        else:
            print(f"⚠️  Skipping lyric_anchors.json (vocal not found)")
    else:
        print(f"⏭️  Skipping lyric_anchors.json (--skip-lyrics)")
    
    # 4. Generate mix_context.json
    if not args.skip_mix:
        total_count += 1
        mix_path = output_dir / "mix_context.json"
        
        if generate_mix_context(stems_dir, mix_path):
            success_count += 1
            print(f"✅ mix_context.json -> {mix_path}")
        else:
            print(f"❌ mix_context.json generation failed")
    else:
        print(f"⏭️  Skipping mix_context.json (--skip-mix)")
    
    print("=" * 60)
    print(f"Stage1 Pipeline Complete: {success_count}/{total_count} successful")
    print("=" * 60)
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())
