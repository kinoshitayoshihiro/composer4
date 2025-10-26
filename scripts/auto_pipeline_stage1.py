#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/auto_pipeline_stage1.py

完全自動Stage1パイプライン（ステムWAV → sections/chordmap/anchors → modular_composer準備完了）

実行フロー：
1. sections_from_audio.py: ステムから自動セクション推定
2. stem_harmony_7th_v2.py: chordmap生成（v4.1機能付き）
3. chordmap_unify.py: 秒単位統一化
4. anchors_from_vocal.py: lyric_anchors生成（Phase 23対応）
5. JSON→modular_composer形式変換（sections統合）

使用例：
python scripts/auto_pipeline_stage1.py \
  --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
  --vocal "data/suno_ai/suno_themesong/song_001/stemswav_001/stem_wav_001_(Vocals).wav" \
  --lyrics data/suno_ai/suno_themesong/song_001/lyric.txt \
  --out-dir data/suno_ai/suno_themesong/song_001/analysis \
  --force-key C \
  --tempo 120
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

def run_cmd(cmd: List[str], description: str) -> None:
    """コマンド実行（失敗時はエラー終了）"""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}", file=sys.stderr)
        sys.exit(1)
    print(f"✅ SUCCESS: {description}")


def merge_chordmap_sections(
    chordmap_path: Path,
    sections_path: Path,
    out_path: Path,
    tempo: float = 120.0
) -> None:
    """chordmap_unified.json + sections.json → modular_composer形式（sections構造）"""
    
    # Load chordmap
    with open(chordmap_path, "r", encoding="utf-8") as f:
        chordmap = json.load(f)
    
    # Load sections
    with open(sections_path, "r", encoding="utf-8") as f:
        sections_data = json.load(f)
    
    # sections.jsonの形式を確認
    sections_list = sections_data.get("sections", [])
    if not sections_list:
        print("⚠️  Warning: sections.json is empty, creating default intro section")
        sections_list = [{"bar": 0, "label": "intro"}]
    
    # unit変換（bar → sec）
    sec_per_beat = 60.0 / tempo
    sec_per_bar = sec_per_beat * 4  # 4/4想定
    
    # セクション構造を構築
    modular_sections = {}
    events = chordmap.get("events", [])
    
    for i, sec_info in enumerate(sections_list):
        bar_start = sec_info["bar"]
        label = sec_info["label"]
        
        # 次のセクションのbar（なければ最後まで）
        if i + 1 < len(sections_list):
            bar_end = sections_list[i + 1]["bar"]
        else:
            # 最後のセクション：全イベント終了まで
            bar_end = float('inf')
        
        # このセクションに属するコードイベントを抽出
        sec_events = []
        for evt in events:
            # イベントの時刻（秒）→ bar換算
            evt_time_sec = evt.get("time", 0.0)
            evt_bar = evt_time_sec / sec_per_bar
            
            if bar_start <= evt_bar < bar_end:
                # セクション内相対時刻に変換（QL単位）
                relative_sec = evt_time_sec - (bar_start * sec_per_bar)
                relative_ql = relative_sec / sec_per_beat
                
                sec_events.append({
                    "chord": evt.get("chord", "N"),
                    "absolute_offset_beats": evt.get("time_ql", relative_ql),  # modular_composerはbeats単位期待
                    "confidence": evt.get("confidence", 0.8),
                })
        
        if not sec_events:
            # 空セクション：デフォルトコード挿入
            sec_events.append({
                "chord": "C",
                "absolute_offset_beats": 0.0,
                "confidence": 0.5,
            })
        
        # セクション追加
        modular_sections[label] = {
            "chords": sec_events,
            "time_signature": "4/4",
            "tempo": tempo,
        }
    
    # modular_composer形式で出力
    output = {
        "sections": modular_sections,
        "global_tempo": tempo,
        "time_signature": "4/4",
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Merged chordmap: {out_path}")
    print(f"   Sections: {len(modular_sections)}")
    for name, data in modular_sections.items():
        print(f"   - {name}: {len(data['chords'])} chords")


def main():
    ap = argparse.ArgumentParser(description="Auto Stage1 Pipeline: stems → sections/chordmap/anchors")
    ap.add_argument("--stems", required=True, help="Stem WAV directory")
    ap.add_argument("--vocal", required=True, help="Vocal WAV file path")
    ap.add_argument("--lyrics", required=True, help="Lyrics text file")
    ap.add_argument("--out-dir", required=True, help="Output directory for analysis files")
    ap.add_argument("--force-key", default="C", help="Force key for chord detection")
    ap.add_argument("--tempo", type=float, default=120.0, help="Tempo (BPM)")
    ap.add_argument("--exclude", action="append", default=[], help="Stems to exclude (e.g., Vocals)")
    ap.add_argument("--min-dwell-ql", type=float, default=0.5, help="Minimum chord duration (QL)")
    ap.add_argument("--skip-sections", action="store_true", help="Skip sections_from_audio (use existing)")
    ap.add_argument("--skip-chordmap", action="store_true", help="Skip chordmap generation (use existing)")
    ap.add_argument("--skip-anchors", action="store_true", help="Skip anchors generation (use existing)")
    args = ap.parse_args()
    
    stems_dir = Path(args.stems)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sections_path = out_dir / "sections.json"
    chordmap_raw_path = out_dir / "chordmap.json"
    chordmap_unified_path = out_dir / "chordmap_unified.json"
    anchors_path = out_dir / "lyric_anchors.json"
    chordmap_modular_path = out_dir / "chordmap_modular.json"
    
    # デフォルト除外リスト
    exclude_list = args.exclude or ["Vocals", "Backing Vocals"]
    
    # Step 1: sections_from_audio
    if not args.skip_sections:
        cmd = [
            sys.executable, "ops/sections_from_audio.py",
            "--stems", str(stems_dir),
            "--out", str(sections_path),
            "--ts-num", "4",
            "--min-bars", "4",
            "--max-sections", "12",
        ]
        for ex in exclude_list:
            cmd.extend(["--exclude", ex])
        run_cmd(cmd, "Step 1: Auto section detection")
    else:
        print(f"⏭️  Skipping sections detection (using {sections_path})")
    
    # Step 2: stem_harmony_7th_v2 (chordmap生成)
    if not args.skip_chordmap:
        cmd = [
            sys.executable, "ops/stem_harmony_7th_v2.py",
            "--stems", str(stems_dir),
            "--out", str(chordmap_raw_path),
            "--sections", str(sections_path),
            "--force-key", args.force_key,
            "--min-dwell-ql", str(args.min_dwell_ql),
            "--emit-confidence",
        ]
        for ex in exclude_list:
            cmd.extend(["--exclude", ex])
        run_cmd(cmd, "Step 2: Chordmap generation (v4.1)")
    else:
        print(f"⏭️  Skipping chordmap generation (using {chordmap_raw_path})")
    
    # Step 3: chordmap_unify (秒単位統一化)
    if not args.skip_chordmap:
        cmd = [
            sys.executable, "ops/chordmap_unify.py",
            "--input", str(chordmap_raw_path),
            "--output", str(chordmap_unified_path),
            "--to-unit", "sec",
        ]
        run_cmd(cmd, "Step 3: Chordmap unification (to sec)")
    else:
        print(f"⏭️  Skipping chordmap unification (using {chordmap_unified_path})")
    
    # Step 4: anchors_from_vocal (lyric_anchors生成)
    if not args.skip_anchors:
        cmd = [
            sys.executable, "ops/anchors_from_vocal.py",
            "--vocal", args.vocal,
            "--lyrics", args.lyrics,
            "--sections", str(sections_path),
            "--out", str(anchors_path),
            "--window-mode", "class",
            "--sibilant-scale", "1.2",
        ]
        run_cmd(cmd, "Step 4: Lyric anchors generation (Phase 23)")
    else:
        print(f"⏭️  Skipping anchors generation (using {anchors_path})")
    
    # Step 5: JSON → modular_composer形式（sections統合）
    merge_chordmap_sections(
        chordmap_unified_path,
        sections_path,
        chordmap_modular_path,
        tempo=args.tempo
    )
    
    print(f"\n{'='*60}")
    print("🎉 Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Output files in: {out_dir}")
    print(f"  - sections.json")
    print(f"  - chordmap.json (raw)")
    print(f"  - chordmap_unified.json (sec)")
    print(f"  - chordmap_modular.json (modular_composer ready)")
    print(f"  - lyric_anchors.json (Phase 23)")
    print(f"\nNext: Run modular_composer with:")
    print(f"  python modular_composer.py \\")
    print(f"    --main-cfg config/main_cfg.yml \\")
    print(f"    --chordmap {chordmap_modular_path} \\")
    print(f"    --output-dir output/")


if __name__ == "__main__":
    main()
