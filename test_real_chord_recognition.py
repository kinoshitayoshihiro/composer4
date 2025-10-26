#!/usr/bin/env python3
"""
Real Chord Recognition Test
実際のSuno AIステムWAVでchord recognitionをテスト
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analysis.stem_harmony import (
    make_beat_grid,
    estimate_chords_per_stem,
    aggregate_stem_chords,
    estimate_activity
)

def test_real_stems():
    """実際のステムWAVでテスト"""
    
    stem_dir = Path("/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/suno_ai/suno_themesong/song_001/stemswav_001")
    
    # ステムファイルマッピング
    stem_files = {
        "bass": stem_dir / "stem_wav_001_(Bass).wav",
        "guitar": stem_dir / "stem_wav_001_(Guitar).wav",
        "piano": stem_dir / "stem_wav_001_(Keyboard).wav",
        "drums": stem_dir / "stem_wav_001_(Drums).wav",
    }
    
    print("=" * 80)
    print("Chord Recognition System Test - Real Suno AI Stems")
    print("=" * 80)
    print()
    
    # 存在確認
    for role, path in stem_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {role:8s}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {role:8s}: NOT FOUND")
    
    print()
    
    # Step 1: Beat Grid作成（Bass WAVから）
    print("-" * 80)
    print("Step 1: Creating Beat Grid...")
    print("-" * 80)
    
    # Bassが最も信頼できるので、beat grid作成に使用
    bass_wav = str(stem_files["bass"])
    
    # ダミーstems dict（make_beat_gridが必要とする形式）
    stems_dict = {"bass": bass_wav}
    
    try:
        beat_grid = make_beat_grid(
            stems_dict,
            default_bpm=120.0,  # 仮のBPM（自動検出される）
            time_sig=(4, 4)
        )
        print(f"✅ BPM: {beat_grid.bpm:.1f}")
        print(f"✅ Time Signature: {beat_grid.time_sig[0]}/{beat_grid.time_sig[1]}")
        print(f"✅ Duration: {beat_grid.duration_ql:.2f} quarter notes")
        print(f"✅ Bars: {len(beat_grid.bars)}")
        print(f"✅ Beats: {len(beat_grid.beats)}")
        print()
    except Exception as e:
        print(f"❌ Error creating beat grid: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: 各ステムからChord推定
    print("-" * 80)
    print("Step 2: Estimating Chords from Each Stem...")
    print("-" * 80)
    print()
    
    all_votes = {}
    all_activities = {}
    
    for role, wav_path in stem_files.items():
        if not wav_path.exists():
            continue
        
        print(f"Processing {role}...")
        
        try:
            # Activity推定
            activity = estimate_activity(str(wav_path), beat_grid)
            all_activities[role] = activity
            avg_activity = sum(a[1] for a in activity) / len(activity) if activity else 0
            print(f"  ✅ Activity: {avg_activity:.2%} (avg)")
            
            # Chord推定（key_hintはオプション）
            votes = estimate_chords_per_stem(
                str(wav_path),
                beat_grid,
                role=role,
                key_hint="C:maj"  # 仮のキー（または自動検出）
            )
            all_votes[role] = votes
            
            # 最初の4小節のコードを表示
            print(f"  ✅ Chords detected: {len(votes)} beat positions")
            print(f"  📊 First 4 bars:")
            
            bar_chords = {}
            for (bar_idx, beat_in_bar), candidates in votes.items():
                if bar_idx >= 4:
                    continue
                if bar_idx not in bar_chords:
                    bar_chords[bar_idx] = []
                if candidates:
                    chord, conf = candidates[0]
                    bar_chords[bar_idx].append(f"{chord}({conf:.0%})")
            
            for bar_idx in sorted(bar_chords.keys()):
                chords_str = " | ".join(bar_chords[bar_idx][:4])  # 最初4拍のみ
                print(f"     Bar {bar_idx}: {chords_str}")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing {role}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Step 3: 複数ステム投票統合
    print("-" * 80)
    print("Step 3: Aggregating Multi-Stem Votes...")
    print("-" * 80)
    print()
    
    try:
        # Bass activityを使用（最も信頼できる）
        bass_activity = all_activities.get("bass", [])
        
        final_chords = aggregate_stem_chords(
            all_votes,
            bass_activity,
            beat_grid
        )
        
        print(f"✅ Final chord sequence: {len(final_chords)} positions")
        print()
        print("📊 Final Chordmap (first 8 bars):")
        print()
        
        bar_final = {}
        for (bar_idx, beat_in_bar), (chord, conf, sources) in final_chords.items():
            if bar_idx >= 8:
                continue
            if bar_idx not in bar_final:
                bar_final[bar_idx] = []
            
            sources_str = "+".join(sources[:2])  # 最初2つのソース
            bar_final[bar_idx].append(f"{chord}({conf:.0%},{sources_str})")
        
        for bar_idx in sorted(bar_final.keys()):
            chords_str = " | ".join(bar_final[bar_idx][:4])  # 最初4拍
            print(f"  Bar {bar_idx:2d}: {chords_str}")
        
        print()
        print("=" * 80)
        print("✅ Chord Recognition Test Completed!")
        print("=" * 80)
        
        # 統計情報
        confidences = [conf for _, (_, conf, _) in final_chords.items()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        low_conf_count = sum(1 for c in confidences if c < 0.5)
        
        print()
        print("📈 Statistics:")
        print(f"  Average Confidence: {avg_conf:.1%}")
        print(f"  Low Confidence (<50%): {low_conf_count}/{len(confidences)}")
        print(f"  Stems Used: {len(all_votes)}")
        print()
        
    except Exception as e:
        print(f"❌ Error aggregating chords: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_stems()
