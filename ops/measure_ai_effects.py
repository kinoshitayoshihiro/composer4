#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from collections import defaultdict

song_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])

print("="*60)
print("AI Effect Measurement")
print("="*60)

report = {"ai_technologies": {}}

# 1. EmotionAI
print("\n1. EmotionAI...")
emotion_result = {"instruments": {}}
for inst in ["bass", "guitar", "piano", "strings"]:
    plan_path = song_dir / f"{inst}_plan.json"
    if plan_path.exists():
        with open(plan_path) as f:
            plan = json.load(f)
        events = []
        if "tracks" in plan:
            for track in plan["tracks"]:
                if "events" in track:
                    events.extend(track["events"])
        if events:
            section_vels = defaultdict(list)
            for e in events:
                section_vels[e.get("section", "unknown")].append(e.get("velocity") or 80)
            section_means = {s: sum(vels)/len(vels) for s, vels in section_vels.items() if vels}
            if section_means:
                vel_range = max(section_means.values()) - min(section_means.values())
                emotion_result["instruments"][inst] = {
                    "velocity_range": round(vel_range, 1),
                    "detected": vel_range > 15
                }
detected = sum(1 for i in emotion_result["instruments"].values() if i["detected"])
total = len(emotion_result["instruments"])
emotion_result["summary"] = f"{detected}/{total}"
print(f"   EmotionAI: {detected}/{total} instruments")
report["ai_technologies"]["emotion_ai"] = emotion_result

# 2. Harmony AI
print("\n2. Harmony AI...")
if Path("usage_history.db").exists():
    import sqlite3
    conn = sqlite3.connect("usage_history.db")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM progression_preferences")
    learned = cur.fetchone()[0]
    conn.close()
    harmony_result = {"status": "active", "learned_prefs": learned}
    print(f"   Learned: {learned}")
else:
    harmony_result = {"status": "not_used"}
    print("   NOT USED")
report["ai_technologies"]["harmony_ai"] = harmony_result

# 3. Magenta
print("\n3. Magenta Groove...")
drums_path = song_dir / "drums_recommendations.json"
if drums_path.exists():
    with open(drums_path) as f:
        drums = json.load(f)
    patterns = [drums[k]["pattern"]["pattern_id"] for k in drums if k.startswith("bar_")]
    magenta_result = {
        "status": "active",
        "total_bars": len(patterns),
        "unique_patterns": len(set(patterns)),
        "diversity": round(len(set(patterns)) / len(patterns), 2)
    }
    print(f"   Patterns: {len(set(patterns))}/{len(patterns)}")
else:
    magenta_result = {"status": "not_found"}
    print("   NOT FOUND")
report["ai_technologies"]["magenta"] = magenta_result

# 4. RhythmAI
print("\n4. RhythmAI...")
rhythm_path = song_dir / "matches_rhythm.json"
if rhythm_path.exists():
    with open(rhythm_path) as f:
        matches = json.load(f)
    match_list = matches.get("matches", [])
    if match_list:
        top_score = max(m["score"] for m in match_list)
        rhythm_result = {
            "status": "active",
            "matches": len(match_list),
            "top_score": round(top_score, 3)
        }
        print(f"   Score: {top_score:.3f}, Matches: {len(match_list)}")
    else:
        rhythm_result = {"status": "no_matches"}
        print("   NO MATCHES")
else:
    rhythm_result = {"status": "not_found"}
    print("   NOT FOUND")
report["ai_technologies"]["rhythm_ai"] = rhythm_result

# 5. CREPE
print("\n5. CREPE...")
crepe_path = song_dir / "vocal_f0_crepe.parquet"
if crepe_path.exists():
    try:
        import pandas as pd
        df = pd.read_parquet(crepe_path)
        is_dummy = len(df) <= 10
        crepe_result = {
            "status": "dummy" if is_dummy else "real_data",
            "frames": len(df)
        }
        print(f"   Frames: {len(df)} ({'DUMMY' if is_dummy else 'REAL'})")
    except:
        crepe_result = {"status": "error"}
        print("   ERROR")
else:
    crepe_result = {"status": "not_found"}
    print("   NOT FOUND")
report["ai_technologies"]["crepe"] = crepe_result

# 6. Onsets-and-Frames
print("\n6. Onsets-and-Frames...")
oaf_path = song_dir / "piano_oaf.json"
if oaf_path.exists():
    with open(oaf_path) as f:
        oaf = json.load(f)
    notes = len(oaf.get("notes", []))
    oaf_result = {"status": "active", "notes": notes}
    print(f"   Notes: {notes}")
else:
    oaf_result = {"status": "not_found"}
    print("   NOT FOUND")
report["ai_technologies"]["onsets_frames"] = oaf_result

# Summary
active = sum(1 for ai in report["ai_technologies"].values() 
             if ai.get("status") in ["active", "real_data"] or "detected" in str(ai))
total = len(report["ai_technologies"])
report["summary"] = {"active": f"{active}/{total}", "rate": round(active/total, 2)}

with open(out_file, 'w') as f:
    json.dump(report, f, indent=2)

print("\n"+"="*60)
print(f"Active AI: {active}/{total} ({report['summary']['rate']*100:.0f}%)")
print(f"Saved: {out_file}")
