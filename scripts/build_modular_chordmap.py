#!/usr/bin/env python3
"""
scripts/build_modular_chordmap.py - sections.json + chordmap.json → modular_composer用chordmapを生成
"""
import argparse
import json
from pathlib import Path


def build_modular_chordmap(sections_path, chordmap_path, output_path, tempo=120):
    """
    sections.json + chordmap_unified.json を統合して、
    modular_composer.py が期待する形式の chordmap を生成する。
    
    modular_composer期待形式:
    {
      "sections": {
        "intro": {
          "processed_chord_events": [
            {
              "absolute_offset_beats": 0.0,
              "original_offset_beats": 0.0,
              "humanized_offset_beats": 0.0,
              "original_duration_beats": 4.0,
              "humanized_duration_beats": 4.0,
              "original_chord_label": "Bm7",
              "chord_symbol_for_voicing": "Bm7",
              "specified_bass_for_voicing": null
            },
            ...
          ]
        },
        "verse_1": { ... },
        ...
      }
    }
    """
    # 1. sections.json 読み込み
    with open(sections_path, encoding="utf-8") as f:
        sections = json.load(f)
    
    # 2. chordmap.json 読み込み
    with open(chordmap_path, encoding="utf-8") as f:
        chordmap = json.load(f)
    
    unit = chordmap.get("unit", "sec")
    events = chordmap.get("events", [])
    
    # 3. 秒→拍変換（tempoベース）
    sec_per_beat = 60.0 / tempo
    
    def sec_to_beats(sec):
        return sec / sec_per_beat
    
    # 4. セクションごとにイベントを振り分け
    modular_sections = {}
    
    for sec_idx, section in enumerate(sections):
        sec_name = section["name"]
        sec_start = section["start"]  # 秒
        sec_end = section["end"]      # 秒
        
        sec_start_beats = sec_to_beats(sec_start)
        sec_end_beats = sec_to_beats(sec_end)
        
        # このセクション内のイベントを抽出
        sec_events = []
        for i, evt in enumerate(events):
            evt_time = evt["time"]
            evt_time_beats = sec_to_beats(evt_time)
            
            # セクション範囲内かチェック
            if sec_start <= evt_time < sec_end:
                # 次のイベントまたはセクション終端までの持続時間
                if i + 1 < len(events):
                    next_time = events[i + 1]["time"]
                    next_time_beats = sec_to_beats(next_time)
                    duration_beats = next_time_beats - evt_time_beats
                else:
                    duration_beats = sec_end_beats - evt_time_beats
                
                # コード表記（root + quality → "Bm7"形式）
                root = evt.get("root", "C")
                quality = evt.get("quality", "maj")
                
                # quality → suffix 変換
                quality_map = {
                    "maj": "",
                    "min": "m",
                    "maj7": "maj7",
                    "min7": "m7",
                    "dom7": "7",
                    "dim": "dim",
                    "aug": "aug",
                    "sus4": "sus4",
                    "sus2": "sus2",
                }
                suffix = quality_map.get(quality, "")
                chord_label = f"{root}{suffix}"
                
                # modular_composer形式のイベント
                processed_evt = {
                    "absolute_offset_beats": evt_time_beats,
                    "original_offset_beats": evt_time_beats,
                    "humanized_offset_beats": evt_time_beats,
                    "original_duration_beats": duration_beats,
                    "humanized_duration_beats": duration_beats,
                    "original_chord_label": chord_label,
                    "chord_symbol_for_voicing": chord_label,
                    "specified_bass_for_voicing": None,
                }
                sec_events.append(processed_evt)
        
        # セクションが空でない場合のみ追加
        if sec_events:
            modular_sections[sec_name] = {
                "processed_chord_events": sec_events
            }
    
    # 5. 出力
    modular_chordmap = {
        "sections": modular_sections
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(modular_chordmap, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Modular chordmap created: {output_path}")
    print(f"   Sections: {len(modular_sections)}")
    total_events = sum(len(s["processed_chord_events"]) for s in modular_sections.values())
    print(f"   Total events: {total_events}")
    
    return modular_chordmap


def main():
    parser = argparse.ArgumentParser(description="Build modular_composer-compatible chordmap")
    parser.add_argument("--sections", required=True, help="sections.json path")
    parser.add_argument("--chordmap", required=True, help="chordmap_unified.json path")
    parser.add_argument("--output", required=True, help="Output modular chordmap path")
    parser.add_argument("--tempo", type=int, default=120, help="Tempo (BPM) for sec→beats conversion")
    
    args = parser.parse_args()
    
    build_modular_chordmap(
        sections_path=args.sections,
        chordmap_path=args.chordmap,
        output_path=args.output,
        tempo=args.tempo
    )


if __name__ == "__main__":
    main()
