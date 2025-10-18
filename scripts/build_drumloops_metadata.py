#!/usr/bin/env python3
"""
【非推奨・レガシー】
クリーニング済みMIDIからメタデータpickleを生成

⚠️ このスクリプトは非推奨です ⚠️

推奨方法:
  clean_midi.py の --pickle-out オプションを使用してください。
  clean_midi.py が直接 Stage2互換の sharded pickle を生成します。

Usage (推奨):
  python -m scripts.clean_midi \
    --in data/loops \
    --out output/drumloops_v3 \
    --quarantine output/drumloops_v3_q \
    --instrument drums \
    --pickle-out output/drums_metadata \
    --shard-size 5000 \
    --emit-meta-json off \
    --resume \
    --jobs 8

このスクリプトは後方互換性のためのみ残されています。
"""

import sys
import pickle
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone

sys.path.append("data/Los-Angeles-MIDI/CODE")
import TMIDIX

print("⚠️  警告: このスクリプトは非推奨です")
print("   推奨: clean_midi.py --pickle-out を使用してください")
print("")

def build_metadata_from_cleaned(input_dir: str, output_pickle: str):
    """クリーニング済みMIDIからメタデータpickleを生成"""
    
    input_path = Path(input_dir)
    midi_files = sorted(list(input_path.rglob("*.mid")) + list(input_path.rglob("*.midi")))
    
    print(f"🎵 Found {len(midi_files)} MIDI files in {input_dir}")
    print(f"📦 Building metadata pickle...")
    
    loops = []
    
    for midi_path in tqdm(midi_files, desc="Processing"):
        try:
            # TMIDIXでMIDIをパース
            score = TMIDIX.Tegridy_MIDI_Processor(str(midi_path)).MIDI_Score
            
            # 基本情報抽出
            ticks_per_beat = score[0]
            
            # ノート抽出
            notes = []
            pitches = []
            for track in score[7]:
                for event in track:
                    if event[0] == 'note':
                        notes.append(event)
                        pitches.append(event[4])
            
            # メタデータ構築
            loop_meta = {
                'md5': midi_path.stem,  # ファイル名をIDとして使用
                'filename': midi_path.name,
                'input_path': str(midi_path.relative_to(input_path.parent)),
                'output_path': str(midi_path.relative_to(input_path.parent)),
                'genre': 'unknown',  # ファイル名から抽出可能なら後で改善
                'bpm': 120,  # デフォルト（後でTMIDIXから抽出可能）
                'note_count': len(notes),
                'duration_ticks': score[1] if len(score) > 1 else 0,
                'pitches': list(set(pitches)),
                'metrics': {}  # Stage2で計算
            }
            
            loops.append(loop_meta)
            
        except Exception as e:
            print(f"\n⚠️  Error processing {midi_path.name}: {e}")
            continue
    
    # Pickle保存
    metadata = {
        'version': '3.0',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'config': {
            'source': 'drumloops_v3_cleaned',
            'instrument': 'drums'
        },
        'shard_index': 0,
        'loop_count': len(loops),
        'summary': {
            'total_files': len(midi_files),
            'processed': len(loops),
            'failed': len(midi_files) - len(loops)
        },
        'loops': loops
    }
    
    output_path = Path(output_pickle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✅ Metadata pickle saved: {output_pickle}")
    print(f"📊 Total loops: {len(loops)}")
    print(f"   Processed: {len(loops)}")
    print(f"   Failed: {len(midi_files) - len(loops)}")

if __name__ == "__main__":
    build_metadata_from_cleaned(
        input_dir="output/drumloops_v3_test",
        output_pickle="output/drumloops_v3_metadata/drumloops_v3_metadata.pickle"
    )
