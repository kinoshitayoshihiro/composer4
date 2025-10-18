
#!/bin/bash
# POP909 簡易クリーニングスクリプト
# 大量のファイルを処理するため、findでファイルリストを事前作成

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "${BASE_DIR}"

echo "🔍 Finding MIDI files in POP909..."
find data/POP909 -name "*.mid" -o -name "*.midi" > /tmp/pop909_files.txt
TOTAL=$(wc -l < /tmp/pop909_files.txt | tr -d ' ')

echo "📁 Found ${TOTAL} MIDI files"
echo ""

# ディレクトリ準備
mkdir -p data/cleaned/pop909
mkdir -p data/quarantine/pop909

# プログレス表示用
COUNTER=0

echo "🎹 Processing POP909 files..."
while IFS= read -r midi_file; do
    COUNTER=$((COUNTER + 1))
    
    # 100ファイルごとに進捗表示
    if [ $((COUNTER % 100)) -eq 0 ]; then
        PCT=$(awk "BEGIN {printf \"%.1f\", (${COUNTER}/${TOTAL})*100}")
        echo "  Progress: ${COUNTER}/${TOTAL} (${PCT}%)"
    fi
    
    # Python スクリプトを直接呼び出す（ファイル列挙をスキップ）
    python3 -c "
import sys
sys.path.append('scripts')
from pathlib import Path
from cleaners.common import common_clean, atomic_write_json
from cleaners.piano import clean_piano
import pretty_midi

midi_path = Path('${midi_file}')
output_dir = Path('data/cleaned/pop909')
quarantine_dir = Path('data/quarantine/pop909')

try:
    # 読み込み
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    
    # 共通クリーニング
    pm, meta_common = common_clean(pm)
    
    # ピアノクリーニング
    pm, meta_piano, reason_codes = clean_piano(pm)
    
    # メタデータ統合
    meta = {**meta_common, **meta_piano, 'reason_codes': reason_codes}
    
    # 隔離判定
    critical = ['note_count_low', 'invalid_midi']
    has_critical = any(c in reason_codes for c in critical)
    has_3_warnings = len([r for r in reason_codes if r not in critical]) >= 3
    should_quarantine = has_critical or has_3_warnings
    
    # 保存先決定
    relative_path = midi_path.relative_to(Path('data/POP909'))
    if should_quarantine:
        output_path = quarantine_dir / relative_path
    else:
        output_path = output_dir / relative_path
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    
    meta_path = output_path.parent / (midi_path.stem + '.meta.json')
    atomic_write_json(meta, meta_path)

except Exception as e:
    print(f'Error processing {midi_path}: {e}', file=sys.stderr)
"
    
done < /tmp/pop909_files.txt

echo ""
echo "✅ Processing complete!"
echo "📁 Cleaned: data/cleaned/pop909"
echo "🗑️ Quarantine: data/quarantine/pop909"

# 統計表示
CLEANED=$(find data/cleaned/pop909 -name "*.mid" | wc -l | tr -d ' ')
QUARANTINED=$(find data/quarantine/pop909 -name "*.mid" | wc -l | tr -d ' ')

echo ""
echo "📊 Statistics:"
echo "  Total: ${TOTAL}"
echo "  Cleaned: ${CLEANED}"
echo "  Quarantined: ${QUARANTINED}"
