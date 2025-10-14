#!/bin/bash
# Nightly CI: Piano Transformer External Benchmark Evaluation
#
# This script runs Piano Transformer evaluation on MAESTRO subset
# and saves results for trend visualization.
#
# Usage:
#   bash scripts/run_piano_external_bench.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
MAESTRO_DIR="${MAESTRO_DIR:-data/maestro_subset}"
MODEL_DIR="${MODEL_DIR:-models/piano_transformer/best}"
OUT_DIR="${OUT_DIR:-output/reports}"
N_SAMPLES="${N_SAMPLES:-10}"
SEED="${SEED:-42}"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# Timestamped output
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_JSON="${OUT_DIR}/piano_external_bench_${TIMESTAMP}.json"

echo "=================================================="
echo "Piano Transformer External Benchmark (Nightly CI)"
echo "=================================================="
echo "MAESTRO Dir: $MAESTRO_DIR"
echo "Model Dir:   $MODEL_DIR"
echo "Output:      $OUT_JSON"
echo "N Samples:   $N_SAMPLES"
echo "Seed:        $SEED"
echo ""

# Check MAESTRO directory
if [ ! -d "$MAESTRO_DIR" ]; then
    echo "[warn] MAESTRO directory not found: $MAESTRO_DIR"
    echo "[warn] Skipping external benchmark evaluation"
    echo "[info] To enable, download MAESTRO subset:"
    echo "       1. Download from https://magenta.tensorflow.org/datasets/maestro"
    echo "       2. Extract 10-20 MIDI files to $MAESTRO_DIR"
    exit 0
fi

# Count MIDI files
MIDI_COUNT=$(find "$MAESTRO_DIR" -type f \( -name "*.mid" -o -name "*.midi" \) | wc -l | tr -d ' ')
echo "[info] Found $MIDI_COUNT MIDI files in MAESTRO subset"

if [ "$MIDI_COUNT" -eq 0 ]; then
    echo "[warn] No MIDI files found, skipping evaluation"
    exit 0
fi

# Run evaluation
echo ""
echo "[info] Running external benchmark evaluation..."
python3 "$SCRIPT_DIR/eval_piano_external.py" \
    --maestro-dir "$MAESTRO_DIR" \
    --out-json "$OUT_JSON" \
    --n-samples "$N_SAMPLES" \
    --seed "$SEED"

# Create symlink to latest (use absolute path to avoid broken symlinks)
LATEST_LINK="${OUT_DIR}/piano_external_bench_latest.json"
ln -sfn "$OUT_JSON" "$LATEST_LINK"
echo "[info] Latest result linked: $LATEST_LINK"

# Append to history (for trend visualization)
HISTORY_FILE="${OUT_DIR}/piano_external_bench_history.jsonl"
if [ -f "$OUT_JSON" ]; then
    # Extract summary and add timestamp
    python3 -c "
import json
from pathlib import Path
from datetime import datetime

out_json = Path('$OUT_JSON')
history_file = Path('$HISTORY_FILE')

data = json.loads(out_json.read_text())
entry = {
    'timestamp': datetime.now().isoformat(),
    'date': datetime.now().strftime('%Y-%m-%d'),
    'summary': data.get('summary', {}),
    'n_samples': data.get('n_samples', 0),
    'provenance': data.get('provenance', {}),
    'schema_version': data.get('schema_version', ''),
    'fileset_hash': data.get('fileset_hash', ''),
    'threshold_flags': data.get('threshold_flags', []),
}

with open(history_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f'[info] Appended to history: {history_file}')
"
fi

echo ""
echo "[done] External benchmark evaluation complete"
echo "       Results: $OUT_JSON"
echo "       History: $HISTORY_FILE"
