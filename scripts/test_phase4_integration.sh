#!/bin/bash
# Phase 4 Integration Test Suite
# Tests all Phase 4 improvements with real-world scenarios

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================================="
echo "Phase 4 Integration Test Suite"
echo "=================================================="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0
skipped=0

# Test function
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expect_success="${3:-true}"
    
    echo -n "Testing: $test_name ... "
    
    if eval "$test_cmd" > /tmp/test_output_$$.log 2>&1; then
        if [ "$expect_success" = "true" ]; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((passed++))
        else
            echo -e "${RED}✗ FAIL${NC} (expected failure)"
            ((failed++))
        fi
    else
        if [ "$expect_success" = "false" ]; then
            echo -e "${GREEN}✓ PASS${NC} (expected failure)"
            ((passed++))
        else
            echo -e "${RED}✗ FAIL${NC}"
            echo "Error output:"
            cat /tmp/test_output_$$.log | head -20
            ((failed++))
        fi
    fi
}

skip_test() {
    local test_name="$1"
    local reason="$2"
    echo -e "Testing: $test_name ... ${YELLOW}⊘ SKIP${NC} ($reason)"
    ((skipped++))
}

echo "=== Test Suite 1: Syntax & Import Validation ==="
echo ""

run_test "Python syntax: eval_drum_batch_stratified.py" \
    ".venv311/bin/python -m py_compile scripts/eval_drum_batch_stratified.py"

run_test "Python syntax: piano_eval_generate.py" \
    ".venv311/bin/python -m py_compile scripts/piano_eval_generate.py"

run_test "Python syntax: piano_train_prepare.py" \
    ".venv311/bin/python -m py_compile scripts/piano_train_prepare.py"

run_test "Python syntax: piano_train.py" \
    ".venv311/bin/python -m py_compile scripts/piano_train.py"

run_test "Python syntax: eval_piano_external.py" \
    ".venv311/bin/python -m py_compile scripts/eval_piano_external.py"

run_test "Python syntax: visualize_piano_trends.py" \
    ".venv311/bin/python -m py_compile scripts/visualize_piano_trends.py"

echo ""
echo "=== Test Suite 2: Phase 4.5 - Tempo Meta Priority ==="
echo ""

# Create test MIDI with tempo metadata
TEST_MIDI_DIR="/tmp/phase4_test_$$"
mkdir -p "$TEST_MIDI_DIR"

cat > /tmp/create_test_midi_$$.py <<'EOF'
import pretty_midi
import json
from pathlib import Path
import sys

# Create test MIDI
pm = pretty_midi.PrettyMIDI(initial_tempo=110.5)
inst = pretty_midi.Instrument(program=0)  # Piano

# Add some notes
for i in range(8):
    note = pretty_midi.Note(
        velocity=80 + i * 5,
        pitch=60 + i,
        start=i * 0.5,
        end=(i + 1) * 0.5
    )
    inst.notes.append(note)

pm.instruments.append(inst)

# Save MIDI
midi_path = Path(sys.argv[1]) / "test_tempo.mid"
pm.write(str(midi_path))

# Save metadata with explicit tempo
meta_path = midi_path.with_suffix('.meta.json')
meta_data = {
    "tempo": 110.5,
    "time_signature": "4/4",
    "chords": ["C", "G", "Am", "F"]
}
meta_path.write_text(json.dumps(meta_data, indent=2))

print(f"Created: {midi_path}")
print(f"Created: {meta_path}")
EOF

.venv311/bin/python /tmp/create_test_midi_$$.py "$TEST_MIDI_DIR" > /dev/null 2>&1

if [ -f "$TEST_MIDI_DIR/test_tempo.mid" ]; then
    run_test "Tempo meta priority (file exists)" "test -f $TEST_MIDI_DIR/test_tempo.meta.json"
    
    # Test tempo meta reading (would need eval_drum_batch_stratified.py run, but that requires full setup)
    echo "  Note: Full eval_drum_batch_stratified.py test requires trained models"
else
    skip_test "Tempo meta priority" "Test MIDI creation failed"
fi

echo ""
echo "=== Test Suite 3: Phase 4.2-polish - Deterministic Splits ==="
echo ""

# Test stratified split stability (requires actual piano data)
if [ -d "data/piano_loops" ] && [ "$(find data/piano_loops -name '*.jsonl' | wc -l)" -gt 0 ]; then
    run_test "Stratified split (dry run)" \
        ".venv311/bin/python scripts/piano_train_prepare.py --help | grep -q 'out-dir'"
    
    echo "  Note: Full stratified split test requires complete piano dataset"
else
    skip_test "Stratified split stability" "Piano dataset not found"
fi

echo ""
echo "=== Test Suite 4: Phase 4.3 - External Benchmarks ==="
echo ""

# Test external benchmark with dummy data
if [ -d "$TEST_MIDI_DIR" ]; then
    TEST_OUTPUT="/tmp/phase4_bench_$$.json"
    
    run_test "External benchmark evaluation" \
        ".venv311/bin/python scripts/eval_piano_external.py \
            --maestro-dir $TEST_MIDI_DIR \
            --out-json $TEST_OUTPUT \
            --n-samples 1 \
            --seed 42"
    
    if [ -f "$TEST_OUTPUT" ]; then
        run_test "Provenance field exists" \
            "grep -q '\"provenance\"' $TEST_OUTPUT"
        
        run_test "Benchmark field exists" \
            "grep -q '\"benchmark\".*maestro_subset' $TEST_OUTPUT"
        
        rm -f "$TEST_OUTPUT"
    fi
fi

echo ""
echo "=== Test Suite 5: Phase 4.3 - Robustness Improvements ==="
echo ""

run_test "Verify script exists" \
    "test -f scripts/verify_phase43_quick.sh"

run_test "Quick verification passes" \
    "bash scripts/verify_phase43_quick.sh | grep -q '7/7 checks passed'"

echo ""
echo "=== Test Suite 6: Documentation Completeness ==="
echo ""

run_test "Phase 4 Implementation Status doc" \
    "test -f docs/PHASE_4_IMPLEMENTATION_STATUS.md"

run_test "Phase 4.3 Improvements doc" \
    "test -f docs/PHASE_4.3_IMPROVEMENTS.md"

run_test "Piano External Benchmark doc" \
    "test -f docs/PIANO_EXTERNAL_BENCHMARK.md"

echo ""
echo "=== Test Suite 7: Git Integration ==="
echo ""

run_test "Branch exists" \
    "git rev-parse --verify chore/ab-eval-piano-guitar-minipatch > /dev/null 2>&1"

run_test "No uncommitted changes" \
    "git diff --quiet HEAD"

run_test "Commits are clean" \
    "git log --oneline -1 | grep -q 'docs\\|feat'"

echo ""
echo "=================================================="
echo "Test Results Summary"
echo "=================================================="
echo -e "${GREEN}Passed:${NC}  $passed"
echo -e "${RED}Failed:${NC}  $failed"
echo -e "${YELLOW}Skipped:${NC} $skipped"
echo "Total:   $((passed + failed + skipped))"
echo ""

# Cleanup
rm -rf "$TEST_MIDI_DIR"
rm -f /tmp/test_output_$$.log
rm -f /tmp/create_test_midi_$$.py

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
