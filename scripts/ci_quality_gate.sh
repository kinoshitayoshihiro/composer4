#!/bin/bash
# CI Quality Gate Integration
# Phase 4.6: Enforce quality gates in CI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "CI Quality Gate Check (Phase 4.6)"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0
WARNINGS=0

# Function to check instrument quality gate
check_instrument_gate() {
    local instrument=$1
    local eval_script=$2
    local output_json=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Checking: $instrument"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if evaluation output exists
    if [ ! -f "$output_json" ]; then
        echo -e "${YELLOW}⚠️  No evaluation output found: $output_json${NC}"
        echo "   Skipping gate check for $instrument"
        ((WARNINGS++))
        return 0
    fi
    
    # Run quality gate check
    if python scripts/quality_gate_checker.py --check "$instrument" --json "$output_json"; then
        echo -e "${GREEN}✅ $instrument: Quality gate PASSED${NC}"
    else
        echo -e "${RED}❌ $instrument: Quality gate FAILED${NC}"
        ((FAILED++))
        
        # Extract threshold_flags if available
        if command -v jq &> /dev/null; then
            FLAGS=$(jq -r '.threshold_flags // [] | join(", ")' "$output_json" 2>/dev/null || echo "")
            if [ -n "$FLAGS" ] && [ "$FLAGS" != "" ]; then
                echo -e "${RED}   Violations: $FLAGS${NC}"
            fi
        fi
    fi
    echo ""
}

# Check Piano (Phase 4.3 complete)
if [ -f "output/reports/piano_external_bench_latest.json" ]; then
    check_instrument_gate "piano" \
        "scripts/eval_piano_external.py" \
        "output/reports/piano_external_bench_latest.json"
fi

# Check Guitar (if eval exists)
if [ -f "output/reports/guitar_eval_latest.json" ]; then
    check_instrument_gate "guitar" \
        "scripts/eval_guitar.py" \
        "output/reports/guitar_eval_latest.json"
fi

# Check Drums (eval exists)
if [ -f "output/reports/drum_eval_latest.json" ]; then
    check_instrument_gate "drums" \
        "scripts/eval_drum_batch_stratified.py" \
        "output/reports/drum_eval_latest.json"
fi

# Check Bass (Phase 4.6 - NEW)
if [ -f "output/reports/bass_eval_latest.json" ]; then
    check_instrument_gate "bass" \
        "scripts/eval_bass.py" \
        "output/reports/bass_eval_latest.json"
fi

# Check Strings (Phase 4.6 - NEW)
if [ -f "output/reports/strings_eval_latest.json" ]; then
    check_instrument_gate "strings" \
        "scripts/eval_strings.py" \
        "output/reports/strings_eval_latest.json"
fi

# Check Bass (if eval exists)
if [ -f "output/reports/bass_eval_latest.json" ]; then
    check_instrument_gate "bass" \
        "scripts/eval_bass.py" \
        "output/reports/bass_eval_latest.json"
fi

# Check Strings (if eval exists)
if [ -f "output/reports/strings_eval_latest.json" ]; then
    check_instrument_gate "strings" \
        "scripts/eval_strings.py" \
        "output/reports/strings_eval_latest.json"
fi

echo "============================================================"
echo "CI Quality Gate Summary"
echo "============================================================"
echo ""
echo "  Failed: $FAILED"
echo "  Warnings: $WARNINGS"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ CI FAILED: $FAILED quality gate(s) failed${NC}"
    echo ""
    echo "💡 Next steps:"
    echo "   1. Review threshold_flags in evaluation JSONs"
    echo "   2. Check config/quality_gates.yaml for thresholds"
    echo "   3. Adjust generator parameters or gate thresholds"
    echo ""
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  CI PASSED with warnings: $WARNINGS evaluation(s) skipped${NC}"
    exit 0
else
    echo -e "${GREEN}✅ CI PASSED: All quality gates passed${NC}"
    exit 0
fi
