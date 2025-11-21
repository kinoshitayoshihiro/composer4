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
ARTIFACT_ERRORS=0

SONG_DIRS=()
if [[ $# -gt 0 ]]; then
    for arg in "$@"; do
        SONG_DIR="${arg%/}"
        SONG_DIRS+=("$SONG_DIR")
    done
else
    if [[ -d "$PROJECT_ROOT/song_packages" ]]; then
        while IFS= read -r path; do
            SONG_DIRS+=("$path")
        done < <(find "$PROJECT_ROOT/song_packages" -maxdepth 2 -type d -name 'song_*' 2>/dev/null | sort)
    fi
fi

require_path_for_song() {
    local song_dir="$1"
    local label="$2"
    shift 2
    local candidates=("$@")
    local found_path=""
    for rel in "${candidates[@]}"; do
        if [[ -s "$song_dir/$rel" ]]; then
            found_path="$song_dir/$rel"
            break
        fi
    done
    if [[ -n "$found_path" ]]; then
        echo -e "${GREEN}   ✅ ${label}: ${found_path}${NC}"
        return 0
    else
        echo -e "${RED}   ❌ Missing ${label} (searched: ${candidates[*]})${NC}"
        return 1
    fi
}

require_plan_for_song() {
    local song_dir="$1"
    local plan_name="$2"
    local match
    match=$(find "$song_dir" -maxdepth 2 -name "$plan_name" -print -quit 2>/dev/null)
    if [[ -n "$match" ]]; then
        echo -e "${GREEN}   ✅ ${plan_name}: ${match}${NC}"
        return 0
    fi
    echo -e "${RED}   ❌ Missing ${plan_name} under $song_dir${NC}"
    return 1
}

check_song_artifacts() {
    local song_dir="$1"
    local rel_path="${song_dir#$PROJECT_ROOT/}"
    local local_failed=0
    [[ -z "$rel_path" ]] && rel_path="$song_dir"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Composer4 artifact check: $rel_path"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    require_path_for_song "$song_dir" "tempo_map.json" "analysis/tempo_map.json" "tempo_map.json" || local_failed=1
    require_path_for_song "$song_dir" "bars_with_slots.parquet" "analysis/bars_with_slots.parquet" "bars_with_slots.parquet" || local_failed=1
    require_plan_for_song "$song_dir" "drums_plan.json" || local_failed=1
    require_plan_for_song "$song_dir" "bass_plan.json" || local_failed=1
    require_plan_for_song "$song_dir" "guitar_plan.json" || local_failed=1
    require_plan_for_song "$song_dir" "piano_plan.json" || local_failed=1
    require_plan_for_song "$song_dir" "strings_plan.json" || local_failed=1
    echo

    if [[ $local_failed -ne 0 ]]; then
        ((ARTIFACT_ERRORS++))
    fi
}

if [[ ${#SONG_DIRS[@]} -gt 0 ]]; then
    echo "============================================================"
    echo "Composer4 Artifact Sanity"
    echo "============================================================"
    echo ""
    for dir in "${SONG_DIRS[@]}"; do
        [[ -d "$dir" ]] || continue
        check_song_artifacts "$dir"
    done
fi

if [[ $ARTIFACT_ERRORS -gt 0 ]]; then
    echo -e "${RED}❌ CI FAILED: Missing composer4 artifacts ($ARTIFACT_ERRORS song package(s))${NC}"
    exit 1
fi

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
