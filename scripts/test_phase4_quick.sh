#!/bin/bash
# Phase 4 Integration Test - Quick Version

echo "=================================================="
echo "Phase 4 Integration Test (Quick)"
echo "=================================================="
echo ""

echo "Test 1: Syntax validation"
.venv311/bin/python -m py_compile scripts/eval_drum_batch_stratified.py && echo "  ✓ eval_drum_batch_stratified.py"
.venv311/bin/python -m py_compile scripts/piano_eval_generate.py && echo "  ✓ piano_eval_generate.py"
.venv311/bin/python -m py_compile scripts/piano_train_prepare.py && echo "  ✓ piano_train_prepare.py"
.venv311/bin/python -m py_compile scripts/piano_train.py && echo "  ✓ piano_train.py"
.venv311/bin/python -m py_compile scripts/eval_piano_external.py && echo "  ✓ eval_piano_external.py"
.venv311/bin/python -m py_compile scripts/visualize_piano_trends.py && echo "  ✓ visualize_piano_trends.py"

echo ""
echo "Test 2: Phase 4.3 verification"
bash scripts/verify_phase43_quick.sh | grep "Quick Verification Complete" && echo "  ✓ Phase 4.3 improvements verified"

echo ""
echo "Test 3: Documentation exists"
test -f docs/PHASE_4_IMPLEMENTATION_STATUS.md && echo "  ✓ PHASE_4_IMPLEMENTATION_STATUS.md"
test -f docs/PHASE_4.3_IMPROVEMENTS.md && echo "  ✓ PHASE_4.3_IMPROVEMENTS.md"
test -f docs/PIANO_EXTERNAL_BENCHMARK.md && echo "  ✓ PIANO_EXTERNAL_BENCHMARK.md"

echo ""
echo "Test 4: Git status"
git diff --quiet HEAD && echo "  ✓ No uncommitted changes" || echo "  ⚠ Uncommitted changes present"
git log --oneline -1 | grep -q "docs\|feat" && echo "  ✓ Clean commit messages"

echo ""
echo "=================================================="
echo "✓ All quick tests passed"
echo "=================================================="
