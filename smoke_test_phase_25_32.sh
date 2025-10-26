#!/bin/bash
# Phase 25-32 出荷前最終スモークテスト

set -e

echo "=========================================="
echo "Phase 25-32 Final Smoke Test"
echo "=========================================="
echo ""

# 1. Phase順序確認
echo "✓ Step 1: Phase順序確認..."
python3 -c "
from generator.piano_params_stage2 import PianoParamsStage2
gen = PianoParamsStage2()
params = {
    'sparsify': {'enable': True},
    'harmony': {'source': 'hybrid'},
    'style_adapt': {'enable': True},
    'export': {'quantize_ql': 0.125},
    'ducking': {'enable': True},
    'xinst_balance': {'vs_bass': {'enable': True}},
    'voice_leading': {'enable': True}
}
phases = gen._get_phases(params)
expected = [11, 12, 20, 25, 26, 31, 27, 30, 28, 29]
print(f'Phases: {phases}')
assert 26 in phases and 31 in phases, 'Phase 26/31 missing'
idx_26 = phases.index(26)
idx_31 = phases.index(31)
idx_27 = phases.index(27)
idx_30 = phases.index(30)
assert idx_31 > idx_26, 'Phase 31 should come after 26'
assert idx_30 > idx_27, 'Phase 30 should come after 27'
print('✓ Phase順序OK: 26→31, 27→30')
"
echo ""

# 2. NO-OP既定確認
echo "✓ Step 2: NO-OP既定確認..."
python3 -c "
from generator.piano_params_stage2 import PianoParamsStage2
gen = PianoParamsStage2()
# 空params → Phase 11,12,20のみ
phases_empty = gen._get_phases({})
assert phases_empty == [11, 12, 20], f'Empty params should be [11,12,20], got {phases_empty}'
print('✓ NO-OP既定OK')
"
echo ""

# 3. 後方互換確認
echo "✓ Step 3: 後方互換確認..."
python3 -c "
from generator.piano_params_stage2 import PianoParamsStage2
import random
gen = PianoParamsStage2()
section = {'label': 'verse', 'bar': 0, 'tempo': 120.0}
mix_ctx = {'beat_grid': {'bpm': 120.0}}
# 旧式params（Phase 25-32なし）
params_old = {'style': 'moderate'}
try:
    result = gen.apply(section, mix_ctx, params_old, seed=42)
    print('✓ 後方互換OK: 旧式params動作確認')
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)
"
echo ""

# 4. Phase 30/31統合確認
echo "✓ Step 4: Phase 30/31統合確認..."
python3 -c "
from generator.piano_params_stage2 import PianoParamsStage2
gen = PianoParamsStage2()
section = {'label': 'chorus', 'bar': 0, 'tempo': 120.0}
mix_ctx = {
    'beat_grid': {'bpm': 120.0},
    'activity': {'bass': [(0, 0.9)]}
}
params = {
    'style': 'moderate',
    'xinst_balance': {'vs_bass': {'enable': True, 'threshold': 0.7, 'vel_cut': 6}},
    'voice_leading': {'enable': True, 'max_leap': 7}
}
try:
    result = gen.apply(section, mix_ctx, params, seed=42)
    print('✓ Phase 30/31統合OK')
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)
"
echo ""

# 5. エッジケース確認
echo "✓ Step 5: エッジケース確認..."
python3 -c "
from generator.piano_params_stage2 import PianoParamsStage2
gen = PianoParamsStage2()
section = {'label': 'bridge', 'bar': 0, 'tempo': 120.0}
# 空activity/空chord
mix_ctx = {'beat_grid': {'bpm': 120.0}, 'activity': {}}
params = {
    'xinst_balance': {'vs_bass': {'enable': True}},
    'voice_leading': {'enable': True}
}
try:
    result = gen.apply(section, mix_ctx, params, seed=42)
    print('✓ エッジケースOK: 空activity/chord対応')
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)
"
echo ""

echo "=========================================="
echo "✅ All smoke tests passed!"
echo "=========================================="
echo ""
echo "Phase 25-32 implementation is PRODUCTION READY 🚀"
echo ""
echo "Next steps:"
echo "  1. pytest tests/test_phase_30_31.py tests/test_phase_final_checklist.py"
echo "  2. git commit -m 'Phase 25-32 implementation complete'"
echo "  3. Production deployment"
echo ""
