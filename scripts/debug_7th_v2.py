#!/usr/bin/env python3
"""Test 7th v2 version"""
import sys
sys.path.insert(0, ".")

try:
    from ops.stem_harmony_7th_v2 import build_loglik_7th_enhanced, estimate_local_key_7th
    print("✓ Functions imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

import numpy as np

# Test estimate_local_key_7th
C_sync = np.random.rand(12, 100)
try:
    local_keys = estimate_local_key_7th(C_sync, window=8, agg_fn="gaussian")
    print(f"✓ estimate_local_key_7th: {local_keys.shape}")
except Exception as e:
    print(f"✗ estimate_local_key_7th failed: {e}")

# Test build_loglik_7th_enhanced
try:
    local_cfg = {"enable": True, "window": 8, "gamma": 0.3}
    n_cfg = {"energy_gamma": 1.0, "conf_gamma": 2.0}
    loglik = build_loglik_7th_enhanced(
        C_sync=C_sync,
        gamma_global=0.15,
        local_cfg=local_cfg,
        include_N=False,
        n_cfg=n_cfg,
        section_for_t=None
    )
    print(f"✓ build_loglik_7th_enhanced: {loglik.shape}")
except Exception as e:
    print(f"✗ build_loglik_7th_enhanced failed: {e}")
    import traceback
    traceback.print_exc()
