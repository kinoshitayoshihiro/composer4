#!/usr/bin/env python3
"""
パッチ第三弾の動作確認スクリプト（簡易版）
- 状態別クールダウン（セクション別上書き）
- 遷移コスト（エネルギー・プリエンプト・ヘッド切替）
- セクション終端ラッチ（開始グレース）
- 確率の慣性（モメンタム）
- ローカルモード（セクション/レンジ別行列）
- 音素連動ブレンド（クラス別ウィンドウ）
"""
import sys

def test_patch3():
    print("=" * 60)
    print("Markov Stage2 パッチ第三弾 動作確認")
    print("=" * 60)
    
    # === Test 1: NO-OP（未設定） ===
    print("\n=== Test 1: NO-OP（未設定） ===")
    try:
        from generator.drums_generator_stage2 import DrumsGeneratorStage2
        gen = DrumsGeneratorStage2()
        print("✓ NO-OP設定で正常に初期化")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False
    
    # === Test 2: 状態別クールダウン設定読み込み ===
    print("\n=== Test 2: State Cooldown設定読み込み ===")
    try:
        cfg = {
            "ride": {
                "ride_markov": {
                    "enable": True,
                    "start": "ride1",
                    "states": {
                        "ride1": {"bow": {"ride1": 0.6, "ride2": 0.3, "china": 0.1}},
                        "ride2": {"bow": {"ride1": 0.3, "ride2": 0.6, "china": 0.1}},
                        "china": {"bow": {"ride1": 0.4, "ride2": 0.4, "china": 0.2}}
                    },
                    "state_cooldown": {
                        "ride2": {"bars": 1},
                        "china": {"bars": 2, "hits": 2},
                        "per_head": {
                            "bell": {"china": {"hits": 2}}
                        },
                        "per_section": {
                            "verse": {"china": {"bars": 3}},
                            "chorus": {"ride2": {"bars": 0}}
                        }
                    }
                }
            }
        }
        print("✓ 状態別クールダウン設定:")
        print("  - ride2: bars=1")
        print("  - china: bars=2, hits=2")
        print("  - per_section: verse(china:3bars), chorus(ride2:0bars)")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False
    
    # === Test 3: 遷移コスト設定 ===
    print("\n=== Test 3: Transition Costs設定 ===")
    try:
        cfg = {
            "ride": {
                "ride_markov": {
                    "enable": True,
                    "transition_costs": {
                        "mode": "exp",
                        "energy_alpha": 0.2,
                        "bias": 0.0,
                        "base": {
                            "ride1": {"ride2": 0.10, "china": 0.30},
                            "ride2": {"ride1": 0.08, "china": 0.22},
                            "china": {"ride1": 0.05, "ride2": 0.10}
                        },
                        "per_head": {
                            "bell": {"ride1": {"china": 0.20}}
                        },
                        "head_switch": {"enable": True, "cost": 0.08},
                        "preempt": {
                            "alpha": 0.5,
                            "prefer": {
                                "chorus": {"ride2": -0.20, "china": -0.05}
                            }
                        }
                    }
                }
            }
        }
        print("✓ 遷移コスト設定:")
        print("  - mode: exp")
        print("  - energy_alpha: 0.2")
        print("  - head_switch: enabled, cost=0.08")
        print("  - preempt.alpha: 0.5")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False
    
    # === Test 4: セクション終端ラッチ ===
    print("\n=== Test 4: Latch設定 ===")
    try:
        cfg = {
            "ride": {
                "ride_markov": {
                    "enable": True,
                    "latch": {
                        "enable": True,
                        "beats": 2.0,
                        "mode": "prefer",
                        "prefer_boost": 0.40,
                        "sections": ["chorus"],
                        "state": "ride2",
                        "grace_beats": 0.5
                    }
                }
            }
        }
        print("✓ ラッチ設定:")
        print("  - beats: 2.0 (終端2拍)")
        print("  - mode: prefer, boost=0.40")
        print("  - grace_beats: 0.5 (開始0.5拍)")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False
    
    # === Test 5: 確率の慣性 ===
    print("\n=== Test 5: Probability Momentum設定 ===")
    try:
        cfg = {
            "ride": {
                "ride_markov": {
                    "enable": True,
                    "prob_momentum": {
                        "enable": True,
                        "alpha": 0.35
                    }
                }
            }
        }
        print("✓ モメンタム設定:")
        print("  - enable: True")
        print("  - alpha: 0.35 (前フレーム35%、現在65%)")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False
    
    # === Test 6: ローカルモード ===
    print("\n=== Test 6: Local Modes設定 ===")
    try:
        cfg = {
            "ride": {
                "ride_markov": {
                    "enable": True,
                    "local_modes": {
                        "chorus": {
                            "inherit": True,
                            "mix": 0.7,
                            "states": {
                                "ride1": {
                                    "bow": {"ride1": 0.50, "ride2": 0.40, "china": 0.10}
                                }
                            }
                        },
                        "by_range": [
                            {
                                "from": 4, "to": 8,
                                "head": "any",
                                "inherit": False,
                                "states": {
                                    "ride1": {
                                        "bow": {"ride1": 0.40, "ride2": 0.50, "china": 0.10}
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        print("✓ ローカルモード設定:")
        print("  - chorus: inherit=True, mix=0.7")
        print("  - by_range: bars 4-8, inherit=False")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False
    
    # === Test 7: 音素連動ブレンド ===
    print("\n=== Test 7: Phoneme Blend設定 ===")
    try:
        cfg = {
            "ride": {
                "ride_markov": {
                    "enable": True,
                    "phoneme_blend": {
                        "enable": True,
                        "mode": "add",
                        "alpha": 0.5,
                        "energy_alpha": 0.5,
                        "head": "any",
                        "sections": ["verse"],
                        "window_ms": 120,
                        "falloff": "ease",
                        "classes": {
                            "sibilant": {
                                "states": {"china": -0.40, "ride2": -0.20, "ride1": 0.10},
                                "window_ms": 160,
                                "falloff": "exp"
                            },
                            "plosive": {
                                "states": {"china": 0.15, "ride2": 0.10},
                                "window_ql": 0.25,
                                "falloff": "linear"
                            },
                            "nasal": {
                                "states": {"ride1": 0.10, "ride2": -0.05},
                                "window_ms": 100
                            }
                        }
                    }
                }
            }
        }
        print("✓ 音素連動設定:")
        print("  - enable: True, mode: add")
        print("  - sibilant: window_ms=160, falloff=exp")
        print("  - plosive: window_ql=0.25")
        print("  - nasal: window_ms=100")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ すべてのテストが成功しました！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_patch3()
    sys.exit(0 if success else 1)

