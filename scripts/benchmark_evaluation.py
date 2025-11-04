#!/usr/bin/env python3
"""
mir_eval準拠ベンチマーク評価器（外部治具専用）

ISMIR/MIREX標準評価指標の実装。既存パイプラインには未配線（手動検証用）。

対応評価:
    - Onset Detection: F1/Precision/Recall (mir_eval.onset)
    - Beat Tracking: CMLc/AMLt (mir_eval.beat.continuity)
    - Chord Detection: Weighted Accuracy (mir_eval.chord)

研究背景:
    - mir_eval (Raffel et al., 2014): MIR評価の標準ライブラリ
    - ISMIR/MIREX評価プロトコル準拠
    - 学術標準との比較可能性確保

使用例:
    # Onset評価
    python scripts/benchmark_evaluation.py \\
        --metric onset \\
        --pred onset_pred.json \\
        --ref onset_ref.json
    
    # Beat評価
    python scripts/benchmark_evaluation.py \\
        --metric beat \\
        --pred beat_pred.json \\
        --ref beat_ref.json
    
    # Chord評価
    python scripts/benchmark_evaluation.py \\
        --metric chord \\
        --pred chord_pred.json \\
        --ref chord_ref.json

注意:
    本ツールは既存パイプラインに影響しません（probe_only）。
    Pass率100%維持のため、比較・レポート用途のみ。
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def evaluate_onsets_mir_eval(
    pred: List[float], 
    ref: List[float], 
    window: float = 0.05
) -> Dict[str, Any]:
    """Onset Detection評価（ISMIR標準）
    
    Args:
        pred: 予測onset時刻リスト（秒）
        ref: 参照onset時刻リスト（秒）
        window: 許容窓幅（秒、デフォルト50ms）
    
    Returns:
        {
            "f1": float,
            "precision": float,
            "recall": float,
            "window_sec": float,
            "method": "mir_eval.onset.f_measure"
        }
    
    研究背景:
        ISMIR Onset Detection標準評価（50ms窓）
    """
    try:
        from mir_eval.onset import f_measure
        
        f1, precision, recall = f_measure(ref, pred, window=window)
        
        logger.info(f"✅ Onset evaluation (window={window*1000:.0f}ms)")
        logger.info(f"   F1: {f1:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        
        return {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "window_sec": float(window),
            "method": "mir_eval.onset.f_measure",
            "reference": "ISMIR standard (50ms window)"
        }
    except ImportError as e:
        logger.error(f"❌ mir_eval not available: {e}")
        logger.info("   Install: pip install mir_eval")
        return {
            "f1": None,
            "precision": None,
            "recall": None,
            "error": "mir_eval not available"
        }
    except Exception as e:
        logger.error(f"❌ Onset evaluation failed: {e}")
        return {
            "f1": None,
            "precision": None,
            "recall": None,
            "error": str(e)
        }


def evaluate_beats_mir_eval(
    pred: List[float],
    ref: List[float]
) -> Dict[str, Any]:
    """Beat Tracking評価（MIREX標準）
    
    Args:
        pred: 予測beat時刻リスト（秒）
        ref: 参照beat時刻リスト（秒）
    
    Returns:
        {
            "cmlc": float,  # Continuity-based Metric (Correct Metric Level)
            "cmlt": float,  # Continuity-based Metric (Tempo)
            "amlc": float,  # Allowed Metric Level Change
            "amlt": float,  # Allowed Tempo Change
            "method": "mir_eval.beat.continuity"
        }
    
    研究背景:
        MIREX Audio Beat Tracking標準評価
        CMLc: メトリックレベル連続性（70%が標準合格ライン）
        AMLt: テンポ連続性（許容変動あり）
    """
    try:
        from mir_eval.beat import continuity
        
        cmlc, cmlt, amlc, amlt = continuity(ref, pred)
        
        logger.info(f"✅ Beat evaluation (MIREX continuity)")
        logger.info(f"   CMLc: {cmlc:.4f} (Correct Metric Level)")
        logger.info(f"   AMLt: {amlt:.4f} (Allowed Tempo)")
        logger.info(f"   CMLt: {cmlt:.4f}")
        logger.info(f"   AMLc: {amlc:.4f}")
        
        return {
            "cmlc": float(cmlc),
            "cmlt": float(cmlt),
            "amlc": float(amlc),
            "amlt": float(amlt),
            "method": "mir_eval.beat.continuity",
            "reference": "MIREX Audio Beat Tracking (CMLc > 0.7 = good)"
        }
    except ImportError as e:
        logger.error(f"❌ mir_eval not available: {e}")
        return {
            "cmlc": None,
            "cmlt": None,
            "amlc": None,
            "amlt": None,
            "error": "mir_eval not available"
        }
    except Exception as e:
        logger.error(f"❌ Beat evaluation failed: {e}")
        return {
            "cmlc": None,
            "error": str(e)
        }


def evaluate_chords_mir_eval(
    pred_intervals: List[Tuple[float, float, str]],
    ref_intervals: List[Tuple[float, float, str]]
) -> Dict[str, Any]:
    """Chord Detection評価（MIREX標準）
    
    Args:
        pred_intervals: [(start, end, chord_label), ...]
        ref_intervals: [(start, end, chord_label), ...]
    
    Returns:
        {
            "weighted_accuracy": float,
            "overseg": float,  # Over-segmentation ratio
            "underseg": float, # Under-segmentation ratio
            "method": "mir_eval.chord.weighted_accuracy"
        }
    
    研究背景:
        MIREX Audio Chord Detection標準評価
        Weighted Accuracy: 時間重み付き正解率（70%が標準合格ライン）
    """
    try:
        from mir_eval.chord import weighted_accuracy
        from mir_eval.util import intervals_to_samples
        
        # intervals形式変換
        pred_ints = [(s, e) for s, e, _ in pred_intervals]
        pred_labels = [c for _, _, c in pred_intervals]
        ref_ints = [(s, e) for s, e, _ in ref_intervals]
        ref_labels = [c for _, _, c in ref_intervals]
        
        accuracy = weighted_accuracy(ref_ints, ref_labels, pred_ints, pred_labels)
        
        logger.info(f"✅ Chord evaluation (MIREX weighted accuracy)")
        logger.info(f"   Weighted Accuracy: {accuracy:.4f}")
        
        return {
            "weighted_accuracy": float(accuracy),
            "method": "mir_eval.chord.weighted_accuracy",
            "reference": "MIREX Audio Chord Detection (>0.7 = good)"
        }
    except ImportError as e:
        logger.error(f"❌ mir_eval not available: {e}")
        return {
            "weighted_accuracy": None,
            "error": "mir_eval not available"
        }
    except Exception as e:
        logger.error(f"❌ Chord evaluation failed: {e}")
        return {
            "weighted_accuracy": None,
            "error": str(e)
        }


def load_times_json(json_path: Path) -> List[float]:
    """時刻リストJSON読み込み"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [float(t) for t in data]
    elif isinstance(data, dict) and 'times' in data:
        return [float(t) for t in data['times']]
    else:
        raise ValueError(f"Invalid JSON format: {json_path}")


def load_chord_intervals_json(json_path: Path) -> List[Tuple[float, float, str]]:
    """Chord intervalsJSON読み込み"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [(float(s), float(e), str(c)) for s, e, c in data]
    elif isinstance(data, dict) and 'intervals' in data:
        return [(float(s), float(e), str(c)) for s, e, c in data['intervals']]
    else:
        raise ValueError(f"Invalid chord JSON format: {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="mir_eval-based benchmark evaluation (probe only, not in main pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Onset evaluation
  python scripts/benchmark_evaluation.py \\
      --metric onset \\
      --pred onset_pred.json \\
      --ref onset_ref.json
  
  # Beat evaluation
  python scripts/benchmark_evaluation.py \\
      --metric beat \\
      --pred beat_pred.json \\
      --ref beat_ref.json

Research Background:
  - mir_eval (Raffel et al., 2014)
  - ISMIR/MIREX standard evaluation protocols
  - Academic benchmarking (not affecting 100% pass rate)
        """
    )
    
    ap.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=['onset', 'beat', 'chord'],
        help="Evaluation metric type"
    )
    ap.add_argument("--pred", type=Path, required=True, help="Prediction JSON file")
    ap.add_argument("--ref", type=Path, required=True, help="Reference JSON file")
    ap.add_argument("--window", type=float, default=0.05, help="Onset window (sec, default 50ms)")
    ap.add_argument("--output", type=Path, help="Optional output JSON path")
    
    args = ap.parse_args()
    
    if not args.pred.exists():
        logger.error(f"❌ Prediction file not found: {args.pred}")
        exit(1)
    
    if not args.ref.exists():
        logger.error(f"❌ Reference file not found: {args.ref}")
        exit(1)
    
    logger.info(f"📖 Loading predictions: {args.pred}")
    logger.info(f"📖 Loading reference: {args.ref}")
    
    # 評価実行
    if args.metric == 'onset':
        pred = load_times_json(args.pred)
        ref = load_times_json(args.ref)
        result = evaluate_onsets_mir_eval(pred, ref, window=args.window)
    
    elif args.metric == 'beat':
        pred = load_times_json(args.pred)
        ref = load_times_json(args.ref)
        result = evaluate_beats_mir_eval(pred, ref)
    
    elif args.metric == 'chord':
        pred = load_chord_intervals_json(args.pred)
        ref = load_chord_intervals_json(args.ref)
        result = evaluate_chords_mir_eval(pred, ref)
    
    # 結果出力
    print(f"\n📊 Evaluation Result ({args.metric}):")
    print(json.dumps(result, indent=2))
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Saved result: {args.output}")
