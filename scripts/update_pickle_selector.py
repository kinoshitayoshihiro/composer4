#!/usr/bin/env python3
"""
Update Pickle Selector - XGB/Sklearn モデルに差し替え

既存のpickle（ルールベースselector）のselector部分を学習済みXGB/Sklearnモデルに差し替えます。

Usage:
    python scripts/update_pickle_selector.py \\
      --in-pickle data/patterns/stage2_guitar.pickle \\
      --model     data/patterns/harmony_baseline_xgb_light.joblib \\
      --meta      data/patterns/harmony_baseline_xgb_light.json \\
      --out       data/patterns/stage2_guitar_v2.pickle

Features:
    - 学習済みモデル（XGBoost/RandomForest）をselectorに統合
    - feature_spec（order/types/encoders）とclass_labelsをmetaから読込
    - patterns辞書はそのまま保持（互換性維持）
    - version情報の自動更新（v1 → v2）
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

try:
    import cloudpickle as _pickle
except ImportError:
    import pickle as _pickle

import joblib

# Logging設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Update pickle selector with trained XGB/Sklearn model"
    )

    parser.add_argument(
        "--in-pickle", type=Path, required=True, help="Input pickle file (stage2_guitar.pickle)"
    )

    parser.add_argument(
        "--model", type=Path, required=True, help="Trained model (joblib): XGB/Sklearn classifier"
    )

    parser.add_argument(
        "--meta",
        type=Path,
        required=False,
        help="Model metadata JSON (feature_spec, class_labels, etc.)",
    )

    parser.add_argument(
        "--out", type=Path, required=True, help="Output pickle file (stage2_guitar_v2.pickle)"
    )

    args = parser.parse_args()

    # Load input pickle
    logger.info(f"Loading input pickle from {args.in_pickle}")
    with open(args.in_pickle, "rb") as f:
        artifact = _pickle.load(f)

    logger.info(f"  Version: {artifact.get('version', 'unknown')}")
    logger.info(f"  Patterns: {len(artifact.get('patterns', {}))}")
    logger.info(f"  Current selector type: {artifact.get('selector', {}).get('type', 'unknown')}")

    # Load trained model
    logger.info(f"Loading trained model from {args.model}")
    model_artifact = joblib.load(args.model)

    # Extract model from artifact (if wrapped in dict)
    encoders_from_artifact = {}
    artifact_meta = {}

    if isinstance(model_artifact, dict):
        model = model_artifact.get("model")
        logger.info(f"  Model artifact type: dict")
        logger.info(f"  Model type: {type(model).__name__}")

        # Extract encoders and metadata from artifact
        if "encoders" in model_artifact:
            encoders_from_artifact = model_artifact["encoders"]
            logger.info(f"  Encoders found: {list(encoders_from_artifact.keys())}")

        if "label_encoder" in model_artifact:
            label_encoder = model_artifact["label_encoder"]
            logger.info(f"  Label encoder found")

        if "metadata" in model_artifact:
            artifact_meta = model_artifact["metadata"]
            logger.info(f"  Metadata found in artifact")
    else:
        model = model_artifact
        logger.info(f"  Model type: {type(model).__name__}")

    # Load metadata
    feature_spec = {
        "order": [
            "section",
            "chord_root",
            "chord_quality",
            "tempo_bin",
            "confidence",
            "time_sig",
            "tempo",
        ],
        "types": {
            "section": "cat",
            "chord_root": "cat",
            "chord_quality": "cat",
            "tempo_bin": "cat",
            "confidence": "num",
            "time_sig": "cat",
            "tempo": "num",
        },
        "encoders": encoders_from_artifact,
    }
    class_labels = None

    if args.meta and args.meta.exists():
        logger.info(f"Loading metadata from {args.meta}")
        meta = load_json(args.meta)

        if "feature_spec" in meta:
            feature_spec = meta["feature_spec"]
            logger.info(f"  Feature order: {feature_spec.get('order', [])}")

        if "class_labels" in meta:
            class_labels = meta["class_labels"]
            logger.info(f"  Class labels loaded from metadata: {len(class_labels)} patterns")

    # Fallback: use model.classes_
    if class_labels is None and hasattr(model, "classes_"):
        class_labels = [str(x) for x in model.classes_.tolist()]
        logger.info(f"  Using model.classes_: {len(class_labels)} classes")

    # Fallback: use label_encoder
    if class_labels is None and "label_encoder" in model_artifact:
        label_encoder = model_artifact["label_encoder"]
        if hasattr(label_encoder, "classes_"):
            class_labels = [str(x) for x in label_encoder.classes_.tolist()]
            logger.info(f"  Using label_encoder.classes_: {len(class_labels)} classes")

    if not class_labels:
        raise RuntimeError(
            "class_labels が見当たりません。"
            "--meta に class_labels を含めるか、model.classes_ を持つ推論器を指定してください。"
        )

    # Create new selector
    selector = {
        "type": "xgboost" if "XGB" in type(model).__name__ else "sklearn",
        "path": str(args.model.resolve()),
        "feature_spec": feature_spec,
        "class_labels": class_labels,
        "predict": "predict_proba" if hasattr(model, "predict_proba") else "predict",
        "notes": f"Trained model: {args.model.name}, replaced from rule-based selector",
    }

    logger.info(f"Creating new selector:")
    logger.info(f"  Type: {selector['type']}")
    logger.info(f"  Predict method: {selector['predict']}")
    logger.info(f"  Features: {len(feature_spec.get('order', []))}")
    logger.info(f"  Classes: {len(class_labels)}")

    # Update artifact
    out = copy.deepcopy(artifact)

    # Update meta
    out.setdefault("meta", {}).update(
        {
            "instrument": out.get("meta", {}).get("instrument", "guitar"),
            "version": "v2",
            "provider": selector["type"],
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selector_model": args.model.name,
            "original_version": artifact.get("version", "v1"),
        }
    )

    # Replace selector
    out["selector"] = selector

    # Save output
    logger.info(f"Saving updated pickle to {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "wb") as f:
        _pickle.dump(out, f)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Selector update complete!")
    logger.info("=" * 60)
    logger.info(f"Output: {args.out}")
    logger.info(f"Selector type: {selector['type']}")
    logger.info(f"Model: {args.model.name}")
    logger.info("")
    logger.info("To use:")
    logger.info(f"  export STAGE2_GUITAR_PATTERNS={args.out.resolve()}")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
