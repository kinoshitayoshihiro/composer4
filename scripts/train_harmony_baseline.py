#!/usr/bin/env python3
"""
Harmony Baseline Training - XGBoost/RandomForest

Stage2 Guitar Pattern Selector用のベースライン学習。
ルールベースselectorをXGB/RandomForestに差し替え。

Usage:
    python scripts/train_harmony_baseline.py \\
      --train harmony_dataset/splits/train.parquet \\
      --val harmony_dataset/splits/val.parquet \\
      --model xgboost \\
      --output data/patterns/harmony_baseline_xgb.joblib \\
      --n-estimators 100 \\
      --max-depth 6

Features:
    - XGBoost/RandomForest対応（sklearn互換API）
    - 特徴量: section, chord_root, chord_quality, tempo, time_sig
    - ターゲット: pattern_id（multi-class classification）
    - 評価: accuracy, top-3 accuracy, weighted F1
    - Feature importance出力
    - joblib形式で保存（差し替え用）
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import joblib

# Logging設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train harmony baseline model (XGB/RandomForest)")

    parser.add_argument("--train", type=Path, required=True, help="Training data parquet")

    parser.add_argument("--val", type=Path, required=True, help="Validation data parquet")

    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "random_forest"],
        default="xgboost",
        help="Model type (default: xgboost)",
    )

    parser.add_argument("--output", type=Path, required=True, help="Output model path (joblib)")

    parser.add_argument(
        "--n-estimators", type=int, default=100, help="Number of estimators (default: 100)"
    )

    parser.add_argument("--max-depth", type=int, default=6, help="Max depth (default: 6)")

    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate for XGBoost (default: 0.1)"
    )

    parser.add_argument(
        "--min-pattern-usage",
        type=int,
        default=10,
        help="Minimum pattern usage count (default: 10)",
    )

    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Sample ratio for training data (0.0-1.0, default: 1.0 = all data)",
    )

    parser.add_argument("--random-seed", type=int, default=42, help="Random seed (default: 42)")

    return parser.parse_args()


def expand_chord_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand chord_sequence array into individual rows (memory-efficient version)

    Args:
        df: DataFrame with chord_sequence column

    Returns:
        Expanded DataFrame
    """
    logger.info("Expanding chord sequences...")

    # Memory-efficient expansion using list comprehension in batches
    batch_size = 1000
    all_rows = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        batch_rows = [
            {
                "song_id": row["song_id"],
                "section": row["section"],
                "tempo": row["tempo"],
                "time_sig": row["time_sig"],
                "bar": chord["bar"],
                "chord_root": chord["root"],
                "chord_quality": chord["quality"],
                "confidence": chord["confidence"],
                "label_strength": chord["label_strength"],
            }
            for _, row in batch.iterrows()
            for chord in row["chord_sequence"]
        ]
        all_rows.extend(batch_rows)

        # Log progress every 10000 sequences
        current = min(i + batch_size, len(df))
        if current % 10000 == 0 or current == len(df):
            logger.info(f"  Processed {current}/{len(df)} sequences...")

    expanded_df = pd.DataFrame(all_rows)
    logger.info(f"  Expanded to {len(expanded_df)} chord events from {len(df)} sequences")

    return expanded_df


def create_pattern_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pattern IDs from (section, root, quality, tempo_bin)

    Args:
        df: Expanded DataFrame

    Returns:
        DataFrame with pattern_id column
    """
    logger.info("Creating pattern IDs...")

    # Tempo binning
    df["tempo_bin"] = pd.cut(
        df["tempo"], bins=[0, 90, 120, 150, 200], labels=["slow", "mid", "fast", "very_fast"]
    )

    # Create pattern_id (vectorized)
    import hashlib

    # Vectorized pattern key creation
    df["pattern_key"] = (
        df["section"].astype(str)
        + "_"
        + df["chord_root"].astype(str)
        + "_"
        + df["chord_quality"].astype(str)
        + "_"
        + df["tempo_bin"].astype(str)
    )

    # Hash pattern keys (vectorized with map)
    def hash_key(key):
        return hashlib.md5(key.encode()).hexdigest()[:12]

    df["pattern_id"] = df["pattern_key"].map(hash_key)
    df = df.drop(columns=["pattern_key"])

    n_unique = df["pattern_id"].nunique()
    logger.info(f"  Created {n_unique} unique pattern IDs")

    return df


def filter_rare_patterns(
    train_df: pd.DataFrame, val_df: pd.DataFrame, min_usage: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out rare patterns (usage < min_usage)

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        min_usage: Minimum usage count

    Returns:
        (filtered_train_df, filtered_val_df)
    """
    logger.info(f"Filtering patterns with usage < {min_usage}...")

    # Count pattern usage in training set
    pattern_counts = train_df["pattern_id"].value_counts()
    valid_patterns = pattern_counts[pattern_counts >= min_usage].index

    # Filter
    train_filtered = train_df[train_df["pattern_id"].isin(valid_patterns)]
    val_filtered = val_df[val_df["pattern_id"].isin(valid_patterns)]

    logger.info(f"  Train: {len(train_df)} -> {len(train_filtered)} events")
    logger.info(f"  Val: {len(val_df)} -> {len(val_filtered)} events")
    logger.info(f"  Valid patterns: {len(valid_patterns)}")

    return train_filtered, val_filtered


def encode_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Encode categorical features

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame

    Returns:
        (X_train, X_val, encoders)
    """
    logger.info("Encoding features...")

    from sklearn.preprocessing import LabelEncoder

    encoders = {}

    # Categorical features
    cat_features = ["section", "chord_root", "chord_quality", "tempo_bin", "time_sig"]

    X_train_parts = []
    X_val_parts = []

    for feat in cat_features:
        le = LabelEncoder()

        # Fit on combined vocabulary
        combined = pd.concat([train_df[feat], val_df[feat]]).astype(str)
        le.fit(combined)

        # Transform
        train_encoded = le.transform(train_df[feat].astype(str))
        val_encoded = le.transform(val_df[feat].astype(str))

        X_train_parts.append(train_encoded.reshape(-1, 1))
        X_val_parts.append(val_encoded.reshape(-1, 1))

        encoders[feat] = le
        logger.info(f"  {feat}: {len(le.classes_)} classes")

    # Numerical features
    num_features = ["tempo", "confidence"]

    for feat in num_features:
        X_train_parts.append(train_df[feat].values.reshape(-1, 1))
        X_val_parts.append(val_df[feat].values.reshape(-1, 1))

    # Concatenate
    X_train = np.hstack(X_train_parts)
    X_val = np.hstack(X_val_parts)

    logger.info(f"  Feature matrix shape: train {X_train.shape}, val {X_val.shape}")

    encoders["feature_names"] = cat_features + num_features

    return X_train, X_val, encoders


def encode_target(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Encode target (pattern_id)

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame

    Returns:
        (y_train, y_val, label_encoder)
    """
    logger.info("Encoding target (pattern_id)...")

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    # Fit on training patterns
    le.fit(train_df["pattern_id"])

    y_train = le.transform(train_df["pattern_id"])
    y_val = le.transform(val_df["pattern_id"])

    logger.info(f"  Target classes: {len(le.classes_)}")

    return y_train, y_val, le


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_seed: int,
) -> Any:
    """
    Train model

    Args:
        X_train: Training features
        y_train: Training target
        model_type: 'xgboost' or 'random_forest'
        n_estimators: Number of estimators
        max_depth: Max depth
        learning_rate: Learning rate (XGBoost only)
        random_seed: Random seed

    Returns:
        Trained model
    """
    logger.info(f"Training {model_type} model...")
    logger.info(f"  n_estimators: {n_estimators}")
    logger.info(f"  max_depth: {max_depth}")

    if model_type == "xgboost":
        try:
            import xgboost as xgb

            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_seed,
                n_jobs=4,  # Limit CPU usage to reduce memory
                tree_method="hist",
                max_bin=256,  # Reduce memory usage
                subsample=0.8,  # Use 80% of data per tree
                colsample_bytree=0.8,  # Use 80% of features per tree
                # Tuning improvements
                min_child_weight=3,  # Prevent overfitting
                gamma=0.1,  # Minimum loss reduction
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
            )
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            sys.exit(1)

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_seed, n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    model.fit(X_train, y_train)

    logger.info("  Training complete")

    return model


def evaluate_model(
    model: Any, X_val: np.ndarray, y_val: np.ndarray, label_encoder: Any
) -> Dict[str, float]:
    """
    Evaluate model

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        label_encoder: Label encoder for pattern_id

    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating model...")

    from sklearn.metrics import accuracy_score, f1_score

    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)

    # Accuracy
    accuracy = accuracy_score(y_val, y_pred)

    # Top-3 accuracy
    top3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:]
    top3_accuracy = np.mean([y_val[i] in top3_indices[i] for i in range(len(y_val))])

    # Weighted F1
    f1 = f1_score(y_val, y_pred, average="weighted")

    metrics = {"accuracy": accuracy, "top3_accuracy": top3_accuracy, "weighted_f1": f1}

    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Top-3 Accuracy: {top3_accuracy:.4f}")
    logger.info(f"  Weighted F1: {f1:.4f}")

    return metrics


def get_feature_importance(model: Any, feature_names: List[str], top_k: int = 20) -> pd.DataFrame:
    """
    Get feature importance

    Args:
        model: Trained model
        feature_names: Feature names
        top_k: Top-K features to show

    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )

        logger.info(f"Top {top_k} feature importances:")
        for _, row in df.head(top_k).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return df

    return None


def main():
    args = parse_args()

    # Load data
    logger.info(f"Loading training data from {args.train}")
    train_df = pd.read_parquet(args.train)
    logger.info(f"  Loaded {len(train_df)} sequences from {train_df['song_id'].nunique()} songs")

    # Sample training data if requested
    if args.sample_ratio < 1.0:
        n_sample = int(len(train_df) * args.sample_ratio)
        train_df = train_df.sample(n=n_sample, random_state=args.random_seed)
        logger.info(f"  Sampled {args.sample_ratio*100:.1f}% -> {len(train_df)} sequences")

    logger.info(f"Loading validation data from {args.val}")
    val_df = pd.read_parquet(args.val)
    logger.info(f"  Loaded {len(val_df)} sequences from {val_df['song_id'].nunique()} songs")

    # Expand chord sequences
    train_expanded = expand_chord_sequences(train_df)
    val_expanded = expand_chord_sequences(val_df)

    # Create pattern IDs
    train_expanded = create_pattern_ids(train_expanded)
    val_expanded = create_pattern_ids(val_expanded)

    # Filter rare patterns
    train_filtered, val_filtered = filter_rare_patterns(
        train_expanded, val_expanded, args.min_pattern_usage
    )

    # Encode features
    X_train, X_val, encoders = encode_features(train_filtered, val_filtered)

    # Encode target
    y_train, y_val, label_encoder = encode_target(train_filtered, val_filtered)

    # Train model
    model = train_model(
        X_train,
        y_train,
        args.model,
        args.n_estimators,
        args.max_depth,
        args.learning_rate,
        args.random_seed,
    )

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val, label_encoder)

    # Feature importance
    feature_importance = get_feature_importance(model, encoders["feature_names"])

    # Save model
    logger.info(f"Saving model to {args.output}")

    model_package = {
        "model": model,
        "encoders": encoders,
        "label_encoder": label_encoder,
        "metrics": metrics,
        "metadata": {
            "model_type": args.model,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_pattern_usage": args.min_pattern_usage,
            "n_classes": len(label_encoder.classes_),
            "n_features": X_train.shape[1],
            "random_seed": args.random_seed,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, args.output)

    logger.info(f"  ✓ Saved model package to {args.output}")

    # Save metadata JSON (with class_labels for pickle update)
    metadata_path = args.output.with_suffix(".json")
    with open(metadata_path, "w") as f:
        # Convert to JSON-serializable format
        metadata_json = {
            "model_type": args.model,
            "metrics": metrics,
            "metadata": model_package["metadata"],
            "feature_spec": {
                "order": encoders.get("feature_names", []),
                "types": {},  # populated by pickle builder
                "encoders": {}  # populated by pickle builder (LabelEncoder objects)
            },
            "class_labels": label_encoder.classes_.tolist(),  # ★ pattern_id list (not ['0','1',...])
        }
        json.dump(metadata_json, f, indent=2)

    logger.info(f"  ✓ Saved metadata to {metadata_path}")
    logger.info(f"  ✓ Class labels: {len(label_encoder.classes_)} patterns")

    # Save feature importance
    if feature_importance is not None:
        importance_path = args.output.parent / f"{args.output.stem}_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"  ✓ Saved feature importance to {importance_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
