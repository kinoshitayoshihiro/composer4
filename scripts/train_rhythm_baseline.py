#!/usr/bin/env python3
"""Rhythm Model Training (Phase 25.2 Task 1)

Train XGBoost/LogReg models for drum pattern recommendation.

Processing:
1. Load train/val parquet
2. Feature engineering & encoding
3. Train XGBoost classifier (family prediction)
4. Train Logistic Regression baseline
5. Cross-validation & hyperparameter tuning
6. Save stage2_drums_v1.pickle

Output:
- stage2_drums_v1.pickle
  - pattern_dict: {pattern_id: pattern_data}
  - xgb_model: trained XGBoost model
  - lr_model: trained LogReg model
  - class_labels: family names
  - feature_names: feature list
  - scaler: StandardScaler for normalization
  - metadata: training info

Usage:
    python train_rhythm_baseline.py \\
        --train-parquet data/datasets/train.parquet \\
        --val-parquet data/datasets/val.parquet \\
        --output-pickle data/patterns/stage2_drums.pickle
    
    # Or use default paths (recommended):
    python train_rhythm_baseline.py
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Optional XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    logger.warning("XGBoost not available. Will use LogReg only.")
    HAS_XGB = False


# ===== Feature Preparation =====

FEATURE_COLUMNS = [
    "tempo_bpm",
    "slots",
    "density_k",
    "density_s",
    "density_h",
    "syncopation",
    "kick_downbeat_rate",
    "snare_backbeat_rate",
    "swing_hint",
    "section_encoded",
]

TARGET_COLUMN = "family"


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """特徴量準備
    
    Args:
        df: train/val DataFrame
    
    Returns:
        (X, y, feature_names)
    """
    # 特徴量抽出
    X = df[FEATURE_COLUMNS].values
    
    # ラベル抽出
    y = df[TARGET_COLUMN].values
    
    logger.info("Features shape: %s", X.shape)
    logger.info("Labels shape: %s", y.shape)
    logger.info("Unique families: %s", np.unique(y))
    
    return X, y, FEATURE_COLUMNS


# ===== XGBoost Training =====

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_encoder: LabelEncoder,
) -> tuple[Any, dict[str, float]]:
    """XGBoost multi-class classification
    
    Args:
        X_train: Training features
        y_train: Training labels (string)
        X_val: Validation features
        y_val: Validation labels (string)
        label_encoder: LabelEncoder for string→int
    
    Returns:
        (xgb_model, metrics)
    """
    if not HAS_XGB:
        logger.warning("XGBoost not available, skipping.")
        return None, {}
    
    logger.info("Training XGBoost classifier...")
    
    # ラベルエンコード
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # XGBoost parameters
    params = {
        "objective": "multi:softprob",
        "num_class": len(label_encoder.classes_),
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss",
    }
    
    model = xgb.XGBClassifier(**params)
    
    # 学習
    model.fit(
        X_train,
        y_train_encoded,
        eval_set=[(X_val, y_val_encoded)],
        verbose=False,
    )
    
    # 評価
    y_pred = model.predict(X_val)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    acc = accuracy_score(y_val, y_pred_labels)
    f1 = f1_score(y_val, y_pred_labels, average="weighted")
    
    logger.info("XGBoost - Accuracy: %.4f, F1: %.4f", acc, f1)
    
    # Feature importance
    importance = model.feature_importances_
    logger.info("Feature importance:")
    for feat, imp in sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: -x[1])[:5]:
        logger.info("  %s: %.4f", feat, imp)
    
    metrics = {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
    }
    
    return model, metrics


# ===== Logistic Regression Training =====

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: StandardScaler,
) -> tuple[LogisticRegression, dict[str, float]]:
    """Logistic Regression baseline
    
    Args:
        X_train: Training features
        y_train: Training labels (string)
        X_val: Validation features
        y_val: Validation labels (string)
        scaler: StandardScaler for normalization
    
    Returns:
        (lr_model, metrics)
    """
    logger.info("Training Logistic Regression baseline...")
    
    # 標準化
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # LogReg学習
    model = LogisticRegression(
        max_iter=500,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    
    # 評価
    y_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    
    logger.info("LogReg - Accuracy: %.4f, F1: %.4f", acc, f1)
    
    metrics = {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
    }
    
    return model, metrics


# ===== Pattern Dictionary Construction =====

def build_pattern_dict(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """パターン辞書構築
    
    各pattern_id_normalizedに対して代表データを保存
    
    Args:
        df: labeled patterns DataFrame
    
    Returns:
        {pattern_id: {
            "kick_vec": list,
            "snare_vec": list,
            "hat_vec": list,
            "family": str,
            "tempo_bpm": float,
            "slots": int,
            "usage_count": int,
        }}
    """
    logger.info("Building pattern dictionary...")
    
    pattern_dict = {}
    
    # pattern_id_normalizedでグループ化
    grouped = df.groupby("pattern_id_normalized")
    
    for pattern_id, group in grouped:
        # 代表パターン（最頻出family）
        family_mode = group["family"].mode()[0] if len(group["family"].mode()) > 0 else group["family"].iloc[0]
        
        # 代表行選択（最初の行）
        representative = group.iloc[0]
        
        # JSON→list変換
        kick_vec = json.loads(representative["kick_vec"]) if isinstance(representative["kick_vec"], str) else representative["kick_vec"]
        snare_vec = json.loads(representative["snare_vec"]) if isinstance(representative["snare_vec"], str) else representative["snare_vec"]
        hat_vec = json.loads(representative["hat_vec"]) if isinstance(representative["hat_vec"], str) else representative["hat_vec"]
        
        pattern_dict[pattern_id] = {
            "kick_vec": kick_vec,
            "snare_vec": snare_vec,
            "hat_vec": hat_vec,
            "family": family_mode,
            "tempo_bpm": float(representative["tempo_bpm"]),
            "slots": int(representative["slots"]),
            "usage_count": len(group),
            "density_k": float(representative["density_k"]),
            "density_s": float(representative["density_s"]),
            "density_h": float(representative["density_h"]),
            "syncopation": float(representative["syncopation"]),
        }
    
    logger.info("Built pattern dictionary with %d unique patterns.", len(pattern_dict))
    return pattern_dict


# ===== Main Training Pipeline =====

def train_rhythm_models(
    train_parquet: Path,
    val_parquet: Path,
    output_pickle: Path,
    save_probas: bool = False,
) -> None:
    """メイン学習パイプライン
    
    Args:
        train_parquet: Train dataset
        val_parquet: Validation dataset
        output_pickle: Output pickle path
        save_probas: Save train-set probas for QC
    """
    logger.info("Loading training data...")
    train_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet)
    
    logger.info("Train: %d patterns, Val: %d patterns", len(train_df), len(val_df))
    
    # 特徴量準備
    X_train, y_train, feature_names = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df)
    
    # LabelEncoder（family名→int）
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_val]))
    class_labels = label_encoder.classes_.tolist()
    logger.info("Class labels: %s", class_labels)
    
    # StandardScaler（LogReg用）
    scaler = StandardScaler()
    
    # XGBoost学習
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val, label_encoder)
    
    # LogReg学習
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val, scaler)
    
    # パターン辞書構築（train + valから）
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    pattern_dict = build_pattern_dict(all_df)
    
    # メタデータ
    metadata = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "num_features": len(feature_names),
        "num_classes": len(class_labels),
        "num_patterns": len(pattern_dict),
        "xgb_metrics": xgb_metrics,
        "lr_metrics": lr_metrics,
        "feature_names": feature_names,
        "class_labels": class_labels,
    }
    
    # Pickle保存
    logger.info("Saving models to %s...", output_pickle)
    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_pickle, "wb") as f:
        pickle.dump({
            "pattern_dict": pattern_dict,
            "xgb_model": xgb_model,
            "lr_model": lr_model,
            "label_encoder": label_encoder,
            "scaler": scaler,
            "feature_names": feature_names,
            "class_labels": class_labels,
            "metadata": metadata,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info("Training complete. Models saved to %s", output_pickle)
    logger.info("Metadata: %s", json.dumps(metadata, indent=2))
    
    # Save train-set probas for QC (optional)
    if save_probas and xgb_model is not None:
        logger.info("Saving train-set probas for QC...")
        try:
            y_train_encoded = label_encoder.transform(y_train)
            probas_train = xgb_model.predict_proba(X_train)
            
            # DataFrame構築
            proba_df = pd.DataFrame(
                probas_train,
                columns=[f"proba_{cls}" for cls in class_labels]
            )
            proba_df["y_true"] = y_train
            proba_df["y_true_encoded"] = y_train_encoded
            
            # Parquet保存
            probas_path = output_pickle.with_suffix(".train_probas.parquet")
            proba_df.to_parquet(probas_path)
            logger.info("Train-set probas saved to %s", probas_path)
        except Exception as exc:
            logger.warning("Failed to save train-set probas: %s", exc)


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train rhythm baseline models (Phase 25.2 Task 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=Path("data/datasets/train.parquet"),
        help="Train dataset parquet",
    )
    parser.add_argument(
        "--val-parquet",
        type=Path,
        default=Path("data/datasets/val.parquet"),
        help="Validation dataset parquet",
    )
    parser.add_argument(
        "--output-pickle",
        type=Path,
        default=Path("data/patterns/stage2_drums.pickle"),
        help="Output pickle path (stage2_drums.pickle)",
    )
    parser.add_argument(
        "--save-probas",
        action="store_true",
        help="Save train-set probas for QC (train_probas.parquet)",
    )
    
    args = parser.parse_args()
    
    if not args.train_parquet.exists():
        logger.error("Train parquet not found: %s", args.train_parquet)
        return 1
    
    if not args.val_parquet.exists():
        logger.error("Val parquet not found: %s", args.val_parquet)
        return 1
    
    try:
        train_rhythm_models(
            train_parquet=args.train_parquet,
            val_parquet=args.val_parquet,
            output_pickle=args.output_pickle,
            save_probas=args.save_probas,
        )
        return 0
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
