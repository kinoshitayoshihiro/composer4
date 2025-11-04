#!/usr/bin/env python3
"""
Train Drum ML Model from Stage2 Features

LAMDA Stage2 rhythm_features.parquet から学習してPickle生成

Usage:
    python scripts/train_drum_ml_model.py \
        --features-parquet output/rhythm_ai/rhythm_features_merged.parquet \
        --out data/patterns/stage2_drums_ml.pickle \
        --algo auto
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def train_model(algo: str, X: np.ndarray, y: np.ndarray):
    """モデル学習（XGBoost→LogisticRegression fallback）"""
    if algo == "auto":
        try:
            from xgboost import XGBClassifier
            
            model = XGBClassifier(
                objective="multi:softprob",
                max_depth=6,
                n_estimators=200,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=42
            )
            model.fit(X, y)
            return model, {"algo": "xgb"}
        except Exception as e:
            print(f"⚠️ XGBoost unavailable ({e}), fallback to LogisticRegression")
    
    # Fallback: LogisticRegression
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(max_iter=4000, random_state=42)
    model.fit(X, y)
    return model, {"algo": "logreg"}


def main():
    parser = argparse.ArgumentParser(
        description="Train Drum ML Model from Stage2 features"
    )
    parser.add_argument(
        '--features-parquet',
        type=Path,
        required=True,
        help='Stage2 rhythm features parquet (rhythm_features_merged.parquet)'
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('data/patterns/stage2_drums_ml.pickle'),
        help='Output pickle path'
    )
    parser.add_argument(
        '--algo',
        type=str,
        default='auto',
        choices=['auto', 'xgb', 'logreg'],
        help='Training algorithm (auto: XGBoost→fallback LogisticRegression)'
    )
    
    args = parser.parse_args()
    
    # データロード
    print(f"📂 Loading: {args.features_parquet}")
    df = pd.read_parquet(args.features_parquet)
    print(f"✅ Loaded: {len(df)} records, {len(df.columns)} columns")
    
    # ターゲットカラム
    target_col = "family_label" if "family_label" in df.columns else "label"
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in parquet")
    
    # 特徴量抽出（数値列のみ）
    ignore_cols = {target_col, 'loop_id'}
    feature_cols = [
        c for c in df.columns 
        if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    print(f"\n📊 Features: {len(feature_cols)}")
    print(f"   {feature_cols[:10]}..." if len(feature_cols) > 10 else f"   {feature_cols}")
    
    # X, y準備
    X = df[feature_cols].astype(float).fillna(0.0).values
    y = df[target_col].astype(str).values
    
    print(f"\n🎯 Target: {target_col}")
    print(f"   Classes: {sorted(pd.unique(y).tolist())}")
    
    # 学習
    print(f"\n🔧 Training with algo='{args.algo}'...")
    model, meta = train_model(args.algo, X, y)
    print(f"✅ Training completed: {meta}")
    
    # クラスラベル
    class_labels = sorted(pd.unique(y).astype(str).tolist())
    
    # Pickle保存（統一フォーマット）
    pkg = {
        "schema_version": "stage2_drums_v1",
        "model_meta": meta,
        "model": model,
        "class_labels": class_labels,
        "feature_names": feature_cols,
        "target_col": target_col
    }
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.out, 'wb') as f:
        pickle.dump(pkg, f)
    
    print(f"\n💾 Saved: {args.out}")
    print(f"   Classes: {len(class_labels)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Schema: {pkg['schema_version']}")


if __name__ == '__main__':
    main()
