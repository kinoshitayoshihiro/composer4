#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 Piano Baseline Trainer: XGBoost / LogisticRegression
Phase: 27.2 (全楽器ML学習)

入力:
  - data/datasets/piano_train.parquet
  - data/datasets/piano_val.parquet
  
出力:
  - data/patterns/stage2_piano.pickle

使用方法:
  python scripts/train_piano_baseline.py
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Any

import pandas as pd
import numpy as np


# ===== アルゴリズム検出＆モデル学習 =====

def _detect_algo(algo: str) -> str:
    if algo and algo.lower() in {"xgb", "logreg"}:
        return algo.lower()
    try:
        import xgboost  # noqa
        return "xgb"
    except ImportError:
        return "logreg"


def _load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")
    df = pd.read_parquet(p)
    print(f"[INFO] Loaded parquet: {p} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def _split_xy(df: pd.DataFrame, targets=("family", "label")) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    # ターゲット列検出
    tgt = None
    for c in targets:
        if c in df.columns:
            tgt = c
            break
    if tgt is None:
        raise KeyError(f"Target column not found (tried: {targets})")
    
    # 特徴量列選択（数値型のみ、ID/ターゲット列除外）
    ignore = {tgt, "pattern_id", "song_id", "track_id", "bar_index"}
    feats = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    # 欠損値埋め
    X = df[feats].astype(float).fillna(0.0).values
    y = df[tgt].astype(str).values
    
    print(f"[INFO] Features: {len(feats)} cols")
    print(f"[INFO] Target: {tgt} ({len(pd.unique(y))} classes)")
    print(f"[INFO] Samples: {len(y)}")
    
    return X, y, feats, tgt


def _train_model(algo: str, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict]:
    algo = _detect_algo(algo)
    
    if algo == "xgb":
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
                n_jobs=-1,
                random_state=42
            )
            
            print(f"[INFO] Training XGBoost (n_estimators=200, max_depth=6)...")
            model.fit(X, y)
            
            meta = {
                "algo": "xgb",
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.08
            }
            
            print(f"[INFO] XGBoost training complete.")
            return model, meta
            
        except ImportError as e:
            print(f"[WARN] XGBoost unavailable, fallback to LogReg: {e}")
    
    # LogisticRegression
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(
        max_iter=4000,
        multi_class="auto",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1
    )
    
    print(f"[INFO] Training LogisticRegression (max_iter=4000)...")
    model.fit(X, y)
    
    meta = {
        "algo": "logreg",
        "max_iter": 4000,
        "solver": "lbfgs"
    }
    
    print(f"[INFO] LogisticRegression training complete.")
    return model, meta


# ===== Pattern Dict構築 =====

def _build_pattern_dict(
    df: pd.DataFrame,
    family: str = "family",
    pid: str = "pattern_id",
    topk: int = 32
) -> Dict[str, List[str]]:
    if family not in df.columns or pid not in df.columns:
        return {}
    
    out = {}
    grp = (
        df.groupby([family, pid])
        .size()
        .reset_index(name="n")
        .sort_values([family, "n"], ascending=[True, False])
    )
    
    for fam, sub in grp.groupby(family):
        out[str(fam)] = [str(x) for x in sub.head(topk)[pid].tolist()]
    
    print(f"[INFO] Pattern dict built: {len(out)} families")
    for fam, pids in out.items():
        print(f"  - {fam}: {len(pids)} patterns")
    
    return out


# ===== Pickle保存 =====

def _save_pickle(
    path: Path,
    model: Any,
    class_labels: List[str],
    feature_names: List[str],
    target_col: str,
    pattern_dict: Dict[str, List[str]],
    model_meta: Dict
):
    pkg = {
        "schema_version": "v1",
        "model_meta": model_meta,
        "model": model,
        "class_labels": class_labels,
        "feature_names": feature_names,
        "target_col": target_col,
        "pattern_dict": pattern_dict
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"[OK] Saved Stage2 pickle: {path}")
    print(f"  - Schema: {pkg['schema_version']}")
    print(f"  - Algo: {model_meta['algo']}")
    print(f"  - Classes: {len(class_labels)}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Patterns: {sum(len(pids) for pids in pattern_dict.values())} total")


# ===== Probas保存 =====

def _save_train_probas(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    class_labels: List[str],
    pickle_path: Path
):
    """学習セット確率保存（QC用）"""
    try:
        if not hasattr(model, "predict_proba"):
            print("[WARN] Model does not support predict_proba, skipping probas save")
            return
        
        probas = model.predict_proba(X)
        proba_df = pd.DataFrame(
            probas,
            columns=[f"proba_{cls}" for cls in class_labels]
        )
        proba_df["y_true"] = y
        
        probas_path = pickle_path.with_suffix(".train_probas.parquet")
        proba_df.to_parquet(probas_path)
        print(f"[OK] Saved train-set probas: {probas_path}")
    except Exception as e:
        print(f"[WARN] Failed to save train-set probas: {e}")


# ===== メイン処理 =====

def main():
    ap = argparse.ArgumentParser(description="Train Piano Baseline (XGBoost/LogReg)")
    ap.add_argument("--train", default="data/datasets/piano_train.parquet", help="Training parquet path")
    ap.add_argument("--val", default="data/datasets/piano_val.parquet", help="Validation parquet path (optional)")
    ap.add_argument("--out", default="data/patterns/stage2_piano.pickle", help="Output pickle path")
    ap.add_argument("--algo", default="auto", choices=["auto", "xgb", "logreg"], help="Algorithm selection")
    ap.add_argument("--topk", type=int, default=32, help="Top-K patterns per family")
    ap.add_argument("--save-probas", action="store_true", help="Save train-set probas for QC")
    args = ap.parse_args()
    
    print("=" * 80)
    print("Stage2 Piano Baseline Trainer (Phase 27.2)")
    print("=" * 80)
    
    # 1. Parquet読み込み
    df = _load_parquet(args.train)
    
    # Validation parquetがあれば結合（オプション）
    if args.val and Path(args.val).exists():
        val_df = _load_parquet(args.val)
        df = pd.concat([df, val_df], ignore_index=True)
        print(f"[INFO] Combined train+val: {len(df)} rows")
    
    # 2. 特徴量とターゲットに分割
    X, y, feat_names, target_col = _split_xy(df)
    
    # 3. モデル学習
    model, meta = _train_model(args.algo, X, y)
    
    # 4. クラスラベル取得
    class_labels = sorted(pd.unique(y).astype(str).tolist())
    print(f"[INFO] Class labels: {class_labels}")
    
    # 5. Pattern Dict構築
    pattern_dict = _build_pattern_dict(df, topk=args.topk)
    
    # 6. Pickle保存
    _save_pickle(
        path=Path(args.out),
        model=model,
        class_labels=class_labels,
        feature_names=feat_names,
        target_col=target_col,
        pattern_dict=pattern_dict,
        model_meta=meta
    )
    
    print("=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    
    # 6.5. 学習セット確率保存（オプション）
    if args.save_probas:
        print("[INFO] Saving train-set probas for QC...")
        _save_train_probas(model, X, y, class_labels, Path(args.out))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
