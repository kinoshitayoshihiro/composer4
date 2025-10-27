#!/usr/bin/env python3
"""
Simple Pattern Recommender - Stage2 Pickle形式対応

新しいpickle形式（version 1.0）用の軽量な推薦システム。
従来のPatternRecommenderと互換性を持ちつつ、新形式に対応。

Pickle構造:
    {
        'version': '1.0',
        'selector': {
            'type': 'rule_based',
            'lookup_table': {(section, root, quality, tempo_bin) -> pattern_id},
            'fallback': 'default_major'
        },
        'patterns': {
            pattern_id: {
                'key': str,
                'voicing': List[int],
                'rhythm': str,
                'metadata': {
                    'section': str,
                    'chord_root': str,
                    'chord_quality': str,
                    'tempo_bin': str,
                    'usage_count': int,
                    'avg_confidence': float,
                    'label_strength': str
                }
            }
        },
        'stats': {...}
    }
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SimplePatternRecommender:
    """Stage2 Pickle形式用パターン推薦システム（XGB/Sklearn対応）"""

    def __init__(self, instrument: str, patterns_path: str | Path):
        """
        Initialize recommender

        Args:
            instrument: 楽器名（bass/guitar/strings/melody/chords）
            patterns_path: パターンファイルパス（pickle）
        """
        self.instrument = instrument
        self.patterns_path = Path(patterns_path)

        # Load pickle
        self.data = self._load_pickle()

        # Extract components
        self.patterns = self.data.get("patterns", {})
        self.selector = self.data.get("selector", {})
        self.stats = self.data.get("stats", {})
        self.meta = self.data.get("meta", {})

        # Initialize model (for XGB/Sklearn selector)
        self._model = None
        self._feature_spec = None
        self._class_labels = None

        selector_type = self.selector.get("type", "rule_based")

        if selector_type in ("xgboost", "sklearn"):
            self._init_ml_selector()

        logger.info(f"Initialized SimplePatternRecommender for {instrument}")
        logger.info(f"  Version: {self.data.get('version')}")
        logger.info(f"  Total patterns: {len(self.patterns)}")
        logger.info(f"  Selector type: {selector_type}")

        if self._model is not None:
            logger.info(f"  Provider: {self.meta.get('provider', 'unknown')}")
            logger.info(f"  Model: {self.meta.get('selector_model', 'unknown')}")
            logger.info(f"  Classes: {len(self._class_labels) if self._class_labels else 0}")

    def _init_ml_selector(self):
        """Initialize ML-based selector (XGB/Sklearn)"""
        try:
            import joblib

            model_path = self.selector.get("path")
            if not model_path:
                logger.warning("ML selector specified but no model path found")
                return

            model_path = Path(model_path)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return

            self._model = joblib.load(model_path)
            
            # If model is a dict, extract the actual model object
            if isinstance(self._model, dict) and 'model' in self._model:
                logger.debug("Model is dict, extracting 'model' key")
                self._model = self._model['model']
            
            self._feature_spec = self.selector.get("feature_spec", {})
            self._class_labels = self.selector.get("class_labels", [])

            logger.info(f"  ✓ Loaded ML model from {model_path.name}")

        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
            self._model = None

    def _load_pickle(self) -> Dict[str, Any]:
        """Load pickle file"""
        if not self.patterns_path.exists():
            raise FileNotFoundError(f"Patterns file not found: {self.patterns_path}")

        with open(self.patterns_path, "rb") as f:
            data = pickle.load(f)

        return data

    def get_pattern(
        self,
        section: str = None,
        chord_root: str = None,
        chord_quality: str = None,
        tempo: float = None,
        confidence: float = 0.5,
        time_sig: str = "4/4",
        topk: int = 1,
        features: dict = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get pattern using ML or rule-based selector

        Args:
            section: Section name (Intro/Verse/Chorus/Bridge/Outro)
            chord_root: Chord root (C/D/E/...)
            chord_quality: Chord quality (maj/min/7/maj7/...)
            tempo: Tempo (BPM)
            confidence: Chord confidence (0.0-1.0)
            time_sig: Time signature
            topk: Number of recommendations (for ML selector)
            features: Optional features dict (merged with named args)

        Returns:
            Pattern dict or None
        """
        # Merge named args with features dict
        if features is None:
            features = {}
        
        if section is not None:
            features.setdefault("section", section)
        if chord_root is not None:
            features.setdefault("chord_root", chord_root)
        if chord_quality is not None:
            features.setdefault("chord_quality", chord_quality)
        if tempo is not None:
            features.setdefault("tempo", tempo)
        if confidence != 0.5:
            features.setdefault("confidence", confidence)
        if time_sig != "4/4":
            features.setdefault("time_sig", time_sig)
        
        # Extract values (with defaults)
        section = features.get("section", "Verse")
        chord_root = features.get("chord_root", "C")
        chord_quality = features.get("chord_quality", "maj")
        tempo = features.get("tempo", 120.0)
        confidence = features.get("confidence", 0.5)
        time_sig = features.get("time_sig", "4/4")
        
        # Try ML selector first
        if self._model is not None:
            result = self._ml_recommend(
                section=section,
                chord_root=chord_root,
                chord_quality=chord_quality,
                tempo=tempo,
                confidence=confidence,
                time_sig=time_sig,
                topk=topk,
                features=features,  # Pass full features for reranking
            )
            if result:
                return result

            logger.debug("ML selector failed, falling back to rule-based")

        # Fallback to rule-based selector
        return self._rule_recommend(
            section=section, chord_root=chord_root, chord_quality=chord_quality, tempo=tempo
        )
    
    def recommend(
        self, 
        features: dict, 
        topk: int = 1, 
        filter_v3_only: bool = False,
        min_proba: float = 0.15,
        min_margin: float = 0.10
    ) -> Optional[Dict[str, Any]]:
        """ML-based recommendation with Top-K reranking
        
        Args:
            features: Features dict with keys: section, chord_root, chord_quality, tempo, 
                     target_accent, target_density_ql, rerank_w_proba, etc.
            topk: Number of recommendations
            filter_v3_only: top1_proba=1.0のパターンのみ推薦（Phase 24横展開、デフォルトFalse）
            min_proba: 最小確率閾値（絶対KPI評価、filter_v3_only=True時有効）
            min_margin: 最小マージン閾値（絶対KPI評価、filter_v3_only=True時有効）
        
        Returns:
            Pattern dict or None (falls back to get_pattern if ML unavailable or threshold not met)
        """
        # Check if ML selector is available
        sel = self.selector or {}
        sel_type = (sel.get("type") or "").lower()
        
        if sel_type not in ("xgboost", "sklearn"):
            # Fallback to rule-based
            logger.debug("ML selector not available, using rule-based")
            return self.get_pattern(features=features, topk=topk)
        
        # Use ML recommendation
        ml_result = self._ml_recommend(
            section=features.get("section", "Unknown"),
            chord_root=features.get("chord_root", "C"),
            chord_quality=features.get("chord_quality", "maj"),
            tempo=features.get("tempo", 120.0),
            confidence=features.get("confidence", 0.8),
            time_sig=features.get("time_sig", "4/4"),
            topk=topk,
            features=features,
            filter_v3_only=filter_v3_only,
            min_proba=min_proba,
            min_margin=min_margin,
        )
        
        # If ML reranking returned None (threshold fallback), use rule-based
        if ml_result is None:
            logger.debug("ML reranking threshold not met, falling back to rule-based")
            return self.get_pattern(features=features, topk=topk)
        
        return ml_result

    def _ml_recommend(
        self,
        section: str,
        chord_root: str,
        chord_quality: str,
        tempo: float,
        confidence: float,
        time_sig: str,
        topk: int,
        features: dict = None,
        filter_v3_only: bool = False,
        min_proba: float = 0.15,
        min_margin: float = 0.10,
    ) -> Optional[Dict[str, Any]]:
        """ML-based recommendation (XGB/Sklearn) with Top-K reranking"""
        try:
            # Encode features (merge with external features)
            base_features = {
                "section": section,
                "chord_root": chord_root,
                "chord_quality": chord_quality,
                "tempo": tempo,
                "tempo_bin": self._get_tempo_bin(tempo),
                "confidence": confidence,
                "time_sig": time_sig,
            }
            
            # Merge external features (for reranking: target_accent, target_density_ql, etc.)
            if features:
                base_features.update(features)

            X = self._encode_features(base_features)

            # Predict Top-K
            if hasattr(self._model, "predict_proba"):
                import numpy as np

                proba = self._model.predict_proba(X)[0]  # shape: [C]
                # Get Top-K candidates (at least 3 for reranking)
                topk_candidates = max(3, topk)
                idx = np.argsort(-proba)[: topk_candidates]
                predictions = [(self._class_labels[i], float(proba[i])) for i in idx]
            else:
                y = self._model.predict(X)[0]
                predictions = [(str(self._class_labels[int(y)]), 1.0)]

            # Rerank with context (accent/density/section fit)
            reranked = self._rerank_with_context(predictions, base_features)
            
            # Phase 24.1: V3フィルタ適用（top1_proba=1.0のみ）
            if filter_v3_only and reranked:
                reranked = self._filter_v3_patterns_simple(reranked, min_proba, min_margin)
                if not reranked:
                    logger.debug("V3 filter: No patterns passed KPI threshold, using fallback")
                    return None
            
            # Debug: 再ランク結果をログ出力
            if reranked:
                logger.debug(f"Reranked Top-3: {[(r['pattern_id'], r.get('confidence', 0.0)) for r in reranked[:3]]}")
            else:
                logger.debug("Reranking returned empty (threshold fallback)")
            
            # Return top-1 from reranked results (empty means fallback to rule-based)
            return reranked[0] if reranked else None

        except Exception as e:
            logger.debug(f"ML recommendation failed: {e}")
            return None

    def _encode_features(self, features: dict) -> list:
        """Encode features for ML model"""
        spec = self._feature_spec or {}
        order = spec.get(
            "order",
            [
                "section",
                "chord_root",
                "chord_quality",
                "tempo_bin",
                "confidence",
                "time_sig",
                "tempo",
            ],
        )
        types = spec.get("types", {})
        encoders = spec.get("encoders", {})

        vec = []
        for key in order:
            val = features.get(key)
            typ = types.get(key, "cat")

            if typ == "num":
                try:
                    vec.append(float(val))
                except:
                    vec.append(0.0)
            else:
                # Categorical encoding
                enc = encoders.get(key, {})
                
                # Handle sklearn LabelEncoder
                if hasattr(enc, 'transform'):
                    try:
                        val_str = str(val) if not isinstance(val, str) else val
                        # Check if value is in classes_
                        if hasattr(enc, 'classes_') and val_str in enc.classes_:
                            idx = float(enc.transform([val_str])[0])
                        else:
                            # Unknown value → use 0
                            idx = 0.0
                        vec.append(idx)
                    except Exception as e:
                        logger.debug(f"LabelEncoder transform failed for {key}={val}: {e}")
                        vec.append(0.0)
                else:
                    # Dict-based encoding
                    if isinstance(val, str):
                        idx = enc.get(val, enc.get("__UNK__", 0))
                    else:
                        idx = enc.get(str(val), enc.get("__UNK__", 0))
                    vec.append(float(idx))

        return [vec]

    def _rerank_with_context(self, preds, features):
        """
        Rerank Top-K predictions by musical context (accent/density/section fit)
        
        Args:
            preds: [(pattern_id, proba), ...]
            features: dict with keys:
                - section (str): Target section (Verse/Chorus/...)
                - target_accent (list[int]): Target accent pattern (16th notes, 0/1)
                - target_density_ql (float): Target density (QL/bar)
                - rerank_w_proba (float): Weight for ML confidence (default: 0.60)
                - rerank_w_accent (float): Weight for accent fit (default: 0.25)
                - rerank_w_density (float): Weight for density fit (default: 0.10)
                - rerank_w_section (float): Weight for section fit (default: 0.05)
                - rerank_conf_thresh (float): Confidence threshold for fallback (default: 0.35)
        
        Returns:
            List of dicts with pattern_id, pattern, confidence, sorted by score
        """
        import numpy as np
        
        section = features.get("section", "Unknown")
        tgt_acc = np.array(features.get("target_accent", []), dtype=float)
        tgt_den = float(features.get("target_density_ql", 0.0))
        
        # If no accent target, just return materialized patterns by probability
        if tgt_acc.size == 0:
            return self._materialize(preds)
        
        # Reranking weights
        w_proba = float(features.get("rerank_w_proba", 0.60))
        w_accent = float(features.get("rerank_w_accent", 0.25))
        w_density = float(features.get("rerank_w_density", 0.10))
        w_section = float(features.get("rerank_w_section", 0.05))
        threshold = float(features.get("rerank_conf_thresh", 0.35))
        
        scored = []
        
        for pid, p in preds:
            pat = self.patterns.get(pid) or self.patterns.get(self._alias(pid))
            if not pat:
                logger.debug(f"Pattern {pid} not found, skipping")
                continue
            
            # Extract pattern metadata
            acc_base = np.array(pat.get("accent_profile", []), dtype=float)
            
            # ▼ パターンメタ正規化: 値域を0..1にクリップ
            if acc_base.size > 0:
                acc_base = np.clip(acc_base, 0.0, 1.0)
            
            if acc_base.size == 0 or acc_base.size != tgt_acc.size:
                # Fallback: downbeat-emphasized pattern (4/4 assumed, 16th notes × 16)
                acc_base = np.array([1.0 if i % (tgt_acc.size // 4) == 0 else 0.0 for i in range(tgt_acc.size)], dtype=float)
            
            # ▼ 円環シフト（circular shift）で最良一致を採用
            # アクセントパターンを1スロットずつずらして最大cos類似度を探索
            best_accent_score = 0.0
            best_shift = 0
            
            if tgt_acc.size > 0 and acc_base.size > 0:
                norm_tgt = np.linalg.norm(tgt_acc)
                if norm_tgt > 1e-6:
                    for shift in range(tgt_acc.size):
                        acc_shifted = np.roll(acc_base, shift)
                        norm_acc = np.linalg.norm(acc_shifted)
                        if norm_acc > 1e-6:
                            cos_sim = float(np.dot(acc_shifted, tgt_acc) / (norm_acc * norm_tgt))
                            if cos_sim > best_accent_score:
                                best_accent_score = cos_sim
                                best_shift = shift
            
            accent_score = best_accent_score
            chosen_shift = best_shift
            
            # Density fit
            den = float(pat.get("density_ql_per_bar", 0.0))
            if tgt_den <= 0.0 or den <= 0.0:
                density_score = 0.5  # Neutral
            else:
                density_score = 1.0 - min(1.0, abs(den - tgt_den) / max(tgt_den, 1.0))
            
            # Section fit
            allow = pat.get("allowed_sections", None)
            section_score = 1.0 if (not allow or section in allow) else 0.0
            
            # Total score
            score = (w_proba * p) + (w_accent * accent_score) + (w_density * density_score) + (w_section * section_score)
            scored.append((pid, p, score, chosen_shift))
        
        if not scored:
            return self._materialize(preds)
        
        # Check top-1 ML confidence threshold (use proba, not total score)
        top1 = max(scored, key=lambda t: t[2])  # Sort by total score
        top1_proba = top1[1]  # ML probability
        
        # ▼ 低確率セーフティ: top1_proba < 0.15 の場合、フォールバック
        #    （本番投入時の保険、ほぼ発動しない想定）
        SAFETY_THRESHOLD = 0.15
        if top1_proba < SAFETY_THRESHOLD:
            logger.warning(f"Low confidence safety: top1_proba={top1_proba:.3f} < {SAFETY_THRESHOLD}, fallback to safe-kit")
            return []  # safe-kitへフォールバック（recommend()で処理）
        
        # 通常のthresholdチェック（本番ではthreshold=0.0なので常にスキップ）
        if top1_proba < threshold:
            logger.debug(f"Top-1 ML proba ({top1_proba:.3f}) < threshold ({threshold}), v1 fallback")
            return []  # Fallback to rule-based in recommend()
        
        # ▼ アクセント劣化防止ガード: トップ候補のアクセント一致度が低く、
        #    より良いアクセント候補が僅差で存在する場合、アクセント優先に差し替え
        accent_scores = []
        for pid, p, s, shift in scored:
            # 位相最適込みのアクセント一致度を再計算
            pat = self.patterns.get(pid, {})
            acc_prof = pat.get("accent_profile", [])
            if acc_prof:
                acc_base = np.array(acc_prof, dtype=float)
                acc_base = np.clip(acc_base, 0.0, 1.0)
                tgt_acc = np.array(features.get("target_accent", []), dtype=float)
                if acc_base.size > 0 and tgt_acc.size > 0:
                    norm_tgt = np.linalg.norm(tgt_acc)
                    if norm_tgt > 1e-6:
                        best_cos = 0.0
                        for sh in range(tgt_acc.size):
                            acc_shifted = np.roll(acc_base, sh)
                            norm_acc = np.linalg.norm(acc_shifted)
                            if norm_acc > 1e-6:
                                cos_sim = float(np.dot(acc_shifted, tgt_acc) / (norm_acc * norm_tgt))
                                best_cos = max(best_cos, cos_sim)
                        accent_scores.append((pid, best_cos))
                    else:
                        accent_scores.append((pid, 0.5))
                else:
                    accent_scores.append((pid, 0.5))
            else:
                accent_scores.append((pid, 0.5))
        
        if accent_scores:
            best_accent_idx = max(range(len(accent_scores)), key=lambda i: accent_scores[i][1])
            best_accent_score = accent_scores[best_accent_idx][1]
            top_accent_only = accent_scores[0][1]
            
            # アクセント劣化が大きい（Δ ≥ 0.10）場合、アクセント最良を採用
            if (best_accent_score - top_accent_only) >= 0.10:
                logger.debug(f"Accent guard: swapping top (accent={top_accent_only:.3f}) with #{best_accent_idx+1} (accent={best_accent_score:.3f})")
                # トップとアクセント最良を入れ替え
                scored[0], scored[best_accent_idx] = scored[best_accent_idx], scored[0]
        
        # Sort by score descending (既にソート済みだが、ガード後に再ソート不要）
        # scored.sort(key=lambda t: -t[2])  # 既にソート済み
        
        # Log top-3
        if scored and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Top-3 reranked patterns for {section}:")
            for i, (pid, p, s, shift) in enumerate(scored[:3], 1):
                logger.debug(f"  #{i}: {pid} (score={s:.3f}, proba={p:.3f}, shift={shift})")
        
        return self._materialize([(pid, p, shift) for pid, p, _, shift in scored])
    
    def _materialize(self, pid_probas):
        """
        Convert (pattern_id, proba[, phase_shift]) list to pattern dict list
        
        Args:
            pid_probas: List of tuples, either:
                - (pattern_id, confidence) for backward compatibility
                - (pattern_id, confidence, phase_slots) for phase-aware reranking
        
        Returns:
            List of dicts with pattern_id, pattern, confidence, phase_slots
        """
        out = []
        for item in pid_probas:
            # Backward compatibility: (pid, conf) or (pid, conf, phase_slots)
            pid = item[0]
            conf = item[1]
            phase = int(item[2]) if len(item) > 2 else 0
            
            pat = self.patterns.get(pid) or self.patterns.get(self._alias(pid))
            if pat:
                out.append({
                    "pattern_id": pid,
                    "pattern": pat,
                    "confidence": float(conf),
                    "phase_slots": phase
                })
        return out
    
    def _alias(self, pid: str) -> str:
        """Pattern ID aliasing (for family/variant support)"""
        # Example: speed family or simple aliasing (extend as needed)
        return pid

    def _get_tempo_bin(self, tempo: float) -> str:
        """Get tempo bin"""
        if tempo < 90:
            return "slow"
        elif tempo < 120:
            return "mid"
        elif tempo < 150:
            return "fast"
        else:
            return "very_fast"

    def _rule_recommend(
        self, section: str, chord_root: str, chord_quality: str, tempo: float
    ) -> Optional[Dict[str, Any]]:
        """Rule-based recommendation (fallback)"""
        if self.selector.get("type") not in ("rule_based", "xgboost", "sklearn"):
            logger.warning(f"Unsupported selector type: {self.selector.get('type')}")
            return None

        tempo_bin = self._get_tempo_bin(tempo)

        # Lookup
        lookup_table = self.selector.get("lookup_table", {})
        key = (section, chord_root, chord_quality, tempo_bin)

        # Try exact match
        result = lookup_table.get(key)

        if result:
            pattern_id = result["pattern_id"]
            pattern = self.patterns.get(pattern_id)
            if pattern:
                return {
                    "pattern_id": pattern_id,
                    "confidence": result.get("confidence", 0.5),
                    **pattern,
                }

        # Fallback to default
        fallback_info = self.selector.get("fallback", {})
        if isinstance(fallback_info, dict):
            fallback_id = fallback_info.get("pattern_id", "default_major")
        else:
            fallback_id = "default_major"

        fallback = self.patterns.get(fallback_id)

        if fallback:
            logger.debug(f"Using fallback pattern: {fallback_id}")
            return {"pattern_id": fallback_id, "confidence": 0.3, **fallback}

        return None
    
    def _filter_v3_patterns_simple(
        self, 
        patterns: List[Dict[str, Any]], 
        min_proba: float, 
        min_margin: float
    ) -> List[Dict[str, Any]]:
        """
        top1_proba=1.0のパターンのみ抽出し、KPI評価（Phase 24.1横展開）
        
        Args:
            patterns: 候補パターンリスト（reranked results）
            min_proba: 最小確率閾値
            min_margin: 最小マージン閾値
        
        Returns:
            KPI合格パターンのみのリスト
        """
        v3_patterns = []
        
        for pattern_dict in patterns:
            pattern_id = pattern_dict.get('pattern_id')
            pattern_data = self.patterns.get(pattern_id, {})
            metadata = pattern_data.get('metadata', {})
            
            # Extract top1_proba, top2_proba
            top1_proba = metadata.get('top1_proba', 0.0)
            top2_proba = metadata.get('top2_proba', 0.0)
            
            # V3フィルタ（top1_proba=1.0）
            if top1_proba < 0.999:
                continue
            
            # KPI評価
            proba_margin = top1_proba - top2_proba
            kpi_passed = (top1_proba >= min_proba) and (proba_margin >= min_margin)
            
            if kpi_passed:
                # Add KPI info to pattern_dict
                pattern_dict['top1_proba'] = top1_proba
                pattern_dict['top2_proba'] = top2_proba
                pattern_dict['proba_margin'] = proba_margin
                pattern_dict['kpi_passed'] = True
                v3_patterns.append(pattern_dict)
                
                logger.debug(f"V3 KPI passed: {pattern_id} proba={top1_proba:.3f} margin={proba_margin:.3f}")
            else:
                logger.debug(f"V3 KPI failed: {pattern_id} proba={top1_proba:.3f} margin={proba_margin:.3f}")
        
        return v3_patterns

    def get_patterns_by_section(self, section: str) -> List[Dict[str, Any]]:
        """
        Get all patterns for a section

        Args:
            section: Section name

        Returns:
            List of patterns
        """
        results = []
        for pattern_id, pattern in self.patterns.items():
            if pattern.get("metadata", {}).get("section") == section:
                results.append({"pattern_id": pattern_id, **pattern})

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pattern statistics

        Returns:
            Statistics dict
        """
        return {
            "total_patterns": len(self.patterns),
            "selector_type": self.selector.get("type"),
            "stats": self.stats,
            "data_source": self.data.get("data_source", {}),
        }
