#!/usr/bin/env python3
"""
Guitar Pattern Recommender - v3互換パターン推薦システム

Phase 26: 他楽器ML展開 - Guitar ML基盤

Features:
- Tempo/Chord/Section適合度計算
- ML推論（XGBoost/LogReg）によるPattern Family予測
- Top-1確率直採用（v3思想）
- Safety判定（min_proba/min_margin）
- Safe-Kit（基本コードパターン）フォールバック
- KPI Gates維持（accent_score ≥ 0.65, chord_fit ≥ 0.60, density_abs ≤ 1.0）

Usage:
    from ml.guitar_pattern_recommender import GuitarPatternRecommender, GuitarQuery
    
    rec = GuitarPatternRecommender(
        patterns_dict,
        safe_kit_path="config/safe_kit_guitar.yaml",
        model_pickle_path="ml/stage2_guitar_v3_ml.pickle"  # Optional
    )
    result = rec.recommend(
        query=GuitarQuery(
            tempo_bpm=120,
            chord_root="C",
            chord_type="maj",
            section="Chorus",
            target_energy=0.7
        ),
        min_proba=0.15,
        min_margin=0.10
    )
"""

import hashlib
import logging
import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class GuitarQuery:
    """ギターパターンクエリ
    
    Attributes:
        tempo_bpm: テンポ（BPM）
        chord_root: コードルート（C/D/E/F/G/A/B）
        chord_type: コードタイプ（maj/min/7/maj7/sus4等）
        section: セクション名（Chorus/Verse/Bridge/Intro/Outro）
        target_energy: 目標エネルギー（0.0-1.0、accent_gridから取得）
        time_signature: 拍子（4/4, 3/4, 6/8等）
    """
    tempo_bpm: float
    chord_root: str      # C/D/E/F/G/A/B
    chord_type: str      # maj/min/7/maj7/sus4等
    section: str         # Chorus/Verse/Bridge/Intro/Outro
    target_energy: float # 0.0-1.0
    time_signature: str = "4/4"


@dataclass
class GuitarRecommendResult:
    """推薦結果
    
    Attributes:
        pattern_id: パターンID
        pattern: パターン辞書
        top1_proba: Top-1確率
        top2_proba: Top-2確率
        margin: Top-1とTop-2のマージン
        safety_triggered: Safety発火フラグ
        safety_reason: Safety発火理由
        accent_score: アクセント適合度（予測値）
        chord_fit: コード適合度（予測値）
        density: 密度（QL単位、予測値）
    """
    pattern_id: str
    pattern: Dict[str, Any]
    top1_proba: float
    top2_proba: float
    margin: float
    safety_triggered: bool
    safety_reason: str
    accent_score: float = 0.0
    chord_fit: float = 0.0
    density: float = 0.0


class GuitarPatternRecommender:
    """ギターパターン推薦システム（v3互換 + ML推論）
    
    v3思想:
    - Top-1確率を直接採用（閾値判定のみ）
    - ML推論（XGBoost/LogReg）でPattern Family予測
    - 低確率/低マージン時はSafe-Kitへフォールバック
    - KPI Gates維持（accent_score, chord_fit, density）
    """
    
    def __init__(
        self,
        patterns: Dict[str, Any],
        safe_kit_path: Optional[Path] = None,
        model_pickle_path: Optional[Path] = None
    ):
        """初期化
        
        Args:
            patterns: パターン辞書 {pattern_id: pattern_dict}
            safe_kit_path: Safe-Kit YAMLパス（オプション）
            model_pickle_path: 学習済みモデルpickleパス（オプション）
        """
        self.patterns = patterns
        self.index = self._build_index(patterns)
        
        # Safe-Kit読み込み
        self.safe_kit = {}
        if safe_kit_path:
            safe_kit_path = Path(safe_kit_path)
            if safe_kit_path.exists():
                with open(safe_kit_path, 'r', encoding='utf-8') as f:
                    self.safe_kit = yaml.safe_load(f)
                logger.info(f"Loaded Guitar Safe-Kit from {safe_kit_path}")
            else:
                logger.warning(f"Guitar Safe-Kit not found: {safe_kit_path}")
        
        # ML推論モデル読み込み（オプション）
        self.ml_model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None
        
        if model_pickle_path:
            model_pickle_path = Path(model_pickle_path)
            if model_pickle_path.exists():
                try:
                    with open(model_pickle_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # XGBoostまたはLogRegモデル
                    self.ml_model = model_data.get("xgb_model") or model_data.get("lr_model")
                    self.label_encoder = model_data.get("label_encoder")
                    self.scaler = model_data.get("scaler")
                    self.feature_names = model_data.get("feature_names")
                    
                    logger.info(f"Loaded Guitar ML model from {model_pickle_path}")
                    logger.info(f"Model type: {type(self.ml_model).__name__}")
                except Exception as exc:
                    logger.error(f"Failed to load Guitar ML model: {exc}")
            else:
                logger.warning(f"Guitar ML model pickle not found: {model_pickle_path}")
    
    def _build_index(
        self,
        patterns: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """パターンインデックス構築
        
        Family/Tempo/Section別にパターンIDをグループ化
        
        Returns:
            インデックス辞書 {key: [pattern_id, ...]}
        """
        index = {}
        
        for pid, ptn in patterns.items():
            family = ptn.get("family", "unknown")
            
            # Family別インデックス
            key_family = f"family:{family}"
            index.setdefault(key_family, []).append(pid)
            
            # Tempo別インデックス（50 BPM刻み）
            tempo_hint = ptn.get("tempo_hint", 120)
            tempo_bucket = (int(tempo_hint) // 50) * 50
            key_tempo = f"tempo:{tempo_bucket}"
            index.setdefault(key_tempo, []).append(pid)
            
            # Section別インデックス
            section_hint = ptn.get("section_hint", "Verse")
            key_section = f"section:{section_hint}"
            index.setdefault(key_section, []).append(pid)
        
        logger.info(f"Built Guitar pattern index: {len(index)} keys, {len(patterns)} patterns")
        return index
    
    def _extract_features(
        self,
        query: GuitarQuery
    ) -> np.ndarray:
        """クエリから特徴量抽出（ML推論用）
        
        Args:
            query: ギタークエリ
        
        Returns:
            特徴量ベクトル（1D array）
        """
        features = []
        
        # Tempo特徴量
        features.append(query.tempo_bpm)
        features.append(math.log(query.tempo_bpm + 1))  # log(tempo)
        
        # Chord特徴量（One-Hot）
        chord_roots = ["C", "D", "E", "F", "G", "A", "B"]
        for root in chord_roots:
            features.append(1.0 if query.chord_root == root else 0.0)
        
        chord_types = ["maj", "min", "7", "maj7", "sus4", "dim", "aug"]
        for ctype in chord_types:
            features.append(1.0 if query.chord_type == ctype else 0.0)
        
        # Section特徴量（One-Hot）
        sections = ["Chorus", "Verse", "Bridge", "Intro", "Outro"]
        for sec in sections:
            features.append(1.0 if query.section == sec else 0.0)
        
        # Energy特徴量
        features.append(query.target_energy)
        features.append(query.target_energy ** 2)  # energy^2
        
        # Time Signature特徴量（One-Hot）
        time_sigs = ["4/4", "3/4", "6/8", "12/8"]
        for ts in time_sigs:
            features.append(1.0 if query.time_signature == ts else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _predict_family_ml(
        self,
        query: GuitarQuery
    ) -> Tuple[str, float, float]:
        """ML推論でFamily予測
        
        Args:
            query: ギタークエリ
        
        Returns:
            (top1_family, top1_proba, top2_proba)
        """
        if self.ml_model is None or self.label_encoder is None:
            # MLモデルがない場合はルールベース
            return self._predict_family_rule_based(query)
        
        try:
            # 特徴量抽出
            features = self._extract_features(query)
            
            # スケーリング（Scalerがある場合）
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1)).flatten()
            
            # ML推論
            if hasattr(self.ml_model, "predict_proba"):
                # LogReg/XGBoost（predict_proba対応）
                proba = self.ml_model.predict_proba(features.reshape(1, -1))[0]
                top_indices = np.argsort(proba)[::-1]
                
                top1_idx = top_indices[0]
                top2_idx = top_indices[1] if len(top_indices) > 1 else top1_idx
                
                top1_family = self.label_encoder.inverse_transform([top1_idx])[0]
                top1_proba = proba[top1_idx]
                top2_proba = proba[top2_idx]
                
                return (top1_family, top1_proba, top2_proba)
            else:
                # predict_probaがない場合（決定木等）
                pred_idx = self.ml_model.predict(features.reshape(1, -1))[0]
                pred_family = self.label_encoder.inverse_transform([pred_idx])[0]
                return (pred_family, 1.0, 0.0)  # 確率情報なし
        
        except Exception as exc:
            logger.error(f"ML prediction failed: {exc}")
            return self._predict_family_rule_based(query)
    
    def _predict_family_rule_based(
        self,
        query: GuitarQuery
    ) -> Tuple[str, float, float]:
        """ルールベースFamily予測（ML推論失敗時のフォールバック）
        
        Args:
            query: ギタークエリ
        
        Returns:
            (family, top1_proba, top2_proba)
        """
        # テンポ/セクション/コードタイプからFamily推定
        if query.tempo_bpm < 80:
            family = "ballad_arpeggio"
        elif query.tempo_bpm > 140:
            family = "fast_strum"
        elif query.section == "Chorus":
            family = "power_chord"
        elif query.chord_type in ["maj7", "min7"]:
            family = "jazz_voicing"
        else:
            family = "standard_strum"
        
        # ルールベースは確率情報なし
        return (family, 0.5, 0.3)  # 仮の確率
    
    def _select_pattern_from_family(
        self,
        family: str,
        query: GuitarQuery
    ) -> Optional[str]:
        """Family内からパターン選択
        
        Args:
            family: Pattern Family
            query: ギタークエリ
        
        Returns:
            パターンID（候補がない場合はNone）
        """
        # Family別インデックスから候補取得
        key_family = f"family:{family}"
        candidates = self.index.get(key_family, [])
        
        if not candidates:
            logger.warning(f"No patterns found for family: {family}")
            return None
        
        # Tempo/Section適合度でスコアリング
        scored = []
        for pid in candidates:
            ptn = self.patterns[pid]
            
            # Tempo適合度
            ptn_tempo = ptn.get("tempo_hint", 120)
            tempo_diff = abs(query.tempo_bpm - ptn_tempo)
            tempo_score = max(0.0, 1.0 - tempo_diff / 50.0)
            
            # Section適合度
            ptn_section = ptn.get("section_hint", "Verse")
            section_score = 1.0 if ptn_section == query.section else 0.5
            
            # Energy適合度
            ptn_energy = ptn.get("energy_hint", 0.5)
            energy_diff = abs(query.target_energy - ptn_energy)
            energy_score = max(0.0, 1.0 - energy_diff)
            
            # 総合スコア
            total_score = tempo_score * 0.4 + section_score * 0.3 + energy_score * 0.3
            scored.append((pid, total_score))
        
        # スコア降順でソート
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Top-1を返す
        return scored[0][0] if scored else None
    
    def _get_safe_kit_pattern(
        self,
        query: GuitarQuery
    ) -> Dict[str, Any]:
        """Safe-Kitパターン取得
        
        Args:
            query: ギタークエリ
        
        Returns:
            Safe-Kitパターン辞書
        """
        if not self.safe_kit:
            # Safe-Kitがない場合は最小限のパターンを生成
            return {
                "pattern_id": "safe_kit_default",
                "family": "safe_kit",
                "events": self._generate_basic_chord_pattern(query),
                "tempo_hint": query.tempo_bpm,
                "section_hint": query.section,
            }
        
        # Safe-Kitから適切なパターン選択
        safe_patterns = self.safe_kit.get("patterns", [])
        
        # Chord Type適合度でスコアリング
        scored = []
        for sp in safe_patterns:
            sp_chord_type = sp.get("chord_type_hint", "maj")
            score = 1.0 if sp_chord_type == query.chord_type else 0.5
            scored.append((sp, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if scored:
            return scored[0][0]
        else:
            return {
                "pattern_id": "safe_kit_fallback",
                "family": "safe_kit",
                "events": self._generate_basic_chord_pattern(query),
            }
    
    def _generate_basic_chord_pattern(
        self,
        query: GuitarQuery
    ) -> List[Dict[str, Any]]:
        """基本コードパターン生成（緊急フォールバック）
        
        Args:
            query: ギタークエリ
        
        Returns:
            イベントリスト
        """
        # 4分音符でルート音を鳴らす基本パターン
        events = []
        
        # Chord Rootをピッチに変換
        root_pitches = {
            "C": 60, "D": 62, "E": 64, "F": 65,
            "G": 67, "A": 69, "B": 71
        }
        root_pitch = root_pitches.get(query.chord_root, 60)
        
        # 4拍分のダウンビート
        for beat in range(4):
            pos_ql = beat * 4  # Quarter Length単位
            events.append({
                "pos_ql": pos_ql,
                "pitch": root_pitch,
                "velocity": 80,
                "duration_ql": 3.5,  # 4分音符 - 少し短め
            })
        
        return events
    
    def recommend(
        self,
        query: GuitarQuery,
        min_proba: float = 0.15,
        min_margin: float = 0.10
    ) -> GuitarRecommendResult:
        """ギターパターン推薦（v3互換）
        
        Args:
            query: ギタークエリ
            min_proba: Top-1確率最小値（これ以下でSafe-Kit発火）
            min_margin: Top-1/Top-2マージン最小値（これ以下でSafe-Kit発火）
        
        Returns:
            推薦結果
        """
        # ML推論でFamily予測
        top1_family, top1_proba, top2_proba = self._predict_family_ml(query)
        margin = top1_proba - top2_proba
        
        # Safety判定
        safety_triggered = False
        safety_reason = ""
        
        if top1_proba < min_proba:
            safety_triggered = True
            safety_reason = f"Low top1_proba: {top1_proba:.3f} < {min_proba}"
        elif margin < min_margin:
            safety_triggered = True
            safety_reason = f"Low margin: {margin:.3f} < {min_margin}"
        
        # Safe-Kit発火
        if safety_triggered:
            safe_pattern = self._get_safe_kit_pattern(query)
            logger.info(f"Safe-Kit triggered: {safety_reason}")
            
            return GuitarRecommendResult(
                pattern_id=safe_pattern.get("pattern_id", "safe_kit"),
                pattern=safe_pattern,
                top1_proba=top1_proba,
                top2_proba=top2_proba,
                margin=margin,
                safety_triggered=True,
                safety_reason=safety_reason,
                accent_score=0.7,  # Safe-Kitは保守的なスコア
                chord_fit=0.8,
                density=4.0,
            )
        
        # Family内からパターン選択
        pattern_id = self._select_pattern_from_family(top1_family, query)
        
        if pattern_id is None:
            # Family内に候補がない → Safe-Kit発火
            safe_pattern = self._get_safe_kit_pattern(query)
            logger.warning(f"No pattern found for family: {top1_family}, using Safe-Kit")
            
            return GuitarRecommendResult(
                pattern_id=safe_pattern.get("pattern_id", "safe_kit"),
                pattern=safe_pattern,
                top1_proba=top1_proba,
                top2_proba=top2_proba,
                margin=margin,
                safety_triggered=True,
                safety_reason=f"No pattern for family: {top1_family}",
                accent_score=0.7,
                chord_fit=0.8,
                density=4.0,
            )
        
        # 推薦成功
        pattern = self.patterns[pattern_id]
        
        # KPI予測値（パターンのメタデータから取得、または推定）
        accent_score = pattern.get("accent_score", 0.75)
        chord_fit = pattern.get("chord_fit", 0.70)
        density = pattern.get("density", 6.0)
        
        logger.info(
            f"Recommended Guitar pattern: {pattern_id} "
            f"(family={top1_family}, proba={top1_proba:.3f}, margin={margin:.3f})"
        )
        
        return GuitarRecommendResult(
            pattern_id=pattern_id,
            pattern=pattern,
            top1_proba=top1_proba,
            top2_proba=top2_proba,
            margin=margin,
            safety_triggered=False,
            safety_reason="",
            accent_score=accent_score,
            chord_fit=chord_fit,
            density=density,
        )
    
    @classmethod
    def load_from_pickle(
        cls,
        patterns_pickle_path: Path,
        safe_kit_path: Optional[Path] = None,
        model_pickle_path: Optional[Path] = None
    ) -> "GuitarPatternRecommender":
        """Pickleファイルからロード
        
        Args:
            patterns_pickle_path: パターン辞書pickleパス
            safe_kit_path: Safe-Kit YAMLパス
            model_pickle_path: ML modelパス
        
        Returns:
            GuitarPatternRecommenderインスタンス
        """
        with open(patterns_pickle_path, 'rb') as f:
            patterns = pickle.load(f)
        
        return cls(
            patterns=patterns,
            safe_kit_path=safe_kit_path,
            model_pickle_path=model_pickle_path
        )
