#!/usr/bin/env python3
"""
Instrument Stage2 Base Class - 共通設計パターン

全楽器（Bass/Piano/Strings/Guitar）のStage2に共通する骨格：
- NO-OP既定
- YAML駆動プリセット
- フェーズ実行（Phase）
- バリデーション
- メトリクス採取
- 安全なtry/except
"""

import importlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
import numpy as np

logger = logging.getLogger(__name__)

# スケールモード機能（Scaler 3風）
try:
    from ops.scale_modes import scale_mask_for_point
except Exception:
    scale_mask_for_point = None  # 無くてもNO-OP


class InstrumentStage2Base:
    """
    楽器Stage2の共通基底クラス
    
    設計原則:
    1. NO-OP既定: 設定未指定時は何もしない
    2. 後方互換: 既存APIに影響なし
    3. 段階導入: Phase単位でON/OFF可能
    4. 安全性: 各Phase失敗でもスキップして完走
    5. 可視化: メトリクス1行ログ＋JSON
    """
    
    def __init__(
        self,
        instrument_name: str = "unknown",
        style_presets: Optional[Dict[str, Any]] = None,
        vocab_presets: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        default_instrument: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        Args:
            instrument_name: 楽器名（"bass"/"piano"/"strings"/"guitar"）
            style_presets: スタイルプリセット辞書
            vocab_presets: 語彙プリセット辞書（任意）
            params: パラメータ辞書（任意）
            overrides: 個別パラメータ上書き（任意）
            default_instrument: Factory互換用（指定があれば優先）
            *args, **kwargs: 未知のキーワード引数を受け入れる（互換性）
        """
        # default_instrument があれば優先
        self.instrument_name = default_instrument or instrument_name
        self.style_presets = style_presets or {}
        self.vocab_presets = vocab_presets or {}
        self.params = params or {}
        self._overrides = overrides or {}
        self.metrics: Dict[str, Any] = {}
        self._rnd = None  # 各楽器で初期化
        self._rpn_written = False  # Phase24: 一度だけRPNを書くためのフラグ
        self._model_cache: Dict[str, Any] = {}  # AI modelキャッシュ
        # 余ったkwargsは将来用に保持（ログ用にも使える）
        self._extra_init_kwargs = dict(kwargs)
        
        # ▼ Stage2 params の自動ロード
        self._params_source = None
        for modname in (
            f"{self.instrument_name}_params_stage2",
            f"generator.{self.instrument_name}_params_stage2",
        ):
            try:
                mod = importlib.import_module(modname)
                defaults = getattr(mod, "DEFAULTS", None)
                if isinstance(defaults, dict):
                    # DEFAULTS を先に、ユーザー指定 params を後勝ちでマージ
                    merged = {**defaults, **self.params}
                    self.params = merged
                    self._params_source = modname
                    logger.info(f"Stage2 Base: Loaded params from {modname} ({len(defaults)} defaults)")
                    break
            except Exception as e:
                logger.debug(f"Stage2 Base: Could not load {modname}: {e}")
                continue
    
    def apply(
        self,
        part: Any,  # stream.Part
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> Any:
        """
        Stage2処理のメインエントリーポイント
        
        Args:
            part: music21 Part
            section_meta: {"label": "Verse", "bar": 0, "emotion": "energetic", ...}
            mix_context: {"sections": [...], "vocal_phrases": [...], "bass_onsets_ql": [...], ...}
            overrides: 個別パラメータ上書き
            seed: 乱数シード（再現性用）
        
        Returns:
            処理後のPart（インプレース変更＋戻り値）
        """
        # overridesを保存（Phase 22/23で使用）
        self._overrides = {"mix_context": mix_context, **(overrides or {})}
        
        # 0) プリセットとemotion/overridesのマージ（NO-OP既定）
        params = self._merge_presets(section_meta, overrides or {})
        
        if not params:
            # 設定なし = NO-OP
            return part
        
        # 1) バリデーション（軽量）
        try:
            warnings = self._validate_params(params)
            if warnings:
                logger.warning(f"[{self.instrument_name}Stage2] Validation warnings: {'; '.join(warnings)}")
        except Exception as e:
            logger.warning(f"[{self.instrument_name}Stage2] Validation failed: {e}")
        
        # 2) PHASES（小粒にtry/except、失敗しても完走）
        # ★ FIX: _get_phases()にparamsを渡して、Phase 13-19の動的有効化を実現
        for phase_num in self._get_phases(params):
            # メソッド名は_phase_{num}または_phase_{num}_{suffix}を許容
            phase_method = f"_phase_{phase_num}"
            
            # 正確な名前がなければ、サフィックス付きメソッドを探す
            if not hasattr(self, phase_method):
                # _phase_13_vocabulary, _phase_14_harmonic_awareness等を探す
                found = False
                for attr_name in dir(self):
                    if attr_name.startswith(f"_phase_{phase_num}_"):
                        phase_method = attr_name
                        found = True
                        break
                
                if not found:
                    continue
            
            try:
                getattr(self, phase_method)(part, section_meta, mix_context, params, seed)
            except Exception as e:
                logger.warning(f"[{self.instrument_name}Stage2] Phase {phase_num} skipped: {e}")
        
        # 3) メトリクス採取
        try:
            self._collect_metrics(part, section_meta, mix_context, params)
            self._log_metrics()
        except Exception as e:
            logger.warning(f"[{self.instrument_name}Stage2] Metrics collection failed: {e}")
        
        return part
    
    def _merge_presets(
        self,
        section_meta: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        プリセット・emotion・overridesを統合
        
        優先順位: overrides > emotion_profile > style_preset
        """
        import sys
        params = {}
        
        # 1) スタイルプリセットから取得
        style_name = section_meta.get(f"{self.instrument_name}_style") or overrides.get("style")
        
        # デバッグ出力
        print(f"[DEBUG _merge_presets] style_name={style_name}", file=sys.stderr)
        print(f"[DEBUG _merge_presets] self.style_presets.keys()={list(self.style_presets.keys())}", file=sys.stderr)
        
        if style_name and style_name in self.style_presets:
            preset = self.style_presets[style_name]
            print(f"[DEBUG _merge_presets] Found preset for {style_name}: keys={list(preset.keys())}", file=sys.stderr)
            params = self._deep_merge_dicts(params, preset)
        else:
            print(f"[DEBUG _merge_presets] No preset found for style={style_name}", file=sys.stderr)
        
        # 2) emotionベースのパラメータ
        emotion_params = section_meta.get(f"{self.instrument_name}_params", {})
        if emotion_params:
            params = self._deep_merge_dicts(params, emotion_params)
        
        # 3) 明示的overrides
        if overrides:
            params = self._deep_merge_dicts(params, overrides)
        
        print(f"[DEBUG _merge_presets] Final params keys={list(params.keys())}", file=sys.stderr)
        
        return params
    
    def _deep_merge_dicts(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """再帰的辞書マージ"""
        result = base.copy()
        for key, val in overlay.items():
            if isinstance(val, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge_dicts(result[key], val)
            else:
                result[key] = val
        return result
    
    def _get_phases(self, params: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        実行するPhase番号のリスト（サブクラスでオーバーライド）
        
        Args:
            params: マージ済みパラメータ辞書（Phase 13-19の動的有効化に使用）
        
        共通Phase番号:
        10: プリセット適用
        11: 密度整形
        12: 音域＆レンジ補正
        13: 語彙の貼り付け
        14: 和声知性
        15: 他パート同期
        16: 遷移平滑化
        17: アーティキュレーション細分化
        18: ダイナミクス整形
        19: グルーヴマイクロタイミング
        20: Humanize
        """
        return [11, 12, 20]  # デフォルトは安全な3Phase
    
    def _validate_params(self, params: Dict[str, Any]) -> List[str]:
        """
        パラメータの軽量バリデーション
        
        Returns:
            警告メッセージのリスト
        """
        warnings = []
        
        # 確率値チェック（0.0〜1.0）
        for key in self._get_probability_keys():
            val = self._deep_get(params, key.split("."))
            if val is not None:
                try:
                    f = float(val)
                    if not (0.0 <= f <= 1.0):
                        warnings.append(f"{key}={f} out of range [0,1]")
                except (ValueError, TypeError):
                    warnings.append(f"{key}={val} not a number")
        
        # ベロシティチェック（1〜127）
        for key in self._get_velocity_keys():
            val = self._deep_get(params, key.split("."))
            if val is not None:
                try:
                    i = int(val)
                    if not (1 <= i <= 127):
                        warnings.append(f"{key}={i} out of range [1,127]")
                except (ValueError, TypeError):
                    warnings.append(f"{key}={val} not an integer")
        
        # 楽器固有のバリデーション
        warnings.extend(self._validate_instrument_specific(params))
        
        return warnings
    
    def _deep_get(self, d: Dict[str, Any], keys: List[str]) -> Any:
        """ネストした辞書からキーで値を取得"""
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
            if d is None:
                return None
        return d
    
    def _get_probability_keys(self) -> List[str]:
        """確率値として検証すべきキー（サブクラスでオーバーライド）"""
        return []
    
    def _get_velocity_keys(self) -> List[str]:
        """ベロシティ値として検証すべきキー（サブクラスでオーバーライド）"""
        return []
    
    def _validate_instrument_specific(self, params: Dict[str, Any]) -> List[str]:
        """楽器固有のバリデーション（サブクラスで実装）"""
        return []
    
    def _collect_metrics(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any]
    ) -> None:
        """メトリクス採取（サブクラスで実装）"""
        try:
            notes = list(part.flatten().notes)
            self.metrics = {
                "instrument": self.instrument_name,
                "section": section_meta.get("label", "?"),
                "bar": section_meta.get("bar", 0),
                "emotion": section_meta.get("emotion", "?"),
                "style": section_meta.get(f"{self.instrument_name}_style", "?"),
                "note_count": len(notes),
            }
            
            if notes:
                velocities = [n.volume.velocity for n in notes if hasattr(n, 'volume') and hasattr(n.volume, 'velocity')]
                if velocities:
                    import numpy as np
                    self.metrics["vel_mean"] = float(np.mean(velocities))
                    self.metrics["vel_std"] = float(np.std(velocities))
        except Exception:
            pass
    
    def _log_metrics(self) -> None:
        """メトリクスを1行ログに出力"""
        if not self.metrics:
            return
        
        parts = [f"{k}={v}" for k, v in self.metrics.items()]
        logger.info(f"[{self.instrument_name}Stage2Metrics] {' '.join(parts)}")
    
    # ========================================================================
    # Phase実装（サブクラスで必要なPhaseだけ実装）
    # ========================================================================
    
    def _phase_11(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 11: 密度整形（サブクラスで実装）"""
        pass
    
    def _phase_12(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 12: 音域＆レンジ補正（サブクラスで実装）"""
        pass
    
    def _phase_20(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 20: Humanize（サブクラスで実装）"""
        pass
    
    # ========================================================================
    # 共通ヘルパー: Phase 16/18/19（全楽器で使用可能）
    # ========================================================================
    
    def _apply_transition_curve(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        params: Dict[str, Any]
    ) -> None:
        """
        Phase 16: セクション境界でのクレッシェンド/デクレッシェンド適用
        
        Args:
            part: music21 Part
            section_meta: セクション情報
            params: パラメータ辞書
        
        動作:
        - セクション最後のN小節でクレッシェンド（velocity +step）
        - セクション最初のN小節でデクレッシェンド（velocity -step）
        - NO-OP既定（transition設定なしなら何もしない）
        """
        tr = params.get("transition") or {}
        if not tr or not tr.get("enable_crescendo"):
            return
        
        try:
            bars_up = int(tr.get("crescendo_bars", 0) or 0)
            bars_dn = int(tr.get("decrescendo_bars", 0) or 0)
            step = int(tr.get("velocity_step", 0) or 0)
            
            if bars_up <= 0 and bars_dn <= 0:
                return
            
            # セクション情報取得
            section_bar_start = section_meta.get("bar", 0)
            section_bars = section_meta.get("bars", 8)
            section_bar_end = section_bar_start + section_bars
            
            notes = list(part.flatten().notes)
            if not notes:
                return
            
            for n in notes:
                if not hasattr(n, 'offset') or not hasattr(n, 'volume'):
                    continue
                
                # 小節番号計算（4/4拍子想定）
                bar_num = int(n.offset / 4.0)
                
                # クレッシェンド適用（セクション末尾）
                if bars_up > 0:
                    bars_from_end = section_bar_end - bar_num
                    if 0 < bars_from_end <= bars_up:
                        multiplier = bars_up - bars_from_end + 1
                        delta = step * multiplier
                        new_vel = min(127, n.volume.velocity + delta)
                        n.volume.velocity = int(new_vel)
                
                # デクレッシェンド適用（セクション先頭）
                if bars_dn > 0:
                    bars_from_start = bar_num - section_bar_start
                    if 0 <= bars_from_start < bars_dn:
                        multiplier = bars_dn - bars_from_start
                        delta = step * multiplier
                        new_vel = max(1, n.volume.velocity - delta)
                        n.volume.velocity = int(new_vel)
        
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 16 transition curve skipped: {e}")
    
    def _apply_dynamics_curve(
        self,
        part: Any,
        params: Dict[str, Any]
    ) -> None:
        """
        Phase 18: セクション全体にベロシティカーブ適用
        
        Args:
            part: music21 Part
            params: パラメータ辞書
        
        カーブ種類:
        - linear_up: 段階的に音量増加
        - linear_down: 段階的に音量減少
        - peak_middle: 中間で最大、前後で弱く
        - NO-OP既定（dynamics設定なしなら何もしない）
        """
        dyn = params.get("dynamics") or {}
        if not dyn:
            return
        
        try:
            curve_type = dyn.get("curve_type") or dyn.get("velocity_curve")
            if not curve_type:
                return
            
            target_min = int(dyn.get("target_min", dyn.get("min_vel", 60)))
            target_max = int(dyn.get("target_max", dyn.get("max_vel", 100)))
            
            notes = list(part.flatten().notes)
            if not notes:
                return
            
            # ノートをonsetでソート
            notes.sort(key=lambda n: n.offset if hasattr(n, 'offset') else 0)
            
            for i, n in enumerate(notes):
                if not hasattr(n, 'volume'):
                    continue
                
                # 進行率 (0.0 ~ 1.0)
                progress = i / max(1, len(notes) - 1)
                
                # カーブ適用
                if curve_type == "linear_up":
                    target_vel = target_min + (target_max - target_min) * progress
                
                elif curve_type == "linear_down":
                    target_vel = target_max - (target_max - target_min) * progress
                
                elif curve_type == "peak_middle":
                    # 放物線カーブ（中央で最大）
                    target_vel = target_min + (target_max - target_min) * (1 - 4 * (progress - 0.5)**2)
                
                else:
                    continue
                
                n.volume.velocity = int(max(1, min(127, target_vel)))
        
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 18 dynamics curve skipped: {e}")
    
    def _apply_groove_timing(
        self,
        part: Any,
        tempo: float,
        params: Dict[str, Any]
    ) -> None:
        """
        Phase 19: グルーヴマイクロタイミング（スウィング/レイドバック/プッシュ）
        
        Args:
            part: music21 Part
            tempo: テンポ (BPM)
            params: パラメータ辞書
        
        調整:
        - swing_amount: 裏拍を遅らせる（Jazz/Blues）
        - laidback_ms: 全体を僅かに遅らせる（Reggae/Funk）
        - push_ms: 16分音符を前にずらす（Metal/Punk）
        - NO-OP既定（groove設定なしなら何もしない）
        """
        grv = params.get("groove") or {}
        if not grv:
            return
        
        try:
            swing_amount = float(grv.get("swing_amount", 0.0) or 0.0)
            laidback_ms = float(grv.get("laidback_ms", grv.get("laid_back_ms", 0.0)) or 0.0)
            push_ms = float(grv.get("push_sixteenth_ms", 0.0) or 0.0)
            
            if swing_amount == 0.0 and laidback_ms == 0.0 and push_ms == 0.0:
                return
            
            notes = list(part.flatten().notes)
            if not notes:
                return
            
            # BPMから1拍の長さ（秒）を計算
            beat_duration = 60.0 / tempo
            
            for n in notes:
                if not hasattr(n, 'offset'):
                    continue
                
                original_offset = n.offset
                
                # スウィング適用（裏拍）
                if swing_amount > 0.0:
                    # 拍内のポジション（0.0 ~ 1.0）
                    beat_position = (original_offset % 1.0)
                    
                    # 裏拍判定（0.5 ~ 1.0）
                    if 0.45 < beat_position < 0.55:  # 裏拍の許容範囲
                        # swing_amount = 0.15 なら 15% 遅らせる
                        swing_shift = (beat_duration / 2) * swing_amount
                        swing_shift_ql = swing_shift / (beat_duration / 4.0)  # quarterLength換算
                        n.offset += swing_shift_ql
                
                # レイドバック適用
                if laidback_ms > 0.0:
                    laidback_ql = (laidback_ms / 1000.0) * (tempo / 60.0)
                    n.offset += laidback_ql
                
                # プッシュ適用（16分音符）
                if push_ms != 0.0:
                    # 16分音符判定（quarterLength < 0.5）
                    if hasattr(n, 'quarterLength') and n.quarterLength < 0.5:
                        push_ql = (push_ms / 1000.0) * (tempo / 60.0)
                        n.offset += push_ql
        
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 19 groove timing skipped: {e}")
    
    # ========================================================================
    # Phase 22/24/23: Emotion mapping / Controls / Prosody
    # ========================================================================
    
    def _emotion_value_at(self, off_ql: float, smooth_ms: float = 0.0, default: float = 0.0) -> float:
        """
        Phase 22: mix_context から E∈[0..1] を取得（vocal_energy か emotion_curve）。
        QL座標で最近傍/簡易平均。未設定は default。
        """
        try:
            mc = (self._overrides or {}).get("mix_context") or {}
            curve = mc.get("emotion_curve") or mc.get("vocal_energy") or []
            if not curve:
                return float(default)
            # curve: [(off_ql, val0_1), ...]
            if smooth_ms and mc.get("beat_grid"):
                bpm = float(mc["beat_grid"].get("bpm", 120.0))
                sec_per_q = 60.0/max(1e-6, bpm)
                win_ql = max((smooth_ms/1000.0)/sec_per_q, 0.0)
                lo, hi = off_ql - 0.5*win_ql, off_ql + 0.5*win_ql
                vals = [float(v) for (t,v) in curve if lo <= float(t) <= hi]
                if vals:
                    return max(0.0, min(1.0, sum(vals)/len(vals)))
            # 既定：最近傍
            t, v = min(curve, key=lambda tv: abs(float(tv[0]) - off_ql))
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return float(default)
    
    def _apply_emotion_map(self, part: Any, params: Dict[str, Any], *, role: str, ql_per_bar: float, bpm: float):
        """
        Phase 22: density/velocity/register へ E(t) を連続写像（NO-OP既定）。
        
        params.emotion_map: {
            density_gain: 0.6,
            register_shift: +3,
            staccato_bias: 0.2,
            smooth_ms: 200
        }
        """
        em = (params.get("emotion_map") or {})
        if not em:
            return
        try:
            gain = float(em.get("density_gain", 0.0))  # 0..1 をこの係数で密度に掛ける想定（各楽器側で解釈）
            rsh  = float(em.get("register_shift", 0.0))  # ピッチ平行移動の上限（半音）
            stb  = float(em.get("staccato_bias", 0.0))   # ノート長さの短縮割合上限
            sm_ms= float(em.get("smooth_ms", 0.0))
            
            # Partからノートを取得
            notes = list(part.flatten().notes) if hasattr(part, 'flatten') else []
            if not notes:
                return
            
            for n in notes:
                if not hasattr(n, 'offset'):
                    continue
                
                off = float(n.offset)
                E = self._emotion_value_at(off, smooth_ms=sm_ms, default=0.0)
                
                # velocity: 緩やかに増減（±12程度の範囲で）
                if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                    dv = int(round(12.0 * (E - 0.5)))
                    n.volume.velocity = int(max(1, min(127, n.volume.velocity + dv)))
                
                # register shift（半音）— 最大 rsh * (E-0.5)*2
                if hasattr(n, 'pitch') and rsh != 0.0:
                    dp = int(round(rsh * (E - 0.5) * 2.0))
                    new_midi = n.pitch.midi + dp
                    n.pitch.midi = int(max(0, min(127, new_midi)))
                
                # staccato bias— durを(1 - stb*E)倍（短くしすぎない）
                if hasattr(n, 'quarterLength') and stb > 0.0:
                    q = float(n.quarterLength)
                    n.quarterLength = max(0.05, q * (1.0 - stb * max(0.0, E)))
            
            # density_gain は各楽器側でパターン選択に使うため、メタ情報として保存
            if not hasattr(part, '_emotion_hints'):
                part._emotion_hints = {}
            part._emotion_hints["density_gain"] = gain
            
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 22 emotion mapping skipped: {e}")
    
    def _emit_cc_lane(self, part: Any, cc: int, points: list):
        """
        Phase 24: points: [(time_sec, value0_127)] を part のCC情報へ追加。
        music21では part の要素として ControlChange を追加。
        """
        try:
            from music21 import midi as m21midi
            for t_sec, v in points:
                if t_sec < 0.0:
                    continue
                # music21ではControlChangeをPartに追加
                cc_event = m21midi.MidiEvent(track=1)
                cc_event.type = "CONTROLLER_CHANGE"
                cc_event.channel = 1
                cc_event.data1 = int(cc)
                cc_event.data2 = int(max(0, min(127, int(v))))
                cc_event.time = int(t_sec * 1000)  # ms単位
                # Partにメタ情報として保存（実際のMIDI書き出しは各Generator側で実装）
                if not hasattr(part, '_cc_events'):
                    part._cc_events = []
                part._cc_events.append(cc_event)
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 24 CC emission skipped: {e}")
    
    def _emit_rpn_bend_range(self, part: Any, semitones: int, *, at_sec: float = 0.0):
        """
        Phase 24: RPN 0,0 (Pitch Bend Sensitivity)。
        
        厳密化仕様:
        - 各トラック最大1回のみ発行（_rpn_written フラグ）
        - 時刻 t ≥ 0（負の時刻を防止）
        - PB存在時はPBより先頭（1μs前）に発行
        """
        try:
            if self._rpn_written:
                return
            self._rpn_written = True
            
            # PB存在確認: PBがある場合はその直前（1e-9秒前）に出す
            pb_events = getattr(part, '_pb_events', [])
            if pb_events:
                first_pb_time = min((float(ev.get("time_sec", 0.0)) for ev in pb_events), default=0.0)
                # PBより先頭、かつ t ≥ 0
                at_sec = max(0.0, min(at_sec, first_pb_time - 1e-9))
            else:
                # PBなし: 指定時刻、ただし t ≥ 0
                at_sec = max(0.0, float(at_sec))
            
            # Partにメタ情報として保存
            if not hasattr(part, '_rpn_events'):
                part._rpn_events = []
            part._rpn_events.append({
                "time_sec": at_sec,
                "msb": 0,
                "lsb": 0,
                "data_msb": int(semitones),
                "data_lsb": 0
            })
            
            logger.debug(f"[{self.instrument_name}] Phase 24 RPN emitted: range={semitones}, time={at_sec:.6f}s")
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 24 RPN emission skipped: {e}")
    
    def _emit_pitchbend_14bit(self, part: Any, pts_norm: list, *, bend_range: int = 2, bpm: float = 120.0):
        """
        Phase 24: pts_norm: [(off_ql, norm[-1..+1])] を 14bit PB に変換。
        PBは±8191スケールへクリップ。
        """
        try:
            PB_MIN, PB_MAX = -8191, 8191
            
            def _to_raw(x):
                v = max(-1.0, min(1.0, float(x)))
                return int(round(v * PB_MAX))
            
            # Partにメタ情報として保存
            if not hasattr(part, '_pb_events'):
                part._pb_events = []
            
            sec_per_q = 60.0/max(1e-6, bpm)
            for off_ql, nrm in pts_norm:
                t = max(0.0, float(off_ql) * sec_per_q)
                part._pb_events.append((t, int(_to_raw(nrm))))
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 24 PB emission skipped: {e}")
    
    def _apply_controls_unified(self, part: Any, params: Dict[str, Any], *, bpm: float):
        """
        Phase 24: 表情CC・サスティン方針・ベンドレンジを統一（NO-OP既定）。
        
        params.controls: {
            expression_curve: 'arch|linear|flat',
            sustain_policy: 'pad_only|off|always',
            bend_range: 2
        }
        """
        cs = (params.get("controls") or {})
        if not cs:
            return
        try:
            br = int(cs.get("bend_range", 2))
            self._emit_rpn_bend_range(part, br, at_sec=0.0)
            
            # expression (CC11) を簡易曲線で出す
            curve = str(cs.get("expression_curve", "flat")).lower()
            
            # Partの長さを推定
            notes = list(part.flatten().notes) if hasattr(part, 'flatten') else []
            if not notes:
                return
            
            sec_per_q = 60.0/max(1e-6, bpm)
            last_end = max((n.offset + n.quarterLength) * sec_per_q for n in notes)
            dur_sec = float(last_end)
            
            if dur_sec > 0.0:
                T = dur_sec
                grid = [0.0, T*0.25, T*0.5, T*0.75, T]
                vals = []
                for t in grid:
                    if curve == "arch":
                        x = t/T if T>0 else 0.0
                        y = -4*(x-0.5)**2 + 1.0  # 0→1→0
                        v = 64 + int(63*y)
                    elif curve == "linear":
                        v = 32 + int(95*(t/T if T>0 else 0.0))
                    else:  # flat
                        v = 96
                    vals.append((t, max(0, min(127, v))))
                self._emit_cc_lane(part, cc=11, points=vals)
            
            # sustain_policy はメタ情報として保存
            if not hasattr(part, '_control_meta'):
                part._control_meta = {}
            part._control_meta["sustain_policy"] = cs.get("sustain_policy", "off")
            
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 24 controls unified skipped: {e}")
    
    def _apply_prosody_alignment(self, part: Any, params: Dict[str, Any], *, bpm: float):
        """
        Phase 23: 子音窓×強勢に合わせて Velや隙間を微調整（v4.1: ProsodyController統合）。
        
        params.prosody: {
            enable: true,
            anchors_path: "analysis/lyric_anchors.json",  # v4.1: アンカーパス
            config: {...}  # ProsodyController設定（任意）
        }
        
        旧フォーマット（互換性維持）:
        params.prosody: {
            enable: true,
            stress_boost: 8,
            sibilant_duck_db: -3,
            plosive_gap_ms: 40,
            window_ms: 120
        }
        """
        pr = (params.get("prosody") or {})
        if not pr or not pr.get("enable", False):
            return
        
        try:
            # v4.1: ProsodyController使用を試みる
            anchors_path = pr.get("anchors_path")
            if anchors_path:
                from pathlib import Path
                from generator.prosody_controller import load_prosody_controller
                
                controller = load_prosody_controller(
                    anchors_path=Path(anchors_path),
                    config_path=Path(pr["config"]) if pr.get("config") else None
                )
                
                if controller:
                    # music21 Part → ノートリスト変換
                    notes_list = []
                    sec_per_q = 60.0 / max(1e-6, bpm)
                    
                    for n in part.flatten().notes:
                        if not hasattr(n, 'offset'):
                            continue
                        
                        time_sec = float(n.offset) * sec_per_q
                        vel = getattr(n.volume, 'velocity', 64) if hasattr(n, 'volume') else 64
                        dur_sec = float(n.quarterLength) * sec_per_q if hasattr(n, 'quarterLength') else 0.5
                        pitch = n.pitch.midi if hasattr(n, 'pitch') else 60
                        
                        notes_list.append({
                            "time": time_sec,
                            "pitch": pitch,
                            "vel": vel,
                            "dur": dur_sec,
                            "_note_obj": n  # 元オブジェクトへの参照
                        })
                    
                    # Prosody制御適用
                    controller.apply_prosody(notes_list, role=self.instrument_name, tempo=bpm)
                    
                    # 結果を反映
                    for note_dict in notes_list:
                        n = note_dict.get("_note_obj")
                        if n and hasattr(n, 'volume'):
                            n.volume.velocity = int(note_dict["vel"])
                        if n and hasattr(n, 'quarterLength'):
                            n.quarterLength = note_dict["dur"] / sec_per_q
                    
                    logger.info(f"[{self.instrument_name}] Phase 23: Applied prosody control ({len(notes_list)} notes)")
                    return
            
            # フォールバック: 旧実装（互換性維持）
            mc = (self._overrides or {}).get("mix_context") or {}
            ph = mc.get("vocal_phonemes") or []   # [(off_ql,'sibilant|plosive|stress'...)]
            if not ph:
                return
            
            w_ms = float(pr.get("window_ms", 120.0))
            sec_per_q = 60.0/max(1e-6, bpm)
            w_q = (w_ms/1000.0)/sec_per_q
            
            # 近傍に応じて処理
            notes = list(part.flatten().notes) if hasattr(part, 'flatten') else []
            
            for n in notes:
                if not hasattr(n, 'offset'):
                    continue
                
                off = float(n.offset)
                close = [lab for (t,lab) in ph if abs(float(t)-off) <= w_q]
                if not close:
                    continue
                
                # stress: Velブースト
                if "stress" in close and hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                    n.volume.velocity = int(max(1, min(127, n.volume.velocity + int(pr.get("stress_boost", 8)))))
                
                # sibilant: 高域衝突を避けるためVel少し下げる
                if "sibilant" in close and hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                    duck = int(round(abs(float(pr.get("sibilant_duck_db", -3))) * 2))  # 簡易：dB→Vel
                    n.volume.velocity = int(max(1, n.volume.velocity - duck))
                
                # plosive直後: 短い隙間
                if "plosive" in close and hasattr(n, 'quarterLength'):
                    gap_q = (float(pr.get("plosive_gap_ms", 40))/1000.0)/sec_per_q
                    n.quarterLength = max(0.05, n.quarterLength - gap_q)
        
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 23 prosody alignment skipped: {e}")
    
    # ===== Phase 25: Sparsify & Collision =================================
    def _thin_notes_even(self, part: Any, *, keep_endpoints=True, min_gap_ms=0.0, step_count=None, bpm=120.0):
        """端点保持・順序不変の等間隔サンプリングで間引く（NO-OP既定）。
        - step_count を優先。無ければ min_gap_ms で近接ノートを抑制。
        - 非破壊: 新しい notes を作って置換。
        """
        try:
            min_gap_ms = float(min_gap_ms or 0.0)  # None→0.0 を強制
            
            from music21 import stream
            if not isinstance(part, stream.Part):
                return
            
            notes = list(part.flatten().notesAndRests.notes)
            if not notes:
                return
            
            sec_per_q = 60.0 / max(1e-6, float(bpm))
            
            def to_sec(n):
                return float(n.offset) * sec_per_q
            
            out_notes = None
            
            if step_count and step_count > 0 and len(notes) > step_count:
                # 端点保持＋均等選択
                keep_set = {0, len(notes)-1} if keep_endpoints and len(notes) >= 2 else set()
                stride = (len(notes)-1) / float(step_count-1) if step_count > 1 else len(notes)
                idxs = {int(round(i*stride)) for i in range(step_count)}
                idxs |= keep_set
                out_notes = [n for i, n in enumerate(notes) if i in idxs]
            elif min_gap_ms and min_gap_ms > 0.0:
                out_notes = []
                last_t = -1e9
                gap = float(min_gap_ms) / 1000.0
                for i, n in enumerate(notes):
                    t = to_sec(n)
                    if keep_endpoints and (i == 0 or i == len(notes)-1):
                        out_notes.append(n)
                        last_t = t
                        continue
                    if (t - last_t) >= gap:
                        out_notes.append(n)
                        last_t = t
            
            # out_notesが作成されなかった場合はNO-OP
            if out_notes is None:
                return
            
            # Partを再構築
            part.clear()
            for n in out_notes:
                part.insert(n.offset, n)
        except Exception as e:
            logger.warning(f"[{self.instrument_name}] _thin_notes_even failed: {e}")
            return
    
    def _avoid_register_collision(self, part: Any, *, band_low=0, band_high=127,
                                  strategy="vel_first", reduce_db=6, drop_prob=0.25, seed=None):
        """指定レンジの密集時にVel減衰→必要なら部分的間引き（NO-OP既定）。"""
        try:
            from music21 import stream
            if not isinstance(part, stream.Part):
                return
            
            notes = list(part.flatten().notesAndRests.notes)
            if not notes:
                return
            
            # 密集度の簡易判定：帯域内の比率
            band_notes = [n for n in notes if band_low <= n.pitch.midi <= band_high]
            if not band_notes or len(band_notes) / max(1, len(notes)) < 0.5:
                return
            
            # まずVelを下げる
            if strategy in ("vel_first", "vel"):
                att = int(max(0, reduce_db)) * 2  # 簡易 dB→Vel
                for n in band_notes:
                    if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                        n.volume.velocity = int(max(1, n.volume.velocity - att))
            
            # まだ多いなら一部を確率でドロップ
            if strategy in ("drop_first", "vel_first", "drop"):
                import random as _r
                if seed is not None:
                    _r.seed(seed)
                keep = []
                for n in notes:
                    if n in band_notes and _r.random() < float(drop_prob):
                        continue
                    keep.append(n)
                
                # Partを再構築
                part.clear()
                for n in keep:
                    part.insert(n.offset, n)
        except Exception as e:
            logger.warning(f"[{self.instrument_name}] _avoid_register_collision failed: {e}")
            return
    
    # ===== Phase 26: Hybrid Harmony =======================================
    def _blend_harmony(self, part: Any, *, audio_chordmap, creative_chordmap, blend=0.5,
                       keep_audio_root=True, allow_text_tensions=None):
        """audio(原曲)×creative(創作) の穏やかな混合。未設定は audio 優先でNO-OP."""
        try:
            allow = set(allow_text_tensions or [])
            
            # 簡易実装: part内のノートを audio/creative の chord情報で変更
            # 実際には各楽器の和声ロジックでこれを参照する想定
            # ここでは mix_context に保存するだけ
            if not hasattr(self, '_overrides'):
                self._overrides = {}
            
            self._overrides['harmony_blend'] = {
                'audio': audio_chordmap,
                'creative': creative_chordmap,
                'blend': blend,
                'keep_root': keep_audio_root,
                'allow_tensions': list(allow)
            }
            
            logger.debug(f"[{self.instrument_name}] Harmony blend prepared (blend={blend:.2f})")
        except Exception as e:
            logger.warning(f"[{self.instrument_name}] _blend_harmony failed: {e}")
            return
    
    # ===== Phase 27: Style Adaptation =====================================
    def _window_activity(self, role, bar_idx, window_bars, default=0.5):
        """bar中心 window の平均 activity レベルを返す（無ければdefault）。"""
        try:
            mc = (getattr(self, '_overrides', {}) or {}).get("mix_context") or {}
            table = ((mc.get("activity") or {}).get(role) or {})
            if not table:
                return float(default)
            
            lo = int(max(0, bar_idx - window_bars // 2))
            hi = int(bar_idx + window_bars // 2)
            d = dict(table) if isinstance(table, list) else table
            vals = [float(d.get(b, default)) for b in range(lo, hi + 1)]
            return sum(vals) / len(vals) if vals else float(default)
        except Exception as e:
            logger.warning(f"[{self.instrument_name}] _window_activity failed: {e}")
            return float(default)
    
    def _adapt_style_params(self, params, presets_dict, level: float, edges=(0.2, 0.7), order=None):
        """level∈[0..1] でプリセットを線形補間し params にディープマージ（NO-OP既定）。"""
        try:
            if not presets_dict:
                return params
            
            order = order or ["simple", "moderate", "complex", "intense"]
            lo, hi = float(edges[0]), float(edges[1])
            t = max(0.0, min(1.0, (level - lo) / max(1e-6, hi - lo)))
            
            import math
            i = int(math.floor(t * (len(order) - 1)))
            j = min(len(order) - 1, i + 1)
            w = t * (len(order) - 1) - i
            
            A = presets_dict.get(order[i], {}) or {}
            B = presets_dict.get(order[j], {}) or {}
            
            # 簡易補間: 数値は線形、それ以外は高重み(B)を優先
            def mix(a, b):
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return (1.0 - w) * a + w * b
                return b if b is not None else a
            
            def deep_merge(X, Y):
                if not isinstance(X, dict) or not isinstance(Y, dict):
                    return Y if Y is not None else X
                keys = set(X) | set(Y)
                result = {}
                for k in keys:
                    x_val = X.get(k)
                    y_val = Y.get(k)
                    if isinstance(x_val, dict) and isinstance(y_val, dict):
                        result[k] = deep_merge(x_val, y_val)
                    elif k in X and k in Y:
                        result[k] = mix(x_val, y_val)
                    else:
                        result[k] = y_val if k in Y else x_val
                return result
            
            merged = deep_merge(A, B)
            
            # params へマージ（上書きは merged を優先）
            def deep_over(base, extra):
                if not isinstance(base, dict) or not isinstance(extra, dict):
                    return extra if extra is not None else base
                out = dict(base)
                for k, v in extra.items():
                    out[k] = deep_over(base.get(k), v)
                return out
            
            return deep_over(params, merged)
        except Exception as e:
            logger.warning(f"[{self.instrument_name}] _adapt_style_params failed: {e}")
            return params
    
    # ===== Phase 28: Export Postprocess ===================================
    def postprocess_export(self, part: Any, role, section_meta, params, *,
                           ql_quant=0.25, quantize_ql=None, track_split=None, name_fmt="{idx:02d}_{role}_{section}", markers=None):
        """量子化＋トラック分割＋命名＋マーカー（NO-OP既定）。"""
        # 互換: 呼び出し側の別名 quantize_ql に対応
        if quantize_ql is not None:
            ql_quant = float(quantize_ql)
        
        try:
            from music21 import stream
            import math
            
            if not isinstance(part, stream.Part):
                return
            
            # 量子化（端点は守る）
            if ql_quant and ql_quant > 0:
                notes = list(part.flatten().notesAndRests.notes)
                # offsetを直接変更（Fraction型に対応）
                for n in notes:
                    q = float(n.offset)
                    d = float(n.duration.quarterLength)
                    # 量子化
                    new_offset = round(q / ql_quant) * ql_quant
                    new_duration = max(0.05, round(d / ql_quant) * ql_quant)
                    
                    # Part内のノートを削除して再挿入
                    part.remove(n)
                    n.offset = new_offset
                    n.duration.quarterLength = new_duration
                    part.insert(new_offset, n)
            
            # --- 追加：自動命名（連番 / 日付 / 任意タグ） 未設定=NO-OP ---
            sec_label = str(section_meta.get("label", "section")).lower() if isinstance(section_meta, dict) else "section"
            idx = int(section_meta.get("index", 1)) if isinstance(section_meta, dict) else 1
            exp_cfg = (params.get("export") or {})
            
            # 連番（インスタンス内で昇順）；セッションを跨ぐ永続はしない=安全
            if not hasattr(self, "_export_seq"):
                self._export_seq = 0
            self._export_seq += 1
            seq_width = int(exp_cfg.get("seq_width", 2))
            seq = f"{self._export_seq:0{seq_width}d}"
            
            # 日付タグ
            date_fmt = str(exp_cfg.get("date_fmt", "%Y%m%d"))
            date_tag = datetime.now().strftime(date_fmt)
            
            # プロジェクトタグ（任意）
            proj = str(exp_cfg.get("project_tag", "")).strip()
            
            # スタイル（任意）
            style = str(exp_cfg.get("style_tag", params.get("style", ""))).strip()
            
            # 拡張トークンを name_fmt で利用可能に
            export_name = name_fmt.format(
                idx=idx, role=role, section=sec_label,
                seq=seq, date=date_tag, project=proj, style=style
            )
            
            # 命名
            if not hasattr(part, 'partName') or not part.partName:
                part.partName = export_name
            
            # メタ情報を保存（track_split等 + export_name）
            if track_split:
                # music21の拡張属性として保存（Editorial.miscの代わりにcomment使用）
                if not hasattr(part, 'comment'):
                    part.comment = ""
                part.comment = f"track_split={','.join(track_split)}"
            
            # export_nameをmetaに保存（バッチスクリプトでも利用可能に）
            if hasattr(part, 'comment'):
                if part.comment:
                    part.comment += f"|export_name={export_name}"
                else:
                    part.comment = f"export_name={export_name}"
            else:
                part.comment = f"export_name={export_name}"
            
            # Phase 32: Export Markers（セクション/歌詞マーカー）
            mk = (params.get("export") or {}).get("markers") or {}
            if mk:
                self._emit_export_markers(part, section_meta, mk)
            
            logger.debug(f"[{self.instrument_name}] Export postprocess: quantize={ql_quant}, name={part.partName}")
        except Exception as e:
            logger.warning(f"[{self.instrument_name}] postprocess_export failed: {e}")
            return
    
    # ===== Phase 29: Vocal-Aware Ducking ===================================
    def _apply_vocal_ducking(self, part: Any, mix_context: Dict[str, Any], cfg: Dict[str, Any]):
        """
        Phase 29: ボーカルが密な瞬間は、鍵盤・ギター・ストリングスのVel/長さを軽く抑える（NO-OP既定）
        
        cfg: {
            enable: true,
            amount_db: 3.0,      # 最大で約3dB相当のVel減
            shorten_ms: 20.0     # 最大で20ms短縮
        }
        
        emotion_curve: [(offset_ql, energy_0_1), ...] から最近傍のエネルギーを取得し、
        Velと長さを軽減。
        """
        try:
            if not (cfg and cfg.get("enable")):
                return
            
            # emotion_curveを取得（ボーカルエネルギー）
            curve = mix_context.get("emotion_curve") or []
            if not curve:
                return
            
            amt = float(cfg.get("amount_db", 3.0))
            shorten_ms = float(cfg.get("shorten_ms", 20.0))
            
            # music21 Part の場合
            from music21 import stream
            if isinstance(part, stream.Part):
                notes = list(part.flatten().notes)
                for n in notes:
                    t = float(n.offset)
                    # 最近傍のエネルギー値を取得
                    if curve:
                        Ev = min(curve, key=lambda x: abs(float(x[0]) - t))[1]
                        k = float(Ev)  # 0..1
                        
                        # Vel減衰（dB≈Vel*2 簡易換算）
                        if hasattr(n, 'volume') and n.volume.velocity:
                            old_vel = n.volume.velocity
                            new_vel = max(1, int(old_vel - k * (amt * 2)))
                            n.volume.velocity = new_vel
                        
                        # 長さ短縮（簡易: quarterLengthを微減）
                        if hasattr(n, 'duration'):
                            sec_per_q = 0.5  # 仮定: 120BPM
                            dur_sec = float(n.duration.quarterLength) * sec_per_q
                            new_dur_sec = max(0.05, dur_sec - k * shorten_ms / 1000.0)
                            n.duration.quarterLength = new_dur_sec / sec_per_q
            
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 29 vocal ducking skipped: {e}")
            return
    
    # ===== Phase 30: Cross-Instrument Balance ==============================
    def _rebalance_against(self, part, mix_context, cfg, *, role: str, against_role: str):
        """
        Phase 30: Cross-Instrument Balance
        他ロールの活動度が高い小節で、軽くVelを下げて"譲る"。未設定はNO-OP。
        """
        try:
            if not (cfg and cfg.get("enable")):
                return
            A = ((mix_context or {}).get("activity") or {}).get(against_role) or []
            if not A:
                return
            thr = float(cfg.get("threshold", 0.7))
            cut = int(cfg.get("vel_cut", 6))
            by_bar = {}
            try:
                # A は [(bar, level), ...] 形式を想定
                by_bar = {int(b): float(v) for (b, v) in A}
            except Exception:
                pass
            for n in (part.get("notes") or []):
                b = int(n.get("bar", 0))
                if float(by_bar.get(b, 0.0)) >= thr and "vel" in n:
                    n["vel"] = max(1, int(n["vel"]) - cut)
        except Exception:
            return
    
    # ===== Phase 31: Voice-Leading Guard ===================================
    def _voice_leading_smooth(self, part, section_meta, chord_now, chord_prev, cfg):
        """
        Phase 31: Voice-Leading Guard
        強拍では和声音を優先、過度な跳躍は抑制（半音寄せ/最大跳躍制限）。未設定はNO-OP。
        """
        try:
            if not (cfg and cfg.get("enable")):
                return
            max_leap = int(cfg.get("max_leap", 7))  # 完全5度(7)以上を抑制
            tones = set((chord_now or {}).get("tones_midi", []) or [])
            ql = float(section_meta.get("ql_per_bar", 4.0) if isinstance(section_meta, dict) else 4.0)

            prev_pitch = None
            for n in (part.get("notes") or []):
                p = int(n.get("pitch", 0))
                # 強拍（bar内off_qlが整数で0に近い）を簡易判定
                off_ql = float(n.get("off_ql", 0.0))
                is_strong = (abs((off_ql % ql)) < 1e-6)
                if is_strong and tones and p not in tones:
                    # 近接和声音へ半音だけ寄せる（穏当）
                    cand = min(tones, key=lambda t: abs(t - p))
                    if abs(cand - p) == 1:
                        p = cand
                        n["pitch"] = p
                if prev_pitch is not None and abs(p - prev_pitch) > max_leap:
                    step = 1 if p > prev_pitch else -1
                    n["pitch"] = prev_pitch + step * max_leap
                    p = n["pitch"]
                prev_pitch = p
            
            # Mode/Scale制約（スケール外音を最近接スケール内音に寄せる）
            # 楽器別のデフォルト強度を取得
            scale_strength = cfg.get("scale_constraint_strength", 1.0)
            self._apply_mode_scale_constraint(part, section_meta, strength=scale_strength)
        except Exception:
            return
    
    # ===== Phase 32: Export Markers ========================================
    def _emit_export_markers(self, part: Any, section_meta: Dict[str, Any], markers_cfg: Dict[str, Any]):
        """
        Phase 32: セクション/歌詞マーカーをMIDIメタとして付与
        
        厳密化仕様:
        - Part拡張属性 _export_markers に必ず書く（エクスポーター側で拾いやすい）
        - part.comment が存在すればそちらにも追記（互換性）
        - sections が空でも例外なし
        - 歌詞マーカーはオフ既定、オン時も time_ql ≥ 0 で出る
        
        markers_cfg: {
            sections: true,   # セクションマーカー有効
            lyrics: false     # 歌詞マーカー無効（既定）
        }
        """
        try:
            mix_ctx = (self._overrides or {}).get("mix_context") or {}
            secs = mix_ctx.get("sections") or []
            
            labels_enabled = markers_cfg.get("sections", True)
            lyrics_enabled = markers_cfg.get("lyrics", False)
            
            # マーカー配列を準備（Part拡張属性として必ず保存）
            if not hasattr(part, '_export_markers'):
                part._export_markers = []
            
            markers = part._export_markers
            
            # セクションマーカー
            if labels_enabled and secs:
                ql_per_bar = float(section_meta.get("ql_per_bar", 4.0))
                for s in secs:
                    t = max(0.0, float(s.get("start_ql", s.get("bar", 0) * ql_per_bar)))
                    label = str(s.get("label", "SECTION")).upper()
                    markers.append({"time_ql": t, "label": label})
            
            # 歌詞マーカー（オプション）
            if lyrics_enabled:
                phonemes = mix_ctx.get("vocal_phonemes") or []
                for p in phonemes:
                    if len(p) >= 3:
                        t = max(0.0, float(p[0]))
                        markers.append({"time_ql": t, "label": f"LYR:{p[2]}"})
            
            # 互換性: part.comment にも埋め込み（存在する場合のみ）
            if hasattr(part, 'comment') and markers:
                labels_str = ",".join(f'{m["label"]}@{m["time_ql"]:.3f}' for m in markers)
                # 既存のcomment（track_split等）に追記
                if part.comment and not part.comment.endswith("|"):
                    part.comment += "|"
                part.comment += f"markers={labels_str}"
            
            logger.debug(f"[{self.instrument_name}] Phase 32 Export markers: {len(markers)} items")
            
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 32 export markers skipped: {e}")
            
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Phase 32 export markers skipped: {e}")
            return


@lru_cache(maxsize=32)
def load_yaml_presets(yaml_path: Path) -> Dict[str, Any]:
    """
    プリセットYAMLを読み込む（2系統スキーマを両対応）
    
    形式A: {presets: {style1:{...}, style2:{...}}}
    形式B: {style1:{...}, style2:{...}}
    
    戻り値: {style_name: {...}} のディクショナリ。
    エラー時/空は {} を返す（NO-OP安全）。
    
    @lru_cache により同一パスの再読込を削減（I/O最適化）
    """
    if not yaml_path.exists():
        logger.warning(f"Preset file not found: {yaml_path}")
        return {}
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        base = data.get("presets", data)
        # 型安全性: dict以外（例: None, list）なら空辞書
        return base if isinstance(base, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load preset {yaml_path}: {e}")
        return {}


# ========================================
# Density表記ゆれ正規化ヘルパー
# ========================================

_DENSITY_ALIASES = {
    "strums_per_bar_range": ("notes_per_bar", "range"),
    "notes_per_bar_range":  ("notes_per_bar", "range"),
    "chords_per_bar":       ("events_per_bar", "obj"),  # Piano: 和音イベント
}


def normalize_density(density_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Density設定の表記ゆれを正規化
    
    入力形式:
    - strums_per_bar_range: [4, 8]         → notes_per_bar: {min: 4, max: 8}
    - notes_per_bar_range: [2, 6]          → notes_per_bar: {min: 2, max: 6}
    - chords_per_bar: {min: 3, max: 6}     → events_per_bar: {min: 3, max: 6}
    - notes_per_bar: {min: 2, max: 8}      → {min: 2, max: 8} (範囲正規化のみ)
    - None / {}                            → {} (NO-OP)
    
    戻り値: {min: int, max: int} または {} (正規化失敗時)
    """
    if not density_cfg:
        return {}
    
    out = dict(density_cfg)
    
    for alias_key, (canonical_key, kind) in _DENSITY_ALIASES.items():
        if alias_key not in out:
            continue
        
        val = out[alias_key]
        
        if kind == "range" and isinstance(val, (list, tuple)) and len(val) == 2:
            # [min, max] → {min: ..., max: ...}
            out[canonical_key] = {"min": val[0], "max": val[1]}
            del out[alias_key]
        
        elif kind == "obj" and isinstance(val, dict):
            # {min: ..., max: ...} → 正規化名に変更
            out[canonical_key] = {"min": val.get("min"), "max": val.get("max")}
            del out[alias_key]
    
    # 既に {min, max} 形式なら範囲の正規化のみ（lo <= hi 保証）
    if 'min' in out or 'max' in out:
        try:
            lo = int(out.get('min', out.get('max', 0)))
            hi = int(out.get('max', out.get('min', 0)))
            if lo > hi:
                lo, hi = hi, lo
            return {"min": lo, "max": hi}
        except (ValueError, TypeError):
            # 数値変換失敗 → NO-OP
            return {}
    
    return out


    # ========================================================================
    # Mode/Scale機能（Scaler 3風）— 最小差分拡張
    # ========================================================================
    
    def _apply_mode_scale_mask_to_probs(
        self,
        probs_12: np.ndarray,
        *,
        t_ql: float,
        chord_root: str,
        chord_quality: str,
    ) -> np.ndarray:
        """
        12半音分布（root相対/絶対どちらでも可）にスケール重みを乗算 → 正規化して返す。
        sections に mode が無ければ NO-OP。
        
        Args:
            probs_12: 12要素の確率分布（0-11 = C-B or root相対）
            t_ql: Quarter-length時刻
            chord_root: コードルート ("D" など)
            chord_quality: コード質 ("maj7" など)
        
        Returns:
            マスク適用後の12要素確率分布（正規化済み）
        """
        if scale_mask_for_point is None:
            return probs_12
        
        try:
            # sections を self._overrides.mix_context.sections から取得
            sections = None
            if hasattr(self, "_overrides") and "mix_context" in self._overrides:
                sections = self._overrides["mix_context"].get("sections")
            
            if sections is None:
                return probs_12
            
            w = scale_mask_for_point(
                t_ql=t_ql,
                sections=sections,
                chord_root=chord_root,
                chord_quality=chord_quality,
            )
            
            if not w:
                return probs_12
            
            w = np.asarray(w, dtype=float)
            if w.shape != probs_12.shape:
                # 形が違う場合は安全にスキップ
                return probs_12
            
            out = probs_12 * np.maximum(w, 0.0)
            s = float(out.sum())
            return out / s if s > 1e-12 else probs_12
        
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Mode/scale mask failed: {e}")
            return probs_12
    
    def _apply_mode_scale_constraint(
        self, 
        part: Any, 
        section_meta: Dict[str, Any],
        strength: float = 1.0
    ):
        """
        Phase 31拡張: 各ノートのピッチをスケール内に寄せる。
        
        sections に mode_hint が無い場合は NO-OP。
        スケール外音を検出し、最も近いスケール内音（±1半音）に修正。
        
        Args:
            part: ノート情報を持つpart
            section_meta: セクションメタデータ（ql_per_bar等）
            strength: 修正強度 (0.0-1.0)
                0.0 = 修正なし（NO-OP）
                1.0 = 完全修正（スケール外音を必ず修正）
                0.5 = 50%の確率で修正
        """
        if scale_mask_for_point is None:
            return
        
        # 強度0.0ならスキップ
        if strength <= 0.0:
            return
        
        try:
            # sections取得
            sections = None
            if hasattr(self, "_overrides") and "mix_context" in self._overrides:
                sections = self._overrides["mix_context"].get("sections")
            
            if not sections:
                return
            
            ql_per_bar = float(section_meta.get("ql_per_bar", 4.0) if isinstance(section_meta, dict) else 4.0)
            
            # コード情報取得（chordmap）
            chordmap = None
            if hasattr(self, "_overrides") and "mix_context" in self._overrides:
                chordmap = self._overrides["mix_context"].get("chordmap")
            
            import random
            
            for n in (part.get("notes") or []):
                pitch = int(n.get("pitch", 0))
                off_ql = float(n.get("off_ql", 0.0))
                
                # 現在のコード情報を取得
                chord_root = None
                chord_quality = None
                if chordmap:
                    bar_num = int(off_ql / ql_per_bar)
                    chord_entry = next((c for c in chordmap if c.get("bar") == bar_num), None)
                    if chord_entry:
                        chord_symbol = chord_entry.get("chord", "")
                        # 簡易パース: "Cmaj7" → root="C", quality="maj7"
                        if chord_symbol:
                            # ops.scale_modes の _parse_chord_root_pc を活用
                            try:
                                from ops.scale_modes import _parse_chord_root_pc
                                chord_root = _parse_chord_root_pc(chord_symbol)
                                # quality判定（簡易）
                                if "maj7" in chord_symbol.lower():
                                    chord_quality = "maj7"
                                elif "min7" in chord_symbol.lower() or "m7" in chord_symbol.lower():
                                    chord_quality = "min7"
                                elif "7" in chord_symbol:
                                    chord_quality = "7"
                                elif "maj" in chord_symbol.lower():
                                    chord_quality = "maj"
                                elif "min" in chord_symbol.lower() or "m" in chord_symbol.lower():
                                    chord_quality = "min"
                            except Exception:
                                pass
                
                # 現在位置のマスクを取得（コード情報活用）
                mask = scale_mask_for_point(
                    t_ql=off_ql,
                    sections=sections,
                    chord_root=chord_root,
                    chord_quality=chord_quality
                )
                
                if not mask:
                    continue  # NO-OP
                
                # ピッチクラス（0-11）を取得
                pc = pitch % 12
                
                # スケール外音かチェック（閾値: 平均の70%以下ならスケール外と判定）
                avg_mask = sum(mask) / len(mask)
                threshold = avg_mask * 0.70
                if mask[pc] <= threshold:
                    # 修正強度に応じて確率的に修正
                    if random.random() > strength:
                        continue  # 修正しない
                    
                    # 最近接のスケール内音を探す（±1半音優先）
                    candidates = []
                    for offset in [1, -1, 2, -2]:  # ±1半音、±2半音の順
                        new_pc = (pc + offset) % 12
                        if mask[new_pc] > threshold:
                            candidates.append((abs(offset), pitch + offset))
                    
                    if candidates:
                        # 最も近い音に修正
                        candidates.sort()
                        new_pitch = candidates[0][1]
                        n["pitch"] = new_pitch
                        logger.debug(f"[{self.instrument_name}] Mode/scale constraint (strength={strength:.2f}): {pitch} → {new_pitch}")
        
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] Mode/scale constraint failed: {e}")
            return


    # ========================================================================
    # AI Model Loading & Compose (追加 2025-10-21)
    # ========================================================================
    
    def get_model(self, name: str, path: Optional[str] = None) -> Any:
        """
        AIモデル（pickle/joblib）をロード＋キャッシュ
        
        Args:
            name: モデル名（"groove_classifier" など）
            path: 明示的なパス（省略時は候補を探索）
        
        Returns:
            ロード済みモデル、見つからなければ None
        """
        import pickle
        
        try:
            import joblib
        except ImportError:
            joblib = None
        
        # キャッシュヒット
        cache_key = path or name
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # パス候補リスト
        candidates = []
        if path:
            candidates.append(path)
        
        models_cfg = self.params.get("models") or {}
        if isinstance(models_cfg, dict) and name in models_cfg:
            candidates.append(models_cfg[name])
        
        for pattern in ["models/{name}", "data/models/{name}", "{name}"]:
            candidates.append(pattern.format(name=name))
        
        # ロード試行
        model = None
        for candidate_path in candidates:
            p = Path(candidate_path)
            if not p.exists():
                continue
            
            try:
                if joblib and (p.suffix == ".joblib" or p.suffix == ".pkl"):
                    model = joblib.load(p)
                else:
                    with open(p, "rb") as f:
                        model = pickle.load(f)
                logger.debug(f"[{self.instrument_name}] Loaded AI model: {p}")
                break
            except Exception as e:
                logger.debug(f"[{self.instrument_name}] Failed to load {p}: {e}")
                continue
        
        self._model_cache[cache_key] = model
        return model
    
    def compose(
        self,
        section_data: Dict[str, Any],
        processed_chord_events: List[Any],
        tempo_map: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        デフォルトのcompose実装（V1ジェネレータ + AI適用）
        
        処理フロー:
        1. build_notes() でノート生成（V1または各楽器の実装）
        2. apply_ai_filters() でAI処理
        3. humanize() でヒューマナイズ
        4. quantize_to_tempo_map() で可変テンポ展開
        
        サブクラスで build_notes() を実装すればOK。
        """
        notes = []
        
        # 1) ノート生成
        if hasattr(self, "build_notes"):
            try:
                notes = self.build_notes(
                    section=section_data,
                    chords=processed_chord_events,
                    tempo_map=tempo_map,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"[{self.instrument_name}] build_notes failed: {e}")
                notes = []
        
        # 2) AI処理
        try:
            notes = self.apply_ai_filters(notes, section=section_data)
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] apply_ai_filters failed: {e}")
        
        # 3) ヒューマナイズ
        try:
            if hasattr(self, "humanize"):
                profile = self.params.get("humanize_profile", self.instrument_name)
                notes = self.humanize(notes, profile=profile)
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] humanize failed: {e}")
        
        # 4) 可変テンポ展開
        try:
            if hasattr(self, "quantize_to_tempo_map") and tempo_map:
                notes = self.quantize_to_tempo_map(notes, tempo_map)
        except Exception as e:
            logger.debug(f"[{self.instrument_name}] quantize_to_tempo_map failed: {e}")
        
        return notes
    
    def apply_ai_filters(self, notes: List[Any], section: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        AI処理の雛形（サブクラスで実装）
        
        デフォルトはNO-OP（ノートをそのまま返す）
        """
        return notes
