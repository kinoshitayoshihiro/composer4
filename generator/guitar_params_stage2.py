#!/usr/bin/env python3
"""
Guitar Params Stage2 - YAML駆動パラメータ適用システム

目的:
- ストラム vs アルペジオ
- Down/Up lag
- Rake/Slide

Phase:
11: 密度（ストラム頻度・アルペジオ率）
12: レンジ/カポ（オクターブ・ポジション）
13: 語彙（rake/slide/hammer-on/pull-off）
14: 和声（トライアド・sus配置）
18: 遷移（pickup/fill）
20: Humanize（down/up lag）
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from music21 import note, stream
except ImportError:
    raise ImportError("music21 required: pip install music21")

from generator.instrument_stage2_base import InstrumentStage2Base, load_yaml_presets, normalize_density

logger = logging.getLogger(__name__)


class GuitarParamsStage2(InstrumentStage2Base):
    """Guitar Params Stage2: Strum＋Arpeggio＋Lag"""
    
    def __init__(
        self,
        style_presets: Optional[Dict[str, Any]] = None,
        vocab_presets: Optional[Dict[str, Any]] = None
    ):
        super().__init__("guitar", style_presets, vocab_presets)
    
    def _get_phases(self, params: Optional[Dict[str, Any]] = None) -> List[int]:
        """実行Phase: 基本11/12/20 + 設定があればPhase 13-28追加"""
        ph = [11, 12, 20]
        
        # Phase 13-19の設定があれば追加
        adv = params or {}
        if any(k in adv for k in ("vocabulary", "harmonic", "cross_sync", "transition", "articulation", "dynamics", "groove")):
            ph = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
        # Phase 22/24/23の設定があれば追加（独立して有効化）
        if adv.get("emotion_map"):
            if 22 not in ph:
                ph.append(22)
        if adv.get("controls"):
            if 24 not in ph:
                ph.append(24)
        if adv.get("prosody", {}).get("enable"):
            if 23 not in ph:
                ph.append(23)
        
        # Phase 25-28の設定があれば追加
        if adv.get("sparsify", {}).get("enable"):
            if 25 not in ph:
                ph.append(25)
        if adv.get("harmony", {}).get("source") == "hybrid":
            if 26 not in ph:
                ph.append(26)
        # Phase 31は26直後が理想（和声情報を使うため）
        if adv.get("voice_leading", {}).get("enable"):
            if 31 not in ph:
                ph.append(31)
        if adv.get("style_adapt", {}).get("enable"):
            if 27 not in ph:
                ph.append(27)
        # Phase 30は27後が理想（スタイル適応後にバランス調整）
        if adv.get("xinst_balance"):
            if 30 not in ph:
                ph.append(30)
        if adv.get("export"):
            if 28 not in ph:
                ph.append(28)
        
        # Phase 29の設定があれば追加
        if adv.get("ducking", {}).get("enable"):
            if 29 not in ph:
                ph.append(29)
        
        # ソート（Phase番号順で実行）
        return sorted(ph)
    
    def _get_probability_keys(self) -> List[str]:
        return [
            "density.arpeggio_ratio",
            "strum.bias_down",
            "voicing.sus_bias",
        ]
    
    def _get_velocity_keys(self) -> List[str]:
        return [
            "dynamics.min_vel",
            "dynamics.max_vel",
        ]
    
    def _validate_instrument_specific(self, params: Dict[str, Any]) -> List[str]:
        warnings = []
        
        # レンジチェック（E2=40, E5=76 → 3オクターブ）
        register = params.get("register", {})
        if register:
            min_midi = register.get("min_midi", 40)
            max_midi = register.get("max_midi", 76)
            if not (36 <= min_midi <= max_midi <= 84):
                warnings.append(f"register [{min_midi},{max_midi}] out of guitar range")
        
        return warnings
    
    def _collect_metrics(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any]
    ) -> None:
        super()._collect_metrics(part, section_meta, mix_context, params)
        
        try:
            notes = list(part.flatten().notes)
            if not notes:
                return
            
            # ストラム判定（同時刻に3音以上）
            from collections import defaultdict
            offset_notes = defaultdict(list)
            for n in notes:
                if hasattr(n, 'offset'):
                    offset_notes[round(n.offset, 3)].append(n)
            
            strums = sum(1 for v in offset_notes.values() if len(v) >= 3)
            self.metrics["strum_count"] = strums
            
            # ダウンストローク比率（仮）
            self.metrics["downstroke_ratio"] = 0.65  # placeholder
        
        except Exception as e:
            logger.warning(f"Guitar metrics collection error: {e}")
    
    def _phase_11(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 11: 密度整形"""
        density_cfg = normalize_density(params.get("density"))
        if not density_cfg:
            return
        
        # 正規化後はnotes_per_barで取得可能
        strums_per_bar = density_cfg.get("notes_per_bar", {"min": 4, "max": 8})
        logger.debug(f"[Guitar] Phase 11: strums_per_bar={strums_per_bar} (normalized)")

    
    def _phase_12(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 12: レンジ補正（カポ）"""
        register_cfg = params.get("register")
        if not register_cfg:
            return
        
        min_midi = int(register_cfg.get("min_midi", 40))  # E2
        max_midi = int(register_cfg.get("max_midi", 76))  # E5
        
        notes = list(part.recurse().notes)
        for n in notes:
            if not isinstance(n, note.Note):
                continue
            
            # 範囲外なら1オクターブシフト
            if n.pitch.midi < min_midi:
                n.pitch.midi += 12
            elif n.pitch.midi > max_midi:
                n.pitch.midi -= 12
    
    def _phase_13_vocabulary(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 13: ギター語彙（ストラム装飾/rake/slide）"""
        v = params.get("vocabulary") or {}
        if not v:
            return
        
        try:
            rake_prob = float(v.get("rake_prob", 0.2))
            slide_prob = float(v.get("slide_prob", 0.15))
            
            if rake_prob <= 0.0 and slide_prob <= 0.0:
                return
            
            rng = random.Random(seed) if seed is not None else random
            
            notes = list(part.flatten().notes)
            for i in range(len(notes)):
                n = notes[i]
                
                # Rake適用（3弦以上のストラムに僅かな時間差）
                if rng.random() < rake_prob:
                    if hasattr(n, 'offset'):
                        n.offset += rng.uniform(0.0, 0.03)  # 最大30ms
                
                # Slide適用（次の音へのスライド）
                if i < len(notes) - 1 and rng.random() < slide_prob:
                    next_n = notes[i + 1]
                    if hasattr(n, 'pitch') and hasattr(next_n, 'pitch'):
                        interval = abs(next_n.pitch.midi - n.pitch.midi)
                        if 2 <= interval <= 5:
                            # velocity 僅かに減少（スライド感）
                            n.volume.velocity = max(50, n.volume.velocity - 8)
            
            logger.debug(f"[Guitar] Phase 13: Applied vocabulary elements")
        
        except Exception as e:
            logger.debug(f"[Guitar] Phase 13 vocabulary skipped: {e}")
    
    def _phase_14_harmonic_awareness(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 14: コード形状選択（パワーコード/開放弦優先）"""
        h = params.get("harmonic") or {}
        if not h:
            return
        
        try:
            power_chord_bias = float(h.get("power_chord_bias", 0.3))
            open_string_bias = float(h.get("open_string_bias", 0.5))
            
            # 簡易実装: パワーコード（Root + 5th）を優先
            # 実際の実装では、コード情報から適切なボイシングを選択
            logger.debug(f"[Guitar] Phase 14: power_chord_bias={power_chord_bias}, open_string_bias={open_string_bias}")
        
        except Exception as e:
            logger.debug(f"[Guitar] Phase 14 harmonic awareness skipped: {e}")
    
    def _phase_15_cross_instrument_sync(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 15: HHの8分グリッドにストロークを寄せる"""
        cs = params.get("cross_sync") or {}
        if not cs or not cs.get("sync_with_hihat"):
            return
        
        try:
            window_ms = float(cs.get("sync_window_ms", 30))
            tempo = section_meta.get("tempo", 120)
            window_ql = (window_ms / 1000.0) * (tempo / 60.0)
            
            # Hi-hat onsets取得
            hh_onsets = mix_context.get("hihat_onsets_ql", [])
            if not hh_onsets:
                return
            
            notes = list(part.flatten().notes)
            for n in notes:
                if not hasattr(n, 'offset'):
                    continue
                
                # 最も近い HH onset を探す
                for hh_off in hh_onsets:
                    if abs(n.offset - hh_off) <= window_ql:
                        # Hi-hatに同期
                        n.offset = hh_off
                        break
        
        except Exception as e:
            logger.debug(f"[Guitar] Phase 15 cross-sync skipped: {e}")
    
    def _phase_16_transition_smoothing(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 16: 遷移平滑化（共通ヘルパー呼び出し）"""
        self._apply_transition_curve(part, section_meta, params)
    
    def _phase_17_articulation_refinement(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 17: アーティキュレーション（ハンマーオン/プルオフ）"""
        art = params.get("articulation") or {}
        if not art:
            return
        
        try:
            hammer_prob = float(art.get("hammer_on_prob", 0.15))
            pull_prob = float(art.get("pull_off_prob", 0.1))
            
            if hammer_prob <= 0.0 and pull_prob <= 0.0:
                return
            
            rng = random.Random(seed) if seed is not None else random
            
            notes = list(part.flatten().notes)
            for i in range(1, len(notes)):
                prev_n = notes[i - 1]
                curr_n = notes[i]
                
                if not hasattr(prev_n, 'pitch') or not hasattr(curr_n, 'pitch'):
                    continue
                
                interval = curr_n.pitch.midi - prev_n.pitch.midi
                
                # Hammer-on（上昇）
                if 1 <= interval <= 3 and rng.random() < hammer_prob:
                    curr_n.volume.velocity = max(40, curr_n.volume.velocity - 15)
                
                # Pull-off（下降）
                if -3 <= interval <= -1 and rng.random() < pull_prob:
                    curr_n.volume.velocity = max(40, curr_n.volume.velocity - 12)
        
        except Exception as e:
            logger.debug(f"[Guitar] Phase 17 articulation skipped: {e}")
    
    def _phase_18_dynamics_shaping(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 18: ダイナミクス整形（共通ヘルパー呼び出し）"""
        self._apply_dynamics_curve(part, params)
    
    def _phase_19_groove_micro_timing(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 19: グルーヴマイクロタイミング（共通ヘルパー呼び出し）"""
        tempo = section_meta.get("tempo", 120)
        self._apply_groove_timing(part, tempo, params)
    
    def _phase_20(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 20: Humanize（ストロークlag）"""
        humanize_cfg = params.get("humanize")
        if not humanize_cfg:
            return
        
        timing_ms = float(humanize_cfg.get("timing_ms", 10.0))
        vel_sigma = float(humanize_cfg.get("vel_sigma", 6.0))
        
        # Guitarは打弦アタックのゆらぎ大きめ
        timing_ms *= 1.2
        
        tempo = section_meta.get("tempo", 120)
        ms_per_quarter = 60000.0 / tempo
        timing_ql = timing_ms / ms_per_quarter
        
        if seed is not None:
            import hashlib
            part_tag = getattr(part, "id", getattr(part, "partName", "guitar"))
            h = hashlib.md5(f"{seed}:{part_tag}".encode()).hexdigest()
            rng = random.Random(int(h[:8], 16))
        else:
            rng = random
        
        notes = list(part.recurse().notes)
        for n in notes:
            if not isinstance(n, note.Note):
                continue
            
            # タイミングゆらぎ
            if hasattr(n, 'offset'):
                offset_shift = rng.uniform(-timing_ql, timing_ql)
                new_off = n.offset + offset_shift
                n.offset = new_off if new_off >= 0.0 else 0.0
            
            # ベロシティゆらぎ（ピッキング強弱）
            if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                vel_shift = int(rng.gauss(0, vel_sigma))
                new_vel = max(1, min(127, n.volume.velocity + vel_shift))
                n.volume.velocity = new_vel
        
        # Phase 29: Vocal-Aware Ducking（ボーカルに寄り添う）
        duck = params.get("ducking") or {}
        self._apply_vocal_ducking(part, mix_context, duck)
    
    def _phase_22(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 22: Emotion mapping（連続写像）"""
        tempo = section_meta.get("tempo", 120)
        ql_per_bar = section_meta.get("ql_per_bar", 4.0)
        self._apply_emotion_map(part, params, role="guitar", ql_per_bar=ql_per_bar, bpm=tempo)
    
    def _phase_24(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 24: Controls（RPN/PB/CC11）"""
        tempo = section_meta.get("tempo", 120)
        self._apply_controls_unified(part, params, bpm=tempo)
    
    def _phase_23(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 23: Prosody（子音窓アライン）"""
        tempo = section_meta.get("tempo", 120)
        self._apply_prosody_alignment(part, params, bpm=tempo)
    
    def _phase_25(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 25: Sparsify（間引き）& Collision Avoidance"""
        sp = params.get("sparsify") or {}
        if not sp.get("enable", False):
            return
        
        bpm = float(section_meta.get("tempo") or (mix_context.get("beat_grid", {}).get("bpm", 120.0)))
        self._thin_notes_even(
            part,
            keep_endpoints=bool(sp.get("keep_endpoints", True)),
            min_gap_ms=float(sp.get("min_gap_ms", 18)),
            step_count=sp.get("step_count"),
            bpm=bpm
        )
        
        # 高域衝突回避（Guitar vs Piano RH）
        self._avoid_register_collision(
            part,
            band_low=int(sp.get("band_low", 64)),
            band_high=int(sp.get("band_high", 90)),
            strategy=str(sp.get("strategy", "vel_first")),
            reduce_db=float(sp.get("reduce_db", 6)),
            drop_prob=float(sp.get("drop_prob", 0.35)),
            seed=seed
        )
    
    def _phase_26(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 26: Hybrid Harmony（audio × creative 混合）"""
        harm = params.get("harmony") or {}
        if harm.get("source") != "hybrid":
            return
        
        self._blend_harmony(
            part,
            audio_chordmap=mix_context.get("audio_chordmap", {}),
            creative_chordmap=mix_context.get("creative_chordmap", {}),
            blend=float(harm.get("blend", 0.5)),
            keep_audio_root=bool(harm.get("keep_audio_root", True)),
            allow_text_tensions=harm.get("allow_text_tensions", [9, 11, 13])
        )
    
    def _phase_27(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 27: Style Adaptation（活動→プリセット自動切替）"""
        ad = params.get("style_adapt") or {}
        if not ad.get("enable", False):
            return
        
        try:
            level = self._window_activity("guitar", int(section_meta.get("bar", 0)), int(ad.get("window_bars", 4)))
            pdict = ad.get("presets_dict", {})
            params.update(
                self._adapt_style_params(
                    params, pdict, level,
                    edges=tuple(ad.get("low_high", [0.25, 0.75])),
                    order=ad.get("order", ["simple", "moderate", "complex", "intense"])
                )
            )
        except Exception:
            return
    
    def _phase_28(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 28: Export Postprocess（量子化・命名）"""
        ex = params.get("export") or {}
        if not ex:
            return
        
        self.postprocess_export(
            part, role="guitar", section_meta=section_meta, params=params,
            ql_quant=float(ex.get("quantize_ql", 0.25)),
            track_split=ex.get("track_split"),
            name_fmt=ex.get("name_fmt"),
            markers=ex.get("markers")
        )
    
    def _phase_30(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 30: Cross-Instrument Balance（他楽器活動度による譲歩）"""
        try:
            role = getattr(self, "role", None) or getattr(self, "_role", None)
            role = (role or self.__class__.__name__).replace("ParamsStage2", "").lower()
            bal = params.get("xinst_balance") or {}
            for k, spec in bal.items():
                if not str(k).startswith("vs_"):
                    continue
                against = str(k)[3:]
                self._rebalance_against(part, mix_context, spec, role=role, against_role=against)
        except Exception:
            return
    
    def _phase_31(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 31: Voice-Leading Guard（和声音優先＋跳躍制限）"""
        try:
            vl = params.get("voice_leading") or {}
            hints = part.get("hints") or {} if isinstance(part, dict) else {}
            chord_now = hints.get("blend_harmony") or {}
            chord_prev = section_meta.get("prev_chord") if isinstance(section_meta, dict) else {}
            self._voice_leading_smooth(part, section_meta, chord_now, chord_prev, vl)
        except Exception:
            return

