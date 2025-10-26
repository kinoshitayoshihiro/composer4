#!/usr/bin/env python3
"""
Strings Params Stage2 - YAML駆動パラメータ適用システム

目的:
- Pad vs Ostinato
- Divisi/レイヤ
- ダイナミクス・スウェル

Phase:
11: 密度（Pad=低密度＆長音価、Ostinato=規則長）
12: レンジ/セクション割り（Vn/Va/Vc/Bass的なレンジ）
13: 語彙（strings_ostinato_presets.yaml）
14: 和声（度数配置指示）
18: 遷移（スウェル/トレモロ）
20: Humanize
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


class StringsParamsStage2(InstrumentStage2Base):
    """Strings Params Stage2: Pad＋Ostinato＋Swell"""
    
    def __init__(
        self,
        style_presets: Optional[Dict[str, Any]] = None,
        vocab_presets: Optional[Dict[str, Any]] = None
    ):
        super().__init__("strings", style_presets, vocab_presets)
    
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
            "density.sustain_ratio_min",
            "density.movement_rate_max",
            "swell.enable",
            "swell.depth",
        ]
    
    def _get_velocity_keys(self) -> List[str]:
        return [
            "dynamics.min_vel",
            "dynamics.max_vel",
        ]
    
    def _validate_instrument_specific(self, params: Dict[str, Any]) -> List[str]:
        warnings = []
        
        # レンジチェック（G2=43, E6=88）
        register = params.get("register", {})
        if register:
            min_midi = register.get("min_midi", 43)
            max_midi = register.get("max_midi", 88)
            if not (36 <= min_midi <= max_midi <= 96):
                warnings.append(f"register [{min_midi},{max_midi}] out of strings range")
        
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
            
            # サステイン比率
            long_notes = sum(1 for n in notes if n.quarterLength >= 2.0)
            self.metrics["sustain_ratio"] = long_notes / len(notes)
            
            # レンジ幅
            pitches = [n.pitch.midi for n in notes if hasattr(n, 'pitch')]
            if pitches:
                self.metrics["register_spread_semitones"] = max(pitches) - min(pitches)
        
        except Exception as e:
            logger.warning(f"Strings metrics collection error: {e}")
    
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
        
        mode = density_cfg.get("mode", "pad")
        logger.debug(f"[Strings] Phase 11: mode={mode} (normalized)")

    
    def _phase_12(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 12: レンジ補正＋セクション割り"""
        register_cfg = params.get("register")
        if not register_cfg:
            return
        
        min_midi = int(register_cfg.get("min_midi", 43))  # G2
        max_midi = int(register_cfg.get("max_midi", 88))  # E6
        
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
        """Phase 13: ストリングス語彙（ミニフィル/リードイン）"""
        v = params.get("vocabulary") or {}
        if not v:
            return
        
        try:
            fill_prob = float(v.get("mini_fill_prob", 0.3))
            leadin_prob = float(v.get("leadin_prob", 0.2))
            
            if fill_prob <= 0.0 and leadin_prob <= 0.0:
                return
            
            rng = random.Random(seed) if seed is not None else random
            
            # セクション末尾にミニフィル（上昇/下降スケール）
            if rng.random() < fill_prob:
                notes = list(part.flatten().notes)
                if notes:
                    last_offset = max(n.offset for n in notes if hasattr(n, 'offset'))
                    
                    # 上昇スケール（半拍 × 4音）
                    fill_pitches = [64, 66, 67, 69]  # E4-F#4-G4-A4
                    for i, pitch in enumerate(fill_pitches):
                        fn = note.Note(pitch, quarterLength=0.5)
                        fn.volume.velocity = 70
                        part.insert(last_offset + i * 0.5, fn)
            
            logger.debug(f"[Strings] Phase 13: Added vocabulary elements")
        
        except Exception as e:
            logger.debug(f"[Strings] Phase 13 vocabulary skipped: {e}")
    
    def _phase_14_harmonic_awareness(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 14: テンション抑制（衝突回避）"""
        h = params.get("harmonic") or {}
        if not h:
            return
        
        try:
            tension_avoid = float(h.get("tension_avoid", 0.7))
            
            # 簡易実装: 半音衝突を検出して velocity 減少
            notes = list(part.flatten().notes)
            for i in range(1, len(notes)):
                prev_n = notes[i - 1]
                curr_n = notes[i]
                
                if not hasattr(prev_n, 'pitch') or not hasattr(curr_n, 'pitch'):
                    continue
                
                interval = abs(curr_n.pitch.midi - prev_n.pitch.midi)
                
                # 半音衝突
                if interval == 1 and random.random() < tension_avoid:
                    curr_n.volume.velocity = max(40, curr_n.volume.velocity - 10)
        
        except Exception as e:
            logger.debug(f"[Strings] Phase 14 harmonic awareness skipped: {e}")
    
    def _phase_15_cross_instrument_sync(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 15: クロス楽器同期（Stringsは同期なし、スウェル専念）"""
        # Stringsは他楽器と同期せず、Phase 16/18でスウェルに専念
        pass
    
    def _phase_16_transition_smoothing(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 16: 遷移平滑化（スウェル = 共通ヘルパー呼び出し）"""
        self._apply_transition_curve(part, section_meta, params)
    
    def _phase_17_articulation_refinement(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 17: アーティキュレーション（トレモロ/ピチカート）"""
        art = params.get("articulation") or {}
        if not art:
            return
        
        try:
            tremolo_prob = float(art.get("tremolo_prob", 0.1))
            pizzicato_prob = float(art.get("pizzicato_prob", 0.05))
            
            if tremolo_prob <= 0.0 and pizzicato_prob <= 0.0:
                return
            
            rng = random.Random(seed) if seed is not None else random
            
            notes = list(part.flatten().notes)
            for n in notes:
                # トレモロ適用（長音）
                if n.quarterLength >= 2.0 and rng.random() < tremolo_prob:
                    # velocity僅かに上げる（トレモロ強度）
                    n.volume.velocity = min(127, n.volume.velocity + 8)
                
                # ピチカート適用（短音）
                if n.quarterLength <= 0.5 and rng.random() < pizzicato_prob:
                    n.quarterLength *= 0.5  # より短く
                    n.volume.velocity = max(60, n.volume.velocity - 10)
        
        except Exception as e:
            logger.debug(f"[Strings] Phase 17 articulation skipped: {e}")
    
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
        """Phase 19: グルーヴマイクロタイミング（Stringsは僅少）"""
        # Stringsはグルーヴを優先せず、僅かなレイドバックのみ
        grv = params.get("groove") or {}
        if not grv:
            return
        
        try:
            laidback_ms = float(grv.get("laidback_ms", grv.get("laid_back_ms", 5.0)))
            
            if laidback_ms <= 0.0:
                return
            
            tempo = section_meta.get("tempo", 120)
            laidback_ql = (laidback_ms / 1000.0) * (tempo / 60.0)
            
            notes = list(part.flatten().notes)
            for n in notes:
                if hasattr(n, 'offset'):
                    n.offset += laidback_ql
        
        except Exception as e:
            logger.debug(f"[Strings] Phase 19 groove timing skipped: {e}")
    
    def _phase_20(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 20: Humanize"""
        humanize_cfg = params.get("humanize")
        if not humanize_cfg:
            return
        
        timing_ms = float(humanize_cfg.get("timing_ms", 8.0))
        vel_sigma = float(humanize_cfg.get("vel_sigma", 5.0))
        
        # Stringsはタイミングゆらぎ僅少
        timing_ms *= 0.5
        
        tempo = section_meta.get("tempo", 120)
        ms_per_quarter = 60000.0 / tempo
        timing_ql = timing_ms / ms_per_quarter
        
        if seed is not None:
            import hashlib
            part_tag = getattr(part, "id", getattr(part, "partName", "strings"))
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
            
            # ベロシティゆらぎ（CCダイナミクス的な滑らかさ）
            if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                vel_shift = int(rng.gauss(0, vel_sigma * 0.8))  # 控えめ
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
        self._apply_emotion_map(part, params, role="strings", ql_per_bar=ql_per_bar, bpm=tempo)
    
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
        """Phase 25: Sparsify（間引き）"""
        sp = params.get("sparsify") or {}
        if not sp.get("enable", False):
            return
        
        bpm = float(section_meta.get("tempo") or (mix_context.get("beat_grid", {}).get("bpm", 120.0)))
        self._thin_notes_even(
            part,
            keep_endpoints=bool(sp.get("keep_endpoints", True)),
            min_gap_ms=float(sp.get("min_gap_ms", 25)),
            step_count=sp.get("step_count"),
            bpm=bpm
        )
        # Strings は通常中域なので衝突回避不要
    
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
            blend=float(harm.get("blend", 0.6)),
            keep_audio_root=bool(harm.get("keep_audio_root", True)),
            allow_text_tensions=harm.get("allow_text_tensions", [9, 11])
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
            level = self._window_activity("strings", int(section_meta.get("bar", 0)), int(ad.get("window_bars", 4)))
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
        """Phase 28: Export Postprocess（量子化・トラック分割・命名）"""
        ex = params.get("export") or {}
        if not ex:
            return
        
        self.postprocess_export(
            part, role="strings", section_meta=section_meta, params=params,
            ql_quant=float(ex.get("quantize_ql", 0.25)),
            track_split=ex.get("track_split", ["Long", "Short"]),
            name_fmt=str(ex.get("name_fmt", "{idx:02d}_{role}_{section}")),
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

