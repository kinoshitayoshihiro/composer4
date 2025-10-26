#!/usr/bin/env python3
"""
Piano Params Stage2 - YAML駆動パラメータ適用システム

目的:
- コンピング（リズム型紙）
- ボイシング（drop2/open/close）
- ペダル処理
- ボイスリーディング

Phase:
11: 密度（chords_per_bar ∈ [2,6]、arpeggio_ratio_max）
12: レンジ/左右分割（split_pitch≈60、重複ノート整理）
13: 語彙（piano_comp_presets.yaml）
14: 和声（3rd/7th露出、テンション挿入率）
18: 遷移（spread/open、cadence強調）
20: Humanize
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from music21 import note, stream, chord
except ImportError:
    raise ImportError("music21 required: pip install music21")

from generator.instrument_stage2_base import InstrumentStage2Base, load_yaml_presets, normalize_density

logger = logging.getLogger(__name__)


class PianoParamsStage2(InstrumentStage2Base):
    """Piano Params Stage2: コンピング＋ボイシング＋ペダル"""
    
    def __init__(
        self,
        style_presets: Optional[Dict[str, Any]] = None,
        vocab_presets: Optional[Dict[str, Any]] = None
    ):
        super().__init__("piano", style_presets, vocab_presets)
    
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
            "density.arpeggio_ratio_max",
            "voicing.tension_rate",
        ]
    
    def _get_velocity_keys(self) -> List[str]:
        return [
            "dynamics.min_vel",
            "dynamics.max_vel",
        ]
    
    def _validate_instrument_specific(self, params: Dict[str, Any]) -> List[str]:
        warnings = []
        
        # 和音密度チェック
        density = params.get("density", {})
        if density:
            cpb = density.get("chords_per_bar", {})
            if isinstance(cpb, dict):
                min_cpb = cpb.get("min", 2)
                max_cpb = cpb.get("max", 6)
                if not (1 <= min_cpb <= max_cpb <= 8):
                    warnings.append(f"chords_per_bar range [{min_cpb},{max_cpb}] unreasonable")
        
        # レンジチェック（A0=21, C8=108）
        register = params.get("register", {})
        if register:
            min_midi = register.get("min_midi", 21)
            max_midi = register.get("max_midi", 108)
            if not (21 <= min_midi <= max_midi <= 108):
                warnings.append(f"register [{min_midi},{max_midi}] out of piano range")
        
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
            # 和音カバレッジ
            chords_count = len([el for el in part.flatten() if isinstance(el, chord.Chord)])
            self.metrics["chord_count"] = chords_count
            
            # ボイスリーディング平均ステップ（簡易）
            # TODO: 実装
            
        except Exception as e:
            logger.warning(f"Piano metrics collection error: {e}")
    
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
        
        logger.debug(f"[Piano] Phase 11: density config detected (normalized)")

    
    def _phase_12(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 12: レンジ補正＋左右手分割"""
        register_cfg = params.get("register")
        if not register_cfg:
            return
        
        split_pitch = int(register_cfg.get("split_pitch", 60))  # Middle C
        
        notes = list(part.recurse().notes)
        for n in notes:
            if not isinstance(n, note.Note):
                continue
            
            # 範囲外なら1オクターブシフト
            if n.pitch.midi < 21:
                n.pitch.midi += 12
            elif n.pitch.midi > 108:
                n.pitch.midi -= 12
    
    def _phase_13_vocabulary(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 13: ピアノ語彙（ターンアラウンド/スケール運指）"""
        v = params.get("vocabulary") or {}
        if not v:
            return
        
        try:
            turnaround_prob = float(v.get("turnaround_prob", 0.3))
            scale_run_prob = float(v.get("scale_run_prob", 0.2))
            
            if turnaround_prob <= 0.0 and scale_run_prob <= 0.0:
                return
            
            rng = random.Random(seed) if seed is not None else random
            
            # セクション末尾にターンアラウンド（簡易: I-vi-ii-V的な1小節フレーズ）
            if rng.random() < turnaround_prob:
                notes = list(part.flatten().notes)
                if notes:
                    last_offset = max(n.offset for n in notes if hasattr(n, 'offset'))
                    
                    # 簡易ターンアラウンド: C-Am-Dm-G (root notes)
                    turnaround_pitches = [60, 57, 62, 55]  # C4, A3, D4, G3
                    for i, pitch in enumerate(turnaround_pitches):
                        tn = note.Note(pitch, quarterLength=1.0)
                        tn.volume.velocity = 75
                        part.insert(last_offset + i, tn)
            
            logger.debug(f"[Piano] Phase 13: Added vocabulary elements")
        
        except Exception as e:
            logger.debug(f"[Piano] Phase 13 vocabulary skipped: {e}")
    
    def _phase_14_harmonic_awareness(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 14: ボイシング和声拘束（ガイドトーン優先）"""
        h = params.get("harmonic") or {}
        if not h:
            return
        
        try:
            guide_tone_emphasis = float(h.get("guide_tone_emphasis", 0.7))
            
            harmony = section_meta.get("harmony", [])
            if not harmony:
                return
            
            notes = list(part.flatten().notes)
            for n in notes:
                if not hasattr(n, 'pitch'):
                    continue
                
                bar_num = int(n.offset / 4.0)
                chord_info = next((c for c in harmony if c.get("bar") == bar_num), None)
                
                if not chord_info:
                    continue
                
                # ガイドトーン（3rd/7th）を強調
                root = chord_info.get("root")
                if root:
                    pitch_class = n.pitch.pitchClass
                    third_pc = (root + 4) % 12  # Major 3rd
                    seventh_pc = (root + 11) % 12  # Major 7th
                    
                    if pitch_class == third_pc or pitch_class == seventh_pc:
                        # ガイドトーンなら velocity +8
                        n.volume.velocity = min(127, n.volume.velocity + 8)
        
        except Exception as e:
            logger.debug(f"[Piano] Phase 14 harmonic awareness skipped: {e}")
    
    def _phase_15_cross_instrument_sync(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 15: アクセントをスネア/キックと同期"""
        cs = params.get("cross_sync") or {}
        if not cs:
            return
        
        try:
            sync_with_snare = cs.get("sync_with_snare", False)
            sync_with_kick = cs.get("sync_with_kick", False)
            
            if not sync_with_snare and not sync_with_kick:
                return
            
            window_ms = float(cs.get("sync_window_ms", 30))
            tempo = section_meta.get("tempo", 120)
            window_ql = (window_ms / 1000.0) * (tempo / 60.0)
            
            # Snare/Kick onsets取得
            snare_onsets = mix_context.get("snare_onsets_ql", []) if sync_with_snare else []
            kick_onsets = mix_context.get("kick_onsets_ql", []) if sync_with_kick else []
            
            sync_onsets = snare_onsets + kick_onsets
            if not sync_onsets:
                return
            
            notes = list(part.flatten().notes)
            for n in notes:
                if not hasattr(n, 'offset'):
                    continue
                
                # 最も近い sync onset を探す
                for sync_off in sync_onsets:
                    if abs(n.offset - sync_off) <= window_ql:
                        # アクセント強調
                        n.volume.velocity = min(127, n.volume.velocity + 10)
                        break
        
        except Exception as e:
            logger.debug(f"[Piano] Phase 15 cross-sync skipped: {e}")
    
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
        """Phase 17: アーティキュレーション（ペダル/スタッカート）"""
        art = params.get("articulation") or {}
        if not art:
            return
        
        try:
            pedal_prob = float(art.get("pedal_prob", 0.5))
            staccato_prob = float(art.get("staccato_prob", 0.1))
            
            if pedal_prob <= 0.0 and staccato_prob <= 0.0:
                return
            
            rng = random.Random(seed) if seed is not None else random
            
            notes = list(part.flatten().notes)
            for n in notes:
                # ペダル適用（長めのノート）
                if n.quarterLength >= 1.0 and rng.random() < pedal_prob:
                    # quarterLength僅かに延長（ペダル効果）
                    n.quarterLength = min(n.quarterLength * 1.1, 4.0)
                
                # スタッカート適用（短めのノート）
                if n.quarterLength <= 0.5 and rng.random() < staccato_prob:
                    n.quarterLength *= 0.7  # 短く
                    n.volume.velocity = max(50, n.volume.velocity - 5)
        
        except Exception as e:
            logger.debug(f"[Piano] Phase 17 articulation skipped: {e}")
    
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
        """Phase 20: Humanize"""
        humanize_cfg = params.get("humanize")
        if not humanize_cfg:
            return
        
        timing_ms = float(humanize_cfg.get("timing_ms", 8.0))
        vel_sigma = float(humanize_cfg.get("vel_sigma", 5.0))
        
        # Pianoはやや後ろ（laidback）
        timing_bias = float(humanize_cfg.get("timing_bias_ms", 2.0))
        
        tempo = section_meta.get("tempo", 120)
        ms_per_quarter = 60000.0 / tempo
        timing_ql = timing_ms / ms_per_quarter
        bias_ql = timing_bias / ms_per_quarter
        
        if seed is not None:
            import hashlib
            part_tag = getattr(part, "id", getattr(part, "partName", "piano"))
            h = hashlib.md5(f"{seed}:{part_tag}".encode()).hexdigest()
            rng = random.Random(int(h[:8], 16))
        else:
            rng = random
        
        notes = list(part.recurse().notes)
        for n in notes:
            if not isinstance(n, note.Note):
                continue
            
            # タイミングゆらぎ + バイアス
            if hasattr(n, 'offset'):
                offset_shift = rng.uniform(-timing_ql, timing_ql) + bias_ql
                new_off = n.offset + offset_shift
                n.offset = new_off if new_off >= 0.0 else 0.0
            
            # ベロシティゆらぎ（拍頭は強め）
            if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                frac = n.offset % 1.0
                is_downbeat = abs(frac - 0.0) < 1e-3
                vel_scale = 1.3 if is_downbeat else 1.0
                
                vel_shift = int(rng.gauss(0, vel_sigma * vel_scale))
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
        self._apply_emotion_map(part, params, role="piano", ql_per_bar=ql_per_bar, bpm=tempo)
    
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
            min_gap_ms=float(sp.get("min_gap_ms", 15)),
            step_count=sp.get("step_count"),
            bpm=bpm
        )
        
        # 高域衝突回避（Piano RH vs Guitar）
        self._avoid_register_collision(
            part,
            band_low=int(sp.get("band_low", 60)),
            band_high=int(sp.get("band_high", 84)),
            strategy=str(sp.get("strategy", "vel_first")),
            reduce_db=float(sp.get("reduce_db", 6)),
            drop_prob=float(sp.get("drop_prob", 0.3)),
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
            level = self._window_activity("piano", int(section_meta.get("bar", 0)), int(ad.get("window_bars", 4)))
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
            part, role="piano", section_meta=section_meta, params=params,
            quantize_ql=ex.get("quantize_ql"), track_split=ex.get("track_split"),
            name_fmt=ex.get("name_fmt"), markers=ex.get("markers")
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

