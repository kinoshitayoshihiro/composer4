#!/usr/bin/env python3
"""
Bass Params Stage2 - YAML駆動パラメータ適用システム

既存のBassGeneratorStage2とは別レイヤー。
生成されたBass Partに対してポストプロセスでパラメータ適用。

目的:
- Kickとロック
- アプローチノート（chromatic/diatonic）
- オクターブ設計
- 歩き（walk）語彙の適用

Phase:
11: 密度（notes_per_bar ∈ [2,12]、syncopation ≤ 0.45）
12: レンジ（E1–E3中心、飛び越し >12半音にソフト制限）
13: 語彙（bass_walks.yaml）
14: 和声（root:third:seventh 比率）
15: キック同期（±30ms以内）
18: 遷移（walk-up/walk-down）
20: Humanize
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from music21 import note, stream, pitch as m21pitch
except ImportError:
    raise ImportError("music21 required: pip install music21")

from generator.instrument_stage2_base import InstrumentStage2Base, load_yaml_presets, normalize_density

logger = logging.getLogger(__name__)


class BassParamsStage2(InstrumentStage2Base):
    """
    Bass Params Stage2: Kickロック＋アプローチ＋Walk語彙
    
    設計:
    - NO-OP既定（設定なしなら何もしない）
    - YAML駆動プリセット（tight_pop等）
    - Phase単位でON/OFF可能
    - 既存BassGeneratorの出力を後処理
    """
    
    def __init__(
        self,
        style_presets: Optional[Dict[str, Any]] = None,
        vocab_presets: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            style_presets: bass_style_presets.yaml から読み込んだプリセット
            vocab_presets: bass_walks.yaml から読み込んだ語彙（任意）
        """
        super().__init__("bass", style_presets, vocab_presets)
    
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
        
        # Phase 25/27/28の設定があれば追加
        if adv.get("sparsify", {}).get("enable"):
            if 25 not in ph:
                ph.append(25)
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
        
        # ソート（Phase番号順で実行）
        return sorted(ph)
    
    def _get_probability_keys(self) -> List[str]:
        """確率値として検証すべきキー"""
        return [
            "approach.chromatic_prob",
            "approach.diatonic_prob",
            "approach.octave_jump_prob",
            "lock_with_kick.strength",
        ]
    
    def _get_velocity_keys(self) -> List[str]:
        """ベロシティ値として検証すべきキー"""
        return [
            "dynamics.min_vel",
            "dynamics.max_vel",
        ]
    
    def _validate_instrument_specific(self, params: Dict[str, Any]) -> List[str]:
        """Bass固有のバリデーション"""
        warnings = []
        
        # 密度チェック
        density = params.get("density", {})
        if density:
            npb = density.get("notes_per_bar", {})
            if isinstance(npb, dict):
                min_npb = npb.get("min", 2)
                max_npb = npb.get("max", 12)
                if not (2 <= min_npb <= max_npb <= 12):
                    warnings.append(f"notes_per_bar range [{min_npb},{max_npb}] unreasonable")
        
        # レンジチェック（E1=40, E3=64）
        register = params.get("register", {})
        if register:
            min_midi = register.get("min_midi", 36)
            max_midi = register.get("max_midi", 64)
            if not (28 <= min_midi <= max_midi <= 72):
                warnings.append(f"register [{min_midi},{max_midi}] out of reasonable bass range")
        
        return warnings
    
    def _collect_metrics(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any]
    ) -> None:
        """Bass固有メトリクス"""
        super()._collect_metrics(part, section_meta, mix_context, params)
        
        try:
            notes = list(part.flatten().notes)
            if not notes:
                return
            
            # オクターブジャンプ率
            jumps = 0
            for i in range(1, len(notes)):
                if hasattr(notes[i], 'pitch') and hasattr(notes[i-1], 'pitch'):
                    diff = abs(notes[i].pitch.midi - notes[i-1].pitch.midi)
                    if diff >= 12:
                        jumps += 1
            
            self.metrics["octave_jump_rate"] = jumps / max(1, len(notes) - 1)
            
            # Kickロック率（mix_contextからkick_onsets取得）
            kick_onsets = mix_context.get("kick_onsets_ql", [])
            if kick_onsets:
                locked = 0
                for n in notes:
                    for k_off in kick_onsets:
                        if abs(float(n.offset) - float(k_off)) < 0.1:  # ±0.1 QL
                            locked += 1
                            break
                self.metrics["lock_ratio_with_kick"] = locked / len(notes)
        
        except Exception as e:
            logger.warning(f"Bass metrics collection error: {e}")
    
    # ========================================================================
    # Phase実装
    # ========================================================================
    
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
        
        # TODO: notes_per_bar / syncopation の調整
        # 現状はプレースホルダ（実装は後段）
        logger.debug(f"[Bass] Phase 11: density config detected (normalized)")

    
    def _phase_12(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 12: レンジ補正"""
        register_cfg = params.get("register")
        if not register_cfg:
            return
        
        min_midi = int(register_cfg.get("min_midi", 36))  # C2
        max_midi = int(register_cfg.get("max_midi", 64))  # E4
        
        notes = list(part.recurse().notes)
        for n in notes:
            if not isinstance(n, note.Note):
                continue
            
            # 範囲外なら1オクターブシフト
            if n.pitch.midi < min_midi:
                n.pitch.midi = min(n.pitch.midi + 12, max_midi)
            elif n.pitch.midi > max_midi:
                n.pitch.midi = max(n.pitch.midi - 12, min_midi)
    
    def _phase_13_vocabulary(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]
    ) -> None:
        """Phase 13: ベース語彙（ピックアップ/アプローチノート）"""
        v = params.get("vocabulary") or {}
        if not v:
            return
        
        try:
            pickup_prob = float(v.get("pickup_prob", 0.3))
            approach_prob = float(v.get("approach_prob", 0.2))
            
            if pickup_prob <= 0.0 and approach_prob <= 0.0:
                return
            
            # RNG初期化
            rng = random.Random(seed) if seed is not None else random
            
            # セクション末尾にピックアップノート追加
            notes = list(part.flatten().notes)
            if not notes or rng.random() > pickup_prob:
                return
            
            # 最後のノートからR→5度への簡易ピックアップ（半拍）
            last_note = notes[-1]
            if not hasattr(last_note, 'pitch'):
                return
            
            # 5度上のピックアップ
            pickup_pitch = last_note.pitch.midi + 7  # Perfect 5th
            pickup = note.Note(pickup_pitch, quarterLength=0.5)
            pickup.volume.velocity = max(60, last_note.volume.velocity - 10)
            
            # 最後のノートの後に追加
            insert_offset = last_note.offset + last_note.quarterLength - 0.5
            part.insert(max(0.0, insert_offset), pickup)
            
            logger.debug(f"[Bass] Phase 13: Added pickup note at offset {insert_offset}")
        
        except Exception as e:
            logger.debug(f"[Bass] Phase 13 vocabulary skipped: {e}")
    
    def _phase_14_harmonic_awareness(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 14: 和声認識（根音/5度優先）"""
        h = params.get("harmonic") or {}
        if not h:
            return
        
        try:
            prefer_root5 = float(h.get("prefer_root5", 0.8))
            
            # ハーモニー情報取得
            harmony = section_meta.get("harmony", [])
            if not harmony:
                return
            
            notes = list(part.flatten().notes)
            for n in notes:
                if not hasattr(n, 'pitch'):
                    continue
                
                # 現在の小節のコード取得
                bar_num = int(n.offset / 4.0)
                chord_info = next((c for c in harmony if c.get("bar") == bar_num), None)
                
                if not chord_info:
                    continue
                
                # 根音または5度に近いピッチを強調
                root = chord_info.get("root")
                if root:
                    # 根音（Root）と完全5度（Perfect 5th）に近い音を強調
                    pitch_class = n.pitch.pitchClass
                    root_pc = (root % 12)
                    fifth_pc = (root + 7) % 12
                    
                    if pitch_class == root_pc or pitch_class == fifth_pc:
                        # 根音/5度なら velocity +5
                        n.volume.velocity = min(127, n.volume.velocity + 5)
                    else:
                        # それ以外なら velocity -3
                        n.volume.velocity = max(40, n.volume.velocity - 3)
        
        except Exception as e:
            logger.debug(f"[Bass] Phase 14 harmonic awareness skipped: {e}")
    
    def _phase_15_cross_instrument_sync(
        self,
        part: Any,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int]) -> None:
        """Phase 15: クロス楽器同期（Kickとロック）"""
        cs = params.get("cross_sync") or {}
        if not cs or not cs.get("lock_with_kick"):
            return
        
        try:
            window_ms = float(cs.get("sync_window_ms", cs.get("window_ms", 30)))
            
            # Kick onsets取得
            kick_onsets = mix_context.get("kick_onsets_ql", mix_context.get("drums_kick_onsets", []))
            if not kick_onsets:
                return
            
            tempo = section_meta.get("tempo", 120)
            window_ql = (window_ms / 1000.0) * (tempo / 60.0)
            
            notes = list(part.flatten().notes)
            for n in notes:
                if not hasattr(n, 'offset'):
                    continue
                
                # 最も近いKick onsetを探す
                for kick_off in kick_onsets:
                    if abs(n.offset - kick_off) <= window_ql:
                        # Kickに同期（タイミング調整）
                        n.offset = kick_off
                        # velocity +5 で強調
                        n.volume.velocity = min(127, n.volume.velocity + 5)
                        break
        
        except Exception as e:
            logger.debug(f"[Bass] Phase 15 cross-sync skipped: {e}")
    
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
        """Phase 17: アーティキュレーション細分化（レガート/スライド）"""
        art = params.get("articulation") or {}
        if not art:
            return
        
        try:
            legato_prob = float(art.get("legato_prob", 0.0))
            slide_prob = float(art.get("slide_prob", 0.0))
            
            if legato_prob <= 0.0 and slide_prob <= 0.0:
                return
            
            rng = random.Random(seed) if seed is not None else random
            
            notes = list(part.flatten().notes)
            for i in range(1, len(notes)):
                prev_n = notes[i - 1]
                curr_n = notes[i]
                
                if not hasattr(prev_n, 'pitch') or not hasattr(curr_n, 'pitch'):
                    continue
                
                interval = abs(curr_n.pitch.midi - prev_n.pitch.midi)
                
                # レガート適用（隣接音）
                if interval <= 2 and rng.random() < legato_prob:
                    # 前のノートの長さを僅かに延長（レガート効果）
                    prev_n.quarterLength = min(prev_n.quarterLength * 1.05, 4.0)
                
                # スライド適用（3半音以上）
                if 3 <= interval <= 7 and rng.random() < slide_prob:
                    # velocity僅かに下げる（スライド感）
                    curr_n.volume.velocity = max(50, curr_n.volume.velocity - 5)
        
        except Exception as e:
            logger.debug(f"[Bass] Phase 17 articulation skipped: {e}")
    
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
        
        # 低域はタイミングゆらぎ少なめ
        timing_ms *= 0.7
        
        tempo = section_meta.get("tempo", 120)
        ms_per_quarter = 60000.0 / tempo
        timing_ql = timing_ms / ms_per_quarter
        
        # パート固有RNG
        if seed is not None:
            import hashlib
            part_tag = getattr(part, "id", getattr(part, "partName", "bass"))
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
            
            # ベロシティゆらぎ（Bassは広め）
            if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                vel_shift = int(rng.gauss(0, vel_sigma * 1.2))  # 1.2倍
                new_vel = max(1, min(127, n.volume.velocity + vel_shift))
                n.volume.velocity = new_vel
    
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
        self._apply_emotion_map(part, params, role="bass", ql_per_bar=ql_per_bar, bpm=tempo)
    
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
            min_gap_ms=float(sp.get("min_gap_ms", 20)),
            step_count=sp.get("step_count"),
            bpm=bpm
        )
        # Bass 自身は衝突回避の対象外（他パート側で調整）
    
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
            level = self._window_activity("bass", int(section_meta.get("bar", 0)), int(ad.get("window_bars", 4)))
            pdict = ad.get("presets_dict", {})  # 事前にロード済み想定
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
            part, role="bass", section_meta=section_meta, params=params,
            ql_quant=float(ex.get("quantize_ql", 0.25)),
            track_split=ex.get("track_split"),
            name_fmt=str(ex.get("name_fmt", "{idx:02d}_{role}_{section}"))
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


# ========================================================================
# デモ実行
# ========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Bass Params Stage2 Demo")
    print("=" * 60)
    
    # プリセット読み込み
    preset_path = Path(__file__).parent.parent / "data" / "presets" / "bass_style_presets.yaml"
    style_presets = load_yaml_presets(preset_path)
    
    gen = BassParamsStage2(style_presets=style_presets)
    
    # ダミーPart作成
    bass_part = stream.Part()
    for i in range(16):
        n = note.Note("E2", quarterLength=1.0)
        n.volume.velocity = 80
        bass_part.insert(float(i), n)
    
    print(f"\n🎸 Original: {len(list(bass_part.flatten().notes))} notes")
    
    # Stage2適用
    section_meta = {
        "label": "Verse",
        "bar": 0,
        "emotion": "energetic",
        "bass_style": "tight_pop",
        "tempo": 120
    }
    
    mix_context = {
        "kick_onsets_ql": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    }
    
    result = gen.apply(
        bass_part,
        section_meta,
        mix_context,
        overrides={"register": {"min_midi": 40, "max_midi": 60}},
        seed=42
    )
    
    print(f"✅ Processed: {len(list(result.flatten().notes))} notes")
    print(f"📊 Metrics: {gen.metrics}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

