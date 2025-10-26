#!/usr/bin/env python3
"""
Drums Params Stage2 - YAML駆動パラメータ適用システム

既存のDrumsGeneratorStage2とは別レイヤー。
生成されたDrums Partに対してポストプロセスでパラメータ適用。

目的:
- 密度コントロール（hits_per_bar）
- ダイナミクスレンジ調整
- ゴーストノート追加
- Humanization（タイミング/ベロシティ揺らぎ）

Phase:
11: 密度（hits_per_bar ∈ [8,32]、ghost_note_prob ≤ 0.3）
12: レンジ（MIDI 35-59、GMドラム標準範囲）
20: Humanize（タイミング±10ms、ベロシティ±8）
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

# GMドラムマップ（drums_generator_stage2.pyと共通）
GM_DRUM_MAP = {
    'kick': [35, 36],           # Bass Drum
    'snare': [38, 40],          # Snare
    'hihat_closed': [42],       # Closed Hi-Hat
    'hihat_open': [46],         # Open Hi-Hat
    'crash': [49, 57],          # Crash Cymbal
    'ride': [51, 59],           # Ride Cymbal
}

# DEFAULTSパラメータ（Base自動ロード用）
DEFAULTS = {
    "humanize_profile": "drums",
    "model": None,  # modelパスが必要な場合はここに指定
    "density": {
        "hits_per_bar": {"min": 8, "max": 32},
        "ghost_note_prob": 0.1
    },
    "register": {
        "min_midi": 35,
        "max_midi": 59
    },
    "dynamics": {
        "min_vel": 40,
        "max_vel": 120
    },
    "humanize": {
        "timing_ms": 8.0,
        "vel_sigma": 6.0
    },
    "groove": {
        "swing_ratio": 0.0,
        "laid_back_ms": 0.0
    }
}


class DrumsParamsStage2(InstrumentStage2Base):
    """
    Drums Params Stage2: 密度＋ダイナミクス＋Humanization
    
    設計:
    - NO-OP既定（設定なしなら何もしない）
    - YAML駆動プリセット（simple/moderate/complex/intense）
    - Phase単位でON/OFF可能
    - 既存DrumsGeneratorの出力を後処理
    """
    
    def __init__(
        self,
        style_presets: Optional[Dict[str, Any]] = None,
        vocab_presets: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            style_presets: drums_style_presets.yaml から読み込んだプリセット
            vocab_presets: 将来の拡張用（現在未使用）
        """
        super().__init__("drums", style_presets, vocab_presets)
    
    def _get_phases(self, params: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        実行Phase: 基本11/12/20 + Phase 13-24を追加
        
        Note: Drumsは常にPhase 13-19を有効化（デフォルト動作）
        """
        # Drumsは常に全Phase有効（後方互換性のため）
        ph = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
        # Phase 22/24/23の設定があれば追加（独立して有効化）
        adv = params or {}
        if adv.get("emotion_map"):
            if 22 not in ph:
                ph.append(22)
        if adv.get("controls"):
            if 24 not in ph:
                ph.append(24)
        if adv.get("prosody", {}).get("enable"):
            if 23 not in ph:
                ph.append(23)
        
        # Phase 25の設定があれば追加（Drumsは25のみ）
        if adv.get("sparsify", {}).get("enable"):
            if 25 not in ph:
                ph.append(25)
        
        # ソート（Phase番号順で実行）
        return sorted(ph)
    
    def _get_probability_keys(self) -> List[str]:
        """確率値として検証すべきキー"""
        return [
            "density.ghost_note_prob",
            "articulation.accent_prob",
            "articulation.flam_prob",
        ]
    
    def _get_velocity_keys(self) -> List[str]:
        """ベロシティ値として検証すべきキー"""
        return [
            "dynamics.min_vel",
            "dynamics.max_vel",
            "dynamics.accent_vel",
        ]
    
    def _validate_instrument_specific(self, params: Dict[str, Any]) -> List[str]:
        """Drums固有のバリデーション"""
        warnings = []
        
        # 密度チェック
        density = params.get("density", {})
        if density:
            hpb = density.get("hits_per_bar", {})
            if isinstance(hpb, dict):
                min_hpb = hpb.get("min", 8)
                max_hpb = hpb.get("max", 32)
                if not (4 <= min_hpb <= max_hpb <= 48):
                    warnings.append(f"hits_per_bar range [{min_hpb},{max_hpb}] unreasonable")
        
        # レンジチェック（MIDI 35-59: GMドラム標準）
        register = params.get("register", {})
        if register:
            min_midi = register.get("min_midi", 35)
            max_midi = register.get("max_midi", 59)
            if not (27 <= min_midi <= max_midi <= 87):
                warnings.append(f"MIDI range [{min_midi},{max_midi}] outside GM drum range")
        
        return warnings
    
    def _phase_11_density(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 11: 密度整形
        - hits_per_bar制限
        - ゴーストノート追加
        """
        density_cfg = params.get("density", {})
        if not density_cfg:
            return
        
        hits_per_bar = density_cfg.get("hits_per_bar", {})
        ghost_prob = density_cfg.get("ghost_note_prob", 0.0)
        
        if not hits_per_bar and ghost_prob == 0.0:
            return
        
        all_notes = list(part.flatten().notes)
        if not all_notes:
            return
        
        logger.debug(f"[Drums Phase11] Original: {len(all_notes)} hits")
        
        # 密度制限
        if hits_per_bar:
            min_hpb = hits_per_bar.get("min", 8)
            max_hpb = hits_per_bar.get("max", 32)
            
            # 小節数推定（4/4前提）
            if all_notes:
                total_duration = max(n.offset + n.quarterLength for n in all_notes)
                bars = max(1, int(total_duration / 4.0))
                target_min = min_hpb * bars
                target_max = max_hpb * bars
                
                current_count = len(all_notes)
                
                # 多すぎる場合: 間引き（ベロシティ低いものから削除）
                if current_count > target_max:
                    sorted_notes = sorted(all_notes, key=lambda n: n.volume.velocity or 64)
                    to_remove = current_count - target_max
                    for n in sorted_notes[:to_remove]:
                        part.remove(n)
                    logger.debug(f"[Drums Phase11] Removed {to_remove} low-velocity hits")
        
        # ゴーストノート追加（スネアに低ベロシティノートを追加）
        if ghost_prob > 0.0:
            snare_notes = [n for n in all_notes if hasattr(n, 'pitch') and n.pitch.midi in GM_DRUM_MAP['snare']]
            ghosts_added = 0
            
            for snare in snare_notes:
                if random.random() < ghost_prob:
                    # 前または後にゴーストノート
                    ghost_offset = snare.offset - 0.25 if random.random() < 0.5 else snare.offset + 0.25
                    if ghost_offset >= 0:
                        ghost = note.Note(GM_DRUM_MAP['snare'][0])
                        ghost.volume.velocity = 40  # ゴーストノートは弱く
                        ghost.duration.quarterLength = 0.125
                        part.insert(ghost_offset, ghost)
                        ghosts_added += 1
            
            if ghosts_added > 0:
                logger.debug(f"[Drums Phase11] Added {ghosts_added} ghost notes")
    
    def _phase_12_register(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 12: レンジ補正
        - GMドラム範囲外のMIDIノートを補正
        """
        register_cfg = params.get("register", {})
        if not register_cfg:
            return
        
        min_midi = register_cfg.get("min_midi", 35)
        max_midi = register_cfg.get("max_midi", 59)
        
        all_notes = list(part.flatten().notes)
        if not all_notes:
            return
        
        corrections = 0
        for n in all_notes:
            if not hasattr(n, 'pitch'):
                continue
            
            original_midi = n.pitch.midi
            
            # 範囲外なら補正
            if original_midi < min_midi:
                n.pitch = m21pitch.Pitch(midi=min_midi)
                corrections += 1
            elif original_midi > max_midi:
                n.pitch = m21pitch.Pitch(midi=max_midi)
                corrections += 1
        
        if corrections > 0:
            logger.debug(f"[Drums Phase12] Corrected {corrections} out-of-range notes")
    
    def _phase_13_vocabulary(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 13: Vocabulary expansion（フィル語彙）
        - セクション境界でフィルイン自動挿入
        """
        vocab_cfg = params.get("vocabulary", {})
        if not vocab_cfg:
            return
        
        insert_fills = vocab_cfg.get("insert_fills", False)
        fill_probability = vocab_cfg.get("fill_probability", 0.0)
        
        if not insert_fills or fill_probability == 0.0:
            return
        
        # セクション境界検出（section_metaから）
        section_label = section_meta.get("label", "")
        bar = section_meta.get("bar", 0)
        
        # フィル挿入判定
        if random.random() > fill_probability:
            logger.debug(f"[Drums Phase13] Skipped fill insertion (prob={fill_probability})")
            return
        
        logger.debug(f"[Drums Phase13] Attempting fill insertion...")
        
        # 簡易フィル: 最後の1小節にスネアロール追加
        all_notes = list(part.flatten().notes)
        if not all_notes:
            return
        
        # 最終小節の開始位置を推定
        total_duration = max(n.offset + n.quarterLength for n in all_notes)
        bars_total = int(total_duration / 4.0)
        last_bar_start = (bars_total - 1) * 4.0
        
        # スネアロールを8分音符で追加
        snare_midi = GM_DRUM_MAP['snare'][0]
        roll_offsets = [last_bar_start + i * 0.5 for i in range(8)]
        roll_velocities = [70, 75, 80, 85, 90, 95, 100, 105]
        
        fills_added = 0
        for offset, vel in zip(roll_offsets, roll_velocities):
            snare = note.Note(snare_midi, quarterLength=0.25)
            snare.volume.velocity = vel
            part.insert(offset, snare)
            fills_added += 1
        
        if fills_added > 0:
            logger.debug(f"[Drums Phase13] Added {fills_added} fill notes")
    
    def _phase_14_harmonic_awareness(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 14: Harmonic awareness（和声認識）
        - コード変化時にクラッシュシンバル追加
        """
        harmonic_cfg = params.get("harmonic", {})
        if not harmonic_cfg:
            return
        
        crash_on_chord_change = harmonic_cfg.get("crash_on_chord_change", False)
        crash_probability = harmonic_cfg.get("crash_probability", 0.0)
        
        if not crash_on_chord_change or crash_probability == 0.0:
            return
        
        # mix_contextからコード進行取得
        chord_changes = mix_context.get("chord_changes", [])
        if not chord_changes:
            return
        
        # コード変化位置にクラッシュ追加
        crash_midi = GM_DRUM_MAP['crash'][0]
        crashes_added = 0
        
        for chord_change in chord_changes:
            if random.random() < crash_probability:
                offset = chord_change.get("offset", 0.0)
                crash = note.Note(crash_midi, quarterLength=1.0)
                crash.volume.velocity = 110
                part.insert(offset, crash)
                crashes_added += 1
        
        if crashes_added > 0:
            logger.debug(f"[Drums Phase14] Added {crashes_added} crash cymbals on chord changes")
    
    def _phase_15_cross_instrument_sync(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 15: Cross-instrument sync（楽器間同期）
        - Bassのキックとタイミング同期強化
        """
        sync_cfg = params.get("cross_sync", {})
        if not sync_cfg:
            return
        
        sync_with_bass = sync_cfg.get("sync_with_bass", False)
        sync_strength = sync_cfg.get("sync_strength", 0.0)
        
        if not sync_with_bass or sync_strength == 0.0:
            return
        
        # mix_contextからBassのonset取得
        bass_onsets = mix_context.get("bass_onsets_ql", [])
        if not bass_onsets:
            return
        
        # Kick notesを取得
        all_notes = list(part.flatten().notes)
        kick_notes = [n for n in all_notes if hasattr(n, 'pitch') and n.pitch.midi in GM_DRUM_MAP['kick']]
        
        if not kick_notes:
            return
        
        # Bassとの同期: 近いKickを微調整
        synced_count = 0
        window = 0.125  # 32分音符程度
        
        for kick in kick_notes:
            for bass_onset in bass_onsets:
                if abs(kick.offset - bass_onset) < window:
                    # 同期強度に応じてタイミング調整
                    if random.random() < sync_strength:
                        kick.offset = bass_onset
                        synced_count += 1
                    break
        
        if synced_count > 0:
            logger.debug(f"[Drums Phase15] Synced {synced_count} kicks with bass")
    
    def _phase_16_transition_smoothing(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 16: Transition smoothing（遷移平滑化）
        - セクション境界でダイナミクス段階的変化
        """
        transition_cfg = params.get("transition", {})
        if not transition_cfg:
            return
        
        enable_crescendo = transition_cfg.get("enable_crescendo", False)
        crescendo_bars = transition_cfg.get("crescendo_bars", 1)
        
        if not enable_crescendo:
            return
        
        all_notes = list(part.flatten().notes)
        if not all_notes:
            return
        
        # 最後のN小節にクレッシェンド適用
        total_duration = max(n.offset + n.quarterLength for n in all_notes)
        bars_total = int(total_duration / 4.0)
        crescendo_start = max(0, bars_total - crescendo_bars) * 4.0
        
        crescendo_notes = [n for n in all_notes if n.offset >= crescendo_start]
        
        if not crescendo_notes:
            return
        
        # ベロシティを徐々に増加
        for i, n in enumerate(sorted(crescendo_notes, key=lambda x: x.offset)):
            if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                original_vel = n.volume.velocity or 64
                # 線形増加
                progress = i / max(1, len(crescendo_notes) - 1)
                boost = int(20 * progress)  # 最大+20
                new_vel = min(127, original_vel + boost)
                n.volume.velocity = new_vel
        
        logger.debug(f"[Drums Phase16] Applied crescendo to {len(crescendo_notes)} notes")
    
    def _phase_17_articulation_refinement(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 17: Articulation refinement（表現細分化）
        - フラム/ゴースト/アクセント自動配置
        """
        artic_cfg = params.get("articulation", {})
        if not artic_cfg:
            return
        
        flam_prob = artic_cfg.get("flam_prob", 0.0)
        accent_prob = artic_cfg.get("accent_prob", 0.0)
        
        if flam_prob == 0.0 and accent_prob == 0.0:
            return
        
        all_notes = list(part.flatten().notes)
        snare_notes = [n for n in all_notes if hasattr(n, 'pitch') and n.pitch.midi in GM_DRUM_MAP['snare']]
        
        if not snare_notes:
            return
        
        # フラム追加
        flams_added = 0
        if flam_prob > 0.0:
            for snare in snare_notes:
                if random.random() < flam_prob:
                    # 直前にゴーストノート追加（フラム効果）
                    grace_offset = snare.offset - 0.03125  # 約20ms前
                    if grace_offset >= 0:
                        grace = note.Note(snare.pitch.midi, quarterLength=0.0625)
                        grace.volume.velocity = 40
                        part.insert(grace_offset, grace)
                        flams_added += 1
        
        # アクセント追加
        accents_added = 0
        if accent_prob > 0.0:
            accent_vel = artic_cfg.get("accent_vel", 120)
            for snare in snare_notes:
                if random.random() < accent_prob:
                    snare.volume.velocity = min(127, accent_vel)
                    accents_added += 1
        
        if flams_added > 0 or accents_added > 0:
            logger.debug(f"[Drums Phase17] Added {flams_added} flams, {accents_added} accents")
    
    def _phase_18_dynamics_shaping(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 18: Dynamics shaping（ダイナミクス整形）
        - セクション別ベロシティカーブ適用
        """
        dynamics_cfg = params.get("dynamics", {})
        if not dynamics_cfg:
            return
        
        velocity_curve = dynamics_cfg.get("velocity_curve", None)
        
        if not velocity_curve:
            return
        
        all_notes = list(part.flatten().notes)
        if not all_notes:
            return
        
        # カーブ適用（"linear_up", "linear_down", "peak_middle"等）
        if velocity_curve == "linear_up":
            # 徐々に強く
            for i, n in enumerate(sorted(all_notes, key=lambda x: x.offset)):
                if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                    progress = i / max(1, len(all_notes) - 1)
                    target_vel = int(60 + 40 * progress)  # 60-100
                    n.volume.velocity = target_vel
        
        elif velocity_curve == "linear_down":
            # 徐々に弱く
            for i, n in enumerate(sorted(all_notes, key=lambda x: x.offset)):
                if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                    progress = i / max(1, len(all_notes) - 1)
                    target_vel = int(100 - 40 * progress)  # 100-60
                    n.volume.velocity = target_vel
        
        elif velocity_curve == "peak_middle":
            # 中間でピーク
            for i, n in enumerate(sorted(all_notes, key=lambda x: x.offset)):
                if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                    progress = i / max(1, len(all_notes) - 1)
                    # 放物線
                    intensity = 1.0 - 4 * (progress - 0.5) ** 2
                    target_vel = int(60 + 40 * intensity)  # 60-100
                    n.volume.velocity = target_vel
        
        logger.debug(f"[Drums Phase18] Applied velocity curve '{velocity_curve}' to {len(all_notes)} notes")
    
    def _phase_19_groove_micro_timing(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 19: Groove micro-timing（グルーヴマイクロタイミング）
        - ジャンル別スウィング適用
        """
        groove_cfg = params.get("groove", {})
        if not groove_cfg:
            return
        
        swing_ratio = groove_cfg.get("swing_ratio", 0.0)
        laid_back_ms = groove_cfg.get("laid_back_ms", 0.0)
        
        if swing_ratio == 0.0 and laid_back_ms == 0.0:
            return
        
        all_notes = list(part.flatten().notes)
        if not all_notes:
            return
        
        # スウィング適用（8分音符の裏拍を遅らせる）
        if swing_ratio > 0.0:
            for n in all_notes:
                # 8分音符グリッドの裏拍判定
                beat_position = n.offset % 1.0
                if 0.4 < beat_position < 0.6:  # 裏拍
                    # スウィング量適用
                    shift = 0.167 * swing_ratio  # 3連符化の度合い
                    n.offset += shift
        
        # レイドバック適用（全体を遅らせる）
        if laid_back_ms > 0.0:
            ms_to_quarter = 0.002  # 120 BPM前提
            laid_back_shift = laid_back_ms * ms_to_quarter
            
            # スネア/ハイハットのみ遅らせる（キックは残す）
            snare_hihat_notes = [n for n in all_notes if hasattr(n, 'pitch') and 
                                n.pitch.midi in (GM_DRUM_MAP['snare'] + GM_DRUM_MAP['hihat_closed'] + GM_DRUM_MAP['hihat_open'])]
            
            for n in snare_hihat_notes:
                n.offset += laid_back_shift
        
        logger.debug(f"[Drums Phase19] Applied groove (swing={swing_ratio}, laid_back={laid_back_ms}ms)")
    
    def _phase_20_humanize(
        self,
        part: stream.Part,
        section_meta: Dict[str, Any],
        mix_context: Dict[str, Any],
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """
        Phase 20: Humanization
        - タイミング揺らぎ（±timing_ms）
        - ベロシティ揺らぎ（±vel_sigma）
        """
        humanize_cfg = params.get("humanize", {})
        if not humanize_cfg:
            return
        
        timing_ms = humanize_cfg.get("timing_ms", 0.0)
        vel_sigma = humanize_cfg.get("vel_sigma", 0.0)
        
        if timing_ms == 0.0 and vel_sigma == 0.0:
            return
        
        all_notes = list(part.flatten().notes)
        if not all_notes:
            return
        
        # タイミング揺らぎ（ms → quarter length）
        if timing_ms > 0.0:
            # 120 BPM前提: 1拍=0.5秒=500ms → 1ms=0.002 quarter
            ms_to_quarter = 0.002
            timing_delta = timing_ms * ms_to_quarter
            
            for n in all_notes:
                shift = random.uniform(-timing_delta, timing_delta)
                new_offset = max(0.0, n.offset + shift)
                n.offset = new_offset
        
        # ベロシティ揺らぎ
        if vel_sigma > 0.0:
            for n in all_notes:
                if hasattr(n, 'volume') and hasattr(n.volume, 'velocity'):
                    original_vel = n.volume.velocity or 64
                    delta = random.gauss(0, vel_sigma)
                    new_vel = int(max(1, min(127, original_vel + delta)))
                    n.volume.velocity = new_vel
        
        logger.debug(f"[Drums Phase20] Humanized {len(all_notes)} hits (timing±{timing_ms}ms, vel±{vel_sigma})")
    
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
        self._apply_emotion_map(part, params, role="drums", ql_per_bar=ql_per_bar, bpm=tempo)
    
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
        """Phase 25: Sparsify（間引き）- レーン構造対応
        
        Drumsは端点保持せず（keep_endpoints=False既定）、
        min_gap_ms未指定時は18.0msをデフォルトとする。
        """
        sp = params.get("sparsify") or {}
        if not sp.get("enable", False):
            return
        
        tempo = section_meta.get("tempo", 120)
        
        # 未設定時にNO-OPにならないよう、必ず既定値を適用
        mg = sp.get("min_gap_ms")
        try:
            mg = float(mg) if mg is not None else 18.0  # 既定: 18ms
        except Exception:
            mg = 18.0
        
        # Drumsは端点保持不要（デフォルトFalse）
        keep_ep = bool(sp.get("keep_endpoints", False))
        
        # 1) トップレベル notes があれば通常どおり
        notes = list(part.flatten().notesAndRests.notes)
        if notes:
            try:
                self._thin_notes_even(
                    part,
                    keep_endpoints=keep_ep,
                    min_gap_ms=mg,
                    step_count=sp.get("step_count"),  # None なら gap モードで動作
                    bpm=tempo
                )
            except Exception as e:
                logger.warning(f"[Drums] Phase 25 sparsify (notes) failed: {e}")
            return
        
        # 2) レーン構造（例: lanes / kit）を安全にハンドル
        # Drumsは通常 music21.stream.Part なので、この分岐は保険
        if hasattr(part, 'recurse'):
            # music21.stream の場合は flatten() で全ノートを取得できている
            return
        
        # dict形式のレーン構造の場合（将来の拡張用）
        lanes = part.get("lanes") if isinstance(part, dict) else None
        if not lanes and isinstance(part, dict):
            lanes = part.get("kit")  # 別名の可能性
        
        if isinstance(lanes, dict) and lanes:
            for lname, lane in lanes.items():
                try:
                    self._thin_notes_even(
                        lane,
                        keep_endpoints=keep_ep,
                        min_gap_ms=mg,
                        step_count=sp.get("step_count"),
                        bpm=tempo
                    )
                except Exception:
                    continue  # 1レーン失敗しても他は続行


def load_drums_presets(
    style_yaml: Optional[Path] = None,
    vocab_yaml: Optional[Path] = None
) -> DrumsParamsStage2:
    """
    YAMLからDrums Params Stage2インスタンスを生成
    
    Args:
        style_yaml: drums_style_presets.yaml へのパス
        vocab_yaml: 将来の拡張用（現在未使用）
    
    Returns:
        DrumsParamsStage2
    """
    style_presets = None
    vocab_presets = None
    
    if style_yaml and style_yaml.exists():
        style_presets = load_yaml_presets(style_yaml)
        logger.info(f"Loaded drums style presets from {style_yaml}")
    
    if vocab_yaml and vocab_yaml.exists():
        vocab_presets = load_yaml_presets(vocab_yaml)
        logger.info(f"Loaded drums vocab from {vocab_yaml}")
    
    return DrumsParamsStage2(
        style_presets=style_presets,
        vocab_presets=vocab_presets
    )


# ============================================================================
# デモ実行
# ============================================================================
if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    print("=" * 70)
    print("  Drums Params Stage2 - Demo")
    print("=" * 70)
    
    # モックパート作成（4小節、16ビート）
    mock_part = stream.Part()
    mock_part.insert(0, note.Note(GM_DRUM_MAP['kick'][0], quarterLength=0.25))
    
    # 16ビートパターン（4小節分）
    for bar in range(4):
        offset_base = bar * 4.0
        
        # Kick: 1拍目, 3拍目
        for beat in [0.0, 2.0]:
            kick = note.Note(GM_DRUM_MAP['kick'][0], quarterLength=0.25)
            kick.volume.velocity = 100
            mock_part.insert(offset_base + beat, kick)
        
        # Snare: 2拍目, 4拍目
        for beat in [1.0, 3.0]:
            snare = note.Note(GM_DRUM_MAP['snare'][0], quarterLength=0.25)
            snare.volume.velocity = 95
            mock_part.insert(offset_base + beat, snare)
        
        # Hi-Hat: 全8分音符
        for eighth in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            hihat = note.Note(GM_DRUM_MAP['hihat_closed'][0], quarterLength=0.25)
            hihat.volume.velocity = 70
            mock_part.insert(offset_base + eighth, hihat)
    
    print(f"\n📊 Mock Part: {len(list(mock_part.flatten().notes))} hits (4 bars)")
    
    # YAMLプリセット読み込み
    preset_path = Path("data/presets/drums_style_presets.yaml")
    if not preset_path.exists():
        print(f"\n⚠️ Preset file not found: {preset_path}")
        print("Creating minimal preset for demo...")
        
        # インラインプリセット
        minimal_presets = {
            "presets": {
                "simple": {
                    "density": {
                        "hits_per_bar": {"min": 8, "max": 16},
                        "ghost_note_prob": 0.1
                    },
                    "register": {
                        "min_midi": 35,
                        "max_midi": 59
                    },
                    "humanize": {
                        "timing_ms": 8.0,
                        "vel_sigma": 6.0
                    }
                }
            }
        }
        
        drums_stage2 = DrumsParamsStage2(style_presets=minimal_presets)
    else:
        drums_stage2 = load_drums_presets(style_yaml=preset_path)
    
    # 適用テスト
    print("\n🎯 Applying 'simple' preset...")
    
    try:
        # section_metaとmix_contextを設定
        section_meta = {
            "label": "Verse",
            "bar": 0,
            "emotion": "energetic",
            "drums_style": "simple"
        }
        mix_context = {
            "sections": [],
            "vocal_phrases": []
        }
        
        result_part = drums_stage2.apply(
            part=mock_part,
            section_meta=section_meta,
            mix_context=mix_context,
            overrides={},
            seed=42
        )
        
        final_notes = list(result_part.flatten().notes)
        print(f"✅ Result: {len(final_notes)} hits after Stage2")
        
        # MIDI出力
        output_path = Path("out/demo_drums_params_stage2.mid")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_part.write('midi', fp=output_path)
        print(f"💾 Saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
