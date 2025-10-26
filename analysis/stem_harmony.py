"""
analysis/stem_harmony.py — Suno stems → mix_context / audio_chordmap / guides

本ファイルは **実コード雛形**（未設定=NO-OPを前提）です。依存は標準+本プロジェクト前提のみ：
- numpy
- pydub
- music21
- pretty_midi

⚠️ 目的：
- Phase13: 拍グリッド生成（簡易・一定テンポの安全フォールバック）
- Phase14: 活動マスク（各stemの barごとの activity∈[0..1]）
- Phase15: 各stemの拍同期コード候補（スケルトン実装：素朴に key_hint を候補化）
- Phase16: stem投票→audio_chordmap統合（安全合成・穴埋めフォールバック）
- Phase17: アクセント格子抽出（簡易：ビート起点/2&4スネア等のプレースホルダ）
- Phase18: ガイドMIDI書き出し（テンポ/マーカー/ブロックコード）

将来的にアルゴリズムを差し替えやすいよう、戻り値の**形だけは最終形**にしています。
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Iterable, Optional
from dataclasses import dataclass
import os
import math

import numpy as np
from pydub import AudioSegment
import pretty_midi
import librosa

try:
    from music21 import chord as m21chord, pitch as m21pitch, key as m21key
except Exception:  # music21 が無い環境でも import 失敗で落ちない
    m21chord = None
    m21pitch = None
    m21key = None

# ------------------------------
# 小ユーティリティ
# ------------------------------

def seconds_per_quarter(bpm: float) -> float:
    return 60.0 / max(1e-6, float(bpm))


def ql_to_ms(ql: float, bpm: float) -> float:
    return seconds_per_quarter(bpm) * ql * 1000.0


def ms_to_ql(ms: float, bpm: float) -> float:
    return (ms / 1000.0) / seconds_per_quarter(bpm)


# ------------------------------
# ビート・バーの表現
# ------------------------------
@dataclass
class BeatGrid:
    bpm: float
    time_sig: Tuple[int, int]  # (numerator, denominator)
    ql_per_bar: float          # 4/4なら4.0、6/8なら3.0 など
    beats: List[float]         # 各拍のオフセット(QL)
    bars: List[float]          # 各小節の先頭(QL)
    duration_ql: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bpm": self.bpm,
            "time_sig": list(self.time_sig),
            "ql_per_bar": self.ql_per_bar,
            "beats": self.beats,
            "bars": self.bars,
            "duration_ql": self.duration_ql,
            "sec_per_q": seconds_per_quarter(self.bpm),
        }


# ------------------------------
# 役割推定（ファイル名 → stem role）
# ------------------------------
ROLE_ALIASES = {
    "vocals": "vocals",
    "backing vocals": "backing_vocals",
    "drums": "drums",
    "percussion": "percussion",
    "bass": "bass",
    "guitar": "guitar",
    "keyboard": "piano",
    "piano": "piano",
    "strings": "strings",
    "synth": "synth",
    "fx": "fx",
}


def guess_role_from_path(path: str) -> str:
    name = os.path.basename(path).lower()
    for k, v in ROLE_ALIASES.items():
        if k in name:
            return v
    return "other"


# ------------------------------
# Phase 13: ビートグリッド
# ------------------------------

def make_beat_grid(stems: Dict[str, str], default_bpm: float = 120.0,
                    time_sig: Tuple[int, int] = (4, 4)) -> Dict[str, Any]:
    """drums優先で拍グリッドを構築（簡易フォールバック）。

    依存を増やさないため、ここでは**一定テンポ**の安全版を返します。
    後で本格推定（オンセット/テンポトラッキング）を差し替えてください。

    Returns: dict( BeatGrid 同等 )
    """
    # 優先: drums → bass → 最初のstem
    sel_path = None
    for k in ("drums", "bass"):
        for p in stems.values():
            if guess_role_from_path(p) == k:
                sel_path = p; break
        if sel_path:
            break
    if sel_path is None and stems:
        sel_path = list(stems.values())[0]

    # 長さ（秒）取得
    try:
        seg = AudioSegment.from_file(sel_path)
        dur_sec = seg.duration_seconds
    except Exception:
        # フォールバック：3分
        dur_sec = 180.0

    num, den = time_sig
    # 4分音符基準の1小節 = 4 * (4/den) * num/4 ???
    # QL換算では、1QL=4分音符。したがって 1bar(QL) = num * (4/den)
    ql_per_bar = float(num) * (4.0 / float(den))

    # 総QL
    spq = seconds_per_quarter(default_bpm)
    duration_ql = dur_sec / spq

    # 拍は 1QL ごと（4/4なら1QL=四分音）
    beats: List[float] = []
    t = 0.0
    while t <= duration_ql + 1e-6:
        beats.append(round(t, 6))
        t += 1.0  # 1QL刻み

    bars: List[float] = []
    t = 0.0
    while t <= duration_ql + 1e-6:
        bars.append(round(t, 6))
        t += ql_per_bar

    grid = BeatGrid(
        bpm=default_bpm,
        time_sig=time_sig,
        ql_per_bar=ql_per_bar,
        beats=beats,
        bars=bars,
        duration_ql=duration_ql,
    )
    return grid.to_dict()


# ------------------------------
# Phase 14: 活動マスク（barごと）
# ------------------------------

def estimate_activity(wav_path: str, beat_grid: Dict[str, Any]) -> List[Tuple[int, float]]:
    """bar単位の活動レベルを 0..1 で返す（RMSベース、簡易）。

    戻り値: [(bar_index, activity_0_1), ...]
    例外時は空リストを返し、呼び出し側が NO-OP として扱えます。
    """
    try:
        seg = AudioSegment.from_file(wav_path)
    except Exception:
        return []

    bpm = float(beat_grid.get("bpm", 120.0))
    bars = [float(x) for x in beat_grid.get("bars", [])]
    if not bars:
        return []

    vals: List[Tuple[int, float]] = []
    # バー区間ごとのRMSを測る
    for i, ql_start in enumerate(bars):
        ql_end = bars[i + 1] if i + 1 < len(bars) else float(beat_grid.get("duration_ql", ql_start + 4.0))
        ms0 = ql_to_ms(ql_start, bpm)
        ms1 = ql_to_ms(ql_end, bpm)
        try:
            chunk = seg[max(0, int(ms0)):max(0, int(ms1))]
            # dBFS は負値（小さいほど小音量）。RMSの簡易正規化に変換
            rms = chunk.rms  # 0..32767 程度
        except Exception:
            rms = 0
        vals.append((i, float(rms)))

    # 0..1へ正規化（ロバストに）
    arr = np.array([v for _, v in vals], dtype=float)
    if arr.size == 0:
        return []
    p95 = np.percentile(arr, 95) if np.any(arr > 0) else 1.0
    norm = (arr / max(1e-6, p95)).clip(0.0, 1.0)
    return [(i, float(x)) for (i, _), x in zip(vals, norm)]


# ------------------------------
# Phase 15: 拍同期コード候補（librosa実装）
# ------------------------------

# ------------------------------
# Viterbi HMM for Chord Smoothing
# ------------------------------

def _build_transition_matrix(n_states: int = 24, stay: float = 0.93, near: float = 0.03) -> np.ndarray:
    """Build HMM transition matrix for chord smoothing.
    
    Based on ChatGPT/VioPTT recommendations:
    - 24 states (12 major + 12 minor)
    - High stay probability (0.93) for temporal stability
    - Small transitions to fifth/fourth (circle of fifths)
    - Minimal probability for other transitions
    
    Args:
        n_states: Number of states (24 = 12 maj + 12 min)
        stay: Probability of staying in same chord (0.90-0.95 recommended)
        near: Probability for fifth/fourth transitions
        
    Returns:
        Transition matrix A[i,j] = P(state_j | state_i), shape [24, 24]
    """
    assert n_states == 24, "Only 24 states (maj/min) supported"
    
    # Base probability for all other transitions
    base = (1.0 - stay - 2 * near) / (n_states - 3)
    A = np.full((n_states, n_states), base, dtype=np.float32)
    
    # Major chords: 0-11
    for root in range(12):
        A[root, root] = stay  # Stay in same chord
        A[root, (root + 7) % 12] += near  # Fifth up (e.g., C -> G)
        A[root, (root + 5) % 12] += near  # Fourth up (e.g., C -> F)
    
    # Minor chords: 12-23
    for root in range(12):
        i = root + 12
        A[i, i] = stay  # Stay in same chord
        A[i, ((root + 7) % 12) + 12] += near  # Fifth up in minor
        A[i, ((root + 5) % 12) + 12] += near  # Fourth up in minor
    
    # Normalize rows to ensure valid probability distribution
    A = np.maximum(A, 1e-12)
    A = A / A.sum(axis=1, keepdims=True)
    
    return A


def _viterbi_decode(loglik: np.ndarray, trans_matrix: np.ndarray) -> np.ndarray:
    """Viterbi algorithm for finding most likely chord sequence.
    
    Args:
        loglik: Log-likelihood matrix [n_states, n_frames]
        trans_matrix: Transition matrix [n_states, n_states]
        
    Returns:
        Most likely state sequence [n_frames]
    """
    n_states, n_frames = loglik.shape
    log_trans = np.log(np.maximum(trans_matrix, 1e-12))
    
    # Dynamic programming tables
    dp = np.zeros((n_states, n_frames), dtype=np.float32)
    backpointer = np.zeros((n_states, n_frames), dtype=np.int32)
    
    # Initialize: uniform prior
    dp[:, 0] = loglik[:, 0]
    
    # Forward pass
    for t in range(1, n_frames):
        # M[i, j] = dp[i, t-1] + log P(j|i)
        M = dp[:, t-1][:, None] + log_trans
        backpointer[:, t] = np.argmax(M, axis=0)
        dp[:, t] = loglik[:, t] + M[backpointer[:, t], np.arange(n_states)]
    
    # Backward pass (trace back)
    path = np.zeros(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(dp[:, -1]))
    
    for t in range(n_frames - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]
    
    return path


def _key_profile_major() -> np.ndarray:
    """Krumhansl-Schmuckler major key profile.
    
    Based on empirical studies of tonal hierarchy in Western music.
    Higher values = stronger tonal function in major key.
    """
    # Values from Krumhansl & Kessler (1982)
    profile = np.array([
        6.35,  # C (tonic)
        2.23,  # C#
        3.48,  # D
        2.33,  # D#
        4.38,  # E
        4.09,  # F
        2.52,  # F#
        5.19,  # G (dominant)
        2.39,  # G#
        3.66,  # A
        2.29,  # A#
        2.88,  # B
    ], dtype=np.float32)
    return profile / profile.sum()


def _estimate_key(chroma_avg: np.ndarray) -> Tuple[int, str]:
    """Estimate key from average chroma vector.
    
    Returns:
        (root_index, mode) where root_index in 0-11, mode in ['maj', 'min']
    """
    profile_maj = _key_profile_major()
    
    # Try all 12 major keys
    scores_maj = np.array([
        np.sum(chroma_avg * np.roll(profile_maj, k))
        for k in range(12)
    ])
    
    # Try all 12 minor keys (using relative minor profile)
    profile_min = np.roll(profile_maj, 3)  # Simplified: rotate by minor 3rd
    scores_min = np.array([
        np.sum(chroma_avg * np.roll(profile_min, k))
        for k in range(12)
    ])
    
    best_maj_idx = int(np.argmax(scores_maj))
    best_min_idx = int(np.argmax(scores_min))
    
    if scores_maj[best_maj_idx] > scores_min[best_min_idx]:
        return best_maj_idx, 'maj'
    else:
        return best_min_idx, 'min'


# Chord templates: major/minor triads for 12 pitch classes
_CHORD_TEMPLATES = None

def _get_chord_templates():
    """Get chord templates (12 major + 12 minor = 24 templates).
    
    Based on VioPTT research: chroma features + template matching.
    Templates are 12-dim vectors representing pitch class distribution.
    """
    global _CHORD_TEMPLATES
    if _CHORD_TEMPLATES is not None:
        return _CHORD_TEMPLATES
    
    # Pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    templates = {}
    
    # Major: root, major 3rd (4 semitones), perfect 5th (7 semitones)
    for root in range(12):
        template = np.zeros(12)
        template[root] = 1.0
        template[(root + 4) % 12] = 0.7  # Major 3rd
        template[(root + 7) % 12] = 0.5  # Perfect 5th
        chord_name = f"{pitch_names[root]}:maj"
        templates[chord_name] = template / np.linalg.norm(template)
    
    # Minor: root, minor 3rd (3 semitones), perfect 5th (7 semitones)
    for root in range(12):
        template = np.zeros(12)
        template[root] = 1.0
        template[(root + 3) % 12] = 0.7  # Minor 3rd
        template[(root + 7) % 12] = 0.5  # Perfect 5th
        chord_name = f"{pitch_names[root]}:min"
        templates[chord_name] = template / np.linalg.norm(template)
    
    _CHORD_TEMPLATES = templates
    return templates


def _match_chroma_to_chord(chroma_vector, templates, top_n=3):
    """Match chroma vector to chord templates using cosine similarity.
    
    Args:
        chroma_vector: 12-dim chroma feature
        templates: Dict of chord_name -> 12-dim template
        top_n: Number of top matches to return
        
    Returns:
        List of (chord_name, score) tuples, sorted by score descending
    """
    if np.sum(chroma_vector) < 1e-6:
        # Silent frame
        return [("N", 0.0)] * top_n
    
    # Normalize input
    chroma_norm = chroma_vector / (np.linalg.norm(chroma_vector) + 1e-8)
    
    # Compute cosine similarity with all templates
    similarities = {}
    for chord_name, template in templates.items():
        sim = np.dot(chroma_norm, template)
        similarities[chord_name] = float(max(0.0, sim))  # Clip negative
    
    # Return top_n matches
    sorted_matches = sorted(similarities.items(), key=lambda x: -x[1])
    return sorted_matches[:top_n]


def estimate_chords_per_stem(
    wav_path: str,
    beat_grid: Dict[str, Any],
    role: str,
    key_hint: Optional[str] = None,
    top_n: int = 2,
    use_viterbi: bool = True,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """各拍のコード候補を返す（librosa + Viterbi HMM実装）。

    ChatGPT推奨の7段階改善手順を実装：
    1. HPSS (Harmonic-Percussive Source Separation) - ハーモニック成分のみ使用
    2. Tuning correction + CQT - チューニング補正とConstant-Q Transform
    3. Beat-synchronous chroma - ビート同期クロマ
    4. Key-conditioned templates - キー条件付きテンプレート
    5. Viterbi/HMM smoothing - HMMによる系列最適化
    6. Modulation detection - キー転調検出（将来実装）
    7. Post-processing - スパー除去等の後処理
    
    精度目標: 70-80% (template matching) → 85%+ (with Viterbi)

    Args:
        wav_path: WAVファイルパス
        beat_grid: ビート情報（bpm, beats, ql_per_bar等）
        role: 楽器役割（bass/guitar/piano/strings/other）
        key_hint: キーヒント（例: "C:maj", "A:min"）
        top_n: 各拍で返す候補数
        use_viterbi: Viterbi smoothingを使用するか（推奨=True）

    Returns:
        Dict[(bar, beat_in_bar)] -> List[{"chord": str, "score": float}]
    """
    beats = [float(x) for x in beat_grid.get("beats", [])]
    bpm = float(beat_grid.get("bpm", 120.0))
    ql_per_bar = float(beat_grid.get("ql_per_bar", 4.0))
    
    if not beats or not os.path.exists(wav_path):
        return {}
    
    # Step 1: Load audio and apply HPSS
    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        # HPSS: Separate harmonic component (より安定したコード推定のため)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
    except Exception as e:
        print(f"[WARNING] Failed to load {wav_path}: {e}")
        return {}
    
    # Step 2: Tuning correction + CQT chroma
    try:
        # チューニング補正（わずかなピッチずれを修正）
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        
        # CQT chroma (bins_per_octave=36 推奨: 高精度な周波数分解能)
        chroma = librosa.feature.chroma_cqt(
            y=y_harmonic, 
            sr=sr, 
            hop_length=512,
            bins_per_octave=36,  # ChatGPT推奨: 3 bins/semitone
            tuning=tuning
        )
    except Exception as e:
        print(f"[WARNING] Failed to extract chroma from {wav_path}: {e}")
        return {}
    
    # Step 3: Beat-synchronous chroma aggregation
    # ビート時刻の計算
    ql_per_sec = bpm / 60.0
    beat_times_sec = np.array([b / ql_per_sec for b in beats])
    
    # フレーム単位のビート位置
    beat_frames = librosa.time_to_frames(beat_times_sec, sr=sr, hop_length=512)
    
    # ビート同期クロマ（median aggregation推奨: 外れ値に強い）
    try:
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    except Exception:
        # フォールバック: 手動で平均
        chroma_sync = np.zeros((12, len(beats)))
        for i, bf in enumerate(beat_frames[:-1]):
            bf_next = beat_frames[i+1] if i+1 < len(beat_frames) else chroma.shape[1]
            chroma_sync[:, i] = np.median(chroma[:, bf:bf_next], axis=1)
    
    # Step 4: Key-conditioned template matching
    templates = _get_chord_templates()
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Estimate key if not provided
    if key_hint:
        k = str(key_hint)
        if ":" in k:
            tonic, mode = k.split(":", 1)
        else:
            tonic, mode = k, "maj"
        key_root = pitch_names.index(tonic)
        key_mode = mode
    else:
        # Global key estimation from average chroma
        chroma_avg = np.mean(chroma_sync, axis=1)
        key_root, key_mode = _estimate_key(chroma_avg)
    
    # Build key-conditioned prior
    key_profile = _key_profile_major()
    if key_mode.startswith('maj'):
        # Diatonic chords in major: I, ii, iii, IV, V, vi, vii°
        scale_degrees = [0, 2, 4, 5, 7, 9, 11]
    else:
        # Natural minor: i, ii°, III, iv, v, VI, VII
        scale_degrees = [0, 2, 3, 5, 7, 8, 10]
    
    # Build 24-state prior (12 maj + 12 min)
    prior = np.ones(24, dtype=np.float32) * 1e-3
    for degree in scale_degrees:
        root_idx = (key_root + degree) % 12
        prior[root_idx] += 1.0  # Major chord
        prior[root_idx + 12] += 0.6  # Minor chord (less likely)
    prior = prior / prior.sum()
    
    # Build log-likelihood matrix [24, n_beats]
    n_beats = chroma_sync.shape[1]
    loglik = np.zeros((24, n_beats), dtype=np.float32)
    
    # Template matching for each beat
    template_array = np.zeros((12, 24), dtype=np.float32)
    chord_names_ordered = []
    for i in range(12):
        # Major
        chord_names_ordered.append(f"{pitch_names[i]}:maj")
        template_array[:, i] = templates[f"{pitch_names[i]}:maj"]
        # Minor
        chord_names_ordered.append(f"{pitch_names[i]}:min")
        template_array[:, i + 12] = templates[f"{pitch_names[i]}:min"]
    
    # Vectorized cosine similarity
    # Normalize chroma_sync
    chroma_norm = chroma_sync / (np.linalg.norm(chroma_sync, axis=0, keepdims=True) + 1e-8)
    # Normalize templates
    template_norm = template_array / (np.linalg.norm(template_array, axis=0, keepdims=True) + 1e-8)
    # Compute similarity [n_beats, 24]
    similarity = chroma_norm.T @ template_norm  # [n_beats, 24]
    similarity = np.maximum(similarity, 1e-9)
    
    # Log-likelihood with key prior
    loglik = np.log(similarity.T) + 0.15 * np.log(prior[:, None])  # [24, n_beats]
    
    # Role-based weight adjustment
    role = (role or "other").lower()
    role_weights = {
        "bass": 1.3,
        "guitar": 1.0,
        "piano": 1.0,
        "strings": 0.9,
        "other": 0.8,
    }.get(role, 0.8)
    
    # Step 5: Viterbi smoothing (if enabled)
    if use_viterbi and n_beats > 1:
        trans_matrix = _build_transition_matrix(n_states=24, stay=0.93, near=0.03)
        path = _viterbi_decode(loglik, trans_matrix)
        
        # Convert path to votes
        votes: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for bi, beat_ql in enumerate(beats):
            bar = int(beat_ql // ql_per_bar)
            beat_in_bar = int(beat_ql - bar * ql_per_bar) + 1
            
            if bi < len(path):
                state = int(path[bi])
                chord = chord_names_ordered[state]
                score = float(np.exp(loglik[state, bi])) * role_weights
                
                # Get top-2 alternatives
                alternatives = []
                for alt_state in np.argsort(-loglik[:, bi])[:top_n]:
                    if alt_state != state:
                        alt_chord = chord_names_ordered[alt_state]
                        alt_score = float(np.exp(loglik[alt_state, bi])) * role_weights
                        alternatives.append({"chord": alt_chord, "score": alt_score})
                
                candidates = [{"chord": chord, "score": score}] + alternatives[:top_n-1]
                votes[(bar, beat_in_bar)] = candidates
    else:
        # Fallback: greedy decoding (no smoothing)
        votes: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for bi, beat_ql in enumerate(beats):
            bar = int(beat_ql // ql_per_bar)
            beat_in_bar = int(beat_ql - bar * ql_per_bar) + 1
            
            # Get top-n chords
            top_indices = np.argsort(-loglik[:, bi])[:top_n]
            candidates = []
            for idx in top_indices:
                chord = chord_names_ordered[idx]
                score = float(np.exp(loglik[idx, bi])) * role_weights
                candidates.append({"chord": chord, "score": score})
            
            votes[(bar, beat_in_bar)] = candidates
    
    return votes


def estimate_chords_per_stem_dummy(
    wav_path: str,
    beat_grid: Dict[str, Any],
    role: str,
    key_hint: Optional[str] = None,
    top_n: int = 2,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """各拍のコード候補を返す（スケルトン・ダミー実装）。

    依存追加なしで動く最小実装として、ここでは
    - key_hint があれば、その I / V / IV を候補
    - なければ C / G / F を候補
    の形で返却します（スコアは役割に応じて微調整）。

    将来的に：拍同期クロマ + HMM/Viterbi へ差し替え予定。
    
    NOTE: このダミー実装は後方互換性のために残しています。
    本番ではestimate_chords_per_stem()を使用してください。
    """
    beats = [float(x) for x in beat_grid.get("beats", [])]
    bpm = float(beat_grid.get("bpm", 120.0))
    ql_per_bar = float(beat_grid.get("ql_per_bar", 4.0))

    # 簡易ダイアトニック候補
    if key_hint is None:
        tonic = "C"
        mode = "maj"
    else:
        # 例: "C" or "C:maj" or "A:min"
        k = str(key_hint)
        if ":" in k:
            tonic, mode = k.split(":", 1)
        else:
            tonic, mode = k, "maj"
    I = f"{tonic}:{'min' if mode.startswith('min') else 'maj'}"
    V = f"{tonic if tonic=='C' else 'G'}:{'maj'}"  # 超簡易
    IV = f"{tonic if tonic=='C' else 'F'}:{'maj'}"
    pool = [I, V, IV]

    # 役割別スコアの微差（bassはIを好む等）
    role = (role or "other").lower()
    bias = {
        "bass": {I: 0.70, V: 0.20, IV: 0.10},
        "guitar": {I: 0.45, V: 0.30, IV: 0.25},
        "piano": {I: 0.45, V: 0.30, IV: 0.25},
        "strings": {I: 0.50, V: 0.25, IV: 0.25},
        "other": {I: 0.50, V: 0.30, IV: 0.20},
    }.get(role, {I: 0.5, V: 0.3, IV: 0.2})

    votes: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    # 各拍に候補を入れる
    for bi, off in enumerate(beats):
        bar = int(off // ql_per_bar)
        beat_in_bar = int(off - bar * ql_per_bar) + 1  # 1-based
        cand = sorted(
            [{"chord": c, "score": float(bias.get(c, 0.1))} for c in pool],
            key=lambda d: -d["score"],
        )[: max(1, top_n)]
        votes[(bar, beat_in_bar)] = cand
    return votes


# ------------------------------
# Phase 16: stem投票の集約 → audio_chordmap
# ------------------------------

def aggregate_stem_chords(
    stem_votes: Dict[str, Dict[Tuple[int, int], List[Dict[str, Any]]]],
    activity: Dict[str, List[Tuple[int, float]]],
    key_hint: Optional[str],
    sections: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """stemごとのコード票を、活動マスク×重みで集約し、
    スムージング（最小持続長=1拍）後に audio_chordmap を返す。

    VioPTT研究のvoting systemを参考に、複数ステムの信頼度重み付け投票を実施。
    低信頼度コード検出時は警告を出力し、手動確認を促す。
    
    Args:
        stem_votes: 各ステムの拍ごとのコード候補
        activity: 各ステムの小節ごとの活動レベル
        key_hint: キーヒント
        sections: セクション情報（未使用）
        cfg: 設定（weights, confidence_threshold等）
        
    Returns:
        audio_chordmap: {"key", "confidence_key", "items", "low_confidence_warnings"}
    """
    weights = cfg.get("weights", {"bass": 0.35, "guitar": 0.35, "piano": 0.2, "strings": 0.1})
    confidence_threshold = cfg.get("confidence_threshold", 0.3)  # Default 0.3
    min_confidence_warn = cfg.get("min_confidence_warn", 0.4)  # Warn if < 0.4

    # 活動レベルを dict[stem][bar] -> 0..1 へ
    act_map: Dict[str, Dict[int, float]] = {}
    for stem, arr in (activity or {}).items():
        act_map[stem] = {int(b): float(v) for b, v in (arr or [])}

    # 拍キーのユニオン
    all_keys: List[Tuple[int, int]] = []
    for d in (stem_votes or {}).values():
        for k in d.keys():
            if k not in all_keys:
                all_keys.append(k)
    all_keys.sort()

    out_items: List[Dict[str, Any]] = []
    low_confidence_warnings = []
    prev_chord = None

    for (bar, beat) in all_keys:
        tally: Dict[str, float] = {}
        stem_contributions = {}  # For debugging
        
        for stem, votes in (stem_votes or {}).items():
            stem_role = stem
            # activity重み
            a = act_map.get(stem, {}).get(bar, 1.0)
            w = float(weights.get(stem_role, 0.1)) * float(a)
            for cand in votes.get((bar, beat), [])[:3]:
                c = str(cand.get("chord", "C:maj"))
                s = float(cand.get("score", 0.0))
                weighted_score = w * s
                tally[c] = tally.get(c, 0.0) + weighted_score
                
                # Track contribution per stem
                if c not in stem_contributions:
                    stem_contributions[c] = []
                stem_contributions[c].append(f"{stem}:{weighted_score:.3f}")
        
        if not tally:
            # 穴埋め：前回 or key_hintのI
            if prev_chord is not None:
                chord = prev_chord
            else:
                tonic = (str(key_hint).split(":")[0] if key_hint else "C")
                chord = f"{tonic}:maj"
            conf = 0.5
            low_confidence_warnings.append({
                "bar": bar,
                "beat": beat,
                "reason": "No stem votes available (fallback to previous or key)",
                "confidence": conf,
                "chord": chord,
            })
        else:
            # Normalize confidence (0..1 range)
            total_weight = sum(tally.values())
            chord, raw_conf = max(tally.items(), key=lambda kv: kv[1])
            conf = raw_conf / (total_weight + 1e-8) if total_weight > 0 else 0.0
            
            # Check for low confidence
            if conf < min_confidence_warn:
                contributions_str = ", ".join(stem_contributions.get(chord, []))
                low_confidence_warnings.append({
                    "bar": bar,
                    "beat": beat,
                    "chord": chord,
                    "confidence": float(conf),
                    "reason": f"Low confidence (< {min_confidence_warn})",
                    "stem_contributions": contributions_str,
                    "alternatives": [
                        {"chord": c, "score": float(s / (total_weight + 1e-8))} 
                        for c, s in sorted(tally.items(), key=lambda kv: -kv[1])[:3]
                    ]
                })
            
            # Apply minimum confidence threshold (filter out very unreliable chords)
            if conf < confidence_threshold:
                if prev_chord is not None:
                    chord = prev_chord
                    conf = 0.5
                    low_confidence_warnings.append({
                        "bar": bar,
                        "beat": beat,
                        "reason": f"Confidence {conf:.3f} below threshold {confidence_threshold}, using previous chord",
                        "confidence": conf,
                        "chord": chord,
                    })
        
        prev_chord = chord
        out_items.append({"bar": bar, "beat": beat, "chord": chord, "confidence": float(conf)})

    # Print warnings if any
    if low_confidence_warnings:
        print(f"\n[WARNING] {len(low_confidence_warnings)} low-confidence chords detected:")
        for w in low_confidence_warnings[:10]:  # Show first 10
            print(f"  Bar {w['bar']}, Beat {w['beat']}: {w['chord']} (conf={w.get('confidence', 0):.3f}) - {w['reason']}")
        if len(low_confidence_warnings) > 10:
            print(f"  ... and {len(low_confidence_warnings) - 10} more warnings")
        print("  Recommendation: Manual review suggested for sections with confidence < 0.4\n")

    return {
        "key": key_hint or "C",
        "confidence_key": 0.5 if key_hint is None else 0.8,
        "items": out_items,
        "low_confidence_warnings": low_confidence_warnings,  # For programmatic access
    }


# ------------------------------
# Phase 17: アクセント格子（簡易）
# ------------------------------

def extract_accent_grid(stems: Dict[str, str], beat_grid: Dict[str, Any]) -> Dict[str, List[Any]]:
    """kick/snare/hihat 等の拍位置（QL）を抽出するスケルトン。

    依存追加なしの簡易版：
    - kick: 各小節の 1拍目
    - snare: 2拍目 & 4拍目（4/4想定）
    - hihat: 全拍（1QLごと）
    将来版で、オンセット検出に差し替えてください。
    """
    ql_per_bar = float(beat_grid.get("ql_per_bar", 4.0))
    beats = [float(x) for x in beat_grid.get("beats", [])]

    kick = []
    snare = []
    hihat = []

    for off in beats:
        bar = math.floor(off / ql_per_bar)
        beat_in_bar = int(off - bar * ql_per_bar) + 1
        if beat_in_bar == 1:
            kick.append(off)
        if beat_in_bar in (2, 4):
            snare.append(off)
        hihat.append(off)

    return {"kick": kick, "snare": snare, "hihat": hihat, "strum_ud": []}


# ------------------------------
# Phase 18: ガイドMIDI書き出し
# ------------------------------

def _parse_chord_root(ch: str) -> int:
    """music21 があればそれを優先。無ければ単純根音推定。戻り：MIDIノート番号（C4=60基準のルート、ここではC3=48に配置）。"""
    try:
        if m21chord is not None and m21pitch is not None:
            root_name = ch.split(":")[0]
            p = m21pitch.Pitch(root_name)
            return int(p.midi)
    except Exception:
        pass
    # フォールバック：A-Gの頭文字だけで判断
    name = ch.split(":")[0].upper()
    order = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
             "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
    semis = order.get(name, 0)
    return 48 + semis  # C3=48


def export_guides_to_midi(out_path: str, beat_grid: Dict[str, Any],
                           sections: List[Dict[str, Any]], audio_chordmap: Dict[str, Any]) -> None:
    """テンポ・セクション・ブロックコード（全音符）・低Velルート を出力。
    本番レンダは各Generatorが行うため、ここは耳チェックのガイド用途。
    例外は握りつぶして関数を無音で終える（CI安全）。
    """
    try:
        bpm = float(beat_grid.get("bpm", 120.0))
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        # マーカー
        for s in (sections or []):
            try:
                bar = int(s.get("bar", 0))
                name = str(s.get("label", "")).upper()
                off_ql = float(beat_grid.get("ql_per_bar", 4.0)) * bar
                pm.markers.append(pretty_midi.Marker(name=name, time=ql_to_ms(off_ql, bpm)/1000.0))
            except Exception:
                continue
        # ブロックコード
        inst = pretty_midi.Instrument(program=0, name="Guide Chords")
        items = (audio_chordmap or {}).get("items", [])
        ql_per_bar = float(beat_grid.get("ql_per_bar", 4.0))
        for it in items:
            try:
                bar = int(it["bar"]) ; beat = int(it["beat"]) ; ch = str(it["chord"]) ;
                off_ql = bar * ql_per_bar + (beat - 1) * 1.0  # 1QL=四分音
                dur_ql = 1.0 * ql_per_bar  # 1小節保持（簡易）
                start = ql_to_ms(off_ql, bpm) / 1000.0
                end   = ql_to_ms(off_ql + dur_ql, bpm) / 1000.0
                # triad (root, third, fifth)
                root_midi = _parse_chord_root(ch)
                quality = (ch.split(":")[1] if ":" in ch else "maj").lower()
                third = 3 if quality.startswith("min") else 4
                notes = [root_midi, root_midi + third, root_midi + 7]
                for n in notes:
                    inst.notes.append(pretty_midi.Note(velocity=40, pitch=int(n), start=start, end=end))
                # 低Velルート（1オクターブ下）
                inst.notes.append(pretty_midi.Note(velocity=25, pitch=int(root_midi - 12), start=start, end=end))
            except Exception:
                continue
        pm.instruments.append(inst)
        pm.write(out_path)
    except Exception:
        return


# ------------- end of file -------------
