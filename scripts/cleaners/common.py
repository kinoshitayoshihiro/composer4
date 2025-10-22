#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共通MIDIクリーニング処理
全楽器に適用される基本的なクリーニングと正規化
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pretty_midi


# =============================================================================
# Schema/Provenance (決定論・監査性)
# =============================================================================

# クリーニング系のスキーマバージョン（外部ベンチ1.1とは分離）
SCHEMA_VERSION = "1.0"


def make_provenance() -> Dict[str, Any]:
    """クリーニングパイプラインの来歴情報を最小限で記録。"""
    return {
        "tool": "cleaning-pipeline",
        "schema_version": SCHEMA_VERSION,
        "git_commit": os.getenv("GIT_COMMIT", ""),
        "git_branch": os.getenv("GIT_BRANCH", ""),
    }


def compute_fileset_hash(paths: Iterable[Path]) -> str:
    """
    対象ファイル集合の決定論ハッシュ（パス＋サイズ）。
    重いので内容ハッシュは省略。
    """
    h = hashlib.sha1()
    for p in paths:
        ap = p.as_posix().encode()
        h.update(ap)
        try:
            st = p.stat()
            h.update(str(st.st_size).encode())
        except FileNotFoundError:
            # クリーニング中の一時的な消滅があっても安定動作
            h.update(b"0")
    return h.hexdigest()[:12]


# =============================================================================
# Determinism (決定論的ファイル列挙・乱数)
# =============================================================================

def stable_list_midis(root: str | Path) -> List[Path]:
    """
    ファイルシステム順に依存しない安定列挙。
    大規模データセット(LAMDA等)向けに find コマンドで高速化。
    """
    import subprocess
    
    root = Path(root)
    
    # find コマンドで高速検索（rglob より 10-100倍速い）
    print(f"🔍 Enumerating MIDI files in {root} (this may take a few minutes)...", flush=True)
    try:
        # -name は複数回指定可能
        result = subprocess.run(
            ["find", str(root), "-type", "f", "-name", "*.mid", "-o", "-name", "*.midi"],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,  # 10分でタイムアウト
        )
        files = [Path(line.strip()) for line in result.stdout.split("\n") if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        # fallback to rglob (slower but compatible)
        print(f"   ⚠️  find command failed ({e}), falling back to rglob...", flush=True)
        files = list(root.rglob("*.mid")) + list(root.rglob("*.midi"))
    
    files.sort(key=lambda p: p.as_posix())
    print(f"✅ Found {len(files)} MIDI files", flush=True)
    return files


def seeded_rng(seed: int | str) -> random.Random:
    """文字列/数値 seed を 32bit に折り畳んだ決定論 RNG。"""
    h = int(hashlib.sha1(str(seed).encode()).hexdigest(), 16) & 0xFFFFFFFF
    return random.Random(h)


# =============================================================================
# Atomic IO (原子的JSON書き込み)
# =============================================================================

def atomic_write_json(obj: Dict, path: Path) -> None:
    """JSON を一時ファイル経由で原子的に保存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, suffix=".tmp", encoding="utf-8"
    ) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp = Path(f.name)
    os.replace(tmp, path)


# =============================================================================
# Common Cleaning Functions
# =============================================================================


def common_clean(
    pm: pretty_midi.PrettyMIDI,
) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any]]:
    """
    全楽器共通のクリーニング処理
    
    Args:
        pm: PrettyMIDI オブジェクト
        
    Returns:
        (cleaned_pm, metadata) のタプル
        metadata には以下が含まれる:
        - clean_actions: 実施した修正のリスト
        - reason_codes: 警告/エラーコード
        - tempo_estimated: テンポ推定したか
        - time_signature: 拍子
        - bars: 小節数
        - notes: 総ノート数
        - density: notes/sec
        - duration_sec: 全体の長さ
    """
    metadata: Dict[str, Any] = {
        "clean_actions": [],
        "reason_codes": [],
    }
    
    # 1. 無効イベントの除去
    pm, invalid_notes = _remove_invalid_notes(pm)
    if invalid_notes > 0:
        metadata["clean_actions"].append(f"remove_invalid_notes:{invalid_notes}")
    
    # 2. テンポ/拍子の正規化
    pm, tempo_meta = _normalize_tempo_timesig(pm)
    metadata.update(tempo_meta)
    
    # 3. 範囲外Pitchの検出
    pitch_warnings = _check_pitch_outliers(pm)
    if pitch_warnings:
        metadata["reason_codes"].extend(pitch_warnings)
    
    # 4. 長さ/密度チェック
    duration_sec = pm.get_end_time()
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    
    metadata["duration_sec"] = round(duration_sec, 2)
    metadata["notes"] = total_notes
    metadata["density"] = round(total_notes / duration_sec, 2) if duration_sec > 0 else 0
    
    # 推定小節数
    tempo = pm.get_tempo_changes()[1][0] if len(pm.get_tempo_changes()[1]) > 0 else 120
    time_sig = pm.time_signature_changes[0] if pm.time_signature_changes else None
    numerator = time_sig.numerator if time_sig else 4
    
    beats_per_minute = tempo
    beats_per_bar = numerator
    bars = (duration_sec / 60) * beats_per_minute / beats_per_bar
    metadata["bars"] = round(bars, 1)
    
    # 5. 最低限チェック
    if bars < 1:
        metadata["reason_codes"].append("too_short")
    if total_notes < 8:  # 最小ノート数
        metadata["reason_codes"].append("too_few_notes")
    
    # 6. 楽器割り当てチェック
    for inst in pm.instruments:
        if inst.program == 0 and inst.is_drum:
            # Drumなのにprogram=0は矛盾
            metadata["reason_codes"].append("drum_program_mismatch")
    
    return pm, metadata


def _remove_invalid_notes(pm: pretty_midi.PrettyMIDI) -> Tuple[pretty_midi.PrettyMIDI, int]:
    """無効ノートを除去"""
    removed_count = 0
    
    for inst in pm.instruments:
        valid_notes = []
        for note in inst.notes:
            # 負/ゼロ長
            if note.end <= note.start:
                removed_count += 1
                continue
            # velocity=0
            if note.velocity == 0:
                removed_count += 1
                continue
            # pitch範囲外
            if note.pitch < 0 or note.pitch > 127:
                removed_count += 1
                continue
            
            valid_notes.append(note)
        
        inst.notes = valid_notes
    
    return pm, removed_count


def _normalize_tempo_timesig(pm: pretty_midi.PrettyMIDI) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any]]:
    """テンポ/拍子を正規化"""
    meta: Dict[str, Any] = {}
    
    # テンポチェック
    tempo_changes = pm.get_tempo_changes()
    if len(tempo_changes[0]) == 0:
        # テンポ未設定 → 120BPMに設定
        pm._tick_scales = [(0, 60.0 / (120.0 * pm.resolution))]
        meta["tempo_estimated"] = True
        meta["tempo"] = 120.0
    elif len(tempo_changes[0]) > 100:
        # 異常な大量テンポ変更
        meta["reason_codes"] = ["tempo_change_excess"]
        meta["tempo"] = statistics.median(tempo_changes[1])
    else:
        meta["tempo_estimated"] = False
        meta["tempo"] = tempo_changes[1][0]
    
    # 拍子チェック
    if not pm.time_signature_changes:
        # 4/4を設定
        pm.time_signature_changes.append(
            pretty_midi.TimeSignature(4, 4, 0)
        )
        meta["time_signature"] = "4/4"
        meta["time_signature_estimated"] = True
    else:
        ts = pm.time_signature_changes[0]
        meta["time_signature"] = f"{ts.numerator}/{ts.denominator}"
        meta["time_signature_estimated"] = False
    
    return pm, meta


def _check_pitch_outliers(pm: pretty_midi.PrettyMIDI) -> List[str]:
    """範囲外Pitchを検出"""
    warnings = []
    
    for inst in pm.instruments:
        if inst.is_drum:
            continue  # ドラムは除外
        
        pitches = [note.pitch for note in inst.notes]
        if not pitches:
            continue
        
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        pitch_range = max_pitch - min_pitch
        
        # GM範囲外チェック (21=A0, 108=C8)
        if min_pitch < 21 or max_pitch > 108:
            warnings.append("pitch_outlier")
        
        # 極端に広い音域 (>7オクターブ)
        if pitch_range > 84:
            warnings.append("pitch_range_excessive")
    
    return warnings


def _deduplicate_overlapping_notes(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """
    同一pitchの重複ノートを処理
    
    Note: レガート許容か分割かは楽器別ポリシーで扱うため、
    ここでは重複を検出してメタデータに記録するのみ
    """
    for inst in pm.instruments:
        # pitch別にグループ化
        pitch_groups: Dict[int, List[pretty_midi.Note]] = defaultdict(list)
        for note in inst.notes:
            pitch_groups[note.pitch].append(note)
        
        # 各pitch内で時系列ソート
        for pitch, notes in pitch_groups.items():
            notes.sort(key=lambda n: n.start)
            
            # オーバーラップ検出
            for i in range(len(notes) - 1):
                if notes[i].end > notes[i + 1].start:
                    # 重複あり → メタデータに記録 (修正は楽器別)
                    pass
    
    return pm


# =============================================================================
# Shard Writer (ストリーミング式pickle生成)
# =============================================================================

class ShardWriter:
    """
    ストリーミング式シャードPickle生成
    
    - バッファが閾値に達したら即座にflush
    - 原子的書き込み (.tmp → rename)
    - レジューム対応（既存シャード検出）
    """
    
    def __init__(self, out_dir: Path, instrument: str, shard_size: int = 5000, resume: bool = False, subfolder_id: str = ""):
        """
        Args:
            out_dir: 出力ディレクトリ
            instrument: 楽器名
            shard_size: シャードあたりの件数
            resume: 既存シャードから再開するか
            subfolder_id: サブフォルダID（例: '0', 'a', 'f'）。指定時は単一pickleファイルを生成
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.instrument = instrument
        self.shard_size = shard_size
        self.buffer: List[Dict[str, Any]] = []
        self.shard_paths: List[Path] = []
        self.subfolder_id = subfolder_id  # サブフォルダモード用
        
        # サブフォルダモードの場合
        if subfolder_id:
            # 既存ファイルがあればスキップ扱い
            expected_pickle = self.out_dir / f"{instrument}_shard_{subfolder_id}.pickle"
            if resume and expected_pickle.exists():
                print(f"📂 Resume mode: Pickle already exists: {expected_pickle}")
                print(f"   This subfolder will be skipped.")
                self.shard_idx = -1  # スキップフラグ
                return
            else:
                self.shard_idx = 0
                return
        
        # 通常モード: 既存シャードを検出して再開
        existing = sorted(self.out_dir.glob(f"{instrument}_*.pkl"))
        # インデックスファイルを除外
        existing = [p for p in existing if "_index.pkl" not in p.name]
        if resume and existing:
            # 最後のシャード番号 + 1 から再開
            last_shard = existing[-1]
            # {instrument}_{idx:05d}.pkl 形式から番号を抽出
            idx_str = last_shard.stem.replace(instrument + "_", "")
            self.shard_idx = int(idx_str) + 1
            self.shard_paths = existing
            print(f"📂 Resume mode: Found {len(existing)} existing shards, starting from shard {self.shard_idx:05d}")
        else:
            self.shard_idx = 0
    
    def add(self, lamda_meta: Dict[str, Any]) -> None:
        """
        LAMDA互換メタデータをバッファに追加
        閾値に達したら自動的にflush
        """
        # スキップモード（既存pickleがある場合）
        if self.shard_idx == -1:
            return
        
        self.buffer.append(lamda_meta)
        
        if len(self.buffer) >= self.shard_size:
            self.flush()
    
    def flush(self) -> None:
        """
        バッファを原子的にpickleファイルに書き込み
        """
        if not self.buffer or self.shard_idx == -1:
            return
        
        # サブフォルダモードの場合
        if self.subfolder_id:
            shard_name = f"{self.instrument}_shard_{self.subfolder_id}.pickle"
        else:
            # Stage2互換の命名規則: {instrument}_{idx:05d}.pkl
            shard_name = f"{self.instrument}_{self.shard_idx:05d}.pkl"
        
        tmp_path = self.out_dir / (shard_name + ".tmp")
        final_path = self.out_dir / shard_name
        
        # Shardデータ構造（Stage2互換）
        shard_data = {
            "version": "lamda_v2_shard",
            "shard_index": self.shard_idx,
            "instrument": self.instrument,
            "loops": self.buffer,
            "count": len(self.buffer),
            "summary": {
                "total_notes": sum(m.get("note_count", 0) for m in self.buffer),
                "avg_bpm": sum(m.get("bpm", 0) for m in self.buffer) / len(self.buffer) if self.buffer else 0,
            }
        }
        
        # 原子的書き込み
        with open(tmp_path, "wb") as f:
            pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())  # ディスクに確実に書き込み
        
        os.replace(tmp_path, final_path)  # アトミックリネーム
        
        self.shard_paths.append(final_path)
        print(f"  ✓ Shard {self.shard_idx:05d}: {len(self.buffer):,} loops → {shard_name}")
        
        # バッファクリア
        self.buffer.clear()
        self.shard_idx += 1
    
    def write_index(self) -> Path:
        """
        全シャードの情報を集約したインデックスpickleを生成（Stage2互換）
        
        Returns:
            インデックスファイルのパス
        """
        # 残りをflush
        self.flush()
        
        if not self.shard_paths:
            raise ValueError("No shards to index")
        
        # 各シャードから情報を収集
        shard_info = []
        total_loops = 0
        
        for shard_path in sorted(self.shard_paths):
            with open(shard_path, "rb") as f:
                shard_data = pickle.load(f)
            
            # Stage2が期待する構造
            shard_info.append({
                "path": shard_path.name,  # 相対パス（ファイル名のみ）
                "index": shard_data.get("shard_index", 0),  # ★ Stage2が必要とするフィールド
                "count": shard_data.get("count", 0),
                "summary": shard_data.get("summary", {}),
                "metrics_summary": shard_data.get("summary", {}),  # ★ 互換性のため両方用意
            })
            total_loops += shard_data.get("count", 0)
        
        # インデックスデータ（Stage2互換）
        index_data = {
            "version": "lamda_v2_index",
            "instrument": self.instrument,
            "total_files": total_loops,
            "shard_size": self.shard_size,
            "base_dir": str(self.out_dir),
            "shards": shard_info,  # ★ Stage2が iter_loop_records() で読むフィールド
        }
        
        index_path = self.out_dir / f"{self.instrument}_index.pkl"
        tmp_path = self.out_dir / f"{self.instrument}_index.pkl.tmp"
        
        # 原子的書き込み
        with open(tmp_path, "wb") as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        
        os.replace(tmp_path, index_path)
        
        return index_path


# =============================================================================
# LAMDA互換メタデータ抽出
# =============================================================================

def extract_lamda_metadata(
    pm: pretty_midi.PrettyMIDI,
    input_path: Path,
    output_path: Path,
    base_dir: Path | None = None,
    genre: str | None = None,
) -> Dict[str, Any]:
    """
    LAMDA互換の詳細メタデータを抽出
    
    Args:
        pm: PrettyMIDI オブジェクト
        input_path: 元ファイルパス（MD5計算用）
        output_path: 出力先パス
        base_dir: 相対パス計算の基準ディレクトリ（Noneの場合は絶対パス）
        genre: ジャンル情報（Noneの場合は "unknown"）
        
    Returns:
        LAMDA形式のメタデータ辞書（md5はMIDIバイトから32桁、output_pathは相対/絶対）
    """
    # ファイル名を抽出
    filename = input_path.stem
    # 全ノート収集
    all_notes = []
    for inst in pm.instruments:
        for note in inst.notes:
            all_notes.append({
                "pitch": note.pitch,
                "velocity": note.velocity,
                "start": note.start,
                "end": note.end,
                "duration": note.end - note.start,
                "is_drum": inst.is_drum,
                "program": inst.program,
            })
    
    if not all_notes:
        return {}
    
    # ピッチ統計（JSON互換のためint()でキャスト）
    pitches = [int(n["pitch"]) for n in all_notes]
    pitches_sum = sum(pitches)
    pitches_counts = {int(p): int(c) for p, c in Counter(pitches).most_common()}
    pitches_and_counts = [[int(p), int(c)] for p, c in pitches_counts.items()]
    
    # ベロシティ統計
    velocities = [int(n["velocity"]) for n in all_notes]
    avg_velocity = sum(velocities) / len(velocities)
    
    # プログラム（パッチ）統計（JSON互換のためint()でキャスト）
    patches = [int(n["program"]) for n in all_notes]
    patches_counts = {int(p): int(c) for p, c in Counter(patches).most_common()}
    
    # タイミング統計（ミリ秒）
    times_ms = []
    all_notes_sorted = sorted(all_notes, key=lambda x: x["start"])
    prev_time = 0
    for note in all_notes_sorted:
        delta_ms = (note["start"] - prev_time) * 1000
        if delta_ms > 0 or prev_time == 0:
            times_ms.append(int(delta_ms))
        prev_time = note["start"]
    
    times_sum = min(10000000, sum(times_ms))
    
    # デュレーション統計（ミリ秒）
    durations_ms = [int(n["duration"] * 1000) for n in all_notes]
    
    # 統計値
    avg_time = int(sum(times_ms) / len(times_ms)) if times_ms else 0
    avg_dur = int(sum(durations_ms) / len(durations_ms)) if durations_ms else 0
    avg_vel = int(avg_velocity)
    
    mode_time = statistics.mode(times_ms) if times_ms else 0
    mode_dur = statistics.mode(durations_ms) if durations_ms else 0
    mode_vel = statistics.mode(velocities) if velocities else 0
    
    median_time = int(statistics.median(times_ms)) if times_ms else 0
    median_dur = int(statistics.median(durations_ms)) if durations_ms else 0
    median_vel = int(statistics.median(velocities)) if velocities else 0
    
    # コード抽出（同時発音ピッチ）
    chords = []
    time_groups = defaultdict(list)
    for note in all_notes:
        if not note["is_drum"]:
            time_key = int(note["start"] * 1000)  # 1ms精度
            time_groups[time_key].append(note["pitch"] % 12)
    
    for time_key in sorted(time_groups.keys()):
        chord = sorted(set(time_groups[time_key]))
        if len(chord) > 1:
            chords.append(chord)
    
    ms_chords_counts = sorted(
        [[list(key), val] for key, val in Counter([tuple(c) for c in chords]).most_common()],
        reverse=True,
        key=lambda x: x[1]
    )
    if not ms_chords_counts:
        ms_chords_counts = [[[0, 0], 0]]
    
    # テンポ情報
    tempo_changes = pm.get_tempo_changes()
    tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
    
    # 拍子情報
    time_sig = pm.time_signature_changes[0] if pm.time_signature_changes else None
    time_signature = f"{time_sig.numerator}/{time_sig.denominator}" if time_sig else "4/4"
    
    # MD5: MIDIバイトから計算（32桁）
    # pm.write() は None を受け付けないため、BytesIO を使用
    from io import BytesIO
    midi_buffer = BytesIO()
    pm.write(midi_buffer)
    midi_bytes = midi_buffer.getvalue()
    md5_full = hashlib.md5(midi_bytes).hexdigest()  # 32桁
    
    # output_path: base_dirからの相対パスまたは絶対パス
    if base_dir is not None:
        try:
            rel_output = output_path.relative_to(base_dir)
            final_output_path = str(rel_output)
        except ValueError:
            final_output_path = str(output_path)
    else:
        final_output_path = str(output_path)
    
    # LAMDA形式メタデータ（Stage2互換）
    metadata = {
        "filename": filename,
        "genre": genre if genre is not None else "unknown",  # ★ Stage2が必要とするフィールド
        "input_path": str(input_path),
        "output_path": final_output_path,
        "cleaned_file": final_output_path,  # ★ Stage2互換のためoutput_pathと同じ値を設定
        "md5": md5_full,  # 32桁
        "bpm": round(tempo, 1),
        "time_signature": time_signature,
        "note_count": len(all_notes),
        "duration_ms": int(pm.get_end_time() * 1000),
        "duration_ticks": int(pm.get_end_time() * pm.resolution * tempo / 60),
        "pitches": {
            "sum": pitches_sum,
            "counts": pitches_counts,
            "distribution": pitches_and_counts,
        },
        "patches_counts": patches_counts,
        "avg_velocity": avg_vel,
        "statistics": {
            "average_median_mode_time_ms": [avg_time, median_time, mode_time],
            "average_median_mode_dur_ms": [avg_dur, median_dur, mode_dur],
            "average_median_mode_vel": [avg_vel, median_vel, mode_vel],
        },
        "pitches_times_sum_ms": times_sum,
        "ms_chords_counts": ms_chords_counts,
        "total_number_of_chords": len(chords),
        "midi_ticks": pm.resolution,
    }
    
    return metadata


def safe_pickle_dump(obj: Any, path: Path) -> None:
    """
    Pickleを安全に書き込み（.tmp → fsync → rename）
    
    Args:
        obj: Pickle化するオブジェクト
        path: 最終保存先パス
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    
    # 一時ファイルに書き込み
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())  # ディスクへ確実に書き込み
    
    # アトミックにリネーム
    os.replace(tmp_path, path)


def write_sharded_pickle(
    metadata_list: List[Dict[str, Any]],
    output_dir: Path,
    shard_index: int,
    instrument: str,
    shard_prefix: str = "shard",
) -> Path:
    """
    Sharded pickleファイルを書き込み（安全版）
    
    Args:
        metadata_list: メタデータのリスト
        output_dir: 出力ディレクトリ
        shard_index: シャード番号
        instrument: 楽器名
        shard_prefix: シャードファイルのプレフィックス
        
    Returns:
        書き込んだpickleファイルのパス
    """
    shard_data = {
        "version": "lamda_v2_shard",
        "shard_index": shard_index,
        "instrument": instrument,
        "loops": metadata_list,
        "count": len(metadata_list),
        "summary": {
            "total_notes": sum(m.get("note_count", 0) for m in metadata_list),
            "avg_bpm": sum(m.get("bpm", 0) for m in metadata_list) / len(metadata_list) if metadata_list else 0,
        }
    }
    
    shard_path = output_dir / f"{instrument}_{shard_prefix}_{shard_index:05d}.pickle"
    safe_pickle_dump(shard_data, shard_path)
    
    return shard_path


def write_index_pickle(
    shard_paths: List[Path],
    base_dir: Path,
    instrument: str,
    shard_size: int,
) -> Path:
    """
    インデックスpickleファイルを書き込み（LAMDA Stage2互換）
    
    Args:
        shard_paths: シャードファイルのパスリスト
        base_dir: 相対パス計算の基準ディレクトリ
        instrument: 楽器名
        shard_size: シャードサイズ（件数/shard）
        
    Returns:
        書き込んだインデックスファイルのパス
    """
    # 各シャードから情報を収集
    shard_info = []
    total_loops = 0
    
    for shard_path in sorted(shard_paths):
        with open(shard_path, "rb") as f:
            shard_data = pickle.load(f)
        
        # 相対パス計算
        try:
            rel_path = shard_path.relative_to(base_dir)
        except ValueError:
            rel_path = shard_path
        
        shard_info.append({
            "path": str(rel_path),
            "shard_index": shard_data.get("shard_index", 0),
            "count": shard_data.get("count", len(shard_data.get("loops", []))),
            "summary": shard_data.get("summary", {}),
        })
        total_loops += shard_data.get("count", len(shard_data.get("loops", [])))
    
    # インデックスデータ
    index_data = {
        "version": "lamda_v2_index",
        "instrument": instrument,
        "total_files": total_loops,
        "shard_size": shard_size,
        "base_dir": str(base_dir),
        "shards": shard_info,
    }
    
    index_path = base_dir / f"{instrument}_metadata_v2.pickle"
    safe_pickle_dump(index_data, index_path)
    
    return index_path
