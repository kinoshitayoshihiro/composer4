#!/usr/bin/env python3
"""
Rhythm AI Stage2 Extractor - リズム専用特徴量抽出

和声AIと異なる特徴量:
- Groove: swing_pct, backbeat_strength, onset_deviation
- Beat Grid: kick/snare/hat onset patterns
- Density: notes per bar, velocity distribution
- Sync: downbeat accuracy, backbeat accuracy

Input:
    - LAMDA Stage1 index pickle (drums_index.pkl)
    - Cleaned MIDI files

Output:
    - rhythm_features.parquet (学習用DataFrame)
    - rhythm_stage2_summary.json (統計)

Usage:
    python scripts/rhythm_stage2_extractor.py \\
        --lamda-index output/rhythm_ai/drumclean_metadata/drums_index.pkl \\
        --input-dir output/rhythm_ai/drumclean_midi \\
        --output-dir output/rhythm_ai/stage2 \\
        --config configs/rhythm_stage2.yaml
"""

import argparse
import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mido
import numpy as np
import pandas as pd
import yaml

VERSION = "rhythm_stage2_v1.0"


# ========== Config ==========


@dataclass
class RhythmStage2Config:
    """Rhythm Stage2設定"""

    # スロット数（量子化）
    slots_4_4: int = 16
    slots_6_8: int = 24
    slots_5_8: int = 20
    slots_9_8: int = 36
    slots_3_4: int = 12

    # 特徴量範囲
    tempo_range: Tuple[float, float] = (60.0, 180.0)
    density_range: Tuple[float, float] = (0.0, 24.0)

    # グルーヴ閾値
    swing_threshold: float = 10.0
    backbeat_threshold: float = 0.7

    # KPI閾値（学習フィルタ用）- 拍子別設定
    # 4/4用（デフォルト）
    kick_downbeat_min: float = 0.40  # 40%以上
    snare_backbeat_min: float = 0.30  # 30%以上
    hat_density_max: float = 24.0  # 最大密度

    # 3/4、6/8等の非4/4拍子用（緩和）
    kick_downbeat_min_alt: float = 0.15  # 15%以上（3/4、5/4、6/8等）
    snare_backbeat_min_alt: float = 0.25  # 25%以上（6/8等）

    # 5/8専用（さらに緩和、複合拍子特性）
    kick_downbeat_min_5_8: float = 0.10  # 10%以上
    snare_backbeat_min_5_8: float = 0.20  # 20%以上

    # ダウンビート/バックビート窓幅係数
    downbeat_window_ratio: float = 0.15  # スロット数の15%（0.08→0.15に拡大）

    # パターン抽出
    min_bars: int = 2
    max_bars: int = 8

    # 出力
    parquet_compression: str = "snappy"
    sample_size: int = 1000


# ========== MIDI Analysis ==========


class RhythmFeatureExtractor:
    """リズム特徴量抽出器"""

    def __init__(self, config: RhythmStage2Config):
        self.config = config

    def extract_features(
        self, midi_path: Path, metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """1ファイルから特徴量抽出"""
        try:
            mid = mido.MidiFile(str(midi_path))

            # 基本情報
            ticks_per_beat = mid.ticks_per_beat
            tempo_bpm = metadata.get("bpm", metadata.get("tempo", 120))

            # time_sig解析（"4/4" → [4, 4]）
            time_sig_str = metadata.get("time_sig", "4/4")
            if isinstance(time_sig_str, str) and "/" in time_sig_str:
                parts = time_sig_str.split("/")
                time_sig = [int(parts[0]), int(parts[1])]
            elif isinstance(time_sig_str, list):
                time_sig = time_sig_str
            else:
                time_sig = [4, 4]

            # スロット数決定
            slots = self._get_slots(time_sig)

            # ノート収集（ドラムのみ）
            drum_notes = self._collect_drum_notes(mid)

            if len(drum_notes) < 4:
                return None

            # 特徴量計算
            features = {
                "loop_id": metadata.get("loop_id", midi_path.stem),
                "tempo_bpm": float(tempo_bpm),
                "time_sig_num": int(time_sig[0]),
                "time_sig_denom": int(time_sig[1]),
                "slots": int(slots),
                "num_notes": len(drum_notes),
            }

            # グリッド分析
            grid_features = self._analyze_grid(drum_notes, ticks_per_beat, time_sig, slots)
            features.update(grid_features)

            # グルーヴ分析
            groove_features = self._analyze_groove(drum_notes, ticks_per_beat)
            features.update(groove_features)

            # 密度分析
            density_features = self._analyze_density(drum_notes, ticks_per_beat, time_sig)
            features.update(density_features)

            # KPI計算
            kpi_features = self._calculate_kpis(drum_notes, ticks_per_beat, time_sig, slots)
            features.update(kpi_features)

            # Family推定（簡易版）
            family = self._estimate_family(features)
            features["family_label"] = family

            return features

        except Exception as e:
            logging.debug(f"Feature extraction failed for {midi_path}: {e}")
            return None

    def _get_slots(self, time_sig: List[int]) -> int:
        """拍子からスロット数を決定"""
        num, denom = time_sig

        if denom == 4:
            if num == 4:
                return self.config.slots_4_4
            elif num == 3:
                return self.config.slots_3_4
        elif denom == 8:
            if num in (6, 12):
                return self.config.slots_6_8
            elif num == 9:
                return self.config.slots_9_8
            elif num == 5:
                return self.config.slots_5_8

        return self.config.slots_4_4

    def _collect_drum_notes(self, mid: mido.MidiFile) -> List[Dict[str, Any]]:
        """ドラムノート収集"""
        notes = []

        for track in mid.tracks:
            current_tick = 0

            for msg in track:
                current_tick += msg.time

                if msg.type == "note_on" and msg.velocity > 0:
                    if msg.channel == 9:  # GM Drum Ch10 (0-indexed=9)
                        notes.append(
                            {
                                "tick": current_tick,
                                "pitch": msg.note,
                                "velocity": msg.velocity,
                                "role": self._pitch_to_role(msg.note),
                            }
                        )

        return sorted(notes, key=lambda x: x["tick"])

    @staticmethod
    def _pitch_to_role(pitch: int) -> str:
        """GM Drum Pitchからロール判定（拡張版 - 人間演奏対応）

        GM Drum Mapping (広い受け口):
        - Kick: 35, 36
        - Snare: 37 (side stick), 38 (acoustic snare), 40 (electric snare)
        - Hat: 42 (closed), 44 (pedal), 46 (open)
        - Ride/Cymbal: 49 (crash 1), 51 (ride 1), 53 (ride bell), 55 (splash), 57 (crash 2), 59 (ride 2)
        - Tom: 41 (low floor), 43 (high floor), 45 (low tom), 47 (low-mid tom), 48 (hi-mid tom), 50 (high tom)
        """
        # Kick (Bass Drum)
        if pitch in (35, 36):
            return "kick"

        # Snare (including side stick)
        elif pitch in (37, 38, 40):
            return "snare"

        # Hi-Hat (Closed, Pedal, Open)
        elif pitch in (42, 44, 46):
            return "hat"

        # Ride/Cymbal系 (hat_densityから除外)
        elif pitch in (49, 51, 53, 55, 57, 59):
            return "cymbal"

        # Tom群
        elif pitch in (41, 43, 45, 47, 48, 50):
            return "tom"

        # Clap/Percussion (39: Hand Clap, 等)
        elif pitch == 39:
            return "clap"

        else:
            return "other"

    def _analyze_grid(
        self, notes: List[Dict], ticks_per_beat: int, time_sig: List[int], slots: int
    ) -> Dict[str, Any]:
        """グリッド分析（オンセットパターン）"""
        num, denom = time_sig
        # bar length in ticks = ticks_per_beat * (quarter-notes per bar)
        # quarter-notes per bar = 4 * num / denom
        bar_ticks = int(round(ticks_per_beat * (4 * num / denom)))

        role_onsets = defaultdict(list)

        for note in notes:
            role = note["role"]
            slot_pos = int((note["tick"] % bar_ticks) / bar_ticks * slots)
            role_onsets[role].append(slot_pos)

        # オンセットヒストグラム
        kick_hist = np.zeros(slots)
        snare_hist = np.zeros(slots)
        hat_hist = np.zeros(slots)

        for pos in role_onsets.get("kick", []):
            kick_hist[pos % slots] += 1
        for pos in role_onsets.get("snare", []):
            snare_hist[pos % slots] += 1
        for pos in role_onsets.get("hat", []):
            hat_hist[pos % slots] += 1

        return {
            "kick_pattern": kick_hist.tolist(),
            "snare_pattern": snare_hist.tolist(),
            "hat_pattern": hat_hist.tolist(),
            "kick_onset_count": len(role_onsets.get("kick", [])),
            "snare_onset_count": len(role_onsets.get("snare", [])),
            "hat_onset_count": len(role_onsets.get("hat", [])),
        }

    def _analyze_groove(self, notes: List[Dict], ticks_per_beat: int) -> Dict[str, Any]:
        """グルーヴ分析（フォールバック版）"""
        # バックビート強度（スネアのベロシティ平均）
        snare_vels = [n["velocity"] for n in notes if n["role"] == "snare"]
        backbeat_strength = float(np.mean(snare_vels)) / 127.0 if snare_vels else 0.0

        # スウィング推定（ハット間隔の分散）
        hat_notes = [n for n in notes if n["role"] == "hat"]
        intervals = []

        for i in range(1, len(hat_notes)):
            intervals.append(hat_notes[i]["tick"] - hat_notes[i - 1]["tick"])

        if intervals:
            swing_pct = float(np.std(intervals) / ticks_per_beat * 100)
            swing_pct = min(swing_pct, 100.0)
        else:
            swing_pct = 0.0

        return {
            "swing_pct": swing_pct,
            "backbeat_strength": backbeat_strength,
            "onset_deviation_mean": 0.0,
            "onset_deviation_std": 0.0,
        }

    def _analyze_density(
        self, notes: List[Dict], ticks_per_beat: int, time_sig: List[int]
    ) -> Dict[str, Any]:
        """密度分析"""
        num, denom = time_sig
        bar_ticks = int(round(ticks_per_beat * (4 * num / denom)))

        max_tick = max(n["tick"] for n in notes)
        num_bars = int(np.ceil(max_tick / bar_ticks))

        notes_per_bar = []

        for bar_idx in range(num_bars):
            bar_start = bar_idx * bar_ticks
            bar_end = (bar_idx + 1) * bar_ticks

            bar_notes = [n for n in notes if bar_start <= n["tick"] < bar_end]
            notes_per_bar.append(len(bar_notes))

        return {
            "density_mean": float(np.mean(notes_per_bar)),
            "density_std": float(np.std(notes_per_bar)),
            "density_min": float(np.min(notes_per_bar)) if notes_per_bar else 0.0,
            "density_max": float(np.max(notes_per_bar)) if notes_per_bar else 0.0,
        }

    def _calculate_kpis(
        self, notes: List[Dict], ticks_per_beat: int, time_sig: List[int], slots: int
    ) -> Dict[str, Any]:
        """KPI計算（人間演奏対応版）

        改善点:
        - ダウンビート窓: ±2スロット（スロット数は拍子に応じて可変）
        - ベロシティ下限: 30（ゴーストノート除外）
        - hat_density: hat系のみ（cymbal除外）
        - backbeat_slots: 拍子に応じて自動推定（4/4は 2拍目・4拍目、6/8は中央など）
        """
        num, denom = time_sig
        # bar length in ticks (quarter-note basis)
        bar_ticks = int(round(ticks_per_beat * (4 * num / denom)))

        # === Backbeat slot positions (meter-aware) ===
        backbeat_slots = set()

        # Simple helper for safe slot conversion
        def beat_index_to_slot(idx: int, total_beats: int) -> int:
            if total_beats <= 0:
                return 0
            return int(round((idx * slots) / total_beats)) % max(slots, 1)

        # Case 1: simple meters with quarter-note beats (e.g., 4/4, 3/4, 2/4)
        quarter_beats_f = (
            4 * num / denom
        )  # may be non-integer in odd meters, but typical meters yield int
        quarter_beats = int(round(quarter_beats_f))

        if denom == 4:
            if quarter_beats >= 4:
                # 4/4: beats [1,3] (0-based: 1 and 3) → 2拍目と4拍目
                backbeat_slots = {
                    beat_index_to_slot(1, quarter_beats),
                    beat_index_to_slot(3, quarter_beats),
                }
            elif quarter_beats in (2, 3):
                # 2/4, 3/4: 2拍目をバックビート扱い
                backbeat_slots = {beat_index_to_slot(1, quarter_beats)}
            else:
                # Fallback: bar center
                backbeat_slots = {int(round(slots / 2))}
        elif denom == 8 and num in (6, 9, 12):
            # Compound meters (6/8, 9/8, 12/8): use dotted-quarter beats
            compound_beats = max(1, num // 3)  # 6/8→2, 9/8→3, 12/8→4
            # backbeats on odd indices: e.g., 6/8 → beat #2 (0-based 1), 12/8 → beats #2 and #4
            backbeat_slots = {
                int(round((b * slots) / compound_beats)) % max(slots, 1)
                for b in range(compound_beats)
                if b % 2 == 1
            }
            if not backbeat_slots:
                backbeat_slots = {int(round(slots / 2))}
        elif denom == 8 and num == 5:
            # 5/8は (2+3) か (3+2) の複合系が一般的。
            # いずれにも耐性を持たせるため、2/5 と 3/5 の位置をバックビート候補に採用。
            backbeat_slots = {
                int(round(slots * 2 / 5)) % max(slots, 1),
                int(round(slots * 3 / 5)) % max(slots, 1),
            }
        else:
            # Generic fallback: bar center
            backbeat_slots = {int(round(slots / 2))}

        # ダウンビート/バックビートの許容窓（config.downbeat_window_ratio、下限2）
        window = max(2, int(round(slots * self.config.downbeat_window_ratio)))
        downbeat_window = window
        backbeat_window = window

        # キックのダウンビート率（窓付き + velocity不問）
        kick_notes = [n for n in notes if n["role"] == "kick"]
        kick_on_downbeat = sum(
            1
            for n in kick_notes
            if abs(int((n["tick"] % bar_ticks) / bar_ticks * slots) - 0) <= downbeat_window
        )
        kick_downbeat_rate = float(kick_on_downbeat / len(kick_notes)) if kick_notes else 0.0

        # 一部ジャンルでバックビートをClap(39)が担う場合を考慮
        snare_notes = [n for n in notes if n["role"] in ("snare", "clap") and n["velocity"] >= 30]
        snare_on_backbeat = sum(
            1
            for n in snare_notes
            if any(
                abs(int((n["tick"] % bar_ticks) / bar_ticks * slots) - slot) <= backbeat_window
                for slot in backbeat_slots
            )
        )
        snare_backbeat_rate = float(snare_on_backbeat / len(snare_notes)) if snare_notes else 0.0

        # ハット密度（hat系のみ、cymbal除外）
        hat_notes = [n for n in notes if n["role"] == "hat"]
        max_tick = max(n["tick"] for n in notes) if notes else bar_ticks
        num_bars = int(np.ceil(max_tick / bar_ticks))
        hat_density = float(len(hat_notes) / max(num_bars, 1))

        return {
            "kick_downbeat_rate": kick_downbeat_rate,
            "snare_backbeat_rate": snare_backbeat_rate,
            "hat_density": hat_density,
        }

    def _estimate_family(self, features: Dict[str, Any]) -> str:
        """Family推定（簡易版）"""
        swing = features["swing_pct"]
        hat_density = features["hat_density"]

        if swing > self.config.swing_threshold:
            if hat_density > 12:
                return "SWING_16"
            else:
                return "SWING_8"
        else:
            if hat_density > 12:
                return "STRAIGHT_16"
            else:
                return "STRAIGHT_8"


# ========== Index Loader ==========


def load_lamda_index(index_path: Path) -> List[Dict[str, Any]]:
    """LAMDA Stage1 indexをロード"""
    with open(index_path, "rb") as f:
        data = pickle.load(f)

    # 形式1: Shardリスト形式（新v2.0+）
    if isinstance(data, dict) and "shards" in data:
        shards = data["shards"]
        index_dir = index_path.parent

        records = []
        for shard_info in shards:
            shard_path = index_dir / shard_info.get("path", "")

            if not shard_path.exists():
                logging.warning(f"Shard not found: {shard_path}")
                continue

            # Shardを読み込み
            with open(shard_path, "rb") as sf:
                shard_data = pickle.load(sf)

            loops = shard_data.get("loops", [])

            for loop in loops:
                midi_path = loop.get("cleaned_file") or loop.get("output_path")

                if not midi_path:
                    continue

                records.append(
                    {
                        "loop_id": loop.get("md5", Path(midi_path).stem),
                        "midi_path": Path(midi_path),
                        "metadata": {
                            "bpm": loop.get("bpm", 120),
                            "tempo": loop.get("bpm", 120),
                            "time_sig": loop.get("time_signature", "4/4"),
                            "note_count": loop.get("note_count", 0),
                        },
                    }
                )

        return records

    # 形式2: 単一インデックス形式
    elif isinstance(data, dict) and "index" in data:
        index = data["index"]

        records = []
        for loop_id, entry in index.items():
            midi_path = entry.get("midi_path")

            if not midi_path:
                logging.warning(f"Entry missing midi_path: {loop_id}")
                continue

            records.append(
                {
                    "loop_id": loop_id,
                    "midi_path": Path(midi_path),
                    "metadata": entry.get("metadata", {}),
                }
            )

        return records

    # 形式3: Shard単体形式
    elif isinstance(data, dict) and "loops" in data:
        loops = data["loops"]

        records = []
        for loop in loops:
            midi_path = loop.get("midi_path") or loop.get("cleaned_file")

            if not midi_path:
                continue

            records.append(
                {
                    "loop_id": loop.get("loop_id", Path(midi_path).stem),
                    "midi_path": Path(midi_path),
                    "metadata": loop.get("metadata", loop),
                }
            )

        return records

    # 形式4: E-GMD形式（Shards参照形式）
    elif isinstance(data, dict) and "num_shards" in data:
        # E-GMD clean_egmd_simple.pyが作成した形式
        index_dir = index_path.parent
        num_shards = data.get("num_shards", 0)

        records = []
        for shard_idx in range(num_shards):
            shard_path = index_dir / f"drums_{shard_idx:04d}.pkl"

            if not shard_path.exists():
                logging.warning(f"Shard not found: {shard_path}")
                continue

            # Shardを読み込み
            with open(shard_path, "rb") as sf:
                shard_data = pickle.load(sf)

            # shard_dataはリスト形式
            for entry in shard_data:
                midi_path = entry.get("cleaned_path") or entry.get("original_path")

                if not midi_path:
                    continue

                records.append(
                    {
                        "loop_id": entry.get("md5", Path(midi_path).stem),
                        "midi_path": Path(midi_path),
                        "metadata": {
                            "filename": entry.get("filename", ""),
                            "file_index": entry.get("file_index", 0),
                        },
                    }
                )

        return records

    else:
        raise ValueError(
            f"Unknown index format: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'N/A'}"
        )


# ========== Main Processor ==========


class RhythmStage2Processor:
    """Rhythm Stage2メインプロセッサ"""

    def __init__(
        self, lamda_index: Path, input_dir: Path, output_dir: Path, config: RhythmStage2Config
    ):
        self.lamda_index = lamda_index
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config

        self.extractor = RhythmFeatureExtractor(config)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self) -> Dict[str, Any]:
        """全処理実行"""
        print(f"📂 Loading index: {self.lamda_index}")
        records = load_lamda_index(self.lamda_index)
        print(f"✅ Loaded {len(records)} records")

        print(f"\n🔍 Extracting rhythm features...")
        features_list = []

        total = len(records)
        for i, record in enumerate(records):
            if i % 100 == 0:
                print(f"  Progress: {i}/{total} ({100*i/total:.1f}%)")

            midi_path = record["midi_path"]

            if not midi_path.is_absolute():
                midi_path = self.input_dir / midi_path

            if not midi_path.exists():
                logging.debug(f"MIDI not found: {midi_path}")
                continue

            features = self.extractor.extract_features(midi_path, record["metadata"])

            if features:
                features_list.append(features)

        print(f"✅ Extracted features from {len(features_list)} files")

        if not features_list:
            raise RuntimeError("No features extracted")

        # DataFrame生成
        df = pd.DataFrame(features_list)

        # 必須カラム保証（学習/マージ互換性）
        needed_cols = [
            "tempo_bpm",
            "swing_pct",
            "backbeat_strength",
            "kick_downbeat_rate",
            "snare_backbeat_rate",
            "hat_density",
            "family_label",
        ]

        for col in needed_cols:
            if col not in df.columns:
                if col == "family_label":
                    df[col] = "UNKNOWN"
                else:
                    df[col] = 0.0

        # カラム順序統一（必須→任意）
        ordered_cols = needed_cols + [c for c in df.columns if c not in needed_cols]
        df = df[ordered_cols]

        # フィルタ統計（拍子別KPI閾値適用）
        total = len(df)

        # 拍子別フィルタ条件
        # 4/4: 標準閾値
        mask_4_4 = (
            (df["time_sig_num"] == 4)
            & (df["time_sig_denom"] == 4)
            & (df["kick_downbeat_rate"] >= self.config.kick_downbeat_min)
            & (df["snare_backbeat_rate"] >= self.config.snare_backbeat_min)
            & (df["hat_density"] <= self.config.hat_density_max)
        )

        # 3/4、5/4、6/8等: 緩和閾値（キック・スネア）
        mask_other = (
            ~((df["time_sig_num"] == 4) & (df["time_sig_denom"] == 4))
            & ~((df["time_sig_num"] == 5) & (df["time_sig_denom"] == 8))
            & (df["kick_downbeat_rate"] >= self.config.kick_downbeat_min_alt)
            & (df["snare_backbeat_rate"] >= self.config.snare_backbeat_min_alt)
            & (df["hat_density"] <= self.config.hat_density_max)
        )

        # 5/8専用: さらに緩和閾値
        mask_5_8 = (
            (df["time_sig_num"] == 5)
            & (df["time_sig_denom"] == 8)
            & (df["kick_downbeat_rate"] >= self.config.kick_downbeat_min_5_8)
            & (df["snare_backbeat_rate"] >= self.config.snare_backbeat_min_5_8)
            & (df["hat_density"] <= self.config.hat_density_max)
        )

        df_filtered = df[mask_4_4 | mask_other | mask_5_8]

        passed = len(df_filtered)
        pass_rate = (passed / total * 100) if total > 0 else 0.0

        print(f"\n📊 Quality Filter:")
        print(f"   Total:  {total}")
        print(f"   Passed: {passed} ({pass_rate:.1f}%)")

        # 保存
        parquet_path = self.output_dir / "rhythm_features.parquet"
        df.to_parquet(str(parquet_path), compression=self.config.parquet_compression, index=False)
        print(f"💾 Saved: {parquet_path}")

        # 合格データも別途保存
        parquet_passed_path = self.output_dir / "rhythm_features_passed.parquet"
        df_filtered.to_parquet(
            str(parquet_passed_path), compression=self.config.parquet_compression, index=False
        )
        print(f"💾 Saved (passed only): {parquet_passed_path}")

        # サマリー（拍子別統計追加）
        df["time_sig"] = df["time_sig_num"].astype(str) + "/" + df["time_sig_denom"].astype(str)
        df_filtered["time_sig"] = (
            df_filtered["time_sig_num"].astype(str)
            + "/"
            + df_filtered["time_sig_denom"].astype(str)
        )

        summary = {
            "version": VERSION,
            "total_records": total,
            "passed_records": passed,
            "pass_rate": pass_rate,
            "config": {
                "kick_downbeat_min_4_4": self.config.kick_downbeat_min,
                "snare_backbeat_min_4_4": self.config.snare_backbeat_min,
                "kick_downbeat_min_other": self.config.kick_downbeat_min_alt,
                "snare_backbeat_min_other": self.config.snare_backbeat_min_alt,
                "hat_density_max": self.config.hat_density_max,
                "downbeat_window_ratio": self.config.downbeat_window_ratio,
            },
            "stats": {
                "tempo_mean": float(df["tempo_bpm"].mean()),
                "tempo_std": float(df["tempo_bpm"].std()),
                "swing_mean": float(df["swing_pct"].mean()),
                "backbeat_mean": float(df["backbeat_strength"].mean()),
                "density_mean": float(df["density_mean"].mean()),
                "family_distribution": df["family_label"].value_counts().to_dict(),
                "family_distribution_passed": df_filtered["family_label"].value_counts().to_dict(),
            },
            "time_sig_stats": {
                ts: {
                    "total": int((df["time_sig"] == ts).sum()),
                    "passed": int((df_filtered["time_sig"] == ts).sum()),
                    "pass_rate": (
                        float(
                            (df_filtered["time_sig"] == ts).sum()
                            / (df["time_sig"] == ts).sum()
                            * 100
                        )
                        if (df["time_sig"] == ts).sum() > 0
                        else 0.0
                    ),
                }
                for ts in sorted(df["time_sig"].unique())
            },
        }

        summary_path = self.output_dir / "rhythm_stage2_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {summary_path}")

        return summary


# ========== CLI ==========


def main():
    parser = argparse.ArgumentParser(description="Rhythm AI Stage2 Extractor")
    parser.add_argument("--lamda-index", type=Path, required=True, help="LAMDA Stage1 index pickle")
    parser.add_argument("--input-dir", type=Path, required=True, help="Cleaned MIDI directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 設定読み込み
    config = RhythmStage2Config()

    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

            for key, value in cfg_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # 処理実行
    processor = RhythmStage2Processor(
        lamda_index=args.lamda_index,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config,
    )

    summary = processor.process()

    print(f"\n{'='*70}")
    print(f"✅ Rhythm Stage2 processing completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
