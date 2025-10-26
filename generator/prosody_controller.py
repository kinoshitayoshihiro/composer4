#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generator/prosody_controller.py - Phase 23: Prosody Control

Vocal STEMの歌詞アンカー（lyric_anchors.json）を使って、
伴奏のノート（Velocity/Duration/CC等）を制御します。

機能:
- アンカーのwindow_ms範囲内のノートを検出
- class別の処理（sibilant=デエッシング、stress=強調、plosive=短縮）
- 窓重なり抑制（近接マージ、最大同時窓数制限）
- セクション別パラメータ対応

使用例:
    from generator.prosody_controller import ProsodyController
    
    controller = ProsodyController(
        anchors_path="analysis/lyric_anchors.json",
        config={
            "sibilant": {"vel_scale": 0.7, "hh_reduce": 0.5},
            "stress": {"vel_scale": 1.2},
            "plosive": {"duration_scale": 0.8}
        }
    )
    
    # ノートリストを処理
    notes = controller.apply_prosody(notes, role="guitar")
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class Anchor:
    """歌詞アンカー"""
    time: float  # 秒
    token: str
    classes: List[str]  # ["sibilant", "stress", "plosive"]
    section: Optional[str] = None
    time_ql: float = 0.0
    window_pre: float = 0.0  # ms
    window_post: float = 0.0  # ms
    
    @property
    def window_start(self) -> float:
        """窓開始時刻（秒）"""
        return self.time - self.window_pre / 1000.0
    
    @property
    def window_end(self) -> float:
        """窓終了時刻（秒）"""
        return self.time + self.window_post / 1000.0
    
    def contains(self, time: float) -> bool:
        """指定時刻が窓範囲内かチェック"""
        return self.window_start <= time <= self.window_end


class ProsodyController:
    """Prosody制御コントローラー
    
    lyric_anchorsを読み込み、ノートのVelocity/Duration/CC等を
    歌詞タイミングに合わせて調整します。
    """
    
    # デフォルト設定
    DEFAULT_CONFIG = {
        "sibilant": {
            "vel_scale": 0.75,      # Velocity 25%減（デエッシング）
            "hh_reduce": 0.6,       # HH/Crash 40%減
            "guitar_hicut": True,   # ギター高域カット
            "duration_scale": 1.0,  # 持続時間変化なし
        },
        "stress": {
            "vel_scale": 1.15,      # Velocity 15%増（強調）
            "duration_scale": 1.0,
            "cc11_boost": 10,       # Expression +10
        },
        "plosive": {
            "vel_scale": 1.0,
            "duration_scale": 0.85,  # 持続時間 15%減（短縮）
            "staccato": True,
        },
        "max_overlaps": 3,          # 最大同時窓数（密集抑制）
        "merge_threshold_ms": 50,   # 近接窓マージ閾値
    }
    
    def __init__(
        self,
        anchors_path: Optional[Path] = None,
        anchors_data: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            anchors_path: lyric_anchors.json のパス
            anchors_data: アンカーデータ（辞書）
            config: 設定辞書（DEFAULT_CONFIGをオーバーライド）
        """
        self.anchors: List[Anchor] = []
        self.config = self._merge_config(config)
        
        # アンカー読み込み
        if anchors_data:
            self._load_from_dict(anchors_data)
        elif anchors_path:
            self._load_from_file(anchors_path)
        
        # 窓重なり抑制
        if self.config.get("merge_threshold_ms", 0) > 0:
            self._merge_overlapping_anchors()
    
    def _merge_config(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """ユーザー設定とデフォルト設定をマージ"""
        config = dict(self.DEFAULT_CONFIG)
        if user_config:
            for key, val in user_config.items():
                if isinstance(val, dict) and key in config:
                    config[key].update(val)
                else:
                    config[key] = val
        return config
    
    def _load_from_file(self, path: Path):
        """ファイルからアンカー読み込み"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._load_from_dict(data)
    
    def _load_from_dict(self, data: Dict[str, Any]):
        """辞書からアンカー読み込み"""
        unit = data.get("unit", "sec")
        for a in data.get("anchors", []):
            time = float(a["time"])
            
            # 単位変換（QL→秒は後で実装）
            if unit == "ql":
                # 暫定：120BPM仮定で変換
                time = time / 4.0 * 60.0 / 120.0
            
            # window_msを取得
            win = a.get("window_ms", {})
            pre = float(win.get("pre", 0.0))
            post = float(win.get("post", 0.0))
            
            # classリスト取得
            classes = a.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            
            anchor = Anchor(
                time=time,
                token=a.get("token", ""),
                classes=classes,
                section=a.get("section"),
                time_ql=float(a.get("time_ql", 0.0)),
                window_pre=pre,
                window_post=post
            )
            self.anchors.append(anchor)
        
        print(f"[ProsodyController] Loaded {len(self.anchors)} anchors", file=sys.stderr)
    
    def _merge_overlapping_anchors(self):
        """近接する窓をマージして密集を抑制"""
        threshold_sec = self.config["merge_threshold_ms"] / 1000.0
        
        merged = []
        i = 0
        while i < len(self.anchors):
            current = self.anchors[i]
            
            # 次のアンカーと近接しているかチェック
            if i + 1 < len(self.anchors):
                next_anchor = self.anchors[i + 1]
                gap = next_anchor.window_start - current.window_end
                
                if gap < threshold_sec:
                    # マージ：windowを拡張、classを結合
                    merged_classes = list(set(current.classes + next_anchor.classes))
                    merged_time = (current.time + next_anchor.time) / 2
                    merged_token = f"{current.token}+{next_anchor.token}"
                    
                    merged.append(Anchor(
                        time=merged_time,
                        token=merged_token,
                        classes=merged_classes,
                        section=current.section,
                        time_ql=(current.time_ql + next_anchor.time_ql) / 2,
                        window_pre=current.window_pre,
                        window_post=next_anchor.window_post
                    ))
                    i += 2  # 2つスキップ
                    continue
            
            merged.append(current)
            i += 1
        
        if len(merged) < len(self.anchors):
            print(f"[ProsodyController] Merged {len(self.anchors) - len(merged)} overlapping anchors", file=sys.stderr)
            self.anchors = merged
    
    def get_anchors_for_time(self, time: float, max_count: Optional[int] = None) -> List[Anchor]:
        """指定時刻に影響するアンカーを取得
        
        Args:
            time: 時刻（秒）
            max_count: 最大取得数（Noneで無制限）
        
        Returns:
            アンカーリスト（時刻順）
        """
        matching = [a for a in self.anchors if a.contains(time)]
        
        if max_count and len(matching) > max_count:
            # 時刻に最も近いmax_count個を返す
            matching.sort(key=lambda a: abs(a.time - time))
            matching = matching[:max_count]
        
        return sorted(matching, key=lambda a: a.time)
    
    def apply_prosody(
        self,
        notes: List[Dict[str, Any]],
        role: str = "generic",
        tempo: float = 120.0
    ) -> List[Dict[str, Any]]:
        """ノートリストにProsody制御を適用
        
        Args:
            notes: ノートリスト [{"time": sec, "pitch": ..., "vel": ..., "dur": ...}, ...]
            role: 楽器ロール（"piano", "guitar", "drums" 等）
            tempo: テンポ（BPM）
        
        Returns:
            処理済みノートリスト
        """
        if not self.anchors:
            return notes  # アンカーなし→そのまま返す
        
        max_overlaps = self.config.get("max_overlaps", 3)
        
        for note in notes:
            time = float(note.get("time", 0.0))
            
            # この時刻に影響するアンカーを取得
            active_anchors = self.get_anchors_for_time(time, max_count=max_overlaps)
            
            if not active_anchors:
                continue
            
            # 各アンカーのclassに基づいて処理
            for anchor in active_anchors:
                self._apply_anchor_to_note(note, anchor, role)
        
        return notes
    
    def _apply_anchor_to_note(
        self,
        note: Dict[str, Any],
        anchor: Anchor,
        role: str
    ):
        """単一アンカーをノートに適用
        
        Args:
            note: ノート辞書（in-place変更）
            anchor: アンカー
            role: 楽器ロール
        """
        # 各classの処理を順番に適用
        for cls in anchor.classes:
            if cls == "sibilant":
                self._apply_sibilant(note, anchor, role)
            elif cls == "stress":
                self._apply_stress(note, anchor, role)
            elif cls == "plosive":
                self._apply_plosive(note, anchor, role)
    
    def _apply_sibilant(self, note: Dict[str, Any], anchor: Anchor, role: str):
        """Sibilant（歯擦音）処理：デエッシング
        
        - Velocity減少（HH/Crash/ギター高域）
        - ギター高域カット（フィルター相当）
        """
        cfg = self.config["sibilant"]
        
        # 現在のVelocity
        vel = float(note.get("vel", 64))
        
        # 楽器別処理
        if role in ["drums"]:
            # HH/Crash系の判定（簡易：pitchで判別）
            pitch = note.get("pitch", 60)
            if pitch >= 42:  # HH/Crash域
                vel *= cfg["hh_reduce"]
        
        elif role in ["guitar", "strings"]:
            # 高域は強めに減衰
            pitch = note.get("pitch", 60)
            if pitch >= 72:  # 高域
                vel *= cfg["vel_scale"] * 0.9
            else:
                vel *= cfg["vel_scale"]
        
        else:
            # その他：標準減衰
            vel *= cfg["vel_scale"]
        
        # 更新
        note["vel"] = int(np.clip(vel, 1, 127))
        
        # メタデータ記録（デバッグ用）
        if "prosody" not in note:
            note["prosody"] = []
        note["prosody"].append(f"sibilant@{anchor.time:.2f}s")
    
    def _apply_stress(self, note: Dict[str, Any], anchor: Anchor, role: str):
        """Stress（強勢）処理：強調
        
        - Velocity増加
        - Expression CC増加
        """
        cfg = self.config["stress"]
        
        # Velocity増加
        vel = float(note.get("vel", 64))
        vel *= cfg["vel_scale"]
        note["vel"] = int(np.clip(vel, 1, 127))
        
        # CC11増加（Expressionがある場合）
        if "cc11" in note and cfg.get("cc11_boost"):
            cc11 = float(note.get("cc11", 100))
            cc11 = min(127, cc11 + cfg["cc11_boost"])
            note["cc11"] = int(cc11)
        
        # メタデータ記録
        if "prosody" not in note:
            note["prosody"] = []
        note["prosody"].append(f"stress@{anchor.time:.2f}s")
    
    def _apply_plosive(self, note: Dict[str, Any], anchor: Anchor, role: str):
        """Plosive（破裂音）処理：短縮
        
        - Duration短縮（スタッカート）
        """
        cfg = self.config["plosive"]
        
        # Duration短縮
        dur = float(note.get("dur", 0.5))
        dur *= cfg["duration_scale"]
        note["dur"] = max(0.05, dur)  # 最小50ms
        
        # メタデータ記録
        if "prosody" not in note:
            note["prosody"] = []
        note["prosody"].append(f"plosive@{anchor.time:.2f}s")
    
    def get_statistics(self) -> Dict[str, Any]:
        """アンカー統計を取得"""
        if not self.anchors:
            return {}
        
        # Class分布
        class_counts = {}
        for anchor in self.anchors:
            for cls in anchor.classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # 窓幅統計
        window_pres = [a.window_pre for a in self.anchors]
        window_posts = [a.window_post for a in self.anchors]
        
        return {
            "total_anchors": len(self.anchors),
            "class_distribution": class_counts,
            "window_stats": {
                "pre_ms": {
                    "min": float(np.min(window_pres)),
                    "max": float(np.max(window_pres)),
                    "mean": float(np.mean(window_pres)),
                },
                "post_ms": {
                    "min": float(np.min(window_posts)),
                    "max": float(np.max(window_posts)),
                    "mean": float(np.mean(window_posts)),
                },
            },
            "sections": list(set(a.section for a in self.anchors if a.section)),
        }


def load_prosody_controller(
    anchors_path: Optional[Path] = None,
    config_path: Optional[Path] = None
) -> Optional[ProsodyController]:
    """ProsodyControllerをロード（ヘルパー関数）
    
    Args:
        anchors_path: lyric_anchors.json パス
        config_path: 設定ファイルパス（YAML/JSON）
    
    Returns:
        ProsodyController インスタンス（ファイルが無い場合はNone）
    """
    if not anchors_path or not anchors_path.exists():
        return None
    
    config = None
    if config_path and config_path.exists():
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    return ProsodyController(anchors_path=anchors_path, config=config)


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Phase 23: Prosody Controller Test")
    ap.add_argument("--anchors", required=True, help="lyric_anchors.json path")
    ap.add_argument("--config", help="Config YAML/JSON (optional)")
    ap.add_argument("--stats", action="store_true", help="Show statistics only")
    
    args = ap.parse_args()
    
    controller = load_prosody_controller(
        anchors_path=Path(args.anchors),
        config_path=Path(args.config) if args.config else None
    )
    
    if not controller:
        print("[ERROR] Failed to load anchors", file=sys.stderr)
        sys.exit(1)
    
    if args.stats:
        stats = controller.get_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(f"✅ Loaded {len(controller.anchors)} anchors")
        print("Run with --stats to see detailed statistics")
