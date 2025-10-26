#!/usr/bin/env python3
"""
LAMDA データソース統合ローダー（NO-OP安全設計）

**設計思想**:
- 全リソースはオプション（None → NO-OP）
- 遅延ロード（初回アクセス時のみ読み込み）
- id_map対応（Pop909/MAESTRO等の自動マッピング）
- 既存パイプライン非破壊（あれば使う/無ければスキップ）

**活用箇所**:
- KILO_CHORDS_DATA → chordmap_external（優先進行）
- META_DATA → patches/groove/controls先験
- SIGNATURES_DATA → timesig救済（1/4→4/4自動補正の裏取り）
- TOTALS_MATRIX → pitch/dur/velの外れ値スコア（品質ゲート）
"""
from __future__ import annotations
import pickle
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

class LamdaSources:
    """LAMDA v2.3 データソースの統一インターフェース（origin + local両対応）"""
    
    def __init__(
        self,
        kilo: Optional[str] = None,
        meta_dir: Optional[str] = None,
        signatures: Optional[str] = None,
        totals: Optional[str] = None,
        id_map_csv: Optional[str] = None,
        # NEW: local bundles
        local_kilo: Optional[str] = None,
        local_meta_dir: Optional[str] = None,
        local_signatures: Optional[str] = None,
        local_totals: Optional[str] = None,
        # NEW: preference
        prefer_local: bool = False,
    ):
        """
        Args:
            kilo: LAMDa_KILO_CHORDS_DATA.pickle へのパス (origin)
            meta_dir: META_DATA/*.pickle のディレクトリ (origin)
            signatures: LAMDa_SIGNATURES_DATA.pickle へのパス (origin)
            totals: LAMDa_TOTALS.pickle へのパス (origin)
            id_map_csv: auto_file_id_map.csv（src_id,target_id形式）
            local_kilo: LOCAL_KILO_CHORDS_DATA.pickle へのパス
            local_meta_dir: LOCAL_META_DATA/*.pickle のディレクトリ
            local_signatures: LOCAL_SIGNATURES_DATA.pickle へのパス
            local_totals: LOCAL_TOTALS.pickle へのパス
            prefer_local: True なら local を優先して解決、False なら origin 優先
        """
        # origin paths
        self.kilo_path = Path(kilo) if kilo else None
        self.meta_dir = Path(meta_dir) if meta_dir else None
        self.sign_path = Path(signatures) if signatures else None
        self.tot_path = Path(totals) if totals else None
        
        # local paths
        self.local_kilo_path = Path(local_kilo) if local_kilo else None
        self.local_meta_dir = Path(local_meta_dir) if local_meta_dir else None
        self.local_sign_path = Path(local_signatures) if local_signatures else None
        self.local_tot_path = Path(local_totals) if local_totals else None
        
        # ID マッピング（Pop909→pop909_001、MAESTRO→maestro_0001等）
        self.id_map = self._load_id_map(id_map_csv) if id_map_csv else {}
        
        # preference
        self.prefer_local = bool(prefer_local)
        
        # 遅延ロード用キャッシュ（origin）
        self._kilo: Optional[Dict[str, List]] = None
        self._meta: Dict[str, Any] = {}
        self._sign: Optional[Dict[str, List]] = None
        self._tot: Optional[Dict[str, Any]] = None
        
        # 遅延ロード用キャッシュ（local）
        self._local_kilo: Optional[Dict[str, List]] = None
        self._local_meta: Dict[str, Any] = {}
        self._local_sign: Optional[Dict[str, List]] = None
        self._local_tot: Optional[Dict[str, Any]] = None

    def _load_id_map(self, csv_path: str) -> Dict[str, str]:
        """auto_file_id_map.csv の読み込み
        
        Format: src_id,target_id
        Example: "001","pop909_001"
        """
        mp = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mp[row["src_id"]] = row["target_id"]
        return mp

    # ========================================
    # KILO_CHORDS_DATA（進行カタログ）
    # ========================================
    def load_kilo(self) -> None:
        """KILO pickle の遅延ロード（origin、初回のみ）"""
        if self._kilo is not None or not self.kilo_path:
            return
        if not self.kilo_path.exists():
            return
        
        with open(self.kilo_path, "rb") as f:
            data = pickle.load(f)
        
        # フォーマット: {file_id: [(root, quality, time_ql), ...]}
        self._kilo = data if isinstance(data, dict) else {}
    
    def load_local_kilo(self) -> None:
        """LOCAL_KILO pickle の遅延ロード（初回のみ）"""
        if self._local_kilo is not None or not self.local_kilo_path:
            return
        if not self.local_kilo_path.exists():
            return
        
        with open(self.local_kilo_path, "rb") as f:
            data = pickle.load(f)
        
        # フォーマット1（dict）: {file_id: [(root, quality, time_ql), ...]}
        # フォーマット2（list）: [[file_id, {"tokens": [[bar, tok], ...]}], ...]
        if isinstance(data, dict):
            self._local_kilo = data
        elif isinstance(data, list):
            self._local_kilo = {}
            for rec in data:
                fid, payload = rec[0], rec[1]
                self._local_kilo[str(fid)] = payload
        else:
            self._local_kilo = {}

    def get_kilo_chords(self, file_id: str) -> Optional[List]:
        """指定file_idのKILO進行を取得（origin + local 両対応）
        
        Args:
            file_id: MIDIファイル名（拡張子なし）
        
        Returns:
            [(root, quality, time_ql), ...] or None
        """
        self.load_kilo()
        self.load_local_kilo()
        
        # id_map適用（Pop909→pop909_001等）
        fid = self.id_map.get(file_id, file_id)
        
        # prefer_local による優先順
        first_dict = self._local_kilo if self.prefer_local else self._kilo
        second_dict = self._kilo if self.prefer_local else self._local_kilo
        
        if first_dict and fid in first_dict:
            return first_dict[fid]
        if second_dict and fid in second_dict:
            return second_dict[fid]
        return None

    # ========================================
    # META_DATA（patches/groove/controls）
    # ========================================
    def _meta_shards(self) -> List[Path]:
        """META_DATA/*.pickle のシャード一覧（origin）"""
        if not self.meta_dir:
            return []
        return sorted(self.meta_dir.glob("LAMDa_META_DATA_*.pickle"))
    
    def _local_meta_shards(self) -> List[Path]:
        """LOCAL_META_DATA/*.pickle のシャード一覧"""
        if not self.local_meta_dir:
            return []
        # WAV版とMIDI版両対応: LOCAL_WAV_META_DATA_*.pickle or LOCAL_META_DATA_*.pickle
        wav_shards = sorted(self.local_meta_dir.glob("LOCAL_WAV_META_DATA_*.pickle"))
        midi_shards = sorted(self.local_meta_dir.glob("LOCAL_META_DATA_*.pickle"))
        return wav_shards if wav_shards else midi_shards

    def load_meta(self) -> None:
        """META シャードの遅延ロード（origin、初回のみ）"""
        if self._meta or not self.meta_dir:
            return
        
        for shard in self._meta_shards():
            if not shard.exists():
                continue
            
            with open(shard, "rb") as f:
                data = pickle.load(f)
            
            # フォーマット: [(file_id, meta_dict), ...]
            for fid, meta in data:
                self._meta[str(fid)] = meta
    
    def load_local_meta(self) -> None:
        """LOCAL_META シャードの遅延ロード（初回のみ）"""
        if self._local_meta or not self.local_meta_dir:
            return
        
        for shard in self._local_meta_shards():
            if not shard.exists():
                continue
            
            with open(shard, "rb") as f:
                data = pickle.load(f)
            
            # フォーマット1（list）: [(file_id, meta_dict), ...]
            # フォーマット2（dict）: {file_id: meta_dict}
            if isinstance(data, list):
                for fid, meta in data:
                    self._local_meta[str(fid)] = meta
            elif isinstance(data, dict):
                for fid, meta in data.items():
                    self._local_meta[str(fid)] = meta

    def get_meta(self, file_id: str) -> Optional[Dict[str, Any]]:
        """指定file_idのMETAデータを取得（origin + local 両対応）
        
        Args:
            file_id: MIDIファイル名（拡張子なし）
        
        Returns:
            {
                "midi_patches": [prog1, prog2, ...],
                "total_patches_counts": {...},
                "groove_signature": {...},
                "controls_summary": {...},
                ...
            } or None
        """
        self.load_meta()
        self.load_local_meta()
        
        fid = self.id_map.get(file_id, file_id)
        
        # prefer_local による優先順
        first_dict = self._local_meta if self.prefer_local else self._meta
        second_dict = self._meta if self.prefer_local else self._local_meta
        
        if fid in first_dict:
            return first_dict[fid]
        if fid in second_dict:
            return second_dict[fid]
        return None

    # ========================================
    # SIGNATURES_DATA（timesig救済の裏取り）
    # ========================================
    def load_signatures(self) -> None:
        """SIGNATURES pickle の遅延ロード（origin、初回のみ）"""
        if self._sign is not None or not self.sign_path:
            return
        if not self.sign_path.exists():
            return
        
        with open(self.sign_path, "rb") as f:
            data = pickle.load(f)
        
        # フォーマット: [(file_id, [[sig_id, count], ...]), ...]
        self._sign = dict(data) if isinstance(data, (list, tuple)) else {}
    
    def load_local_signatures(self) -> None:
        """LOCAL_SIGNATURES pickle の遅延ロード（初回のみ）"""
        if self._local_sign is not None or not self.local_sign_path:
            return
        if not self.local_sign_path.exists():
            return
        
        with open(self.local_sign_path, "rb") as f:
            data = pickle.load(f)
        
        # フォーマット1（dict）: {file_id: ["4/4", "3/4", ...]}
        # フォーマット2（list）: [(file_id, [[sig_id, count], ...]), ...]
        if isinstance(data, dict):
            self._local_sign = data
        elif isinstance(data, (list, tuple)):
            self._local_sign = {}
            for rec in data:
                fid, payload = rec[0], rec[1]
                self._local_sign[str(fid)] = payload
        else:
            self._local_sign = {}

    def get_signatures(self, file_id: str) -> Optional[List[Tuple[int, int]]]:
        """指定file_idのSIGNATURESを取得（origin + local 両対応）
        
        Args:
            file_id: MIDIファイル名（拡張子なし）
        
        Returns:
            [[sig_id, count], ...] or None
            Example: [[155, 48], [211, 8]]  # 155=4/4が48小節、211=3/4が8小節
        """
        self.load_signatures()
        self.load_local_signatures()
        
        fid = self.id_map.get(file_id, file_id)
        
        # prefer_local による優先順
        first_dict = self._local_sign if self.prefer_local else self._sign
        second_dict = self._sign if self.prefer_local else self._local_sign
        
        if first_dict and fid in first_dict:
            return first_dict[fid]
        if second_dict and fid in second_dict:
            return second_dict[fid]
        return None

    # ========================================
    # TOTALS_MATRIX（pitch/dur/velの外れ値）
    # ========================================
    def load_totals(self) -> None:
        """TOTALS pickle の遅延ロード（origin、初回のみ）"""
        if self._tot is not None or not self.tot_path:
            return
        if not self.tot_path.exists():
            return
        
        with open(self.tot_path, "rb") as f:
            data = pickle.load(f)
        
        # フォーマット: {"pitch": [256], "dur": [256], "vel": [256], ...}
        self._tot = data if isinstance(data, dict) else {}
    
    def load_local_totals(self) -> None:
        """LOCAL_TOTALS pickle の遅延ロード（初回のみ）"""
        if self._local_tot is not None or not self.local_tot_path:
            return
        if not self.local_tot_path.exists():
            return
        
        with open(self.local_tot_path, "rb") as f:
            data = pickle.load(f)
        
        # フォーマット: {"format": "local_totals_v1", "pitch_hist_256": [...], ...}
        self._local_tot = data if isinstance(data, dict) else {}

    def get_totals(self) -> Optional[Dict[str, Any]]:
        """TOTALS全体を取得（pitch/dur/velの256bin histograms、origin + local 両対応）
        
        Returns:
            {
                "pitch": [256],  # MIDI note 0-127 × 2 (on/off)
                "dur": [256],    # duration bins
                "vel": [256],    # velocity bins
                ...
            } or None
        """
        self.load_totals()
        self.load_local_totals()
        
        # prefer_local による優先順（totalsは曲別でないため単一オブジェクトを返す）
        first = self._local_tot if self.prefer_local else self._tot
        second = self._tot if self.prefer_local else self._local_tot
        
        return first or second

    # ========================================
    # ユーティリティ
    # ========================================
    def has_kilo(self) -> bool:
        """KILO データが利用可能か（origin or local）"""
        return (
            (self.kilo_path is not None and self.kilo_path.exists()) or
            (self.local_kilo_path is not None and self.local_kilo_path.exists())
        )

    def has_meta(self) -> bool:
        """META データが利用可能か（origin or local）"""
        return (
            (self.meta_dir is not None and bool(self._meta_shards())) or
            (self.local_meta_dir is not None and bool(self._local_meta_shards()))
        )

    def has_signatures(self) -> bool:
        """SIGNATURES データが利用可能か（origin or local）"""
        return (
            (self.sign_path is not None and self.sign_path.exists()) or
            (self.local_sign_path is not None and self.local_sign_path.exists())
        )

    def has_totals(self) -> bool:
        """TOTALS データが利用可能か（origin or local）"""
        return (
            (self.tot_path is not None and self.tot_path.exists()) or
            (self.local_tot_path is not None and self.local_tot_path.exists())
        )

    def summary(self) -> Dict[str, bool]:
        """利用可能なデータソースの一覧"""
        return {
            "kilo": self.has_kilo(),
            "meta": self.has_meta(),
            "signatures": self.has_signatures(),
            "totals": self.has_totals(),
            "id_map": bool(self.id_map),
            "prefer_local": self.prefer_local
        }


# ========================================
# CLI テスト用
# ========================================
if __name__ == "__main__":
    import argparse
    import json
    
    ap = argparse.ArgumentParser(description="LAMDA Sources Test")
    ap.add_argument("--kilo", help="LAMDa_KILO_CHORDS_DATA.pickle")
    ap.add_argument("--meta-dir", help="META_DATA directory")
    ap.add_argument("--signatures", help="LAMDa_SIGNATURES_DATA.pickle")
    ap.add_argument("--totals", help="LAMDa_TOTALS.pickle")
    ap.add_argument("--id-map", help="auto_file_id_map.csv")
    ap.add_argument("--test-file-id", default="001", help="Test file_id")
    args = ap.parse_args()
    
    lamda = LamdaSources(
        kilo=args.kilo,
        meta_dir=args.meta_dir,
        signatures=args.signatures,
        totals=args.totals,
        id_map_csv=args.id_map
    )
    
    print("📊 LAMDA Sources Summary:")
    print(json.dumps(lamda.summary(), indent=2))
    
    print(f"\n🔍 Testing file_id: {args.test_file_id}")
    
    if lamda.has_kilo():
        chords = lamda.get_kilo_chords(args.test_file_id)
        print(f"  KILO chords: {len(chords) if chords else 0} events")
    
    if lamda.has_meta():
        meta = lamda.get_meta(args.test_file_id)
        print(f"  META data: {'✓' if meta else '✗'}")
        if meta:
            print(f"    patches: {len(meta.get('midi_patches', []))}")
    
    if lamda.has_signatures():
        sigs = lamda.get_signatures(args.test_file_id)
        print(f"  SIGNATURES: {sigs}")
    
    if lamda.has_totals():
        tots = lamda.get_totals()
        print(f"  TOTALS keys: {list(tots.keys()) if tots else []}")
