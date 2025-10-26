"""
データ層3ランク管理スキーマ

GOLD / SILVER / BRONZE の品質階層化により、
自己蒸留の劣化を防ぎつつデータを効率的に拡張します。
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# データ品質ランク
DataTier = Literal["GOLD", "SILVER", "BRONZE"]


@dataclass
class ConfidenceScores:
    """信頼度スコア（0.0-1.0）"""
    chord: float = 0.0
    sections: float = 0.0
    roles: float = 0.0
    groove: float = 0.0
    controls: float = 0.0
    key: float = 0.0
    tempo: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @property
    def overall(self) -> float:
        """総合信頼度（平均）"""
        scores = [
            self.chord,
            self.sections,
            self.roles,
            self.groove,
            self.controls,
            self.key,
            self.tempo,
        ]
        return sum(scores) / len(scores)


@dataclass
class ProvenanceInfo:
    """データ出所情報"""
    source: Literal["suno", "lamda", "dawdreamer", "manual"]
    created_at: str  # ISO 8601
    seed: Optional[int] = None
    generator_version: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AudioStemPaths:
    """ステム音声ファイルパス"""
    mix: Optional[str] = None
    vocals: Optional[str] = None
    drums: Optional[str] = None
    bass: Optional[str] = None
    guitar: Optional[str] = None
    piano: Optional[str] = None
    strings: Optional[str] = None
    other: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Optional[str]]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Stage0Paths:
    """Stage0出力パス"""
    sections: Optional[str] = None
    chordmap: Optional[str] = None
    anchors: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Optional[str]]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TieredDataEntry:
    """階層化データエントリ"""
    # 基本情報
    id: str
    tier: DataTier
    
    # 音声ファイル
    audio: AudioStemPaths = field(default_factory=AudioStemPaths)
    
    # Stage0出力
    stage0: Stage0Paths = field(default_factory=Stage0Paths)
    
    # Stage2出力
    stage2_json: Optional[str] = None
    stage2_pkl: Optional[str] = None
    
    # メタデータ
    provenance: ProvenanceInfo = field(default_factory=lambda: ProvenanceInfo(
        source="manual", created_at=""
    ))
    confidence: Optional[ConfidenceScores] = None
    
    # カバレッジ属性
    key: Optional[str] = None
    tempo_bpm: Optional[float] = None
    time_signature: Optional[str] = None
    genre: Optional[str] = None
    emotion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """JSONL出力用"""
        d = {
            "id": self.id,
            "tier": self.tier,
            "audio": self.audio.to_dict(),
            "stage0": self.stage0.to_dict(),
        }
        
        if self.stage2_json:
            d["stage2_json"] = self.stage2_json
        if self.stage2_pkl:
            d["stage2_pkl"] = self.stage2_pkl
        
        d["provenance"] = self.provenance.to_dict()
        
        if self.confidence:
            d["confidence"] = self.confidence.to_dict()
        
        # カバレッジ属性
        for attr in ["key", "tempo_bpm", "time_signature", "genre", "emotion"]:
            val = getattr(self, attr, None)
            if val is not None:
                d[attr] = val
        
        return d
    
    def to_jsonl_line(self) -> str:
        """JSONL 1行"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class TieredDataManager:
    """階層化データ管理"""
    
    def __init__(self, jsonl_path: str):
        self.jsonl_path = Path(jsonl_path)
        self.entries: List[TieredDataEntry] = []
    
    def load(self):
        """JSONL読み込み"""
        if not self.jsonl_path.exists():
            return
        
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # 簡易復元（完全なデシリアライズは省略）
                entry = TieredDataEntry(
                    id=data["id"],
                    tier=data["tier"],
                )
                self.entries.append(entry)
    
    def save(self):
        """JSONL保存"""
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(entry.to_jsonl_line() + "\n")
    
    def add(self, entry: TieredDataEntry):
        """エントリ追加"""
        self.entries.append(entry)
    
    def filter_by_tier(self, tier: DataTier) -> List[TieredDataEntry]:
        """ランクでフィルタ"""
        return [e for e in self.entries if e.tier == tier]
    
    def filter_by_confidence(
        self, min_overall: float = 0.0, min_scores: Optional[Dict[str, float]] = None
    ) -> List[TieredDataEntry]:
        """信頼度でフィルタ"""
        result = []
        for e in self.entries:
            if not e.confidence:
                continue
            
            # 総合信頼度チェック
            if e.confidence.overall < min_overall:
                continue
            
            # 個別スコアチェック
            if min_scores:
                ok = True
                for key, threshold in min_scores.items():
                    if getattr(e.confidence, key, 0.0) < threshold:
                        ok = False
                        break
                if not ok:
                    continue
            
            result.append(e)
        
        return result
    
    def get_coverage_heatmap(self) -> Dict[str, Dict[str, int]]:
        """カバレッジヒートマップ（Key×Tempo×Genre...）"""
        heatmap: Dict[str, Dict[str, int]] = {
            "key": {},
            "tempo_bin": {},
            "time_signature": {},
            "genre": {},
            "emotion": {},
        }
        
        for e in self.entries:
            if e.key:
                heatmap["key"][e.key] = heatmap["key"].get(e.key, 0) + 1
            
            if e.tempo_bpm:
                # テンポビニング (例: 60-80, 80-100, ...)
                tempo_bin = f"{int(e.tempo_bpm // 20) * 20}-{int(e.tempo_bpm // 20 + 1) * 20}"
                heatmap["tempo_bin"][tempo_bin] = heatmap["tempo_bin"].get(tempo_bin, 0) + 1
            
            if e.time_signature:
                heatmap["time_signature"][e.time_signature] = (
                    heatmap["time_signature"].get(e.time_signature, 0) + 1
                )
            
            if e.genre:
                heatmap["genre"][e.genre] = heatmap["genre"].get(e.genre, 0) + 1
            
            if e.emotion:
                heatmap["emotion"][e.emotion] = heatmap["emotion"].get(e.emotion, 0) + 1
        
        return heatmap


# サンプリング方針のための格子定義
COVERAGE_GRID = {
    "keys": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
    "tempo_bins": ["60-80", "80-100", "100-120", "120-140", "140-160", "160-180"],
    "time_signatures": ["4/4", "3/4", "6/8"],
    "genres": [
        "rock",
        "jazz",
        "pop",
        "classical",
        "electronic",
        "hiphop",
        "metal",
        "funk",
        "blues",
        "country",
    ],
    "emotions": [
        "happy",
        "sad",
        "energetic",
        "calm",
        "angry",
        "romantic",
        "mysterious",
        "epic",
    ],
}


def calculate_coverage_score(manager: TieredDataManager) -> float:
    """
    カバレッジスコア算出（0.0-1.0）
    
    格子の全セル数に対する、データが存在するセル数の割合
    """
    heatmap = manager.get_coverage_heatmap()
    
    total_cells = (
        len(COVERAGE_GRID["keys"])
        * len(COVERAGE_GRID["tempo_bins"])
        * len(COVERAGE_GRID["time_signatures"])
        * len(COVERAGE_GRID["genres"])
        * len(COVERAGE_GRID["emotions"])
    )
    
    # 簡易版: 各次元での存在セル数の積
    filled = 1.0
    for dim in ["key", "tempo_bin", "time_signature", "genre", "emotion"]:
        filled *= len(heatmap.get(dim, {}))
    
    return min(1.0, filled / max(1, total_cells))


if __name__ == "__main__":
    # デモ
    manager = TieredDataManager("data/tiered_corpus.jsonl")
    
    # GOLD追加
    entry = TieredDataEntry(
        id="song_000001",
        tier="GOLD",
        audio=AudioStemPaths(
            mix="audio/song_000001/mix.wav",
            drums="audio/song_000001/drums.wav",
            bass="audio/song_000001/bass.wav",
        ),
        stage0=Stage0Paths(
            sections="stage0/song_000001/sections.json",
            chordmap="stage0/song_000001/chordmap.json",
        ),
        stage2_json="stage2/song_000001.stage2.json",
        provenance=ProvenanceInfo(
            source="suno",
            created_at="2025-10-23T12:00:00Z",
            seed=42,
        ),
        key="C",
        tempo_bpm=120.0,
        time_signature="4/4",
        genre="rock",
        emotion="energetic",
    )
    manager.add(entry)
    
    # SILVER追加
    silver = TieredDataEntry(
        id="song_lamda_123456",
        tier="SILVER",
        provenance=ProvenanceInfo(
            source="lamda",
            created_at="2025-10-23T13:00:00Z",
        ),
        confidence=ConfidenceScores(
            chord=0.93,
            sections=0.88,
            roles=0.90,
            groove=0.85,
            controls=0.80,
            key=0.95,
            tempo=0.92,
        ),
        key="Am",
        tempo_bpm=140.0,
        genre="jazz",
    )
    manager.add(silver)
    
    # 保存
    manager.save()
    
    print("✅ Created tiered corpus schema")
    print(f"   GOLD entries: {len(manager.filter_by_tier('GOLD'))}")
    print(f"   SILVER entries: {len(manager.filter_by_tier('SILVER'))}")
    print(f"\n📊 Coverage heatmap:")
    print(json.dumps(manager.get_coverage_heatmap(), indent=2))
