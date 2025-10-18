#!/usr/bin/env python3
"""
Provenance生成ユーティリティ

生成されたMIDI/WAVファイルに隣接するprovenance.jsonを作成し、
生成時刻・入力YAML・パターンID・SF2・seed・Git SHA等を記録します。

使用方法:
    from scripts.provenance import ProvenanceTracker
    
    tracker = ProvenanceTracker()
    tracker.record_midi_generation(
        output_path="out/midi/guitar_strum_chorus.mid",
        structure_yaml="project/song.yaml",
        instrument="guitar",
        technique="strum",
        pattern_id="stage2_guitar_708_strum_120bpm",
        seed=42
    )
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib


class ProvenanceTracker:
    """生成物の系譜情報を記録"""
    
    def __init__(self, root_dir: Optional[Path] = None):
        """
        Args:
            root_dir: プロジェクトルートディレクトリ（Git SHA取得用）
        """
        self.root_dir = root_dir or Path(__file__).parent.parent
    
    def get_git_sha(self) -> str:
        """現在のGitコミットSHAを取得"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "<not-in-git>"
    
    def get_git_branch(self) -> str:
        """現在のGitブランチ名を取得"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "<unknown>"
    
    def compute_file_hash(self, filepath: Path) -> str:
        """ファイルのSHA1ハッシュを計算"""
        if not filepath.exists():
            return "<not-generated>"
        
        sha1 = hashlib.sha1()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()[:16]  # 先頭16文字のみ
    
    def record_midi_generation(
        self,
        output_path: Path,
        structure_yaml: Optional[Path] = None,
        instrument: Optional[str] = None,
        technique: Optional[str] = None,
        section: Optional[str] = None,
        pattern_id: Optional[str] = None,
        seed: Optional[int] = None,
        faithfulness: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        MIDI生成の系譜情報を記録
        
        Args:
            output_path: 出力MIDIファイルパス
            structure_yaml: 入力構造YAMLファイル
            instrument: 楽器名
            technique: 奏法名
            section: セクション名
            pattern_id: 使用したパターンID
            seed: 乱数シード
            faithfulness: 原曲忠実度
            extra_metadata: 追加メタデータ
        
        Returns:
            生成されたprovenance.jsonのパス
        """
        output_path = Path(output_path)
        provenance_path = output_path.with_suffix('.provenance.json')
        
        provenance = {
            "type": "midi_generation",
            "timestamp": datetime.now().isoformat(),
            "git": {
                "commit": self.get_git_sha(),
                "branch": self.get_git_branch()
            },
            "output": {
                "path": str(output_path),
                "hash": self.compute_file_hash(output_path),
                "size_bytes": output_path.stat().st_size if output_path.exists() else 0
            },
            "inputs": {},
            "parameters": {},
            "metadata": {}
        }
        
        # 入力情報
        if structure_yaml:
            structure_yaml = Path(structure_yaml)
            provenance["inputs"]["structure_yaml"] = {
                "path": str(structure_yaml),
                "hash": self.compute_file_hash(structure_yaml)
            }
        
        # パラメータ
        if instrument:
            provenance["parameters"]["instrument"] = instrument
        if technique:
            provenance["parameters"]["technique"] = technique
        if section:
            provenance["parameters"]["section"] = section
        if pattern_id:
            provenance["parameters"]["pattern_id"] = pattern_id
        if seed is not None:
            provenance["parameters"]["seed"] = seed
        if faithfulness is not None:
            provenance["parameters"]["faithfulness"] = faithfulness
        
        # 追加メタデータ
        if extra_metadata:
            provenance["metadata"].update(extra_metadata)
        
        # 保存
        with open(provenance_path, 'w', encoding='utf-8') as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)
        
        return provenance_path
    
    def record_audio_rendering(
        self,
        output_path: Path,
        midi_path: Optional[Path] = None,
        soundfont_path: Optional[Path] = None,
        sample_rate: int = 44100,
        normalize_db: float = -1.0,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        WAV生成の系譜情報を記録
        
        Args:
            output_path: 出力WAVファイルパス
            midi_path: 入力MIDIファイル
            soundfont_path: 使用したSoundFont
            sample_rate: サンプルレート
            normalize_db: 正規化レベル
            extra_metadata: 追加メタデータ
        
        Returns:
            生成されたprovenance.jsonのパス
        """
        output_path = Path(output_path)
        provenance_path = output_path.with_suffix('.provenance.json')
        
        provenance = {
            "type": "audio_rendering",
            "timestamp": datetime.now().isoformat(),
            "git": {
                "commit": self.get_git_sha(),
                "branch": self.get_git_branch()
            },
            "output": {
                "path": str(output_path),
                "hash": self.compute_file_hash(output_path),
                "size_bytes": output_path.stat().st_size if output_path.exists() else 0
            },
            "inputs": {},
            "parameters": {
                "sample_rate": sample_rate,
                "normalize_db": normalize_db
            },
            "metadata": {}
        }
        
        # 入力情報
        if midi_path:
            midi_path = Path(midi_path)
            provenance["inputs"]["midi"] = {
                "path": str(midi_path),
                "hash": self.compute_file_hash(midi_path)
            }
            
            # MIDI付随のprovenanceも参照
            midi_prov_path = midi_path.with_suffix('.provenance.json')
            if midi_prov_path.exists():
                provenance["inputs"]["midi_provenance"] = str(midi_prov_path)
        
        if soundfont_path:
            soundfont_path = Path(soundfont_path)
            provenance["inputs"]["soundfont"] = {
                "path": str(soundfont_path),
                "hash": self.compute_file_hash(soundfont_path)
            }
        
        # 追加メタデータ
        if extra_metadata:
            provenance["metadata"].update(extra_metadata)
        
        # 保存
        with open(provenance_path, 'w', encoding='utf-8') as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)
        
        return provenance_path
    
    def record_structure_extraction(
        self,
        output_yaml: Path,
        vocal_path: Optional[Path] = None,
        accompaniment_path: Optional[Path] = None,
        methods: Optional[list] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        構造抽出の系譜情報を記録
        
        Args:
            output_yaml: 出力構造YAML
            vocal_path: ボーカルstem
            accompaniment_path: 伴奏stem
            methods: 使用した抽出メソッド
            extra_metadata: 追加メタデータ
        
        Returns:
            生成されたprovenance.jsonのパス
        """
        output_yaml = Path(output_yaml)
        provenance_path = output_yaml.with_suffix('.provenance.json')
        
        provenance = {
            "type": "structure_extraction",
            "timestamp": datetime.now().isoformat(),
            "git": {
                "commit": self.get_git_sha(),
                "branch": self.get_git_branch()
            },
            "output": {
                "path": str(output_yaml),
                "hash": self.compute_file_hash(output_yaml)
            },
            "inputs": {},
            "parameters": {},
            "metadata": {}
        }
        
        # 入力情報
        if vocal_path:
            vocal_path = Path(vocal_path)
            provenance["inputs"]["vocal"] = {
                "path": str(vocal_path),
                "hash": self.compute_file_hash(vocal_path)
            }
        
        if accompaniment_path:
            accompaniment_path = Path(accompaniment_path)
            provenance["inputs"]["accompaniment"] = {
                "path": str(accompaniment_path),
                "hash": self.compute_file_hash(accompaniment_path)
            }
        
        # パラメータ
        if methods:
            provenance["parameters"]["extraction_methods"] = methods
        
        # 追加メタデータ
        if extra_metadata:
            provenance["metadata"].update(extra_metadata)
        
        # 保存
        with open(provenance_path, 'w', encoding='utf-8') as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)
        
        return provenance_path


# CLI実行例
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/provenance.py <output_file>")
        sys.exit(1)
    
    tracker = ProvenanceTracker()
    output_file = Path(sys.argv[1])
    
    # デモ: MIDI生成の記録
    if output_file.suffix == '.mid':
        prov_path = tracker.record_midi_generation(
            output_path=output_file,
            instrument="guitar",
            technique="strum",
            seed=42
        )
        print(f"✅ Provenance recorded: {prov_path}")
    
    # デモ: WAV生成の記録
    elif output_file.suffix == '.wav':
        prov_path = tracker.record_audio_rendering(
            output_path=output_file,
            sample_rate=44100,
            normalize_db=-1.0
        )
        print(f"✅ Provenance recorded: {prov_path}")
    
    else:
        print(f"❌ Unsupported file type: {output_file.suffix}")
        sys.exit(1)
