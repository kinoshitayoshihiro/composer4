"""
LAMDA META_DATA Extractor for Stage2
=====================================
META_DATAから7つのStage2拡張メタを抽出

Usage:
    extractor = LAMDAMetaExtractor(meta_data_path)
    meta = extractor.extract_stage2_meta(file_id)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class LAMDAMetaExtractor:
    """LAMDA META_DATAからStage2拡張メタを抽出"""
    
    # General MIDI Patch番号 → 楽器分類
    GM_INSTRUMENT_ROLES = {
        # Piano (0-7)
        **{i: "piano" for i in range(0, 8)},
        # Chromatic Percussion (8-15)
        **{i: "melodic" for i in range(8, 16)},
        # Organ (16-23)
        **{i: "organ" for i in range(16, 24)},
        # Guitar (24-31)
        **{i: "guitar" for i in range(24, 32)},
        # Bass (32-39)
        **{i: "bass" for i in range(32, 40)},
        # Strings (40-47)
        **{i: "strings" for i in range(40, 48)},
        # Ensemble (48-55)
        **{i: "ensemble" for i in range(48, 56)},
        # Brass (56-63)
        **{i: "brass" for i in range(56, 64)},
        # Reed (64-71)
        **{i: "reed" for i in range(64, 72)},
        # Pipe (72-79)
        **{i: "pipe" for i in range(72, 80)},
        # Synth Lead (80-87)
        **{i: "lead" for i in range(80, 88)},
        # Synth Pad (88-95)
        **{i: "pad" for i in range(88, 96)},
        # Synth Effects (96-103)
        **{i: "fx" for i in range(96, 104)},
        # Ethnic (104-111)
        **{i: "ethnic" for i in range(104, 112)},
        # Percussive (112-119)
        **{i: "perc" for i in range(112, 120)},
        # Sound Effects (120-127)
        **{i: "sfx" for i in range(120, 128)},
    }
    
    def __init__(self, meta_data_dir: Path):
        """
        Args:
            meta_data_dir: META_DATAディレクトリのパス
        """
        self.meta_data_dir = Path(meta_data_dir)
        self._meta_index: Dict[str, Tuple[int, int]] = {}  # file_id → (file_idx, entry_idx)
        self._meta_files: List[Path] = []
        self._load_index()
    
    def _load_index(self):
        """META_DATAのインデックスを構築"""
        meta_files = sorted(self.meta_data_dir.glob("LAMDa_META_DATA_*.pickle"))
        
        if not meta_files:
            logger.warning(f"No META_DATA files found in {self.meta_data_dir}")
            return
        
        logger.info(f"Indexing {len(meta_files)} META_DATA files...")
        
        for file_idx, meta_file in enumerate(meta_files):
            self._meta_files.append(meta_file)
            
            # ファイルを読み込んでインデックス構築
            data = pickle.load(open(meta_file, 'rb'))
            for entry_idx, entry in enumerate(data):
                file_id = entry[0]
                self._meta_index[file_id] = (file_idx, entry_idx)
        
        logger.info(f"Indexed {len(self._meta_index)} files")
    
    def _get_meta_entry(self, file_id: str) -> Optional[List]:
        """file_idからMETA_DATAエントリを取得"""
        if file_id not in self._meta_index:
            return None
        
        file_idx, entry_idx = self._meta_index[file_id]
        meta_file = self._meta_files[file_idx]
        
        # ファイルを読み込んで該当エントリを取得
        data = pickle.load(open(meta_file, 'rb'))
        return data[entry_idx][1]  # meta_items
    
    def extract_stage2_meta(self, file_id: str) -> Dict[str, Any]:
        """
        file_idから7つのStage2拡張メタを抽出
        
        Returns:
            {
                'key': str,
                'chord': Dict[str, int],
                'sections': List[Dict],
                'inst': Dict[str, List[int]],
                'groove': Dict[str, float],
                'controls': Dict[str, List],
                'tempo': Dict[str, float]
            }
        """
        meta_items = self._get_meta_entry(file_id)
        if not meta_items:
            return self._empty_stage2_meta()
        
        # メタデータをdict化
        meta_dict = self._parse_meta_items(meta_items)
        
        return {
            'key': self._extract_key(meta_dict),
            'chord': self._extract_chord_stats(meta_dict),
            'sections': self._extract_sections(meta_dict),
            'inst': self._extract_instrument_roles(meta_dict),
            'groove': self._extract_groove(meta_dict),
            'controls': self._extract_controls(meta_dict),
            'tempo': self._extract_tempo(meta_dict)
        }
    
    def _parse_meta_items(self, meta_items: List) -> Dict[str, Any]:
        """meta_itemsをdict形式に変換"""
        result = {}
        for item in meta_items:
            if isinstance(item, list) and len(item) >= 2:
                key = item[0]
                if isinstance(key, str):
                    # 統計データのみ格納（MIDIイベントは除外）
                    if not key.startswith(('note', 'control_change', 'patch_change', 
                                          'sequencer_specific', 'raw_meta_event', 
                                          'sysex_f0', 'text_event', 'track_name')):
                        result[key] = item[1] if len(item) == 2 else item[1:]
        return result
    
    def _extract_key(self, meta_dict: Dict) -> str:
        """調性を推定"""
        # key_signatureからC, C#, D, ... の推定
        key_sig = meta_dict.get('key_signature')
        if key_sig is not None and isinstance(key_sig, (list, tuple)) and len(key_sig) >= 2:
            # key_signature = [time, key, scale]
            # key: -7..7 (♭7..#7)
            # scale: 0=major, 1=minor
            key_value = key_sig[0] if isinstance(key_sig[0], int) else 0
            scale = key_sig[1] if len(key_sig) > 1 and isinstance(key_sig[1], int) else 0
            
            # 五度圏から調名を計算
            keys_major = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 
                         'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
            keys_minor = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#',
                         'D', 'G', 'C', 'F', 'Bb', 'Eb', 'Ab']
            
            if -7 <= key_value <= 7:
                idx = key_value + 7
                if scale == 0:  # major
                    return keys_major[idx % len(keys_major)]
                else:  # minor
                    return keys_minor[idx % len(keys_minor)] + 'm'
        
        # デフォルト: コード統計から推定
        ms_chords = meta_dict.get('ms_chords_counts', [])
        if ms_chords:
            # 最頻コードの根音から推定
            most_common = ms_chords[0][0] if ms_chords else []
            if most_common:
                root = most_common[0] % 12
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                return keys[root]
        
        return 'C'
    
    def _extract_chord_stats(self, meta_dict: Dict) -> Dict[str, int]:
        """コード統計を抽出"""
        return {
            'total': meta_dict.get('total_number_of_chords', 0),
            'unique': len(meta_dict.get('ms_chords_counts', [])),
            'density': meta_dict.get('total_number_of_chords_ms', 0)
        }
    
    def _extract_sections(self, meta_dict: Dict) -> List[Dict]:
        """セクション構造を推定（簡易版）"""
        total_ms = meta_dict.get('pitches_times_sum_ms', 0)
        if total_ms == 0:
            return []
        
        # 8小節ごとにセクション分割（簡易実装）
        # 実際はテンポやイベント密度から判断するべき
        num_sections = max(1, total_ms // 16000)  # 16秒 ≈ 8小節 @ 120BPM
        
        sections = []
        for i in range(min(num_sections, 8)):  # 最大8セクション
            sections.append({
                'start_ms': i * 16000,
                'end_ms': (i + 1) * 16000,
                'type': 'verse' if i % 2 == 0 else 'chorus'
            })
        
        return sections
    
    def _extract_instrument_roles(self, meta_dict: Dict) -> Dict[str, List[int]]:
        """楽器ロールを抽出"""
        patches = meta_dict.get('midi_patches', [])
        patch_counts = meta_dict.get('total_patches_counts', [])
        
        # パッチ番号をロールに分類
        roles: Dict[str, List[int]] = {
            'melody': [],
            'harmony': [],
            'bass': [],
            'drums': []
        }
        
        for patch_info in patch_counts:
            if len(patch_info) < 2:
                continue
            
            patch_num = patch_info[0]
            count = patch_info[1]
            
            # チャンネル10はドラム
            if patch_num == 128 or (patch_num >= 112 and patch_num <= 119):
                roles['drums'].append(patch_num)
            # ロール分類
            elif 32 <= patch_num <= 39:  # Bass
                roles['bass'].append(patch_num)
            elif patch_num in range(0, 8):  # Piano → harmony
                roles['harmony'].append(patch_num)
            elif patch_num in range(40, 56):  # Strings/Ensemble → harmony
                roles['harmony'].append(patch_num)
            else:  # その他 → melody
                roles['melody'].append(patch_num)
        
        return roles
    
    def _extract_groove(self, meta_dict: Dict) -> Dict[str, float]:
        """グルーヴ特性を推定"""
        avg_time = meta_dict.get('average_median_mode_time_ms', [0, 0, 0])
        avg_dur = meta_dict.get('average_median_mode_dur_ms', [0, 0, 0])
        avg_vel = meta_dict.get('average_median_mode_vel', [0, 0, 0])
        
        return {
            'swing': 0.0,  # TODO: タイミング偏差から計算
            'velocity_variation': float(avg_vel[0] - avg_vel[1]) / 127.0 if avg_vel[1] > 0 else 0.0,
            'note_density': float(avg_time[0]) / 1000.0 if avg_time[0] > 0 else 0.0,
            'average_duration': float(avg_dur[0]) / 1000.0 if avg_dur[0] > 0 else 0.0
        }
    
    def _extract_controls(self, meta_dict: Dict) -> Dict[str, List]:
        """コントロール変化を抽出（簡易版）"""
        # TODO: 実際のcontrol_changeイベントをパース
        return {
            'volume': [],
            'pan': [],
            'expression': [],
            'pedal': []
        }
    
    def _extract_tempo(self, meta_dict: Dict) -> Dict[str, float]:
        """テンポ情報を抽出"""
        set_tempo = meta_dict.get('set_tempo')
        tempo_changes = meta_dict.get('tempo_change_count', 0)
        
        # set_tempo = microseconds per quarter note
        bpm = 120.0  # default
        if isinstance(set_tempo, (list, tuple)) and len(set_tempo) >= 1:
            tempo_us = set_tempo[0] if isinstance(set_tempo[0], int) else 500000
            bpm = 60000000.0 / tempo_us if tempo_us > 0 else 120.0
        elif isinstance(set_tempo, int):
            bpm = 60000000.0 / set_tempo if set_tempo > 0 else 120.0
        
        return {
            'bpm': round(bpm, 2),
            'changes': tempo_changes,
            'stable': tempo_changes <= 1
        }
    
    def _empty_stage2_meta(self) -> Dict[str, Any]:
        """空のStage2メタを返す"""
        return {
            'key': 'C',
            'chord': {'total': 0, 'unique': 0, 'density': 0},
            'sections': [],
            'inst': {'melody': [], 'harmony': [], 'bass': [], 'drums': []},
            'groove': {'swing': 0.0, 'velocity_variation': 0.0, 
                      'note_density': 0.0, 'average_duration': 0.0},
            'controls': {'volume': [], 'pan': [], 'expression': [], 'pedal': []},
            'tempo': {'bpm': 120.0, 'changes': 0, 'stable': True}
        }


def test_extractor():
    """テスト実行"""
    import json
    
    meta_dir = Path("/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/Los-Angeles-MIDI/META_DATA")
    
    if not meta_dir.exists():
        print(f"❌ META_DATA directory not found: {meta_dir}")
        return
    
    print("🔍 Initializing LAMDA Meta Extractor...")
    extractor = LAMDAMetaExtractor(meta_dir)
    
    # テスト用file_id（最初のエントリ）
    test_file = meta_dir / "LAMDa_META_DATA_162000.pickle"
    data = pickle.load(open(test_file, 'rb'))
    test_file_id = data[0][0]
    
    print(f"📂 Testing with file_id: {test_file_id}")
    
    # Stage2メタ抽出
    stage2_meta = extractor.extract_stage2_meta(test_file_id)
    
    print("\n✅ Extracted Stage2 Meta:")
    print(json.dumps(stage2_meta, indent=2, ensure_ascii=False))
    
    # 統計情報
    print("\n📊 Statistics:")
    print(f"  Key: {stage2_meta['key']}")
    print(f"  Chords: {stage2_meta['chord']['total']} total, {stage2_meta['chord']['unique']} unique")
    print(f"  Tempo: {stage2_meta['tempo']['bpm']} BPM")
    print(f"  Instruments: {sum(len(v) for v in stage2_meta['inst'].values())} total")
    print(f"  Sections: {len(stage2_meta['sections'])} detected")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_extractor()
