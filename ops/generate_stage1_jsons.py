#!/usr/bin/env python3
"""
generate_stage1_jsons.py - Suno AI WAV Stems → Stage1 JSON生成

Usage:
    python ops/generate_stage1_jsons.py \
        --input data/suno_ai/suno_themesong/song_001/stems \
        --output data/suno_ai/suno_themesong/song_001/analysis \
        --tempo 120

Output:
    - chordmap.json        # 独立した和声進行
    - sections.json        # セクション区間のみ
    - lyric_anchors.json   # 歌詞タイムライン（ボーカルから）
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import librosa
import numpy as np

# パス設定
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class Stage1JsonGenerator:
    """Stage1 JSON生成器（WAV → chordmap/sections/lyric_anchors）"""
    
    def __init__(self, tempo: float = 120.0, sr: int = 22050):
        self.tempo = tempo
        self.sr = sr
        self.beats_per_bar = 4  # 4/4拍子想定
    
    def generate_chordmap(
        self,
        stems_dir: Path,
        exclude_vocals: bool = True
    ) -> Dict[str, Any]:
        """
        ボーカル以外のステムから和声進行を推定
        
        Returns:
            {
                "tempo": 120.0,
                "time_signature": "4/4",
                "chords": [
                    {"bar": 0, "beat": 0, "offset_ql": 0.0, "root": "C", "quality": "maj", "bass": null, "tensions": []},
                    ...
                ]
            }
        """
        logger.info("Generating chordmap from stems...")
        
        # ステムファイル取得（ボーカル除外）
        stem_files = list(stems_dir.glob("*.wav"))
        if exclude_vocals:
            stem_files = [f for f in stem_files if "vocal" not in f.stem.lower()]
        
        logger.info(f"Found {len(stem_files)} non-vocal stems")
        
        # 和声推定（簡易版：全ステムのchroma集約）
        all_chromas = []
        for stem_file in stem_files:
            logger.info(f"  Analyzing: {stem_file.name}")
            y, sr = librosa.load(stem_file, sr=self.sr, mono=True)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
            all_chromas.append(chroma)
        
        if not all_chromas:
            logger.warning("No stems found, returning empty chordmap")
            return self._empty_chordmap()
        
        # Chroma集約（平均）
        mean_chroma = np.mean(all_chromas, axis=0)
        
        # フレーム → 小節境界にスナップ
        frames_per_beat = int(sr * 60 / self.tempo / 512)  # hop_length=512
        frames_per_bar = frames_per_beat * self.beats_per_bar
        
        chords = []
        num_bars = int(mean_chroma.shape[1] / frames_per_bar)
        
        for bar_idx in range(num_bars):
            start_frame = bar_idx * frames_per_bar
            end_frame = min((bar_idx + 1) * frames_per_bar, mean_chroma.shape[1])
            
            # 小節内のchroma平均
            bar_chroma = mean_chroma[:, start_frame:end_frame].mean(axis=1)
            
            # 最大chromaからroot推定（簡易）
            root_idx = int(np.argmax(bar_chroma))
            root_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            root = root_names[root_idx]
            
            # Major/Minor判定（3度の有無で簡易判定）
            major_third = (root_idx + 4) % 12
            minor_third = (root_idx + 3) % 12
            
            if bar_chroma[major_third] > bar_chroma[minor_third]:
                quality = "maj"
            else:
                quality = "min"
            
            chords.append({
                "bar": bar_idx,
                "beat": 0,
                "offset_ql": float(bar_idx * self.beats_per_bar),
                "root": root,
                "quality": quality,
                "bass": None,
                "tensions": []
            })
        
        logger.info(f"Generated {len(chords)} chords")
        
        return {
            "tempo": self.tempo,
            "time_signature": "4/4",
            "chords": chords
        }
    
    def generate_sections(
        self,
        stems_dir: Path,
        min_section_bars: int = 4
    ) -> Dict[str, Any]:
        """
        エネルギー変化からセクション境界を推定
        
        Returns:
            {
                "sections": [
                    {"label": "intro", "start_bar": 0, "end_bar": 4, "start_ql": 0.0, "end_ql": 16.0},
                    {"label": "verse", "start_bar": 4, "end_bar": 12, "start_ql": 16.0, "end_ql": 48.0},
                    ...
                ]
            }
        """
        logger.info("Generating sections from energy analysis...")
        
        # 全ステムのエネルギー合計
        stem_files = list(stems_dir.glob("*.wav"))
        if not stem_files:
            logger.warning("No stems found, returning single section")
            return self._single_section()
        
        # RMS energy計算
        all_rms = []
        for stem_file in stem_files:
            logger.info(f"  Analyzing: {stem_file.name}")
            y, sr = librosa.load(stem_file, sr=self.sr, mono=True)
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            all_rms.append(rms)
        
        # 平均エネルギー
        mean_rms = np.mean(all_rms, axis=0)
        
        # エネルギーの微分（変化点検出）
        rms_diff = np.diff(mean_rms)
        rms_diff_smooth = librosa.decompose.nn_filter(
            rms_diff.reshape(1, -1),
            aggregate=np.median,
            metric='cosine'
        ).flatten()
        
        # 変化点検出（閾値ベース）
        threshold = np.percentile(np.abs(rms_diff_smooth), 80)
        change_frames = np.where(np.abs(rms_diff_smooth) > threshold)[0]
        
        # フレーム → 小節変換
        frames_per_beat = int(sr * 60 / self.tempo / 512)
        frames_per_bar = frames_per_beat * self.beats_per_bar
        
        change_bars = [int(f / frames_per_bar) for f in change_frames]
        change_bars = sorted(set([0] + change_bars))  # 重複除去＋先頭追加
        
        # 最小セクション長で統合
        filtered_bars = [change_bars[0]]
        for bar in change_bars[1:]:
            if bar - filtered_bars[-1] >= min_section_bars:
                filtered_bars.append(bar)
        
        # 終端追加
        total_bars = int(len(mean_rms) / frames_per_bar)
        if filtered_bars[-1] < total_bars:
            filtered_bars.append(total_bars)
        
        # セクションラベル生成
        section_labels = self._generate_section_labels(len(filtered_bars) - 1)
        
        sections = []
        for i in range(len(filtered_bars) - 1):
            start_bar = filtered_bars[i]
            end_bar = filtered_bars[i + 1]
            
            sections.append({
                "label": section_labels[i],
                "start_bar": start_bar,
                "end_bar": end_bar,
                "start_ql": float(start_bar * self.beats_per_bar),
                "end_ql": float(end_bar * self.beats_per_bar)
            })
        
        logger.info(f"Generated {len(sections)} sections")
        
        return {"sections": sections}
    
    def generate_lyric_anchors(
        self,
        vocal_stem: Path
    ) -> Dict[str, Any]:
        """
        ボーカルステムから歌詞タイムライン推定
        
        Returns:
            {
                "anchors": [
                    {"time_ql": 0.0, "time_sec": 0.0, "token": "", "line_id": 0},
                    ...
                ]
            }
        """
        logger.info("Generating lyric anchors from vocal...")
        
        if not vocal_stem.exists():
            logger.warning(f"Vocal stem not found: {vocal_stem}")
            return {"anchors": []}
        
        # ボーカル読み込み
        y, sr = librosa.load(vocal_stem, sr=self.sr, mono=True)
        
        # Onset検出（子音/音節開始）
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=512,
            backtrack=True,
            units='frames'
        )
        
        # フレーム → 時間変換
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        
        # 時間 → Quarter Length変換
        onset_qls = onset_times * self.tempo / 60.0
        
        anchors = []
        for i, (time_sec, time_ql) in enumerate(zip(onset_times, onset_qls)):
            anchors.append({
                "time_ql": float(time_ql),
                "time_sec": float(time_sec),
                "token": "",  # 後で手動入力
                "line_id": i  # 仮ID
            })
        
        logger.info(f"Generated {len(anchors)} lyric anchors")
        
        return {"anchors": anchors}
    
    def _empty_chordmap(self) -> Dict[str, Any]:
        """空のchordmap"""
        return {
            "tempo": self.tempo,
            "time_signature": "4/4",
            "chords": []
        }
    
    def _single_section(self) -> Dict[str, Any]:
        """単一セクション（フォールバック）"""
        return {
            "sections": [{
                "label": "full",
                "start_bar": 0,
                "end_bar": 16,
                "start_ql": 0.0,
                "end_ql": 64.0
            }]
        }
    
    def _generate_section_labels(self, count: int) -> List[str]:
        """セクションラベル生成（パターンベース）"""
        # 典型的なポップス構成
        patterns = [
            "intro", "verse", "pre-chorus", "chorus",
            "verse", "pre-chorus", "chorus",
            "bridge", "chorus", "outro"
        ]
        
        if count <= len(patterns):
            return patterns[:count]
        else:
            # 不足分は section_N で埋める
            return patterns + [f"section_{i}" for i in range(count - len(patterns))]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage1 JSONs from Suno AI WAV stems"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Input stems directory"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output JSON directory"
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=120.0,
        help="Tempo in BPM (default: 120)"
    )
    parser.add_argument(
        "--vocal-stem",
        type=str,
        default="vocals.wav",
        help="Vocal stem filename (default: vocals.wav)"
    )
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Generator初期化
    generator = Stage1JsonGenerator(tempo=args.tempo)
    
    # 1. chordmap.json生成
    logger.info("\n" + "="*60)
    logger.info("Step 1: Generating chordmap.json")
    logger.info("="*60)
    chordmap = generator.generate_chordmap(args.input)
    chordmap_path = args.output / "chordmap.json"
    with open(chordmap_path, 'w', encoding='utf-8') as f:
        json.dump(chordmap, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved: {chordmap_path}")
    
    # 2. sections.json生成
    logger.info("\n" + "="*60)
    logger.info("Step 2: Generating sections.json")
    logger.info("="*60)
    sections = generator.generate_sections(args.input)
    sections_path = args.output / "sections.json"
    with open(sections_path, 'w', encoding='utf-8') as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved: {sections_path}")
    
    # 3. lyric_anchors.json生成
    logger.info("\n" + "="*60)
    logger.info("Step 3: Generating lyric_anchors.json")
    logger.info("="*60)
    vocal_path = args.input / args.vocal_stem
    lyric_anchors = generator.generate_lyric_anchors(vocal_path)
    anchors_path = args.output / "lyric_anchors.json"
    with open(anchors_path, 'w', encoding='utf-8') as f:
        json.dump(lyric_anchors, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved: {anchors_path}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ All Stage1 JSONs generated successfully!")
    logger.info("="*60)
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review and edit: {chordmap_path}")
    logger.info(f"2. Review and edit: {sections_path}")
    logger.info(f"3. Add lyrics to: {anchors_path}")
    logger.info(f"\nThen run modular_composer with these files.")


if __name__ == "__main__":
    main()
