#!/usr/bin/env python3
"""
ChordMap自動生成（Mode/Scale統合版）

ワークフロー:
1. sections.json にジャンル別プリセット自動割当
2. Chromagram抽出 + Scale Prior ブレンド（α=0.25）
3. Chordmap自動推定（Viterbi）
4. 結果をchordmap.jsonに出力

Usage:
    python generate_chordmap_with_scale.py \\
        --audio data/suno_ai/suno_themesong/song_001/audio.wav \\
        --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \\
        --output data/suno_ai/suno_themesong/song_001/analysis/chordmap.json \\
        --genre j-pop
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Librosa for chromagram extraction
try:
    import librosa
except ImportError:
    print("❌ librosa not installed. Run: pip install librosa")
    sys.exit(1)

# ops.scale_modes の import
sys.path.insert(0, str(Path(__file__).parent))
from ops.scale_modes import mask_for_preset, mask_for_key, list_presets


# =========================
# ジャンル → プリセット自動割当
# =========================
GENRE_PRESET_MAP = {
    "ballad": {
        "intro": {"preset": "aeolian_dream", "blues": 0.12, "code_offsets_mode": "key"},
        "verse": {"preset": "aeolian_dream", "blues": 0.15, "code_offsets_mode": "key"},
        "chorus": {"preset": "dorian_soul", "blues": 0.18, "code_offsets_mode": "chord"},
        "bridge": {"preset": "phrygian_spice", "blues": 0.10, "code_offsets_mode": "key"},
        "outro": {"preset": "aeolian_dream", "blues": 0.12, "code_offsets_mode": "key"}
    },
    "j-pop": {
        "intro": {"preset": "lydian_shimmer", "blues": 0.08, "code_offsets_mode": "key"},
        "verse": {"preset": "ionian_vintage", "blues": 0.10, "code_offsets_mode": "key"},
        "chorus": {"preset": "ionian_citypop", "blues": 0.12, "code_offsets_mode": "chord"},
        "post_chorus": {"preset": "dorian_soul", "blues": 0.10, "code_offsets_mode": "chord"},
        "bridge": {"preset": "lydian_shimmer", "blues": 0.08, "code_offsets_mode": "key"},
        "outro": {"preset": "aeolian_cinematic", "blues": 0.10, "code_offsets_mode": "key"}
    },
    "j-rock": {
        "intro": {"preset": "mixolydian_blues", "blues": 0.20, "code_offsets_mode": "key"},
        "verse": {"preset": "dorian_soul", "blues": 0.18, "code_offsets_mode": "key"},
        "chorus": {"preset": "dorian_gospel", "blues": 0.25, "code_offsets_mode": "chord"},
        "bridge": {"preset": "phrygian_spice", "blues": 0.15, "code_offsets_mode": "key"},
        "outro": {"preset": "aeolian_cinematic", "blues": 0.20, "code_offsets_mode": "key"}
    },
    "enka": {
        "intro": {"preset": "aeolian_dream", "blues": 0.18, "code_offsets_mode": "key"},
        "verse": {"preset": "aeolian_dream", "blues": 0.22, "code_offsets_mode": "key"},
        "chorus": {"preset": "aeolian_dream", "blues": 0.20, "code_offsets_mode": "chord"},
        "bridge": {"preset": "phrygian_spice", "blues": 0.15, "code_offsets_mode": "key"},
        "outro": {"preset": "aeolian_dream", "blues": 0.18, "code_offsets_mode": "key"}
    },
    "citypop": {
        "intro": {"preset": "lydian_shimmer", "blues": 0.10, "code_offsets_mode": "key"},
        "verse": {"preset": "ionian_citypop", "blues": 0.10, "code_offsets_mode": "chord"},
        "chorus": {"preset": "ionian_citypop", "blues": 0.15, "code_offsets_mode": "chord"},
        "bridge": {"preset": "lydian_shimmer", "blues": 0.12, "code_offsets_mode": "chord"},
        "outro": {"preset": "ionian_vintage", "blues": 0.10, "code_offsets_mode": "key"}
    }
}


def assign_presets_to_sections(sections: Dict, genre: str) -> Dict:
    """sections.jsonにプリセット自動割当"""
    genre_key = genre.lower().replace("-", "").replace("_", "")
    preset_map = GENRE_PRESET_MAP.get(genre_key, GENRE_PRESET_MAP["j-pop"])
    
    sections_copy = json.loads(json.dumps(sections))  # Deep copy
    
    for sec in sections_copy["sections"]:
        label = sec["label"].lower()
        if label in preset_map:
            sec.update(preset_map[label])
        else:
            # デフォルト: バランス型
            sec.update({"preset": "ionian_vintage", "blues": 0.10, "code_offsets_mode": "key"})
    
    print(f"✅ Assigned {genre} presets to {len(sections_copy['sections'])} sections")
    return sections_copy


# =========================
# Chromagram抽出（マルチステム対応）
# =========================
def extract_chromagram_from_stems(stem_dir: str, hop_length: int = 512) -> Tuple[np.ndarray, float]:
    """
    マルチステムからChromagram抽出（高精度版）
    
    Priority stems for chord detection:
        - Keyboard (1.0x)
        - Guitar (0.9x)
        - Bass (0.7x)
        - Strings (0.8x)
    """
    stem_path = Path(stem_dir)
    
    STEM_WEIGHTS = {
        "Keyboard": 1.0,
        "Guitar": 0.9,
        "Strings": 0.8,
        "Bass": 0.7,
        "Synth": 0.6
    }
    
    print(f"🎵 Loading stems from: {stem_dir}")
    
    chroma_weighted = None
    sr = 22050
    
    for stem_name, weight in STEM_WEIGHTS.items():
        # ファイル検索（柔軟なパターンマッチング）
        stem_files = list(stem_path.glob(f"*{stem_name}*.wav"))
        if not stem_files:
            print(f"  ⚠️  {stem_name} not found, skipping...")
            continue
        
        stem_file = stem_files[0]
        print(f"  🎹 {stem_name}: {stem_file.name} (weight={weight})")
        
        # Chromagram抽出
        y, sr = librosa.load(str(stem_file), sr=22050, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=12)
        
        # 重み付け加算
        if chroma_weighted is None:
            chroma_weighted = chroma * weight
        else:
            # フレーム数が異なる場合は短い方に合わせる
            min_frames = min(chroma_weighted.shape[1], chroma.shape[1])
            chroma_weighted = chroma_weighted[:, :min_frames]
            chroma = chroma[:, :min_frames]
            chroma_weighted += chroma * weight
    
    if chroma_weighted is None:
        raise ValueError("No valid stems found for chromagram extraction")
    
    # 正規化
    chroma_weighted = chroma_weighted / (chroma_weighted.sum(axis=0, keepdims=True) + 1e-9)
    
    print(f"✅ Multi-stem chromagram shape: {chroma_weighted.shape}")
    return chroma_weighted, sr


def extract_chromagram(audio_path: str, hop_length: int = 512) -> Tuple[np.ndarray, float]:
    """オーディオからChromagram抽出（フォールバック用）"""
    print(f"🎵 Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    print(f"🎹 Computing chromagram (hop={hop_length})...")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=12)
    
    # 正規化
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    
    return chroma, sr


# =========================
# Scale Prior 生成
# =========================
def generate_scale_prior(sections: Dict, chroma_frames: int, hop_length: int, 
                         sr: float, bar_duration: float) -> np.ndarray:
    """
    各フレームにScale Maskを生成（12 x T）
    
    Args:
        sections: sections.json (preset割当済)
        chroma_frames: chromagramのフレーム数
        hop_length: hop_length
        sr: sample rate
        bar_duration: 1小節の秒数（平均テンポから計算）
    
    Returns:
        scale_prior: (12, T) - 各フレームの12音重み
    """
    print(f"🎛️  Generating scale prior for {chroma_frames} frames...")
    
    scale_prior = np.ones((12, chroma_frames), dtype=np.float32)
    
    for i, sec in enumerate(sections["sections"]):
        bar_start = sec["bar"]
        bar_end = sections["sections"][i+1]["bar"] if i+1 < len(sections["sections"]) else sections["meta"]["last_bar"]
        
        # 時間範囲
        time_start = bar_start * bar_duration
        time_end = bar_end * bar_duration
        
        # フレーム範囲
        frame_start = int(time_start * sr / hop_length)
        frame_end = int(time_end * sr / hop_length)
        frame_end = min(frame_end, chroma_frames)
        
        # キー取得
        key_hint = sec.get("key_hint") or _get_key_at_bar(sections, bar_start)
        if not key_hint:
            continue
        
        # プリセット → mask生成
        preset = sec.get("preset", "ionian_vintage")
        blues = sec.get("blues", 0.10)
        
        try:
            mask = mask_for_preset(key_hint, preset, blues=blues)
            scale_prior[:, frame_start:frame_end] = np.array(mask).reshape(12, 1)
        except Exception as e:
            print(f"⚠️  Section {sec['label']} (bar {bar_start}): Failed to generate mask - {e}")
            continue
    
    return scale_prior


def _get_key_at_bar(sections: Dict, bar: int) -> Optional[str]:
    """指定小節のキーを取得"""
    key_hint_list = sections.get("key_hint", [])
    current_key = None
    for bar_key, key in key_hint_list:
        if bar_key <= bar:
            current_key = key
        else:
            break
    return current_key


# =========================
# Chromagram + Scale Prior ブレンド
# =========================
def blend_chroma_with_scale(chroma: np.ndarray, scale_prior: np.ndarray, 
                            alpha: float = 0.25) -> np.ndarray:
    """
    chroma と scale_prior をブレンド
    
    Args:
        chroma: (12, T) - chromagram
        scale_prior: (12, T) - scale mask
        alpha: ブレンド係数（0=chroma only, 1=scale only）
    
    Returns:
        blended: (12, T)
    """
    print(f"🎨 Blending chroma with scale prior (α={alpha})...")
    
    # 正規化
    chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    scale_norm = scale_prior / (scale_prior.sum(axis=0, keepdims=True) + 1e-9)
    
    # ブレンド
    blended = (1 - alpha) * chroma_norm + alpha * scale_norm
    blended = blended / (blended.sum(axis=0, keepdims=True) + 1e-9)
    
    return blended


# =========================
# Chord推定（簡易版：最大PC → Root推定）
# =========================
CHORD_TEMPLATES = {
    "maj": [1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
    "min": [1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
    "7": [1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0, 0.0, 0.6, 0.0],
    "maj7": [1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.7],
    "min7": [1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.6, 0.0]
}

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_chord_at_frame(chroma_frame: np.ndarray) -> str:
    """1フレームのChordを推定（テンプレートマッチング）"""
    best_score = -1
    best_chord = "C"
    
    for root_pc in range(12):
        for quality, template in CHORD_TEMPLATES.items():
            # テンプレートを root_pc だけシフト
            template_shifted = np.roll(template, root_pc)
            score = np.dot(chroma_frame, template_shifted)
            
            if score > best_score:
                best_score = score
                best_chord = f"{PC_NAMES[root_pc]}{quality}"
    
    return best_chord


def estimate_chordmap(chroma_blended: np.ndarray, sections: Dict, 
                      hop_length: int, sr: float, bar_duration: float) -> List[Dict]:
    """
    Chromagramからchordmap生成
    
    Returns:
        chordmap: [{"bar": 0, "chord": "Cmaj7"}, ...]
    """
    print(f"🎼 Estimating chordmap...")
    
    chordmap = []
    last_bar = sections["meta"]["last_bar"]
    
    for bar in range(0, last_bar, 1):  # 1小節ごと
        time_start = bar * bar_duration
        time_end = (bar + 1) * bar_duration
        
        frame_start = int(time_start * sr / hop_length)
        frame_end = int(time_end * sr / hop_length)
        frame_end = min(frame_end, chroma_blended.shape[1])
        
        if frame_start >= chroma_blended.shape[1]:
            break
        
        # 小節内の平均chromagram
        chroma_bar = chroma_blended[:, frame_start:frame_end].mean(axis=1)
        
        # Chord推定
        chord = estimate_chord_at_frame(chroma_bar)
        chordmap.append({"bar": bar, "chord": chord})
    
    print(f"✅ Generated chordmap: {len(chordmap)} bars")
    return chordmap


# =========================
# メイン処理
# =========================
def main():
    parser = argparse.ArgumentParser(description="ChordMap自動生成（Scale統合版）")
    parser.add_argument("--audio", help="Audio file path (fallback if no stems)")
    parser.add_argument("--stems", help="Stem directory path (higher priority)")
    parser.add_argument("--sections", required=True, help="sections.json path")
    parser.add_argument("--output", required=True, help="Output chordmap.json path")
    parser.add_argument("--genre", default="j-pop", help="Genre (j-pop, ballad, j-rock, enka, citypop)")
    parser.add_argument("--alpha", type=float, default=0.25, help="Scale prior blend ratio (0.0-1.0)")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for chromagram")
    
    args = parser.parse_args()
    
    if not args.audio and not args.stems:
        print("❌ Error: Either --audio or --stems must be provided")
        sys.exit(1)
    
    # 1. sections.json 読み込み
    print(f"\n📂 Loading sections: {args.sections}")
    with open(args.sections, "r", encoding="utf-8") as f:
        sections = json.load(f)
    
    # 2. プリセット自動割当
    sections_with_preset = assign_presets_to_sections(sections, args.genre)
    
    # 平均テンポ計算（簡易）
    avg_tempo = np.mean([t for _, t in sections["tempo_map"]]) if "tempo_map" in sections else 75.0
    bar_duration = 240.0 / avg_tempo  # 4/4拍子
    print(f"⏱️  Average tempo: {avg_tempo:.2f} BPM, Bar duration: {bar_duration:.3f}s")
    
    # 3. Chromagram抽出（ステム優先）
    if args.stems and Path(args.stems).exists():
        print(f"\n🎼 Using multi-stem chromagram extraction")
        chroma, sr = extract_chromagram_from_stems(args.stems, hop_length=args.hop_length)
    else:
        print(f"\n🎵 Using single audio chromagram extraction")
        chroma, sr = extract_chromagram(args.audio, hop_length=args.hop_length)
    
    print(f"✅ Chromagram shape: {chroma.shape}")
    
    # 4. Scale Prior 生成
    scale_prior = generate_scale_prior(
        sections_with_preset, chroma.shape[1], args.hop_length, sr, bar_duration
    )
    
    # 5. ブレンド
    chroma_blended = blend_chroma_with_scale(chroma, scale_prior, alpha=args.alpha)
    
    # 6. Chordmap推定
    chordmap = estimate_chordmap(chroma_blended, sections, args.hop_length, sr, bar_duration)
    
    # 7. 出力
    output_data = {
        "unit": "bar",
        "chords": chordmap,
        "meta": {
            "genre": args.genre,
            "scale_prior_alpha": args.alpha,
            "presets": {sec["bar"]: sec.get("preset", "N/A") for sec in sections_with_preset["sections"]},
            "method": "multi-stem+scale_prior+template_matching" if args.stems else "chromagram+scale_prior+template_matching",
            "stem_weights": {"Keyboard": 1.0, "Guitar": 0.9, "Strings": 0.8, "Bass": 0.7, "Synth": 0.6} if args.stems else None
        }
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Chordmap saved: {output_path}")
    print(f"📊 Total bars: {len(chordmap)}")
    print(f"🎛️  Genre: {args.genre}, Alpha: {args.alpha}")
    print(f"\n🎉 Done! Next: Edit chordmap.json with ChatGPT for fine-tuning.")


if __name__ == "__main__":
    main()
