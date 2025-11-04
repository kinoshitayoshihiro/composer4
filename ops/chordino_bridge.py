#!/usr/bin/env python3
"""
Chordino/Essentia系Chroma抽出Bridge（安全なフォールバック付き）

Essentia HPCP優先、未導入環境ではlibrosa CQTに自動フォールバック。
既存パイプラインには影響なし（probe_onlyモード）。

使用例:
    # Essentia優先（自動フォールバック）
    python ops/chordino_bridge.py \\
        --audio song_packages/suno_project/song_001/full.wav \\
        --out song_packages/suno_project/song_001/chroma_probe.json
    
    # librosa強制
    python ops/chordino_bridge.py \\
        --audio song_packages/suno_project/song_001/full.wav \\
        --out song_packages/suno_project/song_001/chroma_probe.json \\
        --no-essentia

研究背景:
    - NNLS-Chroma系（Chordino論文）の実務実装
    - Essentia: MIR研究の総合フレームワーク（HPCP = Harmonic Pitch Class Profile）
    - ISMIR/MIREX標準のChroma特徴抽出手法
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _fallback_librosa_chroma(audio_path: Path, sr: int = 44100) -> Dict[str, Any]:
    """librosa CQT Chromaフォールバック（Essentia未導入時）"""
    try:
        import numpy as np
        import librosa
        
        logger.info(f"🔧 Using librosa CQT chroma (fallback mode)")
        
        y, _sr = librosa.load(str(audio_path), sr=sr, mono=True)
        
        # CQT-based Chroma（音楽的に優れた周波数解像度）
        chroma = librosa.feature.chroma_cqt(y=y, sr=_sr, hop_length=512)
        times = librosa.times_like(chroma, sr=_sr, hop_length=512)
        
        return {
            "backend": "librosa_cqt",
            "sample_rate": _sr,
            "hop_length": 512,
            "n_frames": chroma.shape[1],
            "times": times.tolist(),
            "chroma": chroma.T.tolist()  # (n_frames, 12)
        }
    except Exception as e:
        logger.error(f"❌ librosa chroma failed: {e}")
        raise


def _essentia_hpcp(audio_path: Path, sr: int = 44100) -> Dict[str, Any]:
    """Essentia HPCP（Harmonic Pitch Class Profile）
    
    NNLS-Chroma系の実装標準。Chordino VampプラグインのベースとなるEssentiaアルゴリズム。
    
    特徴:
        - 倍音成分を考慮した12次元Chroma（harmonics=4）
        - バンドプリセット最適化（楽器音に適したピーク検出）
        - MIREX Chord Detection推奨手法
    """
    try:
        import numpy as np
        import essentia.standard as es
        
        logger.info(f"✨ Using Essentia HPCP (Chordino-compatible)")
        
        # MonoLoader（Essentiaの標準オーディオローダー）
        y = es.MonoLoader(filename=str(audio_path), sampleRate=sr)()
        
        # フレーム分割パラメータ
        frame_size = 4096  # スペクトル解像度
        hop = 512          # 時間解像度（librosaと揃える）
        
        # Essentia標準処理チェーン
        w = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        
        # HPCP抽出（Chordinoの核心アルゴリズム）
        hpcp = es.HPCP(
            size=12,              # 12次元Chroma
            harmonics=4,          # 第4倍音まで考慮
            bandPreset=True,      # 楽器音最適化
            minFrequency=40.0,    # 低音域下限
            maxFrequency=5000.0,  # 高音域上限
            splitFrequency=500.0  # バンド分割点
        )
        
        # フレームごとにHPCP計算
        frames = es.FrameGenerator(
            y, 
            frameSize=frame_size, 
            hopSize=hop, 
            startFromZero=True
        )
        
        chroma = []
        for f in frames:
            spec = spectrum(w(f))
            chroma.append(hpcp(spec))
        
        # 時間軸生成
        times = [i * hop / sr for i in range(len(chroma))]
        
        return {
            "backend": "essentia_hpcp",
            "sample_rate": sr,
            "hop_length": hop,
            "frame_size": frame_size,
            "n_frames": len(chroma),
            "harmonics": 4,
            "times": times,
            "chroma": chroma  # (n_frames, 12)
        }
    except ImportError as e:
        logger.warning(f"⚠️  Essentia not available: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Essentia HPCP failed: {e}")
        raise


def extract_chroma(
    audio_path: Path, 
    prefer_essentia: bool = True,
    sr: int = 44100
) -> Dict[str, Any]:
    """Chroma特徴抽出（Essentia優先、librosaフォールバック）
    
    Args:
        audio_path: オーディオファイルパス（WAV推奨）
        prefer_essentia: Essentia HPCPを優先（Falseでlibrosa強制）
        sr: サンプリングレート
    
    Returns:
        {
            "backend": "essentia_hpcp" or "librosa_cqt",
            "times": List[float],  # フレーム時刻（秒）
            "chroma": List[List[float]],  # (n_frames, 12)
            ...
        }
    """
    if prefer_essentia:
        try:
            return _essentia_hpcp(audio_path, sr=sr)
        except Exception as e:
            logger.warning(f"⚠️  Essentia failed, falling back to librosa: {e}")
    
    return _fallback_librosa_chroma(audio_path, sr=sr)


def save_chroma_json(
    audio_path: Path, 
    out_json: Path, 
    prefer_essentia: bool = True,
    sr: int = 44100
):
    """Chroma特徴をJSON保存
    
    Args:
        audio_path: 入力オーディオファイル
        out_json: 出力JSONパス
        prefer_essentia: Essentia優先フラグ
        sr: サンプリングレート
    """
    logger.info(f"📖 Loading audio: {audio_path}")
    data = extract_chroma(audio_path, prefer_essentia=prefer_essentia, sr=sr)
    
    logger.info(f"💾 Saving chroma: {out_json}")
    logger.info(f"   Backend: {data['backend']}")
    logger.info(f"   Frames: {data['n_frames']}")
    logger.info(f"   Sample rate: {data['sample_rate']} Hz")
    
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Extract chroma features (Essentia HPCP preferred, librosa CQT fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Essentia優先（自動フォールバック）
  python ops/chordino_bridge.py \\
      --audio data/full.wav \\
      --out data/chroma_probe.json
  
  # librosa強制
  python ops/chordino_bridge.py \\
      --audio data/full.wav \\
      --out data/chroma_probe.json \\
      --no-essentia

Research Background:
  - NNLS-Chroma (Chordino) implementation
  - Essentia HPCP: ISMIR/MIREX standard
  - Harmonic-aware 12-dim chroma extraction
        """
    )
    
    ap.add_argument("--audio", type=Path, required=True, help="Input audio file (WAV)")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON file")
    ap.add_argument("--no-essentia", action="store_true", help="Force librosa (skip Essentia)")
    ap.add_argument("--sr", type=int, default=44100, help="Sample rate (default: 44100)")
    
    args = ap.parse_args()
    
    if not args.audio.exists():
        logger.error(f"❌ Audio file not found: {args.audio}")
        exit(1)
    
    save_chroma_json(
        args.audio, 
        args.out, 
        prefer_essentia=not args.no_essentia,
        sr=args.sr
    )
    
    print(f"✅ Saved chroma: {args.out}")
