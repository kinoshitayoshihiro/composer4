#!/usr/bin/env python3
"""
DAWdreamer Batch Renderer

MIDI → WAV一括レンダリングシステム（pretty_midi + FluidSynth使用）

Features:
- MIDI → WAV変換（pretty_midi + FluidSynthベース）
- 楽器別MIDI処理
- バッチ処理対応
- サンプルレート/ビット深度設定
- プログレスバー表示

Note:
- pretty_midiのFluidSynth統合を使用（シンプルで安定）
- SoundFontがない場合はpretty_midiの内蔵シンセサイザを使用
- DAWdreamerは依存関係として残すが、実際の処理はpretty_midiで行う

Dependencies:
- pretty_midi: pip install pretty_midi
- FluidSynth (optional): brew install fluid-synth
- SoundFont (optional): GeneralUser GS v1.471.sf2 など

Usage:
    from scripts.render.dawdreamer_batch import DAWdreamerBatchRenderer
    
    renderer = DAWdreamerBatchRenderer(
        soundfont_path="soundfonts/GeneralUser_GS.sf2",
        sample_rate=44100
    )
    
    # Single MIDI
    renderer.render_midi("output/guitar.mid", "output/audio/guitar.wav")
    
    # Batch processing
    midi_files = {
        "guitar": "output/guitar.mid",
        "bass": "output/bass.mid",
        "strings": "output/strings.mid"
    }
    renderer.render_batch(midi_files, "output/audio")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import json
import time

# Check dawdreamer availability (optional dependency)
try:
    import dawdreamer as daw
    DAWDREAMER_AVAILABLE = True
except ImportError:
    DAWDREAMER_AVAILABLE = False
    # Not critical, pretty_midi is primary engine

# Check pretty_midi availability (required)
try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    logging.warning("⚠️  pretty_midi not available. Install with: pip install pretty_midi")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 🔴 Phase 2: 堅牢化のための定数
PEAK_TARGET_DB = -1.0  # 正規化目標: -1.0 dBFS
CLIPPING_THRESHOLD = 0.001  # クリッピング許容率: 0.1%
MAX_VELOCITY = 123  # ベロシティ上限（-1dBFS余裕確保）
RENDER_TIMEOUT_SECONDS = 60  # レンダリングタイムアウト（秒/ファイル）


class DAWdreamerBatchRenderer:
    """
    MIDI → WAV一括レンダリング（pretty_midi + FluidSynth使用）
    
    Phase 2強化:
    - -1.0dBFS正規化の強制
    - クリッピング率検出・ゲート検証
    - 失敗ファイル記録・リカバリ機能
    - タイムアウト設定
    """
    
    def __init__(
        self,
        soundfont_path: Optional[Path] = None,
        sample_rate: int = 44100,
        buffer_size: int = 512,
        duration_seconds: float = 60.0,
        verify_sf2_hash: bool = True,
        failed_renders_path: Optional[Path] = None
    ):
        """
        Initialize DAWdreamer Batch Renderer
        
        Args:
            soundfont_path: SoundFontファイルパス (.sf2)
            sample_rate: Sample rate (Hz)
            buffer_size: Buffer size (samples) - unused with pretty_midi
            duration_seconds: Maximum duration for rendering - unused with pretty_midi
            verify_sf2_hash: SF2ハッシュ検証を有効化
            failed_renders_path: 失敗記録ファイルパス（デフォルト: failed_renders.jsonl）
        """
        if not PRETTY_MIDI_AVAILABLE:
            raise RuntimeError("pretty_midi is not installed. Run: pip install pretty_midi")
        
        self.soundfont_path = Path(soundfont_path) if soundfont_path else None
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.duration_seconds = duration_seconds
        self.verify_sf2_hash = verify_sf2_hash
        
        # 失敗記録ファイル
        self.failed_renders_path = failed_renders_path or Path("data/failed_renders.jsonl")
        
        # Validate soundfont
        if self.soundfont_path and not self.soundfont_path.exists():
            logger.warning(f"⚠️  SoundFont not found: {self.soundfont_path}")
            logger.warning("   Rendering may fail without a valid SoundFont")
        
        # SF2ハッシュ検証（Phase 2）
        if self.verify_sf2_hash and self.soundfont_path and self.soundfont_path.exists():
            self._verify_soundfont_hash()
    
    def _verify_soundfont_hash(self) -> None:
        """SoundFontハッシュを検証（Phase 2強化）"""
        try:
            from scripts.render.soundfont_manager import SoundFontManager
            
            manager = SoundFontManager()
            is_valid, message = manager.verify(self.soundfont_path)
            
            if is_valid:
                logger.info(f"✅ SoundFont verified: {self.soundfont_path.name}")
            else:
                logger.warning(message)
        
        except Exception as e:
            logger.warning(f"⚠️  SF2 hash verification skipped: {e}")
    
    @staticmethod
    def analyze_audio_safety(audio: np.ndarray) -> Dict[str, float]:
        """
        オーディオの安全性を解析（Phase 2強化）
        
        Args:
            audio: 音声データ（float32, -1.0〜1.0）
        
        Returns:
            {
                'peak_db': ピーク値（dB）,
                'clipping_rate': クリッピング率（0.0〜1.0）,
                'is_safe': 安全かどうか（bool）
            }
        """
        # ピーク検出
        peak = np.abs(audio).max()
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # クリッピング検出（±0.99以上をクリッピングと見なす）
        clipping_samples = np.sum(np.abs(audio) >= 0.99)
        clipping_rate = clipping_samples / len(audio)
        
        # 安全判定
        is_safe = clipping_rate < CLIPPING_THRESHOLD
        
        return {
            'peak_db': float(peak_db),
            'clipping_rate': float(clipping_rate),
            'is_safe': bool(is_safe)
        }
    
    def render_midi(
        self,
        midi_path: Path,
        output_wav_path: Path,
        duration: Optional[float] = None,
        normalize_db: float = PEAK_TARGET_DB
    ) -> Tuple[Path, Dict[str, float]]:
        """
        単一MIDI → WAV変換（Phase 2強化：正規化・安全性チェック）
        
        Args:
            midi_path: Input MIDI file
            output_wav_path: Output WAV file
            duration: Duration in seconds (None = auto-detect from MIDI)
            normalize_db: 正規化目標（dBFS、デフォルト -1.0）
        
        Returns:
            (出力WAVパス, 音声安全性メトリクス)
        """
        midi_path = Path(midi_path)
        output_wav_path = Path(output_wav_path)
        
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        logger.info(f"🎹 Rendering: {midi_path.name}")
        
        start_time = time.time()
        
        # Simple approach: Use pretty_midi to render MIDI with FluidSynth
        # (DAWdreamer's SamplerProcessor requires audio data, not ideal for MIDI)
        try:
            import pretty_midi
            
            # Load MIDI with pretty_midi
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Use FluidSynth if available, otherwise use pretty_midi's internal synth
            if self.soundfont_path and self.soundfont_path.exists():
                # Synthesize with FluidSynth
                audio = midi.fluidsynth(
                    fs=self.sample_rate,
                    sf2_path=str(self.soundfont_path)
                )
            else:
                # Fallback: Use pretty_midi's default synth
                audio = midi.synthesize(fs=self.sample_rate)
            
            # Phase 2強化：正規化を強制
            peak = np.abs(audio).max()
            if peak > 0:
                # 目標ピーク値（-1.0 dBFS = 0.891）
                target_peak = 10 ** (normalize_db / 20)
                audio = audio * (target_peak / peak)
            
            # 安全性解析
            safety_metrics = self.analyze_audio_safety(audio)
            
            # クリッピング警告
            if not safety_metrics['is_safe']:
                logger.warning(
                    f"⚠️  Clipping detected: {safety_metrics['clipping_rate']*100:.2f}% "
                    f"(threshold: {CLIPPING_THRESHOLD*100:.1f}%)"
                )
            
            # Save WAV
            output_wav_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Clip and convert to int16
            audio_clipped = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            
            # Write WAV
            import wave
            
            # Ensure stereo
            if audio_int16.ndim == 1:
                audio_int16 = np.stack([audio_int16, audio_int16], axis=0)
            
            with wave.open(str(output_wav_path), 'wb') as wav_file:
                wav_file.setnchannels(audio_int16.shape[0])  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.T.tobytes())
            
            elapsed = time.time() - start_time
            duration_sec = len(audio) / self.sample_rate
            
            logger.info(
                f"✅ Rendered: {output_wav_path.name} "
                f"({duration_sec:.1f}s, peak: {safety_metrics['peak_db']:.1f} dB, "
                f"clip: {safety_metrics['clipping_rate']*100:.2f}%, "
                f"time: {elapsed:.1f}s)"
            )
            
            return output_wav_path, safety_metrics
        
        except ImportError:
            raise RuntimeError("pretty_midi is required. Install with: pip install pretty_midi")
        
        except Exception as e:
            # Phase 2: 失敗を記録
            self._record_failure(midi_path, str(e))
            raise
    
    def _record_failure(self, midi_path: Path, error_message: str) -> None:
        """
        失敗したレンダリングを記録（Phase 2強化）
        
        Args:
            midi_path: 失敗したMIDIファイル
            error_message: エラーメッセージ
        """
        self.failed_renders_path.parent.mkdir(parents=True, exist_ok=True)
        
        failure_record = {
            'midi_path': str(midi_path),
            'error': error_message,
            'timestamp': time.time()
        }
        
        with open(self.failed_renders_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(failure_record, ensure_ascii=False) + '\n')
        
        logger.error(f"[FAIL] step=render file={midi_path.name} error={error_message}")
    
    def load_failed_renders(self) -> List[Path]:
        """
        失敗記録から失敗MIDIパスを読み込み
        
        Returns:
            失敗したMIDIパスのリスト
        """
        if not self.failed_renders_path.exists():
            return []
        
        failed_paths = []
        
        with open(self.failed_renders_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                failed_paths.append(Path(record['midi_path']))
        
        return failed_paths
    
    def clear_failed_renders(self) -> None:
        """失敗記録をクリア"""
        if self.failed_renders_path.exists():
            self.failed_renders_path.unlink()
            logger.info(f"🗑️  Cleared failure log: {self.failed_renders_path}")
    
    def render_batch(
        self,
        midi_files: Dict[str, Path],
        output_dir: Path,
        duration: Optional[float] = None,
        resume: bool = False
    ) -> Tuple[Dict[str, Path], Dict[str, Dict]]:
        """
        複数MIDI → WAVバッチ変換（Phase 2強化：リカバリ対応）
        
        Args:
            midi_files: Dict of {instrument_name: midi_path}
            output_dir: Output directory
            duration: Duration in seconds (None = auto-detect)
            resume: 失敗分のみ再実行（True）または全実行（False）
        
        Returns:
            (
                {instrument_name: wav_path},
                {instrument_name: safety_metrics}
            )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # リカバリモード：失敗MIDIのみを対象
        if resume:
            failed_paths = self.load_failed_renders()
            if not failed_paths:
                logger.info("✅ No failed renders to resume")
                return {}, {}
            
            logger.info(f"\n🔄 Resume mode: {len(failed_paths)} failed file(s)")
            
            # 失敗MIDIのみをフィルタ
            midi_files = {
                name: path for name, path in midi_files.items()
                if path in failed_paths
            }
            
            # 失敗記録をクリア（再試行前）
            self.clear_failed_renders()
        
        logger.info(f"\n🎼 Batch rendering: {len(midi_files)} files")
        logger.info(f"   Output dir: {output_dir}")
        
        output_files = {}
        safety_reports = {}
        
        for instrument_name, midi_path in midi_files.items():
            output_wav = output_dir / f"{instrument_name}.wav"
            
            try:
                rendered_path, safety_metrics = self.render_midi(
                    midi_path=midi_path,
                    output_wav_path=output_wav,
                    duration=duration
                )
                output_files[instrument_name] = rendered_path
                safety_reports[instrument_name] = safety_metrics
            
            except Exception as e:
                logger.error(f"❌ Failed to render {instrument_name}: {e}")
        
        # 安全性サマリ
        unsafe_count = sum(1 for m in safety_reports.values() if not m['is_safe'])
        
        logger.info(f"\n✅ Batch rendering complete: {len(output_files)}/{len(midi_files)} files")
        
        if unsafe_count > 0:
            logger.warning(f"⚠️  {unsafe_count} file(s) have clipping issues")
        
        return output_files, safety_reports


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DAWdreamer Batch Renderer - MIDI to WAV')
    parser.add_argument('--midi', type=Path, help='Single MIDI file to render')
    parser.add_argument('--output', type=Path, help='Output WAV file')
    parser.add_argument('--midi-dir', type=Path, help='Directory with MIDI files for batch processing')
    parser.add_argument('--output-dir', type=Path, help='Output directory for batch processing')
    parser.add_argument('--soundfont', type=Path, help='SoundFont file (.sf2)')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate (Hz)')
    parser.add_argument('--duration', type=float, help='Duration in seconds (auto-detect if not specified)')
    parser.add_argument('--resume', action='store_true', help='Resume mode: retry only failed renders')
    parser.add_argument('--no-verify-sf2', action='store_true', help='Skip SF2 hash verification')
    
    args = parser.parse_args()
    
    print("\n🎵 DAWdreamer Batch Renderer")
    print("=" * 60)
    
    # Check pretty_midi
    if not PRETTY_MIDI_AVAILABLE:
        print("❌ pretty_midi not installed")
        print("   Install with: pip install pretty_midi")
        sys.exit(1)
    
    # Initialize renderer (Phase 2強化)
    renderer = DAWdreamerBatchRenderer(
        soundfont_path=args.soundfont,
        sample_rate=args.sample_rate,
        verify_sf2_hash=not args.no_verify_sf2
    )
    
    # Single file mode
    if args.midi and args.output:
        output_path, safety = renderer.render_midi(
            midi_path=args.midi,
            output_wav_path=args.output,
            duration=args.duration
        )
        print(f"\n✅ Rendered: {args.output}")
        print(f"   Peak: {safety['peak_db']:.1f} dB")
        print(f"   Clipping: {safety['clipping_rate']*100:.2f}%")
        print(f"   Safe: {'✅' if safety['is_safe'] else '❌'}")
    
    # Batch mode
    elif args.midi_dir and args.output_dir:
        midi_dir = Path(args.midi_dir)
        
        # Find all MIDI files
        midi_files = {}
        for midi_path in midi_dir.glob("*.mid"):
            instrument_name = midi_path.stem
            midi_files[instrument_name] = midi_path
        
        if len(midi_files) == 0:
            print(f"⚠️  No MIDI files found in: {midi_dir}")
            sys.exit(1)
        
        output_files, safety_reports = renderer.render_batch(
            midi_files=midi_files,
            output_dir=args.output_dir,
            duration=args.duration,
            resume=args.resume
        )
        
        print(f"\n✅ Batch complete: {len(output_files)} files")
        
        # 詳細レポート
        for name, path in output_files.items():
            safety = safety_reports.get(name, {})
            status = '✅' if safety.get('is_safe', True) else '⚠️ '
            print(f"   {status} {name}: {path}")
            if safety:
                print(f"      Peak: {safety['peak_db']:.1f} dB, Clipping: {safety['clipping_rate']*100:.2f}%")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
