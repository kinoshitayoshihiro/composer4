#!/usr/bin/env python3
"""
validate_audio_quality.py - 音声KPI検証スクリプト（Phase 15.3）

WAVファイルから音声KPIを計測し、audio_gate_prod.yaml のSLOに基づいて検証します。

KPI項目:
  - render_rtf: レンダリング実時間比
  - clip_ratio: クリップ率
  - integrated_lufs: 統合ラウドネス
  - crest_factor_db: クレストファクター
  - dc_offset_dbfs: DCオフセット
  - noise_floor_dbfs: ノイズフロア
  - latency_ms_onset: レイテンシー
  - missing_onset_rate: 欠落オンセット率

Usage:
    python3 scripts/validate_audio_quality.py \
      --wav song_packages/<project>/<song>/<instrument>_rendered.wav \
      --midi song_packages/<project>/<song>/<instrument>_merged.mid \
      --gate configs/audio_gate_prod.yaml \
      --out-json song_packages/<project>/<song>/audio_kpi.json
"""

import argparse
import json
import yaml
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:
    import scipy.io.wavfile as wavfile
    import librosa
except ImportError:
    print("ERROR: scipy or librosa not installed")
    print("Install: pip install scipy librosa")
    exit(1)


def load_gate_config(gate_path: Path, profile: str = "default") -> Dict:
    """audio_gate_prod.yaml読み込み + プロファイル選択"""
    with open(gate_path) as f:
        config = yaml.safe_load(f)
    
    # プロファイルが指定されている場合、そのプロファイルを返す
    if profile != "default" and profile in config:
        return config[profile]
    elif "default" in config:
        return config["default"]
    else:
        # プロファイルがない場合はconfig全体を返す（後方互換性）
        return config


def load_wav(wav_path: Path) -> tuple:
    """WAV読み込み（sample_rate, audio_data）"""
    sr, audio = wavfile.read(wav_path)
    
    # ステレオ → モノラル変換（簡易）
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    # 正規化（int16 → float32）
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    
    return sr, audio


def calculate_render_rtf(render_time_sec: float, audio_duration_sec: float) -> float:
    """レンダリング実時間比計算"""
    if audio_duration_sec <= 0:
        return 0.0
    return render_time_sec / audio_duration_sec


def calculate_clip_ratio(audio: np.ndarray, threshold: float = 0.95) -> float:
    """クリップ率計算（|amplitude| > threshold のサンプル率）"""
    clipped = np.abs(audio) > threshold
    return clipped.sum() / len(audio)


def calculate_integrated_lufs(audio: np.ndarray, sr: int) -> float:
    """統合ラウドネス計算（簡易版、ITU-R BS.1770近似）"""
    # RMS計算（ブロック単位）
    block_size = int(0.4 * sr)  # 400ms block
    hop_size = int(0.1 * sr)    # 100ms hop
    
    rms_blocks = []
    for i in range(0, len(audio) - block_size, hop_size):
        block = audio[i:i+block_size]
        rms = np.sqrt(np.mean(block**2))
        if rms > 0:
            rms_blocks.append(rms)
    
    if not rms_blocks:
        return -70.0  # silent
    
    # 統合ラウドネス（dBFS → LUFS近似）
    mean_rms = np.mean(rms_blocks)
    lufs = 20 * np.log10(mean_rms + 1e-10) - 0.691  # ITU-R BS.1770近似
    return float(lufs)


def calculate_crest_factor_db(audio: np.ndarray) -> float:
    """クレストファクター計算（ピーク/RMS）"""
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    
    if rms <= 0:
        return 0.0
    
    crest_factor = peak / rms
    return float(20 * np.log10(crest_factor))


def calculate_dc_offset_dbfs(audio: np.ndarray) -> float:
    """DCオフセット計算（直流成分）"""
    dc = np.mean(audio)
    if abs(dc) < 1e-10:
        return -100.0  # very low
    return float(20 * np.log10(abs(dc)))


def calculate_noise_floor_dbfs(audio: np.ndarray, percentile: float = 5.0) -> float:
    """ノイズフロア計算（低振幅パーセンタイル）"""
    abs_audio = np.abs(audio)
    noise_floor = np.percentile(abs_audio, percentile)
    
    if noise_floor < 1e-10:
        return -100.0
    return float(20 * np.log10(noise_floor))


def calculate_latency_ms_onset(audio: np.ndarray, sr: int, threshold: float = 0.01) -> float:
    """レイテンシー計算（最初の有音サンプルまで）"""
    onset_idx = np.argmax(np.abs(audio) > threshold)
    if onset_idx == 0 and np.abs(audio[0]) <= threshold:
        return 0.0  # no onset detected
    
    latency_sec = onset_idx / sr
    return float(latency_sec * 1000.0)  # ms


def calculate_missing_onset_rate(
    audio: np.ndarray,
    sr: int,
    midi_note_count: int,
    onset_threshold: float = 0.02
) -> float:
    """欠落オンセット率計算（簡易版、librosa onset detection）"""
    try:
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=sr,
            units="frames",
            hop_length=512,
            backtrack=True
        )
        detected_count = len(onset_frames)
    except Exception:
        detected_count = 0
    
    if midi_note_count <= 0:
        return 0.0
    
    missing_rate = max(0.0, 1.0 - detected_count / midi_note_count)
    return float(missing_rate)


def get_midi_note_count(midi_path: Optional[Path]) -> int:
    """MIDI総ノート数取得（簡易版）"""
    if not midi_path or not midi_path.exists():
        return 0
    
    try:
        from mido import MidiFile
        mid = MidiFile(midi_path)
        note_count = 0
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    note_count += 1
        return note_count
    except Exception:
        return 0


def validate_audio(
    wav_path: Path,
    midi_path: Optional[Path],
    gate_config: Dict,
    render_time_sec: Optional[float] = None,
) -> Dict:
    """音声KPI検証"""
    
    # WAV読み込み
    sr, audio = load_wav(wav_path)
    audio_duration_sec = len(audio) / sr
    
    # MIDI情報
    midi_note_count = get_midi_note_count(midi_path)
    
    # KPI計算
    kpi = {
        "file_path": str(wav_path),
        "sample_rate": int(sr),
        "audio_duration_sec": float(audio_duration_sec),
        "midi_note_count": midi_note_count,
    }
    
    # render_rtf
    if render_time_sec is not None:
        kpi["render_time_sec"] = float(render_time_sec)
        kpi["render_rtf"] = calculate_render_rtf(render_time_sec, audio_duration_sec)
    else:
        kpi["render_rtf"] = None
    
    # クリップ率
    kpi["clip_ratio"] = calculate_clip_ratio(audio)
    
    # 統合ラウドネス
    kpi["integrated_lufs"] = calculate_integrated_lufs(audio, sr)
    
    # クレストファクター
    kpi["crest_factor_db"] = calculate_crest_factor_db(audio)
    
    # DCオフセット
    kpi["dc_offset_dbfs"] = calculate_dc_offset_dbfs(audio)
    
    # ノイズフロア
    kpi["noise_floor_dbfs"] = calculate_noise_floor_dbfs(audio)
    
    # レイテンシー
    kpi["latency_ms_onset"] = calculate_latency_ms_onset(audio, sr)
    
    # 欠落オンセット率
    kpi["missing_onset_rate"] = calculate_missing_onset_rate(audio, sr, midi_note_count)
    
    # SLO検証
    audio_gate = gate_config.get("audio", {})
    validation_results = {}
    
    for metric, value in kpi.items():
        if value is None or metric in ["file_path", "sample_rate", "audio_duration_sec", "midi_note_count", "render_time_sec"]:
            continue
        
        gate = audio_gate.get(metric)
        if not gate:
            continue
        
        status = "PASS"
        
        # max check
        if "max" in gate and value > gate["max"]:
            status = "FAIL"
        elif "warn_max" in gate and value > gate["warn_max"]:
            status = "WARNING"
        
        # min check
        if "min" in gate and value < gate["min"]:
            status = "FAIL"
        elif "warn_min" in gate and value < gate["warn_min"]:
            status = "WARNING"
        
        validation_results[metric] = {
            "value": value,
            "status": status,
            "gate": gate,
        }
    
    kpi["validation_results"] = validation_results
    
    return kpi


def main():
    parser = argparse.ArgumentParser(description="音声KPI検証")
    parser.add_argument("--wav", type=Path, required=True, help="WAVファイルパス")
    parser.add_argument("--midi", type=Path, help="MIDIファイルパス（オンセット検証用）")
    parser.add_argument("--gate", type=Path, required=True, help="audio_gate_prod.yaml パス")
    parser.add_argument("--out-json", type=Path, required=True, help="出力JSON パス")
    parser.add_argument("--render-time", type=float, help="レンダリング時間（秒）")
    parser.add_argument("--profile", type=str, default="default", help="KPIプロファイル名（例: piano_kpi）")
    args = parser.parse_args()
    
    if not args.wav.exists():
        print(f"ERROR: WAV not found: {args.wav}")
        exit(1)
    
    if not args.gate.exists():
        print(f"ERROR: Gate config not found: {args.gate}")
        exit(1)
    
    print(f"🎧 Validating audio quality: {args.wav}")
    print(f"   Profile: {args.profile}")
    
    gate_config = load_gate_config(args.gate, args.profile)
    
    kpi = validate_audio(
        wav_path=args.wav,
        midi_path=args.midi,
        gate_config=gate_config,
        render_time_sec=args.render_time,
    )
    
    # 結果保存
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(kpi, f, indent=2)
    
    print(f"✅ Audio KPI saved: {args.out_json}")
    print(f"\n📊 Summary:")
    print(f"   Duration: {kpi['audio_duration_sec']:.2f}s")
    if kpi['render_rtf'] is not None:
        print(f"   Render RTF: {kpi['render_rtf']:.2f}x")
    print(f"   Clip ratio: {kpi['clip_ratio']:.4f} ({kpi['clip_ratio']*100:.2f}%)")
    print(f"   Integrated LUFS: {kpi['integrated_lufs']:.2f}")
    print(f"   Crest factor: {kpi['crest_factor_db']:.2f} dB")
    
    # SLO結果表示
    print(f"\n🚦 SLO Validation:")
    fail_count = 0
    warn_count = 0
    for metric, result in kpi['validation_results'].items():
        status = result['status']
        value = result['value']
        
        if status == "FAIL":
            print(f"   ❌ {metric}: {value:.4f} (FAIL)")
            fail_count += 1
        elif status == "WARNING":
            print(f"   ⚠️  {metric}: {value:.4f} (WARNING)")
            warn_count += 1
        else:
            print(f"   ✅ {metric}: {value:.4f} (PASS)")
    
    print(f"\n📈 Total: {len(kpi['validation_results'])} metrics")
    print(f"   PASS: {len(kpi['validation_results']) - fail_count - warn_count}")
    print(f"   WARNING: {warn_count}")
    print(f"   FAIL: {fail_count}")
    
    if fail_count > 0:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
