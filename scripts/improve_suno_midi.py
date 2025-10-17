#!/usr/bin/env python3
"""
Suno MIDI Quality Improvement Pipeline

Sunoの低品質MIDIを高品質化するための段階的パイプライン:
1. WAV→MIDI変換（複数エンジン）
2. 既存システムでの浄化
3. 品質ゲート適用
4. 反復改善

Usage:
    python scripts/improve_suno_midi.py \
      --input suno_exports/raw \
      --output improved/suno \
      --method iterative \
      --iterations 3
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import pretty_midi
import numpy as np


class SunoMIDIImprover:
    """Suno MIDIの品質改善パイプライン"""
    
    def __init__(self, config_path: Path = None):
        self.config = self._load_config(config_path)
        
    def _load_config(self, path: Path = None) -> Dict:
        """改善設定の読み込み"""
        default_config = {
            "wav_to_midi": {
                "engines": ["basic_pitch", "omnizart", "mt3"],  # 複数エンジンで変換
                "vote_threshold": 2  # 2つ以上のエンジンが同意した音符のみ採用
            },
            "quantization": {
                "grid": 0.0625,  # 16分音符グリッド
                "swing": 0.0,    # スウィング量
                "strength": 0.8  # クオンタイズ強度
            },
            "cleanup": {
                "min_duration": 0.05,    # 最小音符長（秒）
                "remove_overlaps": True,  # 重複除去
                "velocity_smooth": True,  # ベロシティ平滑化
                "tempo_stabilize": True   # テンポ安定化
            },
            "stage2_gate": {
                "threshold_boost": 5.0,  # 実データより5%厳しく
                "required_metrics": ["all"]
            }
        }
        
        if path and path.exists():
            import yaml
            with open(path) as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def wav_to_midi_ensemble(self, wav_path: Path, output_dir: Path) -> Path:
        """
        複数エンジンでWAV→MIDI変換し、投票で信頼性の高い音符のみ採用
        
        Args:
            wav_path: 入力WAVファイル
            output_dir: 出力ディレクトリ
            
        Returns:
            改善されたMIDIファイルのパス
        """
        engines = self.config["wav_to_midi"]["engines"]
        threshold = self.config["wav_to_midi"]["vote_threshold"]
        
        # 各エンジンで変換
        midi_candidates = []
        for engine in engines:
            midi_path = self._run_wav_to_midi_engine(wav_path, engine, output_dir)
            if midi_path:
                midi_candidates.append(midi_path)
        
        # 投票で統合
        ensemble_midi = self._vote_ensemble(midi_candidates, threshold)
        
        # 保存
        ensemble_path = output_dir / f"{wav_path.stem}_ensemble.mid"
        ensemble_midi.write(str(ensemble_path))
        
        return ensemble_path
    
    def _run_wav_to_midi_engine(self, wav_path: Path, engine: str, output_dir: Path) -> Path:
        """個別エンジンでWAV→MIDI変換"""
        output_dir.mkdir(parents=True, exist_ok=True)
        midi_path = output_dir / f"{wav_path.stem}_{engine}.mid"
        
        try:
            if engine == "basic_pitch":
                # Spotify's Basic Pitch
                subprocess.run([
                    "basic-pitch",
                    str(output_dir),
                    str(wav_path)
                ], check=True)
                
            elif engine == "omnizart":
                # Omnizart
                subprocess.run([
                    "omnizart", "music", "transcribe",
                    str(wav_path),
                    "--output", str(output_dir)
                ], check=True)
                
            elif engine == "mt3":
                # MT3 (Music Transcription with Transformers)
                subprocess.run([
                    "python", "scripts/mt3_transcribe.py",
                    "--input", str(wav_path),
                    "--output", str(midi_path)
                ], check=True)
            
            return midi_path if midi_path.exists() else None
            
        except Exception as e:
            print(f"⚠️ {engine} failed: {e}")
            return None
    
    def _vote_ensemble(self, midi_paths: List[Path], threshold: int) -> pretty_midi.PrettyMIDI:
        """
        複数MIDIファイルの投票で信頼性の高い音符のみ採用
        
        Args:
            midi_paths: MIDIファイルのリスト
            threshold: 採用に必要な投票数
            
        Returns:
            統合されたPrettyMIDIオブジェクト
        """
        # 各MIDIを読み込み
        midis = []
        for path in midi_paths:
            try:
                midis.append(pretty_midi.PrettyMIDI(str(path)))
            except:
                continue
        
        if not midis:
            raise ValueError("No valid MIDI files to ensemble")
        
        # 音符の投票集計（時間・ピッチが近い音符をグループ化）
        note_votes = {}
        time_tolerance = 0.05  # 50ms以内は同じタイミングとみなす
        
        for midi in midis:
            for instrument in midi.instruments:
                for note in instrument.notes:
                    # 近い音符を探す
                    key = self._find_note_cluster(
                        note_votes, 
                        note.pitch, 
                        note.start, 
                        time_tolerance
                    )
                    
                    if key not in note_votes:
                        note_votes[key] = {
                            'pitch': note.pitch,
                            'start': note.start,
                            'end': note.end,
                            'velocity': note.velocity,
                            'count': 0,
                            'starts': [],
                            'ends': [],
                            'velocities': []
                        }
                    
                    note_votes[key]['count'] += 1
                    note_votes[key]['starts'].append(note.start)
                    note_votes[key]['ends'].append(note.end)
                    note_votes[key]['velocities'].append(note.velocity)
        
        # 閾値以上の投票を得た音符のみ採用
        ensemble = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        
        for vote_data in note_votes.values():
            if vote_data['count'] >= threshold:
                # 中央値を使用（外れ値に強い）
                start = np.median(vote_data['starts'])
                end = np.median(vote_data['ends'])
                velocity = int(np.median(vote_data['velocities']))
                
                inst.notes.append(pretty_midi.Note(
                    velocity=velocity,
                    pitch=vote_data['pitch'],
                    start=start,
                    end=end
                ))
        
        ensemble.instruments.append(inst)
        return ensemble
    
    def _find_note_cluster(self, note_votes: Dict, pitch: int, start: float, tolerance: float) -> str:
        """時間・ピッチが近い音符クラスタを検索"""
        for key, data in note_votes.items():
            if (data['pitch'] == pitch and 
                abs(data['start'] - start) < tolerance):
                return key
        
        # 新しいクラスタ
        return f"{pitch}_{start:.3f}"
    
    def cleanup_midi(self, midi_path: Path, output_path: Path) -> Path:
        """
        MIDIファイルのクリーンアップ
        - 短すぎる音符除去
        - 重複除去
        - ベロシティ平滑化
        - テンポ安定化
        """
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        config = self.config["cleanup"]
        
        for instrument in midi.instruments:
            # 短すぎる音符除去
            instrument.notes = [
                n for n in instrument.notes 
                if (n.end - n.start) >= config["min_duration"]
            ]
            
            # 重複除去
            if config["remove_overlaps"]:
                instrument.notes = self._remove_overlaps(instrument.notes)
            
            # ベロシティ平滑化
            if config["velocity_smooth"]:
                instrument.notes = self._smooth_velocities(instrument.notes)
        
        # テンポ安定化
        if config["tempo_stabilize"]:
            midi = self._stabilize_tempo(midi)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(output_path))
        return output_path
    
    def _remove_overlaps(self, notes: List) -> List:
        """重複音符の除去"""
        notes = sorted(notes, key=lambda n: n.start)
        cleaned = []
        
        for note in notes:
            # 前の音符と重複チェック
            if cleaned and cleaned[-1].pitch == note.pitch:
                prev = cleaned[-1]
                if prev.end > note.start:
                    # 重複: 短い方を削除
                    if (note.end - note.start) > (prev.end - prev.start):
                        cleaned[-1] = note
                    continue
            
            cleaned.append(note)
        
        return cleaned
    
    def _smooth_velocities(self, notes: List, window: int = 3) -> List:
        """ベロシティの移動平均平滑化"""
        if len(notes) < window:
            return notes
        
        notes = sorted(notes, key=lambda n: n.start)
        
        for i in range(len(notes)):
            start = max(0, i - window // 2)
            end = min(len(notes), i + window // 2 + 1)
            avg_vel = int(np.mean([n.velocity for n in notes[start:end]]))
            notes[i].velocity = avg_vel
        
        return notes
    
    def _stabilize_tempo(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """テンポの安定化（極端な変化を平滑化）"""
        # 簡易実装: 全体を中央値テンポに統一
        if not midi.get_tempo_changes()[1].size:
            return midi
        
        median_tempo = np.median(midi.get_tempo_changes()[1])
        
        # 新しいMIDIを作成（固定テンポ）
        new_midi = pretty_midi.PrettyMIDI(initial_tempo=median_tempo)
        for inst in midi.instruments:
            new_midi.instruments.append(inst)
        
        return new_midi
    
    def iterative_improvement(self, midi_path: Path, output_dir: Path, iterations: int = 3) -> Path:
        """
        反復改善: 
        1. システム通過
        2. 品質ゲート評価
        3. 不合格部分を修正
        4. 再評価
        """
        current_path = midi_path
        
        for i in range(iterations):
            print(f"🔄 Iteration {i+1}/{iterations}")
            
            # Stage2品質評価
            score = self._evaluate_stage2(current_path)
            print(f"   Score: {score:.1f}%")
            
            if score >= self.config["stage2_gate"]["threshold_boost"] + 40.0:
                print(f"✅ Passed quality gate!")
                break
            
            # 改善適用
            improved_path = output_dir / f"{midi_path.stem}_iter{i+1}.mid"
            current_path = self.cleanup_midi(current_path, improved_path)
        
        return current_path
    
    def _evaluate_stage2(self, midi_path: Path) -> float:
        """Stage2品質評価（仮実装）"""
        # TODO: 実際のStage2メトリクスを呼び出す
        # 暫定: ランダムスコア
        import random
        return random.uniform(35.0, 75.0)


def main():
    parser = argparse.ArgumentParser(description="Improve Suno MIDI quality")
    parser.add_argument("--input", required=True, help="Input directory (Suno exports)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--method", choices=["ensemble", "iterative", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--config", help="Config YAML path")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    improver = SunoMIDIImprover(Path(args.config) if args.config else None)
    
    # WAVファイルを処理
    wav_files = list(input_dir.glob("*.wav"))
    print(f"📂 Found {len(wav_files)} WAV files")
    
    for wav_path in wav_files:
        print(f"\n🎵 Processing: {wav_path.name}")
        
        try:
            if args.method in ["ensemble", "both"]:
                # Step 1: WAV→MIDIアンサンブル変換
                ensemble_dir = output_dir / "ensemble"
                midi_path = improver.wav_to_midi_ensemble(wav_path, ensemble_dir)
                print(f"   ✅ Ensemble conversion: {midi_path}")
            else:
                # Suno MIDIを直接使用
                midi_path = input_dir / f"{wav_path.stem}.mid"
            
            if args.method in ["iterative", "both"]:
                # Step 2: 反復改善
                iterative_dir = output_dir / "iterative"
                final_path = improver.iterative_improvement(
                    midi_path, 
                    iterative_dir, 
                    args.iterations
                )
                print(f"   ✅ Iterative improvement: {final_path}")
            else:
                final_path = midi_path
            
            # Step 3: 最終クリーンアップ
            clean_dir = output_dir / "clean"
            clean_path = improver.cleanup_midi(final_path, clean_dir / final_path.name)
            print(f"   ✅ Final cleanup: {clean_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print(f"\n✅ Completed! Output: {output_dir}")


if __name__ == "__main__":
    main()
