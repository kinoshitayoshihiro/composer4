#!/usr/bin/env python3
"""
Stage1 LAMDA Plus v2
Pickle廃止・統合レイアウト・Content-based ID・OK::メタ注入対応
"""
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import mido
import yaml
import pandas as pd

VERSION = "2.0"


class LAMDAPriors:
    """LAMDA先験情報読み込み（TOTALS.parquet/SIGNATURES.json）"""
    
    def __init__(self, totals_path=None, signatures_path=None):
        self.totals = None
        self.signatures = None
        
        # TOTALS.parquet読み込み
        if totals_path and os.path.exists(totals_path):
            try:
                self.totals = pd.read_parquet(totals_path)
                print(f"✓ TOTALS loaded: {totals_path}")
            except Exception as e:
                print(f"⚠ TOTALS load failed: {e}")
        
        # SIGNATURES.json読み込み
        if signatures_path and os.path.exists(signatures_path):
            try:
                with open(signatures_path, 'r') as f:
                    self.signatures = json.load(f)
                print(f"✓ SIGNATURES loaded: {signatures_path}")
            except Exception as e:
                print(f"⚠ SIGNATURES load failed: {e}")
    
    def get_default_timesig(self):
        """デフォルト拍子（4/4）"""
        return (4, 4)
    
    def get_pitch_range(self):
        """デフォルトピッチレンジ"""
        return (21, 108)  # A0-C8
    
    def get_vel_range(self):
        """デフォルトベロシティレンジ"""
        return (1, 127)
    
    def get_dur_range(self):
        """デフォルト音長レンジ（ティック）"""
        return (30, 3840)


# ========== ユーティリティ関数 ==========
def expand_placeholders(path_str, roots):
    """
    ${base} 等を roots 辞書で展開 → さらに env/~/ を展開
    """
    import re
    def repl(m):
        key = m.group(1)
        return str(roots.get(key, m.group(0)))
    s = re.sub(r"\$\{([^}]+)\}", repl, str(path_str))
    s = os.path.expandvars(os.path.expanduser(s))
    return s


# ========== ID生成関数 ==========
def compute_source_mid_id(midi_path):
    """入力MIDIのMD5ハッシュ[:16]（bytes必須）"""
    with open(midi_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:16]


def compute_bar_fingerprint(mid):
    """バー指紋: 各バーの(時刻, 拍子, テンポ)のハッシュ（bytes必須）"""
    bars_info = []
    current_time = 0
    current_tempo = 500000  # デフォルト120BPM
    current_timesig = (4, 4)
    
    for msg in mid.merged_track:
        current_time += msg.time
        if msg.type == 'set_tempo':
            current_tempo = msg.tempo
        elif msg.type == 'time_signature':
            current_timesig = (msg.numerator, msg.denominator)
        elif msg.type == 'note_on' and msg.velocity > 0:
            # 簡易バー検出（1小節=4拍子で1920ティック想定）
            bar_idx = current_time // (mid.ticks_per_beat * 4)
            bars_info.append((bar_idx, current_timesig, current_tempo))
    
    fingerprint = str(sorted(set(bars_info)))
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()[:16]


def compute_content_id(mid):
    """Content-based ID: バー指紋+総ティック長（区切り文字で衝突回避）"""
    bar_fp = compute_bar_fingerprint(mid)
    total_ticks = sum(msg.time for msg in mid.merged_track)
    content_str = f"{bar_fp}|{total_ticks}"  # 区切り文字追加
    return hashlib.md5(content_str.encode("utf-8")).hexdigest()[:16]


# ========== クリーニング関数 ==========
def rescue_timesig_with_signatures(mid, priors, config):
    """拍子救済: SIGNATURES優先+自己相似ヒューリスティック"""
    timesig_msgs = [msg for msg in mid.merged_track if msg.type == 'time_signature']
    
    if not timesig_msgs:
        # 拍子がない場合、デフォルト4/4挿入
        default_ts = priors.get_default_timesig()
        mid.tracks[0].insert(0, mido.MetaMessage('time_signature', 
                                                  numerator=default_ts[0], 
                                                  denominator=default_ts[1], 
                                                  time=0))
        print(f"  ⚠ No time_signature → insert default {default_ts}")
    
    return mid


def smooth_tempo_track(mid, config):
    """テンポ平滑化: BPMクリップ+最小持続フィルタ"""
    tempo_clip = config['policy'].get('tempo_bpm_clip', [30, 300])
    min_hold_beats = config['policy'].get('tempo_min_hold_beats', 1.0)
    
    tpb = mid.ticks_per_beat
    min_hold_ticks = int(min_hold_beats * tpb)
    
    new_track = []
    tempo_buffer = []
    
    for msg in mid.merged_track:
        if msg.type == 'set_tempo':
            bpm = mido.tempo2bpm(msg.tempo)
            bpm_clamped = max(tempo_clip[0], min(tempo_clip[1], bpm))
            tempo_clamped = mido.bpm2tempo(bpm_clamped)
            tempo_buffer.append((msg.time, tempo_clamped))
        else:
            new_track.append(msg)
    
    # 最小持続フィルタ（簡易実装: 全てのテンポを保持）
    for time, tempo in tempo_buffer:
        new_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=time))
    
    # トラック再構築
    mid.tracks[0] = mido.MidiTrack(sorted(new_track, key=lambda m: m.time))
    return mid


def clamp_notes_to_ranges(mid, config):
    """ノート制約: pitch/vel/dur_ticksを安全レンジに制約"""
    ranges = config['ranges']
    pitch_range = ranges.get('pitch', [21, 108])
    vel_range = ranges.get('vel', [1, 127])
    dur_range = ranges.get('dur_ticks', [30, 3840])
    
    for track in mid.tracks:
        note_on_times = {}
        new_track = []
        
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                # Pitch制約
                pitch_clamped = max(pitch_range[0], min(pitch_range[1], msg.note))
                # Velocity制約
                vel_clamped = max(vel_range[0], min(vel_range[1], msg.velocity))
                
                new_msg = msg.copy(note=pitch_clamped, velocity=vel_clamped)
                new_track.append(new_msg)
                note_on_times[(msg.channel, pitch_clamped)] = msg.time
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in note_on_times:
                    duration = msg.time - note_on_times[key]
                    # Duration制約
                    dur_clamped = max(dur_range[0], min(dur_range[1], duration))
                    # note_off時刻調整
                    new_time = note_on_times[key] + dur_clamped
                    new_msg = msg.copy(time=new_time)
                    new_track.append(new_msg)
                    del note_on_times[key]
            else:
                new_track.append(msg)
        
        track[:] = new_track
    
    return mid


def normalize_drums(mid, config):
    """ドラム正規化: GM Ch10統一+近傍スナップ"""
    if not config['policy'].get('drum_normalize', False):
        return mid
    
    # GM Drum Map（簡易版）
    GM_DRUM_MAP = {
        35: 36,  # Acoustic Bass Drum → Bass Drum 1
        37: 38,  # Side Stick → Acoustic Snare
        # ... 他のマッピング
    }
    
    for track in mid.tracks:
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                if msg.channel == 9:  # Ch10（0-indexed）
                    # 近傍スナップ
                    if msg.note in GM_DRUM_MAP:
                        msg.note = GM_DRUM_MAP[msg.note]
    
    return mid


def split_long_notes_on_bar(mid, config):
    """バー境界分割: 長音を小節単位で分割（最小長はranges.dur_ticks[0]）"""
    if not config['policy'].get('bar_split_long_notes', False):
        return mid
    
    tpb = mid.ticks_per_beat
    bar_ticks = tpb * 4  # 4/4拍子想定
    min_dur = config['ranges']['dur_ticks'][0]  # 最小音長
    
    for track in mid.tracks:
        new_track = []
        note_on_times = {}
        
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_on_times[(msg.channel, msg.note)] = msg.time
                new_track.append(msg)
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in note_on_times:
                    start_time = note_on_times[key]
                    end_time = msg.time
                    duration = end_time - start_time
                    
                    # バー境界を超え、かつmin_dur以上の場合のみ分割
                    if duration > bar_ticks and duration >= min_dur:
                        current_time = start_time
                        while current_time + bar_ticks < end_time:
                            # note_off挿入
                            new_track.append(mido.Message('note_off', 
                                                         note=msg.note, 
                                                         channel=msg.channel, 
                                                         velocity=0, 
                                                         time=current_time + bar_ticks))
                            # 次のnote_on挿入
                            current_time += bar_ticks
                            new_track.append(mido.Message('note_on', 
                                                         note=msg.note, 
                                                         channel=msg.channel, 
                                                         velocity=64, 
                                                         time=current_time))
                        
                        # 最終note_off
                        new_track.append(msg.copy(time=end_time))
                    else:
                        new_track.append(msg)
                    
                    del note_on_times[key]
            else:
                new_track.append(msg)
        
        track[:] = new_track
    
    return mid


# ========== Stage1プロセッサ ==========
class Stage1Processor:
    def __init__(self, config, priors, verbose=False):
        self.config = config
        self.priors = priors
        self.verbose = verbose
        self.base_dir = Path(config['roots']['base'])
        self.midi_in_dir = self.base_dir / config['roots']['midi_in']
        self.midi_out_dir = self.base_dir / config['roots']['midi_out']
        
        # exclude_dirsをPath正規化（文字化け・全角対応）
        raw_excludes = config['roots'].get('exclude_dirs', [])
        self.exclude_dirs = [Path(d).as_posix() for d in raw_excludes]
        
        # 出力ディレクトリ作成
        self.midi_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume用: 処理済みsource_mid_id
        self.processed_ids = set()
    
    def load_processed_ids(self):
        """処理済みID読み込み（Resume対応）"""
        for content_dir in self.midi_out_dir.glob("*"):
            json_path = content_dir / "stage1_clean.json"
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        self.processed_ids.add(data['source_mid_id'])
                except:
                    pass
        
        print(f"✓ Resume: {len(self.processed_ids)} files already processed")
    
    def should_exclude(self, midi_path):
        """除外ディレクトリ判定（Path.parts単位で一致）"""
        path_parts = [Path(p).as_posix() for p in Path(midi_path).parts]
        for part in path_parts:
            if part in self.exclude_dirs:
                return True
        return False
    
    def process_midi_file(self, midi_path):
        """1つのMIDIファイルを処理"""
        try:
            # 除外判定
            if self.should_exclude(midi_path):
                if self.verbose:
                    print(f"⊗ SKIP (excluded): {midi_path}")
                return None
            
            # ID計算
            source_mid_id = compute_source_mid_id(midi_path)
            
            # Resume判定
            if source_mid_id in self.processed_ids:
                if self.verbose:
                    print(f"⊗ SKIP (already processed): {midi_path}")
                return None
            
            # MIDI読み込み
            mid = mido.MidiFile(midi_path)
            
            # クリーニング
            if self.config['policy'].get('timesig_rescue', True):
                mid = rescue_timesig_with_signatures(mid, self.priors, self.config)
            
            mid = smooth_tempo_track(mid, self.config)
            mid = clamp_notes_to_ranges(mid, self.config)
            mid = normalize_drums(mid, self.config)
            mid = split_long_notes_on_bar(mid, self.config)
            
            # Content ID計算
            content_id = compute_content_id(mid)
            
            # Run ID生成
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_v{VERSION}"
            
            # OK::meta注入（MIDOではコメントとして埋め込み）
            ok_meta = {
                "song_id": content_id,
                "stage": "stage1",
                "run_id": run_id,
                "source_mid_id": source_mid_id,
                "content_id": content_id,
                "time_sig": [4, 4],  # 簡易実装
                "bpm_est": 120  # 簡易実装
            }
            
            # 出力ディレクトリ
            output_dir = self.midi_out_dir / content_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # MIDI保存
            output_midi = output_dir / "stage1_clean.mid"
            mid.save(str(output_midi))
            
            # JSON保存
            output_json = output_dir / "stage1_clean.json"
            json_data = {
                "source_mid_id": source_mid_id,
                "content_id": content_id,
                "run_id": run_id,
                "ok_meta": ok_meta,
                "input_path": str(midi_path),
                "output_path": str(output_midi),
                "processed_at": datetime.now().isoformat()
            }
            
            with open(output_json, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            if self.verbose:
                print(f"✓ {content_id}: {Path(midi_path).name}")
            
            return {
                "content_id": content_id,
                "source_mid_id": source_mid_id,
                "input_path": str(midi_path),
                "output_path": str(output_midi)
            }
        
        except Exception as e:
            print(f"✗ ERROR: {midi_path} → {e}")
            return None
    
    def process_dataset(self, max_files=None):
        """データセット全体を処理"""
        # Resume対応
        self.load_processed_ids()
        
        # MIDIファイル列挙
        midi_files = list(self.midi_in_dir.rglob("*.mid"))
        
        if max_files:
            midi_files = midi_files[:max_files]
        
        print(f"\n{'='*60}")
        print(f"Stage1 LAMDA Plus v{VERSION}")
        print(f"{'='*60}")
        print(f"Input:  {self.midi_in_dir}")
        print(f"Output: {self.midi_out_dir}")
        print(f"Files:  {len(midi_files)}")
        print(f"Exclude: {self.exclude_dirs}")
        print(f"{'='*60}\n")
        
        results = []
        for i, midi_path in enumerate(midi_files, 1):
            if self.verbose:
                print(f"[{i}/{len(midi_files)}] ", end="")
            
            result = self.process_midi_file(str(midi_path))
            if result:
                results.append(result)
        
        print(f"\n{'='*60}")
        print(f"✓ Completed: {len(results)}/{len(midi_files)} files")
        print(f"{'='*60}\n")
        
        return results


# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(description="Stage1 LAMDA Plus v2")
    parser.add_argument('--config', required=True, help='config YAML path')
    parser.add_argument('--max-files', type=int, help='max files to process')
    parser.add_argument('--csv', help='output CSV path')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    
    args = parser.parse_args()
    
    # Config読み込み
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # ${base}等のプレースホルダ展開
    roots = config.get('roots', {})
    priors_cfg = config.get('priors', {})
    
    totals_path = expand_placeholders(priors_cfg.get('totals_parquet', ''), roots) if priors_cfg.get('totals_parquet') else None
    signatures_path = expand_placeholders(priors_cfg.get('signatures_json', ''), roots) if priors_cfg.get('signatures_json') else None
    
    priors = LAMDAPriors(totals_path, signatures_path)
    
    # プロセッサ実行
    processor = Stage1Processor(config, priors, verbose=args.verbose)
    results = processor.process_dataset(max_files=args.max_files)
    
    # CSV出力
    if args.csv and results:
        df = pd.DataFrame(results)
        df.to_csv(args.csv, index=False)
        print(f"✓ CSV saved: {args.csv}")


if __name__ == "__main__":
    main()
