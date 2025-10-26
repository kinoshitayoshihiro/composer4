#!/usr/bin/env python3
"""
和声学習データセット作成 - Gold/Silver曲からcontext-window形式で抽出
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def load_qa_report(report_path: Path) -> Dict[str, Dict]:
    """QAレポートから曲ごとの品質情報を読み込み"""
    with open(report_path) as f:
        report = json.load(f)
    
    song_quality = {}
    for r in report['results']:
        if r['status'] == 'success':
            song_id = r['song_id']
            bronze_rate = r.get('bronze_rate', 1.0)
            avg_conf = r.get('avg_confidence', 0.0)
            total_events = r.get('total_events', 0)
            
            # Gold/Silver判定
            if bronze_rate <= 0.2 and avg_conf >= 0.5:
                quality = 'gold'
            elif bronze_rate <= 0.4 and avg_conf >= 0.4:
                quality = 'silver'
            else:
                quality = 'bronze'
            
            song_quality[song_id] = {
                'quality': quality,
                'bronze_rate': bronze_rate,
                'avg_confidence': avg_conf,
                'total_events': total_events
            }
    
    return song_quality


def extract_training_sequences(
    song_dir: Path,
    context_bars: int = 8,
    min_events: int = 10
) -> Optional[List[Dict]]:
    """
    1曲から学習用シーケンスを抽出
    
    Args:
        song_dir: midi_guide/{song_id}
        context_bars: コンテキストウィンドウ（小節数）
        min_events: 最小イベント数（これ未満はスキップ）
    
    Returns:
        List of training sequences
    """
    song_id = song_dir.name
    chordmap_path = song_dir / 'chordmap.json'
    bars_path = song_dir / f'{song_id}.bars.parquet'
    sections_path = song_dir / 'sections.json'
    
    # 必須ファイルチェック
    if not all([chordmap_path.exists(), bars_path.exists()]):
        return None
    
    try:
        # 読み込み
        with open(chordmap_path) as f:
            chordmap = json.load(f)
        
        bars_df = pd.read_parquet(bars_path)
        
        sections = {}
        if sections_path.exists():
            with open(sections_path) as f:
                sections = json.load(f)
        
        events = chordmap.get('events', [])
        if len(events) < min_events:
            return None
        
        # barsにイベントをマッピング
        bars_df['chord_root'] = None
        bars_df['chord_quality'] = None
        bars_df['chord_tensions'] = None
        bars_df['chord_confidence'] = None
        bars_df['label_strength'] = None
        
        for event in events:
            time_ql = event['time']
            # 最も近い小節を検索
            closest_bar = bars_df.iloc[(bars_df['time_ql'] - time_ql).abs().argsort()[0]]
            bar_idx = closest_bar['bar_index']
            
            bars_df.loc[bars_df['bar_index'] == bar_idx, 'chord_root'] = event.get('root', 'C')
            bars_df.loc[bars_df['bar_index'] == bar_idx, 'chord_quality'] = event.get('quality', 'maj')
            bars_df.loc[bars_df['bar_index'] == bar_idx, 'chord_tensions'] = str(event.get('tensions', []))
            bars_df.loc[bars_df['bar_index'] == bar_idx, 'chord_confidence'] = event.get('confidence', 0.5)
            bars_df.loc[bars_df['bar_index'] == bar_idx, 'label_strength'] = event.get('label_strength', 'bronze')
        
        # 前方補完（小節内でコードが変わらない想定）
        bars_df['chord_root'] = bars_df['chord_root'].fillna(method='ffill')
        bars_df['chord_quality'] = bars_df['chord_quality'].fillna(method='ffill')
        bars_df['chord_tensions'] = bars_df['chord_tensions'].fillna(method='ffill')
        bars_df['chord_confidence'] = bars_df['chord_confidence'].fillna(method='ffill')
        bars_df['label_strength'] = bars_df['label_strength'].fillna(method='ffill')
        
        # 後方補完（曲頭の空白対策）
        bars_df['chord_root'] = bars_df['chord_root'].fillna(method='bfill')
        bars_df['chord_quality'] = bars_df['chord_quality'].fillna(method='bfill')
        
        # セクション情報追加
        bars_df['section'] = 'Unknown'
        for label_info in sections.get('labels', []):
            section_time = label_info['time']
            section_label = label_info['label']
            # その時刻以降の小節に適用
            bars_df.loc[bars_df['time_ql'] >= section_time, 'section'] = section_label
        
        # テンポ/拍子追加（簡易）
        bars_df['tempo'] = 120.0  # デフォルト
        for tempo_info in sections.get('tempi', []):
            tempo_time = tempo_info['time']
            bpm = tempo_info['bpm']
            bars_df.loc[bars_df['time_ql'] >= tempo_time, 'tempo'] = bpm
        
        bars_df['time_sig_num'] = 4
        bars_df['time_sig_den'] = 4
        for ts_info in sections.get('time_signatures', []):
            ts_time = ts_info['time']
            bars_df.loc[bars_df['time_ql'] >= ts_time, 'time_sig_num'] = ts_info['num']
            bars_df.loc[bars_df['time_ql'] >= ts_time, 'time_sig_den'] = ts_info['den']
        
        # コンテキストウィンドウで分割
        sequences = []
        max_bar = len(bars_df)
        
        for start_bar in range(0, max_bar, context_bars):
            end_bar = min(start_bar + context_bars, max_bar)
            window = bars_df.iloc[start_bar:end_bar]
            
            if len(window) < context_bars // 2:  # 半分未満はスキップ
                continue
            
            # シーケンス作成
            seq = {
                'song_id': song_id,
                'start_bar': int(start_bar),
                'end_bar': int(end_bar),
                'num_bars': len(window),
                'section': window['section'].mode()[0] if len(window['section'].mode()) > 0 else 'Unknown',
                'tempo': float(window['tempo'].mean()),
                'time_sig': f"{int(window['time_sig_num'].mode()[0])}/{int(window['time_sig_den'].mode()[0])}",
                'chord_sequence': [
                    {
                        'bar': int(row['bar_index']),
                        'root': row['chord_root'],
                        'quality': row['chord_quality'],
                        'tensions': row['chord_tensions'],
                        'confidence': float(row['chord_confidence']) if pd.notna(row['chord_confidence']) else 0.5,
                        'label_strength': row['label_strength']
                    }
                    for _, row in window.iterrows()
                ],
                'avg_confidence': float(window['chord_confidence'].mean()),
                'gold_ratio': (window['label_strength'] == 'gold').sum() / len(window),
                'silver_ratio': (window['label_strength'] == 'silver').sum() / len(window),
                'bronze_ratio': (window['label_strength'] == 'bronze').sum() / len(window)
            }
            
            sequences.append(seq)
        
        return sequences
    
    except Exception as e:
        logger.error(f"[{song_id}] Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="和声学習データセット作成")
    parser.add_argument("--input-root", type=Path, required=True, help="midi_guideルート")
    parser.add_argument("--qa-report", type=Path, required=True, help="QAレポート（qa_chordmap_full_reestimation.json）")
    parser.add_argument("--quality", default="gold,silver", help="対象品質（カンマ区切り）")
    parser.add_argument("--context-bars", type=int, default=8, help="コンテキストウィンドウ（小節数）")
    parser.add_argument("--output-dir", type=Path, default=Path("harmony_dataset"), help="出力ディレクトリ")
    parser.add_argument("--workers", type=int, default=4, help="並列数")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # 出力ディレクトリ作成
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # QAレポート読み込み
    logger.info(f"Loading QA report from {args.qa_report}...")
    song_quality = load_qa_report(args.qa_report)
    
    # 対象品質フィルタ
    target_qualities = set(args.quality.split(','))
    target_songs = {
        song_id: info for song_id, info in song_quality.items()
        if info['quality'] in target_qualities
    }
    
    logger.info(f"Target songs: {len(target_songs)} ({', '.join(target_qualities)})")
    for q in target_qualities:
        count = sum(1 for info in target_songs.values() if info['quality'] == q)
        logger.info(f"  {q}: {count}")
    
    # 曲ディレクトリ収集
    all_song_dirs = [
        args.input_root / song_id
        for song_id in target_songs.keys()
        if (args.input_root / song_id).exists()
    ]
    
    logger.info(f"Processing {len(all_song_dirs)} songs...")
    
    # シーケンス抽出
    all_sequences = []
    for song_dir in tqdm(all_song_dirs, desc="Extracting"):
        sequences = extract_training_sequences(song_dir, args.context_bars)
        if sequences:
            all_sequences.extend(sequences)
    
    logger.info(f"Extracted {len(all_sequences)} sequences from {len(all_song_dirs)} songs")
    
    # DataFrame化して保存
    df = pd.DataFrame(all_sequences)
    output_path = args.output_dir / "training_sequences.parquet"
    df.to_parquet(output_path, index=False)
    
    logger.info(f"✓ Saved to {output_path}")
    
    # 統計サマリー
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Total sequences: {len(df)}")
    print(f"Total songs: {df['song_id'].nunique()}")
    print(f"Avg bars/sequence: {df['num_bars'].mean():.1f}")
    print(f"Avg confidence: {df['avg_confidence'].mean():.3f}")
    print(f"\nLabel strength distribution:")
    print(f"  Gold ratio: {df['gold_ratio'].mean()*100:.1f}%")
    print(f"  Silver ratio: {df['silver_ratio'].mean()*100:.1f}%")
    print(f"  Bronze ratio: {df['bronze_ratio'].mean()*100:.1f}%")
    print(f"\nSection distribution:")
    print(df['section'].value_counts().head(10))
    print("="*60)


if __name__ == "__main__":
    main()
