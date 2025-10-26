#!/usr/bin/env python3
"""
全曲のchordmap再推定バッチ - raw/fixed分離出力＋QAレポート
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

# 同じディレクトリのchord_reestimation.pyをインポート
sys.path.insert(0, str(Path(__file__).parent))
from chord_reestimation import reestimate_chordmap

logger = logging.getLogger(__name__)

def process_song(song_dir: Path, write_raw: bool = True) -> Dict:
    """
    1曲の処理
    
    Args:
        song_dir: midi_guide/{song_id}
        write_raw: raw版も保存するか
    
    Returns:
        QAメトリクス（song_id付き）
    """
    song_id = song_dir.name
    chordmap_path = song_dir / "chordmap.json"
    
    if not chordmap_path.exists():
        return {
            "song_id": song_id,
            "status": "no_chordmap",
            "error": "chordmap.json not found"
        }
    
    try:
        # 読み込み
        with open(chordmap_path, "r") as f:
            original = json.load(f)
        
        # raw版保存（初回のみ、既存があればスキップ）
        raw_path = song_dir / "chordmap.raw.json"
        if write_raw and not raw_path.exists():
            with open(raw_path, "w") as f:
                json.dump(original, f, indent=2, ensure_ascii=False)
        
        # 再推定
        fixed, qa = reestimate_chordmap(original)
        
        # fixed版を元のパスに上書き
        with open(chordmap_path, "w") as f:
            json.dump(fixed, f, indent=2, ensure_ascii=False)
        
        # QAメトリクス保存
        qa_dir = song_dir / "qa"
        qa_dir.mkdir(exist_ok=True)
        qa_path = qa_dir / "chordmap_lint.json"
        with open(qa_path, "w") as f:
            json.dump(qa, f, indent=2)
        
        return {
            "song_id": song_id,
            "status": "success",
            **qa
        }
    
    except Exception as e:
        logger.error(f"[{song_id}] Error: {e}")
        return {
            "song_id": song_id,
            "status": "error",
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="全曲chordmap再推定")
    parser.add_argument("--input-root", required=True, help="midi_guideルート")
    parser.add_argument("--workers", type=int, default=4, help="並列数")
    parser.add_argument("--write-raw", action="store_true", default=True, help="raw版も保存")
    parser.add_argument("--output-report", default="qa_chordmap_report.json", help="集約レポート")
    parser.add_argument("--sample", type=int, help="サンプル曲数（テスト用）")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # 全曲列挙
    root = Path(args.input_root)
    all_songs = sorted([d for d in root.iterdir() if d.is_dir()])
    
    if args.sample:
        all_songs = all_songs[:args.sample]
    
    logger.info(f"Processing {len(all_songs)} songs with {args.workers} workers...")
    
    # 並列処理
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_song, song_dir, args.write_raw): song_dir
            for song_dir in all_songs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reestimating"):
            result = future.result()
            results.append(result)
    
    # 集約
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    no_chordmap = [r for r in results if r["status"] == "no_chordmap"]
    
    # 統計
    if success:
        avg_bronze_rate = sum(r.get("bronze_rate", 0) for r in success) / len(success)
        avg_N_rate = sum(r.get("N_rate", 0) for r in success) / len(success)
        avg_confidence = sum(r.get("avg_confidence", 0) for r in success) / len(success)
        
        # Gate判定（学習使用可否）
        gold_songs = [r for r in success if r.get("bronze_rate", 1.0) <= 0.2 and r.get("avg_confidence", 0) >= 0.5]
        silver_songs = [r for r in success if 0.2 < r.get("bronze_rate", 1.0) <= 0.4 and r.get("avg_confidence", 0) >= 0.4]
        bronze_songs = [r for r in success if r.get("bronze_rate", 1.0) > 0.4 or r.get("avg_confidence", 0) < 0.4]
    else:
        avg_bronze_rate = avg_N_rate = avg_confidence = 0.0
        gold_songs = silver_songs = bronze_songs = []
    
    summary = {
        "total_songs": len(all_songs),
        "success": len(success),
        "errors": len(errors),
        "no_chordmap": len(no_chordmap),
        "statistics": {
            "avg_bronze_rate": round(avg_bronze_rate, 3),
            "avg_N_rate": round(avg_N_rate, 3),
            "avg_confidence": round(avg_confidence, 3)
        },
        "quality_gate": {
            "gold_count": len(gold_songs),
            "silver_count": len(silver_songs),
            "bronze_count": len(bronze_songs),
            "gold_rate": len(gold_songs) / len(success) if success else 0.0
        },
        "results": results
    }
    
    # レポート保存
    report_path = Path(args.output_report)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # サマリー表示
    print("\n" + "="*60)
    print("📊 Reestimation Summary")
    print("="*60)
    print(f"Total songs:      {len(all_songs)}")
    print(f"✓ Success:        {len(success)}")
    print(f"✗ Errors:         {len(errors)}")
    print(f"⊘ No chordmap:    {len(no_chordmap)}")
    print()
    print("Statistics (success only):")
    print(f"  Avg bronze rate: {avg_bronze_rate*100:.1f}%")
    print(f"  Avg N-Chord rate: {avg_N_rate*100:.1f}%")
    print(f"  Avg confidence:   {avg_confidence:.3f}")
    print()
    print("Quality Gate (for learning):")
    print(f"  🥇 Gold songs:   {len(gold_songs)} ({len(gold_songs)/len(success)*100:.1f}%)")
    print(f"  🥈 Silver songs: {len(silver_songs)} ({len(silver_songs)/len(success)*100:.1f}%)")
    print(f"  🥉 Bronze songs: {len(bronze_songs)} ({len(bronze_songs)/len(success)*100:.1f}%)")
    print()
    print(f"📄 Report: {report_path}")
    print("="*60)
    
    # Bronze曲リスト出力（学習除外候補）
    if bronze_songs:
        bronze_list_path = report_path.parent / "bronze_songs.txt"
        with open(bronze_list_path, "w") as f:
            for r in bronze_songs:
                f.write(f"{r['song_id']}\t{r.get('bronze_rate', 0):.3f}\t{r.get('avg_confidence', 0):.3f}\n")
        print(f"⚠️  Bronze songs list: {bronze_list_path}")

if __name__ == "__main__":
    main()
