"""
Quick LAMDa DB Build Script for Vertex AI Colab
Vertex AI Colab Enterpriseのノートブックで実行
"""

# セル1: セットアップ
!git clone --depth 1 https://github.com/kinoshitayoshihiro/composer4.git /home/jupyter/composer4
%cd /home/jupyter/composer4
!pip install -q music21 numpy

# セル2: データダウンロード & 解凍
!mkdir -p data/Los-Angeles-MIDI
!gsutil cp gs://otocotoba/data/lamda/CHORDS_DATA.tar.gz data/Los-Angeles-MIDI/
!tar -xzf data/Los-Angeles-MIDI/CHORDS_DATA.tar.gz -C data/Los-Angeles-MIDI/
!ls -lh data/Los-Angeles-MIDI/CHORDS_DATA/ | head -5

# セル3: データベース構築（これが一番時間かかる）
import sys
sys.path.insert(0, '/home/jupyter/composer4')
from lamda_analyzer import LAMDaAnalyzer
from pathlib import Path
import time

chords_dir = Path('data/Los-Angeles-MIDI/CHORDS_DATA')
print(f"Found {len(list(chords_dir.glob('*.pickle')))} pickle files")

start_time = time.time()
analyzer = LAMDaAnalyzer(chords_dir)
analyzer.analyze_all_files()

elapsed = time.time() - start_time
print(f"\n✅ Database built in {elapsed/60:.1f} minutes")

# セル4: 結果をGCSにアップロード
!gsutil cp data/lamda_progressions.db gs://otocotoba/data/lamda/
print("\n✅ Database uploaded to GCS!")
print("Download with: gsutil cp gs://otocotoba/data/lamda/lamda_progressions.db .")

# セル5: 統計確認
from lamda_analyzer import ProgressionRecommender

db_path = Path('data/lamda_progressions.db')
recommender = ProgressionRecommender(db_path)
stats = recommender.get_statistics()

print("\n📊 Database Statistics:")
print(f"  Total progressions: {stats.get('total_progressions', 0):,}")
print(f"  Unique keys: {len(stats.get('key_distribution', {}))}")
print(f"  Top 5 keys:")
for key, count in list(stats.get('key_distribution', {}).items())[:5]:
    print(f"    {key}: {count:,}")
