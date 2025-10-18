"""
test_benchmark_suite.py - ベンチマーク曲集テストスイート

Usage:
    pytest tests/test_benchmark_suite.py -v
"""

import json
from pathlib import Path

import pytest
import yaml


# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / 'configs' / 'benchmarks'
BENCHMARK_JSON = PROJECT_ROOT / 'multi_song_benchmark.json'


class TestBenchmarkYAMLs:
    """ベンチマークYAMLファイルの検証"""
    
    def test_benchmarks_directory_exists(self):
        """benchmarksディレクトリが存在する"""
        assert BENCHMARKS_DIR.exists(), f"Benchmarks directory not found: {BENCHMARKS_DIR}"
    
    def test_all_yaml_files_count(self):
        """12個のYAMLファイルが存在する"""
        yaml_files = list(BENCHMARKS_DIR.glob('*.yaml'))
        assert len(yaml_files) == 12, f"Expected 12 YAML files, found {len(yaml_files)}"
    
    @pytest.mark.parametrize('genre,count', [
        ('pop', 3),
        ('rock', 3),
        ('edm', 3),
        ('ballad', 3),
    ])
    def test_genre_yaml_count(self, genre: str, count: int):
        """各ジャンル3曲ずつ存在する"""
        genre_files = list(BENCHMARKS_DIR.glob(f'{genre}_*.yaml'))
        assert len(genre_files) == count, f"Expected {count} {genre} files, found {len(genre_files)}"
    
    @pytest.mark.parametrize('difficulty', ['simple', 'medium', 'complex'])
    def test_difficulty_yaml_count(self, difficulty: str):
        """各難易度4曲ずつ存在する (4ジャンル × 3難易度)"""
        difficulty_files = [
            f for f in BENCHMARKS_DIR.glob('*.yaml')
            if difficulty in f.stem
        ]
        assert len(difficulty_files) == 4, f"Expected 4 {difficulty} files, found {len(difficulty_files)}"


class TestBenchmarkYAMLStructure:
    """YAML構造の検証"""
    
    @pytest.fixture
    def all_yamls(self):
        """全YAMLファイルを読み込み"""
        yaml_files = sorted(BENCHMARKS_DIR.glob('*.yaml'))
        yamls = []
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                yamls.append((yaml_file.name, data))
        
        return yamls
    
    def test_all_yamls_have_meta(self, all_yamls):
        """全YAMLにmetaセクションが存在する"""
        for name, data in all_yamls:
            assert 'meta' in data, f"{name}: Missing 'meta' section"
    
    def test_all_yamls_have_global(self, all_yamls):
        """全YAMLにglobalセクションが存在する"""
        for name, data in all_yamls:
            assert 'global' in data, f"{name}: Missing 'global' section"
    
    def test_all_yamls_have_sections(self, all_yamls):
        """全YAMLにsectionsが存在する"""
        for name, data in all_yamls:
            assert 'sections' in data, f"{name}: Missing 'sections'"
            assert len(data['sections']) >= 3, f"{name}: Too few sections (< 3)"
    
    def test_all_yamls_have_quality_thresholds(self, all_yamls):
        """全YAMLにquality_thresholdsが存在する"""
        for name, data in all_yamls:
            assert 'quality_thresholds' in data, f"{name}: Missing 'quality_thresholds'"
    
    def test_meta_required_fields(self, all_yamls):
        """metaに必須フィールドが存在する"""
        required_fields = ['title', 'genre', 'style', 'difficulty', 'seed', 'expected_metrics']
        
        for name, data in all_yamls:
            meta = data.get('meta', {})
            
            for field in required_fields:
                assert field in meta, f"{name}: Missing meta.{field}"
    
    def test_expected_metrics_fields(self, all_yamls):
        """expected_metricsに必須フィールドが存在する"""
        required_fields = ['total_bars', 'sections', 'instruments', 'tempo_bpm', 'key']
        
        for name, data in all_yamls:
            expected = data.get('meta', {}).get('expected_metrics', {})
            
            for field in required_fields:
                assert field in expected, f"{name}: Missing expected_metrics.{field}"
    
    def test_unique_seeds(self, all_yamls):
        """全ての曲でseedがユニーク"""
        seeds = [data['meta']['seed'] for name, data in all_yamls]
        assert len(seeds) == len(set(seeds)), "Duplicate seeds found"
    
    def test_seed_ranges_by_genre(self, all_yamls):
        """ジャンル別にseedの範囲が正しい"""
        seed_ranges = {
            'Pop': (1001, 1003),
            'Rock': (2001, 2003),
            'EDM': (3001, 3003),
            'Ballad': (4001, 4003),
        }
        
        for name, data in all_yamls:
            genre = data['meta']['genre']
            seed = data['meta']['seed']
            
            if genre in seed_ranges:
                min_seed, max_seed = seed_ranges[genre]
                assert min_seed <= seed <= max_seed, \
                    f"{name}: Seed {seed} out of range for {genre} ({min_seed}-{max_seed})"


class TestBenchmarkJSON:
    """ベンチマークJSON検証"""
    
    @pytest.fixture
    def benchmark_json(self):
        """ベンチマークJSON読み込み"""
        if not BENCHMARK_JSON.exists():
            pytest.skip(f"Benchmark JSON not found: {BENCHMARK_JSON}")
        
        with open(BENCHMARK_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_json_exists(self):
        """multi_song_benchmark.jsonが存在する"""
        assert BENCHMARK_JSON.exists(), f"Benchmark JSON not found: {BENCHMARK_JSON}"
    
    def test_json_structure(self, benchmark_json):
        """JSON構造が正しい"""
        assert 'version' in benchmark_json
        assert 'generated' in benchmark_json
        assert 'total_songs' in benchmark_json
        assert 'songs' in benchmark_json
    
    def test_json_song_count(self, benchmark_json):
        """12曲すべて含まれている"""
        assert benchmark_json['total_songs'] == 12
        assert len(benchmark_json['songs']) == 12
    
    def test_json_genre_counts(self, benchmark_json):
        """各ジャンル3曲ずつ"""
        genres = benchmark_json.get('genres', {})
        
        assert genres.get('Pop') == 3
        assert genres.get('Rock') == 3
        assert genres.get('EDM') == 3
        assert genres.get('Ballad') == 3
    
    def test_all_songs_have_metadata(self, benchmark_json):
        """全曲にメタデータが存在する"""
        for song in benchmark_json['songs']:
            assert 'id' in song
            assert 'file' in song
            assert 'metadata' in song
            assert 'expected_metrics' in song
            assert 'quality_thresholds' in song


class TestBenchmarkQualityThresholds:
    """品質閾値の妥当性検証"""
    
    @pytest.fixture
    def all_yamls(self):
        """全YAMLファイルを読み込み"""
        yaml_files = sorted(BENCHMARKS_DIR.glob('*.yaml'))
        yamls = []
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                yamls.append((yaml_file.name, data))
        
        return yamls
    
    def test_drums_thresholds_valid(self, all_yamls):
        """ドラム品質閾値が妥当な範囲"""
        for name, data in all_yamls:
            thresholds = data.get('quality_thresholds', {}).get('drums', {})
            
            if 'kick_onbeat_ratio_min' in thresholds:
                ratio = thresholds['kick_onbeat_ratio_min']
                assert 0.0 <= ratio <= 1.0, f"{name}: Invalid kick_onbeat_ratio_min: {ratio}"
            
            if 'quality_score_min' in thresholds:
                score = thresholds['quality_score_min']
                assert 0.0 <= score <= 1.0, f"{name}: Invalid quality_score_min: {score}"
    
    def test_bass_thresholds_valid(self, all_yamls):
        """ベース品質閾値が妥当な範囲"""
        for name, data in all_yamls:
            thresholds = data.get('quality_thresholds', {}).get('bass', {})
            
            if 'root_accuracy_min' in thresholds:
                accuracy = thresholds['root_accuracy_min']
                assert 0.0 <= accuracy <= 1.0, f"{name}: Invalid root_accuracy_min: {accuracy}"
    
    def test_piano_thresholds_valid(self, all_yamls):
        """ピアノ品質閾値が妥当な範囲"""
        for name, data in all_yamls:
            thresholds = data.get('quality_thresholds', {}).get('piano', {})
            
            if 'chord_tone_rate_min' in thresholds:
                rate = thresholds['chord_tone_rate_min']
                assert 0.0 <= rate <= 1.0, f"{name}: Invalid chord_tone_rate_min: {rate}"
            
            if 'velocity_std_range' in thresholds:
                range_val = thresholds['velocity_std_range']
                assert isinstance(range_val, list), f"{name}: velocity_std_range should be list"
                assert len(range_val) == 2, f"{name}: velocity_std_range should have 2 values"
                assert range_val[0] < range_val[1], f"{name}: Invalid velocity_std_range order"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
