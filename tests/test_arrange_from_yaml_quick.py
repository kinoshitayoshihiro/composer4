#!/usr/bin/env python3
"""
YAML→MIDI Arranger Quick Tests

ArrangeFromYAMLの動作確認（軽量テスト）

Tests:
1. 初期化テスト（構造YAML読み込み、Generators初期化）
2. 単一セクション生成テスト（4楽器 × 1セクション）
3. 複数セクション生成テスト（4楽器 × 3セクション）
4. MIDI出力テスト（ファイル保存確認）
5. Emotion推定テスト（セクション名 → Emotion）
"""

import sys
import pathlib
import tempfile
import yaml
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts.arrange.arrange_from_yaml import ArrangeFromYAML


def create_mock_structure_yaml(num_sections: int = 3) -> Dict[str, Any]:
    """
    モック構造YAML作成
    
    Args:
        num_sections: セクション数
    
    Returns:
        構造dict
    """
    sections = []
    chords = {}
    
    section_templates = [
        {"label": "Intro", "duration_measures": 4},
        {"label": "Verse", "duration_measures": 8},
        {"label": "Chorus", "duration_measures": 8},
        {"label": "Bridge", "duration_measures": 4},
        {"label": "Outro", "duration_measures": 4}
    ]
    
    cumulative_time = 0.0
    for i in range(num_sections):
        template = section_templates[i % len(section_templates)]
        section_label = f"{template['label']}{i+1}" if num_sections > len(section_templates) else template['label']
        
        duration = template['duration_measures'] * 2.0  # 2秒/measure仮定
        sections.append({
            "label": section_label,
            "start_time": cumulative_time,
            "end_time": cumulative_time + duration,
            "duration_measures": template['duration_measures']
        })
        
        # Chord progression
        chords[section_label] = [
            {"time": cumulative_time, "chord": "C"},
            {"time": cumulative_time + duration / 4, "chord": "G"},
            {"time": cumulative_time + duration / 2, "chord": "Am"},
            {"time": cumulative_time + 3 * duration / 4, "chord": "F"}
        ]
        
        cumulative_time += duration
    
    structure = {
        "tempo_map": {
            "global_tempo": 120.0,
            "beat_times": [0.5 * i for i in range(int(cumulative_time * 2))]
        },
        "sections": sections,
        "chords": chords,
        "drums_hits": {
            "kick": [0.5 * i for i in range(int(cumulative_time * 2)) if i % 2 == 0],
            "snare": [0.5 * i for i in range(int(cumulative_time * 2)) if i % 4 == 2],
            "hihat": [0.25 * i for i in range(int(cumulative_time * 4))]
        },
        "bass_contour": []
    }
    
    return structure


def test_initialization():
    """Test 1: 初期化テスト"""
    print("\n" + "="*60)
    print("Test 1: Initialization")
    print("="*60)
    
    # Mock YAML作成
    structure = create_mock_structure_yaml(num_sections=2)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(structure, f)
        yaml_path = pathlib.Path(f.name)
    
    try:
        # Arranger初期化
        arranger = ArrangeFromYAML(
            structure_yaml_path=yaml_path,
            enable_stage2=True,
            enable_quality_gates=False,
            verbose=False
        )
        
        # 検証
        assert arranger.structure is not None, "❌ Structure not loaded"
        assert arranger.structure['tempo_map']['global_tempo'] == 120.0, "❌ Tempo mismatch"
        assert len(arranger.structure['sections']) == 2, "❌ Section count mismatch"
        
        # Generators存在確認（現在はGuitar/Strings Stage2のみ）
        assert arranger.guitar_gen is not None, "❌ Guitar generator not initialized"
        assert arranger.strings_gen is not None, "❌ Strings generator not initialized"
        
        print("✅ Initialization successful")
        print(f"   Tempo: {arranger.structure['tempo_map']['global_tempo']} BPM")
        print(f"   Sections: {len(arranger.structure['sections'])}")
        print(f"   Generators: 2 initialized (Guitar, Strings)")
        print("✅ Test 1 Passed!")
    
    finally:
        yaml_path.unlink()


def test_single_section_generation():
    """Test 2: 単一セクション生成テスト"""
    print("\n" + "="*60)
    print("Test 2: Single Section Generation")
    print("="*60)
    
    # Mock YAML作成（1セクション）
    structure = create_mock_structure_yaml(num_sections=1)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(structure, f)
        yaml_path = pathlib.Path(f.name)
    
    try:
        # Arranger初期化
        arranger = ArrangeFromYAML(
            structure_yaml_path=yaml_path,
            enable_stage2=True,
            enable_quality_gates=False,
            verbose=False
        )
        
        # 生成（output_dirは一時ディレクトリ）
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)
            output_files = arranger.generate_all(output_dir)
            
            # 検証
            assert len(output_files) > 0, "❌ No output files generated"
            
            # Full scoreファイル確認
            full_score = output_dir / "full_score.mid"
            assert full_score.exists(), "❌ Full score MIDI not found"
            
            print(f"✅ Generated {len(output_files)} MIDI files")
            print(f"   Section: {structure['sections'][0]['label']}")
            print(f"   Files:")
            for f in output_files:
                print(f"     - {f.name}")
            print("✅ Test 2 Passed!")
    
    finally:
        yaml_path.unlink()


def test_multi_section_generation():
    """Test 3: 複数セクション生成テスト"""
    print("\n" + "="*60)
    print("Test 3: Multi-Section Generation")
    print("="*60)
    
    # Mock YAML作成（3セクション）
    structure = create_mock_structure_yaml(num_sections=3)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(structure, f)
        yaml_path = pathlib.Path(f.name)
    
    try:
        # Arranger初期化
        arranger = ArrangeFromYAML(
            structure_yaml_path=yaml_path,
            enable_stage2=True,
            enable_quality_gates=False,
            verbose=False
        )
        
        # 生成
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)
            output_files = arranger.generate_all(output_dir)
            
            # 検証
            assert len(output_files) > 0, "❌ No output files generated"
            
            # Full scoreファイル読み込み
            from music21 import converter
            full_score_path = output_dir / "full_score.mid"
            assert full_score_path.exists(), "❌ Full score MIDI not found"
            
            score = converter.parse(full_score_path)
            parts = score.parts
            
            print(f"✅ Generated {len(output_files)} MIDI files")
            print(f"   Sections: {[s['label'] for s in structure['sections']]}")
            print(f"   Parts in full score: {len(parts)}")
            
            # 各Part統計
            for part in parts:
                notes = list(part.flatten().notes)
                if len(notes) > 0:
                    print(f"     - {part.partName}: {len(notes)} notes")
            
            print("✅ Test 3 Passed!")
    
    finally:
        yaml_path.unlink()


def test_midi_output():
    """Test 4: MIDI出力テスト"""
    print("\n" + "="*60)
    print("Test 4: MIDI Output")
    print("="*60)
    
    # Mock YAML作成（2セクション）
    structure = create_mock_structure_yaml(num_sections=2)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(structure, f)
        yaml_path = pathlib.Path(f.name)
    
    try:
        # Arranger初期化
        arranger = ArrangeFromYAML(
            structure_yaml_path=yaml_path,
            enable_stage2=True,
            enable_quality_gates=False,
            verbose=False
        )
        
        # 生成
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)
            output_files = arranger.generate_all(output_dir)
            
            # 検証: 各ファイルサイズ > 0
            print(f"✅ Output files: {len(output_files)}")
            for f in output_files:
                assert f.exists(), f"❌ File not found: {f}"
                file_size = f.stat().st_size
                assert file_size > 0, f"❌ Empty file: {f}"
                print(f"   ✓ {f.name}: {file_size} bytes")
            
            print("✅ Test 4 Passed!")
    
    finally:
        yaml_path.unlink()


def test_emotion_estimation():
    """Test 5: Emotion推定テスト"""
    print("\n" + "="*60)
    print("Test 5: Emotion Estimation")
    print("="*60)
    
    # Mock YAML作成（各セクションタイプ）
    structure = create_mock_structure_yaml(num_sections=5)
    structure['sections'][0]['label'] = 'Intro'
    structure['sections'][1]['label'] = 'Verse'
    structure['sections'][2]['label'] = 'Chorus'
    structure['sections'][3]['label'] = 'Bridge'
    structure['sections'][4]['label'] = 'Outro'
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(structure, f)
        yaml_path = pathlib.Path(f.name)
    
    try:
        # Arranger初期化
        arranger = ArrangeFromYAML(
            structure_yaml_path=yaml_path,
            enable_stage2=True,
            enable_quality_gates=False,
            verbose=False
        )
        
        # Emotion推定テスト
        emotions = {
            'Intro': arranger._estimate_emotion('Intro'),
            'Verse': arranger._estimate_emotion('Verse'),
            'Chorus': arranger._estimate_emotion('Chorus'),
            'Bridge': arranger._estimate_emotion('Bridge'),
            'Outro': arranger._estimate_emotion('Outro')
        }
        
        # 検証
        expected = {
            'Intro': 'calm',
            'Verse': 'neutral',
            'Chorus': 'happy',
            'Bridge': 'dramatic',
            'Outro': 'calm'
        }
        
        print("✅ Emotion estimation:")
        for section, emotion in emotions.items():
            exp = expected[section]
            match = "✓" if emotion == exp else "✗"
            print(f"   {match} {section}: {emotion} (expected: {exp})")
            assert emotion == exp, f"❌ Emotion mismatch for {section}"
        
        print("✅ Test 5 Passed!")
    
    finally:
        yaml_path.unlink()


def run_all_tests():
    """全テスト実行"""
    print("\n" + "="*60)
    print("🎼 ArrangeFromYAML Quick Tests")
    print("="*60)
    
    tests = [
        test_initialization,
        test_single_section_generation,
        test_multi_section_generation,
        test_midi_output,
        test_emotion_estimation
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"📊 Test Summary: {passed}/{len(tests)} passed")
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
