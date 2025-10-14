#!/usr/bin/env python3
"""
Phase 5.0: Current State Analysis
Phase 5開始時の現状分析とベースライン品質確立
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import re

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_file_emotion_integration(file_path: Path) -> Dict[str, Any]:
    """
    個別ファイルのemotion統合状態を分析
    
    Returns:
        {
            'has_emotion_loader_import': bool,
            'has_section_param': bool,
            'has_emotion_profile_param': bool,
            'has_adjustments_storage': bool,
            'has_parameter_application': bool,
            'line_counts': dict
        }
    """
    if not file_path.exists():
        return {'error': f'File not found: {file_path}'}
    
    content = file_path.read_text(encoding='utf-8')
    
    # Import確認
    has_emotion_import = bool(re.search(
        r'from utils\.emotion_loader import',
        content
    ))
    
    # Parameter確認
    has_section_param = bool(re.search(
        r'section:\s*str\s*=',
        content
    ))
    
    has_emotion_param = bool(re.search(
        r'emotion_profile:\s*(?:str\s*\|\s*None|Optional\[str\])',
        content
    ))
    
    # Adjustments格納確認
    has_storage = bool(re.search(
        r'section_data\[.?_emotion_adjustments.?\]',
        content
    ))
    
    # パラメータ実適用確認（Phase 5で実装予定）
    # velocity_std, notes_per_bar等への実際の適用
    has_application = bool(re.search(
        r'emotion_adj\.get\([\'"](?:velocity_std_multiplier|notes_per_bar_multiplier|strum_consistency_target|velocity_boost|root_emphasis|legato_rate_target|chord_spread_multiplier|hihat_density_multiplier|kick_emphasis)[\'"]',
        content
    ))
    
    # 行数カウント
    lines = content.split('\n')
    line_counts = {
        'total': len(lines),
        'code': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
        'comments': len([l for l in lines if l.strip().startswith('#')]),
        'blank': len([l for l in lines if not l.strip()])
    }
    
    return {
        'file': str(file_path.relative_to(project_root)),
        'has_emotion_loader_import': has_emotion_import,
        'has_section_param': has_section_param,
        'has_emotion_profile_param': has_emotion_param,
        'has_adjustments_storage': has_storage,
        'has_parameter_application': has_application,
        'line_counts': line_counts,
        'phase_4_9_complete': has_emotion_import and has_section_param and has_emotion_param and has_storage,
        'phase_5_ready': has_emotion_import and has_section_param and has_emotion_param and has_storage and not has_application
    }


def analyze_emotion_integration() -> Dict[str, Any]:
    """
    全Generatorファイルのemotion統合状態を分析
    """
    generators = [
        'generator/piano_generator.py',
        'generator/guitar_generator.py',
        'generator/bass_generator.py',
        'generator/strings_generator.py',
        'generator/drum_generator.py'
    ]
    
    results = {}
    
    for gen_file in generators:
        file_path = project_root / gen_file
        instrument = gen_file.split('/')[-1].replace('_generator.py', '')
        results[instrument] = analyze_file_emotion_integration(file_path)
    
    # サマリー
    summary = {
        'total_generators': len(generators),
        'phase_4_9_complete': sum(1 for r in results.values() if r.get('phase_4_9_complete', False)),
        'phase_5_ready': sum(1 for r in results.values() if r.get('phase_5_ready', False)),
        'has_application': sum(1 for r in results.values() if r.get('has_parameter_application', False))
    }
    
    return {
        'generators': results,
        'summary': summary
    }


def check_emotion_loader_utility() -> Dict[str, Any]:
    """
    utils/emotion_loader.pyの状態を確認
    """
    emotion_loader_path = project_root / 'utils' / 'emotion_loader.py'
    
    if not emotion_loader_path.exists():
        return {'error': 'emotion_loader.py not found'}
    
    content = emotion_loader_path.read_text(encoding='utf-8')
    
    # 関数リスト
    functions = re.findall(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)
    
    # 行数
    lines = content.split('\n')
    
    return {
        'exists': True,
        'path': str(emotion_loader_path.relative_to(project_root)),
        'functions': functions,
        'function_count': len(functions),
        'total_lines': len(lines),
        'expected_functions': [
            'load_emotion_mapping',
            'get_emotion_adjustments',
            'get_section_default_emotion',
            'get_section_alternative_emotions',
            'validate_section_constraints',
            'get_transition_rule',
            'get_emotion_profile_info',
            'apply_adjustments_to_params',
            'get_generation_params'
        ]
    }


def check_emotion_mapping_config() -> Dict[str, Any]:
    """
    config/emotion_mapping.yamlの状態を確認
    """
    config_path = project_root / 'config' / 'emotion_mapping.yaml'
    
    if not config_path.exists():
        return {'error': 'emotion_mapping.yaml not found'}
    
    import yaml
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return {
            'exists': True,
            'path': str(config_path.relative_to(project_root)),
            'emotion_profiles': list(config.get('emotion_profiles', {}).keys()) if config else [],
            'sections': list(config.get('sections', {}).keys()) if config else [],
            'instruments': list(config.get('instruments', {}).keys()) if config else [],
            'profile_count': len(config.get('emotion_profiles', {})) if config else 0,
            'section_count': len(config.get('sections', {})) if config else 0,
            'instrument_count': len(config.get('instruments', {})) if config else 0
        }
    except Exception as e:
        return {'error': f'Failed to load YAML: {e}'}


def run_quality_gate_check(instrument: str) -> Dict[str, Any]:
    """
    個別楽器のQuality Gate確認（実際には実行せずメタ情報のみ）
    
    実際のQG実行は時間がかかるため、ここでは:
    - 最新のeval outputファイルの存在確認
    - Schema version確認
    のみ実施
    """
    # Eval outputファイルパターン
    eval_patterns = {
        'piano': 'results/*piano*eval*.json',
        'guitar': 'results/*guitar*eval*.json',
        'bass': 'results/*bass*eval*.json',
        'strings': 'results/*strings*eval*.json',
        'drums': 'results/*drum*eval*.json'
    }
    
    pattern = eval_patterns.get(instrument)
    if not pattern:
        return {'error': f'Unknown instrument: {instrument}'}
    
    # 最新ファイル検索
    from glob import glob
    matching_files = list(glob(str(project_root / pattern)))
    
    if not matching_files:
        return {
            'instrument': instrument,
            'has_eval_output': False,
            'latest_file': None
        }
    
    # 最新ファイル（更新日時順）
    latest_file = max(matching_files, key=lambda p: Path(p).stat().st_mtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        
        return {
            'instrument': instrument,
            'has_eval_output': True,
            'latest_file': str(Path(latest_file).relative_to(project_root)),
            'schema_version': eval_data.get('schema_version'),
            'metrics': list(eval_data.get('metrics', {}).keys()) if 'metrics' in eval_data else [],
            'threshold_flags': eval_data.get('threshold_flags', {}),
            'timestamp': eval_data.get('provenance', {}).get('timestamp')
        }
    except Exception as e:
        return {
            'instrument': instrument,
            'has_eval_output': True,
            'latest_file': str(Path(latest_file).relative_to(project_root)),
            'error': f'Failed to parse: {e}'
        }


def establish_baseline_quality() -> Dict[str, Any]:
    """
    ベースライン品質を確立（メタ情報のみ）
    """
    instruments = ['piano', 'guitar', 'bass', 'strings', 'drums']
    
    results = {}
    for instrument in instruments:
        results[instrument] = run_quality_gate_check(instrument)
    
    # サマリー
    summary = {
        'total_instruments': len(instruments),
        'with_eval_output': sum(1 for r in results.values() if r.get('has_eval_output', False)),
        'schema_1_1': sum(1 for r in results.values() if r.get('schema_version') == '1.1')
    }
    
    return {
        'instruments': results,
        'summary': summary,
        'note': 'Baseline established from existing eval outputs. Full QG run recommended before Phase 5 implementation.'
    }


def analyze_test_coverage() -> Dict[str, Any]:
    """
    テストファイルの存在確認
    """
    test_patterns = [
        'tests/test_*_emotion*.py',
        'tests/test_*_section*.py',
        'tests/test_emotion_loader.py'
    ]
    
    from glob import glob
    
    found_tests = {}
    for pattern in test_patterns:
        matches = list(glob(str(project_root / pattern)))
        found_tests[pattern] = [
            str(Path(m).relative_to(project_root)) for m in matches
        ]
    
    total_test_files = sum(len(v) for v in found_tests.values())
    
    return {
        'test_patterns': test_patterns,
        'found_tests': found_tests,
        'total_test_files': total_test_files,
        'phase_5_tests_exist': total_test_files > 0
    }


def main():
    """メイン実行"""
    print("=" * 70)
    print("Phase 5.0: Current State Analysis")
    print("=" * 70)
    print()
    
    # 1. Emotion統合分析
    print("[1/5] Analyzing Emotion Integration...")
    emotion_report = analyze_emotion_integration()
    print(f"  ✅ Phase 4.9 Complete: {emotion_report['summary']['phase_4_9_complete']}/5 generators")
    print(f"  ✅ Phase 5 Ready: {emotion_report['summary']['phase_5_ready']}/5 generators")
    print(f"  ⚠️  Parameter Application: {emotion_report['summary']['has_application']}/5 generators (Expected: 0)")
    print()
    
    # 2. Emotion Loader確認
    print("[2/5] Checking Emotion Loader Utility...")
    emotion_loader_info = check_emotion_loader_utility()
    if emotion_loader_info.get('exists'):
        print(f"  ✅ emotion_loader.py found")
        print(f"  ✅ Functions: {emotion_loader_info['function_count']}")
        print(f"  ✅ Total lines: {emotion_loader_info['total_lines']}")
    else:
        print(f"  ❌ {emotion_loader_info.get('error')}")
    print()
    
    # 3. Emotion Mapping Config確認
    print("[3/5] Checking Emotion Mapping Config...")
    config_info = check_emotion_mapping_config()
    if config_info.get('exists'):
        print(f"  ✅ emotion_mapping.yaml found")
        print(f"  ✅ Emotion Profiles: {config_info['profile_count']}")
        print(f"  ✅ Sections: {config_info['section_count']}")
        print(f"  ✅ Instruments: {config_info['instrument_count']}")
    else:
        print(f"  ❌ {config_info.get('error')}")
    print()
    
    # 4. ベースライン品質確認
    print("[4/5] Establishing Baseline Quality...")
    baseline = establish_baseline_quality()
    print(f"  ✅ Instruments with eval output: {baseline['summary']['with_eval_output']}/5")
    print(f"  ✅ Schema 1.1: {baseline['summary']['schema_1_1']}/5")
    print(f"  ℹ️  {baseline['note']}")
    print()
    
    # 5. テストカバレッジ確認
    print("[5/5] Analyzing Test Coverage...")
    test_coverage = analyze_test_coverage()
    print(f"  ✅ Total test files found: {test_coverage['total_test_files']}")
    print()
    
    # 総合レポート
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'phase': '5.0',
        'emotion_integration': emotion_report,
        'emotion_loader': emotion_loader_info,
        'emotion_mapping_config': config_info,
        'baseline_quality': baseline,
        'test_coverage': test_coverage
    }
    
    # Phase 5準備状態判定
    phase_5_ready = (
        emotion_report['summary']['phase_4_9_complete'] == 5 and
        emotion_report['summary']['has_application'] == 0 and
        emotion_loader_info.get('exists', False) and
        config_info.get('exists', False)
    )
    
    report['phase_5_ready'] = phase_5_ready
    
    if phase_5_ready:
        print("✅ Phase 5 Ready!")
        print()
        print("Next Steps:")
        print("  1. Review baseline quality metrics")
        print("  2. Start Phase 5.1: Piano Parameter Application")
        print("  3. Implement velocity_std_multiplier & notes_per_bar_multiplier")
    else:
        print("⚠️  Phase 5 Not Ready")
        print()
        print("Issues:")
        if emotion_report['summary']['phase_4_9_complete'] < 5:
            print(f"  - Phase 4.9 incomplete: {emotion_report['summary']['phase_4_9_complete']}/5 generators")
        if emotion_report['summary']['has_application'] > 0:
            print(f"  - Parameter application already exists (should be 0): {emotion_report['summary']['has_application']}")
        if not emotion_loader_info.get('exists'):
            print("  - emotion_loader.py not found")
        if not config_info.get('exists'):
            print("  - emotion_mapping.yaml not found")
    
    print()
    
    # レポート保存
    output_dir = project_root / 'results'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'phase5_baseline.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Report saved to: {output_file.relative_to(project_root)}")
    print()
    
    # Generator詳細表示
    print("=" * 70)
    print("Generator Details")
    print("=" * 70)
    print()
    
    for instrument, info in emotion_report['generators'].items():
        print(f"[{instrument.upper()}]")
        print(f"  File: {info['file']}")
        print(f"  Import: {'✅' if info['has_emotion_loader_import'] else '❌'}")
        print(f"  Section param: {'✅' if info['has_section_param'] else '❌'}")
        print(f"  Emotion param: {'✅' if info['has_emotion_profile_param'] else '❌'}")
        print(f"  Storage: {'✅' if info['has_adjustments_storage'] else '❌'}")
        print(f"  Application: {'✅' if info['has_parameter_application'] else '⏳ (Phase 5)'}")
        print(f"  Lines: {info['line_counts']['code']} code, {info['line_counts']['comments']} comments")
        print()
    
    print("=" * 70)
    print(f"✅ Phase 5.0 Analysis Complete! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 70)
    
    return 0 if phase_5_ready else 1


if __name__ == '__main__':
    sys.exit(main())
