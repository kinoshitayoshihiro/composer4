"""
test_benchmark_advanced.py - ベンチマーク高度機能テスト

リグレッション検出、CI統合、ダッシュボードのテスト

Usage:
    pytest tests/test_benchmark_advanced.py -v
"""

import json
import subprocess
from pathlib import Path

import pytest


# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
BENCHMARK_OUTPUTS = PROJECT_ROOT / 'benchmark_outputs'


class TestRegressionDetection:
    """リグレッション検出機能のテスト"""
    
    @pytest.fixture
    def sample_baseline(self, tmp_path):
        """サンプルベースラインJSON"""
        baseline = {
            'generated': '2025-10-18T00:00:00',
            'total_benchmarks': 3,
            'passed': 3,
            'failed': 0,
            'pass_rate': 100.0,
            'total_duration_sec': 15.0,
            'results': [
                {'yaml': 'song1.yaml', 'status': 'PASS', 'duration_sec': 5.0},
                {'yaml': 'song2.yaml', 'status': 'PASS', 'duration_sec': 5.0},
                {'yaml': 'song3.yaml', 'status': 'PASS', 'duration_sec': 5.0},
            ]
        }
        
        baseline_path = tmp_path / 'baseline.json'
        with open(baseline_path, 'w') as f:
            json.dump(baseline, f)
        
        return baseline_path
    
    @pytest.fixture
    def sample_current_no_regression(self, tmp_path):
        """リグレッションなしのサンプル"""
        current = {
            'generated': '2025-10-18T01:00:00',
            'total_benchmarks': 3,
            'passed': 3,
            'failed': 0,
            'pass_rate': 100.0,
            'total_duration_sec': 16.0,
            'results': [
                {'yaml': 'song1.yaml', 'status': 'PASS', 'duration_sec': 5.2},
                {'yaml': 'song2.yaml', 'status': 'PASS', 'duration_sec': 5.3},
                {'yaml': 'song3.yaml', 'status': 'PASS', 'duration_sec': 5.5},
            ]
        }
        
        current_path = tmp_path / 'current.json'
        with open(current_path, 'w') as f:
            json.dump(current, f)
        
        return current_path
    
    @pytest.fixture
    def sample_current_with_regression(self, tmp_path):
        """リグレッションありのサンプル"""
        current = {
            'generated': '2025-10-18T01:00:00',
            'total_benchmarks': 3,
            'passed': 2,
            'failed': 1,
            'pass_rate': 66.7,
            'total_duration_sec': 12.0,
            'results': [
                {'yaml': 'song1.yaml', 'status': 'PASS', 'duration_sec': 5.0},
                {'yaml': 'song2.yaml', 'status': 'FAILED', 'duration_sec': 2.0, 'error': 'MIDI generation failed'},
                {'yaml': 'song3.yaml', 'status': 'PASS', 'duration_sec': 5.0},
            ]
        }
        
        current_path = tmp_path / 'current_regression.json'
        with open(current_path, 'w') as f:
            json.dump(current, f)
        
        return current_path
    
    def test_detect_regression_script_exists(self):
        """リグレッション検出スクリプトが存在する"""
        script_path = SCRIPTS_DIR / 'detect_regression.py'
        assert script_path.exists(), f"Script not found: {script_path}"
    
    def test_regression_detection_no_regression(
        self,
        sample_baseline,
        sample_current_no_regression,
        tmp_path
    ):
        """リグレッションなしの検出"""
        output_path = tmp_path / 'report.txt'
        
        result = subprocess.run(
            [
                'python',
                str(SCRIPTS_DIR / 'detect_regression.py'),
                '--baseline', str(sample_baseline),
                '--current', str(sample_current_no_regression),
                '--output', str(output_path),
                '--threshold', '5.0'
            ],
            capture_output=True,
            text=True
        )
        
        # 終了コード0 (リグレッションなし)
        assert result.returncode == 0, f"Unexpected exit code: {result.returncode}"
        
        # レポートファイル生成
        assert output_path.exists(), "Report not generated"
        
        # JSON生成確認
        json_path = output_path.with_suffix('.json')
        assert json_path.exists(), "JSON report not generated"
        
        with open(json_path, 'r') as f:
            report_data = json.load(f)
        
        assert report_data['has_regression'] is False
        assert report_data['total_regressions'] == 0
    
    def test_regression_detection_with_regression(
        self,
        sample_baseline,
        sample_current_with_regression,
        tmp_path
    ):
        """リグレッションありの検出"""
        output_path = tmp_path / 'report_regression.txt'
        
        result = subprocess.run(
            [
                'python',
                str(SCRIPTS_DIR / 'detect_regression.py'),
                '--baseline', str(sample_baseline),
                '--current', str(sample_current_with_regression),
                '--output', str(output_path),
                '--threshold', '5.0',
                '--fail-on-regression'
            ],
            capture_output=True,
            text=True
        )
        
        # 終了コード1 (リグレッション検出)
        assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"
        
        # レポート生成確認
        json_path = output_path.with_suffix('.json')
        assert json_path.exists(), "JSON report not generated"
        
        with open(json_path, 'r') as f:
            report_data = json.load(f)
        
        assert report_data['has_regression'] is True
        assert report_data['total_regressions'] > 0


class TestGitHubActionsWorkflow:
    """GitHub Actionsワークフローのテスト"""
    
    def test_workflow_file_exists(self):
        """ワークフローファイルが存在する"""
        workflow_path = PROJECT_ROOT / '.github' / 'workflows' / 'benchmark.yml'
        assert workflow_path.exists(), f"Workflow not found: {workflow_path}"
    
    def test_workflow_syntax_valid(self):
        """ワークフローYAML構文が正しい"""
        workflow_path = PROJECT_ROOT / '.github' / 'workflows' / 'benchmark.yml'
        
        import yaml
        
        with open(workflow_path, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        # 基本構造確認
        assert 'name' in workflow_data
        assert 'on' in workflow_data
        assert 'jobs' in workflow_data
        
        # ジョブ確認
        assert 'benchmark' in workflow_data['jobs']
        
        job = workflow_data['jobs']['benchmark']
        assert 'runs-on' in job
        assert 'steps' in job
    
    def test_workflow_has_required_steps(self):
        """必須ステップが含まれている"""
        workflow_path = PROJECT_ROOT / '.github' / 'workflows' / 'benchmark.yml'
        
        import yaml
        
        with open(workflow_path, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        steps = workflow_data['jobs']['benchmark']['steps']
        step_names = [step.get('name', '') for step in steps]
        
        # 必須ステップ確認
        assert any('checkout' in name.lower() for name in step_names)
        assert any('python' in name.lower() for name in step_names)
        assert any('benchmark' in name.lower() for name in step_names)


class TestBenchmarkDashboard:
    """ダッシュボードのテスト"""
    
    def test_dashboard_script_exists(self):
        """ダッシュボードスクリプトが存在する"""
        dashboard_path = PROJECT_ROOT / 'streamlit_benchmark_dashboard.py'
        assert dashboard_path.exists(), f"Dashboard not found: {dashboard_path}"
    
    def test_dashboard_imports(self):
        """ダッシュボードが正しくインポートできる"""
        # 基本的なインポートチェック
        import importlib.util
        
        dashboard_path = PROJECT_ROOT / 'streamlit_benchmark_dashboard.py'
        
        spec = importlib.util.spec_from_file_location("dashboard", dashboard_path)
        assert spec is not None, "Cannot load dashboard module"


class TestBenchmarkScripts:
    """ベンチマークスクリプトの統合テスト"""
    
    def test_generate_benchmark_json_exists(self):
        """ベンチマークJSON生成スクリプトが存在する"""
        script_path = SCRIPTS_DIR / 'generate_benchmark_json.py'
        assert script_path.exists()
    
    def test_compare_benchmark_metrics_exists(self):
        """メトリクス比較スクリプトが存在する"""
        script_path = SCRIPTS_DIR / 'compare_benchmark_metrics.py'
        assert script_path.exists()
    
    def test_run_benchmark_suite_exists(self):
        """ベンチマーク実行スクリプトが存在する"""
        script_path = SCRIPTS_DIR / 'run_benchmark_suite.py'
        assert script_path.exists()
    
    def test_all_scripts_have_help(self):
        """全スクリプトが--helpオプションをサポート"""
        scripts = [
            'generate_benchmark_json.py',
            'compare_benchmark_metrics.py',
            'run_benchmark_suite.py',
            'detect_regression.py',
        ]
        
        for script_name in scripts:
            script_path = SCRIPTS_DIR / script_name
            
            result = subprocess.run(
                ['python', str(script_path), '--help'],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, f"{script_name} --help failed"
            assert 'usage' in result.stdout.lower() or 'Usage' in result.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
