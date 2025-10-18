#!/usr/bin/env python3
"""
streamlit_benchmark_dashboard.py - ベンチマーク結果ダッシュボード

Streamlitでベンチマーク結果を可視化します。

Usage:
    streamlit run streamlit_benchmark_dashboard.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Install with: pip install plotly")

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    st.warning("⚠️ mido not installed. Install with: pip install mido")


# ページ設定
st.set_page_config(
    page_title="Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_benchmark_json(path: Path) -> Dict[str, Any]:
    """ベンチマークJSON読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_summary_json(path: Path) -> Optional[Dict[str, Any]]:
    """サマリーJSON読み込み"""
    if not path.exists():
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_midi_info(midi_path: Path) -> Dict[str, Any]:
    """MIDI基本情報抽出"""
    if not MIDO_AVAILABLE or not midi_path.exists():
        return {}
    
    try:
        midi = mido.MidiFile(midi_path)
        
        # トラック数
        track_count = len(midi.tracks)
        
        # ノート数
        note_count = 0
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_count += 1
        
        # 長さ (秒)
        duration_sec = midi.length
        
        return {
            'track_count': track_count,
            'note_count': note_count,
            'duration_sec': duration_sec,
            'ticks_per_beat': midi.ticks_per_beat,
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    st.title("📊 Benchmark Suite Dashboard")
    st.markdown("**ベンチマーク結果の可視化と分析**")
    st.divider()
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ Settings")
        
        project_root = Path.cwd()
        
        # ベンチマークJSON選択
        benchmark_json_path = project_root / 'multi_song_benchmark.json'
        
        if not benchmark_json_path.exists():
            st.error(f"❌ Benchmark JSON not found: {benchmark_json_path}")
            st.info("Run: `python scripts/generate_benchmark_json.py`")
            st.stop()
        
        # サマリーJSON選択
        summary_dir = project_root / 'benchmark_outputs'
        summary_json_path = summary_dir / 'benchmark_summary.json'
        
        if not summary_json_path.exists():
            st.warning("⚠️ No benchmark results found")
            st.info("Run: `python scripts/run_benchmark_suite.py`")
            has_results = False
        else:
            has_results = True
        
        st.divider()
        
        # フィルター
        st.subheader("🔍 Filters")
        
        benchmark_data = load_benchmark_json(benchmark_json_path)
        
        genres = list(benchmark_data.get('genres', {}).keys())
        selected_genres = st.multiselect(
            "Genre",
            options=genres,
            default=genres
        )
        
        difficulties = ['simple', 'medium', 'complex']
        selected_difficulties = st.multiselect(
            "Difficulty",
            options=difficulties,
            default=difficulties
        )
    
    # メインコンテンツ
    tabs = st.tabs(["📊 Overview", "🎵 Songs", "📈 Metrics", "🔍 Regression"])
    
    # Tab 1: Overview
    with tabs[0]:
        st.header("📊 Benchmark Overview")
        
        # 基本統計
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Songs",
                value=benchmark_data.get('total_songs', 0)
            )
        
        with col2:
            genres_dict = benchmark_data.get('genres', {})
            st.metric(
                label="Genres",
                value=len(genres_dict)
            )
        
        with col3:
            if has_results:
                summary = load_summary_json(summary_json_path)
                passed = summary.get('passed', 0)
                st.metric(
                    label="Passed",
                    value=f"{passed}/{summary.get('total_benchmarks', 0)}"
                )
            else:
                st.metric(label="Passed", value="N/A")
        
        with col4:
            if has_results:
                pass_rate = summary.get('pass_rate', 0)
                st.metric(
                    label="Pass Rate",
                    value=f"{pass_rate:.1f}%"
                )
            else:
                st.metric(label="Pass Rate", value="N/A")
        
        st.divider()
        
        # ジャンル分布
        if PLOTLY_AVAILABLE:
            st.subheader("🎸 Genre Distribution")
            
            genre_df = pd.DataFrame([
                {'Genre': genre, 'Songs': count}
                for genre, count in genres_dict.items()
            ])
            
            fig = px.bar(
                genre_df,
                x='Genre',
                y='Songs',
                title='Songs per Genre',
                color='Genre'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 実行結果 (あれば)
        if has_results:
            st.subheader("⏱️ Execution Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Total Duration",
                    value=f"{summary.get('total_duration_sec', 0):.1f}s"
                )
            
            with col2:
                avg_duration = summary.get('total_duration_sec', 0) / max(summary.get('total_benchmarks', 1), 1)
                st.metric(
                    label="Avg Duration/Song",
                    value=f"{avg_duration:.1f}s"
                )
    
    # Tab 2: Songs
    with tabs[1]:
        st.header("🎵 Benchmark Songs")
        
        # フィルター適用
        songs = benchmark_data.get('songs', [])
        
        filtered_songs = [
            song for song in songs
            if song['metadata']['genre'] in selected_genres
            and song['metadata']['difficulty'] in selected_difficulties
        ]
        
        st.info(f"Showing {len(filtered_songs)} / {len(songs)} songs")
        
        # 曲リスト表示
        for song in filtered_songs:
            with st.expander(f"🎵 {song['metadata']['title']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Metadata**")
                    st.write(f"- **Genre**: {song['metadata']['genre']}")
                    st.write(f"- **Style**: {song['metadata']['style']}")
                    st.write(f"- **Difficulty**: {song['metadata']['difficulty']}")
                    st.write(f"- **Seed**: {song['metadata']['seed']}")
                
                with col2:
                    st.markdown("**Expected Metrics**")
                    expected = song.get('expected_metrics', {})
                    st.write(f"- **Total Bars**: {expected.get('total_bars', 'N/A')}")
                    st.write(f"- **Sections**: {expected.get('sections', 'N/A')}")
                    st.write(f"- **Tempo**: {expected.get('tempo_bpm', 'N/A')} BPM")
                    st.write(f"- **Key**: {expected.get('key', 'N/A')}")
                
                # MIDIファイル情報 (あれば)
                if has_results:
                    midi_path = summary_dir / f"{song['id']}.mid"
                    
                    if midi_path.exists():
                        midi_info = extract_midi_info(midi_path)
                        
                        if midi_info:
                            st.markdown("**MIDI Info**")
                            st.write(f"- **Tracks**: {midi_info.get('track_count', 'N/A')}")
                            st.write(f"- **Notes**: {midi_info.get('note_count', 'N/A')}")
                            st.write(f"- **Duration**: {midi_info.get('duration_sec', 0):.1f}s")
                
                # 品質閾値
                st.markdown("**Quality Thresholds**")
                thresholds = song.get('quality_thresholds', {})
                
                for instrument, thresh in thresholds.items():
                    st.write(f"**{instrument.capitalize()}:**")
                    for key, value in thresh.items():
                        st.write(f"  - {key}: {value}")
    
    # Tab 3: Metrics
    with tabs[2]:
        st.header("📈 Metrics Analysis")
        
        if not has_results:
            st.warning("No benchmark results available. Run benchmark suite first.")
        else:
            summary = load_summary_json(summary_json_path)
            results = summary.get('results', [])
            
            # Status分布
            if PLOTLY_AVAILABLE:
                st.subheader("✅ Status Distribution")
                
                status_counts = {}
                for result in results:
                    status = result.get('status', 'UNKNOWN')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                status_df = pd.DataFrame([
                    {'Status': status, 'Count': count}
                    for status, count in status_counts.items()
                ])
                
                fig = px.pie(
                    status_df,
                    values='Count',
                    names='Status',
                    title='Benchmark Status Distribution',
                    color='Status',
                    color_discrete_map={
                        'PASS': 'green',
                        'FAIL': 'red',
                        'FAILED': 'red'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 実行時間分析
            st.subheader("⏱️ Execution Time")
            
            duration_data = [
                {
                    'benchmark': result.get('yaml', 'unknown'),
                    'duration': result.get('duration_sec', 0),
                    'status': result.get('status', 'UNKNOWN')
                }
                for result in results
                if 'duration_sec' in result
            ]
            
            if duration_data and PLOTLY_AVAILABLE:
                duration_df = pd.DataFrame(duration_data)
                
                fig = px.bar(
                    duration_df,
                    x='benchmark',
                    y='duration',
                    title='Execution Time per Benchmark',
                    color='status',
                    color_discrete_map={
                        'PASS': 'green',
                        'FAIL': 'red',
                        'FAILED': 'red'
                    }
                )
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # 詳細テーブル
            st.subheader("📋 Detailed Results")
            
            results_df = pd.DataFrame([
                {
                    'Benchmark': r.get('yaml', 'unknown'),
                    'Status': r.get('status', 'UNKNOWN'),
                    'Duration (s)': f"{r.get('duration_sec', 0):.2f}",
                    'MIDI': r.get('midi', 'N/A'),
                }
                for r in results
            ])
            
            st.dataframe(results_df, use_container_width=True)
    
    # Tab 4: Regression
    with tabs[3]:
        st.header("🔍 Regression Detection")
        
        st.markdown("""
        リグレッション検出機能を使用して、ベースラインとの品質比較を行います。
        
        **使用方法:**
        ```bash
        python scripts/detect_regression.py \\
          --baseline benchmark_outputs/baseline_summary.json \\
          --current benchmark_outputs/benchmark_summary.json \\
          --threshold 5.0
        ```
        """)
        
        # リグレッションレポート表示 (あれば)
        regression_report_path = project_root / 'regression_report.json'
        
        if regression_report_path.exists():
            st.subheader("📄 Latest Regression Report")
            
            with open(regression_report_path, 'r', encoding='utf-8') as f:
                regression_data = json.load(f)
            
            overall = regression_data.get('overall', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pass_rate_diff = overall.get('pass_rate_diff', 0)
                st.metric(
                    label="Pass Rate Change",
                    value=f"{overall.get('current_pass_rate', 0):.1f}%",
                    delta=f"{pass_rate_diff:+.1f}%"
                )
            
            with col2:
                st.metric(
                    label="Regressions",
                    value=regression_data.get('total_regressions', 0)
                )
            
            with col3:
                st.metric(
                    label="Improvements",
                    value=regression_data.get('total_improvements', 0)
                )
            
            # リグレッション詳細
            if regression_data.get('total_regressions', 0) > 0:
                st.error("⚠️ Regressions detected!")
                
                for reg in regression_data.get('regressions', []):
                    st.write(f"- **{reg['benchmark']}**: {reg['baseline_status']} → {reg['current_status']}")
            else:
                st.success("✅ No regressions detected")
        else:
            st.info("No regression report available. Run regression detection first.")
    
    # フッター
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        📊 Benchmark Dashboard v1.0 | 
        <a href='https://github.com'>GitHub</a> | 
        Made with Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
