#!/usr/bin/env python3
"""
Pattern Quality Learner - Phase 24 Meta Feedback

Safe-Kit使用頻度が高いパターンを学習し、v3推薦から除外。
週次バッチで実行し、低品質パターンをブラックリスト化。

Usage:
    from ml.pattern_quality_learner import PatternQualityLearner
    
    learner = PatternQualityLearner(
        shadow_log_path='data/shadow_traffic_log.csv',
        blacklist_threshold=0.05  # 5%以上がSafe-Kit fallbackなら除外
    )
    
    blacklist = learner.analyze_and_blacklist()
    # Returns: {'PATTERN_ID_123': 0.08, ...}
"""

import logging
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import csv
from collections import defaultdict

import numpy as np
import pandas as pd


class PatternQualityLearner:
    """
    パターン品質学習マネージャー
    
    Shadow Traffic LogからSafe-Kit使用頻度を分析し、
    低品質パターンをブラックリスト化してv3推薦から除外。
    
    Features:
    - Safe-Kit fallback率の高いパターンを検出
    - セクション別分析（Chorus/Verse/Bridge）
    - ブラックリスト自動生成（JSON出力）
    - ホワイトリスト対応（除外パターン指定）
    """
    
    def __init__(
        self,
        shadow_log_path: str,
        blacklist_threshold: float = 0.05,
        min_sample_count: int = 10,
        blacklist_output_path: Optional[str] = None,
        whitelist_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            shadow_log_path: Shadow Traffic Log CSV path
            blacklist_threshold: Safe-Kit fallback率閾値（5%以上で除外）
            min_sample_count: 最小サンプル数（10回未満は除外対象外）
            blacklist_output_path: ブラックリスト出力先JSON（デフォルト: data/pattern_blacklist.json）
            whitelist_path: ホワイトリストJSON（除外しないパターン）
            logger: ロガーインスタンス
        """
        self.shadow_log_path = Path(shadow_log_path)
        self.blacklist_threshold = blacklist_threshold
        self.min_sample_count = min_sample_count
        self.logger = logger or logging.getLogger(__name__)
        
        self.blacklist_output_path = Path(blacklist_output_path) if blacklist_output_path else \
            Path('data/pattern_blacklist.json')
        
        # ホワイトリスト読み込み
        self.whitelist = self._load_whitelist(whitelist_path)
        
        self.logger.info(
            f"PatternQualityLearner initialized: "
            f"threshold={blacklist_threshold}, min_samples={min_sample_count}"
        )
    
    def analyze_and_blacklist(
        self,
        days: int = 7,
        section: Optional[str] = None
    ) -> Dict[str, float]:
        """
        過去N日間のログを分析してブラックリスト生成
        
        Args:
            days: 分析対象日数（デフォルト7日）
            section: セクション指定（Noneで全体）
        
        Returns:
            ブラックリスト dict {pattern_id: fallback_rate}
        """
        # Shadow Logからデータ読み込み
        df = self._load_shadow_log(days)
        
        if df.empty:
            self.logger.warning("No shadow log data found")
            return {}
        
        # セクションフィルタ
        if section:
            df = df[df['section'].str.lower() == section.lower()]
        
        # Safe-Kit fallback分析
        blacklist = self._analyze_safe_kit_fallback(df)
        
        # ブラックリスト保存
        self._save_blacklist(blacklist, section)
        
        self.logger.info(
            f"Blacklist generated: {len(blacklist)} patterns "
            f"(threshold={self.blacklist_threshold}, days={days}, section={section})"
        )
        
        return blacklist
    
    def _load_shadow_log(self, days: int = 7) -> pd.DataFrame:
        """Shadow Traffic Logを読み込み"""
        if not self.shadow_log_path.exists():
            self.logger.warning(f"Shadow log not found: {self.shadow_log_path}")
            return pd.DataFrame()
        
        try:
            # CSV読み込み
            df = pd.read_csv(self.shadow_log_path)
            
            # timestamp列をdatetimeに変換
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 過去N日間のみ抽出
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
            
            self.logger.info(f"Loaded {len(df)} records from shadow log (last {days} days)")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to load shadow log: {e}")
            return pd.DataFrame()
    
    def _analyze_safe_kit_fallback(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Safe-Kit fallback率を分析
        
        ロジック:
        1. v3_pattern_idごとにグループ化
        2. Safe-Kit fallback回数（v3_pattern_id == 'SAFE_KIT_*'）をカウント
        3. fallback_rate = safe_kit_count / total_count
        4. fallback_rate > threshold かつ total_count >= min_samples なら除外
        
        Returns:
            {pattern_id: fallback_rate}
        """
        blacklist = {}
        
        # v3_pattern_idごとに集計
        pattern_stats = defaultdict(lambda: {'total': 0, 'safe_kit': 0})
        
        for _, row in df.iterrows():
            pattern_id = row.get('v3_pattern_id', '')
            
            # Safe-Kit使用チェック
            is_safe_kit = pattern_id.startswith('SAFE_KIT_')
            
            # 元のパターンIDを取得（Safe-Kitの場合は除く）
            if is_safe_kit:
                # Safe-KitのパターンIDからは学習しない
                continue
            
            # 統計更新
            pattern_stats[pattern_id]['total'] += 1
            
            # Safety triggered チェック
            safety_triggered = row.get('v3_safety_triggered', 0)
            if safety_triggered == 1:
                pattern_stats[pattern_id]['safe_kit'] += 1
        
        # ブラックリスト判定
        for pattern_id, stats in pattern_stats.items():
            total = stats['total']
            safe_kit = stats['safe_kit']
            
            # 最小サンプル数チェック
            if total < self.min_sample_count:
                continue
            
            # ホワイトリストチェック
            if pattern_id in self.whitelist:
                self.logger.info(f"Skipping whitelisted pattern: {pattern_id}")
                continue
            
            # Fallback率計算
            fallback_rate = safe_kit / total
            
            # 閾値チェック
            if fallback_rate >= self.blacklist_threshold:
                blacklist[pattern_id] = fallback_rate
                self.logger.info(
                    f"Blacklisted: {pattern_id} "
                    f"(fallback_rate={fallback_rate:.2%}, samples={total})"
                )
        
        return blacklist
    
    def _load_whitelist(self, whitelist_path: Optional[str]) -> Set[str]:
        """ホワイトリスト読み込み"""
        if not whitelist_path:
            return set()
        
        try:
            with open(whitelist_path, 'r', encoding='utf-8') as f:
                whitelist_data = json.load(f)
            
            whitelist = set(whitelist_data.get('patterns', []))
            self.logger.info(f"Loaded whitelist: {len(whitelist)} patterns")
            
            return whitelist
        
        except Exception as e:
            self.logger.warning(f"Failed to load whitelist: {e}")
            return set()
    
    def _save_blacklist(self, blacklist: Dict[str, float], section: Optional[str] = None):
        """ブラックリスト保存"""
        blacklist_data = {
            'timestamp': datetime.now().isoformat(),
            'threshold': self.blacklist_threshold,
            'min_sample_count': self.min_sample_count,
            'section': section or 'global',
            'blacklist': [
                {
                    'pattern_id': pattern_id,
                    'fallback_rate': fallback_rate
                }
                for pattern_id, fallback_rate in sorted(
                    blacklist.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
        }
        
        try:
            self.blacklist_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.blacklist_output_path, 'w', encoding='utf-8') as f:
                json.dump(blacklist_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Blacklist saved: {self.blacklist_output_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save blacklist: {e}")
    
    def get_blacklist_summary(self) -> Dict:
        """ブラックリストサマリー取得"""
        if not self.blacklist_output_path.exists():
            return {'blacklist': [], 'count': 0}
        
        try:
            with open(self.blacklist_output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                'timestamp': data.get('timestamp'),
                'threshold': data.get('threshold'),
                'section': data.get('section'),
                'blacklist': data.get('blacklist', []),
                'count': len(data.get('blacklist', []))
            }
        
        except Exception as e:
            self.logger.error(f"Failed to load blacklist summary: {e}")
            return {'blacklist': [], 'count': 0}
    
    def is_blacklisted(self, pattern_id: str) -> bool:
        """
        パターンがブラックリストに含まれるか判定
        
        Args:
            pattern_id: パターンID
        
        Returns:
            True if blacklisted
        """
        summary = self.get_blacklist_summary()
        blacklist_ids = {item['pattern_id'] for item in summary['blacklist']}
        
        return pattern_id in blacklist_ids
    
    def analyze_by_section(self, days: int = 7) -> Dict[str, Dict[str, float]]:
        """
        セクション別にブラックリスト分析
        
        Returns:
            {section: {pattern_id: fallback_rate}}
        """
        results = {}
        
        for section in ['Chorus', 'Verse', 'Bridge', 'Intro', 'Outro']:
            blacklist = self.analyze_and_blacklist(days=days, section=section)
            results[section] = blacklist
        
        return results


# Example usage
if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create learner
    learner = PatternQualityLearner(
        shadow_log_path='data/shadow_traffic_log.csv',
        blacklist_threshold=0.05,  # 5%以上でブラックリスト
        min_sample_count=10
    )
    
    # Analyze and generate blacklist (全体)
    print("\n=== Pattern Quality Analysis (Global) ===")
    blacklist = learner.analyze_and_blacklist(days=7)
    
    print(f"\nBlacklisted Patterns: {len(blacklist)}")
    for pattern_id, fallback_rate in sorted(blacklist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pattern_id}: {fallback_rate:.2%}")
    
    # Section別分析
    print("\n=== Pattern Quality Analysis (By Section) ===")
    section_results = learner.analyze_by_section(days=7)
    
    for section, section_blacklist in section_results.items():
        if section_blacklist:
            print(f"\n{section}: {len(section_blacklist)} patterns blacklisted")
            for pattern_id, fallback_rate in list(section_blacklist.items())[:3]:
                print(f"  {pattern_id}: {fallback_rate:.2%}")
    
    print(f"\n✅ Blacklist saved to {learner.blacklist_output_path}")
