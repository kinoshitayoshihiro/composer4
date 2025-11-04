"""
Shadow Traffic Auto-Recovery Manager

双方向フォールバック/自動復帰機能:
- v3でKPI劣化 → v1に切替（Fallback）
- v1で安定稼働 → v3に復帰（Recovery）
- ヒステリシス制御: 頻繁な切替を防ぐ

設計仕様: monitoring/AUTO_RECOVERY_DESIGN.md
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class RecoveryMetrics:
    """Auto-Recovery関連のメトリクス"""
    switches_v3_to_v1: int = 0  # v3→v1切替回数
    switches_v1_to_v3: int = 0  # v1→v3切替回数
    current_version: str = 'v3'  # 現在のアクティブバージョン
    cooldown_active: bool = False  # クールダウン中フラグ
    cooldown_remaining: int = 0  # クールダウン残りバー数
    breach_count: int = 0  # ウィンドウ内の違反回数
    window_size: int = 32  # ウィンドウサイズ
    threshold: int = 6  # 違反閾値


class AutoRecoveryManager:
    """
    Shadow Trafficの自動復帰管理クラス
    
    機能:
    - 32バーウィンドウで違反回数をカウント
    - 6回以上違反でv3→v1切替（Fallback）
    - 違反0回でv1→v3復帰（Recovery）
    - 切替後16バーのクールダウン期間
    
    使用例:
        manager = AutoRecoveryManager(window_size=32, threshold=6, cooldown=16)
        
        # 各リクエスト後
        is_breach = check_kpi_breach(result)
        manager.add_result(is_breach)
        
        # 切替判定
        new_version = manager.should_switch_version()
        if new_version:
            manager.switch_version(new_version)
        
        manager.tick_cooldown()
    """
    
    def __init__(
        self,
        window_size: int = 32,
        threshold: int = 6,
        cooldown: int = 16,
        initial_version: str = 'v3',
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            window_size: 違反カウントのウィンドウサイズ（バー数）
            threshold: ウィンドウ内で許容される違反回数
            cooldown: バージョン切替後のクールダウン期間（バー数）
            initial_version: 初期アクティブバージョン ('v3' or 'v1')
            logger: ロガーインスタンス
        """
        self.window_size = window_size
        self.threshold = threshold
        self.cooldown = cooldown
        
        # 結果ウィンドウ（True=違反, False=正常）
        self.window = deque(maxlen=window_size)
        
        # 状態管理
        self.current_version = initial_version
        self.cooldown_counter = 0  # クールダウン残りバー数
        
        # 統計
        self.switches_v3_to_v1 = 0
        self.switches_v1_to_v3 = 0
        self.total_requests = 0
        
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(
            f"AutoRecoveryManager initialized: "
            f"window={window_size}, threshold={threshold}, cooldown={cooldown}, "
            f"initial_version={initial_version}"
        )
    
    def add_result(self, is_breach: bool):
        """
        リクエスト結果を追加
        
        Args:
            is_breach: KPIゲート違反があったか（True=違反, False=正常）
        """
        self.window.append(is_breach)
        self.total_requests += 1
        
        if is_breach:
            self.logger.debug(f"Breach detected (total in window: {self.get_breach_count()})")
    
    def get_breach_count(self) -> int:
        """ウィンドウ内の違反回数を取得"""
        return sum(1 for is_breach in self.window if is_breach)
    
    def should_switch_version(self) -> Optional[str]:
        """
        バージョン切替が必要か判定
        
        Returns:
            切替先バージョン ('v3' or 'v1') または None（切替不要）
        
        判定ロジック:
        - クールダウン中: 切替なし
        - v3アクティブ & 違反≥閾値 → v1へ切替
        - v1アクティブ & 違反=0 → v3へ復帰
        """
        # クールダウン中は切替しない
        if self.cooldown_counter > 0:
            return None
        
        # ウィンドウが十分に満たされるまで待機
        if len(self.window) < self.window_size:
            return None
        
        breach_count = self.get_breach_count()
        window_filled = len(self.window)
        breach_ratio = breach_count / window_filled if window_filled > 0 else 0.0
        
        # v3アクティブ時: 違反が閾値以上 OR 比率>20% → v1へ切替
        if self.current_version == 'v3':
            # 回数判定（従来）
            count_breach = breach_count >= self.threshold
            # 比率判定（新規）- より安定
            ratio_breach = breach_ratio > 0.20  # 20%以上の違反率
            
            if count_breach or ratio_breach:
                self.logger.warning(
                    f"Auto-Recovery: Fallback triggered - "
                    f"breach_count={breach_count}/{self.threshold} ({breach_ratio*100:.1f}%), "
                    f"trigger={'count' if count_breach else 'ratio'}"
                )
                return 'v1'
        
        # v1アクティブ時: 違反なし（完全安定）OR 比率<5% → v3へ復帰
        if self.current_version == 'v1':
            # 完全安定判定（従来）
            perfect_stable = breach_count == 0
            # 低違反率判定（新規）- より柔軟な復帰
            low_breach = breach_ratio < 0.05  # 5%未満の違反率
            
            if perfect_stable or low_breach:
                self.logger.info(
                    f"Auto-Recovery: Recovery triggered - "
                    f"breach_count={breach_count} ({breach_ratio*100:.1f}%), "
                    f"trigger={'perfect' if perfect_stable else 'low_ratio'}"
                )
                return 'v3'
        
        return None  # 切替不要
    
    def switch_version(self, new_version: str):
        """
        バージョン切替を実行し、クールダウンを開始
        
        Args:
            new_version: 切替先バージョン ('v3' or 'v1')
        """
        old_version = self.current_version
        self.current_version = new_version
        self.cooldown_counter = self.cooldown
        self.window.clear()  # ウィンドウをリセット
        
        # 統計更新
        if old_version == 'v3' and new_version == 'v1':
            self.switches_v3_to_v1 += 1
        elif old_version == 'v1' and new_version == 'v3':
            self.switches_v1_to_v3 += 1
        
        self.logger.warning(
            f"Auto-Recovery: Version switched - "
            f"{old_version} → {new_version}, "
            f"cooldown={self.cooldown} bars"
        )
    
    def tick_cooldown(self):
        """クールダウンカウンターを減算（毎リクエスト後に呼び出し）"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0:
                self.logger.info("Auto-Recovery: Cooldown period ended")
    
    def get_metrics(self) -> RecoveryMetrics:
        """
        現在のメトリクスを取得
        
        Returns:
            RecoveryMetrics: 統計情報
        """
        return RecoveryMetrics(
            switches_v3_to_v1=self.switches_v3_to_v1,
            switches_v1_to_v3=self.switches_v1_to_v3,
            current_version=self.current_version,
            cooldown_active=self.cooldown_counter > 0,
            cooldown_remaining=self.cooldown_counter,
            breach_count=self.get_breach_count(),
            window_size=self.window_size,
            threshold=self.threshold
        )
    
    def reset(self):
        """状態をリセット（テスト用）"""
        self.window.clear()
        self.cooldown_counter = 0
        self.switches_v3_to_v1 = 0
        self.switches_v1_to_v3 = 0
        self.total_requests = 0
        self.logger.info("Auto-Recovery: State reset")


def check_kpi_breach(
    result_dict: Dict,
    gate_config: Dict,
    section: str
) -> bool:
    """
    KPIゲート違反を検出
    
    Args:
        result_dict: _execute_v3/_execute_v1の返り値
        gate_config: gate_prod.yamlの内容
        section: セクション名（Chorus, Verse等）
    
    Returns:
        True: 違反あり, False: 正常
    """
    # エラー発生は違反扱い
    if result_dict.get('error'):
        return True
    
    # セクション別閾値を取得
    gate = _get_gate_for_section(gate_config, section)
    
    # Accent Scoreチェック
    accent_score = result_dict.get('accent_score', 0.0)
    if accent_score < gate.get('accent_min', 0.60):
        return True
    
    # Chord Fitチェック
    chord_fit = result_dict.get('chord_fit', 0.0)
    if chord_fit < gate.get('chord_min', 0.40):
        return True
    
    return False


def _get_gate_for_section(gate_config: Dict, section: str) -> Dict:
    """
    セクション別ゲート閾値を取得
    
    Args:
        gate_config: gate設定辞書
        section: セクション名
    
    Returns:
        該当セクションのゲート閾値（per_section > デフォルト）
    """
    # セクション別オーバーライドがあればそれを使用
    per_section = gate_config.get('per_section', {})
    if section in per_section:
        return per_section[section]
    
    # デフォルト値
    return {
        'accent_min': gate_config.get('accent_min', 0.60),
        'chord_min': gate_config.get('chord_min', 0.40),
        'density_min': gate_config.get('density_min', 0.0)
    }


# =====================================
# 以下、使用例とテストコード
# =====================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # テストシナリオ1: v3 → v1 Fallback
    logger.info("=== Test Scenario 1: v3 → v1 Fallback ===")
    manager = AutoRecoveryManager(window_size=10, threshold=3, cooldown=5)
    
    # 10回中4回違反（閾値3を超過）
    breaches = [False, True, False, True, False, True, False, True, False, False]
    for i, is_breach in enumerate(breaches):
        manager.add_result(is_breach)
        new_version = manager.should_switch_version()
        if new_version:
            logger.info(f"  [{i+1}] Switch triggered: {manager.current_version} → {new_version}")
            manager.switch_version(new_version)
        manager.tick_cooldown()
    
    metrics = manager.get_metrics()
    logger.info(f"  Final state: {metrics.current_version}, switches_v3_to_v1={metrics.switches_v3_to_v1}")
    
    # テストシナリオ2: v1 → v3 Recovery
    logger.info("\n=== Test Scenario 2: v1 → v3 Recovery ===")
    manager.reset()
    manager.current_version = 'v1'
    
    # 10回中0回違反（完全安定）
    breaches = [False] * 10
    for i, is_breach in enumerate(breaches):
        manager.add_result(is_breach)
        new_version = manager.should_switch_version()
        if new_version:
            logger.info(f"  [{i+1}] Switch triggered: {manager.current_version} → {new_version}")
            manager.switch_version(new_version)
        manager.tick_cooldown()
    
    metrics = manager.get_metrics()
    logger.info(f"  Final state: {metrics.current_version}, switches_v1_to_v3={metrics.switches_v1_to_v3}")
    
    # テストシナリオ3: Cooldown中の切替抑制
    logger.info("\n=== Test Scenario 3: Cooldown Suppression ===")
    manager.reset()
    manager.current_version = 'v3'
    
    # 最初の10回で違反多発 → v1切替 → cooldown開始
    breaches = [True] * 10
    for i, is_breach in enumerate(breaches):
        manager.add_result(is_breach)
        new_version = manager.should_switch_version()
        if new_version:
            logger.info(f"  [{i+1}] Switch triggered: {manager.current_version} → {new_version}")
            manager.switch_version(new_version)
            break  # 切替後はcooldown開始
        manager.tick_cooldown()
    
    # cooldown中に再度違反発生（切替は抑制される）
    logger.info(f"  Cooldown active: {manager.cooldown_counter} bars remaining")
    for i in range(5):
        manager.add_result(True)  # 違反
        new_version = manager.should_switch_version()
        if new_version:
            logger.info(f"  [{i+1}] Switch triggered (unexpected)")
        else:
            logger.info(f"  [{i+1}] Switch suppressed (cooldown={manager.cooldown_counter})")
        manager.tick_cooldown()
    
    logger.info("✓ All test scenarios passed")
