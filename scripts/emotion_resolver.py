"""
emotion_resolver.py

EmotionAI の数値出力（valence, energy）を離散カテゴリに量子化し、
感情タグ（emotion_tag）とロール（bass_role, rhythm_density等）を自動付与するモジュール。

機能:
1. 数値→カテゴリ変換（valence → NEG/NEU/POS, energy → LOW/MID/HIGH）
2. カテゴリ組み合わせから emotion_tag を自動付与（BRIGHT_ENERGETIC 等）
3. emotion_tag から各楽器の default_roles を取得
4. セクション正規化（section_name → 標準セクションタイプ）

使用方法:
    resolver = EmotionResolver()
    
    # 数値からカテゴリへ
    valence_cat = resolver.quantize_valence(0.5)  # "POS"
    arousal_cat = resolver.quantize_arousal(0.8)  # "HIGH"
    
    # カテゴリから感情タグへ
    emotion_tag = resolver.resolve_emotion_tag("POS", "HIGH")  # "BRIGHT_ENERGETIC"
    
    # 感情タグからロール取得
    roles = resolver.get_default_roles(emotion_tag)
    # {"bass_role": "DRIVE_FORWARD", "rhythm_density": "DENSE", ...}
    
    # セクション正規化
    section_type = resolver.normalize_section("verse1")  # "VERSE"
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


class EmotionResolver:
    """感情数値を離散カテゴリ・タグに変換し、楽器ロールを解決するクラス"""
    
    def __init__(
        self,
        emotion_axes_path: str = "config/emotion_axes.yaml",
        emotion_tags_path: str = "config/emotion_tags.yaml",
        section_types_path: str = "config/section_types.yaml",
        phrase_roles_path: str = "config/phrase_roles.yaml",
    ):
        """
        Args:
            emotion_axes_path: emotion_axes.yaml のパス
            emotion_tags_path: emotion_tags.yaml のパス
            section_types_path: section_types.yaml のパス
            phrase_roles_path: phrase_roles.yaml のパス
        """
        self.axes = self._load_yaml(emotion_axes_path)
        self.tags = self._load_yaml(emotion_tags_path)
        self.sections = self._load_yaml(section_types_path)
        self.roles = self._load_yaml(phrase_roles_path)
        
        # 高速検索用のマッピングを構築
        self._build_tag_lookup()
        self._build_section_lookup()
    
    def _load_yaml(self, path: str) -> Dict:
        """YAML ファイルを読み込む"""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _build_tag_lookup(self):
        """valence/arousal の組み合わせから emotion_tag への高速マッピング"""
        self.tag_lookup = {}
        for tag_def in self.tags.get("tags", []):
            key = (tag_def["valence"], tag_def["arousal"])
            self.tag_lookup[key] = tag_def["id"]
    
    def _build_section_lookup(self):
        """section_name → 標準セクションタイプへのマッピング"""
        self.section_lookup = {}
        mapping = self.sections.get("classification_rules", {}).get("mapping", {})
        for name, section_type in mapping.items():
            self.section_lookup[name.lower()] = section_type
    
    # ==========================================
    # 1. 数値 → カテゴリ変換
    # ==========================================
    
    def quantize_valence(self, value: float) -> str:
        """
        valence 数値をカテゴリ（NEG/NEU/POS）に変換
        
        Args:
            value: valence 値 (-1.0 ~ 1.0)
            
        Returns:
            カテゴリ文字列 ("NEG" / "NEU" / "POS")
        """
        return self._quantize_axis("valence", value)
    
    def quantize_arousal(self, value: float) -> str:
        """
        arousal 数値をカテゴリ（LOW/MID/HIGH）に変換
        
        Args:
            value: arousal 値 (0.0 ~ 1.0)
            
        Returns:
            カテゴリ文字列 ("LOW" / "MID" / "HIGH")
        """
        return self._quantize_axis("arousal", value)
    
    def _quantize_axis(self, axis_name: str, value: float) -> str:
        """
        任意の軸の数値をカテゴリに変換
        
        Args:
            axis_name: 軸名 ("valence" or "arousal")
            value: 数値
            
        Returns:
            カテゴリ文字列
        """
        axis_def = self.axes["axes"][axis_name]
        buckets = axis_def["buckets"]
        
        # 各バケツをチェック
        for bucket_name, bucket_range in buckets.items():
            if bucket_range["min"] <= value < bucket_range["max"]:
                return bucket_name
        
        # 境界値の場合（value == max）
        for bucket_name, bucket_range in buckets.items():
            if value == bucket_range["max"]:
                return bucket_name
        
        # デフォルトにフォールバック
        return self.axes["defaults"][axis_name]
    
    # ==========================================
    # 2. カテゴリ → 感情タグ
    # ==========================================
    
    def resolve_emotion_tag(
        self, 
        valence_category: str, 
        arousal_category: str
    ) -> str:
        """
        valence/arousal カテゴリから感情タグを解決
        
        Args:
            valence_category: "NEG" / "NEU" / "POS"
            arousal_category: "LOW" / "MID" / "HIGH"
            
        Returns:
            emotion_tag ID (例: "BRIGHT_ENERGETIC")
        """
        key = (valence_category, arousal_category)
        return self.tag_lookup.get(key, "NEUTRAL_COOL")  # デフォルト
    
    def resolve_emotion_tag_from_values(
        self, 
        valence: float, 
        arousal: float
    ) -> str:
        """
        valence/arousal 数値から直接 emotion_tag を解決
        
        Args:
            valence: valence 値 (-1.0 ~ 1.0)
            arousal: arousal 値 (0.0 ~ 1.0)
            
        Returns:
            emotion_tag ID (例: "MELANCHOLIC")
        """
        val_cat = self.quantize_valence(valence)
        aro_cat = self.quantize_arousal(arousal)
        return self.resolve_emotion_tag(val_cat, aro_cat)
    
    # ==========================================
    # 3. 感情タグ → ロール取得
    # ==========================================
    
    def get_default_roles(self, emotion_tag: str) -> Dict[str, str]:
        """
        emotion_tag から各楽器の default_roles を取得
        
        Args:
            emotion_tag: emotion_tag ID (例: "BRIGHT_ENERGETIC")
            
        Returns:
            ロール辞書 {"bass_role": "DRIVE_FORWARD", "rhythm_density": "DENSE", ...}
        """
        for tag_def in self.tags.get("tags", []):
            if tag_def["id"] == emotion_tag:
                return tag_def.get("default_roles", {})
        
        # デフォルト
        return {
            "bass_role": "ROOT_FOUNDATION",
            "rhythm_density": "MEDIUM",
            "harmony_tension": "MEDIUM",
            "phrase_complexity": "MEDIUM",
        }
    
    def get_bass_role_info(self, bass_role: str) -> Dict:
        """
        bass_role の詳細情報を取得
        
        Args:
            bass_role: bass_role ID (例: "DRIVE_FORWARD")
            
        Returns:
            bass_role の定義辞書
        """
        for role_def in self.roles.get("bass_roles", []):
            if role_def["id"] == bass_role:
                return role_def
        
        # デフォルト
        return {
            "id": "ROOT_FOUNDATION",
            "label_ja": "ルート基盤型",
            "rhythm_density": "SPARSE",
            "note_choice_preference": ["root", "fifth"],
            "fill_frequency": "low",
        }
    
    # ==========================================
    # 4. セクション正規化
    # ==========================================
    
    def normalize_section(self, section_name: str) -> str:
        """
        section_name を標準セクションタイプに正規化
        
        Args:
            section_name: 生のセクション名（例: "verse1", "prechorus"）
            
        Returns:
            標準セクションタイプ（例: "VERSE", "PRE_CHORUS"）
        """
        normalized = section_name.lower().strip()
        return self.section_lookup.get(normalized, "UNKNOWN")
    
    def get_section_info(self, section_type: str) -> Dict:
        """
        標準セクションタイプの詳細情報を取得
        
        Args:
            section_type: 標準セクションタイプ（例: "VERSE"）
            
        Returns:
            セクション定義辞書
        """
        for section_def in self.sections.get("standard_sections", []):
            if section_def["id"] == section_type:
                return section_def
        
        # デフォルト
        return {
            "id": "UNKNOWN",
            "label_ja": "不明",
            "typical_emotion_tags": ["NEUTRAL_COOL"],
            "typical_bass_role": "ROOT_FOUNDATION",
            "energy_level": "mid",
        }
    
    # ==========================================
    # 5. 統合パイプライン
    # ==========================================
    
    def resolve_all(
        self,
        valence: float,
        arousal: float,
        section_name: str,
    ) -> Dict:
        """
        数値・セクション名から全ての情報を一括解決
        
        Args:
            valence: valence 値 (-1.0 ~ 1.0)
            arousal: arousal 値 (0.0 ~ 1.0)
            section_name: セクション名（例: "verse1"）
            
        Returns:
            統合情報辞書:
            {
                "valence_category": "POS",
                "arousal_category": "HIGH",
                "emotion_tag": "BRIGHT_ENERGETIC",
                "section_type": "VERSE",
                "bass_role": "DRIVE_FORWARD",
                "rhythm_density": "DENSE",
                "harmony_tension": "HIGH",
                "phrase_complexity": "MEDIUM",
            }
        """
        val_cat = self.quantize_valence(valence)
        aro_cat = self.quantize_arousal(arousal)
        emotion_tag = self.resolve_emotion_tag(val_cat, aro_cat)
        section_type = self.normalize_section(section_name)
        default_roles = self.get_default_roles(emotion_tag)
        
        return {
            "valence_category": val_cat,
            "arousal_category": aro_cat,
            "emotion_tag": emotion_tag,
            "section_type": section_type,
            **default_roles,
        }


# ==========================================
# サンプル使用例（テスト用）
# ==========================================
if __name__ == "__main__":
    resolver = EmotionResolver()
    
    print("=" * 60)
    print("EmotionResolver サンプル実行")
    print("=" * 60)
    
    # テストケース1: 明るくエネルギッシュ
    print("\n【テストケース1】valence=0.7, arousal=0.85, section=chorus")
    result1 = resolver.resolve_all(valence=0.7, arousal=0.85, section_name="chorus")
    for key, value in result1.items():
        print(f"  {key}: {value}")
    
    # テストケース2: 哀愁・センチメンタル
    print("\n【テストケース2】valence=-0.6, arousal=0.2, section=verse1")
    result2 = resolver.resolve_all(valence=-0.6, arousal=0.2, section_name="verse1")
    for key, value in result2.items():
        print(f"  {key}: {value}")
    
    # テストケース3: クール・中立
    print("\n【テストケース3】valence=0.0, arousal=0.5, section=prechorus")
    result3 = resolver.resolve_all(valence=0.0, arousal=0.5, section_name="prechorus")
    for key, value in result3.items():
        print(f"  {key}: {value}")
    
    # bass_role の詳細情報取得
    print("\n【bass_role 詳細情報】")
    bass_role = result1["bass_role"]
    bass_info = resolver.get_bass_role_info(bass_role)
    print(f"  ID: {bass_info['id']}")
    print(f"  ラベル: {bass_info['label_ja']}")
    print(f"  リズム密度: {bass_info['rhythm_density']}")
    print(f"  音選択優先度: {bass_info['note_choice_preference']}")
    print(f"  フィル頻度: {bass_info['fill_frequency']}")
    
    print("\n" + "=" * 60)
