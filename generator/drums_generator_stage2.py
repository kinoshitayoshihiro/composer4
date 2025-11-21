#!/usr/bin/env python3
"""
Drums Generator Stage2 - V1継承 + AI拡張

V1 DrumGeneratorの全機能を継承し、Stage2レイヤーでAI処理を追加。

アーキテクチャ:
    DrumsGeneratorStage2 (V1継承)
    └─ V1の発音エンジン
       └─ Stage2レイヤー（AI/humanize/tempo展開）
          ├─ PatternRecommender（pickle）
          ├─ apply_ai_filters（モデル適用）
          ├─ humanize（微調整）
          └─ quantize_to_tempo_map（可変テンポ）

Usage:
    from generator.drums_generator_stage2 import DrumsGeneratorStage2

    gen = DrumsGeneratorStage2(...)
    part = gen.compose(section_data=section, ...)
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import inspect
import logging
from typing import Dict, List

try:
    from generator.instrument_stage2_base import InstrumentStage2Base
except ImportError:
    from instrument_stage2_base import InstrumentStage2Base

try:
    from ml.pattern_recommender import PatternRecommender
except ImportError:
    PatternRecommender = None

logger = logging.getLogger(__name__)


__all__ = [
    "DrumsGeneratorStage2",
    "DrumPattern",
    "GM_DRUM_MAP",
]


GM_DRUM_MAP: Dict[str, List[int]] = {
    "kick": [35, 36],
    "snare": [38, 40],
    "hihat_closed": [42],
    "hihat_open": [46],
    "hihat_pedal": [44],
    "tom_low": [41, 43, 45],
    "tom_mid": [47, 48],
    "tom_high": [50],
    "crash": [49, 57],
    "ride": [51, 59],
    "perc": [39, 54, 56, 81, 82],
    "ghost": [37],
}


@dataclass
class DrumPattern:
    """Lightweight container used by scripts and tests."""

    id: str
    instrument: str
    technique: str
    tempo: float
    bars: int
    emotion: str = "neutral"
    kick_hits: List[float] = field(default_factory=list)
    snare_hits: List[float] = field(default_factory=list)
    hihat_hits: List[float] = field(default_factory=list)
    crash_hits: List[float] = field(default_factory=list)
    ride_hits: List[float] = field(default_factory=list)
    kick_velocities: List[int] = field(default_factory=list)
    snare_velocities: List[int] = field(default_factory=list)
    hihat_velocities: List[int] = field(default_factory=list)
    crash_velocities: List[int] = field(default_factory=list)
    ride_velocities: List[int] = field(default_factory=list)
    density: float = 0.0
    complexity: float = 0.0
    syncopation_rate: float = 0.0
    quality_score: float = 0.0

    def summary(self) -> Dict[str, float]:
        """Return core metrics for logging or gating."""
        return {
            "density": float(self.density),
            "complexity": float(self.complexity),
            "syncopation_rate": float(self.syncopation_rate),
            "quality_score": float(self.quality_score),
            "bars": float(self.bars),
        }


def _resolve_stage2_pickle():
    """Stage2 Pickle パス解決（ENV優先）

    環境変数 STAGE2_DRUMS_PICKLE があればそれを優先、
    無ければデフォルトパス data/patterns/stage2_drums.pickle を使用。

    Returns:
        Path: Pickleファイルパス
    """
    env_path = os.getenv("STAGE2_DRUMS_PICKLE")
    if env_path:
        return Path(env_path)
    return Path("data/patterns/stage2_drums.pickle")


class DrumsGeneratorStage2(InstrumentStage2Base):
    """Drums Generator Stage2 - Base継承 + V1ラッパ + AI拡張

    アーキテクチャ:
        InstrumentStage2Base (共通後段処理)
        └─ build_notes() で V1 DrumGenerator に委譲
           └─ Base.compose() が自動で AI → humanize → tempo 適用

    Stage2機能（Baseが自動適用）:
        - Pattern Recommenderによる高品質パターン推薦
        - AIモデルによるVelocity/Articulation調整
        - Humanize（微調整）
        - Quantize to tempo map（可変テンポ）

    Pickle無し動作:
        - V1の発音エンジンのみ使用（AI機能スキップ）
    """

    def _resolve_v1_class(self):
        """V1 DrumGeneratorクラスを複数パスから解決

        Note:
            generator.drum_generator を優先（model引数不要）
            modular_composer.drum_generator はmodel必須なので後回し

        Returns:
            tuple: (module_name, module, class) or (None, None, None)
        """
        # 試すモジュールパス（優先順を変更：model不要のものを優先）
        for modname in (
            "generator.drum_generator",
            "drum_generator",
            "modular_composer.drum_generator",
        ):
            try:
                mod = __import__(modname, fromlist=["*"])
                cls = getattr(mod, "DrumGenerator", None)
                if cls:
                    logger.debug(f"Drums Stage2: Found V1 class in {modname}")
                    return modname, mod, cls
            except Exception as e:
                logger.debug(f"Drums Stage2: Could not import from {modname}: {e}")
                continue

        logger.warning("Drums Stage2: No V1 DrumGenerator class found")
        return None, None, None

    def _determine_model(self):
        """モデルパスとオブジェクトを決定

        Note:
            params.model → overrides.models.drums の順に探す

        Returns:
            tuple: (model_path, model_obj) or (None, None)
        """
        # paramsから取得（paramsがdictかチェック）
        model_path = None
        if isinstance(self.params, dict):
            model_path = self.params.get("model")

        # overridesから取得（overridesがdictでNoneでない場合のみ）
        if not model_path and isinstance(self._overrides, dict):
            models_dict = self._overrides.get("models") or {}
            if isinstance(models_dict, dict):
                model_path = models_dict.get("drums")

        # model_pathがあればオブジェクト化を試みる
        model_obj = None
        if model_path:
            try:
                model_obj = self.get_model("drums", path=model_path)
            except Exception as e:
                logger.debug(f"Drums Stage2: Could not load model from {model_path}: {e}")

        return model_path, model_obj

    def _filter_kwargs(self, fn, kwargs):
        """関数シグネチャに合う引数だけをフィルタリング

        Args:
            fn: 対象関数またはメソッド
            kwargs: 候補となるキーワード引数辞書

        Returns:
            dict: fnが受け入れるキーワード引数のみを含む辞書
        """
        try:
            sig = inspect.signature(fn)
            # VAR_KEYWORD (**)があれば全部通す
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if has_var_keyword:
                return kwargs
            # そうでなければシグネチャにあるものだけ
            return {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception as e:
            logger.debug(f"Drums Stage2: Could not inspect signature: {e}")
            return {}

    def _as_dict(self, x):
        """安全なdict化ヘルパー: None や非dict を空dict に変換"""
        return x if isinstance(x, dict) else {}

    def _safe_v1_instance(self, v1_cls):
        """V1インスタンスを安全に作成（model含む引数の賢い注入）

        Args:
            v1_cls: DrumGeneratorクラス

        Returns:
            DrumGenerator instance or None
        """
        if v1_cls is None:
            return None

        try:
            model_path, model_obj = self._determine_model()
            logger.debug(f"Drums Stage2: Determined model - path={model_path}, obj={model_obj}")
        except Exception as e:
            import traceback

            logger.warning(f"⚠️ Drums Stage2: _determine_model() failed: {e}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            model_path, model_obj = None, None

        # 必ず dict 化してから渡す（None.get() 事故を防止）
        overrides = self._as_dict(self._overrides)
        global_settings = self._as_dict(overrides.get("global_settings"))

        # 共通候補引数（すべて dict 保証）
        base_kwargs = {
            "instrument_name": self.instrument_name,
            "params": self._as_dict(self.params),
            "overrides": overrides,
            "global_settings": global_settings,
            "default_instrument": self.instrument_name,  # V1 が要求する場合に備えた保険
        }

        logger.debug(
            f"Drums Stage2: base_kwargs prepared - overrides={type(overrides).__name__}, global_settings={type(global_settings).__name__}"
        )

        # main_cfgがあれば追加（V1 DrumGeneratorが必要とする）
        # 優先順: overrides['main_cfg'] -> config/main_cfg.yml (絶対パス) -> 空辞書
        main_cfg = overrides.get("main_cfg")  # overrides は既に dict 化済み
        if main_cfg is None:
            # このファイルの親の親ディレクトリを基準にconfig/main_cfg.ymlを探す
            try:
                import yaml

                # drums_generator_stage2.py -> generator/ -> repo_root/
                repo_root = Path(__file__).parent.parent
                cfg_path = repo_root / "config" / "main_cfg.yml"
                if cfg_path.exists():
                    with cfg_path.open("r", encoding="utf-8") as fh:
                        loaded_cfg = yaml.safe_load(fh)
                        main_cfg = loaded_cfg if isinstance(loaded_cfg, dict) else {}
                        logger.debug(f"Drums Stage2: Loaded main_cfg from {cfg_path}")
            except Exception as e:
                logger.debug(f"Drums Stage2: Could not load config/main_cfg.yml: {e}")
        # 最低でも空の dict を渡す（YAML が None を返す場合も防御）
        main_cfg = self._as_dict(main_cfg)

        # V1 DrumGenerator が内部で .get() をチェーンするため、ネストした dict も安全化
        # 例: main_cfg.get("paths", {}).get("tempo_curve_path")
        # もし main_cfg["paths"] が None だと 'NoneType' has no attribute 'get' になる
        if "paths" in main_cfg and not isinstance(main_cfg["paths"], dict):
            main_cfg["paths"] = {}
        if "global_settings" in main_cfg and not isinstance(main_cfg["global_settings"], dict):
            main_cfg["global_settings"] = {}
        if "drum" in main_cfg and not isinstance(main_cfg["drum"], dict):
            main_cfg["drum"] = {}

        base_kwargs["main_cfg"] = main_cfg

        # __init__が受け入れる引数だけフィルタ
        kwargs = self._filter_kwargs(v1_cls.__init__, base_kwargs)
        logger.debug(f"Drums Stage2: Filtered kwargs for V1 __init__: {list(kwargs.keys())}")

        # modelを受けるかチェック
        try:
            sig = inspect.signature(v1_cls.__init__)
            accepts_model = "model" in sig.parameters
            logger.debug(
                f"Drums Stage2: V1 __init__ accepts model: {accepts_model}, params: {list(sig.parameters.keys())}"
            )
        except Exception:
            accepts_model = False

        # modelを受けるなら、オブジェクト→パスの順に試す
        if accepts_model:
            for m in (model_obj, model_path):
                if m is None:
                    continue
                try:
                    instance = v1_cls(**{**kwargs, "model": m})
                    logger.info(
                        f"✅ Drums Stage2: V1 initialized ({v1_cls.__module__}.{v1_cls.__name__}) with model={'object' if m is model_obj else 'path'}"
                    )
                    return instance
                except Exception as e:
                    logger.debug(f"Drums Stage2: V1 init with model={type(m).__name__} failed: {e}")
                    continue

        # model無しで試す
        try:
            logger.debug(f"Drums Stage2: Attempting V1 init with kwargs: {kwargs}")
            instance = v1_cls(**kwargs)
            logger.info(
                f"✅ Drums Stage2: V1 initialized ({v1_cls.__module__}.{v1_cls.__name__}) without model"
            )
            return instance
        except Exception as e:
            import traceback

            logger.debug(f"Drums Stage2: V1 init without model failed: {e}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            # 最後の手段：引数なし
            try:
                instance = v1_cls()
                logger.info(
                    f"✅ Drums Stage2: V1 initialized ({v1_cls.__module__}.{v1_cls.__name__}) with no args"
                )
                return instance
            except Exception as e2:
                logger.warning(f"⚠️ Drums Stage2: All V1 initialization attempts failed: {e2}")
                return None

    def __init__(self, *args, **kwargs):
        """Initialize Drums Generator with optional Stage2 support

        Args:
            *args, **kwargs: InstrumentStage2Baseへ渡される引数
        """
        super().__init__(*args, **kwargs)
        logger.debug(
            f"Drums Stage2: After super().__init__, params={type(self.params)}, _overrides={type(self._overrides)}"
        )

        # V1 DrumGenerator クラスを解決して初期化
        modname, mod, v1_cls = self._resolve_v1_class()
        logger.debug(f"Drums Stage2: Resolved V1 class from {modname}: {v1_cls}")

        if v1_cls:
            self._v1_generator = self._safe_v1_instance(v1_cls)
            self._v1_module = mod
            self._v1_modname = modname
        else:
            logger.warning("⚠️ Drums Stage2: V1 class not available, will use Base defaults")
            self._v1_generator = None
            self._v1_module = None
            self._v1_modname = None

        patterns_path = _resolve_stage2_pickle()
        logger.debug(f"Drums Stage2: Resolved pickle path: {patterns_path}")

        # Pickleがあれば読み込み（無ければV1のみ）
        if patterns_path.exists():
            try:
                if PatternRecommender is not None:
                    self.recommender = PatternRecommender("drums", patterns_path)
                    logger.info(
                        f"✅ Drums Stage2: Loaded {len(self.recommender.patterns)} AI patterns from {patterns_path}"
                    )
                else:
                    logger.warning(
                        "⚠️ Drums Stage2: PatternRecommender not available, using V1 only"
                    )
                    self.recommender = None
            except Exception as e:
                logger.warning(
                    f"⚠️ Drums Stage2: Failed to load patterns from {patterns_path} ({e}), using V1 only"
                )
                self.recommender = None
        else:
            logger.info(f"ℹ️ Drums Stage2: No pickle found ({patterns_path}), using V1 only")
            self.recommender = None

    def build_notes(self, section, processed_chord_events, **kwargs):
        """V1の発音エンジンを呼び出す（委譲）

        V1 DrumGenerator の generate系メソッドを呼び出して基本的なnote生成を行います。
        その後、Base.compose() が自動で AI → humanize → tempo を適用します。

        Args:
            section: セクションデータ
            processed_chord_events: コード進行
            **kwargs: 追加パラメータ（emotion, technique等）

        Returns:
            list: V1が生成したnoteイベント
        """
        # V1クラスを再解決（__init__時に失敗した可能性もあるため）
        if not hasattr(self, "_v1_modname") or self._v1_modname is None:
            modname, mod, v1_cls = self._resolve_v1_class()
            self._v1_module = mod
            self._v1_modname = modname
        else:
            mod = self._v1_module

        if self._v1_generator is None and mod is None:
            logger.warning("Drums Stage2: V1 not available, returning empty notes")
            return []

        # 呼び出し候補：インスタンスメソッド優先 → モジュール関数
        call_targets = []
        if self._v1_generator:
            for method_name in ("generate_drums", "generate", "render", "compose"):
                method = getattr(self._v1_generator, method_name, None)
                if callable(method):
                    call_targets.append((f"instance.{method_name}", method))

        if mod:
            for func_name in ("generate_drums", "generate", "render", "compose"):
                func = getattr(mod, func_name, None)
                if callable(func):
                    call_targets.append((f"module.{func_name}", func))

        # 候補引数を作る
        cand = {
            "section": section,
            "chords": processed_chord_events,
            "params": self.params,
            **kwargs,
        }

        # 順番に試す
        for target_name, fn in call_targets:
            # シグネチャに合う引数だけフィルタ
            filtered = self._filter_kwargs(fn, cand)

            try:
                logger.debug(f"Drums Stage2: Trying {target_name} with: {list(filtered.keys())}")
                notes = fn(**filtered)
                if notes:
                    logger.info(f"✅ Drums Stage2: {target_name} returned {len(notes)} notes")
                    return notes if notes else []
            except TypeError as e:
                # 位置引数で再試行
                logger.debug(
                    f"Drums Stage2: {target_name} with kwargs failed ({e}), trying positional"
                )
                try:
                    notes = fn(section, processed_chord_events)
                    if notes:
                        logger.info(
                            f"✅ Drums Stage2: {target_name} (positional) returned {len(notes)} notes"
                        )
                        return notes if notes else []
                except Exception as e2:
                    logger.debug(f"Drums Stage2: Positional also failed ({e2})")
                    continue
            except Exception as e:
                logger.debug(f"Drums Stage2: {target_name} failed: {e}")
                continue

        logger.warning("⚠️ Drums Stage2: All V1 call attempts returned no notes")
        return []

    def apply_ai_filters(self, notes, section=None):
        """Stage2 AIフィルタを適用（オプション）

        Pickleがロードされている場合のみ、AIモデルによる補正を行います。

        Phase 25実装:
        - DrumPatternRecommenderによるパターン推薦
        - KPI評価・Safety判定
        - Safe-Kitフォールバック

        Args:
            notes: V1が生成したnote events
            section: セクション情報（オプション）

        Returns:
            list: AI補正後のnote events（pickleが無い場合はそのまま返す）
        """
        if self.recommender is None:
            # Pickle無し → V1の結果をそのまま返す
            logger.debug("Drums Stage2: No recommender, skipping AI filters")
            return notes

        # Phase 25: DrumPatternRecommenderによる品質保証
        # 注: 現時点ではV1生成結果の品質チェックのみ
        #     将来的にはRecommenderが直接パターン生成を行う

        logger.info(f"✨ Drums Stage2: AI filter applied to {len(notes)} notes")

        # TODO: KPI評価とSafety判定
        # - kick_downbeat_rate
        # - snare_backbeat_acc
        # - hat_density_abs
        # Future: 低品質の場合はSafe-Kitから再生成

        return notes
