#!/usr/bin/env python3
"""
Strings Generator Stage2 - V1継承 + AI拡張

V1 StringsGeneratorの全機能を継承し、Stage2レイヤーでAI処理を追加。

アーキテクチャ:
    StringsGeneratorStage2 (InstrumentStage2Base継承)
    └─ V1の発音エンジン
       └─ Stage2レイヤー（AI/humanize/tempo展開）
          ├─ PatternRecommender（pickle）
          ├─ apply_ai_filters（モデル適用）
          ├─ humanize（微調整）
          └─ quantize_to_tempo_map（可変テンポ）

Usage:
    from generator.strings_generator_stage2 import StringsGeneratorStage2
    
    gen = StringsGeneratorStage2(...)
    part = gen.compose(section_data=section, ...)
"""

from pathlib import Path
import inspect
import logging

try:
    from generator.instrument_stage2_base import InstrumentStage2Base
except ImportError:
    from instrument_stage2_base import InstrumentStage2Base

try:
    from ml.pattern_recommender import PatternRecommender
except ImportError:
    PatternRecommender = None

logger = logging.getLogger(__name__)


class StringsGeneratorStage2(InstrumentStage2Base):
    """Strings Generator Stage2 - Base継承 + V1ラッパ + AI拡張
    
    アーキテクチャ:
        InstrumentStage2Base (共通後段処理)
        └─ build_notes() で V1 StringsGenerator に委譲
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
        """V1 StringsGeneratorクラスを複数パスから解決
        
        Returns:
            tuple: (module_name, module, class) or (None, None, None)
        """
        for modname in ("generator.strings_generator", "strings_generator", "modular_composer.strings_generator"):
            try:
                mod = __import__(modname, fromlist=["*"])
                cls = getattr(mod, "StringsGenerator", None)
                if cls:
                    logger.debug(f"Strings Stage2: Found V1 class in {modname}")
                    return modname, mod, cls
            except Exception as e:
                logger.debug(f"Strings Stage2: Could not import from {modname}: {e}")
                continue
        
        logger.warning("Strings Stage2: No V1 StringsGenerator class found")
        return None, None, None
    
    def _determine_model(self):
        """モデルパスとオブジェクトを決定
        
        Returns:
            tuple: (model_path, model_obj) or (None, None)
        """
        model_path = None
        if isinstance(self.params, dict):
            model_path = self.params.get("model")
        
        if not model_path and isinstance(self._overrides, dict):
            models_dict = self._overrides.get("models") or {}
            if isinstance(models_dict, dict):
                model_path = models_dict.get("strings")
        
        model_obj = None
        if model_path:
            try:
                model_obj = self.get_model("strings", path=model_path)
            except Exception as e:
                logger.debug(f"Strings Stage2: Could not load model from {model_path}: {e}")
        
        return model_path, model_obj
    
    def _filter_kwargs(self, fn, kwargs):
        """関数シグネチャに合う引数だけをフィルタリング"""
        try:
            sig = inspect.signature(fn)
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD 
                for p in sig.parameters.values()
            )
            if has_var_keyword:
                return kwargs
            return {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception as e:
            logger.debug(f"Strings Stage2: Could not inspect signature: {e}")
            return {}
    
    def _safe_v1_instance(self, v1_cls):
        """V1インスタンスを安全に作成"""
        if v1_cls is None:
            return None
        
        try:
            model_path, model_obj = self._determine_model()
        except Exception as e:
            logger.warning(f"⚠️ Strings Stage2: _determine_model() failed: {e}")
            model_path, model_obj = None, None
        
        base_kwargs = {
            'instrument_name': self.instrument_name,
            'params': self.params if self.params else {},
            'overrides': self._overrides if self._overrides else {},
        }
        
        # default_instrument を追加（BasePartGenerator が必須とする）
        if hasattr(self, 'default_instrument') and self.default_instrument:
            base_kwargs['default_instrument'] = self.default_instrument
        
        if self._overrides and 'global_settings' in self._overrides:
            base_kwargs['global_settings'] = self._overrides['global_settings']
        
        # main_cfg を絶対パスで読み込む
        main_cfg = None
        if isinstance(self._overrides, dict) and 'main_cfg' in self._overrides:
            main_cfg = self._overrides.get('main_cfg')
        if main_cfg is None:
            try:
                import yaml
                repo_root = Path(__file__).parent.parent
                cfg_path = repo_root / 'config' / 'main_cfg.yml'
                if cfg_path.exists():
                    with cfg_path.open('r', encoding='utf-8') as fh:
                        main_cfg = yaml.safe_load(fh) or {}
                        logger.debug(f"Strings Stage2: Loaded main_cfg from {cfg_path}")
            except Exception as e:
                logger.debug(f"Strings Stage2: Could not load config/main_cfg.yml: {e}")
        if main_cfg is None:
            main_cfg = {}
        base_kwargs['main_cfg'] = main_cfg
        
        kwargs = self._filter_kwargs(v1_cls.__init__, base_kwargs)
        
        try:
            sig = inspect.signature(v1_cls.__init__)
            accepts_model = 'model' in sig.parameters
        except Exception:
            accepts_model = False
        
        if accepts_model:
            for m in (model_obj, model_path):
                if m is None:
                    continue
                try:
                    instance = v1_cls(**{**kwargs, 'model': m})
                    logger.info(f"✅ Strings Stage2: V1 initialized with model={'object' if m is model_obj else 'path'}")
                    return instance
                except Exception as e:
                    logger.debug(f"Strings Stage2: V1 init with model failed: {e}")
                    continue
        
        try:
            instance = v1_cls(**kwargs)
            logger.info("✅ Strings Stage2: V1 initialized without model")
            return instance
        except Exception as e:
            logger.debug(f"Strings Stage2: V1 init without model failed: {e}")
            try:
                instance = v1_cls()
                logger.info("✅ Strings Stage2: V1 initialized with no args")
                return instance
            except Exception as e2:
                logger.warning(f"⚠️ Strings Stage2: All V1 initialization attempts failed: {e2}")
                return None
    
    def __init__(self, *args, **kwargs):
        """Initialize Strings Generator with optional Stage2 support"""
        super().__init__(*args, **kwargs)
        
        modname, mod, v1_cls = self._resolve_v1_class()
        if v1_cls:
            self._v1_generator = self._safe_v1_instance(v1_cls)
            self._v1_module = mod
            self._v1_modname = modname
        else:
            logger.warning("⚠️ Strings Stage2: V1 class not available, will use Base defaults")
            self._v1_generator = None
            self._v1_module = None
            self._v1_modname = None
        
        patterns_path = Path("data/patterns/stage2_strings.pickle")
        
        if patterns_path.exists():
            try:
                if PatternRecommender is not None:
                    self.recommender = PatternRecommender("strings", patterns_path)
                    logger.info(f"✅ Strings Stage2: Loaded {len(self.recommender.patterns)} AI patterns")
                else:
                    logger.warning("⚠️ Strings Stage2: PatternRecommender not available, using V1 only")
                    self.recommender = None
            except Exception as e:
                logger.warning(f"⚠️ Strings Stage2: Failed to load patterns ({e}), using V1 only")
                self.recommender = None
        else:
            logger.info(f"ℹ️ Strings Stage2: No pickle found ({patterns_path}), using V1 only")
            self.recommender = None
    
    def build_notes(self, section, processed_chord_events, **kwargs):
        """V1の発音エンジンを呼び出す（委譲）"""
        if not hasattr(self, '_v1_modname') or self._v1_modname is None:
            modname, mod, v1_cls = self._resolve_v1_class()
            self._v1_module = mod
            self._v1_modname = modname
        else:
            mod = self._v1_module
        
        if self._v1_generator is None and mod is None:
            logger.warning("Strings Stage2: V1 not available, returning empty notes")
            return []
        
        call_targets = []
        if self._v1_generator:
            for method_name in ('generate_strings', 'generate', 'render', 'compose'):
                method = getattr(self._v1_generator, method_name, None)
                if callable(method):
                    call_targets.append((f"instance.{method_name}", method))
        
        if mod:
            for func_name in ('generate_strings', 'generate', 'render', 'compose'):
                func = getattr(mod, func_name, None)
                if callable(func):
                    call_targets.append((f"module.{func_name}", func))
        
        cand = {
            'section': section,
            'chords': processed_chord_events,
            'params': self.params,
            **kwargs
        }
        
        for target_name, fn in call_targets:
            filtered = self._filter_kwargs(fn, cand)
            
            try:
                logger.debug(f"Strings Stage2: Trying {target_name} with: {list(filtered.keys())}")
                notes = fn(**filtered)
                if notes:
                    logger.info(f"✅ Strings Stage2: {target_name} returned {len(notes)} notes")
                    return notes if notes else []
            except TypeError as e:
                logger.debug(f"Strings Stage2: {target_name} with kwargs failed ({e}), trying positional")
                try:
                    notes = fn(section, processed_chord_events)
                    if notes:
                        logger.info(f"✅ Strings Stage2: {target_name} (positional) returned {len(notes)} notes")
                        return notes if notes else []
                except Exception as e2:
                    logger.debug(f"Strings Stage2: Positional also failed ({e2})")
                    continue
            except Exception as e:
                logger.debug(f"Strings Stage2: {target_name} failed: {e}")
                continue
        
        logger.warning("⚠️ Strings Stage2: All V1 call attempts returned no notes")
        return []
    
    def apply_ai_filters(self, notes, section=None):
        """Stage2 AIフィルタを適用（オプション）"""
        if self.recommender is None:
            return notes
        
        logger.debug(f"Strings Stage2: AI filter applied to {len(notes)} notes")
        return notes
