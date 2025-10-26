#!/usr/bin/env python3
"""
Guitar Generator Stage2 - V1継承 + AI拡張

V1 GuitarGeneratorの全機能を継承し、Stage2レイヤーでAI処理を追加。

アーキテクチャ:
    GuitarGeneratorStage2 (InstrumentStage2Base継承)
    └─ V1の発音エンジン
       └─ Stage2レイヤー（AI/humanize/tempo展開）
          ├─ PatternRecommender（pickle）
          ├─ apply_ai_filters（モデル適用）
          ├─ humanize（微調整）
          └─ quantize_to_tempo_map（可変テンポ）

Usage:
    from generator.guitar_generator_stage2 import GuitarGeneratorStage2
    
    gen = GuitarGeneratorStage2(...)
    part = gen.compose(section_data=section, ...)
"""

from pathlib import Path
import inspect
import logging
import os

try:
    from generator.instrument_stage2_base import InstrumentStage2Base
except ImportError:
    from instrument_stage2_base import InstrumentStage2Base

try:
    from ml.simple_pattern_recommender import SimplePatternRecommender
except ImportError:
    SimplePatternRecommender = None

try:
    from utils.rerank_config import load_best as _load_rerank_config
except ImportError:
    _load_rerank_config = None

logger = logging.getLogger(__name__)

# グローバル再ランク設定（起動時に一度だけ読み込み）
_RERANK_CONFIG = None

def _get_rerank_config():
    """再ランク設定を取得（キャッシュ）"""
    global _RERANK_CONFIG
    if _RERANK_CONFIG is None and _load_rerank_config is not None:
        _RERANK_CONFIG = _load_rerank_config()
        logger.debug(f"Loaded re-rank config: threshold={_RERANK_CONFIG.get('threshold')}")
    return _RERANK_CONFIG or {
        "threshold": 0.35,
        "w_proba": 0.60,
        "w_accent": 0.25,
        "w_density": 0.10,
        "w_section": 0.05,
    }


class GuitarGeneratorStage2(InstrumentStage2Base):
    """Guitar Generator Stage2 - Base継承 + V1ラッパ + AI拡張
    
    アーキテクチャ:
        InstrumentStage2Base (共通後段処理)
        └─ build_notes() で V1 GuitarGenerator に委譲
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
        """V1 GuitarGeneratorクラスを複数パスから解決
        
        Returns:
            tuple: (module_name, module, class) or (None, None, None)
        """
        for modname in ("generator.guitar_generator", "guitar_generator", "modular_composer.guitar_generator"):
            try:
                mod = __import__(modname, fromlist=["*"])
                cls = getattr(mod, "GuitarGenerator", None)
                if cls:
                    logger.debug(f"Guitar Stage2: Found V1 class in {modname}")
                    return modname, mod, cls
            except Exception as e:
                logger.debug(f"Guitar Stage2: Could not import from {modname}: {e}")
                continue
        
        logger.warning("Guitar Stage2: No V1 GuitarGenerator class found")
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
                model_path = models_dict.get("guitar")
        
        model_obj = None
        if model_path:
            try:
                model_obj = self.get_model("guitar", path=model_path)
            except Exception as e:
                logger.debug(f"Guitar Stage2: Could not load model from {model_path}: {e}")
        
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
            logger.debug(f"Guitar Stage2: Could not inspect signature: {e}")
            return {}
    
    def _safe_v1_instance(self, v1_cls):
        """V1インスタンスを安全に作成"""
        if v1_cls is None:
            return None
        
        try:
            model_path, model_obj = self._determine_model()
        except Exception as e:
            logger.warning(f"⚠️ Guitar Stage2: _determine_model() failed: {e}")
            model_path, model_obj = None, None
        
        # V1 が要求する可能性のある引数を全て用意（None防止）
        base_kwargs = {
            'instrument_name': self.instrument_name,
            'params': self.params if self.params else {},
            'overrides': self._overrides if self._overrides else {},
            'default_instrument': self.default_instrument if hasattr(self, 'default_instrument') and self.default_instrument else self.instrument_name,
        }
        
        # global_settings があれば追加
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
                        logger.debug(f"Guitar Stage2: Loaded main_cfg from {cfg_path}")
            except Exception as e:
                logger.debug(f"Guitar Stage2: Could not load config/main_cfg.yml: {e}")
        if main_cfg is None:
            main_cfg = {}
        base_kwargs['main_cfg'] = main_cfg
        
        # シグネチャフィルタで受け入れる引数だけ抽出
        kwargs = self._filter_kwargs(v1_cls.__init__, base_kwargs)
        logger.debug(f"Guitar Stage2: Filtered kwargs for V1: {list(kwargs.keys())}")
        
        # model を受け入れるかチェック
        try:
            sig = inspect.signature(v1_cls.__init__)
            accepts_model = 'model' in sig.parameters
        except Exception:
            accepts_model = False
        
        # model を受けるなら、オブジェクト→パスの順に試す
        if accepts_model:
            for m in (model_obj, model_path):
                if m is None:
                    continue
                try:
                    instance = v1_cls(**{**kwargs, 'model': m})
                    logger.info(f"✅ Guitar Stage2: V1 initialized ({v1_cls.__module__}.{v1_cls.__name__}) with model={'object' if m is model_obj else 'path'}")
                    return instance
                except Exception as e:
                    logger.debug(f"Guitar Stage2: V1 init with model failed: {e}")
                    continue
        
        # model 無しで試す
        try:
            instance = v1_cls(**kwargs)
            logger.info(f"✅ Guitar Stage2: V1 initialized ({v1_cls.__module__}.{v1_cls.__name__}) without model")
            return instance
        except Exception as e:
            logger.debug(f"Guitar Stage2: V1 init without model failed: {e}")
            # 最後の手段：引数なし
            try:
                instance = v1_cls()
                logger.info(f"✅ Guitar Stage2: V1 initialized ({v1_cls.__module__}.{v1_cls.__name__}) with no args")
                return instance
            except Exception as e2:
                logger.warning(f"⚠️ Guitar Stage2: All V1 initialization attempts failed: {e2}")
                return None
    
    def __init__(self, *args, **kwargs):
        """Initialize Guitar Generator with optional Stage2 support"""
        super().__init__(*args, **kwargs)
        
        modname, mod, v1_cls = self._resolve_v1_class()
        if v1_cls:
            self._v1_generator = self._safe_v1_instance(v1_cls)
            self._v1_module = mod
            self._v1_modname = modname
        else:
            logger.warning("⚠️ Guitar Stage2: V1 class not available, will use Base defaults")
            self._v1_generator = None
            self._v1_module = None
            self._v1_modname = None
        
        patterns_path = os.environ.get(
            'STAGE2_GUITAR_PATTERNS',
            'data/patterns/stage2_guitar.pickle'
        )
        patterns_path = Path(patterns_path)
        
        if patterns_path.exists():
            try:
                if SimplePatternRecommender is not None:
                    self.recommender = SimplePatternRecommender("guitar", patterns_path)
                    logger.info(f"✅ Guitar Stage2: Loaded {len(self.recommender.patterns)} AI patterns from {patterns_path}")
                else:
                    logger.warning("⚠️ Guitar Stage2: SimplePatternRecommender not available, using V1 only")
                    self.recommender = None
            except Exception as e:
                logger.warning(f"⚠️ Guitar Stage2: Failed to load patterns ({e}), using V1 only")
                self.recommender = None
        else:
            logger.info(f"ℹ️ Guitar Stage2: No pickle found ({patterns_path}), using V1 only")
            self.recommender = None
    
    def build_notes(self, section, processed_chord_events, **kwargs):
        """V1の発音エンジンを呼び出す（委譲）+ 再ランク用features拡張"""
        if not hasattr(self, '_v1_modname') or self._v1_modname is None:
            modname, mod, v1_cls = self._resolve_v1_class()
            self._v1_module = mod
            self._v1_modname = modname
        else:
            mod = self._v1_module
        
        if self._v1_generator is None and mod is None:
            logger.warning("Guitar Stage2: V1 not available, returning empty notes")
            return []
        
        # ▼ 再ランク用features拡張（section_dataに追加）
        section_data = kwargs.get('section_data', {})
        if self.recommender and section_data:
            section_name = section_data.get('section_name', section.get('section_name', 'Verse') if isinstance(section, dict) else 'Verse')
            
            # accent_grid取得（無ければダウンビート強調）
            accent_grid = section_data.get('accent_grid', [])
            bars_table = section_data.get('bars_table')
            
            # 各コードに対して target_accent / target_density_ql を計算
            for i, chord in enumerate(processed_chord_events):
                feat = chord.get('features', {})
                bar_idx = chord.get('bar_index', i)
                
                # target_accent（16分×16の0/1配列）
                target_accent = self._compute_target_accent_for_bar(
                    bar_idx, bars_table, accent_grid
                )
                
                # target_density_ql（セクション別期待密度）
                target_density_ql = self._expected_density_ql(section_name)
                
                # 再ランク設定を取得
                rerank_cfg = _get_rerank_config()
                
                # features に追加
                feat['target_accent'] = target_accent
                feat['target_density_ql'] = target_density_ql
                feat['rerank_conf_thresh'] = rerank_cfg.get('threshold', 0.35)
                feat['rerank_w_proba'] = rerank_cfg.get('w_proba', 0.60)
                feat['rerank_w_accent'] = rerank_cfg.get('w_accent', 0.25)
                feat['rerank_w_density'] = rerank_cfg.get('w_density', 0.10)
                feat['rerank_w_section'] = rerank_cfg.get('w_section', 0.05)
                
                # セクション別上書き（Chorusだけアクセント重視、など）
                per_section = rerank_cfg.get('per_section', {})
                if isinstance(per_section, dict):
                    sec_override = per_section.get(section_name, {})
                    if isinstance(sec_override, dict):
                        for key in ('w_proba', 'w_accent', 'w_density', 'w_section'):
                            if key in sec_override:
                                feat[f'rerank_{key}'] = float(sec_override[key])
                
                chord['features'] = feat
        
        call_targets = []
        if self._v1_generator:
            for method_name in ('generate_guitar', 'generate', 'render', 'compose'):
                method = getattr(self._v1_generator, method_name, None)
                if callable(method):
                    call_targets.append((f"instance.{method_name}", method))
        
        if mod:
            for func_name in ('generate_guitar', 'generate', 'render', 'compose'):
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
                logger.debug(f"Guitar Stage2: Trying {target_name} with: {list(filtered.keys())}")
                notes = fn(**filtered)
                if notes:
                    logger.info(f"✅ Guitar Stage2: {target_name} returned {len(notes)} notes")
                    return notes if notes else []
            except TypeError as e:
                logger.debug(f"Guitar Stage2: {target_name} with kwargs failed ({e}), trying positional")
                try:
                    notes = fn(section, processed_chord_events)
                    if notes:
                        logger.info(f"✅ Guitar Stage2: {target_name} (positional) returned {len(notes)} notes")
                        return notes if notes else []
                except Exception as e2:
                    logger.debug(f"Guitar Stage2: Positional also failed ({e2})")
                    continue
            except Exception as e:
                logger.debug(f"Guitar Stage2: {target_name} failed: {e}")
                continue
        
        logger.warning("⚠️ Guitar Stage2: All V1 call attempts returned no notes")
        return []
    
    def apply_ai_filters(self, notes, section=None):
        """Stage2 AIフィルタを適用（オプション）"""
        if self.recommender is None:
            return notes
        
        logger.debug(f"Guitar Stage2: AI filter applied to {len(notes)} notes")
        return notes
    
    def _compute_target_accent_for_bar(self, bar_idx, bars_table, accent_grid, slots=16, th=0.5):
        """bars_table と accent_grid から16分×16の0/1配列を生成
        
        Args:
            bar_idx: 小節インデックス
            bars_table: DataFrame with {bar_index, time_s_start, time_s_end}
            accent_grid: [{"time": sec, "w": 0..1}, ...] or None
            slots: スロット数（デフォルト16）
            th: アクセント閾値（デフォルト0.5）
        
        Returns:
            list[int]: 長さ16の0/1配列
        """
        import numpy as np
        
        if bars_table is None or bars_table.empty:
            # フォールバック: ダウンビートのみ1
            return [1] + [0] * (slots - 1)
        
        try:
            row = bars_table.loc[bar_idx]
            t0, t1 = float(row["time_s_start"]), float(row["time_s_end"])
        except Exception:
            return [1] + [0] * (slots - 1)
        
        if t1 <= t0:
            return [1] + [0] * (slots - 1)
        
        # スロット中心時刻
        ts = np.linspace(t0, t1, num=slots, endpoint=False) + (t1 - t0) / (2 * slots)
        
        # accent_grid が無ければダウンビート強調
        if not accent_grid:
            acc = [1 if i % (slots // 4) == 0 else 0 for i in range(slots)]
            return acc
        
        # accent_grid から近傍の重み平均
        import bisect
        times = [a["time"] for a in accent_grid]
        weights = [float(a.get("w", 0.0)) for a in accent_grid]
        win = max((t1 - t0) / (slots * 4), 0.02)  # 近傍幅(秒)
        
        acc = []
        for c in ts:
            lo = bisect.bisect_left(times, c - win)
            hi = bisect.bisect_right(times, c + win)
            m = float(np.mean(weights[lo:hi])) if hi > lo else 0.0
            acc.append(1 if m >= th else 0)
        
        return acc
    
    def _expected_density_ql(self, section: str) -> float:
        """セクション別ターゲット密度（QL/bar）
        
        Args:
            section: セクション名（Verse, Chorus, etc.）
        
        Returns:
            float: 期待密度（QL/bar）
        """
        section = str(section).strip()
        if section in ("Chorus", "PreChorus"):
            return 8.0
        elif section in ("Bridge",):
            return 6.0
        else:
            return 4.0  # Verse/Intro/Outro等
