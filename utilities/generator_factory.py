# utilities/generator_factory.py
from utilities.drum_map import get_drum_map
from typing import Any
BasePartGenerator: Any = Any
from music21 import instrument as m21instrument
import importlib
import logging
from types import ModuleType

logger = logging.getLogger(__name__)


def _safe_import(candidates: list[tuple[str, str]]):
    """Try importing a list of (module_path, attr) candidates and return the attr from the first successful import.

    Returns None if none succeed (swallow exceptions).
    """
    from pathlib import Path
    import importlib.util

    for modpath, attr in candidates:
        # 1) try normal import
        try:
            mod = importlib.import_module(modpath)
            return getattr(mod, attr)
        except Exception:
            pass

        # 2) try loading directly from generator/ directory by filename to avoid package __init__ side-effects
        try:
            module_name = modpath.split('.')[-1]
            repo_root = Path(__file__).resolve().parent.parent
            candidate_file = repo_root / 'generator' / f"{module_name}.py"
            if candidate_file.exists():
                spec = importlib.util.spec_from_file_location(f"stage2_loader.{module_name}", str(candidate_file))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                    return getattr(mod, attr)
        except Exception:
            pass

    return None

# Stage2版をインポート（全楽器対応）
def _resolve_stage2_classes():
    """Return a mapping role->Stage2 class (or None) by attempting safe imports.

    This is executed at runtime inside GenFactory.build_from_config to avoid
    circular imports at module import time.
    """
    mapping = {}
    mapping['bass'] = _safe_import([
        ("generator.bass_generator_stage2", "BassGeneratorStage2"),
        ("bass_generator_stage2", "BassGeneratorStage2"),
    ])
    mapping['guitar'] = _safe_import([
        ("generator.guitar_generator_stage2", "GuitarGeneratorStage2"),
        ("guitar_generator_stage2", "GuitarGeneratorStage2"),
    ])
    mapping['strings'] = _safe_import([
        ("generator.strings_generator_stage2", "StringsGeneratorStage2"),
        ("strings_generator_stage2", "StringsGeneratorStage2"),
    ])
    mapping['drums'] = _safe_import([
        ("generator.drums_generator_stage2", "DrumsGeneratorStage2"),
        ("drums_generator_stage2", "DrumsGeneratorStage2"),
    ])
    mapping['piano'] = _safe_import([
        ("generator.piano_generator_stage2", "PianoGeneratorStage2"),
        ("piano_generator_stage2", "PianoGeneratorStage2"),
    ])
    return mapping


ROLE_DISPATCH_DEFAULT: dict[str, Any] = {}


class GenFactory:
    @staticmethod
    def build_from_config(main_cfg, rhythm_lib=None, tempo_map=None):
        """main_cfg['part_defaults'] を読み取り各 Generator を初期化

        Parameters
        ----------
        main_cfg : dict
            Parsed configuration dictionary from ``load_main_cfg``.
        rhythm_lib : RhythmLibrary | None
            Optional rhythm library object providing pattern dictionaries for
            each part. If provided, the corresponding pattern set is passed to
            each generator via ``part_parameters``.
        """
        global_settings = main_cfg.get("global_settings", {})
        drum_map = get_drum_map(global_settings.get("drum_map"))
        
        # Stage2クラスを読み込み（AI推薦版を必須使用）
        stage2_map = _resolve_stage2_classes()

        # Melody/Saxは専用クラスを使用（Stage2なし）
        from generator.melody_generator import MelodyGenerator
        from generator.sax_generator import SaxGenerator

        # ROLE_DISPATCH: Stage2クラスを必須使用（AI推薦版）
        ROLE_DISPATCH = {
            "piano": stage2_map['piano'],
            "drums": stage2_map['drums'],  # ← Stage2に変更
            "bass": stage2_map['bass'],
            "guitar": stage2_map['guitar'],
            "strings": stage2_map['strings'],
            "melody": MelodyGenerator,
            "counter": MelodyGenerator,
            "pad": stage2_map['strings'],
            "riff": MelodyGenerator,
            "rhythm": stage2_map['guitar'],
            "unison": stage2_map['strings'],
            "sax": SaxGenerator,
        }
        
        # Stage2選択状況をログ出力
        logger.info("=" * 60)
        logger.info("Generator Factory: Stage2 Class Selection")
        logger.info("=" * 60)
        for role, cls in ROLE_DISPATCH.items():
            if cls is not None:
                class_name = cls.__name__
                is_stage2 = "Stage2" in class_name
                marker = "✅ Stage2" if is_stage2 else "⚠️ V1"
                logger.info(f"{marker} {role:10s} → {class_name}")
            else:
                logger.warning(f"❌ {role:10s} → None (not available)")
        logger.info("=" * 60)
        
        # Stage2クラスが見つからない場合はエラー
        missing = [role for role, cls in ROLE_DISPATCH.items() 
                   if cls is None and role not in ['melody', 'counter', 'riff', 'sax']]
        if missing:
            raise ImportError(
                f"❌ Stage2クラスが見つかりません: {', '.join(missing)}\n"
                f"→ generator/*_generator_stage2.py ファイルを確認してください。"
            )
        gens = {}
        for part_name, part_cfg in main_cfg["part_defaults"].items():
            role = part_cfg.get("role", part_name)  # role が無ければ楽器名と同じ
            try:
                GenCls = ROLE_DISPATCH[role]
            except KeyError as e:
                raise KeyError(f"Unknown role '{role}' for part '{part_name}'") from e
            cleaned_part_cfg = dict(part_cfg)
            cleaned_part_cfg.pop("main_cfg", None)
            inst_spec = cleaned_part_cfg.get("default_instrument", part_name)
            if isinstance(inst_spec, str):
                try:
                    inst_obj = m21instrument.fromString(inst_spec)
                except Exception:
                    try:
                        inst_obj = m21instrument.fromString(part_name)
                    except Exception:
                        inst_obj = m21instrument.Percussion()
            else:
                inst_obj = inst_spec

            lib_params = {}
            if rhythm_lib is not None:
                if part_name == "drums":
                    lib_params = getattr(rhythm_lib, "drum_patterns", {}) or {}
                elif part_name == "bass":
                    lib_params = getattr(rhythm_lib, "bass_patterns", {}) or {}
                elif part_name == "piano":
                    lib_params = getattr(rhythm_lib, "piano_patterns", {}) or {}
                elif part_name in ("guitar", "rhythm"):
                    lib_params = getattr(rhythm_lib, "guitar", {}) or {}

            if lib_params and not isinstance(next(iter(lib_params.values()), {}), dict):
                lib_params = {
                    k: v.model_dump() if hasattr(v, "model_dump") else dict(v)
                    for k, v in lib_params.items()
                }

            part_params = cleaned_part_cfg.get("part_parameters", {})
            if lib_params:
                part_params = {**lib_params, **part_params}
            cleaned_part_cfg["part_parameters"] = part_params

            if part_name == "drums":
                cleaned_part_cfg["drum_map"] = drum_map

            gens[part_name] = GenCls(
                global_settings=global_settings,
                default_instrument=inst_obj,
                part_name=part_name,
                global_tempo=global_settings.get("tempo_bpm"),
                global_time_signature=global_settings.get("time_signature", "4/4"),
                global_key_signature_tonic=global_settings.get("key_tonic"),
                global_key_signature_mode=global_settings.get("key_mode"),
                main_cfg=main_cfg,
                tempo_map=tempo_map,
                **cleaned_part_cfg,
            )
        return gens


# ROLE_DISPATCH is constructed at runtime inside GenFactory.build_from_config
ROLE_DISPATCH: dict[str, Any] = {}
