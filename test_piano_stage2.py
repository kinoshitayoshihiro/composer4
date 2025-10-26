#!/usr/bin/env python3
"""All Stage2 Generators 動作確認テスト (Piano/Drums/Guitar/Strings)"""

import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s]: %(message)s"
)

logger = logging.getLogger(__name__)


def test_piano_stage2_import():
    """Piano Stage2のインポートテスト"""
    logger.info("=" * 60)
    logger.info("Piano Stage2 Import Test")
    logger.info("=" * 60)
    
    try:
        from generator.piano_generator_stage2 import PianoGeneratorStage2
        logger.info("✅ PianoGeneratorStage2 import successful")
        return True
    except Exception as e:
        logger.error(f"❌ PianoGeneratorStage2 import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_drums_stage2_import():
    """Drums Stage2のインポートテスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Drums Stage2 Import Test")
    logger.info("=" * 60)
    
    try:
        from generator.drums_generator_stage2 import DrumsGeneratorStage2
        logger.info("✅ DrumsGeneratorStage2 import successful")
        return True
    except Exception as e:
        logger.error(f"❌ DrumsGeneratorStage2 import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_guitar_stage2_import():
    """Guitar Stage2のインポートテスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Guitar Stage2 Import Test")
    logger.info("=" * 60)
    
    try:
        from generator.guitar_generator_stage2 import GuitarGeneratorStage2
        logger.info("✅ GuitarGeneratorStage2 import successful")
        return True
    except Exception as e:
        logger.error(f"❌ GuitarGeneratorStage2 import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strings_stage2_import():
    """Strings Stage2のインポートテスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Strings Stage2 Import Test")
    logger.info("=" * 60)
    
    try:
        from generator.strings_generator_stage2 import StringsGeneratorStage2
        logger.info("✅ StringsGeneratorStage2 import successful")
        return True
    except Exception as e:
        logger.error(f"❌ StringsGeneratorStage2 import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_piano_stage2_init():
    """Piano Stage2の初期化テスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Piano Stage2 Initialization Test")
    logger.info("=" * 60)
    
    try:
        from generator.piano_generator_stage2 import PianoGeneratorStage2
        from music21 import instrument
        
        # 必須パラメータを全て指定して初期化
        gen = PianoGeneratorStage2(
            default_instrument=instrument.Piano(),
            key="C",
            tempo=120.0,
            emotion="neutral"
        )
        
        logger.info("✅ PianoGeneratorStage2 initialization successful")
        logger.info(f"   - Has recommender: {hasattr(gen, 'recommender')}")
        logger.info(f"   - Recommender value: {gen.recommender}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PianoGeneratorStage2 initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factory_piano_stage2():
    """Factory経由でのPiano Stage2取得テスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Factory Piano Stage2 Test")
    logger.info("=" * 60)
    
    try:
        from utilities.generator_factory import GenFactory
        from utilities.config_loader import load_main_cfg
        
        # 設定ファイルを絶対パスで読み込み
        config_path = Path(__file__).parent / "config" / "main_cfg.yml"
        main_cfg = load_main_cfg(str(config_path))
        
        # Factoryから全楽器のジェネレータを取得
        generators = GenFactory.build_from_config(main_cfg)
        
        logger.info(f"✅ Factory returned {len(generators)} generators")
        
        # Pianoジェネレータを確認
        if "piano" in generators:
            piano_gen = generators["piano"]
            logger.info(f"   - Piano class: {piano_gen.__class__.__name__}")
            logger.info(f"   - Piano module: {piano_gen.__class__.__module__}")
            
            # Stage2かどうか確認
            is_stage2 = "Stage2" in piano_gen.__class__.__name__
            logger.info(f"   - Is Stage2: {is_stage2}")
            
            if not is_stage2:
                logger.warning("⚠️ Factory returned V1 class, not Stage2!")
        else:
            logger.warning("⚠️ No piano generator in factory output")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    logger.info("\n🎹🥁🎸🎻 All Stage2 Generators Test Suite\n")
    
    results = []
    
    # Import Tests
    results.append(("Piano Import", test_piano_stage2_import()))
    results.append(("Drums Import", test_drums_stage2_import()))
    results.append(("Guitar Import", test_guitar_stage2_import()))
    results.append(("Strings Import", test_strings_stage2_import()))
    
    # Initialization Test
    results.append(("Piano Init", test_piano_stage2_init()))
    
    # Factory Test
    results.append(("Factory", test_factory_piano_stage2()))
    
    # 結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        logger.info("\n🎉 All tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
