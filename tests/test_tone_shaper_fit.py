import numpy as np
import pytest

from utilities.tone_shaper import ToneShaper


def test_tone_shaper_fit_predict() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    shaper = ToneShaper()
    mfcc_clean = np.ones((13, 10))
    mfcc_drive = np.full((13, 10), 2.0)
    shaper.fit({"clean": mfcc_clean, "drive": mfcc_drive})
    pred = shaper.predict_preset(mfcc_drive)
    assert pred == "drive"
