"""
OtobonAI: Rulebook-driven composition system
"""

__version__ = "0.1.5"

from .rulebook_engine import Rulebook, Rule, RuleActionEmotion, RuleActionGuideTone
from .emotion_ai import EmotionAI
from .guide_tone_ai import GuideToneAI

__all__ = [
    "Rulebook",
    "Rule",
    "RuleActionEmotion",
    "RuleActionGuideTone",
    "EmotionAI",
    "GuideToneAI",
]
