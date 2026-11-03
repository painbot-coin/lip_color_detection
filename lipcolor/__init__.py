"""Measurement-based lip-colour classification.

Instead of asking a vision-language model to *reason* about lip colour from a
text definition (which drifts on borderline cases), this package *measures* the
vermilion colour with face-mesh segmentation + colour science and classifies it
with deterministic, calibratable rules in CIELAB.
"""

from .color import LipColorFeatures, extract_features
from .classify import ClassifierConfig, LipColorResult, classify
from .pipeline import LipColorPipeline, analyze_image

__all__ = [
    "LipColorFeatures",
    "extract_features",
    "ClassifierConfig",
    "LipColorResult",
    "classify",
    "LipColorPipeline",
    "analyze_image",
]
