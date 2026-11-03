"""End-to-end: image file / array -> LipColorResult."""

from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np

from .segment import FaceLips, Segmentation
from .color import extract_features, LipColorFeatures
from .classify import classify, ClassifierConfig, LipColorResult


def _load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class LipColorPipeline:
    """Reusable pipeline (loads the landmark model once)."""

    def __init__(self, cfg: Optional[ClassifierConfig] = None):
        self.cfg = cfg or ClassifierConfig()
        self.face = FaceLips()

    def analyze(self, image: Union[str, np.ndarray]) -> LipColorResult:
        rgb = _load_rgb(image) if isinstance(image, str) else image
        seg: Segmentation = self.face.segment(rgb)
        if not seg.ok:
            return LipColorResult("other", None, 0.0,
                                  f"segmentation failed: {seg.note}",
                                  LipColorFeatures.empty().as_dict())
        feats = extract_features(seg.lip_rgb, seg.skin_rgb)
        return classify(feats, self.cfg)

    def close(self):
        self.face.close()


def analyze_image(path: str, cfg: Optional[ClassifierConfig] = None) -> LipColorResult:
    """Convenience one-shot (creates and tears down a pipeline)."""
    p = LipColorPipeline(cfg)
    try:
        return p.analyze(path)
    finally:
        p.close()
