"""Vermilion (lip) and skin segmentation via MediaPipe FaceLandmarker.

We isolate the *vermilion* — the coloured lip surface — as the region between
the outer lip contour and the inner mouth opening, so teeth and the dark mouth
interior are excluded geometrically before any colour is read. Skin is sampled
from small cheek/forehead patches to give a per-subject neutral reference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

# MediaPipe FaceMesh 468/478-point topology indices.
# Outer lip ring (vermilion border) and inner lip ring (mouth opening).
OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
             291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Cheek / forehead landmarks for skin sampling (away from brows, lips, nostrils).
SKIN_LANDMARKS = [50, 101, 118, 205, 280, 330, 347, 425, 151, 9]


@dataclass
class Segmentation:
    lip_rgb: np.ndarray                      # (N, 3) sRGB vermilion pixels
    skin_rgb: np.ndarray                     # (M, 3) sRGB skin pixels
    lip_mask: np.ndarray                     # HxW uint8 (debug/overlay)
    skin_mask: np.ndarray                    # HxW uint8
    landmarks: Optional[np.ndarray] = None   # (K, 2) pixel coords
    ok: bool = True
    note: str = ""
    meta: dict = field(default_factory=dict)


class FaceLips:
    """Thin wrapper around the FaceLandmarker; reuse one instance per process."""

    def __init__(self, model_path: str = _MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"face_landmarker.task not found at {model_path}. "
                "Download it from the MediaPipe model hub (see README)."
            )
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _poly(landmarks_px: np.ndarray, idx: list[int]) -> np.ndarray:
        return landmarks_px[idx].astype(np.int32)

    @staticmethod
    def _fill(shape, poly) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [poly], 255)
        return m

    def segment(self, image_rgb: np.ndarray) -> Segmentation:
        """Segment a HxWx3 **RGB** uint8 image."""
        h, w = image_rgb.shape[:2]
        empty = np.zeros((0, 3), np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(image_rgb))
        res = self._landmarker.detect(mp_img)
        if not res.face_landmarks:
            return Segmentation(empty, empty,
                                np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8),
                                ok=False, note="no face detected")

        lm = res.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        # Vermilion = outer lip fill minus inner mouth fill, eroded slightly to
        # pull the sample away from the lip/skin border and the wet line.
        outer = self._fill(image_rgb.shape, self._poly(pts, OUTER_LIP))
        inner = self._fill(image_rgb.shape, self._poly(pts, INNER_LIP))
        verm = cv2.subtract(outer, inner)
        verm = cv2.erode(verm, np.ones((3, 3), np.uint8), iterations=1)

        # Skin patches: small discs around cheek/forehead landmarks.
        skin_mask = np.zeros((h, w), np.uint8)
        r = max(3, int(0.012 * max(h, w)))
        for i in SKIN_LANDMARKS:
            if i < len(pts):
                cv2.circle(skin_mask, tuple(pts[i].astype(int)), r, 255, -1)

        lip_rgb = image_rgb[verm > 0].reshape(-1, 3)
        skin_rgb = image_rgb[skin_mask > 0].reshape(-1, 3)
        note = "" if len(lip_rgb) >= 40 else "few vermilion pixels"
        return Segmentation(
            lip_rgb=lip_rgb, skin_rgb=skin_rgb,
            lip_mask=verm, skin_mask=skin_mask,
            landmarks=pts, ok=len(lip_rgb) > 0, note=note,
            meta={"image_hw": (h, w), "n_landmarks": len(pts)},
        )

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
