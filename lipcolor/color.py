"""Colour measurement: pixels -> robust CIELAB features.

The design goal is *repeatability* and robustness to the three nuisance factors
the task names: skin tone, lighting, and camera white balance.

Key ideas
---------
* Work in **CIELAB** (perceptually organised; a* = green->red, b* = blue->yellow).
* Reject non-lip pixels *before* averaging: specular highlights, teeth, deep
  mouth-interior shadow. A single mean over the raw polygon is badly corrupted
  by these, which is one reason naive colour reads are unstable.
* **White-balance / skin-tone normalise before measuring.** A camera's white
  balance is, to good approximation, a per-channel (diagonal) gain on linear
  RGB; so is much of the illuminant. We estimate that gain from the subject's
  skin and invert it on the lip pixels (von Kries adaptation to a canonical
  skin):

        lip_corrected = ref_skin * (lip_measured / skin_measured)        (1)

  The unknown gain G cancels exactly (lip_measured = G·lip_true,
  skin_measured = G·skin_true), leaving ref_skin·(lip_true/skin_true) — a
  quantity that depends only on the *reflectance ratio* of lips to the person's
  own skin. That is the robust signal: it removes the camera white balance and
  largely normalises skin tone in one step, which is what lets us tell a cool
  blue-leaning red from a warm coral without knowing the white balance.
* Measure in **CIELAB** (perceptual; a* = green->red, b* = blue->yellow) on the
  corrected pixels, after rejecting specular highlights, teeth and deep shadow.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from skimage import color as skcolor


# --------------------------------------------------------------------------- #
# Low-level colour conversion
# --------------------------------------------------------------------------- #
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) uint8/float sRGB array to (N, 3) CIELAB (D65).

    Accepts values in [0, 255] (uint8-style) or [0, 1] (float).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"expected (N, 3) RGB, got {rgb.shape}")
    if rgb.max() > 1.0 + 1e-6:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    # skimage wants an image; treat the pixel list as a 1-row image.
    lab = skcolor.rgb2lab(rgb[None, :, :])[0]
    return lab


def lab_chroma_hue(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (chroma C*, hue angle h in degrees [-180, 180)) for Lab array."""
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    return chroma, hue


# Canonical skin under neutral white balance (sRGB). The normalisation target;
# only its *chromaticity* matters for the warm/cool decision.
REF_SKIN_RGB = (208.0, 170.0, 150.0)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    s = np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)
    return np.clip(s * 255.0, 0.0, 255.0)


def white_balance_gain(
    skin_rgb_center: np.ndarray,
    ref_skin_rgb=REF_SKIN_RGB,
    lum_clip=(0.15, 6.0),
    chroma_clip=(0.55, 1.8),
) -> np.ndarray:
    """Per-channel linear gain mapping measured skin -> canonical skin.

    Inverting this gain on any pixel performs a von Kries chromatic adaptation
    that removes the capture's white balance (eqn (1) in the module docstring).

    The raw gain is split into a **luminance** factor (its geometric mean) and
    the **chromatic** residual (the part that actually corrects the colour
    cast). They are clipped separately: brightness may scale widely (deep vs
    pale skin), but the chromatic correction is kept and only gently bounded.
    Clipping the raw gain directly would, for deep skin, drive all three
    channels to a common ceiling and silently cancel the colour correction.
    """
    skin_lin = srgb_to_linear(np.asarray(skin_rgb_center).reshape(1, 3))[0]
    ref_lin = srgb_to_linear(np.asarray(ref_skin_rgb).reshape(1, 3))[0]
    gain = ref_lin / np.maximum(skin_lin, 1e-4)
    lum = float(np.exp(np.mean(np.log(np.clip(gain, 1e-4, None)))))  # geo-mean
    chroma = gain / lum
    lum = float(np.clip(lum, *lum_clip))
    chroma = np.clip(chroma, *chroma_clip)
    return lum * chroma


def apply_gain(rgb: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Apply a linear per-channel gain to sRGB pixels, return sRGB."""
    lin = srgb_to_linear(rgb)
    return linear_to_srgb(lin * gain.reshape(1, 3))


# --------------------------------------------------------------------------- #
# Robust pixel-set summary
# --------------------------------------------------------------------------- #
def _robust_center(lab: np.ndarray) -> np.ndarray:
    """One-step robust central Lab colour of a pixel set.

    Median, then a trimmed mean of pixels within a moderate radius of the
    median. Deterministic (no RNG), resistant to residual outliers.
    """
    if len(lab) == 0:
        return np.array([np.nan, np.nan, np.nan])
    med = np.median(lab, axis=0)
    d = np.linalg.norm(lab - med, axis=1)
    keep = d <= np.percentile(d, 80)
    if keep.sum() >= max(8, int(0.2 * len(lab))):
        return lab[keep].mean(axis=0)
    return med


def clean_lip_pixels(
    lab: np.ndarray,
    *,
    specular_L: float = 85.0,
    specular_C: float = 18.0,
    teeth_L: float = 72.0,
    teeth_a: float = 12.0,
    shadow_L: float = 12.0,
) -> np.ndarray:
    """Boolean mask of lip pixels worth keeping.

    Drops:
      * specular highlights  (very light & near-neutral: lip gloss glints)
      * teeth                (light & not red: a* low)
      * deep shadow / mouth interior (near black)
    """
    L, a = lab[:, 0], lab[:, 1]
    C = np.hypot(lab[:, 1], lab[:, 2])
    specular = (L > specular_L) & (C < specular_C)
    teeth = (L > teeth_L) & (a < teeth_a)
    shadow = L < shadow_L
    return ~(specular | teeth | shadow)


# --------------------------------------------------------------------------- #
# Feature bundle
# --------------------------------------------------------------------------- #
@dataclass
class LipColorFeatures:
    # Absolute lip colour (robust centre, CIELAB).
    L: float
    a: float
    b: float
    C: float           # chroma  sqrt(a^2 + b^2)
    h: float           # hue angle, degrees [-180, 180)

    # Skin reference (robust centre, CIELAB). NaN if no skin sampled.
    skin_L: float
    skin_a: float
    skin_b: float

    # Relative-to-skin vector (the white-balance / skin-tone robust part).
    dL: float          # lip_L - skin_L
    da: float          # lip_a - skin_a   (>0: lips redder than skin)
    db: float          # lip_b - skin_b   (<0: lips bluer/cooler than skin)
    rel_C: float       # sqrt(da^2 + db^2): how much more chromatic than skin
    rel_h: float       # atan2(db, da): hue of the "lip-ness" direction

    # Bookkeeping.
    n_lip_px: int
    n_skin_px: int

    # White-balance normalisation applied (von Kries to canonical skin).
    wb_corrected: bool = False
    wb_gain: tuple = (1.0, 1.0, 1.0)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def empty(cls) -> "LipColorFeatures":
        nan = float("nan")
        return cls(nan, nan, nan, nan, nan, nan, nan, nan,
                   nan, nan, nan, nan, nan, n_lip_px=0, n_skin_px=0)


def _robust_rgb_center(rgb: np.ndarray) -> np.ndarray:
    """Robust central sRGB of a skin pixel set (specular glints removed)."""
    lab = rgb_to_lab(rgb)
    L = lab[:, 0]
    C = np.hypot(lab[:, 1], lab[:, 2])
    keep = ~((L > 88) & (C < 14))
    sub = rgb[keep] if keep.sum() >= 8 else rgb
    med = np.median(sub, axis=0)
    d = np.linalg.norm(sub - med, axis=1)
    inl = sub[d <= np.percentile(d, 80)]
    return inl.mean(axis=0) if len(inl) >= 8 else med


def extract_features(
    lip_rgb: np.ndarray,
    skin_rgb: Optional[np.ndarray] = None,
    *,
    wb_normalize: bool = False,
) -> LipColorFeatures:
    """Compute robust CIELAB features from lip (and optional skin) pixel sets.

    By default the lip colour is measured **absolutely** (what the photo shows),
    which keeps the label invariant to the wearer's skin tone. Set
    ``wb_normalize=True`` to instead von-Kries-correct the lip to a canonical
    skin (eqn (1)); that buys white-balance invariance but makes the label
    depend on skin tone, so it is off by default. A proper fix for camera white
    balance is a neutral (teeth/sclera/grey-world) reference -- see README.

    Parameters
    ----------
    lip_rgb : (N, 3) array of sRGB lip (vermilion) pixels.
    skin_rgb : (M, 3) array of sRGB skin pixels, or None.
    """
    lip_rgb = np.asarray(lip_rgb, dtype=np.float64).reshape(-1, 3)
    have_skin = skin_rgb is not None and len(skin_rgb) > 0
    gain = np.ones(3)
    wb_done = False

    if have_skin:
        skin_rgb = np.asarray(skin_rgb, dtype=np.float64).reshape(-1, 3)
        skin_center_rgb = _robust_rgb_center(skin_rgb)
        if wb_normalize:
            gain = white_balance_gain(skin_center_rgb)
            lip_rgb = apply_gain(lip_rgb, gain)
            skin_center_rgb = apply_gain(skin_center_rgb.reshape(1, 3), gain)[0]
            wb_done = True
        skin_lab = rgb_to_lab(skin_center_rgb.reshape(1, 3))[0]
        sL_, sa_, sb_ = float(skin_lab[0]), float(skin_lab[1]), float(skin_lab[2])
        n_skin = int(len(skin_rgb))
    else:
        sL_ = sa_ = sb_ = float("nan")
        n_skin = 0

    # Measure lip colour (after any correction).
    lip_lab_all = rgb_to_lab(lip_rgb)
    mask = clean_lip_pixels(lip_lab_all)
    lip_lab = lip_lab_all[mask] if mask.sum() >= 8 else lip_lab_all
    center = _robust_center(lip_lab)
    L, a, b = (float(center[0]), float(center[1]), float(center[2]))
    C = float(np.hypot(a, b))
    h = float(np.degrees(np.arctan2(b, a)))

    if n_skin > 0:
        dL, da, db = L - sL_, a - sa_, b - sb_
        rel_C = float(np.hypot(da, db))
        rel_h = float(np.degrees(np.arctan2(db, da)))
    else:
        dL = da = db = rel_C = rel_h = float("nan")

    return LipColorFeatures(
        L=L, a=a, b=b, C=C, h=h,
        skin_L=sL_, skin_a=sa_, skin_b=sb_,
        dL=dL, da=da, db=db, rel_C=rel_C, rel_h=rel_h,
        n_lip_px=int(len(lip_lab)), n_skin_px=n_skin,
        wb_corrected=wb_done, wb_gain=tuple(float(g) for g in gain),
    )
