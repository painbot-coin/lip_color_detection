"""Deterministic lip-colour classifier.

Maps CIELAB colour features to one of six groups:

    pale | pink-rose | red-coral | berry-mauve | brown | other

and, when the group is ``berry-mauve``, to a sub-shade:

    berry | plum | mauve

Why rules (not an LLM, not a black-box net)
-------------------------------------------
The same input always yields the same output, so the "borderline" cases that
make the VLM wobble become *stable* here: a borderline sample sits near a
threshold and gets a low confidence, but it always lands on the same side.
Thresholds are explicit and calibratable; see ``ClassifierConfig``.

The decisive axes (all in CIELAB)
---------------------------------
* **warm vs cool** -> separates the whole berry-mauve (purple) family from
  coral/pink/brown. The robust signal is ``db`` (lip b* minus skin b*): purple
  lips are *bluer than the subject's own skin* (db < 0); warm lips are yellower
  (db > 0). Falls back to absolute b* when no skin reference is available.
* **chroma C*** -> saturated (coral/berry) vs muted (pink/mauve/pale).
* **lightness L*** -> dark (brown/plum) vs light (pink/mauve).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .color import LipColorFeatures


GROUPS = ("pale", "pink-rose", "red-coral", "berry-mauve", "brown", "other")
SUBSHADES = ("berry", "plum", "mauve")


@dataclass
class ClassifierConfig:
    # --- validity / "other" -------------------------------------------------
    min_lip_px: int = 40          # fewer than this -> segmentation unreliable
    other_min_a: float = 4.0      # a* below this at real chroma = not a lip hue
    other_chroma: float = 12.0    # chroma above which an odd hue counts as odd
    other_hue_hi: float = 82.0    # hues outside [other_hue_lo, other_hue_hi]...
    other_hue_lo: float = -62.0   # ...with chroma are unnatural (green/blue/etc.)

    # --- pale / nude --------------------------------------------------------
    # Pale is judged ABSOLUTELY: light AND low-chroma. (An earlier version keyed
    # pale off chroma *relative to skin*, which made the same lip colour read as
    # pale on deep skin but not on pale skin -- i.e. it broke skin-tone
    # invariance. The absolute rule gives one answer regardless of skin tone.)
    pale_abs_C: float = 21.0      # chroma at/below this ...
    pale_min_L: float = 55.0      # ... and lightness at/above this -> pale
    pale_rel_C: float = -1.0      # legacy skin-relative trigger; <0 disables it

    # --- warm vs cool boundary ---------------------------------------------
    # The decision axis is the lip's b* (yellow<->blue) measured absolutely.
    # Classifying on absolute colour keeps the label invariant to the wearer's
    # skin tone (what the photo *shows* is what we label). The trade-off is
    # sensitivity to an uncorrected camera white balance; normal auto-WB (within
    # ~+/-8% per channel) is tolerated, heavy casts are the known weak point and
    # would need a neutral (teeth/sclera/grey-world) white reference to fix.
    cool_b_boundary: float = 3.5  # absolute b* below this -> cool/purple family

    # --- warm family internal splits ---------------------------------------
    brown_L: float = 44.0         # darker than this (and warm, muted) -> brown
    brown_C_max: float = 46.0     # brown is not highly saturated
    brown_a_max: float = 42.0     # brown is not strongly red
    coral_C: float = 38.0         # saturated warm red -> coral
    coral_h: float = 20.0         # oriented toward orange
    pink_L: float = 52.0          # light + less saturated warm -> pink-rose

    # --- berry-mauve sub-shade splits --------------------------------------
    plum_L: float = 40.0          # dark purple -> plum
    mauve_C: float = 26.0         # low-chroma purple -> mauve
    mauve_rel_C: float = 16.0     # or low chroma relative to skin

    # confidence scaling (CIELAB units over which a decision goes 0->1 sure)
    margin_scale: float = 6.0


@dataclass
class LipColorResult:
    group: str
    subshade: Optional[str]       # only for berry-mauve, else None
    confidence: float             # 0..1, from distance to nearest boundary
    rationale: str
    features: dict                # the LipColorFeatures as a dict

    def __str__(self) -> str:
        sub = f" ({self.subshade})" if self.subshade else ""
        return f"{self.group}{sub}  conf={self.confidence:.2f}  - {self.rationale}"


def _sat_conf(margin: float, scale: float) -> float:
    """Map a signed CIELAB margin to a 0.5..1 confidence via a soft ramp."""
    return float(0.5 + 0.5 * math.tanh(abs(margin) / max(scale, 1e-6)))


def _subshade(f: LipColorFeatures, cfg: ClassifierConfig) -> tuple[str, str]:
    """Pick berry / plum / mauve and give a one-line reason."""
    low_chroma = (f.C < cfg.mauve_C) or (
        not math.isnan(f.rel_C) and f.rel_C < cfg.mauve_rel_C
    )
    if f.L < cfg.plum_L and not low_chroma:
        return "plum", f"deep & purple (L*={f.L:.0f}<{cfg.plum_L:.0f})"
    if low_chroma:
        return "mauve", f"soft/greyish purple (C*={f.C:.0f})"
    return "berry", f"bright purple-red (L*={f.L:.0f}, C*={f.C:.0f})"


def classify(f: LipColorFeatures, cfg: ClassifierConfig | None = None) -> LipColorResult:
    cfg = cfg or ClassifierConfig()
    has_skin = f.n_skin_px > 0 and not math.isnan(f.db)
    feats = f.as_dict()

    def result(group, sub, conf, why):
        return LipColorResult(group, sub, round(float(conf), 3), why, feats)

    # ---- 0. validity / unnatural ("other") --------------------------------
    if f.n_lip_px < cfg.min_lip_px:
        return result("other", None, 0.2,
                      f"only {f.n_lip_px} usable lip px (<{cfg.min_lip_px}); unreliable")
    if f.C > cfg.other_chroma and (f.h > cfg.other_hue_hi or f.h < cfg.other_hue_lo):
        return result("other", None, 0.7,
                      f"hue {f.h:.0f}° outside natural lip range at C*={f.C:.0f}")
    if f.a < cfg.other_min_a and f.C > cfg.other_chroma:
        return result("other", None, 0.6,
                      f"not reddish (a*={f.a:.0f}) yet chromatic; non-lip colour")

    # ---- 1. pale / nude (absolute: light AND low-chroma) ------------------
    rel_small = has_skin and cfg.pale_rel_C >= 0 and f.rel_C < cfg.pale_rel_C
    if (f.C <= cfg.pale_abs_C and f.L >= cfg.pale_min_L) or rel_small:
        margin = min(cfg.pale_abs_C - f.C, f.L - cfg.pale_min_L)
        return result("pale", None, _sat_conf(margin, cfg.margin_scale),
                      f"light & low-chroma (L*={f.L:.0f}>={cfg.pale_min_L:.0f}, "
                      f"C*={f.C:.0f}<={cfg.pale_abs_C:.0f})")

    # ---- 2. warm vs cool (purple) -----------------------------------------
    axis_val = f.b
    axis = f"b*={f.b:+.1f}" + (" (WB-corrected)" if f.wb_corrected else "")
    warm = axis_val >= cfg.cool_b_boundary
    warm_margin = axis_val - cfg.cool_b_boundary
    warm_conf = _sat_conf(warm_margin, cfg.margin_scale)

    # ---- 3a. cool / purple => berry-mauve ---------------------------------
    if not warm:
        sub, sub_why = _subshade(f, cfg)
        return result("berry-mauve", sub, warm_conf,
                      f"blue-leaning / purple-toned [{axis}]; {sub_why}")

    # ---- 3b. warm => brown / coral / pink-rose ----------------------------
    # brown: dark, warm, muted, not strongly red.
    if f.L < cfg.brown_L and f.C < cfg.brown_C_max and f.a < cfg.brown_a_max:
        m = min(cfg.brown_L - f.L, cfg.brown_C_max - f.C)
        return result("brown", None, _sat_conf(m, cfg.margin_scale),
                      f"dark, warm & muted (L*={f.L:.0f}, C*={f.C:.0f}) [{axis}]")

    # red-coral: saturated warm red oriented to orange.
    if f.C >= cfg.coral_C and f.h >= cfg.coral_h:
        m = min(f.C - cfg.coral_C, f.h - cfg.coral_h)
        conf = min(warm_conf, _sat_conf(m, cfg.margin_scale))
        return result("red-coral", None, conf,
                      f"saturated warm red (C*={f.C:.0f}, h={f.h:.0f}°) [{axis}]")

    # pink-rose: lighter / less saturated warm.
    if f.L >= cfg.pink_L or f.C < cfg.coral_C:
        return result("pink-rose", None, warm_conf,
                      f"light, soft warm pink (L*={f.L:.0f}, C*={f.C:.0f}) [{axis}]")

    # warm fallback: dark+saturated but not orange enough -> treat as coral.
    return result("red-coral", None, 0.5 * warm_conf,
                  f"warm & saturated, ambiguous orientation (h={f.h:.0f}°) [{axis}]")
