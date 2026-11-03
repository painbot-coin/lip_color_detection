"""Synthetic-swatch tests for the colour measurement + classifier.

These do NOT need the face model or any image file: they feed known sRGB
swatches straight into extract_features/classify, which is where the decision
logic lives. Run with `pytest` or directly: `python tests/test_classify.py`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lipcolor.color import extract_features          # noqa: E402
from lipcolor.classify import classify               # noqa: E402

SKIN = (208, 170, 150)

# name -> (lip sRGB, expected group, expected subshade or None)
CASES = {
    "pale":      ((205, 165, 158), "pale", None),
    "pink-rose": ((212, 118, 138), "pink-rose", None),
    "red-coral": ((205, 70, 58),   "red-coral", None),
    "berry":     ((176, 36, 96),   "berry-mauve", "berry"),
    "plum":      ((92, 40, 74),    "berry-mauve", "plum"),
    "mauve":     ((176, 128, 150), "berry-mauve", "mauve"),
    "brown":     ((120, 66, 52),   "brown", None),
    "other":     ((70, 90, 170),   "other", None),
}

def _swatch(rgb, n=400):
    return np.tile(np.array(rgb, np.uint8), (n, 1))


def _apply_wb(rgb, gain):
    return tuple(int(np.clip(c * g, 0, 255)) for c, g in zip(rgb, gain))


def test_groups_neutral():
    for name, (rgb, grp, _sub) in CASES.items():
        res = classify(extract_features(_swatch(rgb), _swatch(SKIN)))
        assert res.group == grp, f"{name}: {res.group} != {grp} ({res})"


def test_subshades_neutral():
    for name, (rgb, grp, sub) in CASES.items():
        if grp != "berry-mauve":
            continue
        res = classify(extract_features(_swatch(rgb), _swatch(SKIN)))
        assert res.subshade == sub, f"{name}: subshade {res.subshade} != {sub}"


def test_skin_tone_invariance():
    """Same lip colour shown on different skin tones -> same group.

    This is the property the task actually wants ("go by what the photo shows",
    stable "across skin tones"): a given vermilion colour must get one label no
    matter the wearer's skin. We render the SAME absolute lip colour against
    pale / medium / deep skin and require a single label.
    """
    skins = [(236, 200, 182), (198, 150, 126), (120, 86, 70)]
    for name, (rgb, grp, _sub) in CASES.items():
        labels = {classify(extract_features(_swatch(rgb), _swatch(sk))).group
                  for sk in skins}
        assert labels == {grp}, f"{name}: skin-tone-unstable, got {labels}, want {grp}"


def test_white_balance_tolerance():
    """Saturated categories stay stable under mild white-balance shifts.

    Absolute-colour classification is skin-tone invariant but only *tolerant* to
    white balance (heavy casts are a known limitation). The high-chroma groups,
    which sit far from the warm/cool boundary, must survive a mild +/-8% cast.
    """
    mild = {"neutral": (1.0, 1.0, 1.0), "warm": (1.08, 1.0, 0.94),
            "cool": (0.94, 1.0, 1.08)}
    robust = {"red-coral", "brown"}
    for name, (rgb, grp, _sub) in CASES.items():
        if grp not in robust:
            continue
        labels = {classify(extract_features(_swatch(_apply_wb(rgb, g)),
                                            _swatch(_apply_wb(SKIN, g)))).group
                  for g in mild.values()}
        assert labels == {grp}, f"{name}: WB-unstable under mild cast, got {labels}"


def test_determinism():
    """Identical input -> byte-identical output (the whole point vs the VLM)."""
    a = classify(extract_features(_swatch((176, 36, 96)), _swatch(SKIN)))
    b = classify(extract_features(_swatch((176, 36, 96)), _swatch(SKIN)))
    assert (a.group, a.subshade, a.confidence) == (b.group, b.subshade, b.confidence)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
