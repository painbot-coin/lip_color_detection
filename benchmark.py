"""Honest out-of-sample robustness benchmark for the colour engine.

No images or labels to download: there is **no public dataset** labelled with
these six bespoke lip-colour groups (that is why the task exists). So we build a
controlled test whose ground truth we *know*:

  * a pool of in-category colours chosen to be clearly in each group and
    DIFFERENT from any colour used to set the thresholds (so it is genuinely
    out-of-sample for the decision surface);
  * each rendered many times under randomised **skin tone**, **camera white
    balance**, and **sensor noise** -- the three nuisance factors the task names.

We report group accuracy overall and per group, at several white-balance levels.
This measures the thing that matters versus the VLM: does a known lip colour get
the right, stable label under real-world capture variation?

    python benchmark.py
"""
import numpy as np

from lipcolor.color import extract_features
from lipcolor.classify import classify

# In-category colours (sRGB), intentionally distinct from the unit-test swatches.
POOL = {
    "pale":        [(210, 170, 165), (220, 185, 178), (198, 160, 155),
                    (230, 195, 188), (205, 172, 168)],
    "pink-rose":   [(220, 130, 150), (200, 105, 125), (225, 140, 160),
                    (210, 120, 142), (235, 150, 168)],
    "red-coral":   [(210, 65, 50), (195, 55, 45), (220, 80, 60),
                    (200, 48, 40), (185, 70, 55), (230, 90, 70)],
    "berry-mauve": [(170, 40, 90), (185, 45, 100), (85, 38, 70), (100, 45, 82),
                    (175, 130, 152), (160, 120, 140), (185, 140, 160)],
    "brown":       [(115, 70, 55), (100, 60, 48), (130, 82, 65),
                    (108, 66, 52), (95, 58, 46)],
    "other":       [(60, 120, 90), (80, 90, 180), (150, 150, 40),
                    (120, 200, 210), (90, 70, 160)],
}
SKINS = [(236, 200, 182), (220, 185, 165), (198, 150, 126),
         (170, 120, 95), (140, 100, 80), (120, 86, 70)]
N_PER_COLOR = 30


def _sw(rgb, n=350):
    return np.tile(np.array(rgb, np.uint8), (n, 1))


def main():
    rng = np.random.default_rng(0)  # fixed seed -> reproducible

    def jitter(rgb, gain, sigma):
        px = _sw(rgb).astype(np.float32) * np.array(gain, np.float32)
        px = px + rng.normal(0, sigma, px.shape)
        return np.clip(px, 0, 255).astype(np.uint8)

    def run(wb_amp):
        tot = ok = 0
        per = {g: [0, 0] for g in POOL}
        for g, colors in POOL.items():
            for rgb in colors:
                for _ in range(N_PER_COLOR):
                    gain = rng.uniform(1 - wb_amp, 1 + wb_amp, 3)
                    sigma = rng.uniform(2, 7)
                    skin = SKINS[rng.integers(len(SKINS))]
                    f = extract_features(jitter(rgb, gain, sigma),
                                         jitter(skin, gain, sigma))
                    good = classify(f).group == g
                    tot += 1; ok += good
                    per[g][0] += good; per[g][1] += 1
        return ok / tot, {g: per[g][0] / per[g][1] for g in per}

    print("Out-of-sample accuracy (fresh colours x random skin tone / WB / noise)\n")
    print(f"{'white balance':>16s}  {'overall':>8s}   per-group")
    print("-" * 72)
    for amp, lbl in [(0.0, "none"), (0.05, "+/-5% (typical)"),
                     (0.08, "+/-8%"), (0.15, "+/-15% (bad)")]:
        acc, per = run(amp)
        pg = "  ".join(f"{g[:4]}={per[g]:.0%}" for g in POOL)
        print(f"{lbl:>16s}  {acc:>8.1%}   {pg}")
    print("-" * 72)
    print("Note: pink-rose & berry-mauve are the task's acknowledged hard edges;")
    print("their numbers reflect genuine boundary ambiguity and UNFITTED thresholds.")
    print("Supplying real labelled images would let these thresholds be calibrated.")


if __name__ == "__main__":
    main()
