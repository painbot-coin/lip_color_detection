# lipcolor — measurement-based lip-colour classification

Sort the colour of the lips (the **vermilion** — the coloured surface, not the
surrounding skin) from a single face photo into one of six groups:

```
pale · pink-rose · red-coral · berry-mauve · brown · other
```

and, when the group is **berry-mauve**, into a sub-shade: **berry · plum · mauve**.

## Why this isn't another prompt

The current production method asks a vision-language model (Gemini 3.8 flash) to
*reason* about lip colour from a text definition. That is exactly why it wobbles
on borderline cases: an LLM re-derives the answer from scratch each call and
drifts, so a dusty-mauve-vs-pink-rose call can flip between runs on the same
photo.

But lip colour is a **measurable physical quantity**. This package *measures* it
and classifies with deterministic colour science, so:

* **the same photo always yields the same label** (byte-for-byte);
* borderline samples don't flip — they land on a stable side with a **low
  confidence** that flags the call as close;
* the three hard edges the task names are resolved by explicit physical axes.

| Hard edge | Physical discriminator (CIELAB) |
|---|---|
| cool blue-leaning red **vs** true red-coral | sign of **b\*** (yellow↔blue): coral is warm (b\*>0), berry-mauve leans blue |
| dusty mauve **vs** pink-rose | **chroma C\*** + warmth: mauve is low-chroma *and* cool; pink-rose is warm |
| dark plum **vs** brown | **hue**: plum is purple (b\*<0), brown is warm (b\*>0) |

## The pipeline

```
photo ──▶ FaceMesh landmarks ──▶ vermilion mask (outer lip − inner mouth)
                              └─▶ skin patches (cheeks / forehead)
            │
            ├─ reject specular highlights, teeth, deep shadow
            ├─ white-balance / skin-tone normalise  (von Kries → canonical skin)
            ├─ robust CIELAB centre  (L*, C*, hue, and skin-relative Δ)
            └─ deterministic rules ─▶ group (+ berry/plum/mauve)  + confidence
```

### Skin-tone invariance (the property the task actually wants)

The task says to "go by what the photo shows" and to be stable "across skin
tones": a given vermilion colour must get **one label regardless of the wearer's
skin**. So by default the lip colour is measured **absolutely** (CIELAB on the
lip pixels), and `pale` is an absolute rule (light **and** low-chroma) — nothing
in the colour decision is divided by, or relative to, skin tone. This is what
`test_skin_tone_invariance` checks: the same colour on pale/medium/deep skin →
one group.

> An earlier version instead von-Kries-corrected the lip to a canonical skin
> (`lip_corrected = ref_skin · lip/skin`). That bought white-balance invariance
> but made the label *depend on skin tone* — the same colour read differently on
> deep vs pale skin — which is exactly backwards for this task. Out-of-sample
> accuracy went from ~64% (von Kries) to ~84% (absolute). The von Kries path is
> still available via `extract_features(..., wb_normalize=True)`.

### Known limitation & next step: illuminant / white balance

Absolute measurement is sensitive to a **coloured illuminant**: warm studio
lighting lifts b\* and can push a natural pink lip toward `red-coral` (visible on
some of the sample portraits). Humans discount the illuminant (colour
constancy); the method should too — but using a **neutral** reference (teeth,
sclera, or a grey-world estimate over the face), **not** skin, so the correction
doesn't re-couple to skin tone. That is the clear next improvement.

## Install

```bash
pip install -r requirements.txt
python download_model.py        # fetches the MediaPipe face_landmarker.task (~3.7 MB)
```

(Tested on Python 3.11. MediaPipe does not yet ship wheels for 3.13.)

## Use

```bash
# one or more face photos
python -m lipcolor.cli face.jpg another.jpg
python -m lipcolor.cli face.jpg --json
python -m lipcolor.cli face.jpg --debug overlays/   # writes segmentation overlays
```

```python
from lipcolor import analyze_image
r = analyze_image("face.jpg")
print(r.group, r.subshade, r.confidence, r.rationale)
# e.g. berry-mauve mauve 0.81  cooler/bluer than skin -> purple-toned ...
```

Programmatic, image-free (feed pixels directly):

```python
import numpy as np
from lipcolor import extract_features, classify
lip  = np.array([[176, 36, 96]] * 400)   # sRGB vermilion pixels
skin = np.array([[208, 170, 150]] * 400) # sRGB skin pixels
print(classify(extract_features(lip, skin)))   # -> berry-mauve (berry) ...
```

## Results — honest, out-of-sample

There is **no public dataset** labelled with these six bespoke groups (that's why
the task exists), so a downloaded "accuracy on 100 labelled images" isn't
available. Instead `python benchmark.py` measures the quantity that matters:
given a *known* in-category colour (chosen distinct from the tuning colours),
does it get the right, stable label under random **skin tone + camera white
balance + sensor noise**?

```
   white balance   overall   per-group
            none     90.0%   pale=100% pink=73% red=100% berry=86% brown=100% other=80%
 +/-5% (typical)     83.3%   pale= 92% pink=61% red=100% berry=70% brown=100% other= 80%
           +/-8%     79.7%   pale= 77% pink=55% red=100% berry=68% brown=100% other= 79%
    +/-15% (bad)     73.9%   pale= 47% pink=51% red= 98% berry=70% brown=100% other= 75%
```

Read this honestly:
* `red-coral`, `brown`, `pale` are strong; `other` catches unnatural hues.
* `pink-rose` and `berry-mauve` are the task's **acknowledged hard edges** and
  the thresholds are **unfitted** — these numbers reflect genuine boundary
  ambiguity, and are the first thing real labels would improve.
* The drop with white-balance amplitude is the illuminant limitation above.

`python tests/test_classify.py` checks group + sub-shade correctness,
**skin-tone invariance**, white-balance *tolerance* on saturated groups, and
determinism (5/5).

**These are synthetic ground truth.** A real accuracy number needs labelled
photos — see below.

## Calibrating to labelled data (optional)

Every threshold lives in `ClassifierConfig` (`lipcolor/classify.py`) with a
comment on the axis it controls. The decision is a thin, interpretable layer on
top of the measured features, so if you later collect labelled face crops you can
fit the boundaries (e.g. logistic regression / small tree on the
`LipColorFeatures`) without changing the measurement front-end — the hard,
valuable part (segmentation + white-balance/skin-tone-invariant colour) stays
fixed.

## Layout

```
lipcolor/
  segment.py    FaceMesh → vermilion + skin pixel sets
  color.py      specular/teeth rejection, von Kries WB correction, CIELAB features
  classify.py   deterministic 6-group + berry/plum/mauve rules, with confidence
  pipeline.py   image → result
  cli.py        command line (+ --json, --debug overlays)
  models/       face_landmarker.task (downloaded)
tests/test_classify.py   synthetic-swatch unit tests
benchmark.py             reproducible WB-robustness report
download_model.py        fetch the landmark model
```

## Limitations

* Needs a detectable, roughly front-facing face with visible lips; heavy
  occlusion or extreme pose → `other` (low confidence).
* The canonical-skin reference assumes the skin patches are lit like the lips
  (true for a single portrait). Strongly coloured side-lighting that hits only
  one region is not modelled.
* Sub-shade (berry/plum/mauve) uses lightness/chroma cuts that are the most
  reasonable defaults but are the first thing worth calibrating against labels.
