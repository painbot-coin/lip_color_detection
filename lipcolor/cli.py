"""Command-line entry point.

    python -m lipcolor.cli path/to/face.jpg [more.jpg ...] [--json] [--debug DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

from .pipeline import LipColorPipeline


def _overlay(path: str, seg, out_dir: str):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None or seg.lip_mask is None:
        return
    ov = bgr.copy()
    ov[seg.lip_mask > 0] = (0, 255, 0)
    ov[seg.skin_mask > 0] = (255, 0, 0)
    blended = cv2.addWeighted(bgr, 0.6, ov, 0.4, 0)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{base}_seg.png"), blended)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure and classify lip colour.")
    ap.add_argument("images", nargs="+", help="face image path(s)")
    ap.add_argument("--json", action="store_true", help="emit JSON lines")
    ap.add_argument("--debug", metavar="DIR", help="write segmentation overlays")
    args = ap.parse_args(argv)

    pipe = LipColorPipeline()
    try:
        for path in args.images:
            try:
                rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                seg = pipe.face.segment(rgb)
                from .color import extract_features
                from .classify import classify
                if seg.ok:
                    res = classify(extract_features(seg.lip_rgb, seg.skin_rgb), pipe.cfg)
                else:
                    from .pipeline import LipColorPipeline as _  # noqa
                    res = pipe.analyze(path)
                if args.debug:
                    _overlay(path, seg, args.debug)
            except Exception as e:  # pragma: no cover
                if args.json:
                    print(json.dumps({"image": path, "error": str(e)}))
                else:
                    print(f"{path}\tERROR\t{e}")
                continue

            if args.json:
                out = {"image": path, "group": res.group, "subshade": res.subshade,
                       "confidence": res.confidence, "rationale": res.rationale,
                       "features": res.features}
                print(json.dumps(out))
            else:
                sub = f"/{res.subshade}" if res.subshade else ""
                print(f"{path}\t{res.group}{sub}\tconf={res.confidence:.2f}\t{res.rationale}")
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
