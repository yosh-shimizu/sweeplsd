#!/usr/bin/env python3
"""Render the 3-level comparison (pure SW / HLS-C model / RTL sim) over a corpus
with an identical drawing style, so detection differences are visible and
rendering differences are not.

Inputs per image <name> (directories are configurable, see below):
  SW : <SW_DIR>/<name>.txt                 x0 y0 x1 y1 (float, finalized by detect())
  HLS: <HLS_DIR>/<name>_imp_records.hex    18 hex fields (records; finalized here)
  RTL: <RTL_DIR>/<name>_recs_out.txt       18 dec fields (records; finalized here)

Record finalization is a line-for-line port of hls/host/finalize.hpp with the
hardware-mode flags (endpoint_from_bbox + lattice_half_shift), so all three
columns show the same algorithm stage.

Configuration note: the SW column includes sub-pixel NMS (part of the shipped
`Params{}`); the HLS-C/RTL columns run the hardware configuration, which omits
it. That accounts for ALL SW-vs-HLS count differences: with subpixel_nms=false,
detect() matched the HLS-C counts on every image of the 150-photo Full-HD test
corpus used at the time (verified 2026-07-13). The RTL column is simulated with
real 1080p30 event-FIFO timing, so a saturated dense frame may shed segments.

Outputs: <OUT_DIR>/<name>_{sw,hlsc,rtl}.png, summary.csv, index.html

Source photographs are never redistributed with this repository. Point
SWEEPLSD_DATASET at a directory of Full-HD grayscale PNGs — e.g. the 1080p rung
produced by bench/liu4k/preprocess.py — and the three input directories at
whatever produced them:

  SWEEPLSD_DATASET   image directory                  (required)
  SWEEPLSD_GLOB      filename pattern within it       (default: *.png)
  SWEEPLSD_SW_DIR    SW segment dumps                 (default: <root>/.compare150/sw)
  SWEEPLSD_HLS_DIR   HLS-C record dumps               (default: <root>/rtl/tb/vprof)
  SWEEPLSD_RTL_DIR   RTL record dumps                 (default: <root>/rtl/tb/rtl150)
  SWEEPLSD_OUT_DIR   render output                    (default: <root>/.compare150/render)
"""
import csv
import math
import os
import sys
from glob import glob

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.environ.get("SWEEPLSD_DATASET")
GLOB_PAT = os.environ.get("SWEEPLSD_GLOB", "*.png")
SW_DIR = os.environ.get("SWEEPLSD_SW_DIR", os.path.join(ROOT, ".compare150", "sw"))
HLS_DIR = os.environ.get("SWEEPLSD_HLS_DIR", os.path.join(ROOT, "rtl", "tb", "vprof"))
RTL_DIR = os.environ.get("SWEEPLSD_RTL_DIR", os.path.join(ROOT, "rtl", "tb", "rtl150"))
OUT_DIR = os.environ.get("SWEEPLSD_OUT_DIR", os.path.join(ROOT, ".compare150", "render"))

COLOR = (0, 255, 0)
WIDTH = 2


def finalize(rec, endpoint_from_bbox=True, lattice_half_shift=True):
    (sx, sy, ex, ey, n, xs, ys, xss, yss, xys,
     mnx, mnxy, mxx, mxxy, mny, mnyx, mxy, mxyx) = rec
    W = float(n)
    mux, muy = xs / W, ys / W
    ma = xss * W - xs * xs
    mb = xys * W - xs * ys
    mc = yss * W - ys * ys
    theta = 0.5 * math.atan2(2.0 * mb, ma - mc)
    dx, dy = math.cos(theta), math.sin(theta)

    p0, p1 = (float(sx), float(sy)), (float(ex), float(ey))
    if endpoint_from_bbox:
        cand = [(float(sx), float(sy)), (float(ex), float(ey)),
                (float(mnx), float(mnxy)), (float(mxx), float(mxxy)),
                (float(mnyx), float(mny)), (float(mxyx), float(mxy))]
        tmin, tmax = float("inf"), float("-inf")
        for c in cand:
            t = (c[0] - mux) * dx + (c[1] - muy) * dy
            if t < tmin:
                tmin, p0 = t, c
            if t > tmax:
                tmax, p1 = t, c

    def proj(p):
        t = (p[0] - mux) * dx + (p[1] - muy) * dy
        return (mux + t * dx, muy + t * dy)

    (x0, y0), (x1, y1) = proj(p0), proj(p1)
    if lattice_half_shift:
        x0, y0, x1, y1 = x0 + 0.5, y0 + 0.5, x1 + 0.5, y1 + 0.5
    return (x0, y0, x1, y1)


def load_sw(name):
    path = os.path.join(SW_DIR, name + ".txt")
    if not os.path.exists(path):
        return None
    segs = []
    for ln in open(path):
        p = ln.split()
        if len(p) == 4:
            segs.append(tuple(float(v) for v in p))
    return segs


def load_records(path, base):
    if not os.path.exists(path):
        return None
    segs = []
    for ln in open(path):
        p = ln.split()
        if len(p) != 18:
            continue
        rec = [int(v, base) for v in p]
        segs.append(finalize(rec))
    return segs


def render(name, segs, tag, bg):
    if os.path.exists(os.path.join(OUT_DIR, "%s_%s.png" % (name, tag))):
        return
    img = bg.copy()
    d = ImageDraw.Draw(img)
    for (x0, y0, x1, y1) in segs:
        d.line([(x0, y0), (x1, y1)], fill=COLOR, width=WIDTH)
    img.save(os.path.join(OUT_DIR, "%s_%s.png" % (name, tag)), optimize=True)


def main():
    if not DATASET:
        sys.exit("error: set SWEEPLSD_DATASET to the image directory "
                 "(e.g. data/liu4k_bench/1080). See the module docstring.")
    if not os.path.isdir(DATASET):
        sys.exit("error: SWEEPLSD_DATASET is not a directory: %s" % DATASET)
    os.makedirs(OUT_DIR, exist_ok=True)
    # name -> source path, so the glob may carry any extension.
    srcs = {os.path.splitext(os.path.basename(f))[0]: f
            for f in sorted(glob(os.path.join(DATASET, GLOB_PAT)))}
    if not srcs:
        sys.exit("error: no images matching %s in %s" % (GLOB_PAT, DATASET))
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    rows = []
    for name in sorted(srcs):
        if only and name not in only:
            continue
        sw = load_sw(name)
        hls = load_records(os.path.join(HLS_DIR, name + "_imp_records.hex"), 16)
        rtl = load_records(os.path.join(RTL_DIR, name + "_recs_out.txt"), 10)
        if sw is None and hls is None and rtl is None:
            continue
        bg = Image.open(srcs[name]).convert("L").convert("RGB")
        for segs, tag in ((sw, "sw"), (hls, "hlsc"), (rtl, "rtl")):
            if segs is not None:
                render(name, segs, tag, bg)
        rows.append((name,
                     len(sw) if sw is not None else "",
                     len(hls) if hls is not None else "",
                     len(rtl) if rtl is not None else ""))
        print(name, rows[-1][1:])

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "sw_segments", "hlsc_segments", "rtl_segments"])
        w.writerows(rows)

    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>SW / HLS-C / RTL comparison</title>\n"
                "<style>body{font-family:sans-serif;background:#111;color:#eee}"
                ".row{margin:24px 0}.imgs{display:flex;gap:4px}"
                ".imgs figure{flex:1;margin:0}.imgs img{width:100%;height:auto}"
                "figcaption{font-size:12px;text-align:center;padding:2px}</style>\n"
                "<h1>SweepLSD: pure SW / HLS-C model / RTL sim (identical rendering)</h1>\n"
                "<p style='max-width:72em;color:#bbb'><b>Configurations.</b> "
                "The SW column is <code>detect()</code> with the shipped "
                "<code>Params{}</code>, which includes sub-pixel NMS; the HLS-C and RTL "
                "columns run the hardware configuration, which omits sub-pixel NMS (kept "
                "SW/HLS-only — it does not fit the RTL judge's 128-bit envelope). The small "
                "SW-vs-HLS count differences are exactly this configuration delta: with "
                "sub-pixel NMS disabled, <code>detect()</code> matched the HLS-C counts on "
                "every image of the 150-photo Full-HD test corpus used at the time "
                "(verified 2026-07-13). The RTL column is simulated with real 1080p30 "
                "event-FIFO timing, so a saturated dense frame may shed segments.</p>\n")
        for (name, a, b, c) in rows:
            f.write("<div class='row'><h3>%s &nbsp; <small>SW %s / HLS-C %s / RTL %s</small></h3>"
                    "<div class='imgs'>" % (name, a, b, c))
            for tag, label in (("sw", "pure SW"), ("hlsc", "HLS-C model"), ("rtl", "RTL sim")):
                f.write("<figure><img loading='lazy' src='%s_%s.png'>"
                        "<figcaption>%s</figcaption></figure>" % (name, tag, label))
            f.write("</div></div>\n")

    print("rendered %d images -> %s" % (len(rows), OUT_DIR))


if __name__ == "__main__":
    main()
