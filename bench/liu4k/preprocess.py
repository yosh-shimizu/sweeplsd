"""Preprocess LIU4K-v2 (CC0) images into a uniform multi-resolution benchmark set.

Protocol (see paper Section on speed; justified by SweepLSD's O(width) design):
  1. Load as 8-bit grayscale (the thesis baseline input).
  2. Rotate portrait -> landscape (90 deg CW) so WIDTH is uniform across the corpus
     -- SweepLSD's memory/time are O(width), so orientation must be controlled.
  3. Center-crop to 16:9, then Lanczos-downscale to each rung (never upscale).
Every output image at a given rung is exactly the target size, grayscale PNG.

  python preprocess.py --src E:\\dataset\\LIU4k-v2\\validation \\
      --out ..\\..\\..\\..\\data\\liu4k_bench [--rungs 360,720,1080,1440,2160] [--limit N]
"""
import argparse, glob, os
import cv2
import numpy as np
from PIL import Image

RUNGS = {"360": (640, 360), "720": (1280, 720), "1080": (1920, 1080),
         "1440": (2560, 1440), "2160": (3840, 2160)}
# "lanczos" uses Pillow, whose LANCZOS scales the kernel by the reduction factor
# (proper anti-aliasing on downscale). cv2.INTER_LANCZOS4 does NOT scale its 8x8
# kernel, so it under-anti-aliases large downscales -- use Pillow instead.
CV_INTERP = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC}


def downscale(img, tw, th, interp):
    if interp == "lanczos":
        return np.asarray(Image.fromarray(img).resize((tw, th), Image.LANCZOS))
    return cv2.resize(img, (tw, th), interpolation=CV_INTERP[interp])


def crop_to_aspect(img, aw, ah):
    h, w = img.shape[:2]
    target = aw / ah
    if w / h > target:                       # too wide -> trim width
        nw = int(round(h * target)); x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(round(w / target)); y0 = (h - nh) // 2   # too tall -> trim height
    return img[y0:y0 + nh, :]


def process(path, out_dir, rungs, interp="lanczos"):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("  SKIP (unreadable):", path); return {}
    h, w = img.shape[:2]
    if h > w:                                 # portrait -> landscape
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE); h, w = img.shape[:2]
    name = os.path.splitext(os.path.basename(path))[0]
    done = {}
    for tag in rungs:
        tw, th = RUNGS[tag]
        c = crop_to_aspect(img, tw, th)
        if c.shape[1] < tw or c.shape[0] < th:            # would upscale -> skip
            print(f"  skip {name} @ {tag} (crop {c.shape[1]}x{c.shape[0]} < {tw}x{th})")
            continue
        r = downscale(c, tw, th, interp)
        d = os.path.join(out_dir, tag); os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, name + ".png"), r)
        done[tag] = True
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="source dir (recursive) of images")
    ap.add_argument("--out", required=True, help="output root; writes <out>/<tag>/<name>.png")
    ap.add_argument("--rungs", default="360,720,1080,1440,2160")
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (0=all)")
    ap.add_argument("--interp", default="lanczos", choices=["lanczos", "area", "cubic"],
                    help="downscale filter (lanczos=Pillow proper anti-aliased, recommended)")
    args = ap.parse_args()
    rungs = [r for r in args.rungs.split(",") if r in RUNGS]
    files = sorted(glob.glob(os.path.join(args.src, "**", "*.png"), recursive=True) +
                   glob.glob(os.path.join(args.src, "**", "*.jpg"), recursive=True))
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} source images, rungs={rungs}")
    counts = {t: 0 for t in rungs}
    for i, f in enumerate(files):
        d = process(f, args.out, rungs, args.interp)
        for t in d:
            counts[t] += 1
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(files)}")
    print("per-rung output counts:", counts)


if __name__ == "__main__":
    main()
