"""Generate D4-style transforms of N images and manifests, for the orientation
/ flip equivariance study. Transforms: identity, hflip, vflip, rot180 (all
dimension-preserving) and rot90cw, rot90ccw (dimension-swapping). Writes:
  <out>/imgs/<name>__<T>.png
  <out>/manifest_all.csv   (all transforms; for equivariance + per-frame time)
  <out>/manifest_land.csv  (identity only; landscape memory)
  <out>/manifest_port.csv  (rot90cw only; portrait memory -> O(width) asymmetry)

  python equiv_prep.py --src data/liu4k_bench/1080 --out data/liu4k_equiv --nimg 24
"""
import argparse, csv, glob, os
import cv2

TRANSFORMS = ["identity", "hflip", "vflip", "rot180", "rot90cw", "rot90ccw"]


def apply_t(img, t):
    if t == "identity": return img
    if t == "hflip":     return cv2.flip(img, 1)
    if t == "vflip":     return cv2.flip(img, 0)
    if t == "rot180":    return cv2.rotate(img, cv2.ROTATE_180)
    if t == "rot90cw":   return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if t == "rot90ccw":  return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nimg", type=int, default=24)
    args = ap.parse_args()
    imgs = sorted(glob.glob(os.path.join(args.src, "*.png")))[:args.nimg]
    imgdir = os.path.join(args.out, "imgs"); os.makedirs(imgdir, exist_ok=True)

    rows, fr = [], 0
    for p in imgs:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        name = os.path.splitext(os.path.basename(p))[0]
        for t in TRANSFORMS:
            gt = apply_t(g, t)
            fn = f"{name}__{t}.png"
            cv2.imwrite(os.path.join(imgdir, fn), gt)
            rows.append({"frame": fr, "image_path": fn, "base": name, "transform": t,
                         "W": gt.shape[1], "H": gt.shape[0]})
            fr += 1

    def write(path, sel):
        sub = [r for r in rows if sel(r)]
        # renumber frames 0..k so segs align per manifest
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "image_path", "base", "transform", "W", "H"])
            w.writeheader()
            for i, r in enumerate(sub):
                rr = dict(r); rr["frame"] = i; w.writerow(rr)

    write(os.path.join(args.out, "manifest_all.csv"), lambda r: True)
    write(os.path.join(args.out, "manifest_land.csv"), lambda r: r["transform"] == "identity")
    write(os.path.join(args.out, "manifest_port.csv"), lambda r: r["transform"] == "rot90cw")
    print(f"{len(imgs)} images x {len(TRANSFORMS)} transforms = {len(rows)} frames -> {args.out}")


if __name__ == "__main__":
    main()
