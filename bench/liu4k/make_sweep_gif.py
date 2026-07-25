"""Render the README 'one raster sweep' animation: a scan line moves down the
photo and each segment appears the moment the sweep passes its last row
(SweepLSD emits it there). Grayscale photo, rainbow-coloured segments.

  python make_sweep_gif.py --img IMGP1077.png --segs segs.csv --out sweep.gif
"""
import argparse, colorsys, csv
import cv2, numpy as np
from PIL import Image


def seg_color(i):                       # golden-ratio hue spread -> distinct colours
    h = (i * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.82, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))   # BGR


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--img", required=True)
    ap.add_argument("--segs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--duration", type=int, default=70, help="ms per frame")
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--preview", default=None, help="also save a mid-sweep still here")
    args = ap.parse_args()

    g = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    H0, W0 = g.shape
    W = args.width; H = round(H0 * W / W0)
    base = cv2.cvtColor(cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    dim = (base.astype(np.float32) * 0.55 + 6).clip(0, 255).astype(np.uint8)   # photo, dimmed a touch
    s = W / W0

    segs = []
    with open(args.segs, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            x0, y0, x1, y1 = (float(r["x0"]) * s, float(r["y0"]) * s,
                              float(r["x1"]) * s, float(r["y1"]) * s)
            segs.append((int(x0), int(y0), int(x1), int(y1), max(y0, y1), seg_color(i)))

    frames = []
    for fi in range(args.frames + 1):
        Y = H * fi / args.frames
        c = dim.copy()
        for x0, y0, x1, y1, ey, col in segs:
            if ey <= Y:
                cv2.line(c, (x0, y0), (x1, y1), col, 1, cv2.LINE_AA)
        if fi < args.frames:                                   # bright scan line
            yy = int(min(Y, H - 1))
            cv2.line(c, (0, yy), (W, yy), (235, 235, 130), 1, cv2.LINE_AA)
            if yy + 1 < H:
                cv2.line(c, (0, yy + 1), (W, yy + 1), (120, 120, 70), 1, cv2.LINE_AA)
        if args.preview and fi == int(args.frames * 0.62):
            cv2.imwrite(args.preview, c)
        frames.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                      .convert("P", palette=Image.ADAPTIVE, colors=args.colors))

    durations = [args.duration] * len(frames)
    durations[-1] = 1500                                        # hold the finished frame
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=1)
    print(f"wrote {args.out}  ({W}x{H}, {len(segs)} segments, {args.frames+1} sweep frames)")


if __name__ == "__main__":
    main()
