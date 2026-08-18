"""Arbitrary-angle rotation repeatability (companion to equiv_prep/analyze).

The dihedral study (equiv_prep.py / equiv_analyze.py) tests the transforms
that preserve SweepLSD's two-way quantization axes exactly. This companion
rotates the same photographs by angles that do NOT (15/30/45 deg, bilinear
resampling, same-size canvas), and scores repeatability with the identical
matcher (midpoint <= 6 px, angle <= 5 deg, length ratio >= 0.6) against each
detector's own identity detections from the dihedral run.

Border handling (identical for every detector): the reference set keeps only
identity segments >= --margin2 px inside the original frame whose forward-
rotated endpoints land >= --margin px inside the canvas (fully visible after
rotation); mapped-back detections within --margin2 px of the original frame
border are dropped (they are dominated by the artificial valid-region edge).

  python equiv_rot.py prep    --src data/liu4k_bench/1080 --out data/liu4k_rot [--nimg 123]
  # then export_segs --gt <out>/manifest_rot.csv --root <out>/imgs --out <out>/all_<det> --det <det>
  python equiv_rot.py analyze --rot data/liu4k_rot --eq data/liu4k_equiv123
"""
import argparse, csv, glob, math, os, statistics as st
from collections import defaultdict

ANGLES = [15.0, 30.0, 45.0]
DETS = ["sweep1", "elsed", "edlines", "lsd"]
LABEL = {"sweep1": "SweepLSD", "elsed": "ELSED", "edlines": "EDLines", "lsd": "LSD"}


def rot_matrix(W, H, angle_deg):
    """Forward map (identity coords -> rotated canvas), about the geometric
    center, same convention in prep and analyze."""
    import cv2
    return cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), angle_deg, 1.0)


def apply_m(M, x, y):
    return (M[0][0] * x + M[0][1] * y + M[0][2],
            M[1][0] * x + M[1][1] * y + M[1][2])


# ---- matcher: identical thresholds/features to equiv_analyze.recall ----

def feats(s):
    x0, y0, x1, y1 = s
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1),
            math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0,
            math.hypot(x1 - x0, y1 - y0))


def adiff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def recall(A, B, mid_th=6.0, ang_th=5.0, lr=0.6):
    if not A:
        return None
    Bf = [feats(s) for s in B]
    grid = defaultdict(list)
    for i, (mx, my, _, _) in enumerate(Bf):
        grid[(int(mx // mid_th), int(my // mid_th))].append(i)
    hit = 0
    for s in A:
        amx, amy, aang, aL = feats(s)
        cx, cy = int(amx // mid_th), int(amy // mid_th)
        ok = False
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for i in grid.get((gx, gy), ()):
                    bmx, bmy, bang, bL = Bf[i]
                    if bL >= 1 and math.hypot(amx - bmx, amy - bmy) <= mid_th \
                       and adiff(aang, bang) <= ang_th \
                       and min(aL, bL) / max(aL, bL) >= lr:
                        ok = True
                        break
                if ok: break
            if ok: break
        if ok: hit += 1
    return 100.0 * hit / len(A)


def one_to_one(A, B, mid_th=6.0, ang_th=5.0, lr=0.6):
    """Greedy one-to-one matching (closest midpoints first) with the same
    thresholds as recall(); returns the number of matched pairs. Used by the
    symmetric metrics: unlike recall(), a B segment can serve only one A."""
    Af = [feats(s) for s in A]
    Bf = [feats(s) for s in B]
    grid = defaultdict(list)
    for i, (mx, my, _, _) in enumerate(Bf):
        grid[(int(mx // mid_th), int(my // mid_th))].append(i)
    pairs = []
    for ai, (amx, amy, aang, aL) in enumerate(Af):
        cx, cy = int(amx // mid_th), int(amy // mid_th)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for bi in grid.get((gx, gy), ()):
                    bmx, bmy, bang, bL = Bf[bi]
                    d = math.hypot(amx - bmx, amy - bmy)
                    if d <= mid_th and adiff(aang, bang) <= ang_th \
                       and bL >= 1 and min(aL, bL) / max(aL, bL) >= lr:
                        pairs.append((d, ai, bi))
    pairs.sort(key=lambda t: t[0])
    ua, ub = set(), set()
    n = 0
    for _, ai, bi in pairs:
        if ai in ua or bi in ub:
            continue
        ua.add(ai)
        ub.add(bi)
        n += 1
    return n


def load_segs(path):
    segs = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            segs[int(r["frame"])].append((float(r["x0"]), float(r["y0"]),
                                          float(r["x1"]), float(r["y1"])))
    return segs


def cmd_prep(args):
    import cv2
    os.makedirs(os.path.join(args.out, "imgs"), exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(args.src, "*.png")))[:args.nimg]
    rows = []
    fr = 0
    for p in srcs:
        base = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        for a in args.angles:
            M = rot_matrix(W, H, a)
            out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=128)
            name = f"{base}__rot{a:g}.png"
            cv2.imwrite(os.path.join(args.out, "imgs", name), out)
            rows.append([fr, name, base, a, W, H])
            fr += 1
    with open(os.path.join(args.out, "manifest_rot.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "image_path", "base", "angle", "W", "H"])
        w.writerows(rows)
    print(f"{len(srcs)} images x {len(args.angles)} angles = {fr} frames -> {args.out}")


def cmd_analyze(args):
    # identity detections come from the dihedral run's all_<det> dirs
    idman = {}  # base -> identity frame id
    dims = {}
    with open(os.path.join(args.eq, "manifest_all.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["transform"] == "identity":
                idman[r["base"]] = int(r["frame"])
                dims[r["base"]] = (int(r["W"]), int(r["H"]))
    rot_rows = []
    with open(os.path.join(args.rot, "manifest_rot.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rot_rows.append((int(r["frame"]), r["base"], float(r["angle"]),
                             int(r["W"]), int(r["H"])))

    for det in DETS:
        idsegs = load_segs(os.path.join(args.eq, f"all_{det}", "segs.csv"))
        rotsegs = load_segs(os.path.join(args.rot, f"all_{det}", "segs.csv"))
        per_angle = defaultdict(list)
        per_angle_sym = defaultdict(list)
        for fr, base, ang, W, H in rot_rows:
            if base not in idman:
                continue
            M = rot_matrix(W, H, ang)
            Minv = [[M[1][1], -M[0][1],
                     M[0][1] * M[1][2] - M[1][1] * M[0][2]],
                    [-M[1][0], M[0][0],
                     M[1][0] * M[0][2] - M[0][0] * M[1][2]]]
            # (pure rotation: inverse = transpose of the linear part)
            m, m2 = args.margin, args.margin2
            A = []
            for s in idsegs.get(idman[base], []):
                x0, y0, x1, y1 = s
                if not (m2 <= x0 <= W - 1 - m2 and m2 <= y0 <= H - 1 - m2 and
                        m2 <= x1 <= W - 1 - m2 and m2 <= y1 <= H - 1 - m2):
                    continue
                fx0, fy0 = apply_m(M, x0, y0)
                fx1, fy1 = apply_m(M, x1, y1)
                if (m <= fx0 <= W - 1 - m and m <= fy0 <= H - 1 - m and
                        m <= fx1 <= W - 1 - m and m <= fy1 <= H - 1 - m):
                    A.append(s)
            B = []
            for s in rotsegs.get(fr, []):
                bx0, by0 = apply_m(Minv, s[0], s[1])
                bx1, by1 = apply_m(Minv, s[2], s[3])
                mx, my = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
                if m2 <= mx <= W - 1 - m2 and m2 <= my <= H - 1 - m2:
                    B.append((bx0, by0, bx1, by1))
            r = recall(A, B)
            if r is not None:
                per_angle[ang].append(r)
            # Symmetric one-to-one metrics: B mirror-guarded exactly like A
            # (endpoints >= margin inside the rotated canvas AND mapped-back
            # endpoints >= margin2 inside the original frame), so extra
            # segments invented after rotation are penalized (mapped-back
            # precision), not just missing ones (recall).
            Bs = []
            for s in rotsegs.get(fr, []):
                if not (m <= s[0] <= W - 1 - m and m <= s[1] <= H - 1 - m and
                        m <= s[2] <= W - 1 - m and m <= s[3] <= H - 1 - m):
                    continue
                bx0, by0 = apply_m(Minv, s[0], s[1])
                bx1, by1 = apply_m(Minv, s[2], s[3])
                if (m2 <= bx0 <= W - 1 - m2 and m2 <= by0 <= H - 1 - m2 and
                        m2 <= bx1 <= W - 1 - m2 and m2 <= by1 <= H - 1 - m2):
                    Bs.append((bx0, by0, bx1, by1))
            if A:
                nm = one_to_one(A, Bs) if Bs else 0
                prec = 100.0 * nm / len(Bs) if Bs else None
                rec1 = 100.0 * nm / len(A)
                f1 = 200.0 * nm / (len(A) + len(Bs)) if (A or Bs) else None
                per_angle_sym[ang].append((prec, rec1, f1))
        print(f"=== {LABEL[det]} ===")
        for ang in sorted(per_angle):
            v = per_angle[ang]
            print(f"  rot{ang:>4g}  median {st.median(v):5.1f}%   "
                  f"(n={len(v)} images)")
        pooled = [x for ang in per_angle for x in per_angle[ang]]
        print(f"  pooled median over all angles: {st.median(pooled):5.1f}%")
        print("  symmetric one-to-one (endpoint-guarded both ways):")
        for ang in sorted(per_angle_sym):
            v = per_angle_sym[ang]
            ps = [p for p, _, _ in v if p is not None]
            rs = [r for _, r, _ in v]
            fs = [f for _, _, f in v if f is not None]
            print(f"  rot{ang:>4g}  P {st.median(ps):5.1f}  R {st.median(rs):5.1f}"
                  f"  F1 {st.median(fs):5.1f}   (n={len(v)} images)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nimg", type=int, default=123)
    p.add_argument("--angles", type=lambda s: [float(x) for x in s.split(",")],
                   default=ANGLES)
    a = sub.add_parser("analyze")
    a.add_argument("--rot", required=True)
    a.add_argument("--eq", required=True)
    a.add_argument("--margin", type=float, default=10.0)
    a.add_argument("--margin2", type=float, default=6.0)
    args = ap.parse_args()
    if args.cmd == "prep":
        cmd_prep(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
