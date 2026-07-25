"""Analyze the orientation/flip equivariance experiment.

For each detector and transform T, map the detections on T(I) back to identity
coordinates and measure how well they reproduce the detections on I itself
(equivariance recall: fraction of identity segments with a matching transformed-
back segment within position+angle+length tolerance). Also report median detect
time and segment count per transform (from export_segs times.csv).

  python equiv_analyze.py --eq data/liu4k_equiv
"""
import argparse, csv, math, os, statistics as st
from collections import defaultdict

DETS = ["sweep1", "elsed", "edlines", "lsd"]
LABEL = {"sweep1": "SweepLSD", "elsed": "ELSED", "edlines": "EDLines", "lsd": "LSD"}
TRANSFORMS = ["identity", "hflip", "vflip", "rot180", "rot90cw", "rot90ccw"]


def inv(seg, t, Wid, Hid):
    """Map a transformed-frame segment back to identity coordinates."""
    def m(x, y):
        if t == "identity":  return (x, y)
        if t == "hflip":     return (Wid - 1 - x, y)
        if t == "vflip":     return (x, Hid - 1 - y)
        if t == "rot180":    return (Wid - 1 - x, Hid - 1 - y)
        if t == "rot90cw":   return (y, Hid - 1 - x)
        if t == "rot90ccw":  return (Wid - 1 - y, x)
    x0, y0 = m(seg[0], seg[1]); x1, y1 = m(seg[2], seg[3])
    return (x0, y0, x1, y1)


def feats(s):
    dx, dy = s[2] - s[0], s[3] - s[1]
    L = math.hypot(dx, dy)
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    mx, my = 0.5 * (s[0] + s[2]), 0.5 * (s[1] + s[3])
    return mx, my, ang, L


def adiff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def recall(A, B, mid_th=6.0, ang_th=5.0, lr=0.6):
    """Fraction of A segments with a matching B segment (spatial-grid accelerated)."""
    if not A:
        return float("nan")
    Bf = [feats(b) for b in B]
    grid = defaultdict(list)
    for i, (bmx, bmy, bang, bL) in enumerate(Bf):
        grid[(int(bmx // mid_th), int(bmy // mid_th))].append(i)
    hit = 0
    for a in A:
        amx, amy, aang, aL = feats(a)
        if aL < 1:
            continue
        cx, cy = int(amx // mid_th), int(amy // mid_th)
        ok = False
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for i in grid.get((gx, gy), ()):
                    bmx, bmy, bang, bL = Bf[i]
                    if bL >= 1 and math.hypot(amx - bmx, amy - bmy) <= mid_th \
                       and adiff(aang, bang) <= ang_th and min(aL, bL) / max(aL, bL) >= lr:
                        ok = True
                        break
                if ok: break
            if ok: break
        if ok: hit += 1
    return 100.0 * hit / len(A)


def load(eq, det):
    man = {}
    with open(os.path.join(eq, "manifest_all.csv"), newline="") as f:
        for r in csv.DictReader(f):
            man[int(r["frame"])] = (r["base"], r["transform"], int(r["W"]), int(r["H"]))
    segs = defaultdict(list)
    with open(os.path.join(eq, f"all_{det}", "segs.csv"), newline="") as f:
        for r in csv.DictReader(f):
            segs[int(r["frame"])].append((float(r["x0"]), float(r["y0"]),
                                          float(r["x1"]), float(r["y1"])))
    times = {}
    with open(os.path.join(eq, f"all_{det}", "times.csv"), newline="") as f:
        for r in csv.DictReader(f):
            times[int(r["frame"])] = (float(r["detect_ms"]), int(r["n_segs"]))
    return man, segs, times


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eq", required=True)
    args = ap.parse_args()

    for det in DETS:
        man, segs, times = load(args.eq, det)
        # index frames per base
        by_base = defaultdict(dict)          # base -> transform -> frame
        for fr, (base, t, W, H) in man.items():
            by_base[base][t] = fr
        eqrec = defaultdict(list)            # transform -> [recall%]
        tms = defaultdict(list)              # transform -> [ms]
        ns = defaultdict(list)               # transform -> [n_segs]
        for base, tf in by_base.items():
            if "identity" not in tf:
                continue
            id_fr = tf["identity"]
            Wid, Hid = man[id_fr][2], man[id_fr][3]
            A = segs.get(id_fr, [])
            for t, fr in tf.items():
                B = [inv(s, t, Wid, Hid) for s in segs.get(fr, [])]
                eqrec[t].append(recall(A, B))
                tms[t].append(times[fr][0]); ns[t].append(times[fr][1])
        print(f"\n=== {LABEL[det]} ===")
        print(f"{'transform':<10}{'equiv%':>9}{'detect ms':>11}{'segs':>8}")
        for t in TRANSFORMS:
            if t in eqrec:
                print(f"{t:<10}{st.median(eqrec[t]):>9.1f}{st.median(tms[t]):>11.1f}{st.median(ns[t]):>8.0f}")
        # summary excluding identity
        allr = [r for t in TRANSFORMS if t != "identity" for r in eqrec[t]]
        print(f"  -> median equivariance over all non-identity transforms: {st.median(allr):.1f}%")


if __name__ == "__main__":
    main()
