"""Major 4: analysis of the randomized-scene protocol CSV from
sweeplsd_evaluate --randomized N --csv rand.csv.

Per method: pooled F over all scenes at each knob -> pool-best knob (one
operating point for the whole pool); bootstrap 95% CI of the pooled F
(resampling scenes, 10k, seed 0); slice tables (contrast / blur / noise
bins) at that same knob; pooled matched-geometry errors.

A knob is eligible only if every scene has a row for it (ELSED's runner
aborts below minLineLen ~7-10, mirroring the fixed protocol's handling).
"""
import csv
import sys
from collections import defaultdict

import numpy as np

NBOOT = 10000


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0, p, r


def main(path):
    rows = list(csv.DictReader(open(path, newline="")))
    scenes = sorted({int(r["scene"]) for r in rows})
    nsc = len(scenes)
    meta = {}
    data = defaultdict(dict)  # (method, knob) -> {scene: (tp,fp,fn,slat,sang,nm)}
    for r in rows:
        i = int(r["scene"])
        meta[i] = dict(contrast=float(r["contrast"]), blur=float(r["blur"]),
                       noise=float(r["noise"]), width=float(r["width"]))
        data[(r["method"], float(r["knob"]))][i] = (
            int(r["tp"]), int(r["fp"]), int(r["fn"]),
            float(r["sum_lat"]), float(r["sum_ang"]), int(r["nm"]))

    methods = sorted({m for m, _ in data})
    rng = np.random.default_rng(0)
    print(f"{nsc} scenes")
    print(f"{'method':14s} {'knob':>5s} {'F':>6s} {'95% CI':>15s} {'P':>6s} "
          f"{'R':>6s} {'lat':>6s} {'ang':>6s}")
    best_of = {}
    for m in methods:
        best = None
        for (mm, knob), d in data.items():
            if mm != m or len(d) != nsc:
                continue
            tp = sum(v[0] for v in d.values())
            fp = sum(v[1] for v in d.values())
            fn = sum(v[2] for v in d.values())
            f, p, r = f1(tp, fp, fn)
            if best is None or f > best[0]:
                best = (f, p, r, knob, d)
        if best is None:
            print(f"{m:14s}  (no knob with full coverage)")
            continue
        f, p, r, knob, d = best
        best_of[m] = (knob, d)
        arr = np.array([d[i][:3] for i in scenes])  # (nsc, 3)
        idx = rng.integers(0, nsc, size=(NBOOT, nsc))
        s = arr[idx].sum(axis=1)  # (NBOOT, 3)
        prec = np.where(s[:, 0] + s[:, 1] > 0, s[:, 0] / (s[:, 0] + s[:, 1]), 0)
        rec = np.where(s[:, 0] + s[:, 2] > 0, s[:, 0] / (s[:, 0] + s[:, 2]), 0)
        fb = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
        lo, hi = np.percentile(fb, [2.5, 97.5])
        slat = sum(v[3] for v in d.values())
        sang = sum(v[4] for v in d.values())
        nm = sum(v[5] for v in d.values())
        print(f"{m:14s} {knob:5g} {f:6.3f} [{lo:6.3f},{hi:6.3f}] {p:6.3f} "
              f"{r:6.3f} {slat/nm:5.2f}px {np.degrees(sang/nm):5.2f}d")

    def slice_table(title, key, edges):
        labels = [f"<{edges[0]:g}"] + \
                 [f"{edges[j]:g}-{edges[j+1]:g}" for j in range(len(edges) - 1)] + \
                 [f">={edges[-1]:g}"]
        bins = {i: np.searchsorted(edges, meta[i][key], side="right")
                for i in scenes}
        print(f"\n-- pooled F by {title} (at each method's pool-best knob) --")
        print(f"{'method':14s} " + "".join(f"{l:>12s}" for l in labels))
        for m in methods:
            if m not in best_of:
                continue
            _, d = best_of[m]
            row = f"{m:14s} "
            for b in range(len(labels)):
                sel = [i for i in scenes if bins[i] == b]
                if not sel:
                    row += f"{'---':>12s}"
                    continue
                tp = sum(d[i][0] for i in sel)
                fp = sum(d[i][1] for i in sel)
                fn = sum(d[i][2] for i in sel)
                f, _, _ = f1(tp, fp, fn)
                row += f"{f:12.3f}"
            print(row + f"   (n={[sum(1 for i in scenes if bins[i]==b) for b in range(len(labels))]})"
                  if m == methods[0] else row)

    slice_table("contrast (gray levels)", "contrast", [20.0, 60.0])
    slice_table("blur sigma (px)", "blur", [0.75])
    slice_table("noise sigma", "noise", [10.0])


if __name__ == "__main__":
    main(sys.argv[1])
