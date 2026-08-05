#!/usr/bin/env python3
"""Split-half cross-validated aggregation for the fair "best estimator per
detector" protocol (paper Section 6.2).

Input: one or more CSVs written by sweeplsd_vp_bestcfg (img,method,variant,err
plus, when the harness ran with --time-runs, an est_ms column).
For each method independently:
  1. Sort image names; fold A = even indices, fold B = odd indices.
  2. For each fold as the TRAINING half, select the estimator variant with the
     lowest median error on that half.
  3. Evaluate the selected variant on the OTHER half; pool the two test halves
     (every image is scored exactly once, under a variant chosen without it).
  4. Report the pooled median and the pooled fraction below 2 deg / 5 deg.

This rule was verified to reproduce the 2026-07-06 study's published numbers
exactly from that study's surviving row CSVs (YUD 0.85/0.92/1.00, NYU
5.97/7.64/7.82, and the %<2 / %<5 columns).

Timing columns (optional):
  * est_ms rows (from --time-runs) are pooled the same way as errors — every
    image contributes the estimation time of the variant its fold selected —
    and reported as the pooled median.
  * --det-dir method=DIR attaches per-image detection times for a method from
    runner output files DIR/<img>.txt whose first line is "<count> <ms>"
    (the sweep_one / lsd_one / ELSED / EDLines runner format). total is the
    pooled median of per-image (det + est).

Usage: vp_bestcfg_cv.py rows.csv [more_rows.csv ...] [--methods a,b,c]
       [--det-dir method=DIR ...]
"""
import csv
import os
import statistics
import sys


def read_det_dir(d):
    out = {}
    for fn in os.listdir(d):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(d, fn)) as f:
            head = f.readline().split()
        if len(head) >= 2:
            out[fn[:-4]] = float(head[1])
    return out


def main(argv):
    paths, methods, det_dirs = [], None, {}
    i = 1
    while i < len(argv):
        if argv[i] == "--methods":
            i += 1
            methods = argv[i].split(",")
        elif argv[i] == "--det-dir":
            i += 1
            m, d = argv[i].split("=", 1)
            det_dirs[m] = read_det_dir(d)
        else:
            paths.append(argv[i])
        i += 1
    if not paths:
        print(__doc__)
        return 1

    data = {}  # method -> variant -> {img: (err, est_ms|None)}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                est = r.get("est_ms")
                data.setdefault(r["method"], {}) \
                    .setdefault(r["variant"], {})[r["img"]] = \
                    (float(r["err"]), float(est) if est else None)

    print(f"{'method':10s} {'medErr(CV)':>10s} {'<2deg':>7s} {'<5deg':>7s} "
          f"{'det':>7s} {'est':>7s} {'total':>7s} {'imgs':>5s}  "
          f"picks (fold A train / fold B train)")
    for m in (methods or sorted(data)):
        dm = data[m]
        imgs = sorted(set.intersection(*[set(v) for v in dm.values()]))
        fa, fb = imgs[0::2], imgs[1::2]
        pooled, pooled_est, pooled_tot, picks = [], [], [], []
        det = det_dirs.get(m, {})
        for train, test in ((fa, fb), (fb, fa)):
            best = min(sorted(dm),
                       key=lambda v: statistics.median([dm[v][im][0] for im in train]))
            picks.append(best)
            for im in test:
                err, est = dm[best][im]
                pooled.append(err)
                if est is not None:
                    pooled_est.append(est)
                    if im in det:
                        pooled_tot.append(det[im] + est)
        med = statistics.median(pooled)
        lt2 = 100.0 * sum(1 for e in pooled if e < 2) / len(pooled)
        lt5 = 100.0 * sum(1 for e in pooled if e < 5) / len(pooled)
        med_det = statistics.median([det[im] for im in imgs if im in det]) \
            if det else float("nan")
        med_est = statistics.median(pooled_est) if pooled_est else float("nan")
        med_tot = statistics.median(pooled_tot) if pooled_tot else float("nan")
        print(f"{m:10s} {med:9.2f}° {lt2:6.1f}% {lt5:6.1f}% "
              f"{med_det:6.2f} {med_est:6.2f} {med_tot:6.2f} {len(imgs):5d}  "
              f"{picks[0]} / {picks[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
