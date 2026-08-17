#!/usr/bin/env python3
"""B5: sensitivity of the split-half cross-validated Table 11 (tab:fair) numbers
to the particular split (ChatGPT review, Major 11 / Question 9).

Reads the same rows CSVs as sweeplsd/tools/vp_bestcfg_cv.py
(img,method,variant,err) and applies the IDENTICAL selection rule:
  - split images into two halves;
  - on each half as training, pick the variant with the lowest median error
    (ties broken by sorted variant name, as in vp_bestcfg_cv.py);
  - score the pick on the other half; pool the two test halves.

Step 1 (sanity gate): the paper's deterministic even/odd split must reproduce
Table 11 exactly (medians to 0.01 deg, percentages to 0.1).
Step 2 (Monte Carlo): N random half splits (seeded); report the distribution
of the pooled median / %<2 / %<5 per method, variant pick frequencies, and
pairwise ranking stability.

Usage: split_sensitivity.py rows.csv [more.csv ...] [--n 1000] [--seed 0]
       [--methods a,b,c]
"""
import csv
import random
import statistics
import sys
from collections import Counter


def load(paths):
    data = {}  # method -> variant -> {img: err}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                data.setdefault(r["method"], {}) \
                    .setdefault(r["variant"], {})[r["img"]] = float(r["err"])
    return data


def cv_once(dm, fa, fb):
    """Identical rule to vp_bestcfg_cv.py: returns (pooled_errs, picks)."""
    pooled, picks = [], []
    for train, test in ((fa, fb), (fb, fa)):
        best = min(sorted(dm),
                   key=lambda v: statistics.median([dm[v][im] for im in train]))
        picks.append(best)
        pooled.extend(dm[best][im] for im in test)
    return pooled, picks


def summarize(pooled):
    med = statistics.median(pooled)
    lt2 = 100.0 * sum(1 for e in pooled if e < 2) / len(pooled)
    lt5 = 100.0 * sum(1 for e in pooled if e < 5) / len(pooled)
    return med, lt2, lt5


def quantile(xs, q):
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main(argv):
    paths, methods, n_rep, seed = [], None, 1000, 0
    i = 1
    while i < len(argv):
        if argv[i] == "--methods":
            i += 1; methods = argv[i].split(",")
        elif argv[i] == "--n":
            i += 1; n_rep = int(argv[i])
        elif argv[i] == "--seed":
            i += 1; seed = int(argv[i])
        else:
            paths.append(argv[i])
        i += 1
    if not paths:
        print(__doc__); return 1

    data = load(paths)
    methods = methods or sorted(data)

    # ---- step 1: deterministic even/odd baseline (must match Table 11) ----
    print("== baseline (paper even/odd split) ==")
    print(f"{'method':8s} {'medErr':>7s} {'<2deg':>7s} {'<5deg':>7s}  picks")
    imgs_per_method = {}
    for m in methods:
        dm = data[m]
        imgs = sorted(set.intersection(*[set(v) for v in dm.values()]))
        imgs_per_method[m] = imgs
        pooled, picks = cv_once(dm, imgs[0::2], imgs[1::2])
        med, lt2, lt5 = summarize(pooled)
        print(f"{m:8s} {med:6.2f}° {lt2:6.1f}% {lt5:6.1f}%  {picks[0]} / {picks[1]}")

    # ---- step 2: Monte Carlo over random half splits ----
    rng = random.Random(seed)
    per_method = {m: {"med": [], "lt2": [], "lt5": [], "picks": Counter()}
                  for m in methods}
    # One split per replicate, shared by every method (the image sets are
    # identical across methods in these CSVs), so the pairwise ranking
    # stability below is a PAIRED comparison under a common split.
    shared = sorted(set.intersection(*[set(imgs_per_method[m]) for m in methods]))
    assert all(len(imgs_per_method[m]) == len(shared) for m in methods), \
        "image sets differ across methods; paired splits invalid"
    for rep in range(n_rep):
        imgs = shared[:]
        rng.shuffle(imgs)
        half = len(imgs) // 2
        fa, fb = imgs[:half], imgs[half:]
        for m in methods:
            pooled, picks = cv_once(data[m], fa, fb)
            med, lt2, lt5 = summarize(pooled)
            s = per_method[m]
            s["med"].append(med); s["lt2"].append(lt2); s["lt5"].append(lt5)
            s["picks"].update(picks)

    print(f"\n== Monte Carlo, {n_rep} random half splits (seed {seed}) ==")
    print(f"{'method':8s} {'med(med)':>8s} {'IQR':>13s} {'2.5-97.5%':>15s} "
          f"{'lt2(med)':>8s} {'lt5(med)':>8s}")
    for m in methods:
        s = per_method[m]
        med_med = statistics.median(s["med"])
        q25, q75 = quantile(s["med"], 0.25), quantile(s["med"], 0.75)
        qlo, qhi = quantile(s["med"], 0.025), quantile(s["med"], 0.975)
        print(f"{m:8s} {med_med:7.2f}° [{q25:5.2f},{q75:5.2f}] "
              f"[{qlo:6.2f},{qhi:6.2f}] "
              f"{statistics.median(s['lt2']):7.1f}% "
              f"{statistics.median(s['lt5']):7.1f}%")

    print("\n-- variant pick frequency (top 3 per method) --")
    for m in methods:
        tot = sum(per_method[m]["picks"].values())
        top = per_method[m]["picks"].most_common(3)
        print(f"{m:8s} " + "  ".join(f"{v}:{100.0*c/tot:.0f}%" for v, c in top))

    print("\n-- pairwise ranking stability (P[row median < col median]) --")
    hdr = "         " + "".join(f"{m:>9s}" for m in methods)
    print(hdr)
    for a in methods:
        row = f"{a:8s} "
        for b in methods:
            if a == b:
                row += f"{'-':>9s}"
            else:
                wins = sum(1 for x, y in zip(per_method[a]["med"],
                                             per_method[b]["med"]) if x < y)
                row += f"{100.0*wins/n_rep:8.1f}%"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
