"""Aggregate liu4k_bench/results.csv into per-resolution median time/count tables
and speed ratios (vs SweepLSD one-pass). Emits a console table and summary CSV.

  python agg.py --results data/liu4k_bench/results.csv [--out summary_by_rung.csv]
"""
import argparse, csv, statistics as st
from collections import defaultdict

RUNGS = ["360", "720", "1080", "1440", "2160"]
RES = {"360": "640x360", "720": "1280x720", "1080": "1920x1080",
       "1440": "2560x1440", "2160": "3840x2160 (4K)"}
TIME = {"sw1_ms": "SweepLSD 1-pass", "sw_ms": "SweepLSD multi",
        "el_ms": "ELSED", "ed_ms": "EDLines", "lsd_ms": "LSD"}
CNT = {"sw1_n": "SweepLSD", "el_n": "ELSED", "ed_n": "EDLines", "lsd_n": "LSD"}


def med(xs):
    return st.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = defaultdict(list)
    with open(args.results, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["tag"]].append(r)

    summ = []
    for tag in RUNGS:
        rs = rows.get(tag, [])
        if not rs:
            continue
        t = {k: med([float(x[k]) for x in rs]) for k in TIME}
        c = {k: med([float(x[k]) for x in rs]) for k in CNT}
        sw1 = t["sw1_ms"]
        print(f"\n=== {RES[tag]}  (n={len(rs)}) ===")
        print("  times (median ms):  " +
              "  ".join(f"{lbl} {t[k]:.1f}" for k, lbl in TIME.items()))
        print("  ratio vs 1-pass:    " +
              "  ".join(f"{TIME[k]} {t[k]/sw1:.2f}x" for k in ("el_ms", "ed_ms", "lsd_ms")))
        print("  segments (median):  " +
              "  ".join(f"{lbl} {c[k]:.0f}" for k, lbl in CNT.items()))
        summ.append({"res": RES[tag], "n": len(rs),
                     **{k: round(t[k], 2) for k in TIME},
                     "el_x": round(t["el_ms"]/sw1, 2), "ed_x": round(t["ed_ms"]/sw1, 2),
                     "lsd_x": round(t["lsd_ms"]/sw1, 2),
                     **{k: round(c[k]) for k in CNT}})

    if args.out and summ:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
            w.writeheader(); w.writerows(summ)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
