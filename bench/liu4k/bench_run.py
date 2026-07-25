"""Run the 4-way (SweepLSD/ELSED/EDLines/LSD) timing+count benchmark over a
preprocessed LIU4K rung tree and collect per-image RESULT lines into a CSV.

Resumable: skips (name,tag) pairs already present in the output CSV.

  python bench_run.py --bench-dir <root with <tag>/*.png> \\
      --exe build_mingw/synth_multi.exe --out data/liu4k_bench/results.csv \\
      --rungs 360,720,1080,1440,2160 --runs 5
(Ensure mingw64-15.2/bin is on PATH so synth_multi's runtime DLLs resolve.)
"""
import argparse, csv, glob, os, subprocess, sys

COLS = ["name", "tag", "W", "H", "mp",
        "sw_ms", "sw1_ms", "lsd_ms", "ed_ms", "el_ms",
        "sw_n", "sw1_n", "lsd_n", "ed_n", "el_n"]


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["name"], r["tag"]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", default="360,720,1080,1440,2160")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--scratch", default=None, help="throwaway out-dir for det CSVs")
    args = ap.parse_args()
    rungs = args.rungs.split(",")
    scratch = args.scratch or os.path.join(os.path.dirname(args.out), "_bench_scratch")
    os.makedirs(scratch, exist_ok=True)
    done = load_done(args.out)
    new = not os.path.exists(args.out)
    fout = open(args.out, "a", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(COLS); fout.flush()

    total = 0
    for tag in rungs:
        imgs = sorted(glob.glob(os.path.join(args.bench_dir, tag, "*.png")))
        for img in imgs:
            name = os.path.splitext(os.path.basename(img))[0]
            if (name, tag) in done:
                continue
            try:
                p = subprocess.run([args.exe, img, "--tag", tag, "--out-dir", scratch,
                                    "--runs", str(args.runs)],
                                   capture_output=True, text=True, timeout=1800)
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT {name} @ {tag}", flush=True); continue
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT,")), None)
            if not line:
                print(f"NO RESULT {name} @ {tag}: {p.stdout[-200:]} {p.stderr[-200:]}", flush=True)
                continue
            f = line.split(",")            # RESULT,tag,W,H,mp,sw,sw1,lsd,ed,el,ns...
            w.writerow([name] + f[1:15]); fout.flush()
            total += 1
            if total % 10 == 0:
                print(f"  ...{total} rows ({tag} in progress)", flush=True)
    fout.close()
    print(f"done, {total} new rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
