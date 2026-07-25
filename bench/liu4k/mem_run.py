"""Peak working-set memory per detector per resolution, using export_segs
(one detector per process -> clean peak WS via GetProcessMemoryInfo).

  python mem_run.py --bench-dir data/liu4k_bench --exe build_mingw/export_segs.exe \\
      --out data/liu4k_bench/mem_by_rung.csv --rungs 360,720,1080,1440,2160 --nimg 12
(mingw64-15.2/bin on PATH for runtime DLLs.)
"""
import argparse, csv, glob, os, subprocess

DETS = ["sweep1", "sweeplsd", "elsed", "edlines", "lsd"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", default="360,720,1080,1440,2160")
    ap.add_argument("--nimg", type=int, default=12, help="images per manifest (peak is size-determined)")
    args = ap.parse_args()
    scratch = os.path.join(os.path.dirname(args.out), "_mem_scratch")
    os.makedirs(scratch, exist_ok=True)

    rows = []
    for tag in args.rungs.split(","):
        root = os.path.join(args.bench_dir, tag)
        imgs = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(root, "*.png")))][:args.nimg]
        if not imgs:
            continue
        man = os.path.join(scratch, f"manifest_{tag}.csv")
        with open(man, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["frame", "image_path"])
            for i, n in enumerate(imgs):
                w.writerow([i, n])
        for det in DETS:
            p = subprocess.run([args.exe, "--gt", man, "--root", root,
                                "--out", os.path.join(scratch, f"{tag}_{det}"), "--det", det],
                               capture_output=True, text=True, timeout=3600)
            line = next((l for l in p.stdout.splitlines() if l.startswith("PEAKWS_MB,")), None)
            mb = float(line.split(",")[2]) if line else float("nan")
            rows.append({"tag": tag, "det": det, "peak_mb": round(mb, 2)})
            print(f"  {tag:>5} {det:<9} peak {mb:7.2f} MB", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "det", "peak_mb"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
