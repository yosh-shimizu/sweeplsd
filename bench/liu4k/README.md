# LIU4K-v2 speed / memory / equivariance benchmark

The scripts that produce the public benchmark on the
[`docs/benchmarks.html`](https://ysmz334.github.io/sweeplsd/benchmarks.html) page.
Everything is measured on the **LIU4K-v2** corpus, which is **CC0-licensed** and
freely redownloadable, so the numbers are reproducible by anyone.

## Dataset

LIU4K-v2 (Liu et al., *A Comprehensive Benchmark for Single Image Compression
Artifact Reduction*, IEEE TIP 2020) — <https://structpku.github.io/LIU4K_Dataset/>.
We use the **validation** split's `Building` and `Street` categories (123 unique
structure-rich photographs; 7 byte-identical duplicates across the two categories
are deduplicated by filename). Every image is natively 4K–6K.

## 1. Preprocess (defines the exact benchmark inputs)

`preprocess.py` is the important part: it turns the raw LIU4K-v2 photos into the
uniform, multi-resolution, 8-bit grayscale inputs the benchmark runs on.

* **Rotate portrait → landscape** (90° CW). SweepLSD's memory and time are
  `O(width)`, so the width has to be held constant across the corpus.
* **16:9 centre-crop**, then **downscale** with a properly anti-aliased Lanczos
  filter (Pillow's `LANCZOS`, whose kernel scales with the reduction factor —
  unlike `cv2.INTER_LANCZOS4`, which does not and under-anti-aliases large
  downscales). Never upsamples.

```sh
python bench/liu4k/preprocess.py \
    --src /path/to/LIU4K_v2/validation \
    --out data/liu4k_bench \
    --rungs 360,720,1080,1440,2160        # 640x360 .. 3840x2160 (4K)
```

Output: `data/liu4k_bench/<rung>/<name>.png` (123 images per rung).

## 2. Timing

The repository's own harness times all four genuine detectors on a directory of
PNGs and prints the per-detector median (build with `-DSWEEPLSD_BUILD_BENCH=ON`):

```sh
sweeplsd_time_methods data/liu4k_bench/1080 --runs 5      # Full-HD headline
sweeplsd_time_methods data/liu4k_bench/2160 --runs 5      # 4K, etc.
```

`bench_run.py` / `agg.py` drive a *per-image* 4-way binary that also dumps each
detector's segments (used for the frame-time-spread and count analyses); point
`--exe` at your 4-way timing binary. `agg.py` turns its `results.csv` into the
per-resolution median / ratio table.

## 3. Memory

`mem_run.py` runs one detector per process and reads its peak working set
(`GetProcessMemoryInfo`), so the per-detector, per-resolution footprint is clean.

## 4. Equivariance (rotation / reflection repeatability)

`equiv_prep.py` writes the eight dihedral transforms (flips + 90° rotations) of a
subset; `equiv_analyze.py` maps each detector's detections back to the original
frame and reports the fraction reproduced — SweepLSD is the most repeatable of
the four.

## Notes

* All timings in the published docs are single-thread, i7-8700K, **GCC 15.2 +
  AVX2**, with the OpenCV the baselines link against built by the same compiler.
* `make_sweep_gif.py` renders the README's "one raster sweep" animation from a
  photo and its exported segments.
