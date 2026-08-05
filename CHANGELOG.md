# Changelog

## Unreleased

- **Fair-protocol VP study now prices the downstream too.**
  `sweeplsd_vp_bestcfg` gains `--time-runs R`: each (image, detector, variant)
  row also records the downstream cost (calibrate + Manhattan estimation) as
  the median of R runs in a new `est_ms` CSV column (off by default — output
  unchanged). `tools/vp_bestcfg_cv.py` pools `est_ms` under each image's
  cross-validated pick and, with `--det-dir method=DIR` (per-image runner
  files, `<count> <ms>` header), reports detect / VP-est. / total medians.
  The menu harness's estimator no longer recomputes axis scores inside its
  sort comparator and per-seed scan (~10× faster); comparison outcomes are
  unchanged, so every CSV row is bit-identical (verified against the
  published row CSVs).
- **Manhattan estimator: scoring stage AVX2-vectorized, output
  bit-identical.** The `O(candidates × lines)` inlier tests — the estimator's
  dominant cost — now run 4 lines per AVX2 iteration over an SoA repack of
  the line normals, with the same mul+fma rounding as the compiled scalar
  dot, NaN padding instead of a scalar tail, and a bitwise-fixed-point early
  exit in the triad refinement. Applied to `bench/vp_bestcfg.cpp` and
  `examples/manhattan_frame.cpp` (scalar fallback kept for non-AVX2/FMA
  builds, including MSVC). Verified bit-identical over all 68,100 menu
  evaluations on both datasets plus 30 images of the shipping estimator.
  VGA estimation drops to 1.6–3.7 ms per frame; what remains is the
  multi-start refinement's serial per-seed eigensolves.
- **Docs: end-to-end task times published.** `vp_evaluation.html` §4 gains
  detect / VP est. / total columns (640×480 medians; one toolchain, one
  measurement window, detectors interleaved image-by-image): even with the
  vectorized estimator, at VGA the estimation stage (1.6–3.7 ms) exceeds
  SweepLSD's own detection (1.5–1.8 ms), so the end-to-end margin over ELSED
  compresses to 1.5–2.0×, while at 4K the picture inverts (5–6 ms estimation
  against 30–580 ms detection). `applications.html` now reports the 4K
  horizon lock's complete per-frame compute (detection 32.0 + estimation
  1.1 = 33.2 ms median) and prices the downstream in the resolution sweep.
- **Build-sensitivity disclosures.** ED_Lib: ±1–2 segments per frame between
  compiler builds of the same source and OpenCV version (NYU CV median
  8.09°→8.95°, ranking unaffected). LSD: last-bit detection differences
  between GCC generations move its cross-validated York Urban median between
  0.83° and 0.94° (ELSED's and SweepLSD's medians move ≤0.02°; every NYU
  median reproduces to 0.01°), so LSD's outdoor lead over SweepLSD is within
  build jitter. Root cause of the discovery: the 2026-07 fair-protocol
  harness had been silently built by GCC 8.1 through a stale compiler path
  in an old CMake cache; the time columns and these disclosures re-anchor
  the harness to GCC 15.2.

## v3.0.4 (2026-07-17)

- **Labelling performance: Full-HD one-pass ~12.2 → ~11.5 ms (−5 %), output
  bit-identical.** The Stage-3 row scan in `src/labeling.cpp` now advances a
  word at a time: `Interior` is the only odd `Feature` value, so masking bit 0
  of each byte finds label-relevant pixels directly (skipping endpoint-only
  words too, which an all-zero test cannot) and `ctz` jumps straight to the
  hit. No algorithm, numerics, parameter, or API change; verified
  bit-identical over 13 images / 11,468 segments with both drivers, and the
  HLS core stays bit-exact against the software golden model.
- **Kernel performance: Full-HD one-pass ~13.5 → ~12.0 ms (−11 %), output
  bit-identical.** Three independent rewrites in `src/kernels.hpp`: the edge
  NMS selects its competitor pair with a uint16 mask blend instead of int
  multiplies (edge stage −37 %); `endpointCore` computes its straight-line
  test from eight incremental ring-window parities instead of a 15-term
  thermometer byte + popcount (endpoint stage −17 %, verified exhaustively
  over all 2^16 ring configurations); the sub-pixel NMS indexes its three
  power samples directly with the two border columns peeled (−9 %).
- **Docs/README/paper numbers refreshed to the current build** (they had
  lagged since v3.0.1): Full-HD headline one-pass 11.4 ms / multi-pass
  15.8 ms; ELSED 28.2 ms (2.5×), EDLines 39.4 ms (3.5×, median over the test
  corpus), LSD
  247 ms (22×). The README headline table and the paper's per-scale table now
  include ELSED and EDLines respectively.

## v3.0.2 (2026-07-15)

- **Build portability fixes (no behaviour change).** v3.0.1's cold-throw
  helper used a GCC-only attribute spelling that MSVC cannot parse; the
  load-bearing `noinline` is now emitted through a portable
  `SWEEPLSD_COLD_NOINLINE` macro, so the core library builds under Visual C++
  again.
- **`sweeplsd_hotspots` restricted to x86.** The one-pass hotspot profiler
  uses `rdtsc` and never built on ARM64; its CMake target is now gated on
  `SWEEPLSD_ARCH_X86`, so non-x86 builds skip this diagnostic tool instead of
  failing. CI is green on linux-gcc, linux-clang, windows-msvc, and
  macos-clang (ARM64).

## v3.0.1 (2026-07-15)

- **Labelling performance fix (output bit-identical).** The pool-overflow
  `throw` added in v3.0.0 was inlined into the per-pixel labelling hot loop,
  and its exception landing pads poisoned that loop's codegen even though the
  throw is never taken on valid input. Moving it to a cold, non-inlined
  `[[noreturn]]` helper cuts the streaming detector from ~17.7 ms to ~13.8 ms
  per Full-HD frame (−22 %); the grow-and-report and hard-error behaviours
  are unchanged.

## v3.0.0 (2026-07-15)

- **BREAKING: removed `Params::sparse_label_scan`.** The labeling row scan's
  8 px zero-word skip was mean-neutral but added content-dependent timing
  jitter; the scan is now a single deterministic left-to-right sweep. Code
  that set the flag must drop the assignment; output is unchanged.
  (`sparse_feature_scan` is kept — skipping blank runs in the endpoint stage
  saves real work.)
- **Bounded label pool — O(width) label memory.** The per-frame label table no
  longer grows unbounded (it reached ~25k slots / ~2.5 MB on a dense Full-HD
  frame): both detectors now keep a fixed pool of label records addressed
  through a ring free-list and recycle each label's slot the moment it dies,
  matching the thesis's bounded design. The pool starts at width/4 and grows
  toward the ⌈width/2⌉ theoretical bound only if an input needs it. Working
  set drops to ~15 KB (cache-resident); **output is bit-identical**. Full
  internals and the recycling safety proof: `docs/labeling-internals.md`.
- **New API: `lastPoolGrowthEvents()`.** Pool growths during the most recent
  `detect()`/`detectOnePass()` on the calling thread (0 = normal), surfaced
  with a stderr warning; needing more than width/2 is impossible for a
  correct input, so it throws `std::runtime_error` rather than masking a bug.
- **Streaming labeller speedup.** Templated unweighted-fast moment
  accumulation (bit-identical when `weight_by_gradient` is off).
- **One-pass hotspot profiling tools.** `sweeplsd_hotspots` (per-stage cycle
  breakdown), a self-contained DWARF-based sampling line profiler
  (`tools/line_profiler.cpp`), and `docs/profiling.md`.

## v2.0.0 (2026-07-14)

- **BREAKING: `Params{}` is now the shipped configuration** (all measured
  refinements enabled), matching how the paper presents SweepLSD — one
  detector, one configuration. Code that relied on the old default's
  2014-thesis behaviour should switch to the new
  **`Params::original2014()`**; `Params::improved()` remains as an alias of
  the default, so code written against earlier releases keeps compiling and
  keeps its meaning. The CLI gained `--2014` for the original behaviour, and
  all benches/testbenches/golden-vector generators were moved to
  `original2014()` where they measured the 2014 configuration (their outputs
  are unchanged).

## v1.2.0 (2026-07-13)

- **Stronger collinear linking** (`Params::link_collinear`, still off by
  default). Three changes make larger gap jumps safe: **lateral consistency**
  (`link_lateral_tol`, default 1 px — segments link only if each one's
  endpoints lie on the *line* of the other, not merely parallel to it),
  `link_max_gap` default **4 → 9 px**, and a **two-stage length threshold**
  (`link_admit_pix`, default 5 — short fragments may enter the linker, but a
  chain must evidence the full length as span). Junction cuts and noise
  breaks are jumped without fusing the parallel flanks of thin bars.
  Measured: synthetic-GT F-max (strict one-to-one) σ0 0.966→0.973, σ20
  0.907→**0.935**, geometry errors unchanged; circles stay 0; downstream
  Manhattan-frame medians York 1.05°→**0.99°**, NYU 12.7°→**10.7°** (fixed
  estimator) and **5.23°** under the cross-validated fair protocol.
- **Benchmark harness additions**: genuine-ELSED ingestion (`--elsed-dir`)
  for the synthetic-GT, isotropy, and vanishing-point studies; the
  fair-protocol estimator study is now a committed tool
  (`sweeplsd_vp_bestcfg` + `tools/vp_bestcfg_cv.py`) whose aggregation
  reproduces the published numbers exactly.

## v1.1.0 (2026-07-13)

The FPGA release: the thesis proposed OPLSD as a hardware-oriented method but
left the FPGA form as future work — this release closes that gap.

- **HLS implementation** (`hls/`): the full detector as synthesizable HLS C++
  (Vitis HLS; Artix-7 xc7a35t reports, front-end II=1 at 100 MHz), plus a
  tool-free g++ compatibility shim and golden-parity testbenches against
  `detect()`.
- **Verilog RTL implementation** (`rtl/`): portable hand-written core
  (streaming front-end → event FIFO → labelling back-end → integer judge),
  held **bit-exact** against the HLS C model and the C++ reference — the
  SW == HLS == RTL parity gate covers the full 150-photo Full-HD test corpus.
- **Live board demo** (`rtl/boards/atlys/`): Digilent Atlys (Spartan-6 LX45,
  2009 silicon) — HDMI in → detect → green segment overlay → HDMI out at
  1080p30 and 720p60, no frame buffer, no external memory, single
  recovered-clock domain. The `improved()` refinements that fit the
  streaming/integer model are all in the hardware.
- **Back-end throughput levers** (each measured and bit-exact): dense-frame
  event-FIFO overflow on the LX45 went from ~52 % corpus segment loss (first
  live build) to **~0.2 %** at the shipped 2048-deep FIFO, with zero-drop
  proven in simulation at depth 8192 — see `rtl/DESIGN.md`, "Overflow
  reality check".
- **Live-robustness set**: event-FIFO marker reserve + hysteretic shedding,
  overlay record-FIFO deepening, judge watchdog, per-pass UART telemetry and
  diagnostic LEDs.
- **XST workaround (important for board builders)**: ISE XST's FSM
  re-encoding **mis-synthesizes the back-end FSM** — a silent functional
  divergence at Timing Score 0, invisible to RTL simulation. All ISE builds
  now pass `-fsm_extract NO`; the full story is in `rtl/DESIGN.md`.
- **Fidelity fix (changes default output)**: the outer 3 px of the frame are
  excluded from the edge test (`Params::edge_border_margin = 3`), removing a
  spurious full-frame ring of false edges that the 2×2 gradient manufactures
  at the image border (the original 2014 implementation suppressed a
  comparable band). Baseline segment counts drop ~4 % and the baseline's
  heavy-noise F-max recovers (σ10: 0.47 → 0.94); the improved config is
  unchanged. Evaluation pages regenerated.
- **Evaluation tooling**: 3-level SW / HLS-C / RTL comparison renderer, RTL
  ground-truth burst-overflow simulator, FIFO-drop visualizers.

## v1.0.1 (2026-07-06)

- Fix non-x86 builds and the bench directory listing; add repository links.

## v1.0.0 (2026-07-06)

First public release.

- **Core library** (`sweeplsd::sweeplsd`, MIT, zero dependencies): the
  one-pass line segment detector proposed as *OPLSD* in a 2014 master's
  thesis (Yoshiyasu Shimizu, Waseda University), reimplemented from scratch
  in C++17. Two drivers over shared kernels, tested identical: `detect()`
  (multi-pass, readable) and `detectOnePass()` (streaming single sweep,
  O(width) memory, fastest). Integer-only per-pixel core, no SIMD
  intrinsics — GCC/Clang auto-vectorize the kernels.
- **Thesis-faithful baseline + measured improvements**: `Params{}`
  reproduces the thesis; `Params::improved()` enables sub-pixel NMS,
  adaptive streaming hysteresis, curve rejection, border margin,
  half-pixel lattice correction, and more — each individually benchmarked.
- **I/O and adapters**: `sweeplsd::io` (stb-based PNG/JPG),
  `sweeplsd::opencv` (header-only `cv::Mat` adapter).
- **Tools**: CLI, per-stage profiler, stage-dump.
- **Examples**: calibrated Manhattan-frame estimation with the
  measured-best estimator configuration (`sweeplsd_manhattan`),
  uncalibrated vanishing points, OpenCV integration.
- **Benchmarks** (`-DSWEEPLSD_BUILD_BENCH=ON`): synthetic-GT quality,
  timing, isotropy, and downstream vanishing-point evaluation against the
  genuine author implementations of LSD (AGPL, fetched at configure time —
  never vendored) and ED_Lib EDLines (MIT, fetched; needs OpenCV).
- **Docs** (GitHub Pages): algorithm walkthrough with real stage
  intermediates, speed/quality/vanishing-point evaluations, all numbers
  regenerated from this repository's harness.

Headline numbers (Full-HD, i7-8700K, single thread, AVX2): ~17 ms/frame
one-pass — ~2.5× faster than ED_Lib EDLines, ~14× faster than LSD; best
clean/low-noise F-max and best per-segment direction accuracy of the three;
known limitation (soft low-contrast edges) measured and documented.
