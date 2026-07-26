# SweepLSD RTL — phase-2 design (Digilent Atlys / Spartan-6 LX45)

Hand-written Verilog port of the phase-1 HLS core (`../hls/`), targeting the
2014-era hardware the thesis had in mind: a **live HDMI in → detect → overlay
→ HDMI out** demo on the Digilent Atlys (XC6SLX45), with **no frame buffer and
no external memory** — the detector state is the thesis's ~70 KiB of BRAM.

The architecture is the one already validated end-to-end in phase 1 (bit-exact
C-sim over the full test corpus + C/RTL co-sim): II=1 line-buffered
front-end → sparse event FIFO → elastic labelling back-end → segment records.
This phase re-expresses it in portable Verilog because Spartan-6 is not a
Vitis HLS target (ISE 14.7 only).

> **Building the board bitstream** (ISE 14.7) and the third-party licensing of
> the HDMI glue — in particular the Xilinx **XAPP495** DVI reference design,
> which is git-ignored and must be fetched separately — are documented in
> [`boards/atlys/README.md`](boards/atlys/README.md).

## Verification strategy (same standard as phase 1)

Bit-exactness against `sweeplsd::detect()` stays the acceptance criterion:

- `tools/dump_vectors.cpp` (host, links the sweeplsd library) renders the
  phase-1 test images and dumps each stage boundary as hex vector files:
  gray in, gaussian, gradient(power,dir), edge, feature, event stream,
  accepted segment records.
- `tb/*.v` (Icarus Verilog, OSS CAD Suite — no ISE needed for the inner loop)
  drive each module and the full core from those vectors and fail on the first
  mismatching sample. ISE is only needed for the board build.

Every optimisation described below was admitted only under this gate: the
small synthetic vectors, the full-chain `tb_sweep_core`, and Full-HD golden
runs on five corpus photos, at both CE_DIV 1 and 2. ("The corpus", here and
throughout, is a fixed test set of 150 real Full-HD photographs used across
this project's parity gates; the photographs are not redistributed — Full-HD
golden vectors are regenerated locally with `dump_vectors`.)

## Clocking (Atlys)

Single processing clock = the **recovered HDMI RX pixel clock** (74.25 MHz for
both demo modes), so the whole path RX → core → overlay → TX is synchronous
and frame-buffer-free. Serdes clocks (5×, DDR) come from PLL_ADV + BUFPLL as
in the standard Spartan-6 TMDS designs.

| mode | pixel clk | TMDS bit rate | note |
|---|---|---|---|
| 720p60 (default) | 74.25 MHz | 742.5 Mbps | most compatible source mode |
| 1080p30 (option) | 74.25 MHz | 742.5 Mbps | thesis evaluation format (1920×1080) |

1080p60 (1.485 Gbps) exceeds Spartan-6 serdes capability — out of scope.

**Half-rate core.** On Spartan-6 the lockstep front-end's combinational chain
(gauss BRAM read → vertical → horizontal → gradient → NMS → edge BRAM write)
measures ~25 ns — fine at VGA's 40 ns, not at 720p's 13.3 ns. Rather than
pipelining the stages (which would shift every stage's column-lag constants
and the row-boundary wrap-around), the core runs at HALF the pixel rate: a
detector pass over 720p takes ~25 ms ⇒ segment updates every 2 display frames
(~30 Hz), display always 60 Hz. The portable core exposes this as a global
clock enable `en` (any 1-in-N duty is an exact time dilation; regression-run
at CE_DIV 2); the board build instead drives the core from a REAL half-rate
clock (PLL CLKOUT2 = 37.5 MHz, phase-aligned 2:1 with the 75 MHz pixel clock,
en tied 1) — a multicycle-constraint variant was abandoned after ISE's
ngdbuild silently dropped part of its TNM constraint group, and a real clock
cannot be mis-grouped. The pix<->core handshakes are synchronous 2:1 paths,
fully covered by static timing. The internal-scene build runs the same scheme
from the on-board oscillator (75.0 MHz ⇒ 60.6 Hz, +1 % over CEA — displays
accept it); the live build gets the exact 74.25 MHz from the HDMI RX clock.

## Module hierarchy

```
boards/atlys/atlys_top.v      — pins, PLLs, resets
  hdmi_rx.v                   — ISERDES2 + TMDS decode → pix clk, rgb, de/hs/vs  (from Atlys reference designs)
  rgb2gray.v                  — BT.601 integer luma: (77R + 150G + 29B) >> 8
  core/sweep_core.v           — PORTABLE (no vendor primitives):
    gauss_v.v  gauss_h.v      — 5-tap separable, {16,64,96,64,16}, >>10
    gradient.v                — 2x2, power + H/V dir
    edge_nms.v                — threshold + NMS (3x3 window on power)
    feature.v                 — 5x5 endpoint-candidate (endpoint_core.v = pure LUT logic)
    event_pack.v              — dense feature -> sparse events (+EOR/EOF)
    event_fifo.v              — 2048-deep elastic buffer (backpressures front-end)
    backend.v                 — labelling FSM (below)
  seg_db.v                    — per-frame segment store (ping-pong), feeds overlay
  overlay.v                   — half-res 1-bit mask, ping-pong BRAM + Bresenham drawer
  hdmi_tx.v                   — TMDS encode + OSERDES2                            (from Atlys reference designs)
```

`core/` is simulation-portable Verilog-2001 (inferred BRAM only); everything
vendor-specific (serdes, PLL, TMDS) lives under `boards/atlys/`.

Front-end modules translate 1:1 from `../hls/src/frontend.cpp` — same loop
structure ((h+2)×(w+2) walk with zero-padded borders), same line-buffer
counts, same bit widths (u16 vertical sums, u14 gaussian, u15 power). Each is
a free-running II=1 pipeline with a `valid` handshake and a `stall` input from
the FIFO.

## The hardware configuration

The RTL "improved" configuration is **(a) strict NMS + (j) half-pixel shift +
(f) bbox endpoints + (d) adaptive hysteresis + (h) max-perp-spread +
(i) border margin** — all six three-way (SW/HLS/RTL) bit-exact. (c) sub-pixel
NMS remains a SW/HLS-level refinement (rationale below). (a), (j) and (f)
translate directly; (d), (h) and (i) have hardware-shaped forms:

### (d) Adaptive hysteresis

`core/hyst_hist.v` (instantiated in `stage_edge`) is the RTL twin of
`kernels::AdaptiveLowTh`: a decayed 64-bin power histogram whose low threshold
is ~2× the 80th-percentile power (integer only: counts ×256, decay `v-=v>>8`,
percentile via `cum*5 >= total*4`). The NMS uses this LOW threshold; a
per-pixel `strong` bit (`power >= power_th`, the HIGH threshold) rides the
edge → feature → event path, and the back-end keeps a per-label `strong_cnt`,
rejecting a segment in the judge if `strong_cnt < hyst_strong_min`.

Two hardware-shaped choices, mirrored back into SW/HLS so all three stay
bit-exact: **(1) two-row lag** — row *m*'s threshold is `lowTh(H_{m-2})`,
computed from a snapshot over a full row (needs **width ≥ 64**; narrower test
images fall back to the fixed low threshold). **(2) per-frame clear** — the
histogram is global state that cannot self-clear through the flush rows, so
`frame_start` clears it; each frame cold-starts exactly like a fresh
`detect()` (guarded by a 2-frame Full-HD regression).

### (h) Max-perp-spread + (i) border margin

Both are **once-per-segment judge-level rejections** — no per-pixel or
labelling change — so both fold into the existing back-end at essentially
zero cost.

- **(h) curve rejection.** A curved arc bows off its chord, inflating the
  smaller eigenvalue of the *normalised* covariance; the aspect-ratio test
  alone misses short low-curvature arcs. SW rejects when
  `ev_min = ½(T−R)/N² > max_perp_spread²` (default 1). This is done sqrt-free
  **inside the existing 128-bit judge**: with `A := T − 2·mps²·N²`, reject iff
  `A > 0 && A² > R²`, reusing the aspect test's `T` and `R²`. In
  `judge_unit.v` it is two extra products time-multiplexed onto the same
  shared multiplier — a handful more cycles on the already-off-critical-path
  judge; unlike (c), it does **not** grow the datapath.
- **(i) border margin.** The 2×2 gradient biases the very edge of the frame,
  so a ring of spurious segments traces the border. Defined as a
  **bounding-box rejection**: drop a segment whose bbox reaches within
  `border` (=3) px of the frame — a pure integer compare on the record's own
  extremes, applied at record emission in `backend.v`. *(A per-pixel "skip
  labelling border pixels" form is incompatible with the RTL's
  `Interior ⇒ labelled` invariant; the bbox form removes the identical frame
  artifacts and is used uniformly in SW/HLS/RTL.)*

### (c) Sub-pixel NMS — SW/HLS reference only (out of RTL scope)

Improvement (c) fits a parabola through the three NMS-axis power samples of
each surviving edge pixel and shifts that pixel's moment contribution by the
vertex offset (±0.5 px, in 1/16-px units). It lives in `sweeplsd::detect()`
and the HLS C model but is deliberately NOT ported to this board RTL:

- **Invisible on the demo.** The overlay draws endpoints at half or quarter
  resolution with an integer Bresenham walker. Measured on a Full-HD corpus
  photo, enabling (c) moves endpoints by a mean of 0.064 px (max 0.43 px) and
  changes the accepted count by +1 — nothing on the screen changes.
- **It breaks the judge's 128-bit envelope.** The exact integer test
  `kRejN·T² ≤ kRejD·R²` is calibrated so both sides fit 128 bits (~u127 worst
  case). Sub-pixel accumulation scales every position by 16, growing the
  judge products to ~u143 — past 128 bits for ANY position scaling (even ×2
  overflows). Supporting it would mean ~160-bit arithmetic in the shared
  multiplier plus a wider event word and wider moment/record fields — a large
  datapath change for a sub-visible gain.

## Back-end FSM (from `../hls/src/backend.cpp`)

All state is the phase-1 memory map: 1024-label SoA table, tag-validated
label/feature rows, interior-x lists, touched lists + free-list ring (release
at `last_row ≤ y−2`). The semantic flow per event is the golden one: INGEST →
per interior x: GATHER (≤4 finds, each a BRAM pointer chase; measured chain
depth ≤ 1 on the corpus) → RESOLVE (create / adopt / MERGE) → ACCUMULATE →
CONTACT (endpoint touch → open or close a segment → JUDGE) → SCAVENGE at end
of row. The find loop, the `label0`/`label1` accumulation, the single-hop path
compression, and the merge survivor rule are unchanged from the golden model —
the shipped FSM differs only in *when* the bookkeeping around them happens:

- **Concurrent event ingest.** Ingestion runs as a parallel engine in the
  same clocked process: whenever an event is available it is filed into the
  feature banks / interior lists at 1 event/cycle, concurrently with
  labelling an earlier row (feature banks ×4 and interior-x lists ×3, cycling
  by row so written banks stay disjoint from read banks; the labeller starts
  row `proc_y` once `ingest_y ≥ proc_y+2`). The FIFO only backs up when the
  labeller falls > 3 rows behind the video.
- **GATHER: parallel-skip dispatch, fused end to end.** The four causal
  neighbours are visited in golden order NE, N, NW, W, but absent neighbours
  cost nothing (a combinational present-mask jumps straight to the next
  present one). The entry cycle both captures the right column and dispatches
  the first find; the W (left) neighbour is folded directly from the `w_sav`
  carry — provably already a root, so its 2-cycle find is skipped; and the
  terminal resolve is fused into the last gather action, which branches
  straight to accumulate/adopt/merge.
- **Fetch folded into the accumulate/contact states.** The successor pixel's
  feature/row-buffer column reads ride states the current pixel already pays
  for, so fetch costs **0** cycles on an adjacent successor and **1** on a
  non-adjacent one (the true port floor: three columns through one read port
  at 2-cycle latency leaves one un-hidden slot). All column/label carries are
  gated on the run-continuation `prev_x == px−1`, which is provably false
  after any label-exhaustion skip; the plain 3-state fetch path is kept as
  the row-entry and fallback flow and re-establishes every steady-state
  invariant.

### Throughput summary

The serial bookkeeping was removed lever by lever, each located with a
state-occupancy histogram over Full-HD frames, admitted bit-exact under the
standing gate, and re-checked in `synth_be` (final: 81.7 MHz, 0 failing
endpoints, critical path = the judge DSP; re-run it before a reflash if the
gather/fetch folds change):

| lever | what it removes | measured effect |
|---|---|---|
| gather parallel-skip | dead dispatch cycles on absent neighbours | 18.9 → 16.0 cy/interior (IMGP1033) |
| judge narrowing (below) | multiplier passes + CONT judge stall | back-end −6 % (IMGP1033) |
| INGEST fuse | pop/apply 2 states → 1 | −1 cy/event (−170 k cy, IMGP1033) |
| W-continuation fast-path | the W find on run continuations | −2 cy/fold (−82 k cy, IMGP1033) |
| gi=0 capture+dispatch fusion | 1 gather setup cycle per pixel | −1 cy/interior (−122 k cy, IMGP1033) |
| S_RB bubble removal | 1 fetch wait on adjacent pixels | FETCH −114 k cy (IMGP1077) |
| terminal-resolve fusion | 1 resolve-exit cycle per pixel | GATHER −192 k cy (IMGP1077) |
| fetch fold into S_ACC/S_CONT | the explicit fetch states | FETCH 517 k → 100 k cy (IMGP1077) |
| concurrent event ingest | serial between-row ingestion | ingest 277 k → 1.6 k cy (IMGP1077) |

Net: the worst live frame (IMGP1077) runs at **1.267 M back-end cycles
(6.06 cy/interior)** — ~51 % of the 1080p30 frame-clock budget, down from
2.137 M before the last four levers — and its FIFO peak occupancy is **1**:
events drain into the banks at line rate while the labeller works.

### Judge: one shared sequential multiplier

The exact integer test `361·T² ≤ 441·R²` needs 9 wide products. Rather than
the 79 DSPs the HLS version spends (Artix-7 87 %), phase 2 schedules every
product onto **one 36×36 pipelined multiplier (4 DSP48A1)** decomposed into
18×18 partial products — ~10³ segments/frame ≪ the 2.5 M cycle frame budget.
Total DSP48A1: judge 4 + accumulate (x², x·y, 11×11) ~3 ≈ **7 of 58**.

**The datapath is sized to the empirical worst case, not the field cap.**
`hls/tools/moment_probe` measured the largest moments the judge actually sees
over the corpus — n<2^12, Sx/Sy<2^22, Sxx/Syy/Sxy<2^33 (edges are ≤2 px wide,
so a component is a thin curve) — hence 361·T² / 441·R² < 2^102, and
`judge_unit.v` is sized to that (operands 48 bit, accumulator/compare 104
bit; the field-cap form would need 128). This is **exact, no rounding**: the
removed high bits are provably zero for any input within those bounds. Most
base products then have a zero high half, so the multiplier skips the hi·lo /
lo·hi / hi·hi passes when a half is zero — 1–2 passes instead of 4, which
also trims the back-end's wait-for-judge stall. Worst-case *legitimate* judge
latency measured 61+ cycles; the rescue watchdog window is 255 cycles.

Per-pixel moment accumulate stays combinational-narrow (all adders ≤ 41
bits), and the label table stores Σx²/Σy²/Σxy in 34 bits (probe-bounded
≤2^32; reads zero-extend, the accumulate write truncates losslessly), freeing
2 of the back-end's 18-Kb BRAM sites.

### Overflow reality check (RTL burst simulation)

The back-end labels *events*, so a maximally dense row can outrun it; the
event FIFO (2048 deep) absorbs bursts and, when saturated, sheds data events
rather than stalling live video. The canonical overflow tool is the
cycle-accurate burst testbench `rtl/tb/tb_backend_burst.v` — the real
`backend.v` + real `event_fifo.v` fed the golden event stream at 1080p30
pixel timing, validated to reproduce the exact golden record count when given
an oversized (drop-proof) FIFO. (The host co-sim `hls/tools/fifo_dropsim.cpp`
is a fast estimate only; figures it once produced are superseded by the RTL
numbers.)

With the shipped back-end, overflow is a burst limit, not an
average-throughput one, and it is almost gone: over the 150-photo corpus the
segment loss is **≈0.17 %**, all of it in one pathologically dense frame
(IMGP1032, true backlog 5,975 events); every other frame peaks at ≤ 346 FIFO
entries. At an **8192-deep FIFO the whole corpus is provably lossless**
(verified directly on IMGP1032; every no-drop trajectory is depth-invariant).
Corpus loss across the back-end levers: 52 % (first live build) → 4.8 % →
0.22 % → **0.17 %** at depth 2048 → **0 %** at 8192.

**Board caveat:** the FIFO's first-word-fall-through front is an asynchronous
read, so XST infers distributed (LUT) RAM — at depth 8192 that broke place &
route outright. The shipped core stays at 2048 until the FIFO is rebuilt as a
BRAM (sync-read) FIFO behind a small FWFT skid buffer — the remaining step to
carry the zero-drop result onto the LX45.

### Live robustness: marker reserve & hysteretic shedding

Two `event_fifo` policies protect live video (real HDMI content is noisier
than the PNG-derived vectors, and a live scene can hold the labeller behind
the video indefinitely):

- **EOR-marker reserve = 1152.** Losing a data event only thins detections;
  losing an end-of-row *marker* shears the row bookkeeping — no end-record is
  emitted, and the overlay (which swaps on the end-record) freezes on the
  previous frame's segments. The FIFO reserves capacity for markers first:
  1152 ≥ rows + EOF at 1080p (1081), so marker loss is structurally
  impossible; data capacity is the remaining 896 slots. A hard-full guard
  additionally drops ANY push at count==DEPTH — a sheared row recovers
  through the normal restart path, silent ring corruption does not.
- **Hysteretic shedding.** Dropping data events *individually* under
  saturation cuts every edge run into fragments of expected length (1−p)/p
  px — under `pix_th` = 16 for any sustained loss rate ≳ 6 % — so a saturated
  stretch of frame yields ZERO segments, not thinned ones (on the live board:
  a per-image "cut row" below which nothing was detected). Instead,
  `shedding` latches at `afull` and drops data down to a low watermark, then
  passes everything until `afull` trips again: same average loss, but the
  kept stretches are contiguous, so runs inside them still form segments —
  sustained overload thins instead of wiping. (Transient bursts shed slightly
  more than per-event thinning would, and near-vertical lines can still
  fragment inside a saturated band at depth 2048 — the deeper BRAM FIFO above
  is the remedy. Frames that never assert `afull` are bit-identical under
  either policy.)

### XST FSM-extraction mis-synthesis (why `-fsm_extract NO` is load-bearing)

The hardest bug of the bring-up: on dense scenes the live board lost all
detections below a per-image "cut row" — deterministic, load-dependent,
timing Score 0, RTL simulation clean. Per-pass UART telemetry showed ~92 % of
judge dispatches never receiving `j_done` on the board, and a **gate-level
simulation of the XST netlist** (netgen -sim + iverilog + unisims) reproduced
the board EXACTLY — proving the netlist deviates from the RTL functionally.
Bisected to synthesis: XST's FSM re-encoding mis-synthesizes the big back-end
FSM, and resynthesizing with **`-fsm_extract NO`** restores RTL-identical
gate-level behaviour. The option is set (with a warning comment) in the
`boards/atlys/` build scripts; do not remove it without re-running the
gate-level frame regression. The portable lesson: a deterministic,
load-dependent, board-only anomaly that RTL simulation cannot reproduce ⇒
simulate the *netlist* before blaming physics.

## Overlay (demo path, outside the verified core)

Segments finalised during frame N are drawn into a **half-resolution 1-bit
mask** (720p: 640×360 = 28.8 KiB; ×2 ping-pong = 57.6 KiB BRAM) by a Bresenham
FSM using the raw integer endpoints from the record (the sub-pixel projection
is cosmetic at demo scale); frame N+1 is displayed with mask N mixed in
(green). Endpoints map to mask cells as `(c+1)>>1` (`(c+2)>>2` at the FullHD
/4 scale) — the cell nearest the TRUE edge position `c+0.5`, since the 2×2
gradient samples at pixel corners (improvement (j); clamped at the
right/bottom edge). Total BRAM: core ~70 KiB + overlay ~58 KiB + video I/O ≈
140 KiB of 261 KiB — still no external memory.

## Board bring-up (completed)

The bring-up followed three staged milestones, each verified before the next:
(1) a known-good HDMI pass-through skeleton under ISE 14.7 (XAPP495 RX → TX
untouched, FPGA-served EDID), (2) rgb2gray + `sweep_core` inserted in the
recovered-clock domain, (3) the overlay and live tuning. The result is the
live demo described above (`boards/atlys/atlys_rx_top.v`, 1080p30 / 720p60);
build instructions are in `boards/atlys/README.md`. Programming via Digilent
Adept 2 (`djtgcfg prog`).

## Repository layout

```
rtl/
  DESIGN.md          — this file
  core/              — portable detector RTL (the deliverable)
  tb/                — Icarus Verilog testbenches (golden-vector parity)
  tools/             — dump_vectors.cpp (golden vector generator)
  boards/atlys/      — top level, HDMI PHY, UCF constraints, ISE build script
```
