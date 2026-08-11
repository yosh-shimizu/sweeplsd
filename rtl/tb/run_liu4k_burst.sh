#!/bin/sh
# §6 overload replacement: ground-truth RTL burst sim (tb_backend_burst =
# real backend.v + real event_fifo.v at 1080p30 pixel timing, drop_mode=1)
# over the LIU4K-FHD corpus in the hardware configuration. Per image:
# generate golden vectors, keep events/records/meta, simulate at the given
# FIFO depth, record "golden=<n> RESULT ..." per image, keep the per-row
# burst CSV, delete the bulky stage hex. Resumable via .res markers. This is
# the corpus-wide overload-limit measurement (event-FIFO loss vs depth).
#   usage: [DEPTH_AW=11] [DUMP=path/to/sweeplsd_dump_vectors] \
#          run_liu4k_burst.sh [imgdir] [outdir]
# imgdir defaults to the 1080p rung produced by bench/liu4k/preprocess.py.
set -u
cd "$(dirname "$0")/.."   # rtl/

OSS=${OSS_CAD_SUITE:-/e/dev/claude/tools/oss-cad-suite}
PATH="$OSS/bin:$OSS/lib:$PATH"
export PATH

IMGDIR=${1:-../data/liu4k_bench/1080}
OUT=${2:-tb/liu4k_burst}
DEPTH_AW=${DEPTH_AW:-11}   # 11 = shipped 2048
DUMP=${DUMP:-../build/rtl/sweeplsd_dump_vectors.exe}
mkdir -p "$OUT"

JOBS=8
running=0
for img in "$IMGDIR"/*.png; do
    name=$(basename "$img" .png)
    [ -f "$OUT/$name.res" ] && continue
    (
        d="$OUT/vec_$name"
        mkdir -p "$d"
        if ! "$DUMP" "$d" --improved "$img" > "$OUT/$name.log" 2>&1; then
            mv "$OUT/$name.log" "$OUT/$name.FAIL"; rm -rf "$d"; exit 1
        fi
        rm -f "$d"/lines64_imp_* "$d"/noise64_imp_* "$d"/tiny16_imp_* \
              "$d"/*_gray.hex "$d"/*_gauss.hex "$d"/*_power.hex \
              "$d"/*_dir.hex "$d"/*_edge.hex "$d"/*_feat.hex
        golden=$(wc -l < "$d/${name}_imp_records.hex")
        if iverilog -g2005 -o "$OUT/$name.vvp" \
              -DIMG_W=1920 -DIMG_H=1080 -DPIX_TH=15 \
              -DHYST_ON=1 -DHYST_MIN=3 -DMPS_2SQ=2 -DDEPTH_AW="$DEPTH_AW" \
              "-DVEC=\"$d/${name}_imp\"" \
              tb/tb_backend_burst.v core/event_fifo.v core/backend.v core/judge_unit.v \
              2>> "$OUT/$name.log" \
           && vvp "$OUT/$name.vvp" >> "$OUT/$name.log" 2>&1 \
           && grep -q "^RESULT" "$OUT/$name.log"; then
            res=$(grep "^RESULT" "$OUT/$name.log" | tail -1)
            mv "$d/${name}_imp_burst.csv" "$OUT/$name.burst.csv" 2>/dev/null
            echo "$name golden=$golden $res" > "$OUT/$name.res"
        else
            mv "$OUT/$name.log" "$OUT/$name.FAIL"
        fi
        rm -f "$OUT/$name.vvp"
        rm -rf "$d"
    ) &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
done
wait

# Aggregate: corpus golden vs emitted, loss %, images with drops, peak occ.
awk -v depth="$DEPTH_AW" '
{
    for (i = 1; i <= NF; ++i) {
        if ($i ~ /^golden=/)      { g = substr($i, 8) + 0 }
        if ($i ~ /^rec_emitted=/) { e = substr($i, 13) + 0 }
        if ($i ~ /^total_drop=/)  { d = substr($i, 12) + 0 }
        if ($i ~ /^peak_occ=/)    { p = substr($i, 10) + 0 }
    }
    G += g; E += e; n += 1
    if (g > e) { lossy += 1; printf "  lossy: %s golden=%d emitted=%d (-%d)\n", $1, g, e, g - e }
    if (p > maxp) maxp = p
    if (g == e && p > maxp_nodrop) maxp_nodrop = p
}
END {
    loss_pct = 0
    if (G > 0) loss_pct = 100.0 * (G - E) / G
    printf "LIU4K burst @depth_aw=%s: %d images, golden %d -> emitted %d, loss %d (%.3f%%), %d lossy images, peak_occ max %d (max among lossless %d)\n",
           depth, n, G, E, G - E, loss_pct, lossy, maxp, maxp_nodrop
}' "$OUT"/*.res
fail=$(ls "$OUT"/*.FAIL 2>/dev/null | wc -l)
[ "$fail" -eq 0 ] || { echo "$fail FAILURES"; exit 1; }
