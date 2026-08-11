#!/bin/sh
# Tier-2 RTL leg over the LIU4K-FHD corpus: for every image, generate the
# --improved (HW configuration) golden vectors, run the tb_sweep_core parity
# testbench (2 frames, full core), record PASS/FAIL, and delete the vectors
# (53 MB/set). 8-way parallel like run_rtl150.sh. Resumable: images with a
# .done marker are skipped. This is the RTL leg of the corpus-wide
# SW==HLS==RTL bit-exactness verification.
#   usage: [DUMP=path/to/sweeplsd_dump_vectors] run_liu4k_t2.sh [imgdir] [outdir]
# imgdir defaults to the 1080p rung produced by bench/liu4k/preprocess.py.
set -u
cd "$(dirname "$0")/.."   # rtl/

OSS=${OSS_CAD_SUITE:-/e/dev/claude/tools/oss-cad-suite}
PATH="$OSS/bin:$OSS/lib:$PATH"
export PATH

IMGDIR=${1:-../data/liu4k_bench/1080}
OUT=${2:-tb/liu4k_t2}
DUMP=${DUMP:-../build/rtl/sweeplsd_dump_vectors.exe}
mkdir -p "$OUT"

JOBS=8
running=0
for img in "$IMGDIR"/*.png; do
    name=$(basename "$img" .png)
    [ -f "$OUT/$name.done" ] && continue
    (
        d="$OUT/vec_$name"
        mkdir -p "$d"
        if ! "$DUMP" "$d" --improved "$img" > "$OUT/$name.log" 2>&1; then
            mv "$OUT/$name.log" "$OUT/$name.FAIL"; rm -rf "$d"; exit 1
        fi
        # drop the always-dumped procedural sets so only the photo runs
        rm -f "$d"/lines64_imp_meta.txt "$d"/noise64_imp_meta.txt "$d"/tiny16_imp_meta.txt
        if VECDIR="$d" sh tb/run_tb.sh tb_sweep_core core/*.v >> "$OUT/$name.log" 2>&1 \
           && grep -q "PASS" "$OUT/$name.log"; then
            mv "$OUT/$name.log" "$OUT/$name.done"
        else
            mv "$OUT/$name.log" "$OUT/$name.FAIL"
        fi
        rm -rf "$d"
    ) &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
done
wait
pass=$(ls "$OUT"/*.done 2>/dev/null | wc -l)
fail=$(ls "$OUT"/*.FAIL 2>/dev/null | wc -l)
echo "LIU4K Tier-2 RTL leg: $pass PASS, $fail FAIL"
[ "$fail" -eq 0 ]
