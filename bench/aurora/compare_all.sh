#!/bin/bash
# Runs the whole SparseDirectSolver.jl side of the cuDSS comparison on Aurora as a chain of
# debug-scaling jobs (bench/aurora/compare.pbs; 1 h walltime, one running job per user), into
# bench/comparison/sds_aurora.csv. Login node, from the repository root, after the preparation
# in the header of compare.pbs:
#
#   nohup bench/aurora/compare_all.sh <project> > compare_all.log 2>&1 &
#
# Per feature of bench/comparison/cudss.csv it submits a job for the matrices that have a cuDSS
# row and no Aurora row yet, waits for it (qsub -W block=true) and repeats until none is left.
# compare.jl writes the CSV after every matrix, so a job stopped by the walltime keeps what it
# measured and the next one goes on from there; one job at a time, so the jobs share one CSV. A
# job that adds no row skips the matrix it was on (logged). Rerunning the script resumes.
# Optional: SDS_COMPARE_FEATURES="a b" restricts the features, SDS_COMPARE_EXTRA adds
# compare.jl options (e.g. "--samples=3").

set -u
export LC_ALL=C  # one collation for sort and comm
project=${1:?usage: compare_all.sh <project>}
dir=bench/comparison
csv=$dir/sds_aurora.csv
logs=$dir/aurora_logs
mkdir -p "$logs"

# matrices of feature $1 in file $2 (columns 2 and 4 of the CSVs written by compare.jl)
matrices() { [ -f "$2" ] && awk -F, -v f="$1" 'NR > 1 && $2 == f { print $4 }' "$2" | sort -u; }

features=${SDS_COMPARE_FEATURES:-$(awk -F, 'NR > 1 && !seen[$2]++ { print $2 }' "$dir/cudss.csv")}
for f in $features; do
    skip=""
    while :; do
        todo=$(comm -23 <(matrices "$f" "$dir/cudss.csv") <(matrices "$f" "$csv"))
        for m in $skip; do todo=$(grep -vxF "$m" <<< "$todo"); done
        [ -z "$todo" ] && break
        before=$( [ -f "$csv" ] && wc -l < "$csv" || echo 0)
        only=$(paste -sd, <<< "$todo")
        echo "$(date '+%F %T') $f: $(wc -l <<< "$todo") matrices: $only"
        log=$logs/$f-$(date +%s).log
        jid=$(qsub -W block=true -A "$project" -q debug-scaling -l walltime=01:00:00 -N "sds-$f" \
                   -o "$log" \
                   -v "SDS_JULIA_DEPOT=${JULIA_DEPOT_PATH:-},SDS_COMPARE_ARGS='--features=$f --only=$only ${SDS_COMPARE_EXTRA:-}'" \
                   bench/aurora/compare.pbs)
        if [ -z "$jid" ]; then  # refused (queue limit, maintenance): try again later
            echo "$(date '+%F %T') $f: qsub refused, retrying in 5 min"
            sleep 300
            continue
        fi
        echo "$(date '+%F %T') $f: job $jid finished"
        after=$( [ -f "$csv" ] && wc -l < "$csv" || echo 0)
        if [ "$after" -le "$before" ]; then
            # compare.jl prints "<feature> <matrix>" before measuring it: the last such line
            first=$(grep -aE "^$f +[^ ]" "$log" 2>/dev/null | tail -1 | awk '{ print $2 }')
            grep -qxF "$first" <<< "$todo" || first=$(head -1 <<< "$todo")
            echo "$(date '+%F %T') $f: no row added, skipping $first"
            skip="$skip $first"
        fi
    done
done
echo "$(date '+%F %T') done: $(($(wc -l < "$csv") - 1)) rows in $csv"
