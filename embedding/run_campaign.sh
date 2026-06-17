#!/usr/bin/env bash
# run_campaign.sh — self-driving experiment campaign for the r(n,d) matrix.
# Designed to run unattended for days/weeks in a screen session.
#
#   STAGE A: calibrate repulsion in R^3 (must reproduce: certify ring 5,
#            fail ring 6). Adaptive grid, up to 6 attempts.
#   STAGE B: R^4 + calibrated repulsion to ring 12 — the surface r(2,4)
#            measurement. (The headline number.)
#   STAGE C: R^3 control, no repulsion, to ring 9 — graph-realization depth.
#   STAGE D: R^4 no repulsion, fresh resumable run to ring 12 — graph record.
#
# Results: campaign_<date>/SUMMARY.md + per-run JSONL. After each stage and
# at exit (even on crash) it best-effort notifies via: ntfy.sh, mail, and a
# git commit+push of the campaign directory if it's inside a git repo.
#
# USAGE:
#   1. Pick a private topic name and subscribe at https://ntfy.sh/<topic>
#      from your phone/browser (works off-site, no account).
#   2. screen -dmS campaign env NTFY_TOPIC=<topic> MAILTO=you@western.edu \
#          bash run_campaign.sh
#   3. Watch your phone. Reattach in two weeks: screen -r campaign
#
# Requires probe_depth.py + generate_unit_distance_library.py in cwd.

set -uo pipefail

NTFY_TOPIC="${NTFY_TOPIC:-}"
MAILTO="${MAILTO:-}"
STAMP="$(date +%Y%m%d_%H%M)"
DIR="campaign_${STAMP}"
mkdir -p "$DIR"
SUMMARY="$DIR/SUMMARY.md"
echo "# r(n,d) campaign $STAMP" > "$SUMMARY"

notify() {
    local msg="$1"
    echo "[notify] $msg"
    echo "$(date '+%F %T')  $msg" >> "$DIR/notify.log"
    if [ -n "$NTFY_TOPIC" ]; then
        curl -fsS -m 20 -d "$msg" "https://ntfy.sh/$NTFY_TOPIC" \
            >/dev/null 2>&1 || true
    fi
    if [ -n "$MAILTO" ] && command -v mail >/dev/null 2>&1; then
        echo "$msg" | mail -s "[237 campaign] update" "$MAILTO" || true
    fi
}

push_results() {
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git add "$DIR" >/dev/null 2>&1 || true
        git commit -m "campaign $STAMP: $1" >/dev/null 2>&1 || true
        git push >/dev/null 2>&1 || true
    fi
}

# Parse a probe JSONL: prints "deepest_certified first_failed" (-1 if none)
probe_outcome() {
    python3 - "$1" << 'PYEOF'
import json, sys
deep, fail = -1, -1
try:
    for line in open(sys.argv[1]):
        r = json.loads(line)
        if r['certified']:
            deep = max(deep, r['ring'])
        elif fail < 0:
            fail = r['ring']
except FileNotFoundError:
    pass
print(deep, fail)
PYEOF
}

on_exit() {
    notify "campaign ended. summary: $(tail -n +2 "$SUMMARY" | tr '\n' ' | ')"
    push_results "final"
}
trap on_exit EXIT

# ---------------------------------------------------------------- STAGE A
notify "STAGE A start: calibrating repulsion in R3 (target: pass 5, fail 6)"
CAL_R=""; CAL_W=""
R=0.45; W=0.2
for attempt in 1 2 3 4 5 6; do
    TAG="$DIR/calA_${attempt}_r${R}_w${W}"
    rm -f "${TAG}.jsonl"
    python3 probe_depth.py --q 7 --dim 3 --max-ring 7 --retries 3 \
        --repulse --repulse-r "$R" --repulse-w "$W" \
        --out "$TAG" > "${TAG}.log" 2>&1
    read DEEP FAIL <<< "$(probe_outcome "${TAG}.jsonl")"
    echo "- calibration $attempt: r=$R w=$W -> deepest=$DEEP failed_at=$FAIL" \
        >> "$SUMMARY"
    notify "cal $attempt (r=$R w=$W): deepest=$DEEP fail=$FAIL"
    if [ "$DEEP" -eq 5 ] && [ "$FAIL" -eq 6 ]; then
        CAL_R=$R; CAL_W=$W
        echo "- CALIBRATED: repulse-r=$R repulse-w=$W reproduces r(2,3)=5" \
            >> "$SUMMARY"
        break
    elif [ "$DEEP" -ge 6 ]; then
        # too weak: strengthen
        R=$(python3 -c "print(round($R+0.1,2))")
        W=$(python3 -c "print(round($W*1.6,3))")
    else
        # failed at <=5: too strong, weaken
        R=$(python3 -c "print(round(max($R-0.1,0.2),2))")
        W=$(python3 -c "print(round($W*0.6,3))")
    fi
done
if [ -z "$CAL_R" ]; then
    echo "- calibration DID NOT CONVERGE; using last (r=$R w=$W) for stage B" \
        >> "$SUMMARY"
    notify "WARNING: calibration did not converge; B runs with r=$R w=$W"
    CAL_R=$R; CAL_W=$W
fi
push_results "stage A done"

# ---------------------------------------------------------------- STAGE B
notify "STAGE B start: R4 + repulse (r=$CAL_R w=$CAL_W) to ring 12 — the r(2,4) measurement"
TAGB="$DIR/B_q7_d4_rep"
python3 probe_depth.py --q 7 --dim 4 --max-ring 12 --max-verts 540000 \
    --retries 3 --repulse --repulse-r "$CAL_R" --repulse-w "$CAL_W" \
    --out "$TAGB" > "${TAGB}.log" 2>&1
read DEEP FAIL <<< "$(probe_outcome "${TAGB}.jsonl")"
echo "- STAGE B (surface r(2,4) proxy): deepest=$DEEP failed_at=$FAIL" \
    >> "$SUMMARY"
notify "STAGE B DONE: R4 surface-mode deepest=$DEEP fail=$FAIL  <-- headline"
push_results "stage B done"

# ---------------------------------------------------------------- STAGE C
notify "STAGE C start: R3 control (no repulse) to ring 9"
TAGC="$DIR/C_q7_d3_graph"
python3 probe_depth.py --q 7 --dim 3 --max-ring 9 --max-verts 30000 \
    --retries 3 --out "$TAGC" > "${TAGC}.log" 2>&1
read DEEP FAIL <<< "$(probe_outcome "${TAGC}.jsonl")"
echo "- STAGE C (R3 graph realization): deepest=$DEEP failed_at=$FAIL" \
    >> "$SUMMARY"
notify "STAGE C done: R3 graph deepest=$DEEP fail=$FAIL"
push_results "stage C done"

# ---------------------------------------------------------------- STAGE D
notify "STAGE D start: R4 no-repulse fresh resumable run to ring 12"
TAGD="$DIR/D_q7_d4_graph"
python3 probe_depth.py --q 7 --dim 4 --max-ring 12 --max-verts 540000 \
    --retries 3 --out "$TAGD" > "${TAGD}.log" 2>&1
read DEEP FAIL <<< "$(probe_outcome "${TAGD}.jsonl")"
echo "- STAGE D (R4 graph realization): deepest=$DEEP failed_at=$FAIL" \
    >> "$SUMMARY"
notify "STAGE D done: R4 graph deepest=$DEEP fail=$FAIL"
push_results "stage D done"

echo "- campaign complete $(date '+%F %T')" >> "$SUMMARY"
