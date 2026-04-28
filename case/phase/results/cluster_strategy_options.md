---
title: Cluster strategy options when seas_gpu/barak_lab queue is stuck
date: 2026-04-28
status: actionable_options_for_user
authoritative_data: squeue / sshare output at 2026-04-28 08:51 EDT
---

# Cluster strategy options — for when the queue stays blocked

The 6 phase PENDING jobs (`9008462, 9016098, 9029967, 9030513, 9033655,
9033656`) have all been PENDING (Priority) for >12 hours on
`--partition=seas_gpu --account=barak_lab`, with `START_TIME=N/A` —
Slurm has no scheduled time for them.

This document lists the recovery options, ordered by intrusiveness.
**The author of this doc (loop) is not authorized to take any of
these actions; this is for the user to decide.**

## Diagnosis

```
sshare -u hanlinzhang -A barak_lab    →  FairShare = 0.000838
                                         (very low; lab's allocation
                                          shared across users; user has
                                          consumed most of own slice)
```

The user has 9 RUNNING jobs on `kempner_sham_lab` account
(`kempner` and `kempner_h1` partitions) right now — so the *user* has
plenty of cluster capacity, but the *barak_lab phase jobs* are
stuck behind other-user `seas_gpu` priority.

## Option 1 — Wait

START_TIME is N/A but other user-of-mine jobs in `seas_gpu` queue
have estimated starts of 14:38–19:16 EDT (5–10 h from doc time).
Our phase jobs are below those in priority. Realistic ETA: probably
**after 19:00 today, possibly tomorrow**.

Cost: nothing, but the loop's binary-test predictions (P17, P18-P21,
P19-P22, P13-P23, P24) all sit pending until then.

## Option 2 — Resubmit on `kempner_h1` / `kempner_sham_lab`

The user already has 9 RUNNING jobs on `kempner_h1` with the
`kempner_sham_lab` account. That cluster has GPU slots available
to this account RIGHT NOW.

To resubmit one of our phase pilots there:

```bash
# Cancel the seas_gpu copy first (don't double-run)
scancel 9033655   # pilot_b4p0_g0p6
# Resubmit to kempner_h1
OVR='sweep.seeds=[1] plan.name=single plan.beta=4.0 plan.gamma=0.6'
B64=$(printf '%s' "$OVR" | base64 -w0)
sbatch \
  --job-name=phase-pilot-b4p0-g0p6 \
  --partition=kempner_h1 \
  --account=kempner_sham_lab \
  --time=02:00:00 \
  --export="ALL,RUN_NAME=pilot_b4p0_g0p6,OVERRIDES_B64=$B64" \
  case/phase/sweep.slurm
```

The 4 single-cell pilots (each 1 h compute) are the cheapest items to
move first — they would resolve P17/P18-P21/P19-P22/P13-P23 in 1–2 h
each. The 24-h `complement` and the 30k pilot are heavier; only worth
moving if the smaller probes succeed and we want to commit kempner
slots to the bigger experiments.

**Risks of moving to kempner_h1:**
- Uses `kempner_sham_lab` allocation (different lab); user may not
  want to spend that budget on this work.
- May compete with user's already-running protein jobs on kempner_h1.
- If kempner_h1 has shorter wall caps or different node specs (e.g.
  H100 vs A100), runtime estimates may differ from seas_gpu.

## Option 3 — Demote our long-wall pending jobs to free queue position

The 24-h `complement` (job 9008462) and 24-h `gamma_axis_b8p0`
(9016098) are likely the priority anchors holding back the smaller
2-h / 4-h pilots. Slurm scheduling considers wall time when packing.

```bash
scontrol update jobid=9008462 TimeLimit=12:00:00
scontrol update jobid=9016098 TimeLimit=06:00:00
```

This shortens their declared wall time, possibly improving backfill
chance. The complement has 28 cells × 51 min/cell ≈ 24 h actual, so
a 12-h cap would force it to time out at ~14 cells. Acceptable —
14 cells is enough to fit a regression on β > 1, which is what the
boundary mapping needs.

Risk: if user's other PENDING jobs (e.g. `8998035` etc.) jump our
priority, this doesn't help.

## Option 4 — Cancel and re-submit with `--qos=...` if available

Some clusters have `--qos=priority` or `--qos=interactive` for
short-wall, short-queue work.

```bash
sacctmgr list assoc user=hanlinzhang account=barak_lab format=qos
```

would show what QOS levels are available. If `interactive` exists,
short pilots could backfill there.

## Recommendation

In order of "does it solve the immediate problem":

1. **Option 2 for the 4 small pilots** (cheap; ~4 h compute total
   on kempner_h1; resolves 5 binary-test predictions). Requires user
   to authorize use of `kempner_sham_lab` allocation.

2. **Option 1 (wait) for the 24 h jobs** (complement, b8p0). They are
   not on the critical path for the loop's binary tests; they fill in
   the boundary map but the linear law is already validated by P15.

3. Defer Options 3 and 4 unless waiting > 24 h with no movement.

## Loop note (iter 26)

The loop will continue checking PENDING status each fire. If user
does Option 2, the new kempner_h1 jobs' `runs/<name>/` data will be
auto-discovered by `aggregate_report.py` and `cross_variant_scatter.py`
exactly as the seas_gpu jobs would have been. No code changes needed.
