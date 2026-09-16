# Mounted four-seed GPU suite

- Start: **2026-09-16 01:00:00 Asia/Shanghai**.
- Live tmux session: `masked_r1_0100`; scheduler PID at mounting: `4073591`.
- Plan: `/data/dyl/sotamodelv3_mask/workspaces/scheduled/masked_r1_20260916_0100/plan.json`.
- Live status: same directory, `status.json`; scheduler log: `scheduler.log`.
- Seeds: **834958859, 1729822690, 1487209254, 1459867472**. OS CSPRNG, distinct from previous local suite seeds, identical paired seed list on each dataset.
- FIFO dataset order: GossipCop → Weibo21 → Weibo; 12 complete seed pipelines. Each free card receives one seed. When one finishes/fails, that card can accept the next pending job without waiting for the other cards.
- Each seed: three sequential reference folds → independent target construction → R1 student → validation mechanisms → explicitly frozen final test. A failure is preserved as failed; a dataset with any failed seed is marked incomplete and is not summarized as four successful seeds.
- Epoch limit **30**, patience **8**, original per-device batch **1**. Single-GPU accumulation is **32 for GossipCop**, **16 for Weibo21/Weibo**, preserving the baseline default four-rank effective batches. These budgets are starting settings, not validated optima.
- No GPU query or allocation occurs during the timer wait. At 01:00 the scheduler checks current GPU processes/UUIDs and waits for free cards; it never kills someone else's process.
- Startup checks code, scheduled config and dataset-manifest hashes. Changes fail closed. Each reference run also hashes actual pretrained/processor/image bytes before fitting.
- Actual seed-specific fold preflight: `planned_fold_preflight.json` beside the plan; all 36 outer/inner fold partitions were checked on CPU.
- At delivery status was `scheduled`, active GPU jobs **0**. This is not a completed GPU experiment.

```bash
# Inspect without allocating GPU:
cat /data/dyl/sotamodelv3_mask/workspaces/scheduled/masked_r1_20260916_0100/status.json
tmux attach -t masked_r1_0100
# Cancel only this scheduler if later required (not executed during implementation):
kill -TERM 4073591
```

The scheduler's test invocation is explicitly `--stage test --frozen`; the ordinary `all` stage remains validation-only. The plan records exact commands, paths, seeds, model/config/data identity and per-job status. No earlier ordinary-v3 or legacy-masked queue was replaced or cancelled.
