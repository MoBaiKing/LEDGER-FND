"""CPU-only regression tests: never load a model or start a training process."""
import contextlib
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import scheduled_random4 as scheduler


class QueueTest(unittest.TestCase):
    def simulate(self, fail_seed=False, blocked_gpu=False):
        with tempfile.TemporaryDirectory(prefix="v3-queue-test-") as directory:
            root = Path(directory)
            jobs = []
            for dataset in scheduler.DATASETS:
                group = root / dataset
                group.mkdir()
                for seed in range(1, 5):
                    jobs.append(dict(dataset=dataset, seed=seed, run_dir=str(group / str(seed)),
                                     group_dir=str(group), log_path=str(group / f"seed{seed}.log"),
                                     command=["mock-training", dataset, str(seed)], status="pending"))
            plan = dict(directory=directory, project=directory, python="mock-python",
                        datasets=list(scheduler.DATASETS), gpu_uuids={str(i): f"GPU-{i}" for i in range(4)})
            state = dict(status="running", jobs=jobs, aggregates={})
            tick = [0]
            processes, launches, finishes = [], [], []
            peak = [0]

            class Process:
                def __init__(self, job=None, gpu=None, output=None):
                    self.pid = 100000 + len(processes)
                    self.job, self.gpu, self.output = job, gpu, output
                    self.done = False
                    self.code = 1 if job and fail_seed and job["dataset"] == "weibo21" and job["seed"] == 1 else 0
                    duration = 12 if job and job["dataset"] == "weibo21" and job["seed"] == 4 else 2
                    self.end = tick[0] + duration

                def poll(self):
                    if tick[0] < self.end:
                        return None
                    if not self.done:
                        self.done = True
                        if self.job:
                            finishes.append((self.job["dataset"], self.job["seed"], tick[0]))
                            if self.code == 0:
                                for name in ("final_summary.json", "evaluation/final/test/metrics.json"):
                                    target = Path(self.job["run_dir"]) / name
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    target.write_text("{}")
                        elif self.output:
                            Path(self.output).write_text("{}")
                    return self.code

            def popen(command, **kwargs):
                if command[0] == "mock-training":
                    dataset, seed = command[1], int(command[2])
                    job = next(j for j in jobs if j["dataset"] == dataset and j["seed"] == seed)
                    gpu = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"].removeprefix("GPU-"))
                    alive = [p for p in processes if p.job and p.poll() is None]
                    self.assertNotIn(gpu, [p.gpu for p in alive])
                    self.assertLess(len(alive), 4)
                    self.assertEqual(kwargs["env"]["WORLD_SIZE"], "1")
                    if blocked_gpu and gpu == 0:
                        self.assertGreaterEqual(tick[0], 6)
                    peak[0] = max(peak[0], len(alive) + 1)
                    launches.append((dataset, seed, tick[0], gpu))
                    process = Process(job=job, gpu=gpu)
                else:
                    self.assertEqual(Path(command[1]).name, "aggregate_seeds.py")
                    self.assertEqual(len(command[4:]), 4)
                    process = Process(output=command[3])
                processes.append(process)
                return process

            def sleep(seconds):
                tick[0] += seconds
                self.assertLess(tick[0], 100, "Queue failed to make progress")

            def probe(uuids):
                return {0} if blocked_gpu and tick[0] < 6 else set()

            with contextlib.redirect_stdout(io.StringIO()):
                scheduler.run_queue(plan, state, lambda s: None, probe=probe, popen=popen, sleep=sleep)
            self.assertEqual([(d, s) for d, s, _, _ in launches],
                             [(d, s) for d in scheduler.DATASETS for s in range(1, 5)])
            self.assertEqual(peak[0], 4)
            first_gossip = min(t for d, s, t, g in launches if d == "gossipcop")
            last_weibo21 = max(t for d, s, t in finishes if d == "weibo21")
            self.assertLess(first_gossip, last_weibo21, "Must refill across dataset boundaries")
            self.assertEqual(state["pending_count"], 0)
            self.assertEqual(state["active_count"], 0)
            self.assertEqual(state["status"], "completed_with_errors" if fail_seed else "completed")
            for dataset in scheduler.DATASETS:
                expected = "incomplete" if fail_seed and dataset == "weibo21" else "completed"
                self.assertEqual(state["aggregates"][dataset]["status"], expected)

    def test_all_success_fifo_and_immediate_refill(self):
        self.simulate()

    def test_failed_seed_does_not_block_other_datasets(self):
        self.simulate(fail_seed=True)

    def test_busy_gpu_is_not_used_until_free(self):
        self.simulate(blocked_gpu=True)

    def test_cancelled_timer_does_not_start_training(self):
        with tempfile.TemporaryDirectory(prefix="v3-timer-test-") as directory:
            root = Path(directory)
            plan = dict(start_at=(datetime.now(scheduler.TZ) + timedelta(hours=1)).isoformat(), jobs=[{}])
            scheduler.write_json(root / "plan.json", plan)
            scheduler.write_json(root / "status.json", dict(status="scheduled", jobs=[dict(status="pending")]))
            def interrupt(seconds):
                signal.raise_signal(signal.SIGTERM)
            with patch.object(scheduler.time, "sleep", side_effect=interrupt), \
                    patch.object(scheduler, "run_queue") as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(InterruptedError):
                    scheduler.run_plan(root / "plan.json")
                run.assert_not_called()
            state = json.loads((root / "status.json").read_text())
            self.assertEqual(state["status"], "cancelled")
            self.assertEqual(state["jobs"][0]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main(verbosity=2)
