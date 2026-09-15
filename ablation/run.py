"""Serial ablation scheduling using actual training plans/seeds; dry-run by default."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

from ablation.registry import ROOT, VARIANTS, CONTROLS, LEGACY, DATASETS, canonical, resolve_config, config_hash, states


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_once(path, payload):
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError(f"Protocol/config changed: {path}; use a new --suite")
    write_json(path, payload)


def git_info():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def option(command, key, default=None):
    return command[command.index(key) + 1] if key in command else default


def training_source(dataset, source=None, base_path=None):
    """Import runtime overrides too, not the stale hardcoded multi-seed defaults."""
    if source is None:
        candidates = sorted((ROOT / "workspaces/scheduled").glob("*/plan.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        source = next((p for p in candidates if dataset in json.loads(p.read_text()).get("datasets", [])), None)
    payload = json.loads(source.read_text()) if source is not None else {}
    jobs = [job for job in payload.get("jobs", []) if job["dataset"] == dataset]
    if payload.get("datasets") and dataset not in payload["datasets"]:
        raise ValueError(f"{dataset} is absent from seed/training source {source}")
    if jobs:
        commands = [job["command"] for job in jobs]
        config_paths = {option(command, "--config") for command in commands}
        if len(config_paths) != 1:
            raise ValueError("Training seeds use different configs")
        imported = Path(next(iter(config_paths)))
        overrides = {"--epochs": "epochs", "--early-stop-patience": "early_stop_patience",
                     "--per-gpu-batch-size": "per_gpu_batch_size", "--grad-accum-steps": "grad_accum_steps"}
        for flag in overrides:
            if len({option(command, flag) for command in commands}) != 1:
                raise ValueError(f"Training seeds use different {flag}")
    else:
        imported = (source.parent / "configs" / f"{dataset}.json") if source else ROOT / "configs/datasets" / f"{dataset}.json"
        if not imported.is_file():
            imported = ROOT / "configs/datasets" / f"{dataset}.json"
    path = base_path or imported
    base = json.loads(path.read_text())
    if jobs and base_path is None:
        for flag, key in overrides.items():
            value = option(commands[0], flag)
            if value is not None:
                base["train"][key] = int(value)
    seeds = payload.get("seeds", [base["seed"]])
    if jobs and {job["seed"] for job in jobs} != set(seeds):
        raise ValueError("Training plan seeds and jobs disagree")
    info = {"path": str(source.resolve()) if source else str(path.resolve()),
            "sha256": digest(source or path), "kind": "training_plan" if jobs else "seed_file_or_config",
            "base_config": str(path.resolve()), "base_sha256": digest(path), "imported_seeds": seeds}
    return base, seeds, info


def make_command(config_path, dataset, manifest, seed, nproc, smoke_steps=0, resume=None):
    command = [sys.executable]
    if nproc > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "--module", "ablation.entrypoint"]
    else:
        command += ["-m", "ablation.entrypoint"]
    command += ["train", "--dataset", dataset, "--config", str(config_path), "--manifest-dir", str(manifest),
                "--seed", str(seed), "--run-name", f"seed{seed}"]
    if smoke_steps:
        command += ["--smoke-steps", str(smoke_steps)]
    if resume:
        command += ["--resume", str(resume)]
    return command


def completed(run):
    required = ("final_summary.json", "evaluation/final/test/metrics.json", "evaluation/final/val/metrics.json",
                "evaluation/final/test/predictions.jsonl", "checkpoints/final_averaged.pth")
    return all((run / name).is_file() for name in required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", "--dataset", nargs="+", choices=DATASETS, required=True)
    parser.add_argument("--variants", "--experiments", nargs="+", choices=(*VARIANTS, *CONTROLS, *LEGACY), default=list(VARIANTS))
    parser.add_argument("--training-plan", "--seed-source", type=Path, dest="source")
    parser.add_argument("--base-config", type=Path, help="Common override; only with one dataset")
    parser.add_argument("--manifest-dir", type=Path, help="Only with one dataset")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit override; otherwise import actual training seeds")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--suite", default="ablation_v2")
    parser.add_argument("--output-root", type=Path, default=ROOT / "ablation/outputs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete last.pth; complete runs are always skipped")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--with-path-controls", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--fixed-mix-coefficient", type=float, default=0.5)
    parser.add_argument("--non-llm-config", type=Path)
    parser.add_argument("--subset-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if (args.base_config or args.manifest_dir) and len(args.datasets) != 1:
        parser.error("base-config/manifest-dir require exactly one dataset")
    if args.execute and args.dry_run:
        parser.error("Use either execute or dry-run")
    if args.nproc_per_node < 1 or args.smoke_steps < 0 or not 0 < args.subset_fraction < 1:
        parser.error("Invalid nproc/smoke/subset setting")
    if not args.suite or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.suite):
        parser.error("suite must be a plain name")
    variants = list(dict.fromkeys(canonical(name) for name in args.variants))
    if args.with_path_controls:
        variants = list(dict.fromkeys([*variants, *CONTROLS]))
    # Exact equivalence: strict direct already has the common downstream setup.
    if "path_direct_reference" in variants and "direct_only" not in variants:
        variants.append("direct_only")
    non_llm = json.loads(args.non_llm_config.read_text()) if args.non_llm_config else None
    source_files = [ROOT / "train.py", ROOT / "evaluate.py", *sorted((ROOT / "mmfnd").glob("*.py")),
                    *sorted((ROOT / "ablation").glob("*.py"))]
    code = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    for dataset in args.datasets:
        base, imported_seeds, source = training_source(dataset, args.source, args.base_config)
        seeds = args.seeds if args.seeds is not None else imported_seeds
        if not seeds or len(set(seeds)) != len(seeds) or any(type(x) is not int or not 0 <= x < 2**31 for x in seeds):
            raise ValueError("Seeds must be distinct integers in [0,2^31)")
        manifest = (args.manifest_dir or ROOT / "datasets" / dataset / "ready").resolve()
        suite = args.output_root.resolve() / dataset / args.suite
        protocol = {"version": 2, "dataset": dataset, "seed_source": source, "seeds": seeds,
                    "seed_override": args.seeds is not None, "source_sha256": code, "git": git_info(),
                    "nproc_per_node": args.nproc_per_node, "smoke_steps": args.smoke_steps,
                    "fixed_mix_coefficient": args.fixed_mix_coefficient,
                    "subset_definition": {"reference_seed": seeds[0], "fraction": args.subset_fraction,
                                          "rule": "Full validation quantiles, frozen on common test IDs"},
                    "manifest_dir": str(manifest),
                    "manifest_sha256": {name: digest(manifest / name) if (manifest / name).is_file() else None
                                        for name in ("dataset_manifest.json", "train.jsonl", "val.jsonl", "test.jsonl")}}
        write_once(suite / "protocol.json", protocol)
        registry_path = suite / "runs.json"
        runs = json.loads(registry_path.read_text()) if registry_path.exists() else {}
        planned = []
        for variant in variants:
            for seed in seeds:
                key = f"{variant}/seed{seed}"
                run = suite / variant / f"seed{seed}"
                if variant == "path_direct_reference":
                    runs[key] = {"dataset": dataset, "variant": variant, "seed": seed, "status": "alias",
                                 "alias_of": f"direct_only/seed{seed}", "reason": "Identical model, initialization and loss; reuse strict direct result"}
                    continue
                try:
                    cfg = resolve_config(base, variant, seed, suite / variant, args.fixed_mix_coefficient, non_llm)
                except ValueError as error:
                    runs[key] = {"dataset": dataset, "variant": variant, "seed": seed,
                                 "status": "not_applicable" if "not_applicable" in str(error) else "invalid_config", "reason": str(error)}
                    write_json(registry_path, runs)
                    raise
                cfg["ablation_provenance"] = protocol
                cfg["ablation_resolved_config_hash"] = config_hash(cfg)
                config_path = suite / "configs" / f"{variant}_seed{seed}.json"
                write_once(config_path, cfg)
                command = make_command(config_path, dataset, manifest, seed, args.nproc_per_node, args.smoke_steps)
                record = {"dataset": dataset, "variant": variant, "seed": seed, "run_dir": str(run),
                          "config": str(config_path), "config_hash": cfg["ablation_resolved_config_hash"],
                          "command": command, "status": "planned", "module_states": states(variant)}
                if key in runs and runs[key].get("config_hash") != record["config_hash"]:
                    raise ValueError(f"Run config changed: {key}")
                runs[key] = {**record, **runs.get(key, {})}
                planned.append((key, run, command))
        write_json(registry_path, runs)
        print(f"dataset={dataset} seed_source={source['path']} seeds={seeds} n={len(seeds)}", flush=True)
        for key, run, command in planned:
            if completed(run):
                saved_config = json.loads((run / "config.json").read_text())
                if saved_config.get("ablation_resolved_config_hash") != runs[key]["config_hash"]:
                    runs[key].update(status="invalid_result", reason="Completed run has a different configuration hash")
                    write_json(registry_path, runs)
                    raise ValueError(f"Completed run config mismatch: {key}")
                runs[key]["status"] = "completed"
                write_json(registry_path, runs)
                print(f"completed: {key}; skipped", flush=True)
                continue
            if args.resume and (run / "checkpoints/last.pth").is_file():
                command = [*command, "--resume", str(run / "checkpoints/last.pth")]
            print(shlex.join(command), flush=True)
            if not args.execute:
                continue
            if run.exists() and "--resume" not in command:
                runs[key].update(status="blocked", reason="Incomplete run exists; use --resume or a new suite")
                write_json(registry_path, runs)
                raise FileExistsError(runs[key]["reason"])
            if any(v is None for v in protocol["manifest_sha256"].values()):
                runs[key].update(status="failed", reason="Missing data manifests")
                write_json(registry_path, runs)
                raise FileNotFoundError(manifest)
            log = suite / "logs" / (key.replace("/", "_") + ".log")
            log.parent.mkdir(exist_ok=True)
            runs[key].update(status="running", command=command, started_at=datetime.now(timezone.utc).isoformat(), log=str(log))
            write_json(registry_path, runs)
            env = {**os.environ, "PYTHONHASHSEED": str(runs[key]["seed"]), "PYTHONUNBUFFERED": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
            with log.open("a") as stream:
                try:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                except OSError as error:
                    runs[key].update(status="failed", reason=f"Could not launch process: {error}")
                    write_json(registry_path, runs)
                    if not args.continue_on_error:
                        raise
                    continue
            success = result.returncode == 0 and (completed(run) or args.smoke_steps and (run / "checkpoints/smoke.pth").is_file())
            runs[key].update(status=("smoke_passed" if args.smoke_steps else "completed") if success else "failed",
                             returncode=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
            write_json(registry_path, runs)
            if not success and not args.continue_on_error:
                raise RuntimeError(f"Failed {key}; inspect {log}")


if __name__ == "__main__":
    main()
