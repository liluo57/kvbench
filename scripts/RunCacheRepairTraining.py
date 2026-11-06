"""Durable full-training supervisor: CPU preparation, 30-minute GPU wait, train.

Uses only the standard library until child commands run. Never preempts another
GPU process. SIGINT/SIGTERM forwards to the active child for a clean checkpoint.
Restart this command with the same job directory to continue a prepared run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def merge_inventory(sources, destination):
    """Include both static inputs and the actual 640 frozen evaluation inputs."""
    destination = Path(destination)
    if (destination / "manifest.json").exists():
        return
    destination.mkdir(exist_ok=True)
    connection = sqlite3.connect(destination / "contents.sqlite")
    connection.executescript("DROP TABLE IF EXISTS contents; DROP TABLE IF EXISTS refs;"
                            "CREATE TABLE contents(hash TEXT PRIMARY KEY,text TEXT NOT NULL);"
                            "CREATE TABLE refs(hash TEXT,workload TEXT,record_id TEXT,kind TEXT);"
                            "CREATE INDEX refs_hash ON refs(hash);")
    manifests, coverage = [], {}
    for source in sources:
        source = Path(source)
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        for item in manifest.get("files", []):
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError(f"Evaluation source changed: {item['path']}")
        coverage.update(manifest.get("coverage", {}))
        with sqlite3.connect(source / "contents.sqlite") as original:
            connection.executemany("INSERT OR IGNORE INTO contents VALUES (?,?)", original.execute("SELECT hash,text FROM contents"))
            connection.executemany("INSERT INTO refs VALUES (?,?,?,?)", original.execute("SELECT hash,workload,record_id,kind FROM refs"))
        manifests.append({"path": str(source.resolve()), "manifest": manifest,
                          "sqlite_sha256": sha256_file(source / "contents.sqlite")})
    connection.commit()
    count = connection.execute("SELECT COUNT(*) FROM contents").fetchone()[0]
    connection.close()
    atomic_json(destination / "manifest.json", {"sources": manifests, "coverage": coverage,
                "content_sha256": sha256_file(destination / "contents.sqlite"), "content_units": count})


def gpu_snapshot():
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True)
    busy = {line.split(",")[0].strip() for line in processes.stdout.splitlines() if line.strip()}
    snapshots = []
    for line in query.stdout.splitlines():
        index, uuid, total, used, utilization = [v.strip() for v in line.split(",")]
        snapshots.append({"index": int(index), "uuid": uuid, "total_mib": int(total),
                          "used_mib": int(used), "utilization": int(utilization), "has_compute_process": uuid in busy})
    return snapshots


def available_gpus(snapshot, allowed=None):
    return [g["index"] for g in snapshot if (allowed is None or g["index"] in allowed)
            and g["total_mib"] >= 44000 and g["used_mib"] <= 1024 and g["utilization"] <= 5
            and not g["has_compute_process"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=".cache/cacherepair-qwen-env/bin/python")
    parser.add_argument("--evaluation-inventory", action="append", required=True)
    parser.add_argument("--allowed-gpus", help="Physical IDs, otherwise any two free L40-class GPUs")
    parser.add_argument("--poll-seconds", type=int, default=1800)
    parser.add_argument("--candidates", type=int, default=150000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.poll_seconds < 1800:
        parser.error("GPU polling must be at least 1800 seconds")
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    job = Path(args.job).resolve()
    job.mkdir(parents=True, exist_ok=True)
    lock = (job / "supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This training supervisor is already running")
    plan = {**vars(args), "job": str(job), "config_sha256": sha256_file(args.config), "initialization": "fresh"}
    plan_path = job / "launch-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing job launch plan differs")
    atomic_json(plan_path, plan)
    source_paths = sorted((root / "methods/CacheRepair/qwen38").glob("*.py")) + [
        root / "methods/CacheRepair/backend/cacherepair/train.py",
        root / "methods/CacheRepair/backend/cacherepair/model.py", Path(__file__).resolve()]
    source_hashes = {str(path): sha256_file(path) for path in source_paths}
    source_manifest = job / "source-provenance.json"
    if source_manifest.exists() and json.loads(source_manifest.read_text()) != source_hashes:
        raise ValueError("Queued job implementation changed; inspect before relaunching")
    atomic_json(source_manifest, source_hashes)
    atomic_json(job / "supervisor.json", {"pid": os.getpid(), "job": str(job)})
    stop = threading.Event()
    child = None

    def status(state, **details):
        event = {"time": datetime.now(timezone.utc).isoformat(), "status": state, **details}
        atomic_json(job / "job-status.json", event)
        with (job / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    def interrupt(number, frame):
        stop.set()
        if child is not None and child.poll() is None:
            child.send_signal(number)

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupt)

    def run(stage, command, env=None):
        nonlocal child
        if stop.is_set():
            return False
        if sha256_file(args.config) != plan["config_sha256"] or any(sha256_file(p) != h for p, h in source_hashes.items()):
            raise ValueError("Configuration or implementation changed while the job was queued")
        status(stage, command=command)
        with (job / f"{stage}.log").open("a", buffering=1) as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            code = child.wait()
            child = None
        if stop.is_set():
            status("interrupted", stage=stage, exit_code=code)
            return False
        if code:
            raise RuntimeError(f"{stage} failed with exit code {code}; see {job / (stage + '.log')}")
        return True

    try:
        evaluation, dataset, training = job / "evaluation-inventory", job / "data", job / "training"
        if not (dataset / "dataset.json").exists():
            status("preparing_evaluation_inventory")
            merge_inventory(args.evaluation_inventory, evaluation)
            if not run("preparing_data", [args.python, "-u", "-m", "methods.CacheRepair.qwen38.full_data", "build",
                "--model", args.model, "--cache", args.cache, "--evaluation-inventory", str(evaluation),
                "--output", str(dataset), "--candidates", str(args.candidates), "--workers", str(args.workers)]):
                return
        settings = json.loads(Path(args.config).read_text())
        actual = json.loads((dataset / "dataset.json").read_text())
        if actual["sizes"] != {"train": settings["data"]["train_examples"], "validation": settings["data"]["validation_examples"]}:
            raise ValueError("Prepared split sizes do not match full training config")
        if (training / "status.json").exists() and json.loads((training / "status.json").read_text())["status"] == "completed":
            status("completed", training=str(training))
            return
        allowed = set(map(int, args.allowed_gpus.split(","))) if args.allowed_gpus else None
        gpu_locks = []
        while not stop.is_set():
            try:
                snapshot = gpu_snapshot()
                free = available_gpus(snapshot, allowed)
            except (subprocess.SubprocessError, ValueError) as error:
                status("waiting_for_gpu", error=repr(error), next_check_seconds=args.poll_seconds)
                stop.wait(args.poll_seconds)
                continue
            if len(free) >= 2:
                for index in free:
                    handle = Path(f"/tmp/kvbench-cacherepair-gpu-{index}.lock").open("a")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        continue
                    gpu_locks.append((index, handle))
                    if len(gpu_locks) == 2:
                        break
                if len(gpu_locks) == 2:
                    break
                for _, handle in gpu_locks:
                    handle.close()
                gpu_locks = []
            status("waiting_for_gpu", snapshot=snapshot, next_check_seconds=args.poll_seconds)
            stop.wait(args.poll_seconds)
        if stop.is_set():
            status("interrupted", stage="waiting_for_gpu")
            return
        physical = [index for index, _ in gpu_locks]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, physical)),
                       "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
        atomic_json(job / "gpu-assignment.json", {"physical_ids": physical, "logical_ids": [0, 1]})
        common = ["--model", args.model, "--dataset", str(dataset), "--config", args.config, "--gpus", "0,1"]
        if not (job / "preflight.json").exists():
            if not run("preflight", [args.python, "-u", "-m", "methods.CacheRepair.qwen38.preflight", *common,
                                     "--output", str(job / "preflight.json")], env=environment):
                return
        command = [args.python, "-u", "-m", "methods.CacheRepair.qwen38.train", *common,
                   "--output", str(training), "--checkpoint-steps", "250", "--keep-restarts", "2"]
        if training.exists():
            command.extend(["--resume", "auto"])
        if not run("training", command, env=environment):
            return
        result = json.loads((training / "status.json").read_text())
        status(result["status"], training=str(training))
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
