"""Run a real, isolated vLLM training/restart smoke across 6/4/2 GPUs.

This utility uses the production Qwen3.8 train CLI and its committed optimizer,
RNG, cursor, and repairer checkpoint implementation. It deliberately writes the
RunFast six-gpu-smoke.json gate only after all child runs and checks pass.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import heapq
import json
import math
import os
import pwd
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB_ROOT = ROOT / "outputs/cacherepair-qwen38"
BATCH_SIZE = 4
LONG_MIN_TOKENS = 16_000
SMOKE_ROWS = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_key(chunks) -> str:
    payload = json.dumps(chunks, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_physical_gpus(value: str) -> tuple[int, ...]:
    try:
        ids = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("physical GPU IDs must be comma-separated integers") from exc
    if len(ids) != 6 or len(set(ids)) != 6 or min(ids) < 0:
        raise ValueError("the restart smoke requires six distinct physical GPU IDs")
    return ids


def select_smoke_rows(train_jsonl: Path, *, long_min_tokens: int = LONG_MIN_TOKENS,
                      short_count: int = SMOKE_ROWS - 1):
    """Pick one longest real row and distinct shortest rows, in stable file order."""
    longest = None
    # Keep a bounded heap of candidate short rows; input data can be multi-GB.
    shortest = []
    seen_short_keys = set()
    row_count = 0
    with Path(train_jsonl).open() as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            chunks = row.get("chunks")
            if not isinstance(chunks, list) or len(chunks) < 2 or any(not c for c in chunks):
                continue
            token_count = sum(map(len, chunks))
            if token_count < 1:
                continue
            row_count += 1
            candidate = (token_count, index, row)
            if longest is None or (token_count, -index) > (longest[0], -longest[1]):
                longest = candidate
            key = row_key(chunks)
            if key in seen_short_keys:
                continue
            seen_short_keys.add(key)
            # Reverse-sort retained tuples by using a max-heap over token count
            # and source index. Keep the globally shortest distinct rows.
            item = (-token_count, -index, key, row)
            if len(shortest) < short_count:
                heapq.heappush(shortest, item)
            elif item > shortest[0]:
                removed = heapq.heapreplace(shortest, item)
                seen_short_keys.discard(removed[2])
    if longest is None or longest[0] < long_min_tokens:
        raise ValueError(f"No real training row reaches {long_min_tokens} tokens")
    long_key = row_key(longest[2]["chunks"])
    candidates = sorted(shortest, key=lambda item: (-item[0], -item[1]))
    candidates = [item for item in candidates if item[2] != long_key]
    if len(candidates) < short_count:
        raise ValueError("Could not find enough distinct short rows for the smoke batch")
    short_rows = [item[3] for item in candidates[:short_count]]
    if any(sum(map(len, row["chunks"])) >= long_min_tokens for row in short_rows):
        raise ValueError("Short-row selection unexpectedly includes a long example")
    return [longest[2], *short_rows], {
        "source_rows_scanned": row_count,
        "source_indices": [longest[1], *[-item[1] for item in candidates[:short_count]]],
        "tokens": [sum(map(len, row["chunks"])) for row in [longest[2], *short_rows]],
        "row_keys": [row_key(row["chunks"]) for row in [longest[2], *short_rows]],
    }


def place_long_row_in_first_batch(rows, selection, *, seed: int, batch_size: int = BATCH_SIZE):
    """Reorder only the selected smoke subset so 6-GPU phase sees the long row."""
    if not rows or len(rows) != len(selection["row_keys"]):
        raise ValueError("Smoke selection metadata does not match its rows")
    long_key = selection["row_keys"][0]
    if row_key(rows[0]["chunks"]) != long_key:
        raise ValueError("The first selected row must be the longest prefill example")
    order = list(range(len(rows)))
    random.Random(seed * 1_000_003).shuffle(order)
    first_index = order[0]
    if first_index >= batch_size:
        raise ValueError("Deterministic first batch does not contain its first shuffled row")
    if first_index:
        rows = list(rows)
        rows[0], rows[first_index] = rows[first_index], rows[0]
        for field in ("source_indices", "tokens", "row_keys"):
            selection[field][0], selection[field][first_index] = (
                selection[field][first_index], selection[field][0])
    if row_key(rows[order[0]]["chunks"]) != long_key:
        raise AssertionError("Long smoke row was not moved into the first training example")
    return rows, selection


def expected_training_order(rows, seed: int, epochs: int):
    keys = [row_key(row["chunks"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Smoke rows must have unique prompt occurrences for order auditing")
    result = []
    for epoch in range(epochs):
        order = list(range(len(rows)))
        random.Random(seed * 1_000_003 + epoch * 10_007).shuffle(order)
        result.extend(keys[index] for index in order)
    return result


def validate_prompt_events(events, rows, *, seed: int, epochs: int):
    """Check consumed request keys and token lengths against shuffled row occurrences."""
    expected_keys = expected_training_order(rows, seed, epochs)
    actual_keys = [event.get("pair_key") for event in events]
    if actual_keys != expected_keys:
        raise ValueError("vLLM prefill request order differs from the deterministic epoch cursor")
    tokens_by_key = {
        row_key(row["chunks"]): sum(map(len, row["chunks"])) for row in rows
    }
    if len(tokens_by_key) != len(rows):
        raise ValueError("Smoke rows must have unique prompt keys for token-length auditing")
    actual_tokens = [event.get("tokens") for event in events]
    expected_tokens = [tokens_by_key[key] for key in expected_keys]
    if actual_tokens != expected_tokens:
        raise ValueError("vLLM prefill token lengths differ from the matching prompt rows")
    return True


def validate_restart_state(state, *, expected_step: int, expected_epoch: int,
                           expected_offset: int, train_size: int, cuda_rng_count: int):
    if state.get("format") != "kvbench-qwen38-trainer-v1":
        raise ValueError("Restart checkpoint has an unknown trainer-state format")
    if (state.get("step"), state.get("epoch"), state.get("offset")) != (
            expected_step, expected_epoch, expected_offset):
        raise ValueError("Restart step/epoch/offset differs from the smoke cursor")
    if state["offset"] > train_size or state["offset"] % BATCH_SIZE:
        raise ValueError("Restart cursor is outside a complete optimizer batch boundary")
    if not isinstance(state.get("optimizer"), dict) or not state["optimizer"].get("state"):
        raise ValueError("Restart checkpoint has no populated AdamW state")
    torch_rng = state.get("torch_rng")
    if (not isinstance(state.get("python_rng"), tuple) or torch_rng is None
            or (hasattr(torch_rng, "numel") and torch_rng.numel() == 0)):
        raise ValueError("Restart checkpoint is missing Python/Torch RNG state")
    if not isinstance(state.get("cuda_rng"), list) or len(state["cuda_rng"]) != cuda_rng_count:
        raise ValueError("Restart checkpoint CUDA RNG topology differs from the trainer devices")
    if not math.isfinite(float(state.get("horizon", 0))) or state["horizon"] <= expected_step:
        raise ValueError("Restart learning-rate horizon is not valid for continued training")
    if not state.get("identity"):
        raise ValueError("Restart checkpoint lacks its immutable run identity")
    return True


def validate_gate_checks(checks):
    required = {
        "six_gpu_update", "four_gpu_exact_resume_update", "two_gpu_exact_resume_update",
        "optimizer_state_restored", "rng_state_present", "progress_cursor_restored",
        "model_updated", "rms_unchanged", "init_checkpoint_unchanged",
        "finite_losses", "prompt_order_verified", "source_unchanged",
    }
    missing = sorted(required - set(checks))
    if missing:
        raise ValueError(f"Smoke checks are missing required results: {missing}")
    failed = sorted(key for key in required if checks[key] is not True)
    if failed:
        raise ValueError(f"Smoke checks did not pass: {failed}")
    return True


def _atomic_json(path: Path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_smoke_dataset(source_manifest: Path, work_dir: Path, rows, config_path: Path):
    work_data = work_dir / "data"
    work_data.mkdir(parents=True, exist_ok=False)
    train_path = work_data / "train.jsonl"
    validation_path = work_data / "validation.jsonl"
    with train_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    # A short real prompt suffices as the validation fixture; stop-after-steps
    # prevents reaching the epoch-end validation callback during this smoke.
    with validation_path.open("x") as stream:
        stream.write(json.dumps(rows[1], separators=(",", ":")) + "\n")
    manifest = json.loads(Path(source_manifest).read_text())
    manifest = {
        "format": "cacherepair-vllm-restart-smoke-v1",
        "tokenizer": manifest["tokenizer"],
        "sizes": {"train": len(rows), "validation": 1},
        "files_sha256": {"train.jsonl": sha256_file(train_path),
                         "validation.jsonl": sha256_file(validation_path)},
    }
    (work_data / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    settings = json.loads(Path(config_path).read_text())
    settings["training"]["epochs"] = 6
    settings["training"]["schedule_epochs"] = 6
    settings["training"]["effective_batch_size"] = BATCH_SIZE
    settings["training"]["statistics_examples"] = None
    settings["training"]["train_diagnostic_examples"] = None
    settings["data"]["train_examples"] = len(rows)
    settings["data"]["validation_examples"] = 1
    settings["data"]["max_total_tokens"] = max(sum(map(len, row["chunks"])) for row in rows)
    settings["data"]["max_chunk_tokens"] = max(len(chunk) for row in rows for chunk in row["chunks"])
    settings["data"]["max_chunks"] = max(len(row["chunks"]) for row in rows)
    smoke_config = work_dir / "smoke-config.json"
    smoke_config.write_text(json.dumps(settings, indent=2) + "\n")
    return work_data, smoke_config


def _load_trainer_state(checkpoint: Path):
    import torch
    return torch.load(Path(checkpoint) / "trainer.pt", map_location="cpu", weights_only=True)


def _adamw_step_values(state):
    values = []
    for slot in state["optimizer"]["state"].values():
        step = slot.get("step")
        if step is not None:
            values.append(int(step.item()) if hasattr(step, "item") else int(step))
    if not values:
        raise ValueError("AdamW checkpoint contains no per-parameter update counters")
    return values


def _load_model_tensors(path: Path):
    from safetensors.torch import load_file
    return load_file(str(path), device="cpu")


def _phase_groups(count: int, *, dedicated: bool):
    if dedicated and count >= 4:
        return ";".join(f"{index},{index + 1}" for index in range(2, count, 2))
    return ";".join(f"{index},{index + 1}" for index in range(0, count, 2))


def restart_cli_arguments(train_output: Path, resume_checkpoint: Path | None,
                          init_checkpoint: Path):
    if resume_checkpoint is None:
        return ["--init-from", str(init_checkpoint)]
    latest_path = Path(train_output) / "latest.json"
    if not latest_path.is_file():
        raise ValueError("Exact-resume output has no committed latest pointer")
    latest = json.loads(latest_path.read_text())
    if Path(latest.get("checkpoint", "")).resolve() != Path(resume_checkpoint).resolve():
        raise ValueError("Exact-resume parent is not the current committed latest checkpoint")
    # The production auto-resume path reuses the immutable initialization
    # provenance recorded by the first --init-from invocation.
    return ["--resume", "auto"]


def _run_phase(args, *, visible_gpus, work_dir, smoke_data, smoke_config,
               train_output, stop_after, source_checkpoint, resume_checkpoint,
               dedicated_repairer):
    group_count = len(visible_gpus)
    command = [
        args.python, "-u", "-m", "methods.CacheRepair.qwen38.train",
        "--model", str(args.model), "--dataset", str(smoke_data),
        "--config", str(smoke_config), "--output", str(train_output),
        "--gpus", "0,1", "--pair-groups", _phase_groups(group_count, dedicated=dedicated_repairer),
        "--supervision-backend", "vllm", "--vllm-memory-utilization", str(args.vllm_memory_utilization),
        "--vllm-max-batched-tokens", str(args.vllm_max_batched_tokens),
        "--vllm-prefetch-per-worker", str(args.vllm_prefetch_per_worker),
        "--vllm-scheduler", args.vllm_scheduler,
        "--checkpoint-steps", "1", "--keep-restarts", "4",
        "--stop-after-steps", str(stop_after), "--memory-efficient-repairer",
        "--cpu-threads", "1",
    ]
    if args.vllm_prefetch_limit is not None:
        command.extend(["--vllm-prefetch-limit", str(args.vllm_prefetch_limit)])
    if args.vllm_worker_pending_tokens is not None:
        command.extend(["--vllm-worker-pending-tokens", str(args.vllm_worker_pending_tokens)])
    if args.vllm_global_pending_tokens is not None:
        command.extend(["--vllm-global-pending-tokens", str(args.vllm_global_pending_tokens)])
    command.extend(restart_cli_arguments(train_output, resume_checkpoint, source_checkpoint))
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, visible_gpus)),
               OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_ALLOC_CONF="backend:native")
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    log_path = work_dir / f"phase-{group_count}-gpus-step-{stop_after}.log"
    with log_path.open("x") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Trainer phase failed ({result.returncode}); see {log_path}")
    status_path = train_output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get("status") != "interrupted":
        raise RuntimeError(f"Trainer did not stop at the requested optimizer step; see {log_path}")
    latest = json.loads((train_output / "latest.json").read_text())
    if latest.get("step") != stop_after:
        raise RuntimeError(f"Latest committed step is {latest.get('step')}, expected {stop_after}")
    checkpoint = Path(latest["checkpoint"])
    if not checkpoint.is_dir():
        raise RuntimeError("Trainer latest pointer references a missing checkpoint")
    return checkpoint, log_path


def gpu_snapshot():
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    busy = {}
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid = (value.strip() for value in line.split(",", 1))
        busy.setdefault(uuid, []).append(int(pid))
    snapshots = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, total, used, utilization = (value.strip() for value in line.split(","))
        snapshots.append({"index": int(index), "uuid": uuid, "total_mib": int(total),
                          "used_mib": int(used), "utilization": int(utilization),
                          "compute_pids": busy.get(uuid, [])})
    return snapshots


def acquire_gpu_locks(physical_ids, snapshot=None):
    """Refuse occupied cards, then take RunFast's per-physical-GPU locks."""
    snapshot = gpu_snapshot() if snapshot is None else snapshot
    by_index = {item["index"]: item for item in snapshot}
    missing = sorted(set(physical_ids) - set(by_index))
    if missing:
        raise RuntimeError(f"nvidia-smi did not report requested GPUs: {missing}")
    busy = []
    for index in physical_ids:
        item = by_index[index]
        owners = []
        for pid in item.get("compute_pids", []):
            try:
                uid = Path(f"/proc/{pid}").stat().st_uid
                owner = pwd.getpwuid(uid).pw_name
            except FileNotFoundError:
                owner = "exited"
            except KeyError:
                owner = f"uid-{uid}"
            owners.append({"pid": pid, "owner": owner, "uid": uid if owner != "exited" else None})
        if (owners or item["total_mib"] < 44000 or item["used_mib"] > 128
                or item["utilization"] > 5):
            busy.append({"gpu": index, "used_mib": item["used_mib"],
                         "utilization": item["utilization"], "processes": owners})
    if busy:
        foreign = [row for row in busy if any(proc["owner"] not in ("exited", pwd.getpwuid(os.getuid()).pw_name)
                                               for proc in row["processes"])]
        detail = "non-owned compute use" if foreign else "GPU use"
        raise RuntimeError(f"Refusing to claim GPUs with {detail}: {busy}")
    handles = []
    try:
        for index in sorted(physical_ids):
            handle = Path(f"/tmp/kvbench-cacherepair-gpu-{index}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise RuntimeError(f"GPU {index} is held by another CacheRepair job")
            handles.append(handle)
        # Detect a non-cooperating process racing the lock acquisition.
        recheck = gpu_snapshot()
        recheck_busy = []
        for item in recheck:
            if item["index"] in physical_ids and (item.get("compute_pids") or item["used_mib"] > 128
                                                   or item["utilization"] > 5):
                recheck_busy.append({"gpu": item["index"], "pids": item.get("compute_pids", []),
                                     "used_mib": item["used_mib"], "utilization": item["utilization"]})
        if recheck_busy:
            raise RuntimeError(f"GPU use appeared while acquiring locks; releasing without launch: {recheck_busy}")
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def release_gpu_locks(handles):
    for handle in handles:
        handle.close()


def _verify_fast_preflight(job: Path):
    path = job / "fast-validation-vllm.json"
    if not path.is_file() or json.loads(path.read_text()).get("status") != "passed":
        raise ValueError("A passed fast-validation-vllm.json is required before the restart smoke")
    validation = json.loads(path.read_text())
    recorded = validation.get("source_sha256", {})
    for relative, expected in recorded.items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"vLLM preflight source is stale: {relative}")
    return sha256_file(path)


def _snapshot_sources():
    paths = sorted((ROOT / "methods/CacheRepair/qwen38").glob("*.py")) + [
        ROOT / "methods/CacheRepair/backend/cacherepair/train.py",
        ROOT / "methods/CacheRepair/backend/cacherepair/model.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def run_smoke(args):
    job = Path(args.job).resolve()
    if not job.is_dir():
        raise ValueError(f"Job directory does not exist: {job}")
    gate_path = job / "six-gpu-smoke.json"
    if gate_path.exists():
        raise FileExistsError(f"Refusing to replace existing smoke gate: {gate_path}")
    validation_sha = _verify_fast_preflight(job)
    physical = parse_physical_gpus(args.physical_gpus)
    locks = acquire_gpu_locks(physical)
    try:
        return _run_smoke_locked(args, job, physical, validation_sha)
    finally:
        release_gpu_locks(locks)


def _run_smoke_locked(args, job, physical, validation_sha):
    gate_path = job / "six-gpu-smoke.json"
    source_checkpoint = Path(args.init_from).resolve()
    for name in ("model.safetensors", "config.json"):
        if not (source_checkpoint / name).is_file():
            raise FileNotFoundError(f"Initialization checkpoint lacks {name}: {source_checkpoint}")
    source_training = json.loads((source_checkpoint / "config.json").read_text()).get("training", {})
    if source_training.get("checkpoint_epoch") != 3:
        raise ValueError("The initialization checkpoint must be a committed epoch-3 repairer checkpoint")
    source_checkpoint_hashes = {name: sha256_file(source_checkpoint / name)
                                for name in ("model.safetensors", "config.json")}
    source_checkpoint_tensors = _load_model_tensors(source_checkpoint / "model.safetensors")
    rms_names = ("sigma_stale", "sigma_delta")
    if not all(name in source_checkpoint_tensors for name in rms_names):
        raise ValueError("Initialization checkpoint does not contain both RMS buffers")
    source_rms = {name: source_checkpoint_tensors[name].clone() for name in rms_names}

    work_root = job / "vllm-resume-smoke"
    work_root.mkdir(exist_ok=True)
    work_dir = work_root / f"run-{time.time_ns()}"
    work_dir.mkdir()
    rows, selection = select_smoke_rows(Path(args.dataset) / "train.jsonl")
    training_seed = json.loads(Path(args.config).read_text())["training"]["seed"]
    rows, selection = place_long_row_in_first_batch(rows, selection, seed=training_seed)
    smoke_data, smoke_config = _write_smoke_dataset(
        Path(args.dataset) / "dataset.json", work_dir, rows, Path(args.config))
    train_output = work_dir / "training"
    source_hashes = _snapshot_sources()
    phase_details = []
    expected_steps = (1, 2, 3)
    expected_progress = ((0, BATCH_SIZE), (0, 2 * BATCH_SIZE), (1, 0))
    previous_checkpoint = None
    try:
        for stage_index, (gpu_count, stop_step) in enumerate(zip((6, 4, 2), expected_steps, strict=True)):
            visible = physical[:gpu_count]
            checkpoint, log_path = _run_phase(
                args, visible_gpus=visible, work_dir=work_dir, smoke_data=smoke_data,
                smoke_config=smoke_config, train_output=train_output, stop_after=stop_step,
                source_checkpoint=source_checkpoint, resume_checkpoint=previous_checkpoint,
                dedicated_repairer=args.dedicated_repairer)
            state = _load_trainer_state(checkpoint)
            epoch, offset = expected_progress[stage_index]
            validate_restart_state(state, expected_step=stop_step, expected_epoch=epoch,
                                   expected_offset=offset, train_size=len(rows), cuda_rng_count=2)
            if _adamw_step_values(state) != [stop_step] * len(_adamw_step_values(state)):
                raise ValueError("AdamW per-parameter counters did not continue at the expected global step")
            if stage_index and state["identity"] != previous_state["identity"]:
                raise ValueError("Exact-resume identity changed across GPU allocations")
            if stage_index and state["step"] != previous_state["step"] + 1:
                raise ValueError("Exact-resume progress did not advance by one optimizer update")
            phase_details.append({"visible_gpu_count": gpu_count, "physical_gpus": list(visible),
                                  "step": stop_step, "epoch": epoch, "offset": offset,
                                  "checkpoint": str(checkpoint), "log": str(log_path),
                                  "optimizer_slots": len(state["optimizer"]["state"]),
                                  "adamw_steps": sorted(set(_adamw_step_values(state))),
                                  "cuda_rng_state_count": len(state["cuda_rng"])})
            previous_state = state
            previous_checkpoint = checkpoint

        events_path = work_dir / "training-pair-events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        expected_order = expected_training_order(rows, json.loads(smoke_config.read_text())["training"]["seed"], 1)
        validate_prompt_events(events, rows,
                               seed=json.loads(smoke_config.read_text())["training"]["seed"], epochs=1)
        event_keys = [entry.get("pair_key") for entry in events]

        losses = [json.loads(line) for line in (train_output / "loss.jsonl").read_text().splitlines() if line.strip()]
        if [entry.get("step") for entry in losses] != [1, 2, 3]:
            raise ValueError("Loss log does not contain the three committed optimizer updates")
        if any(not math.isfinite(float(entry.get("mse", float("nan")))) for entry in losses):
            raise ValueError("Training produced a nonfinite loss")
        final_tensors = _load_model_tensors(previous_checkpoint / "model.safetensors")
        rms_unchanged = all(__import__("torch").equal(final_tensors[name], source_rms[name]) for name in rms_names)
        if not rms_unchanged:
            raise ValueError("RMS buffers changed during the smoke updates")
        changed_parameters = [name for name, tensor in final_tensors.items()
                              if name not in rms_names and name in source_checkpoint_tensors
                              and not __import__("torch").equal(tensor, source_checkpoint_tensors[name])]
        # If the full source state was not retained, compare model hashes as a
        # coarse additional check; AdamW state already proves a real update.
        del final_tensors
        if not changed_parameters:
            raise ValueError("No repairer model tensor changed from the initialization weights")
        if any(sha256_file(source_checkpoint / name) != digest
               for name, digest in source_checkpoint_hashes.items()):
            raise ValueError("Initialization checkpoint changed during the smoke")
        if _snapshot_sources() != source_hashes:
            raise ValueError("Training source changed during the smoke")
        if sha256_file(job / "fast-validation-vllm.json") != validation_sha:
            raise ValueError("Fast vLLM preflight evidence changed during the smoke")

        checks = {
            "six_gpu_update": True,
            "four_gpu_exact_resume_update": True,
            "two_gpu_exact_resume_update": True,
            "optimizer_state_restored": all(item["adamw_steps"] == [item["step"]] for item in phase_details),
            "rng_state_present": all(item["cuda_rng_state_count"] == 2 for item in phase_details),
            "progress_cursor_restored": [(item["step"], item["epoch"], item["offset"])
                                          for item in phase_details] == [(1, 0, 4), (2, 0, 8), (3, 1, 0)],
            "model_updated": bool(changed_parameters),
            "rms_unchanged": rms_unchanged,
            "init_checkpoint_unchanged": True,
            "finite_losses": len(losses) == 3,
            "prompt_order_verified": event_keys == expected_order,
            "source_unchanged": True,
        }
        validate_gate_checks(checks)
        artifact = {
            "schema": "cacherepair-vllm-restart-smoke-v1",
            "status": "passed", "passed": True,
            "created_unix": time.time(),
            "scope": "Real-target vLLM CacheRepair training from epoch-3 weights/RMS with actual AdamW save/restart across six-, four-, and two-GPU allocations; isolated 12-row subset selected from real training rows.",
            "job": str(job), "model": str(Path(args.model).resolve()),
            "dataset": str(Path(args.dataset).resolve()), "init_from": str(source_checkpoint),
            "init_checkpoint_sha256": source_checkpoint_hashes,
            "fast_validation_sha256": validation_sha,
            "source_sha256": source_hashes,
            "selection": selection,
            "phase_details": phase_details,
            "optimizer_steps": [entry["step"] for entry in losses],
            "losses_finite": all(math.isfinite(float(entry["mse"])) for entry in losses),
            "changed_model_tensors": changed_parameters[:32],
            "changed_model_tensor_count": len(changed_parameters),
            "checks": checks,
            "work_dir": str(work_dir),
        }
        _atomic_json(gate_path, artifact)
        return artifact
    except BaseException as exc:
        failure = {"status": "failed", "error": repr(exc), "work_dir": str(work_dir),
                   "phase_details": phase_details, "selection": selection}
        try:
            _atomic_json(work_dir / "smoke-failure.json", failure)
        except FileExistsError:
            pass
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True,
                        help="Prepared RunFast job; writes six-gpu-smoke.json only on full success")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Production dataset containing train.jsonl and dataset.json")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, required=True,
                        help="Real epoch-3 repairer checkpoint with model.safetensors/config.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--physical-gpus", default="0,1,2,3,4,5")
    parser.add_argument("--dedicated-repairer", action="store_true")
    parser.add_argument("--vllm-memory-utilization", type=float, default=.70)
    parser.add_argument("--vllm-max-batched-tokens", type=int, default=32768)
    parser.add_argument("--vllm-prefetch-per-worker", type=int, default=8)
    parser.add_argument("--vllm-prefetch-limit", type=int, default=9)
    parser.add_argument("--vllm-worker-pending-tokens", type=int, default=49152)
    parser.add_argument("--vllm-global-pending-tokens", type=int, default=65536)
    parser.add_argument("--vllm-scheduler", choices=("round_robin", "least_work"), default="least_work")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}), file=sys.stderr, flush=True)
        raise
    print(json.dumps({"status": result["status"], "passed": result["passed"],
                      "gate": str(Path(args.job).resolve() / "six-gpu-smoke.json"),
                      "work_dir": result["work_dir"]}), flush=True)


if __name__ == "__main__":
    main()
