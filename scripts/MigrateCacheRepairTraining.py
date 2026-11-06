"""Carry a stopped run across the explicitly audited recovery-only source update.

Model/optimizer/RNG/cursor values remain identical. This is not a generic bypass
for identity checks: changed source hashes must match the reviewed candidate.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same_tree(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def audited_identity(previous, audit, root):
    approved = json.loads((audit / "candidate-sha256.json").read_text())
    log = (audit / "full-candidate-tests.log").read_text()
    if not re.search(r"\b\d+ passed\b", log) or re.search(r"\b\d+ failed\b", log):
        raise ValueError("Audited candidate lacks passing regression evidence")
    for name, hashes in approved.items():
        if digest(root / name) != hashes["candidate"] or digest(audit / "candidate" / name) != hashes["candidate"]:
            raise ValueError(f"Current source differs from reviewed candidate: {name}")
    identity = copy.deepcopy(previous)
    changed = []
    for name, old_hash in previous["implementation_sha256"].items():
        current = digest(root / name)
        if current != old_hash:
            if name not in approved or approved[name]["before"] != old_hash:
                raise ValueError(f"Unaudited implementation change: {name}")
            changed.append(name)
        identity["implementation_sha256"][name] = current
    shared = "methods/CacheRepair/backend/cacherepair/model.py"
    if digest(root / shared) != digest(audit / "candidate" / shared):
        raise ValueError("Shared official repair blocks differ from audited source")
    identity["implementation_sha256"][shared] = digest(root / shared)
    execution = audit / "execution-update.json"
    backend = identity.get("backend_spec")
    if execution.exists() and backend and backend.get("pipeline_depth"):
        proof = json.loads(execution.read_text())
        if not (proof.get("status") == "passed" and proof.get("optimizer_update_bitwise_equal")
                and proof.get("root_depth_before") == backend.get("root_training_pipeline_depth")
                and proof.get("root_depth_after") == min(3, backend["pipeline_depth"])
                and all(digest(root / name) == value for name, value in proof["implementation_sha256"].items())):
            raise ValueError("Prefetch execution update lacks matching numerical evidence")
        backend["root_training_pipeline_depth"] = proof["root_depth_after"]
    return identity, changed


def migrate(source, destination, audit, *, root=ROOT):
    source, destination, audit, root = map(lambda p: Path(p).resolve(), (source, destination, audit, root))
    if destination.exists():
        raise FileExistsError(destination)
    with (source.parent / "fast-supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = json.loads((source / "status.json").read_text())
        if status["status"] not in {"interrupted", "completed"}:
            raise ValueError("Source run must be stopped at a committed optimizer update")
        previous = json.loads((source / "run-identity.json").read_text())
        identity, changed = audited_identity(previous, audit, root)
        pointer = json.loads((source / "latest.json").read_text())
        checkpoint = Path(pointer["checkpoint"]).resolve()
        if checkpoint.parent != source / "resume-checkpoints":
            raise ValueError("Source latest points outside this run")
        state = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=True)
        if state["format"] != "kvbench-qwen38-trainer-v1" or state["identity"] != previous:
            raise ValueError("Source trainer identity differs")
        if state["step"] != pointer["step"] or digest(checkpoint / "model.safetensors") != state["model_sha256"]:
            raise ValueError("Source checkpoint commit or model checksum differs")
        updates = math.ceil(previous["train_size"] / 4)
        epoch, batch = divmod(state["step"], updates)
        if (state["epoch"], state["offset"], state["horizon"]) != (epoch, batch * 4, updates * previous["schedule_epochs"]):
            raise ValueError("Source update/data cursor or LR horizon differs")
        stats = load_file(str(source / "statistics.safetensors"))
        model = load_file(str(checkpoint / "model.safetensors"))
        for name in ("sigma_stale", "sigma_delta"):
            if not torch.equal(stats[name], model[name]) or not torch.isfinite(stats[name]).all() or not (stats[name] > 0).all():
                raise ValueError("Final RMS and trained model normalization differ")
        del model
        rows = [json.loads(line) for line in (source / "loss.jsonl").read_text().splitlines()]
        if [row["step"] for row in rows] != list(range(1, state["step"] + 1)):
            raise ValueError("Source loss log is not the exact committed update prefix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".migrating-", dir=destination.parent))
        try:
            migrated = temporary / "resume-checkpoints" / checkpoint.name
            shutil.copytree(checkpoint, migrated)
            replacement = {**state, "identity": identity}
            torch.save(replacement, migrated / "trainer.pt")
            reloaded = torch.load(migrated / "trainer.pt", map_location="cpu", weights_only=True)
            if reloaded["identity"] != identity or not same_tree(
                    {k: v for k, v in state.items() if k != "identity"},
                    {k: v for k, v in reloaded.items() if k != "identity"}):
                raise AssertionError("Numerical trainer state changed during migration")
            for name in ("statistics.safetensors", "loss.jsonl", "target-provenance.json", "training-config.json", "experiment-config.json"):
                if (source / name).exists():
                    shutil.copyfile(source / name, temporary / name)
                    if digest(source / name) != digest(temporary / name):
                        raise AssertionError(f"Copied artifact changed: {name}")
            final_checkpoint = destination / "resume-checkpoints" / checkpoint.name
            receipt = {"time_unix": time.time(), "source": str(source), "source_checkpoint": str(checkpoint),
                "step": state["step"], "epoch": state["epoch"], "offset": state["offset"],
                "horizon": state["horizon"], "changed_implementation": changed,
                "model_sha256": state["model_sha256"], "statistics_sha256": digest(source / "statistics.safetensors"),
                "optimizer_rng_cursor_bitwise_preserved": True, "rms_recomputed": False,
                "source_identity": previous, "destination_identity": identity,
                "audit_tests_sha256": digest(audit / "full-candidate-tests.log")}
            documents = {"run-identity.json": identity, "latest.json": {"checkpoint": str(final_checkpoint), "step": state["step"]},
                "migration.json": receipt, "status.json": {"status": "training", "step": state["step"],
                    "epoch": state["epoch"] + 1, "next_offset": state["offset"], "latest_restart": str(final_checkpoint)}}
            for name, value in documents.items():
                (temporary / name).write_text(json.dumps(value, indent=2) + "\n")
            for path in temporary.rglob("*"):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            for path in [*sorted((p for p in temporary.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True), temporary]:
                descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            os.rename(temporary, destination)
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return receipt
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    receipt = migrate(args.source, args.destination, args.audit)
    print(json.dumps({key: value for key, value in receipt.items() if key not in {"source_identity", "destination_identity"}}), flush=True)


if __name__ == "__main__":
    main()
