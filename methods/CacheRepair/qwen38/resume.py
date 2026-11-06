"""Explicit, checked additions to upstream control flow for restartable jobs.

The vendored source stays unchanged. AST edits add state I/O at optimizer-update
boundaries and replace only the starting epoch/batch and append-log mode.
Loss, optimizer, accumulation, shuffle and LR formula remain upstream code.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from .model import load_repairer, save_repairer


class TrainingInterrupted(Exception):
    """A clean stop after a committed optimizer update or statistics batch."""


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def statistics_semantics(identity):
    values = {key: value for key, value in identity.items() if key != "implementation_sha256"}
    values["backend_spec"] = {key: value for key, value in (values.get("backend_spec") or {}).items()
                              if key not in ("pipeline_depth", "root_training_pipeline_depth")}
    return values


def validated_statistics_prefix(source, identity, rows, validation):
    """Reuse FP64 sums only after a recorded real-target arithmetic check."""
    source = Path(source)
    previous = json.loads((source / "run-identity.json").read_text())
    if statistics_semantics(previous) != statistics_semantics(identity):
        raise ValueError("Imported RMS data, target, or numerical configuration differs")
    report = json.loads(Path(validation).read_text())
    pipeline = Path(__file__).parent / "experimental/pipeline.py"
    progress = source / "statistics-progress.pt"
    if not (report.get("status") == "passed" and report.get("kv_bitwise_equal")
            and report.get("rms_sums_bitwise_equal")
            and report.get("target_geometry") == identity["target_config"]
            and report.get("source_run_identity_sha256") == fingerprint(previous)
            and report.get("source_progress_sha256") == hashlib.sha256(progress.read_bytes()).hexdigest()
            and report.get("pipeline_source_sha256") == hashlib.sha256(pipeline.read_bytes()).hexdigest()):
        raise ValueError("RMS import lacks matching real-target equivalence evidence")
    state = torch.load(progress, map_location="cpu", weights_only=True)
    if state["identity"] != fingerprint({"run": previous, "rows": rows}):
        raise ValueError("Imported RMS prefix has a different row identity")
    index = state["index"]
    count = sum(sum(map(len, row["chunks"])) for row in rows[:index])
    geometry = identity["target_config"]
    shape = (len(geometry["full_attention_layer_indices"]), geometry["n_kv_heads"], 2, geometry["head_dim"])
    if not (0 < index <= len(rows) and state["count"] == count
            and set(state["sums"]) == {"sigma_stale", "sigma_delta"}
            and all(value.dtype == torch.float64 and tuple(value.shape) == shape and torch.isfinite(value).all()
                    for value in state["sums"].values())):
        raise ValueError("Imported RMS cursor, token weights, or FP64 sums are invalid")
    return state


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
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


def resumed_main(function, bindings):
    tree = ast.parse(inspect.getsource(function))
    changed = set()

    class Patch(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node) == "step = 0":
                changed.add("restore")
                return ast.parse("step, first_epoch, first_offset = restore_training(model, optimizer, horizon)").body
            return self.generic_visit(node)

        def visit_For(self, node):
            text = ast.unparse(node.iter)
            if text == "range(args.epochs)":
                node.iter = ast.parse("range(first_epoch, args.epochs)", mode="eval").body
                position = next(i for i, statement in enumerate(node.body)
                                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                                and ast.unparse(statement.value.func).endswith(".shuffle")) + 1
                node.body[position:position] = ast.parse("begin_epoch(epoch, order, first_offset if epoch == first_epoch else 0)").body
                changed.add("epoch")
            elif text == "range(0, len(order), batch_size)":
                node.iter = ast.parse("range(first_offset if epoch == first_epoch else 0, len(order), batch_size)", mode="eval").body
                node.body.extend(ast.parse("after_update(model, optimizer, step, epoch, start + len(batch), horizon)").body)
                changed.add("batch")
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node) == "save_file(stats, str(args.output / 'statistics.safetensors'))":
                node.func = ast.Name(id="save_statistics", ctx=ast.Load())
                changed.add("statistics_commit")
            if ast.unparse(node.func) == "args.output.mkdir":
                for kw in node.keywords:
                    if kw.arg == "exist_ok":
                        kw.value = ast.Constant(True)
                changed.add("directory")
            if ast.unparse(node.func) == "(args.output / 'loss.jsonl').open":
                node.args[0] = ast.Constant("a")
                changed.add("log")
            return self.generic_visit(node)

    tree = Patch().visit(tree)
    if changed != {"restore", "epoch", "batch", "directory", "log", "statistics_commit"}:
        raise RuntimeError(f"Upstream training source changed; refusing an unchecked adaptation: {changed}")
    ast.fix_missing_locations(tree)
    namespace = dict(bindings)
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    return namespace[function.__name__]


def resumed_stats(function, bindings):
    function = getattr(function, "__wrapped__", function)
    tree = ast.parse(inspect.getsource(function))
    tree.body[0].decorator_list = []
    changed = set()

    class Patch(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node) == "sums, count = ({}, 0)":
                changed.add("restore")
                return ast.parse("sums, count, stats_start = restore_statistics(rows)").body
            return self.generic_visit(node)

        def visit_For(self, node):
            if ast.unparse(node.iter) == "enumerate(rows)":
                node.iter = ast.parse("enumerate(rows[stats_start:], start=stats_start)", mode="eval").body
                node.body.extend(ast.parse("after_statistics(index + 1, sums, count)").body)
                changed.add("progress")
            return self.generic_visit(node)

    tree = Patch().visit(tree)
    if changed != {"restore", "progress"}:
        raise RuntimeError(f"Upstream statistics source changed: {changed}")
    ast.fix_missing_locations(tree)
    namespace = dict(bindings)
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    return torch.no_grad()(namespace[function.__name__])


def statistics_accumulator(function, bindings):
    """Run the official FP64 accumulation, returning its unnormalized sums."""
    function = getattr(function, "__wrapped__", function)
    tree = ast.parse(inspect.getsource(function))
    definition = tree.body[0]
    definition.decorator_list = []
    if not isinstance(definition.body[-1], ast.Return) or not isinstance(definition.body[-1].value, ast.DictComp):
        raise RuntimeError("Upstream RMS finalization changed")
    definition.body[-1] = ast.parse("return sums, count").body[0]
    ast.fix_missing_locations(tree)
    namespace = dict(bindings)
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    return torch.no_grad()(namespace[function.__name__])


class Restart:
    def __init__(self, output, identity, target, model_id, *, resume=None, initialize=None,
                 checkpoint_steps=250, keep=2, gpu_ids=(), stop_after_steps=None):
        if resume and initialize:
            raise ValueError("resume and initialize are mutually exclusive")
        if checkpoint_steps < 1 or keep < 1:
            raise ValueError("Checkpoint interval and retention must be positive")
        self.output, self.identity = Path(output), identity
        self.target, self.model_id = target, model_id
        self.auto_requested = resume == "auto"
        if self.auto_requested:
            latest = self.output / "latest.json"
            identity_path = self.output / "run-identity.json"
            if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
                raise ValueError("Automatic resume run identity differs or is missing")
            pointer = json.loads(latest.read_text()) if latest.exists() else {}
            resume = pointer.get("checkpoint")
            # Directory rename commits all checkpoint files before the small
            # latest pointer is published. Recover that crash window explicitly.
            root = self.output / "resume-checkpoints"
            candidates = []
            for path in root.glob("step-*"):
                suffix = path.name.removeprefix("step-")
                if path.is_dir() and suffix.isdecimal() and int(suffix) > pointer.get("step", -1):
                    candidates.append((int(suffix), path))
            if candidates:
                step, candidate = max(candidates)
                state = self.checked_state(candidate)
                if state["step"] != step:
                    raise ValueError("Committed checkpoint directory and step differ")
                resume = str(candidate.resolve())
                atomic_json(latest, {"checkpoint": resume, "step": step})
                atomic_json(self.output / "recovered-latest.json", {"previous": pointer,
                    "checkpoint": resume, "step": step, "reason": "Committed directory preceded latest pointer"})
        self.resume = Path(resume).resolve() if resume else None
        self.initialization_record_present = "initialization" in identity
        self.initialization_record = identity.get("initialization")
        recorded_source = None
        if self.initialization_record is not None:
            record = self.initialization_record
            if (not isinstance(record, dict) or record.get("format") != "qwen38-initialization-v1"
                or not isinstance(record.get("source"), str)
                or not isinstance(record.get("model_sha256"), str)):
                raise ValueError("Initial checkpoint provenance is missing or invalid")
            recorded_source = Path(record["source"]).resolve()
            source_model = recorded_source / "model.safetensors"
            if (not source_model.is_file()
                or hashlib.sha256(source_model.read_bytes()).hexdigest() != record["model_sha256"]):
                raise ValueError("Initial checkpoint model changed since this run was created")
        self.initialize = Path(initialize).resolve() if initialize else None
        if self.initialize is not None:
            if recorded_source is None or self.initialize != recorded_source:
                raise ValueError("Initial checkpoint differs from immutable run provenance")
        elif self.auto_requested and self.resume is None and recorded_source is not None:
            # No durable training state yet: restore the original weights and
            # RMS and re-create the fresh optimizer rather than starting random.
            self.initialize = recorded_source
        self.interval, self.keep, self.gpu_ids = checkpoint_steps, keep, gpu_ids
        self.stop_after_steps = stop_after_steps
        self.requested = None
        self.last_checkpoint = str(self.resume) if self.resume else None
        self.previous_signals = {}

    def restore_statistics(self, rows):
        path = self.output / "statistics-progress.pt"
        self.stats_identity = fingerprint({"run": self.identity, "rows": rows})
        if not path.exists() and getattr(self, "statistics_from", None):
            state = validated_statistics_prefix(self.statistics_from, self.identity, rows, self.statistics_validation)
            state["identity"] = self.stats_identity
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            atomic_json(self.output / "statistics-import.json", {"source": str(self.statistics_from),
                "validation": str(self.statistics_validation), "examples": state["index"], "tokens": state["count"],
                "scope": "FP64 RMS sums only; fresh repairer and optimizer; verified unchanged numerical semantics"})
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state["identity"] != self.stats_identity:
                raise ValueError("Statistics resume inputs differ")
            self.last_statistics_checkpoint = state["index"]
            return state["sums"], state["count"], state["index"]
        self.last_statistics_checkpoint = 0
        return {}, 0, 0

    def after_statistics(self, index, sums, count):
        if index - self.last_statistics_checkpoint < 100 and not self.requested:
            return
        path = self.output / "statistics-progress.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"identity": self.stats_identity, "index": index,
                    "sums": sums, "count": count}, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.last_statistics_checkpoint = index
        atomic_json(self.output / "status.json", {"status": "statistics", "examples_done": index})
        if self.requested:
            raise TrainingInterrupted(self.requested)

    def install(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous_signals[number] = signal.getsignal(number)
            signal.signal(number, lambda n, frame: setattr(self, "requested", signal.Signals(n).name))

    def close(self):
        for number, handler in self.previous_signals.items():
            signal.signal(number, handler)

    def state(self):
        return self.checked_state(self.resume)

    def checked_state(self, directory):
        directory = Path(directory)
        state = torch.load(directory / "trainer.pt", map_location="cpu", weights_only=True)
        if state["format"] != "kvbench-qwen38-trainer-v1" or state["identity"] != self.identity:
            raise ValueError("Resume dataset, target, architecture or training schedule differs")
        if state["model_sha256"] != hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest():
            raise ValueError("Resume model checksum differs")
        return state

    def existing_statistics(self):
        path = self.output / "statistics.safetensors"
        if not path.exists():
            return None
        try:
            values = load_file(str(path))
            geometry = self.identity["target_config"]
            shape = (len(geometry["full_attention_layer_indices"]), geometry["n_kv_heads"], 2, geometry["head_dim"])
            if (set(values) != {"sigma_stale", "sigma_delta"} or any(
                value.dtype != torch.float32 or tuple(value.shape) != shape
                or not torch.isfinite(value).all() or not (value > 0).all() for value in values.values())):
                raise ValueError("Invalid finalized RMS tensor values")
        except (SafetensorError, ValueError, OSError) as error:
            archive = path.with_name(f"statistics.invalid-{time.time_ns()}.safetensors")
            os.replace(path, archive)
            atomic_json(self.output / "recovered-statistics.json", {"archive": str(archive),
                "reason": repr(error), "action": "Rebuild from durable FP64 statistics prefix"})
            return None
        return path

    def save_statistics(self, values, path):
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        save_file(values, str(temporary))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def restore(self, model, optimizer, horizon):
        if (self.auto_requested and self.resume is None and self.initialize is None
            and not self.initialization_record_present):
            raise ValueError("Automatic resume has no durable restart or recorded initialization")
        if self.resume or self.initialize:
            source = self.resume or self.initialize
            loaded = load_repairer(source, self.target, target_model_id=self.model_id)
            if loaded.config != model.config:
                raise ValueError("Initial repairer architecture differs")
            model.load_state_dict(loaded.state_dict(), strict=True)
            del loaded
        if not self.resume:
            if self.auto_requested:
                self.trim_log(0)
            return 0, 0, 0
        state = self.state()
        if state["horizon"] != horizon:
            raise ValueError("Resume learning-rate horizon differs")
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if len(state["cuda_rng"]) != len(self.gpu_ids):
            raise ValueError("Resume GPU count differs")
        for gpu, rng in zip(self.gpu_ids, state["cuda_rng"]):
            torch.cuda.set_rng_state(rng, gpu)
        self.trim_log(state["step"])
        return state["step"], state["epoch"], state["offset"]

    def trim_log(self, committed_step):
        # A crash can leave log entries beyond the last durable checkpoint.
        # Preserve those entries separately before replaying their updates.
        log = self.output / "loss.jsonl"
        if log.exists():
            rows = log.read_text().splitlines(keepends=True)
            retained, replayed = [], []
            for row in rows:
                try:
                    step = json.loads(row)["step"]
                except (ValueError, KeyError):
                    replayed.append(row)
                    continue
                (retained if step <= committed_step else replayed).append(row)
            if replayed:
                with (self.output / "replayed-loss.jsonl").open("a") as stream:
                    stream.writelines(replayed)
                log.write_text("".join(retained))

    def after_update(self, model, optimizer, step, epoch, offset, horizon, *, force=False):
        size = self.identity["train_size"]
        next_epoch, next_offset = (epoch + 1, 0) if offset >= size else (epoch, offset)
        stop = self.requested or (self.stop_after_steps is not None and step >= self.stop_after_steps)
        if step % 25 == 0:
            atomic_json(self.output / "status.json", {"status": "training", "step": step,
                        "epoch": epoch + 1, "next_offset": next_offset, "latest_restart": self.last_checkpoint})
        if step % self.interval and offset < size and not stop and not force:
            return
        root = self.output / "resume-checkpoints"
        root.mkdir(exist_ok=True)
        destination = root / f"step-{step:08d}"
        if destination.exists():
            raise FileExistsError(f"Refusing to replace a committed restart: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=root))
        try:
            save_repairer(temporary, model)
            if self.identity.get("provenance") is not None:
                atomic_json(temporary / "target-provenance.json", self.identity["provenance"])
            state = {"format": "kvbench-qwen38-trainer-v1", "identity": self.identity,
                     "step": step, "epoch": next_epoch, "offset": next_offset, "horizon": horizon,
                     "optimizer": optimizer.state_dict(), "python_rng": random.getstate(),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": [torch.cuda.get_rng_state(gpu) for gpu in self.gpu_ids],
                     "model_sha256": hashlib.sha256((temporary / "model.safetensors").read_bytes()).hexdigest()}
            torch.save(state, temporary / "trainer.pt")
            # Rename only after every file has reached the filesystem.
            for path in temporary.iterdir():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.rename(temporary, destination)
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            atomic_json(self.output / "latest.json", {"checkpoint": str(destination.resolve()), "step": step})
            self.last_checkpoint = str(destination.resolve())
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        for path in sorted(root.glob("step-*"))[:-self.keep]:
            if path.resolve() != self.resume:
                shutil.rmtree(path)
        print(json.dumps({"restart_checkpoint": str(destination), "step": step}), flush=True)
        if stop:
            raise TrainingInterrupted(str(self.requested or "requested update limit"))
