"""Thin Qwen3.8 adapter around the unchanged official CacheRepair training loop.

No alternate optimizer loop: main(), compute_stats(), learning_rate(), and loss
execute official function code with a private dependency binding dictionary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types

import torch
from safetensors.torch import load_file, save_file

from ..backend.cacherepair import train as official
from ..backend.cacherepair.model import CacheRepair as OfficialRepairer
from .model import QwenCacheRepair, RepairConfig, load_repairer, save_repairer
from .sources import local_checkpoint_spec, local_tokenizer_spec, write_json
from .supervision import make_pair
from .target import TargetGeometry
from .resume import Restart, TrainingInterrupted, atomic_json, fingerprint, resumed_main, resumed_stats


UPSTREAM_COMMIT = "4eb96cd92df4be4f679ad8ac42a51442249683c2"


class _Proxy:
    def __init__(self, original, **overrides):
        self.original, self.overrides = original, overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.original, name)


class _FrozenDispatchedTarget(_Proxy):
    def to(self, device):
        if device != "cuda":
            raise ValueError("Unexpected upstream target device request")
        # Already placed by the native loader. Moving the whole 27B model to
        # one GPU would invalidate the deliberately dispatched layer mapping.
        return self

    def eval(self):
        self.original.eval()
        return self


def _bind(function, bindings):
    function = getattr(function, "__wrapped__", function)
    bound = types.FunctionType(function.__code__, bindings, function.__name__,
                               function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def _parameter_sample(target):
    digest = hashlib.sha256()
    for name, parameter in target.named_parameters():
        if "embed_tokens" in name or "layers.0." in name or "layers.3.self_attn" in name:
            digest.update(name.encode())
            digest.update(parameter.detach().flatten()[:32].float().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def validation_diagnostics(model, target, rows, pair_function):
    if hasattr(model, '_long_training'):
        from .long_training import validation_diagnostics as tiled_diagnostics
        return tiled_diagnostics(model,target,rows,pair_function)
    totals = {name: 0.0 for name in ("normalized_mse", "zero_residual_mse", "error_ss", "true_ss", "pred_ss", "stale_ss")}
    per_layer, per_layer_baseline, tokens = None, None, 0
    model.eval()
    for row in rows:
        pair = pair_function(target, row["chunks"])
        prediction = model(pair["stale"], pair["tokens"], pair["chunk_ids"]).float()
        delta = pair["joint"] - pair["stale"]
        normalized = delta / model.sigma_delta
        error = (prediction - normalized).square()
        totals["normalized_mse"] += error.mean().item()
        totals["zero_residual_mse"] += normalized.square().mean().item()
        predicted = prediction * model.sigma_delta
        for key, value in (("error_ss", predicted - delta), ("true_ss", delta),
                           ("pred_ss", predicted), ("stale_ss", pair["stale"])):
            totals[key] += value.double().square().sum().item()
        layer_error = error.sum(dim=(0, 1, 3, 5)).double().cpu()
        per_layer = layer_error if per_layer is None else per_layer + layer_error
        layer_baseline = normalized.square().sum(dim=(0, 1, 3, 5)).double().cpu()
        per_layer_baseline = layer_baseline if per_layer_baseline is None else per_layer_baseline + layer_baseline
        tokens += pair["tokens"].numel()
    count = len(rows)
    scale = tokens * model.target_config.n_kv_heads * model.target_config.head_dim
    return {"examples": count, "tokens": tokens,
            "normalized_residual_mse": totals["normalized_mse"] / count,
            "zero_residual_baseline_mse": totals["zero_residual_mse"] / count,
            "relative_residual_l2_error": (totals["error_ss"] / max(totals["true_ss"], 1e-30)) ** 0.5,
            "predicted_to_true_residual_l2": (totals["pred_ss"] / max(totals["true_ss"], 1e-30)) ** 0.5,
            "predicted_to_stale_kv_l2": (totals["pred_ss"] / max(totals["stale_ss"], 1e-30)) ** 0.5,
            "per_attention_layer_k_v_normalized_mse": (per_layer / scale).tolist(),
            "per_attention_layer_k_v_zero_residual_mse": (per_layer_baseline / scale).tolist()}


def run_upstream(target, train_rows, validation_rows, output, *, model_id,
                 repair_config=None, seed=7, initialization_seed=853119240,
                 epochs=1, schedule_epochs=1, stats_examples=None, device=None,
                 gpu_ids=(), max_pair_seconds=120, validate_train=True, cache_pairs=False,
                 resume_from=None, init_from=None, checkpoint_steps=250, keep_restarts=2,
                 stop_after_steps=None, provenance=None, train_diagnostic_examples=None,
                 pair_pool=None, backend_spec=None,
                 statistics_from=None, statistics_validation=None,
                 learning_rate_scale=1.0, evaluation_plan=None,
                 scope="Training-source residual validation; no generation-quality or TTFT experiment"):
    """Execute original upstream main with only target/cache/model/storage adapters."""
    target.requires_grad_(False).eval()
    geometry = TargetGeometry.from_target(target, target_model_id=model_id)
    device = torch.device(device or target.get_input_embeddings().weight.device)
    output = Path(output)
    if output.exists() and not resume_from:
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not train_rows or not validation_rows:
        raise ValueError("Training and validation rows must both be nonempty")
    for row in [*train_rows, *validation_rows]:
        if len(row.get("chunks", [])) < 2:
            raise ValueError("Training/validation examples need at least two chunks")
    if stats_examples is not None and not 0 < stats_examples <= len(train_rows):
        raise ValueError("stats_examples must be positive and no greater than training size")
    repair_config = repair_config or RepairConfig()
    if not 0 < learning_rate_scale <= 1:
        raise ValueError('learning_rate_scale must be in (0,1]')
    settings = {"target_model_id": model_id, "target_config": geometry.to_dict(),
                "repair_config": asdict(repair_config),
                "training": {"initialization_seed": initialization_seed, "seed": seed,
                             "learning_rate_scale":learning_rate_scale}}
    # Persist initialization provenance in new run identities. On auto-resume,
    # reuse the immutable record so a relaunch checks the exact parent weights
    # even though the supervisor selects --resume instead of --init-from.
    initialization_record = None
    include_initialization_record = True
    if resume_from:
        identity_path = Path(output) / "run-identity.json"
        previous_identity = None
        if identity_path.is_file():
            previous_identity = json.loads(identity_path.read_text())
        elif str(resume_from) != "auto":
            # An explicit durable checkpoint can seed a new output directory.
            # Its embedded identity is the provenance source; deployment
            # checkpoints without trainer state are not exact-resume inputs.
            state_path = Path(resume_from) / "trainer.pt"
            if not state_path.is_file():
                raise ValueError("Explicit resume checkpoint lacks embedded run identity")
            previous_state = torch.load(state_path, map_location="cpu", weights_only=True)
            previous_identity = previous_state.get("identity")
            if not isinstance(previous_identity, dict):
                raise ValueError("Explicit resume checkpoint has no valid run identity")
        if previous_identity is not None and "initialization" in previous_identity:
            initialization_record = previous_identity["initialization"]
        else:
            # Preserve pre-metadata identities without rewriting them.
            include_initialization_record = False
    elif init_from:
        source = Path(init_from).resolve()
        source_model = source / "model.safetensors"
        initialization_record = {"format": "qwen38-initialization-v1",
            "source": str(source), "model_sha256": hashlib.sha256(source_model.read_bytes()).hexdigest()}

    identity = {**settings, "train_sha256": fingerprint(train_rows),
                "validation_sha256": fingerprint(validation_rows), "train_size": len(train_rows),
                "epochs": epochs, "schedule_epochs": schedule_epochs,
                "statistics_examples": stats_examples, "provenance": provenance,
                "implementation_sha256": {str(path.relative_to(Path(__file__).parent.parent.parent.parent)):
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                        Path(__file__), Path(__file__).with_name("resume.py"), Path(__file__).with_name("model.py"),
                        Path(__file__).with_name("supervision.py"), Path(__file__).with_name("target.py"),
                        Path(__file__).with_name("cache.py"), Path(__file__).with_name("pair_pool.py"),
                        Path(__file__).with_name("packed.py"), Path(official.__file__),
                        Path(sys.modules[OfficialRepairer.__module__].__file__),
                        *([Path(__file__).with_name('long_training.py')]
                          if (backend_spec or {}).get('memory_efficient_repairer') else []),
                        *([Path(__file__).with_name('scheduled_validation.py')] if evaluation_plan else []),
                        *([Path(__file__).with_name(name) for name in
                           ('vllm_pool.py', 'vllm_target.py', 'vllm_capture.py')]
                          if (backend_spec or {}).get('supervision_backend') == 'vllm' else []),
                        *([Path(__file__).parent / "experimental/pipeline.py"]
                          if (backend_spec or {}).get("pipeline_depth") else []))},
                "torch_version": str(torch.__version__),
                "native_library_versions": {name: importlib.metadata.version(name) for name in
                                            ("transformers", "accelerate", "safetensors")},
                "backend_spec": backend_spec}
    if include_initialization_record:
        identity["initialization"] = initialization_record
    if evaluation_plan:
        identity['evaluation_plan'] = evaluation_plan
    restart = Restart(output, identity, target, model_id, resume=resume_from, initialize=init_from,
                      checkpoint_steps=checkpoint_steps, keep=keep_restarts, gpu_ids=gpu_ids,
                      stop_after_steps=stop_after_steps)
    restart.statistics_from = statistics_from
    restart.statistics_validation = statistics_validation
    resume_state = restart.state() if restart.resume else None  # Validate before touching output.
    report = {"upstream_commit": UPSTREAM_COMMIT, "target_geometry": geometry.to_dict(),
              "repair_config": asdict(repair_config), "seed": seed, "initialization_seed": initialization_seed,
              "effective_batch_size": 4, "epochs": epochs, "schedule_epochs": schedule_epochs,
              "learning_rate_scale":learning_rate_scale,
              "initialization": (f"exact resume: {resume_from}" if resume_from else
                                 f"weights and RMS initialized from: {init_from}; fresh optimizer/schedule" if init_from
                                 else "fresh repairer; no checkpoint or optimizer state resumed"), "scope": scope,
              "statistics_examples": len(train_rows) if stats_examples is None else stats_examples,
              "gradient_steps": [], "pair_examples": 0, "pair_tokens": 0, "pair_seconds": 0.,
              "pair_cache_hits": 0, "pair_cache": "run-local disk cache" if cache_pairs else "disabled",
              "checkpoints": [], "target_parameter_sample_before": _parameter_sample(target)}
    target_parameters = {id(parameter) for parameter in target.parameters()}
    schedule = None
    if evaluation_plan and evaluation_plan.get('enabled'):
        from .scheduled_validation import ScheduledValidation
        schedule = ScheduledValidation(output, validation_rows, evaluation_plan)

    def synchronize():
        if device.type == "cuda":
            for gpu in gpu_ids or (device.index,):
                torch.cuda.synchronize(gpu)

    def pairs(_target, chunks):
        cached = None
        if cache_pairs:
            key = hashlib.sha256(json.dumps(chunks, separators=(",", ":")).encode()).hexdigest()
            directory = output / "supervision-pairs"
            directory.mkdir(exist_ok=True)
            cached = directory / f"{key}.safetensors"
            if cached.is_file():
                report["pair_cache_hits"] += 1
                return {key: value.to(device) for key, value in load_file(str(cached)).items()}
        if not (backend_spec or {}).get("pipeline_depth") and not getattr(pair_pool, "asynchronous", False):
            synchronize()
        started = time.perf_counter()
        result = pair_pool.get(chunks) if pair_pool is not None else make_pair(
            target, chunks, geometry=geometry, chunk_batch_size=(backend_spec or {}).get("chunk_batch_size", 1),
            pack_chunks=(backend_spec or {}).get("pack_chunks", False))
        if not (backend_spec or {}).get("pipeline_depth") and not getattr(pair_pool, "asynchronous", False):
            synchronize()
        elapsed = pair_pool.last_event["compute_seconds"] if pair_pool is not None else time.perf_counter() - started
        report["pair_examples"] += 1
        report["pair_tokens"] += result["tokens"].numel()
        report["pair_seconds"] += elapsed
        if cached is not None:
            save_file({key: value.cpu().contiguous() for key, value in result.items()}, str(cached))
        if elapsed > max_pair_seconds:
            raise RuntimeError(f"Supervision pair cost {elapsed:.1f}s exceeds {max_pair_seconds}s budget; stopping without changing objective")
        if report["pair_examples"] == 1 or report["pair_examples"] % 8 == 0:
            print(json.dumps({"supervision_pairs": report["pair_examples"], "last_pair_seconds": elapsed,
                              "kv_shape": list(result["stale"].shape)}), flush=True)
        return result

    # Keep every original module global unchanged. Official code functions run
    # against this private dictionary, avoiding process-global monkeypatches.
    bindings = dict(official.__dict__)
    bindings["make_pair"] = pairs
    def restore_statistics(rows):
        values = restart.restore_statistics(rows)
        if pair_pool is not None:
            pair_pool.set_sequence(rows[values[2]:])
        return values

    def begin_epoch(epoch, order, offset):
        if pair_pool is not None:
            if hasattr(pair_pool, "set_training"):
                pair_pool.set_training(True)
            pair_pool.set_sequence([train_rows[index] for index in order[offset:]])

    bindings.update(restore_statistics=restore_statistics, after_statistics=restart.after_statistics,
                    begin_epoch=begin_epoch)
    official_stats = resumed_stats(official.compute_stats, bindings)

    def stats(_target, rows):
        indices = list(range(len(rows)))
        if stats_examples is not None:
            import random
            indices = sorted(random.Random(seed).sample(indices, stats_examples))
        report["statistics_training_row_indices"] = indices
        started = time.perf_counter()
        selected = [rows[i] for i in indices]
        if pair_pool is not None and hasattr(pair_pool, "statistics"):
            sums, count = pair_pool.statistics(selected, restart)
            # Execute the official RMS final expression once on the combined
            # FP64 token-weighted sums, without moving full pairs across IPC.
            finalize = resumed_stats(official.compute_stats, {
                **bindings, "restore_statistics": lambda rows: (sums, count, 0)})
            result = finalize(_target, [])
        else:
            result = official_stats(_target, selected)
        report["statistics_seconds"] = time.perf_counter() - started
        return result

    class GeometryAdapter:
        @staticmethod
        def from_hf(config):
            if config is not target.config:
                raise ValueError("Unexpected target config passed by upstream")
            return geometry

        def __new__(cls, **values):
            return TargetGeometry.from_dict(values)

    class RepairerAdapter(QwenCacheRepair):
        # The official normalized residual MSE is reused verbatim. Requests
        # have variable lengths and accumulate individually, with no padding.
        def forward(self,stale,tokens,chunk_ids):
            if hasattr(self,'_long_training'):
                from .long_training import forward
                return forward(self,stale,tokens,chunk_ids)
            return super().forward(stale,tokens,chunk_ids)

        def loss(self,stale,joint,tokens,chunk_ids):
            if hasattr(self,'_long_training'):
                from .long_training import loss
                return loss(self,stale,joint,tokens,chunk_ids)
            return OfficialRepairer.loss(self,stale,joint,tokens,chunk_ids)

        def to(self, *args, **kwargs):
            if args and args[0] == "cuda":
                args = (device, *args[1:])
            return super().to(*args, **kwargs)

    trained = {}
    epoch_clock = {}

    def construct(*args):
        model = RepairerAdapter(*args)
        if (backend_spec or {}).get('memory_efficient_repairer'):
            from .long_training import configure
            configure(model,tile=backend_spec.get('repairer_tile',256),efficient=device.type=='cuda')
        trained["model"] = model
        report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        return model

    base_adamw = torch.optim.AdamW

    def optimizer_factory(parameters, **kwargs):
        parameters = list(parameters)
        if any(id(parameter) in target_parameters for parameter in parameters):
            raise AssertionError("Target parameters entered the repairer optimizer")
        optimizer = base_adamw(parameters, **kwargs)
        epoch_clock["started"] = time.perf_counter()

        def verify_gradients(optimizer, args, kwargs):
            model = trained["model"]
            gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                         if parameter.grad is not None]
            if not gradients:
                raise FloatingPointError("Repairer gradients are absent or nonfinite")
            # Collect the small status tensor in one transfer; individual
            # .item() calls would serialize the GPU once per parameter/flag.
            checks = torch.stack([torch.stack((torch.isfinite(gradient).all(),
                torch.count_nonzero(gradient).ne(0))) for _, gradient in gradients]).cpu().tolist()
            if not all(finite for finite, _ in checks):
                raise FloatingPointError("Repairer gradients are absent or nonfinite")
            if any(parameter.grad is not None for parameter in target.parameters()):
                raise AssertionError("Frozen target received gradients")
            nonzero = [name for (name, _), (_, present) in zip(gradients, checks) if present]
            if not nonzero:
                raise AssertionError("All repairer gradients are zero")
            global_step = (resume_state["step"] if resume_state else 0) + len(report["gradient_steps"]) + 1
            report["gradient_steps"].append({"step": global_step,
                                             "parameters_with_grad": len(gradients),
                                             "parameters_with_nonzero_grad": nonzero,
                                             "target_gradients": 0})
            print(json.dumps({"optimizer_step": global_step,
                              "nonzero_gradient_parameters": len(nonzero)}), flush=True)
        optimizer.register_step_pre_hook(verify_gradients)
        return optimizer

    def checkpoint(directory, model, *, target_model_id=None, training=None):
        elapsed = time.perf_counter() - epoch_clock["started"]
        import random
        order = list(range(len(train_rows)))
        random.Random(seed * 1_000_003 + (training["checkpoint_epoch"] - 1) * 10_007).shuffle(order)
        offset = resume_state["offset"] if resume_state and training["checkpoint_epoch"] == resume_state["epoch"] + 1 else 0
        measured_rows = [train_rows[i] for i in order[offset:]]
        tokens = sum(sum(map(len, row["chunks"])) for row in measured_rows)
        report.setdefault("training_epochs", []).append({
            "epoch": training["checkpoint_epoch"], "seconds": elapsed,
            "segment_examples": len(measured_rows), "segment_tokens": tokens,
            "examples_per_second": len(measured_rows) / elapsed, "tokens_per_second": tokens / elapsed,
        })
        save_repairer(directory, model)
        if provenance is not None:
            write_json(Path(directory) / "target-provenance.json", provenance)
        config_path = Path(directory) / "config.json"
        values = json.loads(config_path.read_text())
        values.update({"training": training, "upstream_commit": UPSTREAM_COMMIT})
        write_json(config_path, values)
        validation_started = time.perf_counter()
        original_mode = model.training
        if pair_pool is not None:
            pair_pool.set_sequence(validation_rows)
        diagnostics = validation_diagnostics(model, target, validation_rows, pairs)
        # Reload validation constructs a temporary model. Keep those initializer
        # draws out of the training RNG, including when an epoch callback is
        # completed after resuming the final update of that epoch.
        with torch.random.fork_rng(devices=list(gpu_ids)):
            loaded = load_repairer(directory, target, target_model_id=model_id)
        if hasattr(model,'_long_training'):
            from .long_training import configure,forward as tiled_forward
            configure(loaded,**model._long_training)
        pair = pairs(target, validation_rows[0]["chunks"])
        with torch.no_grad():
            expected = model(pair["stale"], pair["tokens"], pair["chunk_ids"])
            actual = (tiled_forward(loaded,pair['stale'],pair['tokens'],pair['chunk_ids'])
                if hasattr(model,'_long_training') else loaded(pair["stale"],pair["tokens"],pair["chunk_ids"]))
        if not torch.equal(expected, actual):
            raise AssertionError("Reloaded repairer output differs from saved checkpoint")
        del pair, loaded
        entry = {"path": str(directory), "validation": diagnostics,
                 "reload_exact": True, "bytes": (Path(directory) / "model.safetensors").stat().st_size}
        report["checkpoints"].append(entry)
        # Publish completion only after all artifacts, validation and reload
        # comparison succeeded. An existing directory alone is not a commit.
        for path in Path(directory).iterdir():
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        atomic_json(Path(directory) / "validation.json", entry)
        model.train(original_mode)
        if schedule is not None:
            schedule.record(training['checkpoint_step'], 'epoch', diagnostics,
                            time.perf_counter()-validation_started)
        epoch_clock["started"] = time.perf_counter()
        from .scheduled_validation import request_epoch_quality
        if not restart.requested and request_epoch_quality(output, directory, training, evaluation_plan):
            raise TrainingInterrupted('epoch_quality_evaluation')

    def restore_training(model, optimizer, horizon):
        step, epoch, offset = restart.restore(model, optimizer, horizon)
        # A clean stop on the epoch's final update may precede upstream's
        # deployment-checkpoint/validation callback. Finish that callback now.
        if restart.resume and epoch and not offset and not (output / f"checkpoint-{step}/validation.json").is_file():
            checkpoint(output / f"checkpoint-{step}", model, training={
                "checkpoint_epoch": epoch, "checkpoint_step": step, "examples_per_epoch": len(train_rows),
                "seed": seed, "effective_document_batch": 4, "learning_rate_horizon_updates": horizon})
        # Recover a crash between checkpoint validation and quality request.
        if restart.resume and epoch and not offset:
            from .scheduled_validation import request_epoch_quality
            if request_epoch_quality(output, output/f'checkpoint-{step}',
                    dict(checkpoint_epoch=epoch,checkpoint_step=step), evaluation_plan):
                raise TrainingInterrupted('epoch_quality_evaluation')
        # Commit the imported parent weights/RMS and fresh optimizer before
        # long baseline validation. A crash here can resume from step zero.
        if not restart.resume and restart.initialize:
            restart.after_update(model, optimizer, step, epoch, offset, horizon, force=True)
        if schedule is not None and not schedule.state['baseline_done']:
            validation_started = time.perf_counter()
            if pair_pool is not None:
                pair_pool.set_sequence(validation_rows)
            schedule.run(model,target,pairs,validation_diagnostics,step=step,kind='baseline',gpu_ids=gpu_ids)
            epoch_clock['started'] += time.perf_counter()-validation_started
        return step, epoch, offset

    def after_update(model, optimizer, step, epoch, offset, horizon):
        due = schedule is not None and offset < len(train_rows) and schedule.due()
        # An evaluator failure must not discard the updates preceding it.
        restart.after_update(model, optimizer, step, epoch, offset, horizon, force=due)
        # The epoch callback already validates the same complete holdout.
        if not due:
            return
        atomic_json(output/'status.json',dict(status='validating',kind='hourly',step=step,
            epoch=epoch+1,examples=len(validation_rows),latest_restart=restart.last_checkpoint))
        validation_started = time.perf_counter()
        if pair_pool is not None:
            pair_pool.set_sequence(validation_rows)
        try:
            schedule.run(model,target,pairs,validation_diagnostics,step=step,kind='hourly',gpu_ids=gpu_ids)
        finally:
            import random
            order = list(range(len(train_rows)))
            random.Random(seed * 1_000_003 + epoch * 10_007).shuffle(order)
            begin_epoch(epoch, order, offset)
            epoch_clock['started'] += time.perf_counter()-validation_started
        atomic_json(output/'status.json',dict(status='training',step=step,epoch=epoch+1,
            next_offset=offset,latest_restart=restart.last_checkpoint))

    bindings.update({"AutoModelForCausalLM": types.SimpleNamespace(
                        from_pretrained=lambda *a, **kw: _FrozenDispatchedTarget(target)),
                     "TargetConfig": GeometryAdapter, "RepairConfig": RepairConfig,
                     "CacheRepair": construct, "compute_stats": stats, "save_repairer": checkpoint,
                     "restore_training": restore_training, "after_update": after_update,
                     "learning_rate":lambda step,horizon:official.learning_rate(step,horizon)*learning_rate_scale,
                     "save_statistics": restart.save_statistics,
                     "torch": _Proxy(torch, optim=_Proxy(torch.optim, AdamW=optimizer_factory))})
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for gpu in gpu_ids:
        torch.cuda.reset_peak_memory_stats(gpu)
    previous_argv = sys.argv
    restart.install()
    try:
        output.mkdir(parents=True, exist_ok=bool(resume_from))
        atomic_json(output / "run-identity.json", identity)
        if provenance is not None:
            write_json(output / "target-provenance.json", provenance)
        with tempfile.TemporaryDirectory(prefix=".qwen38-training-", dir=output.parent) as temporary:
            temporary = Path(temporary)
            write_json(temporary / "config.json", settings)
            with (temporary / "train.jsonl").open("x") as stream:
                for row in train_rows:
                    stream.write(json.dumps(row) + "\n")
            sys.argv = ["cacherepair.train", "--model", model_id,
                        "--data", str(temporary / "train.jsonl"), "--config", str(temporary / "config.json"),
                        "--output", str(output), "--epochs", str(epochs), "--schedule-epochs", str(schedule_epochs)]
            if restart.resume or restart.initialize:
                source = Path(restart.resume or restart.initialize)
                tensors = load_file(str(source / "model.safetensors"))
                save_file({name: tensors[name] for name in ("sigma_stale", "sigma_delta")},
                          str(temporary / "stats.safetensors"))
                sys.argv.extend(["--stats", str(temporary / "stats.safetensors")])
            elif (statistics_file := restart.existing_statistics()) is not None:
                sys.argv.extend(["--stats", str(statistics_file)])
            resumed_main(official.main, bindings)()
            shutil.copyfile(temporary / "config.json", output / "training-config.json")
        synchronize()
        report["upstream_loop_including_stats_validation_seconds"] = time.perf_counter() - started
        if validate_train:
            diagnostic_rows = train_rows
            if train_diagnostic_examples is not None and train_diagnostic_examples < len(train_rows):
                import random
                indices = sorted(random.Random(seed).sample(range(len(train_rows)), train_diagnostic_examples))
                diagnostic_rows = [train_rows[i] for i in indices]
                report["training_diagnostic_row_indices"] = indices
            if pair_pool is not None:
                pair_pool.set_sequence(diagnostic_rows)
            report["training_diagnostics"] = validation_diagnostics(trained["model"], target, diagnostic_rows, pairs)
        report["target_parameter_sample_after"] = _parameter_sample(target)
        if report["target_parameter_sample_after"] != report["target_parameter_sample_before"]:
            raise AssertionError("Sampled frozen target weights changed")
        report["target_frozen"] = not any(parameter.requires_grad or parameter.grad is not None for parameter in target.parameters())
        report["gpu_memory"] = {str(gpu): {"peak_allocated_bytes": torch.cuda.max_memory_allocated(gpu),
                                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(gpu)} for gpu in gpu_ids}
        report["total_seconds_including_train_diagnostics"] = time.perf_counter() - started
        write_json(output / "diagnostics.json", report)
        atomic_json(output / "status.json", {"status": "completed", "latest_restart": restart.last_checkpoint})
        return report
    except TrainingInterrupted as error:
        report.update({"interrupted": True, "reason": str(error), "latest_restart": restart.last_checkpoint})
        atomic_json(output / "status.json", {"status": "interrupted", "latest_restart": restart.last_checkpoint})
        write_json(output / "interrupted.json", report)
        return report
    except Exception as error:
        if output.exists():
            write_json(output / "failure.json", {**report, "error": repr(error)})
        raise
    finally:
        sys.argv = previous_argv
        restart.close()


def load_native_target(path, gpu_ids, *, attention_backend="eager", require_fast=False):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    if require_fast:
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
        if not is_fast_path_available:
            raise RuntimeError("Fast GDN kernels are required; refusing the Torch fallback")

    config = Qwen3_5Config.from_pretrained(path, local_files_only=True)
    count = config.text_config.num_hidden_layers
    mapping = {"model.visual": "cpu", "model.language_model.embed_tokens": f"cuda:{gpu_ids[0]}",
               "model.language_model.rotary_emb": f"cuda:{gpu_ids[0]}",
               "model.language_model.norm": f"cuda:{gpu_ids[-1]}", "lm_head": f"cuda:{gpu_ids[-1]}"}
    mapping.update({f"model.language_model.layers.{i}": f"cuda:{gpu_ids[min(len(gpu_ids) - 1, i * len(gpu_ids) // count)]}"
                    for i in range(count)})
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, config=config, dtype=torch.bfloat16, device_map=mapping,
        attn_implementation=attention_backend, local_files_only=True, output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"Target checkpoint did not load completely: {loading}")
    return model.requires_grad_(False).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/Qwen3.8-27B-StageA.json"))
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--stats-examples", type=int)
    parser.add_argument("--max-pair-seconds", type=float, default=120)
    parser.add_argument("--cache-pairs", action="store_true", help="Run-local disk KV cache; never keeps the corpus in GPU memory")
    parser.add_argument("--cpu-threads", type=int, default=1)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--resume", help="Committed restart checkpoint including optimizer/RNG/data cursor")
    initial.add_argument("--init-from", help="Repairer weights and RMS only; new optimizer and LR schedule")
    parser.add_argument("--checkpoint-steps", type=int, default=250)
    parser.add_argument("--keep-restarts", type=int, default=2)
    parser.add_argument("--stop-after-steps", type=int, help="Clean stop at an optimizer update, useful for restart checks")
    parser.add_argument("--pair-groups", help="Disjoint logical GPU pairs, e.g. 0,1;2,3;4,5")
    parser.add_argument('--supervision-backend', choices=('native','vllm'), default='native')
    parser.add_argument('--vllm-memory-utilization', type=float, default=.70)
    parser.add_argument('--vllm-max-batched-tokens', type=int, default=32768)
    parser.add_argument('--vllm-prefetch-per-worker',type=int,default=1)
    parser.add_argument('--vllm-prefetch-limit',type=int)
    parser.add_argument('--vllm-batch-examples',type=int,default=1)
    parser.add_argument('--vllm-batch-input-tokens',type=int,default=8192)
    parser.add_argument('--vllm-remote-cache-gib',type=float,default=1)
    parser.add_argument('--vllm-remote-max-seqs',type=int,default=32)
    parser.add_argument('--vllm-dynamic-batching',action='store_true')
    parser.add_argument('--vllm-worker-pending-tokens',type=int)
    parser.add_argument('--vllm-global-pending-tokens',type=int)
    parser.add_argument('--vllm-scheduler',choices=('round_robin','least_work'),default='round_robin')
    parser.add_argument("--chunk-batch-size", type=int, default=1)
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--require-fast-gdn", action="store_true")
    parser.add_argument("--pack-chunks", action="store_true")
    parser.add_argument("--pipeline-depth", type=int, default=0)
    parser.add_argument("--statistics-from", help="Import a numerically verified RMS prefix into a fresh experiment")
    parser.add_argument("--statistics-validation", help="Real-target bitwise KV and RMS equivalence report")
    parser.add_argument('--memory-efficient-repairer',action='store_true',
                        help='Exact FP32 SDPA, tiled encoder/head loss and activation recomputation')
    parser.add_argument('--repairer-tile',type=int,default=256)
    parser.add_argument('--evaluation-plan',type=Path,help='Hourly holdout MSE and epoch quality schedule')
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if args.statistics_from and (args.resume or args.init_from or not args.statistics_validation):
        parser.error("statistics-from requires a fresh experiment and statistics-validation")
    if args.supervision_backend == 'vllm' and (args.pipeline_depth or args.pack_chunks):
        parser.error('Native pack/pipeline options cannot be used with vLLM')
    if args.supervision_backend == 'vllm' and not args.pair_groups:
        parser.error('vLLM requires explicit --pair-groups')
    torch.set_num_threads(args.cpu_threads)
    dataset, output = Path(args.dataset), Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}")
    settings = json.loads(Path(args.config).read_text())
    manifest = json.loads((dataset / "dataset.json").read_text())
    if manifest["tokenizer"] != local_tokenizer_spec(args.model):
        raise ValueError("Dataset tokenizer differs from the actual target tokenizer")
    splits = {}
    for split in ("train", "validation"):
        path = dataset / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files_sha256"][path.name]:
            raise ValueError("Training/validation file differs from its decontaminated manifest")
        splits[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data_settings = settings.get("data", {})
    for split, rows in splits.items():
        if len(rows) != data_settings.get(f"{split}_examples", len(rows)):
            raise ValueError(f"{split} size differs from the experiment configuration")
        for row in rows:
            chunks = row["chunks"]
            if not 2 <= len(chunks) <= data_settings.get("max_chunks", float("inf")):
                raise ValueError(f"{split} chunk count differs from the experiment configuration")
            if sum(map(len, chunks)) > data_settings.get("max_total_tokens", float("inf")):
                raise ValueError(f"{split} total token count exceeds the experiment configuration")
            if any(not chunk or len(chunk) > data_settings.get("max_chunk_tokens", float("inf")) for chunk in chunks):
                raise ValueError(f"{split} chunk length exceeds the experiment configuration")
    if settings["training"].get("effective_batch_size", 4) != 4:
        raise ValueError("The unchanged upstream loop requires effective_batch_size=4")
    stats_examples = args.stats_examples if args.stats_examples is not None else settings["training"].get("statistics_examples")
    gpu_ids = tuple(int(value) for value in args.gpus.split(","))
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("Specify distinct target GPU IDs")
    # PairPool readiness is published before run_upstream creates its output.
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(gpu_ids[0])
    started = time.perf_counter()
    if args.supervision_backend == 'vllm':
        from .vllm_target import VllmTarget
        target = VllmTarget(args.model, f'cuda:{gpu_ids[0]}', settings['target_model_id'])
    else:
        target = load_native_target(args.model, gpu_ids, attention_backend=args.attention_backend,
                                    require_fast=args.require_fast_gdn)
    load_seconds = time.perf_counter() - started
    geometry = TargetGeometry.from_target(target, target_model_id=settings["target_model_id"])
    if list(geometry.kv_shape) != settings["expected_kv_shape"]:
        raise ValueError("Loaded target geometry differs from the experiment assertion")
    pool = None
    if args.pair_groups:
        from .pair_pool import PairPool
        groups = [tuple(map(int, value.split(","))) for value in args.pair_groups.split(";")]
        if args.supervision_backend == "native" and groups[0] != gpu_ids:
            raise ValueError("First supervision replica must match the training target GPUs")
        if args.supervision_backend == 'vllm':
            from .vllm_pool import VllmPairPool
            actual_prefetch = args.vllm_prefetch_per_worker
            pool = VllmPairPool(target, geometry, groups, model_path=args.model,
                memory_utilization=args.vllm_memory_utilization,
                max_batched_tokens=args.vllm_max_batched_tokens,
                prefetch_per_worker=actual_prefetch,scheduler=args.vllm_scheduler,max_pending_pairs=args.vllm_prefetch_limit,
                max_batch_examples=args.vllm_batch_examples,max_batch_input_tokens=args.vllm_batch_input_tokens,
                remote_kv_cache_bytes=int(args.vllm_remote_cache_gib*1024**3),remote_max_num_seqs=args.vllm_remote_max_seqs,dynamic_batching=args.vllm_dynamic_batching,
                max_pending_input_tokens_per_worker=args.vllm_worker_pending_tokens,
                max_pending_input_tokens=args.vllm_global_pending_tokens,
                log_path=output.parent/(output.name+'-pair-events.jsonl'))
        else:
            pool = PairPool(target, geometry, groups, model_path=args.model,
                chunk_batch_size=args.chunk_batch_size, attention_backend=args.attention_backend,
                pack_chunks=args.pack_chunks, pipeline_depth=args.pipeline_depth,
                log_path=output.parent / (output.name + "-pair-events.jsonl"))
    def version_or_none(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    backend_spec = {"attention_backend": args.attention_backend, "chunk_batch_size": args.chunk_batch_size,
                    "pack_chunks": args.pack_chunks,
                    "require_fast_gdn": args.require_fast_gdn,
                    "gdn_arithmetic": "official HF/FLA fused BF16 target; FP32 recurrent state",
                    "kernel_versions": {name: version_or_none(name) for name in
                                        ("flash-linear-attention", "fla-core", "causal-conv1d")}}
    if args.supervision_backend == 'vllm':
        backend_spec.update(supervision_backend='vllm', tensor_parallel_size=2,
            kv_cache_dtype='bfloat16', mamba_ssm_cache_dtype='float32',kv_cache_memory_bytes=1024**3,
            gdn_arithmetic='vLLM fused BF16 GDN; FP32 recurrent state',
            transfer='BF16 CUDA IPC/P2P; canonical FP32 on repairer GPU',
            memory_utilization=args.vllm_memory_utilization,
            max_batched_tokens=args.vllm_max_batched_tokens,
            prefetch_per_worker=args.vllm_prefetch_per_worker,scheduler=args.vllm_scheduler,max_pending_pairs=args.vllm_prefetch_limit,
            max_batch_examples=args.vllm_batch_examples,max_batch_input_tokens=args.vllm_batch_input_tokens,
            remote_kv_cache_bytes=int(args.vllm_remote_cache_gib*1024**3),remote_max_num_seqs=args.vllm_remote_max_seqs,dynamic_batching=args.vllm_dynamic_batching,
            max_pending_input_tokens_per_worker=args.vllm_worker_pending_tokens,
            max_pending_input_tokens=args.vllm_global_pending_tokens,
            shared_max_batch_examples=1,shared_max_batched_tokens=min(32768,args.vllm_max_batched_tokens),
            shared_long_sequence_safety_threshold=8192,
            enable_prefix_caching=False,enable_chunked_prefill=False,enforce_eager=True,
            vllm_version=version_or_none('vllm'))
    if args.memory_efficient_repairer or settings['training'].get('memory_efficient_repairer'):
        backend_spec.update(memory_efficient_repairer=True,repairer_tile=args.repairer_tile)
    if args.pipeline_depth:
        backend_spec["pipeline_depth"] = args.pipeline_depth
        backend_spec["root_training_pipeline_depth"] = min(3, args.pipeline_depth)
    if pool is None and args.require_fast_gdn:
        raise ValueError("Fast production jobs require explicit --pair-groups, including one pair")
    if pool is not None:
        atomic_json(output.parent / "resources-ready.json", {"time_unix": time.time(), "groups": groups,
                    "backend_spec": backend_spec, "runtime_prefetch_per_worker": getattr(pool,"prefetch_per_worker",None),
                    "pid": __import__("os").getpid()})
    try:
        report = run_upstream(
            target, splits["train"], splits["validation"], output,
            model_id=settings["target_model_id"], repair_config=RepairConfig(**settings["repair_config"]),
            seed=settings["training"]["seed"], initialization_seed=settings["training"]["initialization_seed"],
            epochs=settings["training"]["epochs"], schedule_epochs=settings["training"]["schedule_epochs"],
            stats_examples=stats_examples, gpu_ids=gpu_ids, max_pair_seconds=args.max_pair_seconds,
            cache_pairs=args.cache_pairs,
            resume_from=args.resume, init_from=args.init_from, checkpoint_steps=args.checkpoint_steps,
            keep_restarts=args.keep_restarts, stop_after_steps=args.stop_after_steps,
            provenance=local_checkpoint_spec(args.model),
            train_diagnostic_examples=settings["training"].get("train_diagnostic_examples"),
            pair_pool=pool, backend_spec=backend_spec,
            statistics_from=args.statistics_from, statistics_validation=args.statistics_validation,
            learning_rate_scale=settings['training'].get('learning_rate_scale',1.0),
            evaluation_plan=json.loads(args.evaluation_plan.read_text()) if args.evaluation_plan else None,
            scope=settings.get("scope", "Training-source residual validation; no generation-quality or TTFT experiment"),
        )
    finally:
        if pool is not None:
            pool.close()
    if report.get("interrupted"):
        print(json.dumps({"output": str(output), "status": "interrupted", "latest_restart": report["latest_restart"]}), flush=True)
        return
    report.update({"model_load_seconds": load_seconds, "dataset": str(dataset.resolve()),
                   "tokenizer": manifest["tokenizer"], "model_path": str(Path(args.model).resolve()),
                   "dataset_files_sha256": manifest["files_sha256"],
                   "experiment_config": settings,
                   "target_provenance": local_checkpoint_spec(args.model),
                   "environment": {name: importlib.metadata.version(name) for name in (
                       "torch", "transformers", "huggingface-hub", "safetensors", "accelerate")}})
    write_json(output / "target-provenance.json", report["target_provenance"])
    write_json(output / "experiment-config.json", settings)
    for checkpoint in report["checkpoints"]:
        write_json(Path(checkpoint["path"]) / "target-provenance.json", report["target_provenance"])
    write_json(output / "diagnostics.json", report)
    print(json.dumps({"output": str(output), "checkpoints": report["checkpoints"], "target_frozen": report["target_frozen"]}), flush=True)


if __name__ == "__main__":
    main()
