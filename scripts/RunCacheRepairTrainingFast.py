"""Run CacheRepair supervision on 2/4/6 GPUs until explicitly stopped.

Three, two, or one frozen two-GPU targets feed the official single optimizer.
An optional explicit deadline pauses; the default has no automatic stop.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.RunCacheRepairTraining import atomic_json, available_gpus, gpu_snapshot, sha256_file


def due_for_pause(policy, now):
    deadline = policy.get("deadline_unix", policy.get("six_gpu_deadline_unix"))
    return deadline is not None and now >= deadline


def gpu_groups(ids):
    if len(ids) not in (2, 4, 6) or len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError("Specify 2, 4, or 6 distinct physical GPU IDs")
    return ";".join(f"{i},{i + 1}" for i in range(0, len(ids), 2))


def supervision_groups(ids, *, dedicated=False, backend='native'):
    groups=gpu_groups(ids)
    if not dedicated:
        return groups
    if backend!='vllm':
        raise ValueError('Dedicated repairer requires vLLM')
    # A two-GPU allocation must share GPU0 to fit the 27B target. A larger
    # allocation preserves the dedicated-GPU0 mode and uses complete pairs.
    return ';'.join(f'{i},{i+1}' for i in range(2,len(ids),2)) if len(ids)>=4 else groups


def protected_source_paths(root, pipeline_depth=0, evaluation=False):
    paths = sorted((root / "methods/CacheRepair/qwen38").glob("*.py")) + [
        root / "methods/CacheRepair/backend/cacherepair/train.py",
        root / "methods/CacheRepair/backend/cacherepair/model.py"]
    if pipeline_depth:
        paths.append(root / "methods/CacheRepair/qwen38/experimental/pipeline.py")
    if evaluation:
        paths.append(root / 'scripts/EvaluateCacheRepairEpoch.py')
    return paths


def immutable_plan(plan):
    # GPU allocation is allowed to change on exact training/statistics resume.
    return {**{key: plan[key] for key in ("model", "dataset", "config", "python", "chunk_batch_size",
                                       "config_sha256", "dataset_manifest_sha256", "initialization")},
            "supervision_backend":plan.get('supervision_backend','native'),
            "vllm_memory_utilization":plan.get('vllm_memory_utilization',.70),
            "vllm_max_batched_tokens":plan.get('vllm_max_batched_tokens',32768),
            "vllm_prefetch_per_worker":plan.get('vllm_prefetch_per_worker',1),
            "vllm_scheduler":plan.get('vllm_scheduler','round_robin'),
            "vllm_prefetch_limit":plan.get('vllm_prefetch_limit'),
            "vllm_batch_examples":plan.get('vllm_batch_examples',1),
            "vllm_batch_input_tokens":plan.get('vllm_batch_input_tokens',8192),
            "vllm_remote_cache_gib":plan.get('vllm_remote_cache_gib',1),
            "vllm_remote_max_seqs":plan.get('vllm_remote_max_seqs',32),
            "vllm_dynamic_batching":plan.get('vllm_dynamic_batching',False),
            "vllm_worker_pending_tokens":plan.get('vllm_worker_pending_tokens'),
            "vllm_global_pending_tokens":plan.get('vllm_global_pending_tokens'),
            "dedicated_repairer":plan.get('dedicated_repairer',False),
            "pipeline_depth": plan.get("pipeline_depth", 0),
            "statistics_from": plan.get("statistics_from"),
            "statistics_validation": plan.get("statistics_validation"),
            "init_from":plan.get('init_from'),
            "init_from_sha256":plan.get('init_from_sha256'),
            "pack_chunks":plan.get('pack_chunks',True),
            "memory_efficient_repairer":plan.get('memory_efficient_repairer',False),
            "evaluation_plan_sha256":plan.get('evaluation_plan_sha256')}


def quality_request(job):
    path = Path(job)/'training/epoch-quality-request.json'
    if not path.exists():
        return None
    request = json.loads(path.read_text())
    done = Path(job)/'quality-evaluations'/f"step-{request['step']:08d}"/'completed.json'
    if not done.exists():
        return path
    result=json.loads(done.read_text())
    if result.get('status')!='completed' or result.get('checkpoint')!=request['checkpoint']:
        raise ValueError('Invalid epoch quality completion')
    return None


def initialization_arguments(training,init_from=None,statistics_from=None,statistics_validation=None):
    if Path(training).exists():return ['--resume','auto']
    if init_from:return ['--init-from',str(init_from)]
    if statistics_from:
        if not statistics_validation:raise ValueError('Imported statistics require real-target equivalence evidence')
        return ['--statistics-from',statistics_from,'--statistics-validation',statistics_validation]
    return []


def allocation_policy(previous, hours, physical, new_window=False):
    old = {} if new_window else previous
    return {"hours": hours, "after_window": "pause_for_review" if hours is not None else "continue_until_requested",
            "physical_ids": physical,
            "started_unix": old.get("started_unix", old.get("six_gpu_started_unix")),
            "deadline_unix": old.get("deadline_unix", old.get("six_gpu_deadline_unix")) if hours is not None else None,
            "clock_scope": "From all requested target replicas ready; includes full RMS and gradient updates"}


def process_identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[0], fields[19]


def validate_attachment(pid, job, physical):
    process = Path(f"/proc/{pid}")
    if process.stat().st_uid != os.getuid():
        raise ValueError("Attached training process has a different owner")
    command = (process / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    ready = json.loads((job / "resources-ready.json").read_text())
    output = command[command.index("--output") + 1] if "--output" in command else None
    if (ready["pid"] != pid or "methods.CacheRepair.qwen38.train" not in command
            or output is None or Path(output).resolve() != job / "training" or os.getpgid(pid) != pid):
        raise ValueError("Attached PID is not this job's independent training process")
    environment = dict(part.split("=", 1) for part in (process / "environ").read_bytes().decode().split("\0") if "=" in part)
    if environment.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, physical)):
        raise ValueError("Attached training process uses different GPUs")
    state, birth = process_identity(pid)
    if state == "Z":
        raise ValueError("Attached training process has exited")
    return birth


class AttachedChild:
    """Monitor an existing owned trainer without reloading or interrupting it."""
    def __init__(self, pid, birth, training):
        self.pid, self.birth, self.training, self.returncode = pid, birth, training, None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        try:
            state, birth = process_identity(self.pid)
            running = state != "Z" and birth == self.birth
        except FileNotFoundError:
            running = False
        if not running:
            status = self.training / "status.json"
            state = json.loads(status.read_text())["status"] if status.exists() else "unknown"
            self.returncode = 0 if state in ("completed", "interrupted") else 1
        return self.returncode

    def send_signal(self, number):
        if self.poll() is None:
            os.kill(self.pid, number)

    def wait(self, timeout):
        until = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= until:
                raise subprocess.TimeoutExpired(f"attached trainer {self.pid}", timeout)
            time.sleep(.2)
        return self.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=".cache/cacherepair-qwen-fast-env/bin/python")
    parser.add_argument("--physical-gpus", default="0,1,2,3,4,5")
    parser.add_argument("--run-hours", "--six-hours", dest="six_hours", type=float,
                        help="Optional automatic pause; omitted means run until explicitly stopped")
    parser.add_argument('--supervision-backend',choices=('native','vllm'),default='native')
    parser.add_argument('--vllm-memory-utilization',type=float,default=.70)
    parser.add_argument('--vllm-max-batched-tokens',type=int,default=32768)
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
    parser.add_argument('--dedicated-repairer',action='store_true',help='Reserve GPU0 for repairer on 4/6 GPUs; automatically share it on a 2-GPU resume')
    parser.add_argument("--chunk-batch-size", type=int, default=8)
    parser.add_argument("--new-window", action="store_true", help="Explicitly start a fresh allocation after review")
    parser.add_argument("--attach-pid", type=int, help="Adopt this job's running trainer without restarting it")
    parser.add_argument("--pipeline-depth", type=int, default=0)
    parser.add_argument("--statistics-from")
    parser.add_argument("--statistics-validation")
    parser.add_argument('--init-from',help='New fine-tuning job from checkpoint weights and RMS')
    parser.add_argument('--pack-chunks',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--memory-efficient-repairer',action='store_true')
    parser.add_argument('--monitor-seconds',type=int,default=3600)
    parser.add_argument('--evaluation-plan',type=Path)
    args = parser.parse_args()
    if args.six_hours is not None and args.six_hours <= 0:
        parser.error("run-hours must be positive")
    if args.init_from and args.statistics_from:parser.error('init-from already inherits checkpoint RMS')
    if args.pipeline_depth and not args.pack_chunks:parser.error('pipeline requires packed chunks')
    if args.supervision_backend=='vllm' and (args.pipeline_depth or args.pack_chunks):
        parser.error('vLLM requires --no-pack-chunks and --pipeline-depth 0')
    if args.monitor_seconds<1:parser.error('monitor-seconds must be positive')
    if args.evaluation_plan:
        evaluation_plan=json.loads(args.evaluation_plan.read_text())
        if not evaluation_plan.get('enabled'):parser.error('Evaluation plan is disabled')
        from scripts.EvaluateCacheRepairEpoch import verify_quality_inputs
        verify_quality_inputs(evaluation_plan)
    else:
        evaluation_plan=None
    if args.init_from:args.init_from=str(Path(args.init_from).resolve())
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    job = Path(args.job).resolve()
    job.mkdir(parents=True, exist_ok=True)
    lock = (job / "fast-supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    physical = list(map(int, args.physical_gpus.split(",")))
    groups = supervision_groups(physical,dedicated=args.dedicated_repairer,backend=args.supervision_backend)
    attached_birth = validate_attachment(args.attach_pid, job, physical) if args.attach_pid else None
    validation_name = 'fast-validation-vllm.json' if args.supervision_backend=='vllm' else 'fast-validation-native-norm.json'
    validation = json.loads((job / validation_name).read_text())
    if validation["status"] != "passed":
        raise ValueError("Real-target fast-path validation did not pass")
    test = json.loads((job / "six-gpu-smoke.json").read_text())
    if not test.get("passed"):
        raise ValueError("Six-GPU training and two-GPU resume smoke did not pass")
    source_paths = protected_source_paths(root, args.pipeline_depth, evaluation_plan is not None)
    hashes = {str(path): sha256_file(path) for path in source_paths}
    manifest = job / "fast-source-provenance.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        previous_training = {key: value for key, value in previous.items() if key != str(Path(__file__).resolve())}
        if previous_training != hashes:
            raise ValueError("Fast training source code changed")
        if previous != hashes:
            atomic_json(job / "fast-source-provenance-before-pause-fix.json", previous)
    atomic_json(manifest, hashes)
    with (job / "supervisor-provenance.jsonl").open("a") as stream:
        stream.write(json.dumps({"time_unix": time.time(), "path": str(Path(__file__).resolve()),
                                 "sha256": sha256_file(__file__)}) + "\n")
    plan = immutable_plan({**vars(args), "config_sha256": sha256_file(args.config),
            "dataset_manifest_sha256": sha256_file(Path(args.dataset) / "dataset.json"),
            "initialization":(f'weights/RMS from {args.init_from}; fresh optimizer and LR schedule'
                if args.init_from else "fresh; old Torch-fallback statistics preserved separately, not mixed"),
            "init_from_sha256":sha256_file(Path(args.init_from)/'model.safetensors') if args.init_from else None,
            "evaluation_plan_sha256":sha256_file(args.evaluation_plan) if args.evaluation_plan else None})
    plan_path = job / "fast-launch-plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if immutable_plan(previous) != plan:
            raise ValueError("Fast training plan differs")
        if previous != plan:
            atomic_json(job / "fast-launch-plan-before-pause-fix.json", previous)
    atomic_json(plan_path, plan)
    atomic_json(job / "fast-supervisor.json", {"pid": os.getpid(), "job": str(job)})
    policy_path = job / "resource-policy.json"
    previous = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    with (job / "allocation-history.jsonl").open("a") as stream:
        stream.write(json.dumps({"time_unix": time.time(), "previous": previous,
                                 "physical_ids": physical, "new_window": args.new_window}) + "\n")
    policy = allocation_policy(previous, args.six_hours, physical, args.new_window)
    atomic_json(policy_path, policy)
    stopped = threading.Event()
    child = None
    gpu_handles = {}

    def status(state, **details):
        event = {"time": datetime.now(timezone.utc).isoformat(), "status": state,
                 "resource_policy": policy, **details}
        atomic_json(job / "job-status.json", event)
        with (job / "fast-events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    def signal_handler(number, frame):
        stopped.set()
        if child is not None and child.poll() is None:
            child.send_signal(number)

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, signal_handler)

    try:
        if due_for_pause(policy, time.time()):
            status("paused_for_review", reason="Allocation expired; use --new-window after review")
            return
        if args.attach_pid:
            for index in physical:
                handle = Path(f"/tmp/kvbench-cacherepair-gpu-{index}.lock").open("a")
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                gpu_handles[index] = handle
        while not gpu_handles and not stopped.is_set():
            snapshot = gpu_snapshot()
            if set(physical) <= set(available_gpus(snapshot, set(physical))):
                for index in physical:
                    handle = Path(f"/tmp/kvbench-cacherepair-gpu-{index}.lock").open("a")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        break
                    gpu_handles[index] = handle
                if set(physical) <= set(gpu_handles):
                    break
                for handle in gpu_handles.values():
                    handle.close()
                gpu_handles.clear()
            status("waiting_for_gpu", snapshot=snapshot, required=physical, next_check_seconds=1800)
            stopped.wait(1800)
            if due_for_pause(policy, time.time()):
                status("paused_for_review", reason="Allocation expired while queued")
                return
        if stopped.is_set():
            status("interrupted")
            return
        if any(sha256_file(path) != digest for path, digest in hashes.items()):
            raise ValueError("Source changed while the job was queued")
        if sha256_file(args.config) != plan["config_sha256"]:
            raise ValueError("Training configuration changed")
        training = job / "training"
        ready = job / "resources-ready.json"
        if ready.exists() and not args.attach_pid:
            ready.rename(job / f"resources-ready-{time.time_ns()}.json")
        command = [args.python, "-u", "-m", "methods.CacheRepair.qwen38.train", "--model", args.model,
                   "--dataset", args.dataset, "--config", args.config, "--output", str(training),
                   "--gpus", "0,1", "--pair-groups", groups, "--require-fast-gdn", "--attention-backend", "sdpa",
                   "--chunk-batch-size", str(args.chunk_batch_size),
                   "--checkpoint-steps", "100", "--keep-restarts", "2"]
        if args.supervision_backend=='vllm':
            command.extend(['--supervision-backend','vllm','--vllm-memory-utilization',
                str(args.vllm_memory_utilization),'--vllm-max-batched-tokens',str(args.vllm_max_batched_tokens),
                '--vllm-prefetch-per-worker',str(args.vllm_prefetch_per_worker),'--vllm-scheduler',args.vllm_scheduler])
        if args.supervision_backend=='vllm':
            command.extend(['--vllm-batch-examples',str(args.vllm_batch_examples),
                '--vllm-batch-input-tokens',str(args.vllm_batch_input_tokens),
                '--vllm-remote-cache-gib',str(args.vllm_remote_cache_gib),
                '--vllm-remote-max-seqs',str(args.vllm_remote_max_seqs)])
            for flag,value in [('--vllm-worker-pending-tokens',args.vllm_worker_pending_tokens),
                               ('--vllm-global-pending-tokens',args.vllm_global_pending_tokens)]:
                if value is not None:command.extend([flag,str(value)])
        if args.vllm_dynamic_batching:command.append('--vllm-dynamic-batching')
        if args.vllm_prefetch_limit is not None:
            command.extend(['--vllm-prefetch-limit',str(args.vllm_prefetch_limit)])
        if args.pack_chunks:command.append('--pack-chunks')
        if args.memory_efficient_repairer:command.append('--memory-efficient-repairer')
        if args.evaluation_plan:command.extend(['--evaluation-plan',str(args.evaluation_plan.resolve())])
        if args.pipeline_depth:
            command.extend(["--pipeline-depth", str(args.pipeline_depth)])
        command.extend(initialization_arguments(training,args.init_from,args.statistics_from,args.statistics_validation))
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, physical)),
                       "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
        if args.supervision_backend=='vllm':
            environment['PYTORCH_ALLOC_CONF']='backend:native'
            environment.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        status("adopting_running_trainer" if args.attach_pid else "loading_targets", physical_ids=physical,
               command=command, attached_pid=args.attach_pid)
        deadline_stop = False
        shutdown_started = None
        while True:
            with (job / "training-fast.log").open("a", buffering=1) as log:
                child = AttachedChild(args.attach_pid, attached_birth, training) if args.attach_pid else subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                    stdin=subprocess.DEVNULL, start_new_session=True)
                last_monitor = 0.
                announced_ready = False
                while child.poll() is None:
                    now = time.time()
                    if ready.exists() and not announced_ready:
                        if policy["started_unix"] is None:
                            began = json.loads(ready.read_text())["time_unix"]
                            policy.update(started_unix=began, deadline_unix=began + args.six_hours * 3600
                                          if args.six_hours is not None else None)
                            atomic_json(policy_path, policy)
                        announced_ready = True
                        status("running", physical_ids=physical)
                    if due_for_pause(policy, now) and not deadline_stop and not stopped.is_set():
                        deadline_stop = True
                        shutdown_started = now
                        child.send_signal(signal.SIGTERM)
                        status("checkpointing_for_review", reason="Allocation ended; pause and release all GPUs")
                    if stopped.is_set() and shutdown_started is None:
                        shutdown_started = now
                    if shutdown_started is not None and now - shutdown_started > 600:
                        os.killpg(child.pid, signal.SIGKILL)
                        status("checkpoint_timeout", reason="Resume last durable snapshot after shutdown timeout")
                    if now - last_monitor >= args.monitor_seconds:
                        item = {"time_unix": now, "physical_ids": physical, "snapshot": gpu_snapshot()}
                        if (training / "status.json").exists():
                            item["training"] = json.loads((training / "status.json").read_text())
                        with (job / "gpu-telemetry.jsonl").open("a") as stream:
                            stream.write(json.dumps(item) + "\n")
                        last_monitor = now
                    stopped.wait(1.) if not stopped.is_set() else time.sleep(.2)
                code = child.returncode
                child = None
            request = quality_request(job) if evaluation_plan else None
            if code or stopped.is_set() or deadline_stop or request is None:
                break
            training_status = json.loads((training/'status.json').read_text())
            if training_status['status'] != 'interrupted':
                raise ValueError('Epoch quality requires the trainer to exit cleanly')
            reason=json.loads((training/'interrupted.json').read_text()).get('reason')
            if reason != 'epoch_quality_evaluation':
                break  # External stop must never automatically restart training.
            latest=json.loads((training/'latest.json').read_text())
            epoch_request=json.loads(request.read_text())
            if latest['step'] != epoch_request['step']:
                raise ValueError('Epoch quality pause lost its exact training cursor')
            status('epoch_quality_evaluation',request=epoch_request)
            # Targets/prefetch workers have exited. Keep the GPU leases while
            # Engine runs on the selected physical devices in its own process.
            eval_command=[sys.executable,str(root/'scripts/EvaluateCacheRepairEpoch.py'),
                '--job',str(job),'--plan',str(args.evaluation_plan.resolve()),
                '--request',str(request),'--physical-gpus',','.join(map(str,physical))]
            eval_env={k:v for k,v in os.environ.items() if k not in
                ('CUDA_VISIBLE_DEVICES','PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')}
            with (job/'epoch-quality.log').open('a',buffering=1) as log:
                child=subprocess.Popen(eval_command,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                    env=eval_env,stdin=subprocess.DEVNULL,start_new_session=True)
                while child.poll() is None:
                    if stopped.wait(1.):
                        child.send_signal(signal.SIGTERM)
                eval_code=child.returncode
                child=None
            if stopped.is_set():
                break
            if eval_code or quality_request(job) is not None:
                raise RuntimeError('Epoch quality failed; durable training snapshot retained for manual resume')
            # Resuming is authorized only for this scheduler-owned epoch stop.
            args.attach_pid=None
            init_args=initialization_arguments(training,args.init_from,args.statistics_from,args.statistics_validation)
            index=next((i for i,x in enumerate(command) if x in ('--resume','--init-from','--statistics-from')),len(command))
            command=command[:index]+init_args
            if ready.exists():ready.rename(job/f'resources-ready-{time.time_ns()}.json')
            shutdown_started=None
            if due_for_pause(policy,time.time()):
                deadline_stop=True
                break
            status('resuming_after_epoch_quality',physical_ids=physical,command=command)
        if deadline_stop:
            progress = json.loads((training / "status.json").read_text()) if (training / "status.json").exists() else {}
            atomic_json(job / "five-hour-review.json", {"time_unix": time.time(),
                "decision": "pause for user review; release all GPUs", "progress": progress,
                "resource_policy": policy, "automatic_restart": False})
            status("paused_for_review", physical_ids=[])
        elif stopped.is_set():
            status("interrupted")
        elif code:
            raise RuntimeError(f"Fast training exited with code {code}; see training-fast.log")
        else:
            status(json.loads((training / "status.json").read_text())["status"])
    except Exception as error:
        status("failed", error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for handle in gpu_handles.values():
            handle.close()


if __name__ == "__main__":
    main()
