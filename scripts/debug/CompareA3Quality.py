#!/usr/bin/env python3
"""Run paired A3 quality/TTFT checks with frozen task inputs.

The optional reference worker is an archived, pre-vendoring worker. Its Python
environment and checkout are used only by this comparison, never by A3Repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def summarize(rows):
    scores = sorted({key for row in rows for key in row["scores"]})
    latency = [row["performance"]["ttft"] for row in rows]
    return {
        "samples": len(rows),
        **{key: statistics.mean(row["scores"][key] for row in rows) for key in scores},
        "mean_ttft": statistics.mean(latency),
        "median_ttft": statistics.median(latency),
        "fallbacks": sum(bool(row["result_metadata"].get("full_prefill")) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclusive-gpu", action="store_true",
                        help="reject latency samples when another process uses this GPU")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--recomp-ratio", type=float, default=0.15)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--ruler-length", type=int, default=4096)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-python")
    parser.add_argument("--reference-worker")
    parser.add_argument("--reference-repo")
    parser.add_argument("--arms", nargs="+", choices=["bundled", "reference"])
    args = parser.parse_args()
    reference_paths = (args.reference_python, args.reference_worker, args.reference_repo)
    if any(reference_paths) and not all(reference_paths):
        parser.error("provide all three reference paths together")
    if args.arms is None:
        args.arms = ["bundled", "reference"] if all(reference_paths) else ["bundled"]
    if "reference" in args.arms and not all(reference_paths):
        parser.error("reference arm requires Python, worker and repo paths")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def check_gpu(worker_pid=None):
        if not args.exclusive_gpu:
            return
        result = subprocess.run(
            ["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        owners = {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
        others = owners - ({worker_pid} if worker_pid is not None else set())
        if others:
            raise RuntimeError(f"GPU {args.gpu} is shared with PIDs {sorted(others)}; sample rejected")

    from core import Config

    cfg = Config.LoadConfig()
    cfg.update(ModelPath=args.model, ModelConfig={"mode": "greedy"}, MaxModelLen=args.max_model_len)
    cfg.setdefault("A3", {})["Python"] = None

    from methods.A3Repo import A3Repo
    from tasks.Niah import NIAHShuffleTask
    from tasks.Cwe import CWEShuffleTask
    from tasks.Vt import VTShuffleTask
    from tasks.Musique import MusiqueTask
    from tasks.Samsum import SamsumTask
    from tasks.TwoWikiMultiHopQA import TwoWikiMultiHopQATask
    from tasks.GovReport import GovReportTask
    from tasks.HotpotQA import HotpotQATask
    from tasks.MultiNews import MultiNewsTask
    from tasks.TriviaQA import TriviaQATask

    task_classes = {
        "niah_shuffle": NIAHShuffleTask, "cwe_shuffle": CWEShuffleTask,
        "vt_shuffle": VTShuffleTask, "musique": MusiqueTask,
        "samsum": SamsumTask, "2wikimultihopqa": TwoWikiMultiHopQATask,
        "govreport": GovReportTask, "hotpotqa": HotpotQATask,
        "multinews": MultiNewsTask, "triviaqa": TriviaQATask,
    }

    class ReferenceA3(A3Repo):
        def _startWorker(self):
            if not all((args.reference_python, args.reference_worker, args.reference_repo)):
                raise ValueError("reference arm requires Python, worker and repo paths")
            self._stderrTail.clear()
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env.pop("LD_PRELOAD", None)
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            env["PATH"] = str(Path(args.reference_python).absolute().parent) + os.pathsep + env.get("PATH", "")
            self._proc = subprocess.Popen(
                [args.reference_python, args.reference_worker, "--repo_root", args.reference_repo,
                 "--model", self.modelPath, "--max_new_tokens", str(self.maxNewTokens),
                 "--recomp_ratio", str(self.recompRatio), "--reuse_method", self.reuseMethod,
                 "--sampling_config", json.dumps(self.samplingConfig)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=env, cwd=str(ROOT),
            )
            self._drainThread = threading.Thread(target=self._drainStderr, daemon=True)
            self._drainThread.start()
            for line in self._proc.stdout:
                if line.startswith("[a3-repo-helper] ready"):
                    return
                print(line.rstrip(), flush=True)
            raise RuntimeError("reference worker failed: " + "".join(self._stderrTail))

    for task_name in args.tasks:
        if task_name not in task_classes:
            parser.error(f"unknown task: {task_name}")
        is_ruler = task_name.endswith("_shuffle")
        dataset_dir = Path(args.data_root) / (f"ruler/{args.ruler_length}" if is_ruler else task_name)
        kwargs = {"dataDir": str(dataset_dir), "maxSamples": args.samples,
                  "startIdx": args.start_index}
        if is_ruler:
            kwargs["maxSeqLength"] = args.ruler_length
        if task_name == "govreport":
            kwargs.update(nChunks=4, maxSampleLength=32768)
        if args.max_new_tokens is not None:
            kwargs["maxNewTokens"] = args.max_new_tokens
        task = task_classes[task_name](**kwargs)
        cases = list(task.Cases())
        for case in cases:
            case.workflow.case_id += args.start_index
            if "case_id" in case.metadata:
                case.metadata["case_id"] += args.start_index
        if len(cases) != args.samples:
            raise RuntimeError(f"{task_name}: expected {args.samples} cases, got {len(cases)}")
        if max(len(case.input.run_input) for case in cases) + task.maxNewTokens > args.max_model_len:
            raise ValueError(f"{task_name}: prompt plus generation exceeds model budget")
        inputs_path = output_dir / f"{task_name}.inputs.jsonl"
        if args.resume and inputs_path.exists():
            previous = {row["case_id"]: row for row in
                        map(json.loads, inputs_path.read_text().splitlines())}
            for case in cases:
                saved = previous.get(case.workflow.case_id)
                if saved is None or saved["input_ids"] != case.input.run_input or saved["chunk_ids"] != case.input.prepare_input:
                    raise ValueError("resume inputs differ from the saved manifest")
        with inputs_path.open("w") as stream:
            for case in cases:
                stream.write(json.dumps({"case_id": case.workflow.case_id,
                    "metadata": case.metadata, "input_ids": case.input.run_input,
                    "chunk_ids": case.input.prepare_input}, ensure_ascii=False) + "\n")
        report = {"task": task_name, "model": args.model, "task_parameters": kwargs,
                  "recomp_ratio": args.recomp_ratio, "reuse_method": "debug",
                  "sampling": "greedy", "dtype": "bfloat16", "batch_size": 1,
                  "gpu": args.gpu, "max_new_tokens": task.maxNewTokens,
                  "max_model_len": args.max_model_len,
                  "exclusive_gpu": args.exclusive_gpu,
                  "input_manifest_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
                  "prompt_tokens": [len(case.input.run_input) for case in cases],
                  "summaries": {}, "started_unix": time.time()}
        report_path = output_dir / f"{task_name}.json"
        write_json(report_path, report)

        for arm in args.arms:
            cls = A3Repo if arm == "bundled" else ReferenceA3
            method = cls(maxNewTokens=task.maxNewTokens, maxModelLen=args.max_model_len,
                         recompRatio=args.recomp_ratio)
            rows = {"full": [], "reuse": []} if arm == "bundled" else {"reuse": []}
            log_path = output_dir / f"{task_name}.{arm}.samples.jsonl"
            completed = set()
            if args.resume and log_path.exists():
                for row in map(json.loads, log_path.read_text().splitlines()):
                    key = (row["case_id"], row["mode"])
                    if key in completed:
                        raise ValueError("duplicate saved sample")
                    if args.start_index <= row["case_id"] < args.start_index + args.samples:
                        rows[row["mode"]].append(row)
                        completed.add(key)
                if len(completed) == len(cases) * len(rows):
                    for mode, samples in rows.items():
                        report["summaries"][f"{arm}_{mode}"] = summarize(samples)
                    write_json(report_path, report)
                    continue
            print(f"[{task_name}/{arm}] initialize GPU {args.gpu}; tokens={task.maxNewTokens}", flush=True)
            try:
                check_gpu()
                method.Initialize([args.gpu])
                check_gpu(method._proc.pid)
                # Warm the full/reuse prefill and autoregressive decode kernels.
                if arm == "bundled":
                    method.Run([cases[0].input.run_input], maxNewTokens=min(8, task.maxNewTokens))
                    method.Reset()
                method.Prepare([cases[0].input.prepare_input])
                method.Run([cases[0].input.run_input], maxNewTokens=min(8, task.maxNewTokens))
                method.Reset()
                with log_path.open("a" if args.resume else "w", buffering=1) as stream:
                    for index, case in enumerate(cases):
                        for mode in rows:
                            if (case.workflow.case_id, mode) in completed:
                                continue
                            check_gpu(method._proc.pid)
                            start = time.perf_counter()
                            if mode == "reuse":
                                method.Prepare([case.input.prepare_input])
                            prepared = time.perf_counter()
                            result = method.Run([case.input.run_input], retainOutput=[True],
                                                maxNewTokens=task.maxNewTokens)[0]
                            finished = time.perf_counter()
                            check_gpu(method._proc.pid)
                            if result.metadata.get("n_input") != len(case.input.run_input):
                                raise RuntimeError("worker consumed different input tokens")
                            if mode == "reuse" and result.metadata.get("full_prefill"):
                                raise RuntimeError(f"{task_name}/{index}: unexpected full-prefill fallback")
                            row = {"case_id": case.workflow.case_id, "arm": arm, "mode": mode,
                                   "prompt_tokens": len(case.input.run_input), "output": result.output,
                                   "scores": task.Evaluate(result, case.metadata),
                                   "performance": result.performance, "result_metadata": result.metadata,
                                   "prepare_seconds": prepared - start,
                                   "wall_seconds": finished - start,
                                   "exclusive_gpu_worker_pid": method._proc.pid if args.exclusive_gpu else None}
                            rows[mode].append(row)
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            method.Reset()
                        if (index + 1) % 8 == 0 or index + 1 == len(cases):
                            print(f"[{task_name}/{arm}] {index + 1}/{len(cases)}", flush=True)
            finally:
                method.Close()
            for mode, samples in rows.items():
                report["summaries"][f"{arm}_{mode}"] = summarize(samples)
            report["finished_unix"] = time.time()
            write_json(report_path, report)
            print(f"[{task_name}/{arm}] {report['summaries']}", flush=True)


if __name__ == "__main__":
    main()
