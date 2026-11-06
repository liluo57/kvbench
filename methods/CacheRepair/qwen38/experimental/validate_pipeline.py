"""Real 27B serial/pipeline equivalence, RMS throughput, and memory preflight."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import torch

from ..supervision import make_pair
from ..target import TargetGeometry
from ..train import load_native_target
from .pipeline import NativePipeline


def accumulate(pair, sums):
    for name, value in [("sigma_stale", pair["stale"]), ("sigma_delta", pair["joint"] - pair["stale"])]:
        term = value.double().square().sum(dim=(0, 1)).cpu()
        sums[name] = sums.get(name, 0) + term


def measured(rows, get, gpu_ids, label):
    stop, telemetry = threading.Event(), []

    def monitor():
        while not stop.is_set():
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                               "--format=csv,noheader,nounits"], text=True)
            telemetry.append({"time_unix": time.time(), "gpus": [list(map(int, line.split(",")))
                                                                   for line in output.splitlines()]})
            stop.wait(.5)

    for gpu in gpu_ids:
        torch.cuda.synchronize(gpu)
        torch.cuda.reset_peak_memory_stats(gpu)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    sums, tokens = {}, 0
    try:
        for i, row in enumerate(rows):
            pair = get(row["chunks"])
            accumulate(pair, sums)
            tokens += pair["tokens"].numel()
            del pair
            if (i + 1) % 8 == 0:
                print(json.dumps({"phase": label, "rows": i + 1, "seconds": time.perf_counter() - started}), flush=True)
        for gpu in gpu_ids:
            torch.cuda.synchronize(gpu)
    finally:
        stop.set()
        thread.join()
    elapsed = time.perf_counter() - started
    visible = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")))
    return sums, {"phase": label, "rows": len(rows), "seconds": elapsed,
        "rows_per_second": len(rows) / elapsed, "tokens_per_second": tokens / elapsed,
        "gpu_mean_utilization": {str(g): sum(sample["gpus"][visible[g]][1] for sample in telemetry) / len(telemetry)
                                 for g in gpu_ids}, "telemetry": telemetry,
        "physical_gpu_ids": [visible[g] for g in gpu_ids],
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "gpu_peak_memory_mib": {str(g): max(sample["gpus"][visible[g]][2] for sample in telemetry)
                                for g in gpu_ids},
        "peak_allocated_bytes": {str(g): torch.cuda.max_memory_allocated(g) for g in gpu_ids},
        "peak_reserved_bytes": {str(g): torch.cuda.max_memory_reserved(g) for g in gpu_ids}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--depths", default="2,3")
    parser.add_argument("--skip-serial-benchmark", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = {}
    indices = set(range(8)) | {17911}
    with (Path(args.dataset) / "train.jsonl").open() as stream:
        for i, line in enumerate(stream):
            if i in indices:
                selected[i] = json.loads(line)
            if i == max(indices):
                break
    rows = [selected[i] for i in sorted(selected)]
    target = load_native_target(args.model, (0, 1), attention_backend="sdpa", require_fast=True)
    geometry = TargetGeometry.from_target(target, target_model_id="Qwen/Qwen3.8-27B")
    pipeline_sha256 = hashlib.sha256(Path(__file__).with_name("pipeline.py").read_bytes()).hexdigest()
    baseline = lambda chunks: make_pair(target, chunks, geometry=geometry, chunk_batch_size=8, pack_chunks=True)
    references = []
    for row in rows:
        pair = baseline(row["chunks"])
        references.append({key: value.cpu() for key, value in pair.items()})
        del pair
    benchmark = [rows[i % len(rows)] for i in range(args.rows)]
    if args.skip_serial_benchmark:
        terms = []
        for pair in references:
            sums = {}
            accumulate(pair, sums)
            terms.append(sums)
        expected = {}
        for i in range(len(benchmark)):
            for key, value in terms[i % len(rows)].items():
                expected[key] = expected.get(key, 0) + value
        serial = None
    else:
        expected, serial = measured(benchmark, baseline, (0, 1), "serial")
    report = {"status": "in_progress", "serial": serial, "pipelines": [],
              "target_geometry": geometry.to_dict(), "model_path": args.model,
              "pipeline_source_sha256": pipeline_sha256}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for depth in map(int, args.depths.split(",")):
        gc.collect()
        for gpu in (0, 1):
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()
        pipeline = NativePipeline(target, geometry, (0, 1), depth=depth, record=True)
        try:
            pipeline.set_sequence(rows)
            differences = []
            for row, reference in zip(rows, references):
                actual = pipeline.get(row["chunks"])
                for key in reference:
                    value = actual[key].cpu()
                    equal = torch.equal(value, reference[key])
                    differences.append({"key": key, "equal": equal})
                    if not equal:
                        raise AssertionError(f"Concurrent pipeline changed {key}; refusing RMS reuse")
                del actual
            pipeline.set_sequence(benchmark)
            actual, result = measured(benchmark, pipeline.get, (0, 1), f"pipeline-depth-{depth}")
            result["kv_bitwise_equal"] = all(v["equal"] for v in differences)
            # CPU reference reduction can have a different FP64 reduction
            # tree; the full serial-GPU comparison is the strict RMS gate.
            result["rms_sums_bitwise_equal"] = all(torch.equal(actual[key], expected[key]) for key in expected)
            if args.skip_serial_benchmark:
                result["rms_fp64_relative_difference"] = max(
                    ((actual[key] - expected[key]).abs() / expected[key].abs().clamp_min(1e-20)).max().item()
                    for key in expected)
                if result["rms_fp64_relative_difference"] > 1e-12:
                    raise AssertionError("RMS differs from CPU FP64 reference")
            if not result["rms_sums_bitwise_equal"] and not args.skip_serial_benchmark:
                raise AssertionError("RMS accumulation changed")
            result["speedup"] = serial["seconds"] / result["seconds"] if serial is not None else None
            result["events"] = pipeline.events
            report["pipelines"].append(result)
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value for key, value in result.items() if key not in ("events", "telemetry")}), flush=True)
        finally:
            pipeline.close()
    report["status"] = "passed"
    if hashlib.sha256(Path(__file__).with_name("pipeline.py").read_bytes()).hexdigest() != pipeline_sha256:
        raise AssertionError("Pipeline implementation changed during validation")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
