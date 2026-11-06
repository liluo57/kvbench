#!/usr/bin/env python3
"""Microbenchmark the NCCL PyNccl all-reduce path used by vLLM.

Run as a fresh two-process job for each NCCL_PROTO setting. For example:

    CUDA_VISIBLE_DEVICES=0,1 NCCL_PROTO=Simple \
      torchrun --standalone --nproc-per-node=2 \
      scripts/ProfileCacheRepairCollectives.py

This intentionally measures collectives only; it does not load a model or
write tensors/results to disk. NCCL environment settings must be set before
launch so each fresh worker initializes a communicator under that setting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from typing import Any


DEFAULT_ROWS = (1024, 4096, 16384, 32768)
HIDDEN_SIZE = 5120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=DEFAULT_ROWS)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261029)
    return parser.parse_args()


def tensor_sha256(tensor: Any, torch: Any) -> str:
    """Hash the raw contiguous tensor bytes without writing them to disk."""
    raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be nonnegative and --iterations positive")
    if any(rows <= 0 for rows in args.rows) or args.hidden_size <= 0:
        raise SystemExit("rows and hidden size must be positive")

    # Import CUDA packages only inside the launched worker process. NCCL_PROTO
    # is reported from the environment inherited at process start.
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("launch exactly two processes (torchrun --nproc-per-node=2)")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and len([part for part in visible.split(",") if part.strip()]) != 2:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly two GPUs")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = int(os.environ["RANK"])

    # Gloo carries only communicator bootstrap and small timing metadata. The
    # measured all-reduces use vLLM's direct PyNcclCommunicator API.
    dist.init_process_group(backend="gloo")
    comm = PyNcclCommunicator(group=dist.group.WORLD, device=device)
    if comm.disabled:
        raise RuntimeError("vLLM PyNcclCommunicator was disabled")

    results: list[dict[str, Any]] = []
    try:
        for rows in args.rows:
            shape = (rows, args.hidden_size)
            # Input ranks contain 1 and 2, so SUM must produce exactly 3 in
            # bfloat16. Both buffers persist only in this process/GPU memory.
            inp = torch.full(shape, rank + 1, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)

            dist.barrier()
            for _ in range(args.warmup):
                comm.all_reduce(inp, out)
            torch.cuda.synchronize(device)
            dist.barrier()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                result = comm.all_reduce(inp, out)
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)

            if result is None or result.data_ptr() != out.data_ptr():
                raise AssertionError("PyNcclCommunicator did not return the supplied output buffer")

            # Check every element from the final operation. The exact values
            # are representable in BF16, so zero tolerance is appropriate.
            if not torch.all(out == 3).item():
                max_error = (out.float() - 3.0).abs().max().item()
                raise AssertionError(f"all-reduce output mismatch (max error {max_error})")

            # Independent deterministic validation: both ranks recreate the
            # same pair of signed BF16 inputs. Expected is formed on CPU with
            # FP32 addition and one final BF16 rounding. This does not touch
            # the timed constant inputs or their output buffer.
            generator = torch.Generator(device="cpu").manual_seed(args.seed + rows)
            x0 = torch.randn(shape, generator=generator, dtype=torch.float32).to(
                torch.bfloat16
            )
            x1 = torch.randn(shape, generator=generator, dtype=torch.float32).to(
                torch.bfloat16
            )
            expected = (x0.float() + x1.float()).to(torch.bfloat16)
            signed_inputs = bool((x0 < 0).any() and (x0 > 0).any() and (x1 < 0).any() and (x1 > 0).any())
            if not signed_inputs:
                raise AssertionError("seeded validation inputs must contain positive and negative values")
            validation_input = (x0 if rank == 0 else x1).to(device=device)
            validation_output = torch.empty_like(validation_input)
            validation_result = comm.all_reduce(validation_input, validation_output)
            if (
                validation_result is None
                or validation_result.data_ptr() != validation_output.data_ptr()
            ):
                raise AssertionError("PyNcclCommunicator validation call returned an unexpected output")
            torch.cuda.synchronize(device)
            actual_cpu = validation_output.cpu()
            if not torch.equal(actual_cpu, expected):
                max_error = (actual_cpu.float() - expected.float()).abs().max().item()
                raise AssertionError(
                    f"random BF16 all-reduce mismatch (max abs error {max_error})"
                )
            validation_hashes = {
                "expected_sha256": tensor_sha256(expected, torch),
                "output_sha256": tensor_sha256(actual_cpu, torch),
            }
            all_validation_hashes: list[dict[str, str] | None] = [
                None
            ] * dist.get_world_size()
            dist.all_gather_object(all_validation_hashes, validation_hashes)
            if any(item != all_validation_hashes[0] for item in all_validation_hashes):
                raise AssertionError(
                    f"ranks produced different validation hashes: {all_validation_hashes}"
                )

            rank_samples: list[float] = [0.0] * dist.get_world_size()
            dist.all_gather_object(rank_samples, elapsed_ms / args.iterations)
            if rank == 0:
                latency_ms = max(rank_samples)
                payload_bytes = rows * args.hidden_size * 2  # BF16 is two bytes.
                # Standard all-reduce effective bus-bandwidth estimate:
                # 2 * (P - 1) / P * payload_bytes / elapsed_time.
                bus_bytes = 2 * (dist.get_world_size() - 1) / dist.get_world_size() * payload_bytes
                seconds = latency_ms / 1000.0
                results.append(
                    {
                        "rows": rows,
                        "hidden_size": args.hidden_size,
                        "dtype": "bfloat16",
                        "payload_bytes_per_rank": payload_bytes,
                        "iterations": args.iterations,
                        "latency_ms_max_rank": latency_ms,
                        "latency_ms_rank_median": statistics.median(rank_samples),
                        "payload_gib_s": (payload_bytes / seconds) / (1024**3),
                        "effective_bus_gib_s": (bus_bytes / seconds) / (1024**3),
                        "correct_sum": True,
                        "random_validation_seed": args.seed + rows,
                        "random_validation_bitwise_equal": True,
                        "expected_sha256": validation_hashes["expected_sha256"],
                        "output_sha256": validation_hashes["output_sha256"],
                    }
                )
            del inp, out, x0, x1, expected, validation_input, validation_output, actual_cpu
            torch.cuda.empty_cache()

        if rank == 0:
            print(
                json.dumps(
                    {
                        "benchmark": "vllm_pynccl_all_reduce",
                        "world_size": dist.get_world_size(),
                        "device": torch.cuda.get_device_name(device),
                        "torch_version": torch.__version__,
                        "nccl_version_raw": comm.nccl_version,
                        "nccl_proto_environment": os.environ.get("NCCL_PROTO", "<unset/default>"),
                        "cuda_visible_devices": visible,
                        "warmup": args.warmup,
                        "results": results,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        comm.destroy()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # torchrun collects worker stderr; keep a concise failure context.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
