"""Native GPU memory/throughput preflight using the frozen full-data manifest."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from ..backend.cacherepair.model import CacheRepair as OfficialRepairer
from .model import QwenCacheRepair, RepairConfig
from .resume import atomic_json
from .supervision import make_pair
from .target import TargetGeometry
from .train import load_native_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings = json.loads(Path(args.config).read_text())
    rows = [json.loads(line) for line in (Path(args.dataset) / "train.jsonl").read_text().splitlines()]
    length = lambda row: sum(map(len, row["chunks"]))
    selected = {}
    for band, lo, hi in (("short", 1, 1024), ("medium", 1025, 2048), ("long", 2049, 4096)):
        candidates = [i for i, row in enumerate(rows) if lo <= length(row) <= hi]
        if not candidates:
            raise ValueError(f"Missing preflight length band: {band}")
        # Test longest sequence and most chunk-prefill calls in every band.
        selected[band] = sorted(set([max(candidates, key=lambda i: length(rows[i])),
                                     max(candidates, key=lambda i: len(rows[i]["chunks"]))]))
    gpus = tuple(map(int, args.gpus.split(",")))
    torch.cuda.set_device(gpus[0])
    started = time.perf_counter()
    target = load_native_target(args.model, gpus)
    geometry = TargetGeometry.from_target(target, target_model_id=settings["target_model_id"])
    if list(geometry.kv_shape) != settings["expected_kv_shape"]:
        raise ValueError("Unexpected target geometry")
    config = RepairConfig(**settings["repair_config"])
    torch.manual_seed(settings["training"]["initialization_seed"])
    repairer = QwenCacheRepair(geometry, config, target.get_input_embeddings()).to(f"cuda:{gpus[0]}")
    repairer.loss = OfficialRepairer.loss.__get__(repairer)
    optimizer = torch.optim.AdamW(repairer.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    for gpu in gpus:
        torch.cuda.reset_peak_memory_stats(gpu)

    def sync():
        for gpu in gpus:
            torch.cuda.synchronize(gpu)

    observations = []
    for band, indices in selected.items():
        for index in indices:
            row = rows[index]
            sync()
            clock = time.perf_counter()
            pair = make_pair(target, row["chunks"], geometry=geometry)
            sync()
            pair_seconds = time.perf_counter() - clock
            clock = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = repairer.loss(**pair)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native preflight loss")
            loss.backward()
            gradients = [p.grad for p in repairer.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise FloatingPointError("Nonfinite or absent native preflight gradients")
            torch.nn.utils.clip_grad_norm_(repairer.parameters(), 1.)
            optimizer.step()
            if not all(torch.isfinite(p).all().item() for p in repairer.parameters()):
                raise FloatingPointError("Nonfinite native preflight parameters")
            sync()
            observations.append({"band": band, "training_row": index, "tokens": length(row),
                                 "chunks": len(row["chunks"]), "pair_seconds": pair_seconds,
                                 "repair_forward_backward_seconds": time.perf_counter() - clock,
                                 "loss": loss.item()})
            print(json.dumps(observations[-1]), flush=True)
            del pair, loss
    memory = {str(g): {"peak_allocated_bytes": torch.cuda.max_memory_allocated(g),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(g),
                       "total_bytes": torch.cuda.get_device_properties(g).total_memory} for g in gpus}
    if any(p.requires_grad or p.grad is not None for p in target.parameters()):
        raise AssertionError("Target gradients in preflight")
    # A conservative estimate from the tested extremes, not a measured full run.
    per_band = {band: {key: max(o[key] for o in observations if o["band"] == band)
                      for key in ("pair_seconds", "repair_forward_backward_seconds")} for band in selected}
    pair_mean = sum(per_band[b]["pair_seconds"] * w for b, w in (("short", .1), ("medium", .1), ("long", .8)))
    repair_mean = sum(per_band[b]["repair_forward_backward_seconds"] * w for b, w in (("short", .1), ("medium", .1), ("long", .8)))
    estimate = pair_mean * len(rows) * 7 + repair_mean * len(rows) * 6
    atomic_json(args.output, {"status": "passed", "observations": observations, "gpu_memory": memory,
                             "target_frozen": True, "repair_config": asdict(config),
                             "estimated_stats_plus_six_epochs_wall_hours": estimate / 3600,
                             "estimate_scope": "Conservative length-band extremes; excludes validation/checkpoints/loading/I/O; no pair disk cache",
                             "total_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
