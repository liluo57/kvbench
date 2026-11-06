"""Real-target validation of fused GDN, SDPA and independent-chunk batching."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from .model import QwenCacheRepair, RepairConfig
from .resume import atomic_json
from .supervision import make_pair
from .target import TargetGeometry, text_backbone
from .train import load_native_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--chunk-batch-size", type=int, default=8)
    parser.add_argument("--pack-chunks", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    gpus = tuple(map(int, args.gpus.split(",")))
    torch.cuda.set_device(gpus[0])
    target = load_native_target(args.model, gpus, attention_backend="sdpa", require_fast=True)
    geometry = TargetGeometry.from_target(target, target_model_id="Qwen/Qwen3.8-27B")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    modules = [layer.linear_attn for layer in text_backbone(target).layers if hasattr(layer, "linear_attn")]
    fast = [(m.causal_conv1d_fn, m.chunk_gated_delta_rule, m.norm) for m in modules]
    last_logits = {}
    def capture_logits(module, inputs, output):
        last_logits["value"] = output.logits[:, -1].detach().float().cpu()
    target.register_forward_hook(capture_logits)

    def backend(fused):
        text_backbone(target).config._attn_implementation = "sdpa" if fused else "eager"
        for m, (conv, delta, norm) in zip(modules, fast):
            m.causal_conv1d_fn = conv if fused else None
            m.chunk_gated_delta_rule = delta if fused else native.torch_chunk_gated_delta_rule
            if fused:
                m.norm = norm
            else:
                replacement = native.Qwen3_5RMSNormGated(norm.weight.numel(), eps=m.config.rms_norm_eps
                    if hasattr(m, "config") else 1e-6).to(norm.weight)
                with torch.no_grad():
                    replacement.weight.copy_(norm.weight)
                m.norm = replacement.requires_grad_(False)
        target.requires_grad_(False).eval()

    def synchronize():
        for gpu in gpus:
            torch.cuda.synchronize(gpu)

    def pair(chunks, batch):
        synchronize()
        start = time.perf_counter()
        result = make_pair(target, chunks, geometry=geometry, chunk_batch_size=batch,
                           pack_chunks=args.pack_chunks and batch > 1)
        synchronize()
        seconds = time.perf_counter() - start
        return {**{key: value.cpu() for key, value in result.items()}, "last_logits": last_logits["value"]}, seconds

    def compare(reference, actual):
        report = {}
        for key in ("tokens", "chunk_ids"):
            if not torch.equal(reference[key], actual[key]):
                raise AssertionError("Chunk batching changed token order or boundaries")
        for key in ("stale", "joint"):
            error = (actual[key] - reference[key]).double()
            rel = (error.square().sum() / reference[key].double().square().sum()).sqrt().item()
            report[key] = {"relative_l2": rel, "max_absolute_error": error.abs().max().item()}
            if not torch.isfinite(actual[key]).all():
                raise AssertionError(f"Excessive {key} numerical drift: {report[key]}")
        ref_logits, actual_logits = reference["last_logits"], actual["last_logits"]
        report["logits"] = {"cosine_similarity": torch.nn.functional.cosine_similarity(ref_logits, actual_logits).item(),
                             "top1_equal": ref_logits.argmax(-1).item() == actual_logits.argmax(-1).item(),
                             "top10_overlap": len(set(ref_logits.topk(10).indices.flatten().tolist()) &
                                                  set(actual_logits.topk(10).indices.flatten().tolist())) / 10}
        return report

    rows = [json.loads(line) for line in (Path(args.dataset) / "train.jsonl").read_text().splitlines()]
    selected = [0, max(range(len(rows)), key=lambda i: sum(map(len, rows[i]["chunks"])))]
    # Warm the compiled path separately from steady-state timings.
    pair(rows[0]["chunks"], args.chunk_batch_size)
    report = {"status": "running", "observations": [], "chunk_batch_size": args.chunk_batch_size,
              "numerical_policy": "Official fused BF16 teacher is the new reference; no old RMS is reused. Packed-vs-single KV L2 <=5%, next-token logit cosine >=.999 and top1 unchanged; old/new logit cosine >=.99 and top10 overlap >=.5. FP32 unit reference separately checks exact chunk boundaries. All measured drift is reported.",
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "fla-core", "causal-conv1d")}}
    for index in selected:
        chunks = rows[index]["chunks"]
        backend(False)
        reference, old_seconds = pair(chunks, 1)
        backend(True)
        unbatched, fused_seconds = pair(chunks, 1)
        batched, batched_seconds = pair(chunks, args.chunk_batch_size)
        observation = {"training_row": index, "tokens": sum(map(len, chunks)), "chunks": len(chunks),
                       "torch_unbatched_seconds": old_seconds, "fused_unbatched_seconds": fused_seconds,
                       "fused_batched_seconds": batched_seconds,
                       "speedup": old_seconds / batched_seconds,
                       "kernel_drift": compare(reference, unbatched), "batch_drift": compare(unbatched, batched)}
        report["observations"].append(observation)
        print(json.dumps(observation), flush=True)
        atomic_json(args.output, report)
        if (any(observation["batch_drift"][key]["relative_l2"] > .05 for key in ("stale", "joint"))
            or observation["batch_drift"]["logits"]["cosine_similarity"] < .999
            or not observation["batch_drift"]["logits"]["top1_equal"]
            or observation["kernel_drift"]["logits"]["cosine_similarity"] < .99
            or observation["kernel_drift"]["logits"]["top10_overlap"] < .5):
            report["status"] = "failed_numerical_drift"
            atomic_json(args.output, report)
            raise AssertionError("Fast path failed batch equivalence or next-token logit stability")
        del reference, unbatched, batched
    backend(True)
    # Exercise the largest input's real forward/backward while the target lives
    # on the same GPUs, including the unchanged official residual loss.
    from ..backend.cacherepair.model import CacheRepair as OfficialRepairer
    model = QwenCacheRepair(geometry, RepairConfig(), target.get_input_embeddings()).to(f"cuda:{gpus[0]}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.01)
    inputs = make_pair(target, rows[selected[-1]]["chunks"], geometry=geometry, chunk_batch_size=args.chunk_batch_size,
                       pack_chunks=args.pack_chunks)
    loss = OfficialRepairer.loss(model, **inputs)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Fast-path training has nonfinite loss or gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    if any(p.grad is not None or p.requires_grad for p in target.parameters()):
        raise AssertionError("Fast-path target is not frozen")
    report.update(status="passed", largest_input_backward_finite=True,
                  gpu_memory={str(g): {"peak_allocated_bytes": torch.cuda.max_memory_allocated(g),
                                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(g)} for g in gpus})
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
