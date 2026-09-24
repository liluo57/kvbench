"""Bisect one HotpotQA PIC request against a direct Full baseline.

This is deliberately independent of KVBench.  It uses HYPIC's optional
per-layer diagnostic dump to identify where a fully cached request first
diverges from the same prompt recomputed from scratch.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


SEP = "<<PIC_SEP>>"
SYSTEM = "You are a helpful assistant.\n\n"
POST = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
PREFIX = (
    "Answer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\nThe following are given "
    "passages.\n"
)
QUERY_PREFIX = (
    "\n\nAnswer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\nQuestion: "
)


def build_case(sample: dict) -> tuple[list[str], str]:
    context = str(sample["context"])
    parts = [
        part
        for part in re.split(
            r"(?m)(?=^Passage(?: \d+)?:[ \t]*$)", context
        )
        if part
    ]
    parts[0] = PREFIX + parts[0]
    question = str(sample["input"]).strip()
    return parts, f"{QUERY_PREFIX}{question}\nAnswer:"


def run_once(
    label: str,
    prompt: str,
    warmups: list[str],
    output_path: Path,
    model: str,
    mode: str | None,
) -> None:
    input_path = output_path.with_suffix(".input.json")
    gdn_path = output_path.with_suffix(".gdn.jsonl")
    input_path.write_text(json.dumps({"prompt": prompt, "warmups": warmups}))
    child = f"""
import json, os
os.environ["PIC_DIAG_DUMP"] = {str(output_path)!r}
os.environ["PIC_DIAG_GDN"] = {str(gdn_path)!r}
os.environ["PIC_DIAG_GDN_LAYER"] = "0"
os.environ["PIC_DIAG_FORCE_SPLIT"] = "1"
inp = json.load(open({str(input_path)!r}))
import sglang as sgl
common = dict(
    model_path={model!r}, tp_size=1, dtype="bfloat16",
    context_length=32768, max_prefill_tokens=32768,
    max_running_requests=1, mem_fraction_static=0.80,
    trust_remote_code=True, enable_multimodal=False, page_size=1,
    chunked_prefill_size=-1, disable_cuda_graph=True,
    log_level="error",
)
"""
    if mode is None:
        child += (
            "engine = sgl.Engine(**common, pic_enable=False, "
            "mamba_radix_cache_strategy='no_buffer', "
            "disable_radix_cache=True, disable_overlap_schedule=True)\n"
        )
    else:
        child += (
            "engine = sgl.Engine(**common, pic_enable=True, "
            f"pic_mode={mode!r}, pic_separator_str={SEP!r}, "
            "max_mamba_cache_size=32)\n"
        )
    child += f"""
if {mode is not None!r} and {os.environ.get("DIRECT_NO_WARMUP", "0")!r} != "1":
    for warmup in inp["warmups"]:
        engine.generate(warmup, sampling_params={{"temperature": 0, "max_new_tokens": 1}})
open({str(output_path)!r}, "w").close()
open({str(gdn_path)!r}, "w").close()
out = engine.generate(inp["prompt"], sampling_params={{"temperature": 0, "max_new_tokens": 3}})
print({label!r}, out.get("meta_info", {{}}), repr(out.get("text", "")), flush=True)
engine.shutdown()
"""
    subprocess.run([sys.executable, "-c", child], check=True, env=os.environ.copy())


def compare(base: Path, pic: Path) -> None:
    base_rows = [json.loads(line) for line in base.read_text().splitlines() if line]
    pic_rows = [json.loads(line) for line in pic.read_text().splitlines() if line]
    print(f"layers: full={len(base_rows)} ttr={len(pic_rows)}")
    first = None
    for index, (full, ttr) in enumerate(zip(base_rows, pic_rows)):
        rel = abs(full["norm"] - ttr["norm"]) / max(full["norm"], 1e-6)
        dot = sum(a * b for a, b in zip(full["head"], ttr["head"]))
        nf = sum(a * a for a in full["head"]) ** 0.5
        nt = sum(a * a for a in ttr["head"]) ** 0.5
        cosine = dot / (nf * nt + 1e-12)
        diverged = rel > 1e-4 or cosine < 0.999
        if diverged and first is None:
            first = (index, full["layer"], rel, cosine)
        print(
            f"layer={full['layer']:2d} kind={full['kind']:<6} "
            f"rel_norm={rel:.3e} cosine8={cosine:.6f}"
            + ("  DIVERGE" if diverged else "")
        )
    print(f"first divergence: {first}")


def compare_gdn(base: Path, pic: Path) -> None:
    def read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    base_rows, pic_rows = read(base), read(pic)
    print("layer 0 GDN diagnostics:")
    for tag in ("base_ssm_final", "pic_ssm_final"):
        rows = [row for row in (base_rows + pic_rows) if row.get("tag") == tag]
        if rows:
            row = rows[-1]
            print(f"  {tag}: norm={row['norm']:.6f} shape={row.get('shape')}")
    base_ssm = [row for row in base_rows if row.get("tag") == "base_ssm_final"]
    pic_ssm = [row for row in pic_rows if row.get("tag") == "pic_ssm_final"]
    if base_ssm and pic_ssm:
        full, cached = base_ssm[-1], pic_ssm[-1]
        diff = sum((a - b) ** 2 for a, b in zip(full["head"], cached["head"])) ** 0.5
        denom = sum(a * a for a in full["head"]) ** 0.5 + 1e-12
        print(f"  ssm head8 relative L2={diff / denom:.6e}")
    for prefix, rows in (("base_", base_rows), ("pic_", pic_rows)):
        counts = {}
        for row in rows:
            tag = row.get("tag", "")
            if tag.startswith(prefix) and "norms" in row:
                counts[tag] = counts.get(tag, 0) + row.get("T", 0)
        print(f"  {prefix} token counts: {counts}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--data", default="data/hotpotqa/hotpotqa.jsonl")
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3.5-35b")
    parser.add_argument("--mode", default="transition_rope_recompute")
    parser.add_argument("--output-dir", default="outputs/direct-hypic-diag")
    args = parser.parse_args()

    samples = [
        json.loads(line)
        for line in Path(args.data).read_text().splitlines()
        if line.strip()
    ]
    chunks, suffix = build_case(samples[args.index])
    full = SYSTEM + "".join(chunks) + suffix + POST
    segmented = SYSTEM + SEP + SEP.join(chunks) + SEP + suffix + POST
    warmups = [SYSTEM + SEP + chunk + SEP + suffix + POST for chunk in chunks]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "full.jsonl"
    pic_path = output_dir / "ttr.jsonl"
    for path in (base_path, pic_path):
        path.write_text("")

    print(f"index={args.index} chunks={len(chunks)} model={args.model}")
    run_once("full", full, [], base_path, args.model, None)
    run_once(args.mode, segmented, warmups, pic_path, args.model, args.mode)
    compare(base_path, pic_path)
    compare_gdn(base_path.with_suffix(".gdn.jsonl"), pic_path.with_suffix(".gdn.jsonl"))


if __name__ == "__main__":
    main()
