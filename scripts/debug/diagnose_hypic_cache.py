"""Compare one HYPIC segment's committed cache under isolated/joint priming.

Run this script in a fresh process for each case, for example:

  python scripts/debug/diagnose_hypic_cache.py --priming isolated --mode transition_rope
  python scripts/debug/diagnose_hypic_cache.py --priming joint --mode transition_rope

The HYPIC-side opt-in instrumentation writes a .pt entry dump.  This runner
uses the official raw Qwen prompt shape and intentionally keeps the measured
request separate from the priming request.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.Config import Get  # noqa: E402

_HYPIC_CONFIG = Get("Hypic", {}) or {}
_HYPIC_ROOT = os.environ.get("HYPIC_REPO_PATH") or _HYPIC_CONFIG.get("RepoPath")
if not _HYPIC_ROOT:
    raise RuntimeError("Set Hypic.RepoPath or HYPIC_REPO_PATH to the HYPIC checkout")
HYPIC = Path(_HYPIC_ROOT).expanduser().resolve()
sys.path.insert(0, str(HYPIC / "python"))

import sglang as sgl  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from sglang.srt.pic.segmenter import segment_hash  # noqa: E402


SEP = "<<PIC_SEP>>"
SYSTEM = "You are a helpful assistant."
POST = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
MODEL_DEFAULT = os.environ.get("PIC_MODEL")
OUT_DEFAULT = os.environ.get("PIC_CACHE_DEBUG_DUMP", str(ROOT / "outputs" / "cache-debug"))

# Long enough to exercise Qwen's convolutional history, but deliberately small
# enough that each diagnostic run remains inexpensive.
C1 = ("C1: The first document establishes a fictional fact about amber foxes. " * 48).strip()
C2 = ("C2: The reusable target document says the answer is cobalt. " * 48).strip()
C3 = ("C3: A third document provides distractor facts about rivers. " * 48).strip()
TAIL = "Question: what answer is stated in the target document? Answer:"
MEASURED = SYSTEM + SEP + C1 + SEP + C2 + SEP + C3 + SEP + TAIL + POST


def official_prompt(parts: list[str]) -> str:
    return SYSTEM + SEP + SEP.join(parts) + SEP + TAIL + POST


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priming", choices=("isolated", "joint"), required=True)
    ap.add_argument(
        "--mode",
        choices=("addition", "transition", "transition_rope", "transition_rope_recompute"),
        default="transition_rope",
    )
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    if not args.model:
        ap.error("--model or PIC_MODEL is required")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    c2_ids = tok.encode(C2, add_special_tokens=False)
    c2_hash = segment_hash(c2_ids).hex()
    os.environ["PIC_CACHE_DEBUG_DUMP"] = str(Path(args.out).resolve())
    os.environ["PIC_CACHE_DEBUG_HASH"] = c2_hash
    os.environ["PIC_CACHE_DEBUG_LABEL"] = f"{args.mode}_{args.priming}"

    kwargs = dict(
        model_path=args.model,
        tp_size=int(os.environ.get("PIC_TP", "1")),
        dtype="bfloat16",
        context_length=32768,
        max_prefill_tokens=32768,
        max_running_requests=1,
        mem_fraction_static=float(os.environ.get("PIC_MEM", "0.80")),
        trust_remote_code=True,
        enable_multimodal=False,
        page_size=1,
        chunked_prefill_size=-1,
        cuda_graph_backend_prefill="disabled",
        log_level="error",
        pic_enable=True,
        pic_mode=args.mode,
        pic_separator_str=SEP,
        max_mamba_cache_size=int(os.environ.get("PIC_MAMBA", "32")),
    )
    warm = (
        official_prompt([C2])
        if args.priming == "isolated"
        else official_prompt([C1, C2, C3])
    )
    print(
        f"mode={args.mode} priming={args.priming} C2_tokens={len(c2_ids)} "
        f"C2_hash={c2_hash} warm_tokens={len(tok.encode(warm, add_special_tokens=False))}",
        flush=True,
    )
    engine = sgl.Engine(**kwargs)
    try:
        warm_out = engine.generate(
            warm,
            sampling_params={"temperature": 0.0, "max_new_tokens": 4},
        )
        measured = engine.generate(
            MEASURED,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        )
        print(
            "warm_meta=" + repr(warm_out.get("meta_info", {})) +
            " measured_meta=" + repr(measured.get("meta_info", {})),
            flush=True,
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
