"""Load ``HypicMethod`` and run one tiny prepare/reorder/generate smoke test.

Run from the KVBench root::

    python scripts/SmokeHypic.py --gpu 0

The real file entry point and ``__main__`` guard are required because HYPIC's
SGLang engine starts its scheduler with Python's ``spawn`` context.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import HypicMethod  # noqa: E402


def Main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=12000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lines-per-chunk", type=int, default=192)
    parser.add_argument("--pic-mode", default="addition")
    parser.add_argument("--full-prefill", action="store_true")
    parser.add_argument(
        "--fresh-gap",
        action="store_true",
        help="run the FreshGap case directly and print its generated output",
    )
    parser.add_argument(
        "--through-engine",
        action="store_true",
        help=(
            "run through KVBench's spawned worker instead of calling "
            "the Method directly"
        ),
    )
    args = parser.parse_args()

    method = HypicMethod(
        maxNewTokens=args.max_new_tokens,
        maxModelLen=args.max_model_len,
        memFractionStatic=0.80,
        picMode=args.pic_mode,
        fullPrefill=args.full_prefill,
        tag="full_prefill" if args.full_prefill else None,
    )
    if args.through_engine:
        from core.engine import Engine
        from metrics import TTFTMetric, ThroughputMetric
        from tasks.FreshGap import FreshGapTask

        report = Engine(
            availableGpuIds=[args.gpu],
            batchSize=1,
            initializeTimeout=600,
            taskTimeout=600,
            shutdownGracePeriod=90,
            tui=False,
        ).Evaluate(
            tasks=[
                FreshGapTask(nCases=1, linesPerChunk=args.lines_per_chunk)
            ],
            methods=[method],
            metrics=[TTFTMetric(), ThroughputMetric()],
        )
        print(json.dumps(report["cores"], ensure_ascii=False, indent=2))
        return

    try:
        print("[smoke] initialize", flush=True)
        method.Initialize([args.gpu])
        if args.fresh_gap:
            from tasks.FreshGap import FreshGapTask

            case = next(
                FreshGapTask(nCases=1, linesPerChunk=args.lines_per_chunk).Cases()
            )
            print("[smoke] fresh-gap prepare", flush=True)
            method.Prepare([case.input.prepare_input])
            print("[smoke] fresh-gap run", flush=True)
            result = method.Run([case.input.run_input])[0]
            print(
                json.dumps(
                    {
                        "expected": case.metadata["answer"],
                        "output": result.output,
                        "performance": result.performance,
                        "metadata": result.metadata,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            method.Reset()
            return
        first = "Document A says the answer is amber. " * 32
        second = "Document B discusses rivers and mountains. " * 32
        print("[smoke] prepare", flush=True)
        method.Prepare([[first, second]])
        print("[smoke] reordered run", flush=True)
        result = method.Run(
            [second + first + "\nWhat answer does document A give? Answer briefly."]
        )[0]
        print(
            json.dumps(
                {
                    "output": result.output,
                    "performance": result.performance,
                    "metadata": result.metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        method.Reset()
    finally:
        print("[smoke] close", flush=True)
        method.Close()


if __name__ == "__main__":
    Main()
