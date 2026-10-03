"""Load ``HypicMethod`` and run one tiny prepare/reorder/generate smoke test.

Run from the KVBench root::

    python scripts/debug/SmokeHypic.py --gpu 0

The real file entry point and ``__main__`` guard are required because HYPIC's
SGLang engine starts its scheduler with Python's ``spawn`` context.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import HypicMethod, LinearKVMethod  # noqa: E402


def Main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--lines-per-chunk", type=int, default=192)
    parser.add_argument("--pic-mode", default="transition_rope_recompute")
    parser.add_argument("--linearkv", action="store_true")
    parser.add_argument(
        "--task",
        choices=["niah", "cwe", "vt", "musique", "samsum", "2wikimultihopqa"],
        help="run the first case of a Main.py task instead of the synthetic smoke",
    )
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

    method = (LinearKVMethod if args.linearkv else HypicMethod)(
        memFractionStatic=args.mem_fraction_static,
        **({} if args.linearkv else {"picMode": args.pic_mode}),
        fullPrefill=args.full_prefill,
        tag="full_prefill" if args.full_prefill else None,
    )
    if args.through_engine:
        from core.engine import Engine
        from core.Config import LoadConfig
        from metrics import TTFTMetric, ThroughputMetric
        from tasks.FreshGap import FreshGapTask

        LoadConfig().setdefault("Engine", {}).update(
            AvailableGpuIds=[args.gpu],
            BatchSize=1,
            InitializeTimeoutSec=600,
            TaskTimeoutSec=600,
            ShutdownGracePeriodSec=90,
            Tui=False,
            TuiWaitForQuit=False,
        )
        report = Engine().Evaluate(
            tasks=[
                FreshGapTask(
                    nCases=1,
                    linesPerChunk=args.lines_per_chunk,
                    maxNewTokens=args.max_new_tokens,
                )
            ],
            methods=[method],
            metrics=[TTFTMetric(), ThroughputMetric()],
        )
        print(json.dumps(report["cores"], ensure_ascii=False, indent=2))
        return

    try:
        print("[smoke] initialize", flush=True)
        method.Initialize([args.gpu])
        if args.task:
            from tasks import (
                CWEShuffleTask,
                MusiqueTask,
                NIAHShuffleTask,
                SamsumTask,
                VTShuffleTask,
                TwoWikiMultiHopQATask,
            )

            taskType = {
                "niah": NIAHShuffleTask,
                "cwe": CWEShuffleTask,
                "vt": VTShuffleTask,
                "musique": MusiqueTask,
                "samsum": SamsumTask,
                "2wikimultihopqa": TwoWikiMultiHopQATask,
            }[args.task]
            task = taskType(maxSamples=1, maxNewTokens=args.max_new_tokens)
            case = next(task.Cases())
            print(
                f"[smoke] task={args.task} segments="
                f"{len(case.input.prepare_input)} prepare",
                flush=True,
            )
            method.Prepare([case.input.prepare_input])
            result = method.Run(
                [case.input.run_input], maxNewTokens=task.maxNewTokens
            )[0]
            print(
                json.dumps(
                    {
                        "expected": case.metadata.get("answer"),
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
        if args.fresh_gap:
            from tasks.FreshGap import FreshGapTask

            task = FreshGapTask(
                nCases=1,
                linesPerChunk=args.lines_per_chunk,
                maxNewTokens=args.max_new_tokens,
            )
            case = next(task.Cases())
            print("[smoke] fresh-gap prepare", flush=True)
            method.Prepare([case.input.prepare_input])
            print("[smoke] fresh-gap run", flush=True)
            result = method.Run(
                [case.input.run_input], maxNewTokens=task.maxNewTokens
            )[0]
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
        from helpers.backends.Tokenization import EncodeChunks, EncodeText
        chunks = EncodeChunks([first, second], method.modelPath)
        method.Prepare([chunks])
        print("[smoke] reordered run", flush=True)
        result = method.Run(
            [chunks[1] + chunks[0] + EncodeText("\nWhat answer does document A give? Answer briefly.", method.modelPath)],
            maxNewTokens=args.max_new_tokens,
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
