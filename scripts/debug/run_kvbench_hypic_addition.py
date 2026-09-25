"""Run the target HotpotQA HYPIC pair through KVBench's Engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.Config import LoadConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "addition",
            "transition",
            "transition_rope",
            "transition_rope_recompute",
            "full",
        ),
        default="addition",
    )
    args = parser.parse_args()

    config = LoadConfig()
    engine_config = config.setdefault("Engine", {})
    engine_config["Tui"] = False
    engine_config["TuiWaitForQuit"] = False
    engine_config["Verbose"] = False

    from core.engine import Engine
    from metrics import TTFTMetric, ThroughputMetric
    from methods.Hypic import HypicMethod
    from tasks import HotpotQATask

    method_kwargs = {
        "gpuNums": 1,
        "memFractionStatic": 0.90,
        "tag": args.mode,
    }
    if args.mode == "full":
        method_kwargs["fullPrefill"] = True
    else:
        method_kwargs["picMode"] = args.mode

    report = Engine().Evaluate(
        tasks=[HotpotQATask(maxSamples=64)],
        methods=[HypicMethod(**method_kwargs)],
        metrics=[TTFTMetric(), ThroughputMetric()],
    )
    print(json.dumps(report["cores"], ensure_ascii=False, indent=2))
    print(f"OUTPUT_DIR {report['output_dir']}")


if __name__ == "__main__":
    main()
